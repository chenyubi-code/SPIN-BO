"""Run simulated discovery campaigns and save their numerical results."""
from __future__ import annotations

import resource
import sys
import time

import numpy as np

from .design import METHODS, SOURCE_IDS, output_dir
from .evaluation import discovery, terminal_summary
from .inputs import load_inputs
from .models import fit_model, load_geometry, model_config, prepare_geometry, save_geometry
from .storage import cache_root, common_initialization, get_basis, get_landscape, landscape_dir, lock
from .util import ROOT, digest, read_json, rng, seed, write_csv, write_json


def peak_memory_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def run_dir(c, config, landscape, campaign, method):
    return output_dir(c) / "Results" / config["simulation_id"] / c["id"] / f"landscape_{landscape}" / f"campaign_{campaign}" / method


def geometry_for(inputs, data, meta, config, c, landscape, *, source_free=False, encode=True):
    if source_free:
        partner_ids = ["MMP10"]
        source = np.empty((len(inputs["ids"]), 0), dtype=np.float64)
        path = cache_root(config) / f"landscape_{landscape}/Geometry/source_free"
        encode = False
    else:
        partner_ids = ["MMP10", *meta["source_ids"]]
        source = data["source_observed"]
        path = landscape_dir(c, config, landscape) / "Geometry" / ("encoder_v1" if encode else "raw")
    partners = inputs["partner_raw"][[inputs["partner_lookup"][p] for p in partner_ids]]
    public_sources = inputs["partner_raw"][[inputs["partner_lookup"][p] for p in SOURCE_IDS]]
    encoder_seed = seed(config["master_seed"], "source_encoder", landscape)
    settings = model_config(config["model"])
    identity = {"inputs": inputs["cache_inputs"], "condition": None if source_free else c,
                "generator": config, "partner_ids": partner_ids,
                "public_source_partner_ids": SOURCE_IDS,
                "encoder_seed": encoder_seed, "encode": encode, "settings": settings}
    with lock(path.parent / (path.name + ".lock")):
        stamp = path / "cache_identity.json"
        if stamp.exists():
            if read_json(stamp)["identity"] != digest(identity):
                raise ValueError(f"Geometry cache differs from current inputs/configuration: {path}")
            return load_geometry(path)[0]
        geometry, artifacts = prepare_geometry(inputs["mutant_raw"], partners, source, encoder_seed,
                                                settings, train_encoder=encode, mutant_sequences=inputs["sequences"],
                                                public_source_partner_raw=public_sources,
                                                public_source_partner_ids=SOURCE_IDS)
        save_geometry(path, geometry, artifacts)
        write_json(stamp, {"identity": digest(identity)})
        return geometry


def campaign(inputs, data, geometry, config, c, landscape, campaign_seed, method, initial):
    directory = run_dir(c, config, landscape, campaign_seed, method)
    settings = model_config(config["model"])
    identity_fields = {"configuration": config, "model": settings, "condition": c,
                       "landscape_seed": landscape, "campaign_seed": campaign_seed, "method": method,
                       "inputs": inputs["cache_inputs"], "initial": initial["indices"]}
    identity = digest(identity_fields)
    started = time.perf_counter()
    with lock(directory / ".run.lock"):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path).get("stage_failure_only"):
            # A preparation failure can be retried with the current configuration.
            manifest_path.unlink()
        if manifest_path.exists():
            if read_json(manifest_path)["identity"] != identity:
                raise ValueError(f"Run inputs/configuration changed: {directory}. Move the old run to an archive first.")
            if (directory / "status.json").exists() and read_json(directory / "status.json")["state"] == "complete":
                return {"state": "skipped_complete", "path": str(directory.relative_to(ROOT))}
        else:
            write_json(manifest_path, {**identity_fields, "identity": identity,
                                      "target_direction": "maximize_dimensionless_simulated_score"})
        write_json(directory / "status.json", {"state": "running", "runtime_seconds": 0})
        observed = list(map(int, initial["indices"]))
        truth = data["target_truth"]
        # The evaluator owns this table. Models get only its currently observed slice.
        assay = truth.copy()
        rows, query_rows = [], []
        for i in observed:
            query_rows.append({"round": 0, "slot": None, "candidate_index": i,
                               "sequence_id": str(inputs["ids"][i]), "observed_y": float(assay[i])})
        try:
            round_index = 0
            while True:
                n = len(observed)
                observed_array = np.asarray(observed, dtype=np.int64)
                mask = np.ones(len(truth), dtype=bool)
                mask[observed_array] = False
                # Sorting eligible indices by canonical ID makes every deterministic tie reproducible.
                eligible = np.flatnonzero(mask)
                eligible = eligible[np.argsort(inputs["ids"][eligible], kind="stable")]
                training_seed = seed(config["master_seed"], "method_training", landscape, campaign_seed, method, round_index)
                fit_start = time.perf_counter()
                fitted = fit_model(method, geometry, observed_array, assay[observed_array].copy(), training_seed, settings)
                fit_seconds = time.perf_counter()-fit_start
                row = discovery(truth, assay, observed_array, inputs["ids"], config["recall_ks"])
                row.update(round=round_index, fit_seconds=fit_seconds,
                           fit_converged=fitted.diagnostics.get("selected_endpoint_converged"),
                           amplitude_lower_boundary=fitted.diagnostics.get("amplitude_lower_boundary"),
                           runtime_seconds=time.perf_counter()-started, peak_memory_mb=peak_memory_mb())
                rows.append(row)
                write_csv(directory / "trajectory.csv", rows)
                write_csv(directory / "queries.csv", query_rows)
                if n >= config["budget"]:
                    break
                batch_size = min(config["batch_size"], config["budget"]-n)
                acquisition_seed = seed(config["master_seed"], "acquisition_factor", landscape, campaign_seed, method, round_index)
                slots = [seed(config["master_seed"], "acquisition_slot", landscape, campaign_seed, method, round_index, slot)
                         for slot in range(batch_size)]
                selected, diagnostics = fitted.select_batch(eligible, batch_size,
                                   rng(config["master_seed"], "acquisition_factor", landscape, campaign_seed, method, round_index),
                                   sampler=config["sampler"], slot_seeds=slots)
                selected = list(map(int, selected))
                if len(selected) != batch_size or len(set(selected)) != batch_size or not set(selected).issubset(set(eligible)):
                    raise ValueError("Acquisition returned an invalid/repeated/previously observed batch.")
                batch_path = directory / f"Batches/round_{round_index+1:02d}.json"
                write_json(batch_path, {"selected_indices": selected, "selected_ids": inputs["ids"][selected],
                           "training_seed": training_seed, "factor_seed": acquisition_seed,
                           "slot_seeds": slots, "diagnostics": diagnostics})
                # Simultaneous reveal occurs only after the complete batch is selected and recorded.
                observed.extend(selected)
                for slot, i in enumerate(selected):
                    query_rows.append({"round": round_index+1, "slot": slot, "candidate_index": i,
                                       "sequence_id": str(inputs["ids"][i]), "observed_y": float(assay[i])})
                round_index += 1
            summary = terminal_summary(rows, config["budget"], config.get("intermediate_budgets", [200]))
            write_json(directory / "summary.json", summary)
            status = {"state": "complete", "runtime_seconds": time.perf_counter()-started,
                      "peak_memory_mb": peak_memory_mb(), "budget": len(observed),
                      "nonconvergent_fits": sum(r.get("fit_converged") is False for r in rows),
                      "amplitude_boundary_fits": sum(r.get("amplitude_lower_boundary") is True for r in rows)}
            write_json(directory / "status.json", status)
            return status
        except Exception as exc:
            status = {"state": "failed", "runtime_seconds": time.perf_counter()-started,
                      "peak_memory_mb": peak_memory_mb(), "budget": len(observed),
                      "error_type": type(exc).__name__, "error": str(exc), "error_detail": type(exc).__name__}
            write_json(directory / "summary.json", terminal_summary(rows, config["budget"], config.get("intermediate_budgets", [200])))
            write_json(directory / "status.json", status)
            return status


def execute(config, selected_conditions, landscapes=None, campaigns=None, methods=None, *, prepare_only=False):
    inputs = load_inputs(config)
    methods = methods or METHODS
    unknown = set(methods)-set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    landscapes = landscapes or config["landscape_seeds"]
    campaigns = campaigns or config["campaign_seeds"]
    if not set(landscapes).issubset(config["landscape_seeds"]) or not set(campaigns).issubset(config["campaign_seeds"]):
        raise ValueError("Requested seeds are absent from the experiment configuration.")
    statuses = []
    for landscape in landscapes:
        basis, basis_meta = get_basis(inputs, config, landscape)
        initials = {c: common_initialization(inputs, config, landscape, c, basis) for c in campaigns}
        source_free_geometry = None
        for c in selected_conditions:
            print(f"[{landscape}] {c['id']}: preparing source bank", flush=True)
            data, meta = get_landscape(inputs, config, c, landscape, basis, basis_meta)
            if prepare_only:
                continue
            geometries = {}
            # The raw mean-channel model remains independent of encoder training success.
            for encode, group in ((False, ("mean_only",)), (True, ("full", "encoder_only"))):
                affected = [m for m in group if m in methods]
                if not affected:
                    continue
                try:
                    geometry = geometry_for(inputs, data, meta, config, c, landscape, encode=encode)
                    for method in affected:
                        geometries[method] = geometry
                except Exception as exc:
                    for campaign_seed in campaigns:
                        for method in affected:
                            directory = run_dir(c, config, landscape, campaign_seed, method)
                            status = {"state": "failed", "stage": "source_geometry", "error": str(exc),
                                      "error_type": type(exc).__name__, "runtime_seconds": None, "error_detail": type(exc).__name__}
                            old_status = directory / "status.json"
                            if not old_status.exists() or read_json(old_status).get("state") != "complete":
                                write_json(directory / "manifest.json", {"condition": c, "landscape_seed": landscape,
                                           "campaign_seed": campaign_seed, "method": method, "configuration": config,
                                           "identity": None, "stage_failure_only": True})
                                write_json(old_status, status)
                            statuses.append(status)
            if any(m in ("target_only", "alde", "evolvepro", "random") for m in methods) and source_free_geometry is None:
                source_free_geometry = geometry_for(inputs, data, meta, config, c, landscape, source_free=True, encode=False)
            for campaign_seed in campaigns:
                for method in methods:
                    is_transfer = method in ("full", "mean_only", "encoder_only")
                    if is_transfer and method not in geometries:
                        continue
                    print(f"[{landscape}/{campaign_seed}] {c['id']}: {method}", flush=True)
                    geometry = geometries[method] if is_transfer else source_free_geometry
                    statuses.append(campaign(inputs, data, geometry, config, c, landscape, campaign_seed,
                                             method, initials[campaign_seed]))
    return statuses
