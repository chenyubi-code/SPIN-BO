"""Numerical CSV summaries for the final manuscript; no typesetting or plotting."""
from __future__ import annotations
import numpy as np
from .design import FOCAL, GP_METHODS, METHODS, METHOD_LABELS, conditions
from .util import ROOT, read_json, write_csv

METRICS = ("INR", "OptSuccess_408", "Recall20_200", "Recall20_408")


def read_campaigns(config):
    records = []
    for c in conditions():
        for method in METHODS if c["id"] == FOCAL else GP_METHODS:
            for landscape in config["landscape_seeds"]:
                for campaign in config["campaign_seeds"]:
                    p = ROOT / "Outputs/Results" / config["simulation_id"] / c["id"] / f"landscape_{landscape}" / f"campaign_{campaign}" / method
                    row = dict(condition=c["id"], rho=c["rho"], gamma=c["gamma"], method=method,
                               landscape=landscape, campaign=campaign, status="missing")
                    if (p / "status.json").exists():
                        row["status"] = read_json(p / "status.json")["state"]
                    if row["status"] == "complete":
                        s = read_json(p / "summary.json")
                        row.update(INR=s["INR"], OptSuccess_408=s["top1_success"],
                                   Recall20_200=s["recall_20_at_200"], Recall20_408=s["recall_20"])
                    records.append(row)
    return records


def aggregate(records, config):
    rows = []
    for c in conditions():
        for method in METHODS if c["id"] == FOCAL else GP_METHODS:
            matched = [r for r in records if r["condition"] == c["id"] and r["method"] == method and r["status"] == "complete"]
            row = {"condition": c["id"], "rho": c["rho"], "gamma": c["gamma"], "method": method,
                   "method_label": METHOD_LABELS[method], "complete_campaigns": len(matched), "expected_campaigns": 10}
            identities = [(int(r["landscape"]), int(r["campaign"])) for r in matched]
            expected = {(l, c) for l in config["landscape_seeds"] for c in config["campaign_seeds"]}
            complete = len(identities) == len(expected) and set(identities) == expected
            row["status"] = "complete" if complete else "incomplete"
            for metric in METRICS:
                values = [np.mean([float(r[metric]) for r in matched if int(r["landscape"]) == l])
                          for l in config["landscape_seeds"]] if complete else []
                row[metric+"_mean"] = float(np.mean(values)) if complete else None
                row[metric+"_se"] = float(np.std(values, ddof=1)/np.sqrt(len(values))) if complete else None
            rows.append(row)
    return rows


def report(config):
    records = read_campaigns(config)
    rows = aggregate(records, config)
    out = ROOT / "Outputs" / "Summary"
    write_csv(out / "campaign_metrics.csv", records)
    write_csv(out / "grid_summary.csv", [r for r in rows if r["method"] in GP_METHODS])
    write_csv(out / "focal_summary.csv", [r for r in rows if r["condition"] == FOCAL])
    differences = []
    for c in conditions():
        full = next(r for r in rows if r["condition"] == c["id"] and r["method"] == "full")
        target = next(r for r in rows if r["condition"] == c["id"] and r["method"] == "target_only")
        complete = full["status"] == target["status"] == "complete"
        differences.append({"condition":c["id"],"rho":c["rho"],"gamma":c["gamma"],
                            "delta_INR":full["INR_mean"]-target["INR_mean"] if complete else None,
                            "delta_Recall20_408":full["Recall20_408_mean"]-target["Recall20_408_mean"] if complete else None,
                            "status":"complete" if complete else "incomplete"})
    write_csv(out / "grid_differences.csv", differences)
    print(f"Numeric summaries written to {out.relative_to(ROOT)}")
    return rows
