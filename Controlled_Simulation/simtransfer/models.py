"""Simulation adapters: Encoder-v1, exact REML/UK, joint TS and target-only baselines.

Scientific routines are retained from the implementation used for the paper.
This module requires numpy/scipy/torch/scikit-learn.
Only revealed target labels enter fit_model.  The complete source dictionary is
kept on its supplied assay scale; the SVD changes basis, not its column space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from scipy.stats import qmc

METHODS = ("full", "mean_only", "encoder_only", "target_only", "alde", "evolvepro", "random")
GP_METHODS = METHODS[:4]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MUTATION_POSITIONS = (4, 35, 38, 68, 71, 97, 99)
ENCODER_PREPROCESSING = "center_coordinate_fixed_public_sources_v1"
MODEL_DEFAULTS = {
    "encoder_hidden": 128, "encoder_bottleneck": 32, "encoder_epochs": 100,
    "encoder_batch_size": 256, "encoder_learning_rate": 1e-3,
    "encoder_weight_decay": 1e-4, "encoder_gradient_clip": 5.0,
    "encoder_device": "cpu", "encoder_prediction_batch_size": 1024,
    "encoder_preprocessing": ENCODER_PREPROCESSING,
    "encoder_std_floor_fraction": 0.1, "encoder_public_source_count": 8,
    "encoder_train_candidates": 4096,
    "encoder_split_seed": 17201,
    "gp_amplitude_bounds": [1e-3, 10.0],
    "gp_length_relative_bounds": [0.1, 10.0],
    "gp_fit_noise": False, "gp_nugget_std": 0.0,
    "gp_noise_bounds": [1e-3, 2.0], "gp_sobol_starts": 16,
    "gp_optimizer_maxiter": 500, "gp_jitter_initial_factor": 1e-7,
    "gp_jitter_max_factor": 1e-3, "mean_rank_relative_tolerance": 1e-10,
    "distance_pair_count": 100000, "distance_seed": 314159,
    "sampler": "exact", "prediction_batch_size": 1024,
    "alde_ensemble_size": 5, "alde_hidden": 30,
    "alde_learning_rate": 1e-3, "alde_max_epochs": 300, "alde_patience": 30,
    "rf_estimators": 100, "rf_n_jobs": 1,
}


def model_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = dict(MODEL_DEFAULTS)
    if config is not None:
        result.update(config)
    if int(result["gp_sobol_starts"]) < 1:
        raise ValueError("GP restart count must be positive")
    if float(result["gp_nugget_std"]) < 0:
        raise ValueError("A declared GP nugget cannot be negative")
    for key in ("encoder_epochs", "encoder_batch_size", "encoder_prediction_batch_size", "prediction_batch_size",
                "encoder_hidden", "encoder_bottleneck", "gp_optimizer_maxiter", "distance_pair_count",
                "encoder_train_candidates", "encoder_public_source_count",
                "alde_ensemble_size", "alde_hidden", "alde_max_epochs", "alde_patience", "rf_estimators"):
        if int(result[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < float(result["gp_jitter_initial_factor"]) <= float(result["gp_jitter_max_factor"]):
        raise ValueError("Numerical jitter factors must be positive and ordered")
    if not 0 < float(result["mean_rank_relative_tolerance"]) < 1:
        raise ValueError("Observed mean rank tolerance must lie strictly between zero and one")
    if result["encoder_preprocessing"] != ENCODER_PREPROCESSING:
        raise ValueError("The revised encoder requires centered coordinate scaling with the fixed public source panel")
    if not 0 < float(result["encoder_std_floor_fraction"]) <= 1:
        raise ValueError("Encoder standard-deviation floor fraction must lie in (0, 1]")
    if int(result["encoder_public_source_count"]) != 8:
        raise ValueError("The fixed public encoder reference panel must contain all eight source partners")
    if int(result["encoder_split_seed"]) < 0:
        raise ValueError("Encoder candidate split seed cannot be negative")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def exact_mean_basis(source_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep every numerically nonzero full-library direction, including near duplicates.

    The only cutoff is LAPACK-style machine precision: max(shape)*eps*sigma_max.
    A much larger scientific truncation tolerance is deliberately not used here.
    H @ transform = basis; thus transform @ fitted_beta is the minimum-norm raw
    coefficient representative (individual coefficients are unidentifiable when
    raw columns contain exact duplicates).
    """
    sources = np.asarray(source_y, dtype=np.float64)
    if sources.ndim != 2 or not np.isfinite(sources).all():
        raise ValueError("source_y must be a finite candidate-by-source matrix")
    raw = np.column_stack((np.ones(len(sources)), sources))
    u, singular, vt = np.linalg.svd(raw, full_matrices=False)
    tolerance = max(raw.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    scale = np.sqrt(len(raw))
    transform = (vt[:rank].T / singular[:rank]) * scale
    basis = u[:, :rank] * scale
    return basis, transform, {
        "raw_columns": raw.shape[1], "effective_rank": rank,
        "basis_convention": "U_retained_times_sqrt_N",
        "source_response_preprocessing": "none_raw_assay_scale",
        "raw_coefficients_identifiable": rank == raw.shape[1],
    }


def make_one_hot(sequences: Sequence[str]) -> np.ndarray:
    indices = np.array([[AMINO_ACIDS.index(str(seq)[pos - 1]) for pos in MUTATION_POSITIONS]
                        for seq in sequences], dtype=np.int64)
    return np.eye(len(AMINO_ACIDS), dtype=np.float32)[indices].reshape(len(indices), -1)


@dataclass
class Geometry:
    raw_mutant: np.ndarray
    raw_target: np.ndarray
    encoded_target: np.ndarray | None
    source_y: np.ndarray
    mean_basis: np.ndarray
    mean_transform: np.ndarray
    one_hot: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return len(self.raw_mutant)


def _device(name: str) -> str:
    import torch
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested encoder CUDA device is unavailable")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested encoder MPS device is unavailable")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown encoder device {name}")
    return name


def encoder_training_indices(candidate_count: int, config: Mapping[str, Any]) -> np.ndarray:
    """Select the fixed training candidates independently of response values."""
    train_count = int(config["encoder_train_candidates"])
    if train_count < 1 or train_count > candidate_count:
        raise ValueError("Encoder training candidate count must be positive and fit in the library")
    order = np.random.default_rng(int(config["encoder_split_seed"])).permutation(candidate_count)
    return order[:train_count]


def encoder_preprocessing(mutant: np.ndarray, public_source_partners: np.ndarray,
                          config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Fit feature-only block statistics using all candidates and all eight sources.

    A constant coordinate uses the block's positive-standard-deviation floor. A
    completely constant block uses unit scales, avoiding NaNs for the fixed source panel.
    The target partner and all response values are absent from this API.
    """
    values = np.asarray(mutant, dtype=np.float64)
    partners = np.asarray(public_source_partners, dtype=np.float64)
    if (values.ndim != 2 or len(values) < 2
            or partners.shape != (int(config["encoder_public_source_count"]), values.shape[1])
            or not np.isfinite(values).all() or not np.isfinite(partners).all()):
        raise ValueError("Encoder preprocessing requires all finite public candidate features and the fixed eight-source reference panel")
    statistics: dict[str, np.ndarray] = {}
    for name, block in (("mutant", values), ("partner", partners)):
        mean = block.mean(axis=0)
        std = block.std(axis=0, ddof=0)
        positive = std[std > 0]
        floor = float(config["encoder_std_floor_fraction"]) * float(np.median(positive)) if len(positive) else 1.0
        scale = np.maximum(std, floor) if len(positive) else np.ones_like(std)
        statistics.update({f"{name}_mean": mean, f"{name}_scale": scale,
                           f"{name}_std": std, f"{name}_std_floor": np.array(floor)})
    return statistics


def frozen_encoder_coordinates(mutant: np.ndarray, target_partner: np.ndarray,
                               encoder_state: Mapping[str, Any],
                               batch_size: int = 1024, *,
                               preprocessing: Mapping[str, np.ndarray] | None = None
                               ) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate the same frozen Encoder-v1 accurately, up to a GP-invariant shift.

    Training remains the original float32 GELU MLP. For negative activations,
    float32 1+erf(x/sqrt(2)) can round to zero; adding the final bias can also
    erase differences. Use float64 x*Phi(x), subtract the first candidate's
    hidden activation before the final matrix product, and cancel the output
    bias. The result is mathematically z(x)-z(x_ref), preserving every distance
    and the kernel. No residual feature, noise or substitute geometry is added.
    Calls supply the centered coordinate preprocessing.
    """
    def as_array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    x = np.asarray(mutant, dtype=np.float64)
    a = np.asarray(target_partner, dtype=np.float64)
    w1, b1 = as_array(encoder_state["0.weight"]), as_array(encoder_state["0.bias"])
    w2 = as_array(encoder_state["2.weight"])
    if (x.ndim != 2 or len(x) < 2 or a.shape != (x.shape[1],)
            or w1.shape[1] != 2*x.shape[1] or w2.shape[1] != w1.shape[0]
            or batch_size <= 0):
        raise ValueError("Frozen Encoder-v1 inputs/weights have invalid dimensions or values")
    if not all(np.isfinite(v).all() for v in (x, a, w1, b1, w2)):
        raise ValueError("Frozen Encoder-v1 inputs or weights contain nonfinite values")
    width = x.shape[1]
    if preprocessing is None:
        raise ValueError("Frozen Encoder-v1 requires preprocessing statistics")
    xm, xs = as_array(preprocessing["mutant_mean"]), as_array(preprocessing["mutant_scale"])
    pm, ps = as_array(preprocessing["partner_mean"]), as_array(preprocessing["partner_scale"])
    if (any(v.shape != (width,) or not np.isfinite(v).all() for v in (xm, xs, pm, ps))
            or np.any(xs <= 0) or np.any(ps <= 0)):
        raise ValueError("Frozen Encoder-v1 preprocessing statistics are invalid")
    input_protocol = ENCODER_PREPROCESSING
    partner_offset = ((a-pm)/ps) @ w1[:, :width].T + b1
    reference_pre = ((x[0]-xm)/xs) @ w1[:, width:].T + partner_offset
    reference_hidden = reference_pre*ndtr(reference_pre)
    encoded = np.empty((len(x), w2.shape[0]), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        pre = ((x[start:start+batch_size]-xm)/xs) @ w1[:, width:].T + partner_offset
        encoded[start:start+len(pre)] = (pre*ndtr(pre)-reference_hidden) @ w2.T
    if not np.isfinite(encoded).all():
        raise ValueError("Stable frozen Encoder-v1 produced nonfinite coordinates")
    return encoded, {"frozen_inference_device": "cpu", "frozen_inference_dtype": "float64",
                     "input_preprocessing": input_protocol,
                     "gelu_evaluation": "x_times_stable_normal_cdf",
                     "output_transform": "subtract_reference_bottleneck_translation_only",
                     "reference_candidate_index": 0, "output_bias_cancels_in_distances": True}


def _train_encoder(mutant: np.ndarray, partners: np.ndarray, source_y: np.ndarray,
                   seed: int, config: Mapping[str, Any], *,
                   public_source_partner_raw: np.ndarray,
                   public_source_partner_ids: Sequence[str] | None = None,
                   reference_origin: str = "explicit_fixed_public_panel"
                   ) -> tuple[np.ndarray, dict[str, Any]]:
    """Source-only Encoder-v1 with a fixed candidate split and public feature scaling."""
    import torch
    from torch import nn
    count, width = mutant.shape
    source_count = source_y.shape[1]
    if source_count < 1:
        raise ValueError("Encoder-v1 requires at least one complete source profile")
    device = _device(str(config["encoder_device"]))
    reference = np.asarray(public_source_partner_raw, dtype=np.float32)
    preprocessing = encoder_preprocessing(mutant, reference, config)
    if any(not np.any(np.all(reference == row, axis=1)) for row in partners[1:]):
        raise ValueError("Every trained source partner must belong to the fixed public source reference panel")
    reference_ids = None if public_source_partner_ids is None else list(map(str, public_source_partner_ids))
    if reference_ids is not None and (len(reference_ids) != 8 or len(set(reference_ids)) != 8
                                      or "MMP10" in reference_ids):
        raise ValueError("Public source reference IDs must name eight distinct sources and exclude target MMP10")
    if np.any(np.all(reference == partners[0], axis=1)):
        raise ValueError("The target partner cannot enter the fixed public source reference panel")
    train_indices = encoder_training_indices(count, config)
    train_count = len(train_indices)
    normalized_mutant = ((np.asarray(mutant, dtype=np.float64)-preprocessing["mutant_mean"])
                         / preprocessing["mutant_scale"]).astype(np.float32)
    normalized_partners = ((np.asarray(partners[1:], dtype=np.float64)-preprocessing["partner_mean"])
                           / preprocessing["partner_scale"]).astype(np.float32)
    torch.manual_seed(int(seed) % (2**63 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) % (2**63 - 1))
    encoder = nn.Sequential(nn.Linear(2 * width, int(config["encoder_hidden"])), nn.GELU(),
                            nn.Linear(int(config["encoder_hidden"]), int(config["encoder_bottleneck"])))
    head = nn.Linear(int(config["encoder_bottleneck"]), 1)
    network = nn.Sequential(encoder, head)
    for module in network.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    network.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(config["encoder_learning_rate"]),
                                  weight_decay=float(config["encoder_weight_decay"]),
                                  betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
                                  maximize=False, foreach=None, capturable=False,
                                  differentiable=False, fused=None)
    batch_size = int(config["encoder_batch_size"])
    for epoch in range(int(config["encoder_epochs"])):
        network.train()
        rng = np.random.default_rng(int(seed) + epoch)
        source_order = rng.permutation(source_count)
        order = np.column_stack([int(s)*train_count+rng.permutation(train_count)
                                 for s in source_order]).reshape(-1)
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            source_rows, candidate_positions = np.divmod(rows, train_count)
            candidate_rows = train_indices[candidate_positions]
            values = np.concatenate((normalized_partners[source_rows], normalized_mutant[candidate_rows]), axis=1)
            x = torch.as_tensor(values, dtype=torch.float32, device=device)
            y = torch.as_tensor(source_y[candidate_rows, source_rows], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(network(x).squeeze(-1), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Encoder loss became nonfinite at epoch {epoch + 1}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(network.parameters(), float(config["encoder_gradient_clip"]))
            if not torch.isfinite(norm):
                raise RuntimeError("Encoder gradient became nonfinite")
            optimizer.step()
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    latent, inference_metadata = frozen_encoder_coordinates(
        mutant, partners[0], encoder_state, batch_size=int(config["encoder_prediction_batch_size"]),
        preprocessing=preprocessing)
    return latent, {
        "encoder_metadata": {
            "name": "Encoder-v1",
            "architecture": [2 * width, int(config["encoder_hidden"]), int(config["encoder_bottleneck"])],
            "activation": "GELU", "preprocessing_protocol": ENCODER_PREPROCESSING,
            "device": device, "std_floor_fraction": float(config["encoder_std_floor_fraction"]),
            "partner_reference_ids": reference_ids, "partner_reference_origin": reference_origin,
            "training_source_count": source_count, "candidate_training_count": train_count,
            "seed": int(seed), "checkpoint_selection": "final_epoch",
            "input_order": "partner_then_mutant", **inference_metadata,
            "loss": "unweighted_source_row_MSE_raw_source_responses",
            "sampler": "source_balanced_interleaved_permutation"}}


def prepare_geometry(mutant_raw: np.ndarray, partner_raw: np.ndarray | None,
                     source_y: np.ndarray | None, seed: int, config: Mapping[str, Any] | None = None,
                     *, train_encoder: bool = True, mutant_sequences: Sequence[str] | None = None,
                     public_source_partner_raw: np.ndarray | None = None,
                     public_source_partner_ids: Sequence[str] | None = None
                     ) -> tuple[Geometry, dict[str, Any]]:
    cfg = model_config(config)
    mutant = np.asarray(mutant_raw, dtype=np.float32)
    if mutant.ndim != 2 or not np.isfinite(mutant).all() or len(mutant) < 2:
        raise ValueError("mutant_raw must be a finite N by D matrix with N >= 2")
    sources = np.empty((len(mutant), 0)) if source_y is None else np.asarray(source_y, dtype=np.float64)
    if sources.ndim != 2 or len(sources) != len(mutant):
        raise ValueError("source_y must have candidate-major shape (N,S)")
    basis, transform, rank_metadata = exact_mean_basis(sources)
    raw_rms = float(np.sqrt(np.mean(np.square(mutant, dtype=np.float64))))
    if raw_rms <= 0:
        raise ValueError("Raw mutant RMS must be positive")
    artifacts: dict[str, Any] = {}
    encoded = None
    if train_encoder:
        partners = np.asarray(partner_raw, dtype=np.float32)
        if partners.shape != (sources.shape[1] + 1, mutant.shape[1]) or not np.isfinite(partners).all():
            raise ValueError("partner_raw shape must be (1+S,D), target partner first")
        reference_origin = "explicit_fixed_public_panel"
        if public_source_partner_raw is None:
            # The paper uses all eight sources. Use the complete public panel
            # if the caller has not supplied the same reference explicitly.
            if sources.shape[1] != int(cfg["encoder_public_source_count"]):
                raise ValueError("Provide public_source_partner_raw for the fixed eight-source reference panel, containing all eight paper sources")
            public_source_partner_raw = partners[1:]
            reference_origin = "current_bank_complete_eight_sources"
        encoded, artifacts = _train_encoder(
            mutant, partners, sources, seed, cfg, public_source_partner_raw=public_source_partner_raw,
            public_source_partner_ids=public_source_partner_ids, reference_origin=reference_origin)
    one_hot = None if mutant_sequences is None else make_one_hot(mutant_sequences)
    if one_hot is not None and len(one_hot) != len(mutant):
        raise ValueError("Sequence rows do not match raw embedding rows")
    metadata = {"source_mean": rank_metadata, "raw_mutant_rms": raw_rms,
                "encoder_seed": int(seed), "model_config": cfg,
                "one_hot_positions": list(MUTATION_POSITIONS), "one_hot_amino_acid_order": AMINO_ACIDS}
    metadata.update({k: v for k, v in artifacts.items() if k.endswith("metadata")})
    raw_target = np.asarray(mutant / raw_rms, dtype=np.float64)
    geometry = Geometry(mutant, raw_target, encoded, sources, basis, transform, one_hot, metadata)
    metadata["raw_median_distance"] = median_distance(raw_target, cfg)
    if encoded is not None:
        metadata["encoded_median_distance"] = median_distance(encoded, cfg)
    return geometry, artifacts


def save_geometry(path: str | Path, geometry: Geometry, artifacts: Mapping[str, Any]) -> None:
    """Cache the fixed arrays and scales used by subsequent experiment runs."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in vars(geometry).items() if isinstance(value, np.ndarray)}
    np.savez_compressed(destination / "geometry.npz", **arrays)
    (destination / "metadata.json").write_text(json.dumps(_jsonable(geometry.metadata), indent=2) + "\n")


def load_geometry(path: str | Path) -> tuple[Geometry, dict[str, Any]]:
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text())
    with np.load(source / "geometry.npz", allow_pickle=False) as data:
        fields = ("raw_mutant", "raw_target", "encoded_target", "source_y",
                  "mean_basis", "mean_transform", "one_hot")
        geometry = Geometry(**{key: data[key].copy() if key in data.files else None for key in fields},
                            metadata=metadata)
    return geometry, {}


def median_distance(features: np.ndarray, config: Mapping[str, Any]) -> float:
    """Sample the outcome-blind distance scale used by the GP length bounds."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Distance geometry requires a finite matrix with at least two rows")
    count = len(values)
    rng = np.random.default_rng(int(config["distance_seed"]))
    pairs = int(config["distance_pair_count"])
    i = rng.integers(0, count, pairs)
    j = rng.integers(0, count - 1, pairs)
    j += j >= i
    # Bound peak memory for 1280-dimensional embeddings: a full pair matrix
    # would allocate several gigabytes for the default 100,000 distance pairs.
    distances = np.empty(pairs, dtype=np.float64)
    for start in range(0, pairs, 2048):
        stop = min(start + 2048, pairs)
        difference = values[i[start:stop]] - values[j[start:stop]]
        distances[start:stop] = np.sqrt(np.einsum("ij,ij->i", difference, difference))
    result = float(np.median(distances))
    if result <= 0 or not np.isfinite(result):
        raise ValueError("Outcome-blind median pairwise distance must be finite and positive")
    return result


def _euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Float64 Euclidean distances, accelerating large products through BLAS.

    The common translation and near-pair repair follow the selected development
    implementation used for the paper. This changes only the
    arithmetic used to evaluate the same distances, never the covariance model.
    Direct differences repair every cancellation-dominated pair (including all
    duplicate rows), using a dimension-aware roundoff bound. Temporary repair
    storage is limited to 4,096 pairs regardless of the eligible pool size.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]
            or left.shape[1] < 1 or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("Kernel features must be finite two-dimensional matrices with the same positive width")
    if not len(left) or not len(right):
        return np.empty((len(left), len(right)), dtype=np.float64)
    if len(left)*len(right)*left.shape[1] < 2_000_000:
        distances = cdist(left, right)
        if not np.isfinite(distances).all():
            raise ValueError("Feature magnitudes exceed finite float64 Euclidean-distance arithmetic")
        return distances
    try:
        with np.errstate(over="raise", invalid="raise"):
            origin = left[0]
            centered_left, centered_right = left-origin, right-origin
            left_norm = np.einsum("ij,ij->i", centered_left, centered_left)
            right_norm = np.einsum("ij,ij->i", centered_right, centered_right)
            squared = centered_left @ centered_right.T
            squared *= -2.0
            squared += left_norm[:, None]
            squared += right_norm[None, :]
            # A dot product accumulates roundoff with feature width. This bound
            # safely includes pairs whose tiny distance was lost by subtraction.
            bound = (8.0*left.shape[1]+64.0)*np.finfo(np.float64).eps
            for row in range(len(left)):
                suspect = np.flatnonzero(squared[row] <= bound*(left_norm[row]+right_norm))
                for start in range(0, len(suspect), 4096):
                    columns = suspect[start:start+4096]
                    delta = left[row]-right[columns]
                    squared[row, columns] = np.einsum("ij,ij->i", delta, delta)
            if not np.isfinite(squared).all():
                raise ValueError("Feature magnitudes exceed finite float64 Euclidean-distance arithmetic")
            # Every negative entry was included in the direct repair above.
            if np.any(squared < 0):
                raise ValueError("Direct Euclidean-distance repair produced a negative squared distance")
            np.sqrt(squared, out=squared)
            return squared
    except FloatingPointError as exc:
        raise ValueError("Feature magnitudes exceed finite float64 Euclidean-distance arithmetic") from exc


def matern52(left: np.ndarray, right: np.ndarray, amplitude: float, length_scale: float) -> np.ndarray:
    """Exact Matérn-5/2 covariance, with no feature approximation or shortlist."""
    amplitude, length_scale = float(amplitude), float(length_scale)
    if (not np.isfinite(amplitude) or amplitude < 0 or amplitude > np.sqrt(np.finfo(float).max)
            or not np.isfinite(length_scale) or length_scale <= 0):
        raise ValueError("Matérn amplitude must be finite and nonnegative, and length scale finite and positive")
    r = _euclidean_distances(left, right)
    if amplitude == 0:
        return np.zeros_like(r)
    with np.errstate(over="ignore", under="ignore"):
        r /= length_scale
        root5 = np.sqrt(5.0)*r
        # Ordinary distances retain the reference expression. Log evaluation at
        # large radius avoids inf*0 while preserving its mathematical limit.
        ordinary = root5 <= 50.0
        result = np.empty_like(r)
        correlation = (1+root5[ordinary]+root5[ordinary]**2/3)*np.exp(-root5[ordinary])
        result[ordinary] = amplitude**2*correlation
        finite_tail = ~ordinary & np.isfinite(root5)
        a = root5[finite_tail]
        log_polynomial = 2*np.log(a)-np.log(3.0)+np.log1p(3/a+3/a**2)
        result[finite_tail] = np.exp(2*np.log(amplitude)+log_polynomial-a)
        result[~np.isfinite(root5)] = 0.0
    return result


class ModelNumericalError(RuntimeError):
    """A recorded failed run; there is no silent fallback to another algorithm."""


def _gls(h: np.ndarray, y: np.ndarray, chol: np.ndarray, tolerance: float) -> dict[str, Any]:
    whitened_h = solve_triangular(chol, h, lower=True, check_finite=False)
    whitened_y = solve_triangular(chol, y, lower=True, check_finite=False)
    qmat, rmat, piv = qr(whitened_h, mode="economic", pivoting=True, check_finite=False)
    rank = int(np.count_nonzero(np.abs(np.diag(rmat)) > tolerance * abs(rmat[0, 0])))
    if rank != h.shape[1]:
        raise ModelNumericalError(f"Observed mean basis is rank deficient: {rank}/{h.shape[1]}")
    beta_pivot = solve_triangular(rmat, qmat.T @ whitened_y, lower=False, check_finite=False)
    beta = np.empty(h.shape[1]); beta[piv] = beta_pivot
    inverse_r = solve_triangular(rmat, np.eye(h.shape[1]), lower=False, check_finite=False)
    beta_covariance = np.empty((h.shape[1], h.shape[1]))
    beta_covariance[np.ix_(piv, piv)] = inverse_r @ inverse_r.T
    return {"beta": beta, "beta_covariance": beta_covariance,
            "residual": y - h @ beta, "rank": rank,
            "log_information": float(2 * np.log(np.abs(np.diag(rmat))).sum()),
            "information_condition": float(np.linalg.cond(whitened_h)**2)}


def _chol(kernel: np.ndarray, noise_var: float, config: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    scale = float(np.mean(np.diag(kernel)))
    factor = float(config["gp_jitter_initial_factor"])
    while factor <= float(config["gp_jitter_max_factor"]) * (1 + 1e-12):
        try:
            jitter = factor * scale
            return cholesky(kernel + (noise_var + jitter) * np.eye(len(kernel)), lower=True,
                            check_finite=False), jitter
        except np.linalg.LinAlgError:
            factor *= 10
    raise ModelNumericalError("Observed covariance Cholesky failed within the declared jitter schedule")


def _solve(chol: np.ndarray, values: np.ndarray) -> np.ndarray:
    first = solve_triangular(chol, values, lower=True, check_finite=False)
    return solve_triangular(chol.T, first, lower=False, check_finite=False)


@dataclass
class FittedGP:
    method: str
    geometry: Geometry
    observed: np.ndarray
    features: np.ndarray
    mean_design: np.ndarray
    config: dict[str, Any]
    amplitude: float
    length_scale: float
    noise_std: float
    jitter: float
    chol: np.ndarray
    beta: np.ndarray
    beta_covariance: np.ndarray
    residual_weights: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def observation_covariance(self) -> np.ndarray:
        return self.chol @ self.chol.T

    @property
    def noise_variance(self) -> float:
        return self.noise_std**2

    def _parts(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        cross = matern52(self.features[selected], self.features[self.observed], self.amplitude, self.length_scale)
        weights = _solve(self.chol, cross.T).T
        basis_residual = self.mean_design[selected] - weights @ self.mean_design[self.observed]
        return cross, weights, basis_residual

    def predict(self, indices: Sequence[int] | None = None, full_cov: bool = False
                ) -> tuple[np.ndarray, np.ndarray]:
        selected = np.arange(len(self.features)) if indices is None else np.asarray(indices, dtype=np.int64)
        if not full_cov and len(selected) > int(self.config["prediction_batch_size"]):
            batch = int(self.config["prediction_batch_size"])
            pieces = [self.predict(selected[start:start+batch], False) for start in range(0, len(selected), batch)]
            return np.concatenate([p[0] for p in pieces]), np.concatenate([p[1] for p in pieces])
        cross, weights, bstar = self._parts(selected)
        mean = self.mean_design[selected] @ self.beta + cross @ self.residual_weights
        if full_cov:
            covariance = (matern52(self.features[selected], self.features[selected], self.amplitude, self.length_scale)
                          - weights @ cross.T + bstar @ self.beta_covariance @ bstar.T)
            covariance = 0.5 * (covariance + covariance.T)
            variance = np.diag(covariance)
        else:
            variance = self.amplitude**2 - np.einsum("ij,ij->i", weights, cross)
            variance += np.einsum("ij,ij->i", bstar @ self.beta_covariance, bstar)
            covariance = variance
        tolerance = 1e-8 * max(1.0, self.amplitude**2)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or np.min(variance) < -tolerance:
            raise ModelNumericalError("Nonfinite or materially negative GP posterior variance")
        if not full_cov:
            covariance = np.maximum(covariance, 0)
        return mean, covariance


    def exact_factor(self, indices: Sequence[int]) -> tuple[np.ndarray, dict[str, Any]]:
        from .exact_sampling import exact_posterior_factor
        return exact_posterior_factor(self, indices)


    def select_batch(self, eligible_indices: Sequence[int], batch_size: int,
                     rng: np.random.Generator, sampler: str = "exact",
                     slot_seeds: Sequence[int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        pool = _candidate_pool(eligible_indices, self.observed, len(self.features), batch_size)
        if slot_seeds is not None and len(slot_seeds) < batch_size:
            raise ValueError("One separate acquisition seed is required per batch slot")
        if sampler != "exact":
            raise ValueError("The paper protocol requires exact joint posterior sampling")
        mean, _ = self.predict(pool)
        factor, factor_metadata = self.exact_factor(pool)
        chosen, slots = select_factor_mean_fantasy(mean, factor, pool, batch_size, rng,
                                                 self.noise_variance, slot_seeds=slot_seeds)
        return chosen, {"sampler": sampler, "factor_rank": factor.shape[1],
                        "candidate_pool_size": len(pool), "factorization": factor_metadata, "slots": slots}


def fit_gp(method: str, geometry: Geometry, observed: np.ndarray, y: np.ndarray,
           seed: int, config: Mapping[str, Any]) -> FittedGP:
    features = geometry.encoded_target if method in ("full", "encoder_only") else geometry.raw_target
    if features is None:
        raise ValueError(f"{method} requires a frozen source-trained encoder")
    features = np.asarray(features, dtype=np.float64)
    h_all = geometry.mean_basis if method in ("full", "mean_only") else np.ones((len(features), 1))
    h = h_all[observed]
    if len(y) <= h.shape[1]:
        raise ModelNumericalError("REML requires n_observed > effective mean rank")
    # The complete-library feature distance scale is outcome blind and fixed.
    geometry_key = "encoded_median_distance" if method in ("full", "encoder_only") else "raw_median_distance"
    # Geometry is immutable during fitting: campaign cache identities must not
    # depend on which method happened to fit first. Old/manual geometries get
    # a local fallback calculation, never an in-place metadata mutation.
    d_med = float(geometry.metadata[geometry_key]) if geometry_key in geometry.metadata else median_distance(features, config)
    distances = cdist(features[observed], features[observed])
    bounds_list = [config["gp_amplitude_bounds"], np.asarray(config["gp_length_relative_bounds"]) * d_med]
    fit_noise = bool(config["gp_fit_noise"])
    if fit_noise:
        bounds_list.append(config["gp_noise_bounds"])
    bounds = np.log(np.asarray(bounds_list, dtype=np.float64))
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("GP log bounds must be finite, strictly ordered, and positive")
    def evaluate(log_parameters: np.ndarray) -> dict[str, Any]:
        parameters = np.exp(log_parameters)
        tau, ell = map(float, parameters[:2])
        noise = float(parameters[2]) if fit_noise else float(config["gp_nugget_std"])
        scaled = distances / ell
        kernel = tau*tau * (1 + np.sqrt(5)*scaled + 5*scaled*scaled/3) * np.exp(-np.sqrt(5)*scaled)
        chol, jitter = _chol(kernel, noise*noise, config)
        regression = _gls(h, y, chol, float(config["mean_rank_relative_tolerance"]))
        white = solve_triangular(chol, regression["residual"], lower=True, check_finite=False)
        objective = 0.5 * (2*np.log(np.diag(chol)).sum() + regression["log_information"]
                           + float(white @ white) + (len(y)-h.shape[1])*np.log(2*np.pi))
        if not np.isfinite(objective):
            raise ModelNumericalError("REML objective is nonfinite")
        return {"objective": float(objective), "amplitude": tau, "length_scale": ell,
                "noise_std": noise, "chol": chol, "jitter": jitter, **regression}
    def objective(parameters: np.ndarray) -> float:
        try:
            return evaluate(parameters)["objective"]
        except (ValueError, np.linalg.LinAlgError, ModelNumericalError):
            return 1e100
    sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=int(seed) % (2**32))
    count = int(config["gp_sobol_starts"])
    units = sampler.random_base2(int(np.log2(count))) if count & (count-1) == 0 else sampler.random(count)
    starts = bounds[:, 0] + units * (bounds[:, 1]-bounds[:, 0])
    restarts, states = [], []
    for index, start in enumerate(starts):
        fit = minimize(objective, start, method="L-BFGS-B", bounds=[tuple(row) for row in bounds],
                       options={"maxiter": int(config["gp_optimizer_maxiter"])})
        record = {"restart": index, "start_log_parameters": start.tolist(), "optimizer_success": bool(fit.success),
                  "message": str(fit.message), "iterations": int(fit.nit), "function_evaluations": int(fit.nfev)}
        try:
            state = evaluate(fit.x)
            states.append(state)
            record.update({"numerically_valid": True, "objective": state["objective"],
                           "parameters": [state["amplitude"], state["length_scale"], state["noise_std"]],
                           "jitter": state["jitter"]})
        except (ValueError, np.linalg.LinAlgError, ModelNumericalError) as exc:
            record.update({"numerically_valid": False, "failure": str(exc)})
        restarts.append(record)
    if not states:
        raise ModelNumericalError("Every bounded REML restart failed; no fallback method was selected")
    best = min(states, key=lambda row: row["objective"])
    raw_beta = geometry.mean_transform @ best["beta"] if method in ("full", "mean_only") else best["beta"]
    diagnostics = {"estimator": "REML_exact_unpenalized_signed_GLS", "kernel": "isotropic_Matern52",
        "objective": best["objective"], "amplitude_sd": best["amplitude"], "length_scale": best["length_scale"],
        "noise_sd": best["noise_std"], "noise_policy": "fitted_homoscedastic" if fit_noise else "fixed_declared_nugget",
        "numerical_jitter": best["jitter"], "tau_zero_face": False,
        "amplitude_lower_boundary": bool(np.isclose(best["amplitude"], config["gp_amplitude_bounds"][0], rtol=1e-5)),
        "rank": best["rank"], "n_minus_rank": len(y)-best["rank"],
        "information_condition": best["information_condition"], "observed_design_condition": float(np.linalg.cond(h)),
        "beta_basis": best["beta"].tolist(), "beta_raw_minimum_norm": raw_beta.tolist(),
        "raw_beta_identifiable": geometry.metadata["source_mean"]["raw_coefficients_identifiable"] if method in ("full", "mean_only") else True,
        "d_med": d_med, "restarts": restarts, "training_seed": int(seed),
        "selected_endpoint_converged": any(r["optimizer_success"] and r.get("objective") == best["objective"] for r in restarts),
        "nonconvergence_policy": "retain_finite_endpoint_and_flag"}
    return FittedGP(method, geometry, observed.copy(), features, h_all, dict(config),
                    best["amplitude"], best["length_scale"], best["noise_std"], best["jitter"], best["chol"],
                    best["beta"], best["beta_covariance"], _solve(best["chol"], best["residual"]), diagnostics)




def _candidate_pool(eligible: Sequence[int], observed: np.ndarray, count: int, batch_size: int) -> np.ndarray:
    pool = np.asarray(eligible, dtype=np.int64)
    if pool.ndim != 1 or len(np.unique(pool)) != len(pool) or batch_size < 0 or batch_size > len(pool):
        raise ValueError("Malformed candidate pool or batch size")
    if np.any(pool < 0) or np.any(pool >= count) or np.intersect1d(pool, observed).size:
        raise ValueError("Candidate pool contains out-of-range or already-observed indices")
    # The caller supplies canonical mutant-ID order. Preserve it so lexical ID
    # ties do not silently turn into integer embedding-row ties.
    return pool.copy()


def select_factor_mean_fantasy(mean: np.ndarray, factor: np.ndarray, pool: np.ndarray,
                               batch_size: int, rng: np.random.Generator, noise_variance: float,
                               *, slot_seeds: Sequence[int] | None = None
                               ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Exact mean fantasies with O(batch_size * factor_rank) auxiliary storage.

    The weight square root is L=P_0 ... P_s, where P_i=I-a_i v_i v_i'
    is a symmetric rank-one contraction. Store its vectors instead of an R by R
    matrix. Apply contractions in reverse order for L w and forward order for
    L' g. This preserves Q_new=Q-Q g g' Q/(g' Q g+noise_variance), with Q=L L'.
    """
    factor = np.asarray(factor, dtype=np.float64)
    contractions: list[tuple[np.ndarray, float]] = []
    blocked = np.zeros(len(pool), dtype=bool)
    selected, records = [], []
    for slot in range(batch_size):
        slot_rng = rng if slot_seeds is None else np.random.Generator(np.random.Philox(int(slot_seeds[slot])))
        random_weights = slot_rng.normal(size=factor.shape[1])
        for direction, contraction in reversed(contractions):
            random_weights -= contraction * float(direction @ random_weights) * direction
        scores = np.asarray(mean) + factor @ random_weights
        scores[blocked] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            raise ModelNumericalError("Acquisition produced no finite eligible candidate score")
        projected = factor[best].copy()
        for direction, contraction in contractions:
            projected -= contraction * float(direction @ projected) * direction
        variance = float(projected @ projected)
        total = variance + float(noise_variance)
        skipped = total <= np.finfo(float).tiny
        if not skipped and variance > 0:
            contraction = 1 - np.sqrt(max(float(noise_variance), 0) / total)
            contractions.append((projected / np.sqrt(variance), float(contraction)))
        selected.append(int(pool[best]))
        blocked[best] = True
        records.append({"slot": slot, "selected_index": int(pool[best]), "selected_score": float(scores[best]),
                        "mean_fantasy": float(mean[best]), "latent_variance_before": variance,
                        "fantasy_noise_variance": float(noise_variance), "zero_variance_update_skipped": skipped,
                        "seed": None if slot_seeds is None else int(slot_seeds[slot])})
    return np.asarray(selected, dtype=np.int64), records


@dataclass
class FittedComparator:
    method: str
    observed: np.ndarray
    candidate_count: int
    predictions: np.ndarray | None
    member_predictions: np.ndarray | None
    diagnostics: dict[str, Any]

    def predict(self, indices: Sequence[int] | None = None, full_cov: bool = False
                ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.predictions is None:
            return None, None
        selected = np.arange(self.candidate_count) if indices is None else np.asarray(indices, dtype=np.int64)
        # Ensemble disagreement is descriptive, never a calibrated GP interval.
        uncertainty = None
        if self.member_predictions is not None:
            members = self.member_predictions[:, selected]
            uncertainty = np.var(members, axis=0, ddof=0)
            if full_cov:
                centered = members - members.mean(axis=0)
                uncertainty = centered.T @ centered / len(members)
        return self.predictions[selected], uncertainty

    def select_batch(self, eligible_indices: Sequence[int], batch_size: int,
                     rng: np.random.Generator, sampler: str = "exact",
                     slot_seeds: Sequence[int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        pool = _candidate_pool(eligible_indices, self.observed, self.candidate_count, batch_size)
        if self.method == "random":
            selected, remaining = [], pool.copy()
            for slot in range(batch_size):
                draw_rng = rng if slot_seeds is None else np.random.Generator(np.random.Philox(int(slot_seeds[slot])))
                position = int(draw_rng.integers(len(remaining)))
                selected.append(int(remaining[position]))
                remaining = np.delete(remaining, position)
            return np.asarray(selected, dtype=np.int64), {"acquisition": "uniform_without_replacement"}
        if self.method == "evolvepro":
            order = np.argsort(-self.predictions[pool], kind="stable")
            return pool[order[:batch_size]], {"acquisition": "greedy_topN_mean", "within_batch_retraining": False}
        if self.method != "alde" or self.member_predictions is None:
            raise ValueError(f"Unknown comparator {self.method}")
        selected, records = [], []
        blocked = np.zeros(len(pool), dtype=bool)
        for slot in range(batch_size):
            draw_rng = rng if slot_seeds is None else np.random.Generator(np.random.Philox(int(slot_seeds[slot])))
            member = int(draw_rng.integers(len(self.member_predictions)))
            scores = self.member_predictions[member, pool].copy()
            scores[blocked] = -np.inf
            best = int(np.argmax(scores))
            selected.append(int(pool[best])); blocked[best] = True
            records.append({"slot": slot, "member": member, "selected_index": int(pool[best]),
                            "seed": None if slot_seeds is None else int(slot_seeds[slot])})
        return np.asarray(selected), {"acquisition": "uniform_fresh_ensemble_member_per_slot",
                                      "mean_fantasies": False, "slots": records}


def _fit_alde(geometry: Geometry, observed: np.ndarray, y: np.ndarray,
              seed: int, config: Mapping[str, Any]) -> FittedComparator:
    import torch
    from torch import nn
    from sklearn.model_selection import train_test_split
    if geometry.one_hot is None:
        raise ValueError("ALDE requires mutant_sequences passed to prepare_geometry (seven-site one-hot)")
    maximum = float(np.max(y))
    maximum_absolute = float(np.max(np.abs(y)))
    denominator = maximum if maximum > 1e-8 else (maximum_absolute if maximum_absolute > 1e-8 else 1.0)
    seed_rng = np.random.Generator(np.random.Philox(seed))
    predictions, members = [], []
    hidden = int(config["alde_hidden"])
    for member in range(int(config["alde_ensemble_size"])):
        member_seed = int(seed_rng.integers(0, 2**32-1))
        train_rows, _ = train_test_split(np.arange(len(observed)), test_size=0.1,
                                               shuffle=True, random_state=member_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(member_seed)
            network = nn.Sequential(nn.Linear(geometry.one_hot.shape[1], hidden), nn.LeakyReLU(0.01),
                                    nn.Linear(hidden, hidden), nn.LeakyReLU(0.01), nn.Linear(hidden, 1))
        network.to(device="cpu", dtype=torch.float32)
        optimizer = torch.optim.Adam(network.parameters(), lr=float(config["alde_learning_rate"]),
                                     weight_decay=0.0)
        x_train = torch.from_numpy(geometry.one_hot[observed[train_rows]])
        y_train = torch.as_tensor(y[train_rows] / denominator, dtype=torch.float32)
        best_loss, stale, best_epoch = float("inf"), 0, 0
        epoch_losses = []
        for epoch in range(int(config["alde_max_epochs"])):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(network(x_train).squeeze(-1), y_train)
            if not torch.isfinite(loss):
                raise ModelNumericalError(f"ALDE member {member} has nonfinite training loss")
            loss.backward(); optimizer.step()
            network.eval()
            with torch.inference_mode():
                monitored = float(nn.functional.mse_loss(network(x_train).squeeze(-1), y_train))
            if not np.isfinite(monitored):
                raise ModelNumericalError("ALDE has nonfinite monitored loss")
            epoch_losses.append(monitored)
            if monitored < best_loss:
                best_loss, stale, best_epoch = monitored, 0, epoch + 1
            else:
                stale += 1
            if stale >= int(config["alde_patience"]):
                break
        with torch.inference_mode():
            prediction = network(torch.from_numpy(geometry.one_hot)).squeeze(-1).numpy().astype(np.float64) * denominator
        if not np.isfinite(prediction).all():
            raise ModelNumericalError("ALDE predictions are nonfinite")
        predictions.append(prediction)
        members.append({"member": member, "seed": member_seed, "train_indices": observed[train_rows].tolist(),
                        "epochs": len(epoch_losses),
                        "best_epoch": best_epoch, "best_training_loss": best_loss,
                        "final_training_loss": epoch_losses[-1], "training_loss_history": epoch_losses,
                        "restored_best_weights": False})
    prediction_matrix = np.vstack(predictions)
    return FittedComparator("alde", observed, geometry.candidate_count, prediction_matrix.mean(axis=0),
                            prediction_matrix, {"architecture": [140, hidden, hidden, 1],
        "activation": "LeakyReLU(0.01)", "activation_reason": "paper_reference_implementation",
        "optimizer": "Adam", "learning_rate": config["alde_learning_rate"], "weight_decay": 0.0,
        "dropout": 0.0, "members": members, "outcome_denominator": denominator,
        "target_scale_rule": "positive_observed_max_else_maxabs_else_one", "device": "cpu",
        "member_subset": "90_percent_without_replacement",
        "monitor": "post_step_training_MSE", "bootstrap": False, "source_information_used": False,
        "predictive_uncertainty": "uncalibrated_ensemble_disagreement", "training_seed": int(seed)})


def fit_model(method: str, geometry: Geometry, observed_indices: Sequence[int],
              observed_y: Sequence[float], seed: int, config: Mapping[str, Any] | None = None
              ) -> FittedGP | FittedComparator:
    """Fit from a reveal-only target slice. Hidden truth is absent from this API."""
    start = perf_counter()
    cfg = model_config(config)
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {METHODS}")
    observed = np.asarray(observed_indices, dtype=np.int64)
    y = np.asarray(observed_y, dtype=np.float64)
    if observed.ndim != 1 or y.shape != observed.shape or len(np.unique(observed)) != len(observed):
        raise ValueError("Observed indices and revealed labels must be aligned unique vectors")
    if np.any(observed < 0) or np.any(observed >= geometry.candidate_count) or not np.isfinite(y).all():
        raise ValueError("Observed rows are out of range or labels are nonfinite")
    if method in GP_METHODS:
        result = fit_gp(method, geometry, observed, y, seed, cfg)
    elif method == "alde":
        result = _fit_alde(geometry, observed, y, seed, cfg)
    elif method == "evolvepro":
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(n_estimators=int(cfg["rf_estimators"]), criterion="friedman_mse",
            max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
            oob_score=False, n_jobs=int(cfg["rf_n_jobs"]), random_state=int(seed) % (2**32), verbose=0,
            warm_start=False, ccp_alpha=0.0, max_samples=None)
        estimator.fit(geometry.raw_mutant[observed], y)
        prediction = np.asarray(estimator.predict(geometry.raw_mutant), dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise ModelNumericalError("EVOLVEpro-style RF predictions are nonfinite")
        result = FittedComparator(method, observed, geometry.candidate_count, prediction, None,
            {"estimator": "RandomForestRegressor", "parameters": estimator.get_params(),
             "features": "raw_ESM2_650M_seven_position_mean_embedding", "source_information_used": False,
             "predictive_uncertainty": None, "training_seed": int(seed)})
    else:
        result = FittedComparator(method, observed, geometry.candidate_count, None, None,
                                  {"source_information_used": False, "predictive_uncertainty": None})
    result.diagnostics.update({"method": method, "observed_count": len(observed),
                               "fit_seconds": perf_counter()-start})
    return result
