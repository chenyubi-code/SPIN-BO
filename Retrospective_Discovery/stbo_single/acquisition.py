from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import StudyConfig
from .io_utils import derive_uint64


@dataclass(frozen=True)
class BatchSelection:
    indices: tuple[int, ...]


def _stable_fantasy_covariance(
    covariance: np.ndarray,
    *,
    relative_tolerance: float,
    context: str,
) -> np.ndarray:
    """Symmetrize the latent covariance while rejecting material PSD errors."""

    raw = np.asarray(covariance, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise RuntimeError(f"{context} must be square")
    if raw.shape[0] == 0:
        return raw.copy()
    if not np.all(np.isfinite(raw)):
        raise RuntimeError(f"{context} contains NaN/Inf")
    entry_scale = max(float(np.max(np.abs(raw))), 1.0)
    symmetry_max_abs = float(np.max(np.abs(raw - raw.T)))
    if symmetry_max_abs > float(relative_tolerance) * entry_scale:
        raise RuntimeError(
            f"{context} is materially asymmetric: max_abs={symmetry_max_abs}"
        )
    values = 0.5 * (raw + raw.T)
    eigenvalues = np.linalg.eigvalsh(values)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    minimum_diagonal = float(np.min(np.diag(values)))
    minimum_eigenvalue = float(eigenvalues[0])
    threshold = -float(relative_tolerance) * spectral_scale
    if minimum_diagonal < threshold:
        raise RuntimeError(
            f"{context} has a materially negative variance: {minimum_diagonal}"
        )
    if minimum_eigenvalue < threshold:
        raise RuntimeError(
            f"{context} is materially indefinite: min_eigenvalue={minimum_eigenvalue}"
        )
    return values


def stable_best(pool: np.ndarray, scores: np.ndarray, pair_ids: np.ndarray) -> int:
    indices = np.asarray(pool, dtype=int)
    values = np.asarray(scores, dtype=np.float64)
    identifiers = np.asarray(pair_ids, dtype=str)
    if indices.ndim != 1 or values.shape != indices.shape:
        raise ValueError("pool and score vectors do not align")
    if indices.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("selection scores must be nonempty and finite")
    order = np.lexsort((identifiers[indices], -values))
    return int(indices[order[0]])


def _joint_draw(
    mean: np.ndarray,
    covariance: np.ndarray,
    rng: np.random.Generator,
    config: StudyConfig,
) -> np.ndarray:
    covariance = 0.5 * (
        np.asarray(covariance, dtype=np.float64)
        + np.asarray(covariance, dtype=np.float64).T
    )
    mean = np.asarray(mean, dtype=np.float64)
    if covariance.shape != (mean.size, mean.size):
        raise ValueError("joint TS covariance has the wrong shape")
    diagonal = np.diag(covariance)
    scale = max(float(np.mean(np.clip(diagonal, 0.0, None))), 1.0)
    identity = np.eye(mean.size, dtype=np.float64)
    factor = float(config.ts_jitter_initial_factor)
    while factor <= config.ts_jitter_max_factor * (1.0 + 16.0 * np.finfo(float).eps):
        jitter = factor * scale
        try:
            root = np.linalg.cholesky(covariance + jitter * identity)
            sample = mean + root @ rng.standard_normal(mean.size)
            return sample
        except np.linalg.LinAlgError:
            factor *= 10.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), scale, 1.0)
    minimum = float(eigenvalues[0])
    if minimum < -config.ts_negative_eigen_relative_tolerance * spectral_scale:
        raise RuntimeError(
            f"latent covariance is materially indefinite: min_eigenvalue={minimum}"
        )
    clipped = np.clip(eigenvalues, 0.0, None)
    root = eigenvectors * np.sqrt(clipped)[None, :]
    sample = mean + root @ rng.standard_normal(mean.size)
    return sample


def select_slotwise_joint_ts(
    mean: np.ndarray,
    covariance: np.ndarray,
    noise_std: float,
    observed_indices: np.ndarray,
    feasible: np.ndarray,
    pair_ids: np.ndarray,
    *,
    batch_size: int,
    namespace: str,
    method_id: str,
    campaign_seed: int,
    round_id: int,
    config: StudyConfig,
) -> BatchSelection:
    posterior_mean = np.asarray(mean, dtype=np.float64)
    fantasy_covariance = _stable_fantasy_covariance(
        covariance,
        relative_tolerance=config.ts_negative_eigen_relative_tolerance,
        context="initial latent posterior covariance",
    )
    feasible_mask = np.asarray(feasible, dtype=bool)
    identifiers = np.asarray(pair_ids, dtype=str)
    if fantasy_covariance.shape != (posterior_mean.size, posterior_mean.size):
        raise ValueError("posterior covariance has the wrong shape")
    if feasible_mask.shape != posterior_mean.shape or identifiers.shape != posterior_mean.shape:
        raise ValueError("feasibility or pair IDs do not align with the posterior")
    blocked = set(int(value) for value in np.asarray(observed_indices, dtype=int))
    pending: list[int] = []
    noise_variance = float(noise_std) ** 2

    for slot in range(1, batch_size + 1):
        pool = np.asarray(
            [
                index
                for index in range(posterior_mean.size)
                if feasible_mask[index] and index not in blocked
            ],
            dtype=int,
        )
        if pool.size < batch_size - slot + 1:
            raise RuntimeError("remaining feasible pool cannot fill the pending batch")
        subseed, _ = derive_uint64(
            namespace,
            method_id,
            int(campaign_seed),
            int(round_id),
            "slot_joint_latent_ts",
            slot,
        )
        rng = np.random.default_rng(subseed)
        pool_covariance = fantasy_covariance[np.ix_(pool, pool)]
        sample = _joint_draw(
            posterior_mean[pool], pool_covariance, rng, config
        )
        selected = stable_best(pool, sample, identifiers)
        variance = float(fantasy_covariance[selected, selected])
        denominator = variance + noise_variance
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("Kriging-believer conditioning denominator is nonpositive")
        remaining = pool[pool != selected]
        cross = fantasy_covariance[:, selected].copy()
        updated_covariance = fantasy_covariance - np.outer(cross, cross) / denominator
        remaining_covariance = _stable_fantasy_covariance(
            updated_covariance[np.ix_(remaining, remaining)],
            relative_tolerance=config.ts_negative_eigen_relative_tolerance,
            context=f"fantasy covariance after round {round_id}, slot {slot}",
        )
        fantasy_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        fantasy_covariance[np.ix_(remaining, remaining)] = remaining_covariance
        pending.append(selected)
        blocked.add(selected)
    return BatchSelection(indices=tuple(pending))
