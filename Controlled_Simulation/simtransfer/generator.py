"""Exact finite-library primary generator; teacher objects stay evaluator-side."""
from __future__ import annotations

import numpy as np
from scipy import linalg

from .design import SOURCE_IDS, source_order
from .util import rng


def pca_coordinates(x, dimension=64, floor=1e-6):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    # Exact thin SVD: all candidates, no response values.
    _, singular, vt = linalg.svd(centered, full_matrices=False, lapack_driver="gesdd")
    eigenvalues = singular**2 / len(x)
    if len(eigenvalues) < dimension or eigenvalues[dimension-1] <= eigenvalues[0]*1e-12:
        raise ValueError("Candidate embeddings lack 64 nonzero PCA directions.")
    components = vt[:dimension].copy()
    signs = np.where(components[np.arange(dimension), np.abs(components).argmax(axis=1)] < 0, -1., 1.)
    components *= signs[:, None]
    eigenvalues = eigenvalues[:dimension]
    coordinates = (centered @ components.T) / np.sqrt(eigenvalues + floor*eigenvalues[0])
    return {"coordinates": coordinates, "mean": mean, "components": components,
            "eigenvalues": eigenvalues, "floor": np.asarray(floor)}


def teacher_coordinates(coordinates, config):
    """Select and standardize PC4/PC8/PC12 using only public candidate features."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    pcs = np.asarray(config["generator_teacher_pcs"], dtype=int)
    if (coordinates.ndim != 2 or pcs.shape != (3,) or len(set(pcs.tolist())) != 3
            or np.any(pcs < 0) or np.any(pcs >= coordinates.shape[1])
            or not np.isfinite(coordinates).all()):
        raise ValueError("Teacher requires three distinct valid outcome-blind PCA columns.")
    selected = coordinates[:, pcs]
    mean = selected.mean(axis=0)
    scale = np.sqrt(np.mean((selected-mean)**2, axis=0))
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("Selected teacher PCA coordinates have zero variance.")
    return (selected-mean)/scale, mean, scale


def _teacher_orthonormal(a, tolerance=1e-9):
    """QR of the nine landscape functions with a fixed sign convention."""
    a = np.asarray(a, dtype=np.float64)
    a = a-a.mean(axis=0)
    q, r = linalg.qr(a, mode="economic")
    diagonal = np.diag(r)
    if len(diagonal) != 9 or np.any(np.abs(diagonal) < tolerance):
        raise ValueError("Numerically degenerate nine-function teacher; seed must be recorded, not retried.")
    return q*np.where(diagonal < 0, -1.0, 1.0), diagonal


def make_basis(coordinates, config, landscape):
    """Shared 3D spectral teacher, independent of every learned representation.

    ``feature_basis`` denotes the nine-function smooth span, not all cosines.
    """
    master, D = config["master_seed"], int(config["generator_features"])
    frequency = float(config["generator_frequency"])
    random = rng(master, "landscape_basis", landscape)
    q, coordinate_mean, coordinate_scale = teacher_coordinates(coordinates, config)
    n, d = q.shape
    if n <= 19 or D < 9 or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Teacher requires N>19, at least nine cosine features, and positive frequency.")
    frequencies = random.normal(size=(d, D))*frequency
    phases = random.uniform(0, 2*np.pi, D)
    loadings = random.normal(size=(D, 9))
    functions = np.cos(q @ frequencies + phases) @ loadings / np.sqrt(D)
    tolerance = float(config.get("generator_qr_absolute_tolerance", 1e-9))
    qe, _ = _teacher_orthonormal(functions, tolerance)
    smooth = np.sqrt(n)*qe
    z = random.normal(size=(n, 9))
    z -= z.mean(axis=0)
    # Two passes reduce projection roundoff for the rough subspace.
    for _ in range(2):
        z -= qe @ (qe.T @ z)
        z -= z.mean(axis=0)
    qv, _ = _teacher_orthonormal(z, tolerance)
    rough = np.sqrt(n)*qv
    arrays = {"smooth": smooth, "rough": rough, "feature_basis": qe,
              "teacher_coordinates": q, "coordinate_mean": coordinate_mean,
              "coordinate_scale": coordinate_scale,
              "frequencies": frequencies, "phases": phases, "loadings": loadings}
    return arrays, {"feature_rank": 9,
                    "generator_family": "shared_lowdim_spectral_v2",
                    "cosine_features": D, "coordinate_dimension": d,
                    "teacher_pcs": list(config["generator_teacher_pcs"]),
                    "frequency": frequency, "median_distance_bandwidth_used": False,
                    "rough_projection_dimension": 9,
                    "generator_uses_trained_encoder": False,
                    "partner_coefficient_rule": "source-profile assignment; no partner-geometry smoothness guarantee"}


def make_landscape(basis, c, config, landscape):
    if c["family"] != "primary":
        raise ValueError("Only the manuscript generator is supported.")
    gamma, rho, S, k = c["gamma"], c["rho"], c["S"], c["k"]
    u = np.sqrt(gamma)*basis["smooth"] + np.sqrt(1-gamma)*basis["rough"]
    target = u[:, 0].copy()
    strengths = np.zeros(S)
    strengths[:k] = rho
    source = target[:, None]*strengths + u[:, 1:S+1]*np.sqrt(1-strengths**2)
    signs = np.ones(S)
    # Core noise arrays are precisely zero, not random draws with tiny variance.
    if c["sigma_source"] != 0 or c["sigma_target"] != 0:
        raise ValueError("The paper protocol uses noiseless responses.")
    observed_source = source.copy()
    index = config["landscape_seeds"].index(landscape)
    order = source_order(config["master_seed"], index, config)
    meta = {"condition": c, "landscape_seed": landscape,
            "source_ids": [SOURCE_IDS[i] for i in order[:S]],
            "source_identity_indices": order[:S], "source_signs": signs,
            "informative_source_ids": [SOURCE_IDS[i] for i in order[:k]] if rho else [],
            "target_noise": "identically_zero"}
    return {"target_truth": target, "source_truth": source, "source_observed": observed_source,
            "source_noise": np.zeros_like(source), "target_noise": np.zeros_like(target)}, meta
