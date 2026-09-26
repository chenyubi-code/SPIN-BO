from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import numpy as np

from .design import output_dir
from .generator import make_basis, make_landscape, pca_coordinates, teacher_coordinates
from .util import ROOT, digest, read_json, write_json, write_npz


@contextmanager
def lock(path):
    """Advisory process lock, automatically released on process exit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_cached_npz(path, identity):
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not path.exists() and not meta_path.exists():
        return None
    if not path.exists() or not meta_path.exists():
        raise ValueError(f"Incomplete artifact pair at {path}; move it aside before regenerating.")
    meta = read_json(meta_path)
    if meta["identity"] != digest(identity):
        raise ValueError(f"Cached inputs/configuration changed at {path}; use a new simulation_id/output directory.")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return arrays, meta


def save_cached_npz(path, arrays, identity, metadata):
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    if any(v.dtype.hasobject for v in arrays.values()):
        raise ValueError("Cached arrays must not contain pickled Python objects.")
    if any(not np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in "fc"):
        raise ValueError("Cached numeric arrays must be finite.")
    write_npz(path, **arrays)
    meta = {**metadata, "identity": digest(identity)}
    write_json(Path(path).with_suffix(".json"), meta)
    return arrays, meta


def cache_root(config):
    return ROOT / "Outputs/Cache" / config["simulation_id"]


def get_teacher(inputs, config):
    path = cache_root(config) / "teacher_coordinates.npz"
    ident = {"embeddings": inputs["cache_inputs"], "pca_dim": config["generator_pca_dim"],
             "pca_floor": config["generator_pca_floor"], "master_seed": config["master_seed"],
             "teacher_pcs": config["generator_teacher_pcs"],
             "teacher_standardization": "full_library_center_population_std"}
    with lock(path.with_suffix(".lock")):
        cached = load_cached_npz(path, ident)
        if cached:
            return cached
        arrays = pca_coordinates(inputs["mutant_raw"], config["generator_pca_dim"], config["generator_pca_floor"])
        active, center, scale = teacher_coordinates(arrays["coordinates"], config)
        arrays.update(teacher_coordinates=active,
                      teacher_coordinate_mean=center, teacher_coordinate_scale=scale,
                      teacher_pc_indices=np.asarray(config["generator_teacher_pcs"], dtype=np.int64))
        return save_cached_npz(path, arrays, ident, {"primary_teacher_coordinates": "teacher_coordinates"})


def get_basis(inputs, config, landscape):
    teacher, teacher_meta = get_teacher(inputs, config)
    path = cache_root(config) / f"landscape_{landscape}/basis.npz"
    ident = {"teacher": teacher_meta["identity"], "landscape_seed": landscape,
             "master_seed": config["master_seed"], "features": config["generator_features"],
             "teacher_pcs": config["generator_teacher_pcs"],
             "frequency": config["generator_frequency"],
             "qr_absolute_tolerance": config["generator_qr_absolute_tolerance"]}
    with lock(path.with_suffix(".lock")):
        cached = load_cached_npz(path, ident)
        if cached:
            return cached
        arrays, meta = make_basis(teacher["coordinates"], config, landscape)
        return save_cached_npz(path, arrays, ident, meta)


def landscape_dir(c, config, landscape):
    return output_dir(c) / "Data" / config["simulation_id"] / c["id"] / f"landscape_{landscape}"


def get_landscape(inputs, config, c, landscape, basis=None, basis_meta=None):
    if basis is None:
        basis, basis_meta = get_basis(inputs, config, landscape)
    path = landscape_dir(c, config, landscape) / "landscape.npz"
    ident = {"basis": basis_meta["identity"], "condition": c,
             "master_seed": config["master_seed"],
             "source_identity_base_order": config["source_identity_base_order"],
             "source_identity_offsets": config["source_identity_offsets"],
             "landscape_seeds": config["landscape_seeds"], "inputs": inputs["cache_inputs"]}
    with lock(path.with_suffix(".lock")):
        cached = load_cached_npz(path, ident)
        if cached:
            return cached
        arrays, meta = make_landscape(basis, c, config, landscape)
        return save_cached_npz(path, arrays, ident, meta)


def common_initialization(inputs, config, landscape, campaign, basis):
    """Load the common initial assays used by all methods."""
    records = read_json(ROOT / "Configs/initializations.json")
    saved = next(r for r in records if r["landscape_seed"] == landscape and r["campaign_seed"] == campaign)
    chosen = np.asarray(saved["indices"], dtype=np.int64)
    if (chosen.shape != (config["n0"],) or len(np.unique(chosen)) != len(chosen)
            or np.any(chosen < 0) or np.any(chosen >= len(inputs["ids"]))):
        raise ValueError("Initial assays must be distinct candidate indices of the configured size")
    return saved
