"""The 20 conditions and seven methods appearing in the paper."""
from __future__ import annotations

import copy
from pathlib import Path
from .util import ROOT, read_json

METHODS = ["full", "mean_only", "encoder_only", "target_only", "alde", "evolvepro", "random"]
GP_METHODS = METHODS[:4]
METHOD_LABELS = {
    "full": "SPIN-BO (Both)", "mean_only": "SPIN-BO (Response Transfer)",
    "encoder_only": "SPIN-BO (Representation Transfer)", "target_only": "Target-only BO",
    "alde": "ALDE-style", "evolvepro": "EVOLVEpro-style", "random": "Random search",
}
SOURCE_IDS = ["MMP1", "MMP2", "MMP3", "MMP7", "MMP8", "MMP9", "MMP13", "MMP14"]
SOURCE_IDENTITY_BASE_ORDER = ["MMP13", "MMP2", "MMP8", "MMP14", "MMP1", "MMP9", "MMP3", "MMP7"]
SOURCE_IDENTITY_OFFSETS = [0, 1, 3, 5, 7]
FOCAL = "main_r0p6_g0p9"


def tag(x):
    return f"{x:g}".replace(".", "p")


def conditions():
    return [dict(id=f"main_r{tag(rho)}_g{tag(gamma)}", block="main", family="primary",
                 S=8, k=2, rho=rho, gamma=gamma, sigma_source=0.0, sigma_target=0.0,
                 force_wt=False, enabled=True)
            for rho in (0.0, 0.3, 0.6, 0.9) for gamma in (0.0, 0.3, 0.7, 0.9, 1.0)]


def select(block="main", condition_id=None):
    if block != "main":
        raise ValueError("Only the manuscript's 20-condition main grid is distributed")
    if condition_id is not None:
        found = [c for c in conditions() if c["id"] == condition_id]
        if not found:
            raise ValueError(f"Unknown paper condition: {condition_id}")
        return found
    return conditions()


def source_order(master, landscape_index, config=None):
    """Same outcome-blind cyclic source assignment as the paper implementation."""
    config = config or {}
    names = config.get("source_identity_base_order", SOURCE_IDENTITY_BASE_ORDER)
    offsets = config.get("source_identity_offsets", SOURCE_IDENTITY_OFFSETS)
    if len(names) != 8 or set(names) != set(SOURCE_IDS):
        raise ValueError("Source schedule must contain all eight sources exactly once")
    if landscape_index < 0 or landscape_index >= len(offsets):
        raise ValueError("Landscape index is outside the frozen schedule")
    permutation = [SOURCE_IDS.index(name) for name in names]
    shift = (-offsets[landscape_index]) % len(permutation)
    return permutation[-shift:] + permutation[:-shift] if shift else permutation


def output_dir(c):
    return ROOT / "Outputs"


def load_config(path=None):
    config = read_json(Path(path) if path else ROOT / "Configs/paper.json")
    if config.get("schema_version") != 2:
        raise ValueError("Expected schema_version=2")
    return copy.deepcopy(config)
