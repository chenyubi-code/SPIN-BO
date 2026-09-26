"""Load the paper sequence library and locally generated ESM2 features."""
from __future__ import annotations

import csv
import numpy as np

from .design import SOURCE_IDS
from .util import ROOT


def load_inputs(config):
    embedding = ROOT / config["embedding_dir"]
    sequence_path = ROOT / config["sequence_csv"]
    with sequence_path.open(newline="") as stream:
        registry = list(csv.DictReader(stream))
    candidates = [row for row in registry if row["sequence_type"] == "candidate"]
    partners = [row for row in registry if row["sequence_type"] == "partner"]
    if len(candidates) != 7715 or len(partners) != 9:
        raise ValueError("The experiment requires 7715 candidate sequences and 9 partners.")
    if [row["partner_id"] for row in partners] != [*SOURCE_IDS, "MMP10"]:
        raise ValueError("Partner order must be the eight paper sources followed by MMP10.")
    if len({row["sequence_id"] for row in registry}) != len(registry):
        raise ValueError("Sequence IDs must be unique.")

    arrays = {}
    for role, rows in (("candidate", candidates), ("partner", partners)):
        with (embedding / f"{role}_index.csv").open(newline="") as stream:
            index = list(csv.DictReader(stream))
        matrix = np.load(embedding / f"{role}_embeddings.npy", allow_pickle=False)
        if matrix.shape != (len(rows), 1280) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            raise ValueError(f"Expected finite float32 {role} features with shape ({len(rows)}, 1280).")
        row_ids = {row["sequence_id"] for row in rows}
        index_ids = [row["sequence_id"] for row in index]
        if len(index_ids) != len(rows) or set(index_ids) != row_ids:
            raise ValueError(f"The {role} feature index must contain each sequence exactly once.")
        if [int(row["embedding_row"]) for row in index] != list(range(len(index))):
            raise ValueError("Embedding index rows must be contiguous and match the matrix.")
        lookup = {row["sequence_id"]: i for i, row in enumerate(index)}
        arrays[role] = matrix[[lookup[row["sequence_id"]] for row in rows]]

    return {
        "mutant_raw": arrays["candidate"], "partner_raw": arrays["partner"],
        "source_partner_reference_raw": arrays["partner"][:len(SOURCE_IDS)].copy(),
        "source_partner_reference_ids": list(SOURCE_IDS),
        "candidates": candidates, "partners": partners,
        "ids": np.asarray([row["sequence_id"] for row in candidates]),
        "sequences": [row["sequence"] for row in candidates],
        "partner_lookup": {row["partner_id"]: i for i, row in enumerate(partners)},
        "cache_inputs": {"sequence_csv": config["sequence_csv"], "embedding_dir": config["embedding_dir"]},
    }
