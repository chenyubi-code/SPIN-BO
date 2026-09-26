from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stbo_single.data import DatasetBundle
from stbo_single.gp import distance_matrix, median_pairwise_distance


AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class ComparatorFeatures:
    raw_mutant_esm: np.ndarray
    target_gp_scaled_esm: np.ndarray
    target_gp_rms: float
    target_gp_distances: np.ndarray
    target_gp_d_med: float
    alde_onehot: np.ndarray
    alde_positions: np.ndarray

    def metadata(self) -> dict[str, object]:
        return {
            "raw_mutant_esm": {
                "shape": list(self.raw_mutant_esm.shape),
            },
            "target_gp": {
                "global_rms": self.target_gp_rms,
                "scaled_shape": list(self.target_gp_scaled_esm.shape),
                "d_med": self.target_gp_d_med,
            },
            "alde": {
                "positions": self.alde_positions.tolist(),
                "amino_acid_order": AMINO_ACID_ORDER,
                "shape": list(self.alde_onehot.shape),
            },
        }


def build_alde_onehot(dataset: DatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    positions = np.unique(np.asarray(dataset.mutation_positions, dtype=int))
    positions.sort()
    if positions.size == 0:
        raise ValueError("ALDE requires at least one mutable position")
    amino_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACID_ORDER)}
    tensor = np.zeros(
        (dataset.candidate_count, positions.size, len(AMINO_ACID_ORDER)),
        dtype=np.float32,
    )
    for row, sequence in enumerate(dataset.mutant_sequences):
        if not set(sequence).issubset(AMINO_ACID_ORDER):
            raise ValueError(f"candidate {row} contains a noncanonical amino acid")
        for column, position in enumerate(positions):
            if not 1 <= int(position) <= len(sequence):
                raise ValueError(f"ALDE position {position} is outside candidate {row}")
            amino_acid = sequence[int(position) - 1]
            tensor[row, column, amino_index[amino_acid]] = 1.0
    flattened = tensor.reshape(dataset.candidate_count, -1, order="C")
    return flattened, positions


def prepare_comparator_features(
    dataset: DatasetBundle, raw_mutant_esm: np.ndarray
) -> ComparatorFeatures:
    raw = np.asarray(raw_mutant_esm, dtype=np.float32)
    if raw.shape != (dataset.candidate_count, 1280):
        raise ValueError(
            f"mutant ESM matrix must have shape ({dataset.candidate_count},1280)"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("mutant ESM matrix contains NaN/Inf")
    squared_sum = np.sum(np.square(raw, dtype=np.float64), dtype=np.float64)
    rms = float(np.sqrt(squared_sum / raw.size))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError("target-only GP global RMS is nonfinite/nonpositive")
    scaled = (raw / rms).astype(np.float64)
    distances = distance_matrix(scaled)
    d_med = median_pairwise_distance(distances=distances)
    onehot, positions = build_alde_onehot(dataset)
    return ComparatorFeatures(
        raw_mutant_esm=raw,
        target_gp_scaled_esm=scaled,
        target_gp_rms=rms,
        target_gp_distances=distances,
        target_gp_d_med=d_med,
        alde_onehot=onehot,
        alde_positions=positions,
    )

