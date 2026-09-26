from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ModelSnapshot:
    method_id: str
    predictive_mean: np.ndarray
    predictive_covariance: np.ndarray | None = None
    noise_std: float = 0.0
    member_predictions: np.ndarray | None = None


@dataclass(frozen=True)
class BatchDecision:
    indices: tuple[int, ...]


class SequentialMethod(Protocol):
    method_id: str

    def fit_predict(
        self,
        observed_indices: np.ndarray,
        observed_y: np.ndarray,
        *,
        seed: int,
        round_id: int,
    ) -> ModelSnapshot: ...

    def select_batch(
        self,
        snapshot: ModelSnapshot,
        observed_indices: np.ndarray,
        feasible: np.ndarray,
        pair_ids: np.ndarray,
        *,
        seed: int,
        round_id: int,
        batch_size: int,
    ) -> BatchDecision: ...

    def configuration(self) -> dict[str, Any]: ...
