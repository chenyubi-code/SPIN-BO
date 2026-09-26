from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StudyConfig:
    """Frozen method and campaign settings for one single-target example."""

    initial_batch: int = 20
    batch_size: int = 20
    total_budget: int = 200
    seeds: tuple[int, ...] = tuple(range(10))
    recall_ks: tuple[int, ...] = (5, 20)
    study_namespace: str = "similar-target-equal-mutant-single-target-v1"
    initial_max_redraws: int = 1000

    encoder_seed: int = 20260826
    encoder_hidden: int = 128
    encoder_bottleneck: int = 32
    encoder_batch_size: int = 256
    encoder_epochs: int = 200
    encoder_learning_rate: float = 1.0e-3
    encoder_weight_decay: float = 1.0e-4
    encoder_gradient_clip: float = 5.0
    encoder_device: str = "auto"

    gp_amplitude_bounds: tuple[float, float] = (1.0e-3, 10.0)
    gp_length_relative_bounds: tuple[float, float] = (0.1, 10.0)
    gp_noise_bounds: tuple[float, float] = (0.03, 2.0)
    gp_sobol_starts: int = 16
    gp_optimizer_seed: int = 20260826
    gp_optimizer_maxiter: int = 500
    gp_jitter_initial_factor: float = 1.0e-7
    gp_jitter_max_factor: float = 1.0e-3
    mean_rank_relative_tolerance: float = 1.0e-10

    ts_jitter_initial_factor: float = 1.0e-10
    ts_jitter_max_factor: float = 1.0e-6
    ts_negative_eigen_relative_tolerance: float = 1.0e-8

    alde_ensemble_size: int = 5
    alde_hidden: int = 30
    alde_learning_rate: float = 1.0e-3
    alde_max_epochs: int = 300
    alde_patience: int = 30

    rf_estimators: int = 100
    top_tie_rule: str = "descending_value_then_lexicographic_pair_id"
    response_direction: str = "maximize"
    allow_missing_embedding_generation: bool = False
    embedding_batch_size: int = 8
    embedding_device: str = "auto"

    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, candidate_count: int | None = None) -> None:
        if self.initial_batch <= 0:
            raise ValueError("initial_batch must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.total_budget < self.initial_batch:
            raise ValueError("total_budget must be at least initial_batch")
        if (self.total_budget - self.initial_batch) % self.batch_size != 0:
            raise ValueError(
                "total_budget - initial_batch must be divisible by batch_size"
            )
        if candidate_count is not None and self.total_budget > candidate_count:
            raise ValueError(
                f"total budget {self.total_budget} exceeds {candidate_count} candidates"
            )
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("campaign seeds must be nonnegative")
        if self.initial_max_redraws < 0:
            raise ValueError("initial_max_redraws must be nonnegative")
        if any(k <= 0 for k in self.recall_ks):
            raise ValueError("recall cutoffs must be positive")
        if self.response_direction != "maximize":
            raise ValueError("this implementation is frozen to maximization")

    @property
    def budgets(self) -> tuple[int, ...]:
        return tuple(
            range(self.initial_batch, self.total_budget + 1, self.batch_size)
        )

    @property
    def adaptive_rounds(self) -> int:
        return (self.total_budget - self.initial_batch) // self.batch_size

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["recall_ks"] = list(self.recall_ks)
        return payload
