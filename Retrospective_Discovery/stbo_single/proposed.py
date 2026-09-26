from __future__ import annotations

from typing import Any

import numpy as np

from .acquisition import select_slotwise_joint_ts
from .config import StudyConfig
from .gp import fit_single_target_gp
from .protocol import BatchDecision, ModelSnapshot


OUR_METHOD_ID = "our_method_single_target_v1"


class ProposedMethod:
    """Proposal-faithful single-target transfer BO model and acquisition."""

    method_id = OUR_METHOD_ID

    def __init__(
        self,
        mean_design: np.ndarray,
        encoder_latent: np.ndarray,
        distances: np.ndarray,
        d_med: float,
        raw_from_basis_transform: np.ndarray,
        source_names: tuple[str, ...],
        config: StudyConfig,
    ) -> None:
        self.mean_design = np.asarray(mean_design, dtype=np.float64)
        self.representation = np.asarray(encoder_latent, dtype=np.float64)
        self.distances = np.asarray(distances, dtype=np.float64)
        self.d_med = float(d_med)
        self.raw_from_basis_transform = np.asarray(
            raw_from_basis_transform, dtype=np.float64
        )
        self.source_names = tuple(str(value) for value in source_names)
        self.config = config
        candidate_count = self.mean_design.shape[0]
        if (
            self.representation.shape[0] != candidate_count
            or self.distances.shape != (candidate_count, candidate_count)
        ):
            raise ValueError("proposed-method arrays are not aligned")
        if self.raw_from_basis_transform.shape != (
            self.mean_design.shape[1],
            len(self.source_names) + 1,
        ):
            raise ValueError(
                "source-mean basis transform does not match the source labels"
            )

    def fit_predict(
        self,
        observed_indices: np.ndarray,
        observed_y: np.ndarray,
        *,
        seed: int,
        round_id: int,
    ) -> ModelSnapshot:
        observed = np.asarray(observed_indices, dtype=int)
        fitted = fit_single_target_gp(
            self.mean_design[observed],
            self.representation[observed],
            np.asarray(observed_y, dtype=np.float64),
            d_med=self.d_med,
            config=self.config,
            include_tau_zero_face=True,
            train_distances=self.distances[np.ix_(observed, observed)],
        )
        all_indices = np.arange(len(self.distances))
        mean, covariance = fitted.predict(
            self.mean_design,
            self.representation,
            cross_distances=self.distances[np.ix_(observed, all_indices)],
            test_distances=self.distances,
            include_noise=False,
        )
        return ModelSnapshot(
            method_id=self.method_id,
            predictive_mean=mean,
            predictive_covariance=covariance,
            noise_std=fitted.noise_std,
        )

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
    ) -> BatchDecision:
        if snapshot.predictive_covariance is None:
            raise AssertionError("proposed-method selection requires full covariance")
        selected = select_slotwise_joint_ts(
            snapshot.predictive_mean,
            snapshot.predictive_covariance,
            snapshot.noise_std,
            observed_indices,
            feasible,
            pair_ids,
            batch_size=batch_size,
            namespace=self.config.study_namespace,
            method_id=self.method_id,
            campaign_seed=seed,
            round_id=round_id,
            config=self.config,
        )
        return BatchDecision(selected.indices)

    def configuration(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "family": "single_output_exact_gp",
            "estimator": "reml",
            "mean_columns": int(self.mean_design.shape[1]),
            "feature_definition": "frozen_source_trained_encoder_v1_raw_32d",
            "feature_shape": list(self.representation.shape),
            "d_med": self.d_med,
            "kernel": "isotropic_matern52",
            "amplitude_sd_bounds": list(self.config.gp_amplitude_bounds),
            "length_scale_bounds": [
                self.config.gp_length_relative_bounds[0] * self.d_med,
                self.config.gp_length_relative_bounds[1] * self.d_med,
            ],
            "noise_sd_bounds": list(self.config.gp_noise_bounds),
            "sobol_starts": self.config.gp_sobol_starts,
            "optimizer_seed": self.config.gp_optimizer_seed,
            "optimizer": "L-BFGS-B",
            "optimizer_maxiter": self.config.gp_optimizer_maxiter,
            "include_exact_tau_zero_face": True,
            "posterior": "full_latent_universal_kriging",
            "acquisition": "fresh_slotwise_joint_latent_thompson_sampling",
            "fantasy": "posterior_mean_covariance_only_with_learned_noise",
            "diversity_penalty": None,
            "source_names": list(self.source_names),
        }
