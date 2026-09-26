"""Real-data component ablations using the selected cohort's frozen GP engine.

Mean-only retains SPIN-BO's source mean and replaces its encoder geometry by
the raw ESM geometry used by Target-only. Encoder-only retains SPIN-BO's frozen
encoder geometry and replaces the source mean by an estimated intercept.

The positive-amplitude fit, learned-noise policy, universal-kriging posterior,
and slotwise Thompson sampler come directly from the historical real-data
implementation. The exact tau=0 face follows the retained mean family:
Mean-only enables it like SPIN-BO; Encoder-only disables it like Target-only.
These are the historical real-data conventions, not Simulation overrides.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stbo_benchmark.models import _GPMethod
from stbo_single.config import StudyConfig


MEAN_ONLY_METHOD_ID = "mean_only_gp_ts"
ENCODER_ONLY_METHOD_ID = "encoder_only_gp_ts"
METHOD_IDS = (MEAN_ONLY_METHOD_ID, ENCODER_ONLY_METHOD_ID)
SOURCE_INFORMED_METHOD_IDS = METHOD_IDS
# The Bio cohort declares its protocol in another module. Both implementations
# use this same GP interface, so this annotation remains independent of layout.
SequentialMethod = _GPMethod


class MeanOnlyGP(_GPMethod):
    def __init__(
        self,
        mean_basis: np.ndarray,
        raw_features: np.ndarray,
        raw_distances: np.ndarray,
        raw_d_med: float,
        raw_rms: float,
        config: StudyConfig,
    ) -> None:
        self.rms = float(raw_rms)
        if not np.isfinite(self.rms) or self.rms <= 0:
            raise ValueError("raw ESM global RMS must be finite and positive")
        super().__init__(
            method_id=MEAN_ONLY_METHOD_ID,
            mean_design=mean_basis,
            representation=raw_features,
            distances=raw_distances,
            d_med=raw_d_med,
            config=config,
            include_tau_zero_face=True,
            feature_definition="mutant_only_esm2_650m_mean_pooled_divided_by_global_rms",
        )

    def configuration(self) -> dict[str, Any]:
        result = super().configuration()
        result.update({
            "ablation": "source_mean_with_raw_esm_residual_gp",
            "mean_definition": "same_frozen_source_mean_basis_as_spin_bo",
            "global_rms": self.rms,
            "partner_embedding_used": False,
            "source_information_loaded": True,
            "source_outcomes_used_for_mean": True,
            "source_trained_encoder_used": False,
            "outcome_scale": "raw_y",
            "tau_zero_face_policy": "same_as_historical_spin_bo_source_mean_gp",
        })
        return result


class EncoderOnlyGP(_GPMethod):
    def __init__(
        self,
        encoder_latent: np.ndarray,
        encoder_distances: np.ndarray,
        encoder_d_med: float,
        config: StudyConfig,
    ) -> None:
        super().__init__(
            method_id=ENCODER_ONLY_METHOD_ID,
            mean_design=np.ones((len(encoder_latent), 1), dtype=np.float64),
            representation=encoder_latent,
            distances=encoder_distances,
            d_med=encoder_d_med,
            config=config,
            include_tau_zero_face=False,
            feature_definition="frozen_source_trained_encoder_v1_raw_32d",
        )

    def configuration(self) -> dict[str, Any]:
        result = super().configuration()
        result.update({
            "ablation": "intercept_mean_with_frozen_source_encoder_residual_gp",
            "mean_definition": "estimated_intercept_only",
            "partner_embedding_used": True,
            "source_information_loaded": True,
            "source_outcomes_used_for_mean": False,
            "source_trained_encoder_used": True,
            "encoder_retrained": False,
            "outcome_scale": "raw_y",
            "tau_zero_face_policy": "same_as_historical_target_only_intercept_gp",
        })
        return result


def create_methods(
    mean_basis: np.ndarray,
    raw_features: np.ndarray,
    raw_distances: np.ndarray,
    raw_d_med: float,
    raw_rms: float,
    encoder_latent: np.ndarray,
    encoder_distances: np.ndarray,
    encoder_d_med: float,
    config: StudyConfig,
) -> dict[str, SequentialMethod]:
    """Construct ablations from the prepared real-data arrays."""
    matrices = {
        "mean_basis": np.asarray(mean_basis),
        "raw_features": np.asarray(raw_features),
        "raw_distances": np.asarray(raw_distances),
        "encoder_latent": np.asarray(encoder_latent),
        "encoder_distances": np.asarray(encoder_distances),
    }
    for name, values in matrices.items():
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite matrix")
    count = len(matrices["mean_basis"])
    if not count or any(len(values) != count for values in matrices.values()):
        raise ValueError("ablation arrays must share the candidate row order")
    if matrices["mean_basis"].shape[1] < 1:
        raise ValueError("source mean basis must include its retained intercept span")
    for name in ("raw_distances", "encoder_distances"):
        if matrices[name].shape != (count, count):
            raise ValueError(f"{name} must be candidate-by-candidate")
    if any(not np.isfinite(value) or value <= 0 for value in (raw_d_med, encoder_d_med)):
        raise ValueError("GP median distances must be finite and positive")
    methods = (
        MeanOnlyGP(mean_basis, raw_features, raw_distances, raw_d_med, raw_rms, config),
        EncoderOnlyGP(encoder_latent, encoder_distances, encoder_d_med, config),
    )
    return {method.method_id: method for method in methods}
