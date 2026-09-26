from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch import nn

from stbo_single.acquisition import select_slotwise_joint_ts, stable_best
from stbo_single.config import StudyConfig
from stbo_single.gp import fit_single_target_gp
from stbo_single.io_utils import derive_uint64
from stbo_single.proposed import OUR_METHOD_ID
from stbo_single.protocol import BatchDecision, ModelSnapshot


TARGET_GP_METHOD_ID = "target_only_gp_ts"
ALDE_METHOD_ID = "alde_dnn_ts"
EVOLVEPRO_METHOD_ID = "evolvepro_rf_topn"
METHOD_IDS = (
    OUR_METHOD_ID,
    TARGET_GP_METHOD_ID,
    ALDE_METHOD_ID,
    EVOLVEPRO_METHOD_ID,
)


class _GPMethod:
    def __init__(
        self,
        *,
        method_id: str,
        mean_design: np.ndarray,
        representation: np.ndarray,
        distances: np.ndarray,
        d_med: float,
        config: StudyConfig,
        include_tau_zero_face: bool,
        feature_definition: str,
    ) -> None:
        self.method_id = method_id
        self.mean_design = np.asarray(mean_design, dtype=np.float64)
        self.representation = np.asarray(representation, dtype=np.float64)
        self.distances = np.asarray(distances, dtype=np.float64)
        self.d_med = float(d_med)
        self.config = config
        self.include_tau_zero_face = bool(include_tau_zero_face)
        self.feature_definition = feature_definition
        count = self.mean_design.shape[0]
        if self.representation.shape[0] != count or self.distances.shape != (count, count):
            raise ValueError("GP method arrays are not aligned")

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
            include_tau_zero_face=self.include_tau_zero_face,
            train_distances=self.distances[np.ix_(observed, observed)],
        )
        mean, covariance = fitted.predict(
            self.mean_design,
            self.representation,
            cross_distances=self.distances[np.ix_(observed, np.arange(len(self.distances)))],
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
            raise AssertionError("GP selection requires a full latent covariance")
        noise_std = snapshot.noise_std
        selected = select_slotwise_joint_ts(
            snapshot.predictive_mean,
            snapshot.predictive_covariance,
            noise_std,
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
            "feature_definition": self.feature_definition,
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
            "include_exact_tau_zero_face": self.include_tau_zero_face,
            "posterior": "full_latent_universal_kriging",
            "acquisition": "fresh_slotwise_joint_latent_thompson_sampling",
            "fantasy": "posterior_mean_covariance_only_with_learned_noise",
            "diversity_penalty": None,
        }


class TargetOnlyGPTS(_GPMethod):
    def __init__(
        self,
        scaled_mutant_esm: np.ndarray,
        distances: np.ndarray,
        d_med: float,
        rms: float,
        config: StudyConfig,
    ) -> None:
        self.rms = float(rms)
        super().__init__(
            method_id=TARGET_GP_METHOD_ID,
            mean_design=np.ones((len(scaled_mutant_esm), 1), dtype=np.float64),
            representation=scaled_mutant_esm,
            distances=distances,
            d_med=d_med,
            config=config,
            include_tau_zero_face=False,
            feature_definition="mutant_only_esm2_650m_mean_pooled_divided_by_global_rms",
        )

    def configuration(self) -> dict[str, Any]:
        result = super().configuration()
        result.update(
            {
                "global_rms": self.rms,
                "partner_embedding_used": False,
                "source_information_loaded": False,
                "outcome_scale": "raw_y",
            }
        )
        return result


class _ALDENetwork(nn.Module):
    def __init__(self, input_dimension: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, hidden, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden, hidden, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden, 1, bias=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def _subseed_with_empty_slot(
    namespace: str, method_id: str, campaign_seed: int, round_id: int, stream: str
) -> tuple[int, str]:
    # The final empty field is part of the frozen comparator seed namespace.
    return derive_uint64(namespace, method_id, campaign_seed, round_id, stream, "")


class ALDEDNNTS:
    method_id = ALDE_METHOD_ID

    def __init__(
        self,
        features: np.ndarray,
        positions: np.ndarray,
        config: StudyConfig,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.positions = np.asarray(positions, dtype=int)
        self.config = config
        if self.features.ndim != 2 or not np.all(np.isfinite(self.features)):
            raise ValueError("ALDE features must be a finite matrix")
        if not np.all(np.isin(np.unique(self.features), [0.0, 1.0])):
            raise ValueError("ALDE features must be raw binary one-hot values")

    @staticmethod
    def _outcome_denominator(y: np.ndarray) -> float:
        maximum = float(np.max(y))
        if maximum > 1.0e-8:
            return maximum
        maximum_absolute = float(np.max(np.abs(y)))
        return maximum_absolute if maximum_absolute > 1.0e-8 else 1.0

    def fit_predict(
        self,
        observed_indices: np.ndarray,
        observed_y: np.ndarray,
        *,
        seed: int,
        round_id: int,
    ) -> ModelSnapshot:
        observed = np.asarray(observed_indices, dtype=int)
        y = np.asarray(observed_y, dtype=np.float64)
        if observed.size != y.size or observed.size < 2:
            raise ValueError("ALDE observations are malformed")
        denominator = self._outcome_denominator(y)
        all_predictions: list[np.ndarray] = []
        for member in range(self.config.alde_ensemble_size):
            member_seed, _ = _subseed_with_empty_slot(
                self.config.study_namespace,
                self.method_id,
                seed,
                round_id,
                f"ensemble_member_{member}",
            )
            split_seed = int(member_seed % (2**32))
            torch_seed = int(member_seed % (2**63 - 1))
            train_indices, _ = train_test_split(
                observed,
                test_size=0.1,
                shuffle=True,
                random_state=split_seed,
            )
            x_train = torch.from_numpy(self.features[np.asarray(train_indices, dtype=int)])
            y_lookup = dict(zip(observed.tolist(), y.tolist()))
            y_train = torch.tensor(
                [y_lookup[int(index)] / denominator for index in train_indices],
                dtype=torch.float32,
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch_seed)
                network = _ALDENetwork(self.features.shape[1], self.config.alde_hidden)
            network.to(device="cpu", dtype=torch.float32)
            optimizer = torch.optim.Adam(
                network.parameters(),
                lr=self.config.alde_learning_rate,
                weight_decay=0.0,
            )
            objective = nn.MSELoss(reduction="mean")
            best_loss = float("inf")
            epochs_without_improvement = 0
            final_loss = float("nan")
            for epoch in range(1, self.config.alde_max_epochs + 1):
                network.train()
                optimizer.zero_grad(set_to_none=True)
                loss = objective(network(x_train), y_train)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"ALDE loss became nonfinite for member {member}, epoch {epoch}"
                    )
                loss.backward()
                optimizer.step()
                network.eval()
                with torch.inference_mode():
                    monitored = objective(network(x_train), y_train)
                final_loss = float(monitored)
                if not np.isfinite(final_loss):
                    raise RuntimeError("ALDE monitored training loss is nonfinite")
                if final_loss < best_loss:
                    best_loss = final_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.config.alde_patience:
                        break
            network.eval()
            with torch.inference_mode():
                prediction = (
                    network(torch.from_numpy(self.features)).cpu().numpy().astype(np.float64)
                    * denominator
            )
            if prediction.shape != (len(self.features),) or not np.all(np.isfinite(prediction)):
                raise RuntimeError("ALDE produced malformed/nonfinite predictions")
            all_predictions.append(prediction)
        member_predictions = np.vstack(all_predictions)
        predictive_mean = np.mean(member_predictions, axis=0)
        return ModelSnapshot(
            method_id=self.method_id,
            predictive_mean=predictive_mean,
            member_predictions=member_predictions,
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
        member_predictions = snapshot.member_predictions
        if member_predictions is None:
            raise ValueError("ALDE selection requires ensemble member predictions")
        batch_seed, _ = _subseed_with_empty_slot(
            self.config.study_namespace,
            self.method_id,
            seed,
            round_id,
            "ensemble_member_ts_batch",
        )
        rng = np.random.default_rng(batch_seed)
        blocked = set(int(value) for value in np.asarray(observed_indices, dtype=int))
        feasible_mask = np.asarray(feasible, dtype=bool)
        identifiers = np.asarray(pair_ids, dtype=str)
        chosen: list[int] = []
        for slot in range(1, batch_size + 1):
            pool = np.asarray(
                [
                    index
                    for index in range(len(identifiers))
                    if feasible_mask[index] and index not in blocked
                ],
                dtype=int,
            )
            member = int(rng.integers(0, member_predictions.shape[0]))
            scores = member_predictions[member, pool]
            selected = stable_best(pool, scores, identifiers)
            chosen.append(selected)
            blocked.add(selected)
        return BatchDecision(tuple(chosen))

    def configuration(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "feature_definition": "full_sequence_one_hot_at_sorted_mutable_positions",
            "positions": self.positions.tolist(),
            "amino_acid_order": "ACDEFGHIKLMNPQRSTVWY",
            "feature_shape": list(self.features.shape),
            "architecture": [
                int(self.features.shape[1]),
                self.config.alde_hidden,
                self.config.alde_hidden,
                1,
            ],
            "activation": "LeakyReLU(0.01)",
            "ensemble_size": self.config.alde_ensemble_size,
            "member_subset": "90_percent_without_replacement",
            "optimizer": "Adam",
            "learning_rate": self.config.alde_learning_rate,
            "weight_decay": 0.0,
            "maximum_epochs": self.config.alde_max_epochs,
            "patience": self.config.alde_patience,
            "early_stopping_monitor": "training_loss_after_step",
            "restore_best_weights": False,
            "device": "cpu",
            "dtype": "float32",
            "acquisition": "uniform_ensemble_member_per_slot_then_stable_argmax",
            "partner_embedding_used": False,
            "source_information_loaded": False,
        }


class EvolveProRFTOPN:
    method_id = EVOLVEPRO_METHOD_ID

    def __init__(
        self,
        raw_mutant_esm: np.ndarray,
        pair_ids: np.ndarray,
        config: StudyConfig,
    ) -> None:
        self.features = np.asarray(raw_mutant_esm, dtype=np.float32)
        self.pair_ids = np.asarray(pair_ids, dtype=str)
        self.config = config
        if self.features.ndim != 2 or not np.all(np.isfinite(self.features)):
            raise ValueError("EVOLVEpro features must be a finite matrix")
        if self.pair_ids.shape != (len(self.features),):
            raise ValueError("EVOLVEpro pair IDs do not align with its features")
        if np.unique(self.pair_ids).size != self.pair_ids.size:
            raise ValueError("EVOLVEpro pair IDs must be unique")

    def _model(self, seed: int) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.config.rf_estimators,
            criterion="friedman_mse",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=int(seed),
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
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
        y = np.asarray(observed_y, dtype=np.float64)
        model = self._model(seed)
        model.fit(self.features[observed], y)
        if len(model.estimators_) != self.config.rf_estimators:
            raise RuntimeError("EVOLVEpro forest has the wrong number of trees")
        prediction = np.asarray(model.predict(self.features), dtype=np.float64)
        if prediction.shape != (len(self.features),) or not np.all(np.isfinite(prediction)):
            raise RuntimeError("EVOLVEpro produced malformed/nonfinite predictions")
        return ModelSnapshot(
            method_id=self.method_id,
            predictive_mean=prediction,
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
        blocked = set(int(value) for value in np.asarray(observed_indices, dtype=int))
        feasible_mask = np.asarray(feasible, dtype=bool)
        identifiers = np.asarray(pair_ids, dtype=str)
        if not np.array_equal(identifiers, self.pair_ids):
            raise ValueError("EVOLVEpro selection pair IDs changed after fitting")
        pool = np.asarray(
            [
                index
                for index in range(len(identifiers))
                if feasible_mask[index] and index not in blocked
            ],
            dtype=int,
        )
        if pool.size < batch_size:
            raise RuntimeError("EVOLVEpro pool cannot fill the requested batch")
        order = np.lexsort((identifiers[pool], -snapshot.predictive_mean[pool]))
        ranked = pool[order]
        chosen = tuple(int(value) for value in ranked[:batch_size])
        return BatchDecision(chosen)

    def configuration(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "feature_definition": "raw_mutant_only_esm2_650m_mean_pooled",
            "feature_shape": list(self.features.shape),
            "feature_preprocessing": "none",
            "partner_embedding_used": False,
            "source_information_loaded": False,
            "outcome_scale": "raw_y",
            "model": "sklearn.RandomForestRegressor",
            "model_parameters": self._model(0).get_params(deep=False),
            "random_state_rule": "campaign_seed_at_every_checkpoint",
            "acquisition": "stable_topn_by_mean_prediction",
            "stable_rank_tie_rule": self.config.top_tie_rule,
            "diversity_penalty": None,
            "automatic_wild_type_append": False,
        }
