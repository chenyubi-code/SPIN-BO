from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import qmc

from .config import StudyConfig


class GPNumericalError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


_COVARIANCE_RELATIVE_TOLERANCE = 1.0e-8


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (values + values.T)


def _stable_covariance(
    matrix: np.ndarray,
    *,
    context: str,
    relative_tolerance: float = _COVARIANCE_RELATIVE_TOLERANCE,
) -> np.ndarray:
    """Symmetrize a covariance while rejecting material numerical errors.

    Symmetry is checked on the matrix as computed, rather than after silently
    symmetrizing it.  Positive semidefiniteness is then checked on the
    symmetrized matrix with a scale-relative tolerance for floating-point
    roundoff.
    """

    raw = np.asarray(matrix, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] == 0:
        raise GPNumericalError(f"{context} must be a nonempty square matrix")
    if not np.all(np.isfinite(raw)):
        raise GPNumericalError(f"{context} contains NaN/Inf")
    entry_scale = max(float(np.max(np.abs(raw))), 1.0)
    symmetry_max_abs = float(np.max(np.abs(raw - raw.T)))
    if symmetry_max_abs > float(relative_tolerance) * entry_scale:
        raise GPNumericalError(
            f"{context} is materially asymmetric: max_abs={symmetry_max_abs}"
        )
    values = _symmetrize(raw)
    eigenvalues = np.linalg.eigvalsh(values)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    minimum_diagonal = float(np.min(np.diag(values)))
    if minimum_diagonal < -float(relative_tolerance) * spectral_scale:
        raise GPNumericalError(
            f"{context} has a materially negative variance: {minimum_diagonal}"
        )
    if float(eigenvalues[0]) < -float(relative_tolerance) * spectral_scale:
        raise GPNumericalError(
            f"{context} is materially indefinite: min_eigenvalue={eigenvalues[0]}"
        )
    return values


def distance_matrix(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("features must be a finite matrix with at least two rows")
    return squareform(pdist(values, metric="euclidean")).astype(np.float64)


def median_pairwise_distance(
    features: np.ndarray | None = None, distances: np.ndarray | None = None
) -> float:
    if distances is None:
        if features is None:
            raise ValueError("features or distances must be supplied")
        condensed = pdist(np.asarray(features, dtype=np.float64), metric="euclidean")
    else:
        matrix = np.asarray(distances, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError("distance matrix must be square with at least two rows")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("distance matrix contains NaN/Inf")
        condensed = matrix[np.triu_indices(matrix.shape[0], 1)]
    value = float(np.median(condensed))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("median pairwise distance must be finite and positive")
    return value


def matern52_from_distances(
    distances: np.ndarray, amplitude: float, length_scale: float
) -> np.ndarray:
    tau = float(amplitude)
    ell = float(length_scale)
    if tau <= 0.0 or ell <= 0.0 or not np.isfinite(tau + ell):
        raise ValueError("Matérn amplitude and length scale must be finite and positive")
    scaled = np.asarray(distances, dtype=np.float64) / ell
    root5 = np.sqrt(5.0) * scaled
    correlation = (1.0 + root5 + (5.0 / 3.0) * scaled**2) * np.exp(-root5)
    return np.asarray((tau**2) * correlation, dtype=np.float64)


@dataclass(frozen=True)
class _Regression:
    beta: np.ndarray
    beta_covariance: np.ndarray
    residual: np.ndarray
    rank: int
    information_condition: float
    log_information_determinant: float


@dataclass(frozen=True)
class _State:
    face: str
    amplitude: float
    length_scale: float | None
    noise_std: float
    nll: float
    jitter: float
    jitter_factor: float
    cholesky: np.ndarray
    regression: _Regression
    residual_weights: np.ndarray


@dataclass(frozen=True)
class FittedGP:
    face: str
    amplitude: float
    length_scale: float | None
    noise_std: float
    nll: float
    beta: np.ndarray
    beta_covariance: np.ndarray
    rank: int
    information_condition: float
    d_med: float
    jitter: float
    jitter_factor: float
    h_train: np.ndarray
    z_train: np.ndarray
    cholesky: np.ndarray
    residual_weights: np.ndarray
    positive_restart_log: tuple[dict[str, Any], ...]
    boundary_face_log: tuple[dict[str, Any], ...]

    def predict(
        self,
        h_test: np.ndarray,
        z_test: np.ndarray,
        *,
        cross_distances: np.ndarray | None = None,
        test_distances: np.ndarray | None = None,
        include_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_new = np.asarray(h_test, dtype=np.float64)
        z_new = np.asarray(z_test, dtype=np.float64)
        if h_new.ndim != 2 or h_new.shape[1] != self.beta.size:
            raise ValueError("prediction mean design has the wrong shape")
        if z_new.ndim != 2 or z_new.shape[0] != h_new.shape[0]:
            raise ValueError("prediction features do not align with the mean design")
        if self.face == "tau_zero":
            mean = h_new @ self.beta
            covariance = h_new @ self.beta_covariance @ h_new.T
        else:
            if self.length_scale is None:
                raise AssertionError("positive GP face has no length scale")
            cross = (
                cdist(self.z_train, z_new, metric="euclidean")
                if cross_distances is None
                else np.asarray(cross_distances, dtype=np.float64)
            )
            test = (
                cdist(z_new, z_new, metric="euclidean")
                if test_distances is None
                else np.asarray(test_distances, dtype=np.float64)
            )
            if cross.shape != (self.z_train.shape[0], z_new.shape[0]):
                raise ValueError("cross-distance matrix has the wrong shape")
            if test.shape != (z_new.shape[0], z_new.shape[0]):
                raise ValueError("test-distance matrix has the wrong shape")
            kernel_cross = matern52_from_distances(
                cross, self.amplitude, self.length_scale
            )
            mean = h_new @ self.beta + kernel_cross.T @ self.residual_weights
            whitened = solve_triangular(
                self.cholesky, kernel_cross, lower=True, check_finite=False
            )
            solved = solve_triangular(
                self.cholesky.T, whitened, lower=False, check_finite=False
            )
            kernel_test = matern52_from_distances(
                test, self.amplitude, self.length_scale
            )
            basis_residual = h_new.T - self.h_train.T @ solved
            covariance = (
                kernel_test
                - kernel_cross.T @ solved
                + basis_residual.T @ self.beta_covariance @ basis_residual
            )
        covariance = _stable_covariance(
            covariance, context="GP latent posterior covariance"
        )
        if include_noise:
            covariance = covariance + (self.noise_std**2) * np.eye(len(h_new))
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise GPNumericalError("GP posterior contains NaN/Inf")
        return np.asarray(mean, dtype=np.float64), covariance



def _cholesky_with_jitter(
    kernel: np.ndarray,
    noise_variance: float,
    initial_factor: float,
    maximum_factor: float,
) -> tuple[np.ndarray, float, float]:
    kernel = _symmetrize(kernel)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] == 0:
        raise ValueError("kernel covariance must be nonempty and square")
    mean_kernel_diagonal = float(np.mean(np.diag(kernel)))
    if not np.isfinite(mean_kernel_diagonal) or mean_kernel_diagonal <= 0.0:
        raise GPNumericalError("positive GP face has a nonpositive mean kernel diagonal")
    identity = np.eye(kernel.shape[0], dtype=np.float64)
    factor = float(initial_factor)
    last_error = "unknown"
    while factor <= maximum_factor * (1.0 + 16.0 * np.finfo(float).eps):
        jitter = factor * mean_kernel_diagonal
        try:
            matrix = kernel + (float(noise_variance) + jitter) * identity
            chol = cholesky(matrix, lower=True, check_finite=False)
            return np.asarray(chol), float(jitter), float(factor)
        except (np.linalg.LinAlgError, ValueError) as exc:
            last_error = str(exc)
            factor *= 10.0
    raise GPNumericalError(f"observation covariance Cholesky failed: {last_error}")


def _gls(
    h: np.ndarray,
    y: np.ndarray,
    chol: np.ndarray,
    relative_tolerance: float,
) -> _Regression:
    whitened_h = solve_triangular(chol, h, lower=True, check_finite=False)
    whitened_y = solve_triangular(chol, y, lower=True, check_finite=False)
    q_matrix, r_matrix, pivots = qr(
        whitened_h, mode="economic", pivoting=True, check_finite=False
    )
    columns = h.shape[1]
    diagonal = np.abs(np.diag(r_matrix))
    reference = float(diagonal[0]) if diagonal.size else 0.0
    rank = int(np.sum(diagonal > relative_tolerance * reference)) if reference > 0 else 0
    singular = np.linalg.svd(whitened_h, compute_uv=False)
    condition = (
        float("inf")
        if singular.size == 0 or singular[-1] <= 0.0
        else float((singular[0] / singular[-1]) ** 2)
    )
    if rank < columns or r_matrix.shape[0] < columns:
        raise GPNumericalError(
            f"mean design rank deficient: rank={rank}, columns={columns}, condition={condition}"
        )
    leading = np.asarray(r_matrix[:columns, :columns], dtype=np.float64)
    projected_y = np.asarray(q_matrix.T @ whitened_y, dtype=np.float64)
    beta_pivoted = solve_triangular(
        leading, projected_y[:columns], lower=False, check_finite=False
    )
    beta = np.empty(columns, dtype=np.float64)
    beta[pivots] = beta_pivoted
    inverse_r = solve_triangular(
        leading, np.eye(columns), lower=False, check_finite=False
    )
    covariance_pivoted = inverse_r @ inverse_r.T
    beta_covariance = np.empty((columns, columns), dtype=np.float64)
    beta_covariance[np.ix_(pivots, pivots)] = covariance_pivoted
    beta_covariance = _symmetrize(beta_covariance)
    log_information = float(2.0 * np.sum(np.log(np.abs(np.diag(leading)))))
    return _Regression(
        beta=beta,
        beta_covariance=beta_covariance,
        residual=y - h @ beta,
        rank=rank,
        information_condition=condition,
        log_information_determinant=log_information,
    )


def _positive_state(
    log_parameters: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    train_distances: np.ndarray,
    config: StudyConfig,
) -> _State:
    parameters = np.exp(np.asarray(log_parameters, dtype=np.float64))
    if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
        raise GPNumericalError("GP parameters are malformed/nonfinite")
    amplitude, length_scale, noise_std = map(float, parameters)
    kernel = matern52_from_distances(train_distances, amplitude, length_scale)
    chol, jitter, jitter_factor = _cholesky_with_jitter(
        kernel,
        noise_std**2,
        config.gp_jitter_initial_factor,
        config.gp_jitter_max_factor,
    )
    regression = _gls(h, y, chol, config.mean_rank_relative_tolerance)
    whitened_residual = solve_triangular(
        chol, regression.residual, lower=True, check_finite=False
    )
    quadratic = float(whitened_residual @ whitened_residual)
    log_determinant = float(2.0 * np.sum(np.log(np.diag(chol))))
    degrees = y.size - h.shape[1]
    if degrees <= 0:
        raise GPNumericalError("REML requires n greater than the mean-basis rank")
    nll = 0.5 * (
        log_determinant
        + regression.log_information_determinant
        + quadratic
        + degrees * np.log(2.0 * np.pi)
    )
    if not np.isfinite(nll):
        raise GPNumericalError("REML objective is nonfinite")
    residual_weights = solve_triangular(
        chol.T, whitened_residual, lower=False, check_finite=False
    )
    return _State(
        face="positive",
        amplitude=amplitude,
        length_scale=length_scale,
        noise_std=noise_std,
        nll=float(nll),
        jitter=jitter,
        jitter_factor=jitter_factor,
        cholesky=chol,
        regression=regression,
        residual_weights=np.asarray(residual_weights),
    )


def _fit_positive_face(
    h: np.ndarray,
    y: np.ndarray,
    train_distances: np.ndarray,
    d_med: float,
    config: StudyConfig,
) -> tuple[_State | None, tuple[dict[str, Any], ...]]:
    bounds = np.log(
        np.asarray(
            [
                config.gp_amplitude_bounds,
                (
                    config.gp_length_relative_bounds[0] * d_med,
                    config.gp_length_relative_bounds[1] * d_med,
                ),
                config.gp_noise_bounds,
            ],
            dtype=np.float64,
        )
    )
    sampler = qmc.Sobol(d=3, scramble=True, seed=config.gp_optimizer_seed)
    count = int(config.gp_sobol_starts)
    if count > 0 and count & (count - 1) == 0:
        units = sampler.random_base2(int(np.log2(count)))
    else:
        units = sampler.random(count)
    starts = bounds[:, 0] + units * (bounds[:, 1] - bounds[:, 0])
    records: list[dict[str, Any]] = []
    valid_states: list[_State] = []

    for restart, start in enumerate(starts):
        last_failure: list[str | None] = [None]

        def objective(value: np.ndarray) -> float:
            try:
                state = _positive_state(value, h, y, train_distances, config)
                last_failure[0] = None
                return state.nll
            except (GPNumericalError, ValueError, np.linalg.LinAlgError) as exc:
                last_failure[0] = str(exc)
                return 1.0e100

        try:
            optimized = minimize(
                objective,
                np.asarray(start),
                method="L-BFGS-B",
                bounds=[tuple(row) for row in bounds],
                options={"maxiter": int(config.gp_optimizer_maxiter)},
            )
            endpoint = np.asarray(optimized.x, dtype=np.float64)
            try:
                final_state = _positive_state(endpoint, h, y, train_distances, config)
                valid = True
                failure = None
                valid_states.append(final_state)
            except (GPNumericalError, ValueError, np.linalg.LinAlgError) as exc:
                final_state = None
                valid = False
                failure = str(exc)
            records.append(
                {
                    "restart": restart,
                    "start_log_parameters": start.tolist(),
                    "start_parameters": np.exp(start).tolist(),
                    "final_log_parameters": endpoint.tolist(),
                    "final_parameters": None
                    if final_state is None
                    else [
                        final_state.amplitude,
                        final_state.length_scale,
                        final_state.noise_std,
                    ],
                    "objective": None if final_state is None else final_state.nll,
                    "optimizer_success": bool(optimized.success),
                    "numerically_valid": valid,
                    "iterations": int(getattr(optimized, "nit", 0)),
                    "function_evaluations": int(getattr(optimized, "nfev", 0)),
                    "message": str(optimized.message),
                    "jitter": None if final_state is None else final_state.jitter,
                    "jitter_factor": None
                    if final_state is None
                    else final_state.jitter_factor,
                    "failure_reason": failure or last_failure[0],
                }
            )
        except Exception as exc:
            records.append(
                {
                    "restart": restart,
                    "start_log_parameters": start.tolist(),
                    "start_parameters": np.exp(start).tolist(),
                    "final_log_parameters": None,
                    "final_parameters": None,
                    "objective": None,
                    "optimizer_success": False,
                    "numerically_valid": False,
                    "iterations": 0,
                    "function_evaluations": 0,
                    "message": "L-BFGS-B raised an exception",
                    "jitter": None,
                    "jitter_factor": None,
                    "failure_reason": str(exc),
                }
            )
    best = min(valid_states, key=lambda state: state.nll) if valid_states else None
    return best, tuple(records)


def _fit_zero_face(h: np.ndarray, y: np.ndarray, config: StudyConfig) -> _State:
    identity_chol = np.eye(y.size, dtype=np.float64)
    unscaled = _gls(h, y, identity_chol, config.mean_rank_relative_tolerance)
    degrees = y.size - h.shape[1]
    if degrees <= 0:
        raise GPNumericalError("tau=0 REML face requires positive residual degrees of freedom")
    residual_sum_squares = float(unscaled.residual @ unscaled.residual)
    lower, upper = map(float, config.gp_noise_bounds)
    variance = float(np.clip(residual_sum_squares / degrees, lower**2, upper**2))
    noise_std = float(np.sqrt(variance))
    chol = noise_std * identity_chol
    regression = _gls(h, y, chol, config.mean_rank_relative_tolerance)
    whitened_residual = regression.residual / noise_std
    quadratic = float(whitened_residual @ whitened_residual)
    log_determinant = float(y.size * np.log(variance))
    nll = 0.5 * (
        log_determinant
        + regression.log_information_determinant
        + quadratic
        + degrees * np.log(2.0 * np.pi)
    )
    residual_weights = regression.residual / variance
    return _State(
        face="tau_zero",
        amplitude=0.0,
        length_scale=None,
        noise_std=noise_std,
        nll=float(nll),
        jitter=0.0,
        jitter_factor=0.0,
        cholesky=chol,
        regression=regression,
        residual_weights=np.asarray(residual_weights),
    )


def fit_single_target_gp(
    h_train: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    *,
    d_med: float,
    config: StudyConfig,
    include_tau_zero_face: bool,
    train_distances: np.ndarray | None = None,
) -> FittedGP:
    h = np.asarray(h_train, dtype=np.float64)
    z = np.asarray(z_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if h.ndim != 2 or z.ndim != 2 or y.ndim != 1:
        raise ValueError("GP training arrays have invalid dimensions")
    if h.shape[0] != z.shape[0] or h.shape[0] != y.size:
        raise ValueError("GP training arrays have inconsistent row counts")
    if y.size <= h.shape[1]:
        raise ValueError("REML needs more observations than mean-basis columns")
    if not all(np.all(np.isfinite(value)) for value in (h, z, y)):
        raise ValueError("GP training arrays contain NaN/Inf")
    distances = (
        cdist(z, z, metric="euclidean")
        if train_distances is None
        else np.asarray(train_distances, dtype=np.float64)
    )
    if distances.shape != (y.size, y.size):
        raise ValueError("training distance matrix has the wrong shape")
    positive, restart_log = _fit_positive_face(h, y, distances, d_med, config)
    candidates: list[_State] = []
    boundary_log: list[dict[str, Any]] = []
    if positive is not None:
        candidates.append(positive)
        boundary_log.append(
            {
                "face": "positive",
                "valid": True,
                "negative_restricted_log_likelihood": positive.nll,
                "amplitude_sd": positive.amplitude,
                "length_scale": positive.length_scale,
                "noise_sd": positive.noise_std,
            }
        )
    else:
        boundary_log.append(
            {
                "face": "positive",
                "valid": False,
                "negative_restricted_log_likelihood": None,
                "failure_reason": "all deterministic Sobol endpoints were invalid",
            }
        )
    if include_tau_zero_face:
        try:
            zero = _fit_zero_face(h, y, config)
            candidates.append(zero)
            boundary_log.append(
                {
                    "face": "tau_zero",
                    "valid": True,
                    "negative_restricted_log_likelihood": zero.nll,
                    "amplitude_sd": 0.0,
                    "length_scale": None,
                    "noise_sd": zero.noise_std,
                }
            )
        except (GPNumericalError, ValueError, np.linalg.LinAlgError) as exc:
            boundary_log.append(
                {
                    "face": "tau_zero",
                    "valid": False,
                    "negative_restricted_log_likelihood": None,
                    "failure_reason": str(exc),
                }
            )
    if not candidates:
        raise GPNumericalError(
            "no numerically valid REML face was found",
            diagnostics={
                "positive_restart_log": list(restart_log),
                "boundary_face_log": boundary_log,
            },
        )
    # Exact objective ties prefer the simpler boundary face, as preregistered.
    selected = min(
        candidates, key=lambda state: (state.nll, 0 if state.face == "tau_zero" else 1)
    )
    for row in boundary_log:
        row["selected"] = bool(row["face"] == selected.face)
    return FittedGP(
        face=selected.face,
        amplitude=selected.amplitude,
        length_scale=selected.length_scale,
        noise_std=selected.noise_std,
        nll=selected.nll,
        beta=selected.regression.beta.copy(),
        beta_covariance=selected.regression.beta_covariance.copy(),
        rank=selected.regression.rank,
        information_condition=selected.regression.information_condition,
        d_med=float(d_med),
        jitter=selected.jitter,
        jitter_factor=selected.jitter_factor,
        h_train=h.copy(),
        z_train=z.copy(),
        cholesky=selected.cholesky.copy(),
        residual_weights=selected.residual_weights.copy(),
        positive_restart_log=restart_log,
        boundary_face_log=tuple(boundary_log),
    )

