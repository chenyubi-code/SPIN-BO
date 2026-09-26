"""Explicit finite-library joint sampling of the exact UK covariance.

One complete candidate-covariance factorization is used per batch.
No prediction jitter is introduced.
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dpotrf, dpstrf


def exact_posterior_factor(fitted, indices):
    from .models import ModelNumericalError, matern52

    selected = np.asarray(indices, dtype=np.int64)
    n = len(selected)
    if selected.ndim != 1 or not n or len(np.unique(selected)) != n:
        raise ValueError("Exact sampling requires a nonempty pool of distinct indices")
    x = fitted.features[selected]
    cross = matern52(x, fitted.features[fitted.observed], fitted.amplitude, fitted.length_scale)
    whitened = solve_triangular(fitted.chol, cross.T, lower=True, check_finite=False).T
    weights = solve_triangular(fitted.chol.T, whitened.T, lower=False, check_finite=False).T
    residual = fitted.mean_design[selected] - weights @ fitted.mean_design[fitted.observed]
    mean_factor = residual @ cholesky(fitted.beta_covariance, lower=True, check_finite=False)
    del cross, weights, residual
    block_size = min(256, int(fitted.config["prediction_batch_size"]))

    def covariance_rows(start, stop):
        block = matern52(x[start:stop], x, fitted.amplitude, fitted.length_scale)
        block -= whitened[start:stop] @ whitened.T
        block += mean_factor[start:stop] @ mean_factor.T
        return block

    def build():
        # Fortran layout permits LAPACK to overwrite the same buffer.
        matrix = np.empty((n, n), dtype=np.float64, order="F")
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            block = covariance_rows(start, stop)
            if not np.isfinite(block).all():
                raise ModelNumericalError("Exact posterior covariance contains nonfinite entries")
            matrix[start:stop] = block
        return matrix

    work = build()
    diagonal = work.diagonal().copy()
    scale = max(float(np.max(np.abs(diagonal))), np.finfo(float).tiny)
    if float(np.min(diagonal)) < -1e-10 * scale:
        raise ModelNumericalError("Exact posterior covariance has a negative diagonal")
    factor, info = dpotrf(work, lower=1, clean=0, overwrite_a=1)
    if info < 0:
        raise ModelNumericalError(f"LAPACK Cholesky invalid argument: {info}")
    if info == 0:
        for column in range(n):
            factor[:column, column] = 0
        rank = n
        algorithm = "blocked_covariance_unpivoted_Cholesky"
    else:
        # A failed potrf has modified its input. Rebuild before the rank-revealing
        # factorization, then independently check EVERY residual block. A small
        # stopping pivot alone would not exclude an indefinite residual.
        del factor, work
        work = build()
        pivot_tolerance = np.finfo(float).eps * n * scale
        pivoted, permutation, rank, pivot_info = dpstrf(
            work, lower=1, tol=pivot_tolerance, overwrite_a=1)
        if pivot_info < 0:
            raise ModelNumericalError(f"LAPACK pivoted Cholesky invalid argument: {pivot_info}")
        for column in range(rank):
            pivoted[:column, column] = 0
        factor = np.empty((n, rank), dtype=np.float64, order="F")
        factor[permutation - 1] = pivoted[:, :rank]
        del pivoted, work
        squared_error = squared_reference = 0.0
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            reference = covariance_rows(start, stop)
            error = factor[start:stop] @ factor.T - reference
            squared_error += float(np.sum(error * error))
            squared_reference += float(np.sum(reference * reference))
        residual_error = float(np.sqrt(squared_error / max(squared_reference, np.finfo(float).tiny)))
        if not np.isfinite(residual_error) or residual_error > 1e-10:
            raise ModelNumericalError(
                f"Exact covariance factorization residual exceeds roundoff tolerance: {residual_error}")
        algorithm = "roundoff_rank_revealing_Cholesky"
    if not np.isfinite(factor).all():
        raise ModelNumericalError("Exact posterior factor contains nonfinite entries")
    return factor, {"algorithm": algorithm, "factor_rank": int(rank)}
