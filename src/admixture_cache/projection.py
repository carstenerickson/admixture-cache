"""NumPy supervised-ADMIXTURE projection (per-target hot path).

Given a fixed allele-frequency matrix P (panel-only, precomputed) and
a target's genotype dosage vector, solve for the target's K-vector
admixture proportions via scipy SLSQP under the standard binomial
admixture likelihood. NO ADMIXTURE binary needed at projection time.

This is the "fast" half of the two-phase workflow: build the cache
once (slow), project new targets in ~0.02 sec apiece (excluding
plink2-based target alignment + dosage load).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from admixture_cache.errors import PanelCacheError


@dataclass(frozen=True)
class ProjectionResult:
    """Per-target projection output. Q vector + cluster names from
    cached manifest. Panel-stability metric carried through from build
    time."""

    target_q: np.ndarray  # shape (K,)
    cluster_order: list[str]
    panel_stability_max_sd: float  # from cached restart_sd metadata
    n_snps_used: int  # non-missing SNPs after mask
    optimization_iterations: int
    converged: bool


def numpy_supervised_projection(
    *, target_dosage: np.ndarray, p_matrix: np.ndarray, k: int,
    eps: float = 1e-9, maxiter: int = 200, ftol: float = 1e-9,
) -> tuple[np.ndarray, int, bool]:
    """Pure NumPy/scipy supervised-ADMIXTURE projection.

    Given target genotype dosage ``target_dosage`` (M-vector,
    values 0/1/2 with NaN for missing) and fixed allele-frequency
    matrix ``p_matrix`` (M × K, P[s,k] = freq of allele 1 in pop k
    at SNP s), compute the target's K-vector admixture proportions
    q via maximum-likelihood under the binomial model:

        L(q) = ∏_s Binomial(g_s; 2, q^T P_s)

    Subject to: sum(q) = 1, q_k >= 0.

    Matches stock ``admixture --supervised`` Q to within ~1e-5
    absolute on representative panels. SLSQP converges in ~9
    iterations / ~0.02 sec on 850K SNPs at K=4.

    Returns (q, n_iter, converged).
    """
    assert target_dosage.shape == (p_matrix.shape[0],), (
        f"dosage shape {target_dosage.shape} != P rows {p_matrix.shape[0]}"
    )
    assert p_matrix.shape[1] == k, (
        f"P has {p_matrix.shape[1]} columns but k={k}"
    )

    mask = ~np.isnan(target_dosage)
    g_obs = target_dosage[mask]
    P_obs = p_matrix[mask]

    if g_obs.size == 0:
        raise PanelCacheError(
            "numpy_supervised_projection: target has zero non-missing "
            "SNPs after mask; cannot project (no data).",
        )

    def neg_log_lik(q: np.ndarray) -> float:
        f = np.clip(P_obs @ q, eps, 1 - eps)
        return float(-(g_obs * np.log(f) + (2 - g_obs) * np.log(1 - f)).sum())

    def grad_neg_log_lik(q: np.ndarray) -> np.ndarray:
        f = np.clip(P_obs @ q, eps, 1 - eps)
        score = g_obs / f - (2 - g_obs) / (1 - f)
        result: np.ndarray = -P_obs.T @ score
        return result

    result = minimize(
        neg_log_lik, np.ones(k) / k, jac=grad_neg_log_lik,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{
            "type": "eq",
            "fun": lambda q: q.sum() - 1.0,
            "jac": lambda q: np.ones(k),
        }],
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return result.x, result.nit, result.success


__all__ = ["ProjectionResult", "numpy_supervised_projection"]
