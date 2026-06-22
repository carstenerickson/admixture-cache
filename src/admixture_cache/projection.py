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
    # Observed heterozygosity rate (fraction of non-missing genotypes equal
    # to 1) of the projected target. NaN when not computed. An
    # essentially-zero rate flags pseudo-haploid or very low-coverage input
    # (project_target warns; see SCIENCE.md D17). Defaulted so older direct
    # constructions keep working.
    heterozygosity: float = float("nan")


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

    **Diploid hard-call model.** The dosage g in {0,1,2} is treated as
    the count of allele 1 out of 2. Pseudo-haploid ancient-DNA data (one
    sampled read coded as a homozygous 0/2 genotype) is accepted but note
    that the diploid likelihood is then exactly twice the correct
    Bernoulli (n=1) likelihood at every site, so the MLE point estimate of
    Q is IDENTICAL either way (the constant factor cannot move the
    argmax); only the unreported likelihood magnitude / implied confidence
    differs. For low-coverage data where per-site uncertainty should
    actually change the estimate, use a genotype-likelihood method
    (see SCIENCE.md D17). project_target warns when a target's
    heterozygosity looks pseudo-haploid / very low coverage.

    Matches stock ``admixture --supervised`` Q to within ~1e-3
    absolute on representative panels (≈0.002 max-component error on
    a real 1.14M-SNP K=4 panel). SLSQP converges in ~10-15
    iterations / ~0.02 sec on 850K SNPs at K=4.

    **Objective is the MEAN per-SNP negative log-likelihood, not the
    sum.** The argmax is identical — scaling the objective by the
    constant 1/M can't move the optimum — but it keeps the gradient
    O(1) regardless of panel size. The summed form's gradient scales
    with the SNP count (~1e6 at 1.1M SNPs); SLSQP doesn't auto-scale,
    so against the O(1) sum-to-1 constraint Jacobian the QP subproblem
    is badly conditioned and the optimizer stalls at a corner with
    ``success=True`` — returning a confidently-wrong Q. Discovered
    projecting an interior 4-way mixture against a real 1.1M-SNP panel:
    the summed form returned ``[0, 0, 1, 0]`` (true ``[.2, .5, .25,
    .05]``), the mean form recovers it to ~0.002. Normalizing makes
    SLSQP's ``ftol`` behave identically at 100 SNPs and 1.1M SNPs.

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

    # Normalize by the observed-SNP count so the objective is the MEAN
    # per-SNP NLL. Keeps the gradient O(1) at any panel size → SLSQP's
    # tolerances behave the same whether M=100 or M=1.1M. See docstring.
    inv_m = 1.0 / g_obs.size

    def neg_log_lik(q: np.ndarray) -> float:
        f = np.clip(P_obs @ q, eps, 1 - eps)
        return float(
            -inv_m * (g_obs * np.log(f) + (2 - g_obs) * np.log(1 - f)).sum()
        )

    def grad_neg_log_lik(q: np.ndarray) -> np.ndarray:
        f = np.clip(P_obs @ q, eps, 1 - eps)
        score = g_obs / f - (2 - g_obs) / (1 - f)
        result: np.ndarray = -inv_m * (P_obs.T @ score)
        return result

    return _minimize_on_simplex(neg_log_lik, grad_neg_log_lik, k, maxiter, ftol)


def numpy_supervised_projection_gl(
    *, gl: np.ndarray, p_matrix: np.ndarray, k: int,
    eps: float = 1e-9, maxiter: int = 200, ftol: float = 1e-9,
) -> tuple[np.ndarray, int, bool]:
    """Genotype-likelihood supervised-ADMIXTURE projection (NGSadmix model).

    Given per-SNP genotype likelihoods ``gl`` (M × 3, columns = P(reads |
    genotype = 0/1/2 copies of allele 1); rows of NaN are treated as missing)
    and a fixed allele-frequency matrix ``p_matrix`` (M × K, P[s,k] = freq of
    allele 1 in pop k), solve for the target's K-vector admixture proportions q
    by maximum likelihood. Per SNP the unknown genotype is marginalized out
    under a Hardy-Weinberg prior at the admixed frequency f_s = q^T P_s:

        L_s(q) = GL_s(0)·(1-f)^2 + GL_s(1)·2f(1-f) + GL_s(2)·f^2

    maximizing prod_s L_s subject to sum(q)=1, q_k >= 0 (Skotte et al. 2013,
    doi:10.1534/genetics.113.154138; fastNGSadmix is this fixed-P projection
    setting). Unlike collapsing to hard 0/1/2 calls, this downweights
    low-confidence sites, so for genuinely low-coverage data it changes (and
    improves) the estimate. Like the hard-call objective the per-SNP NLL is
    MEAN-normalized (÷ observed-SNP count) so SLSQP's tolerances behave the same
    at any panel size. Returns (q, n_iter, converged).
    """
    assert gl.shape == (p_matrix.shape[0], 3), (
        f"gl shape {gl.shape} != (P rows {p_matrix.shape[0]}, 3)"
    )
    assert p_matrix.shape[1] == k, (
        f"P has {p_matrix.shape[1]} columns but k={k}"
    )

    mask = ~np.isnan(gl).any(axis=1)
    gl_obs = gl[mask]
    P_obs = p_matrix[mask]
    if gl_obs.shape[0] == 0:
        raise PanelCacheError(
            "numpy_supervised_projection_gl: target has zero usable GL sites "
            "after masking; cannot project (no data).",
        )

    g0 = gl_obs[:, 0]
    g1 = gl_obs[:, 1]
    g2 = gl_obs[:, 2]
    inv_m = 1.0 / gl_obs.shape[0]

    def neg_log_lik(q: np.ndarray) -> float:
        f = np.clip(P_obs @ q, eps, 1 - eps)
        like = g0 * (1 - f) ** 2 + g1 * 2 * f * (1 - f) + g2 * f * f
        like = np.clip(like, eps, None)
        return float(-inv_m * np.log(like).sum())

    def grad_neg_log_lik(q: np.ndarray) -> np.ndarray:
        f = np.clip(P_obs @ q, eps, 1 - eps)
        like = g0 * (1 - f) ** 2 + g1 * 2 * f * (1 - f) + g2 * f * f
        like = np.clip(like, eps, None)
        # dL/df = -2 g0 (1-f) + 2 g1 (1-2f) + 2 g2 f; d(-log L)/dq via chain rule.
        dlike_df = -2 * g0 * (1 - f) + 2 * g1 * (1 - 2 * f) + 2 * g2 * f
        score = dlike_df / like
        result: np.ndarray = -inv_m * (P_obs.T @ score)
        return result

    return _minimize_on_simplex(neg_log_lik, grad_neg_log_lik, k, maxiter, ftol)


def _minimize_on_simplex(
    neg_log_lik: object, grad_neg_log_lik: object, k: int,
    maxiter: int, ftol: float,
) -> tuple[np.ndarray, int, bool]:
    """Minimize ``neg_log_lik`` over the probability simplex (sum(q)=1,
    0<=q_k<=1) from the uniform start via SLSQP. Shared by the hard-call and
    genotype-likelihood projections so both use the identical constraint setup."""
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


__all__ = [
    "ProjectionResult",
    "numpy_supervised_projection",
    "numpy_supervised_projection_gl",
]
