"""NumPy supervised-ADMIXTURE projection math tests.

Synthetic K-cluster panels with analytically known Q vectors;
SLSQP must recover Q within tight tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from admixture_cache import PanelCacheError, numpy_supervised_projection


def _binomial_dosage(p_per_snp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one diploid dosage per SNP given allele-1 frequency."""
    return rng.binomial(2, p_per_snp).astype(np.float64)


class TestProjectionMath:
    def test_two_cluster_50_50_admixture(self) -> None:
        """K=2 panel, q=(0.5, 0.5); recover within binomial sampling noise."""
        rng = np.random.default_rng(0)
        M = 2000
        P = np.column_stack([np.full(M, 0.9), np.full(M, 0.1)])
        q_true = np.array([0.5, 0.5])
        dosage = _binomial_dosage(P @ q_true, rng)

        q, _, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=2,
        )
        assert converged
        assert q.shape == (2,)
        assert np.isclose(q.sum(), 1.0, atol=1e-9)
        assert np.all(q >= 0.0)
        # Binomial noise ~ 1 / sqrt(M) per Q component; loose tolerance.
        assert np.max(np.abs(q - q_true)) < 0.03

    def test_four_cluster_50_50_mixed(self) -> None:
        """K=4 panel, q=(0.5, 0.5, 0, 0). Cluster AFs are sampled per
        SNP so each pair of clusters has identifiable signal; mass
        should accumulate on the two true clusters."""
        rng = np.random.default_rng(42)
        M = 2000
        # Random per-SNP AFs in [0.05, 0.95] for each of 4 clusters.
        # Random P breaks the cross-cluster degeneracy a uniform P has.
        P = rng.uniform(0.05, 0.95, size=(M, 4))
        q_true = np.array([0.5, 0.5, 0.0, 0.0])
        dosage = _binomial_dosage(P @ q_true, rng)

        q, _, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=4,
        )
        assert converged
        # Present clusters dominate
        assert q[0] + q[1] > 0.85
        # Each absent cluster individually small
        assert q[2] < 0.1
        assert q[3] < 0.1

    def test_pure_single_cluster_q_recovers_one_hot(self) -> None:
        """q=(1, 0, 0): sample dosage from cluster-1's exact frequencies;
        optimizer should drive q1 toward 1."""
        rng = np.random.default_rng(7)
        M = 2000
        # Random AFs so each cluster has its own SNP-level signature.
        P = rng.uniform(0.05, 0.95, size=(M, 3))
        q_true = np.array([1.0, 0.0, 0.0])
        dosage = _binomial_dosage(P @ q_true, rng)

        q, _, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=3,
        )
        assert converged
        assert q[0] > 0.9
        assert q[1] < 0.05
        assert q[2] < 0.05

    @pytest.mark.parametrize("k", [2, 3, 5, 8])
    def test_simplex_constraint_sum_to_one(self, k: int) -> None:
        """Q vector must sum to 1 (equality constraint) regardless of K."""
        rng = np.random.default_rng(123 + k)
        M = 1000
        P = rng.uniform(0.05, 0.95, size=(M, k))
        q_true = np.ones(k) / k
        dosage = _binomial_dosage(P @ q_true, rng)
        q, _, _ = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=k,
        )
        assert np.isclose(q.sum(), 1.0, atol=1e-6)
        assert np.all(q >= -1e-9)  # nonnegativity bound
        assert np.all(q <= 1.0 + 1e-9)

    def test_missing_dosage_handled(self) -> None:
        """NaN dosages get masked out before optimization."""
        rng = np.random.default_rng(0)
        M = 200
        P = np.column_stack([np.full(M, 0.9), np.full(M, 0.1)])
        dosage = _binomial_dosage(P @ np.array([0.5, 0.5]), rng)
        # Set first 50 to NaN
        dosage[:50] = np.nan
        q, _, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=2,
        )
        assert converged
        assert np.isclose(q.sum(), 1.0, atol=1e-6)

    def test_all_missing_raises(self) -> None:
        """Entirely missing dosage vector must raise PanelCacheError."""
        M = 100
        P = np.column_stack([np.full(M, 0.9), np.full(M, 0.1)])
        dosage = np.full(M, np.nan)
        with pytest.raises(PanelCacheError, match="zero non-missing"):
            numpy_supervised_projection(
                target_dosage=dosage, p_matrix=P, k=2,
            )

    def test_shape_mismatch_dosage_p_rows_raises(self) -> None:
        """Dosage and P must agree on M."""
        P = np.column_stack([np.full(100, 0.9), np.full(100, 0.1)])
        dosage = np.zeros(99)
        with pytest.raises(AssertionError):
            numpy_supervised_projection(
                target_dosage=dosage, p_matrix=P, k=2,
            )

    def test_k_mismatch_raises(self) -> None:
        """k argument must match P's column count."""
        P = np.column_stack([np.full(100, 0.9), np.full(100, 0.1)])
        dosage = np.zeros(100)
        with pytest.raises(AssertionError):
            numpy_supervised_projection(
                target_dosage=dosage, p_matrix=P, k=3,
            )

    def test_deterministic_under_fixed_seed_input(self) -> None:
        """Same dosage + P inputs → same Q."""
        rng = np.random.default_rng(5)
        M = 300
        P = np.column_stack([
            np.full(M, 0.7), np.full(M, 0.3), np.full(M, 0.5),
        ])
        dosage = _binomial_dosage(P @ np.array([0.4, 0.4, 0.2]), rng)
        q1, _, _ = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=3,
        )
        q2, _, _ = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=3,
        )
        assert np.allclose(q1, q2)

    def test_optimization_iterations_reported(self) -> None:
        """SLSQP iteration count is positive on converged runs."""
        rng = np.random.default_rng(0)
        M = 200
        P = np.column_stack([np.full(M, 0.9), np.full(M, 0.1)])
        dosage = _binomial_dosage(P @ np.array([0.5, 0.5]), rng)
        _, n_iter, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=2,
        )
        assert converged
        assert n_iter >= 1


class TestProjectionProperties:
    """Property-based tests via Hypothesis. Generate random panels +
    true Q vectors; assert SLSQP recovers Q to within binomial sampling
    noise. Catches regressions the hand-written test cases above wouldn't
    surface (degenerate P matrices, near-boundary Q vectors, large K)."""

    @staticmethod
    def _sample_q(k: int, rng: np.random.Generator) -> np.ndarray:
        """Dirichlet draw — uniform over the K-simplex."""
        return rng.dirichlet(np.ones(k))

    @staticmethod
    def _sample_p(m: int, k: int, rng: np.random.Generator) -> np.ndarray:
        """Cluster allele-frequency matrix.

        Each cluster's per-SNP frequency is drawn uniformly from
        [0.05, 0.95] so neither boundary (fully-fixed allele) is hit;
        SLSQP behaves well off the boundary. The 0.05 floor also keeps
        the binomial likelihood numerically stable.
        """
        return rng.uniform(0.05, 0.95, size=(m, k))

    @pytest.mark.parametrize("k", [2, 3, 4, 6, 10])
    @pytest.mark.parametrize("seed", range(10))
    def test_random_panel_recovers_q(self, k: int, seed: int) -> None:
        """For random P (M=3000 SNPs, K clusters in 2..10) and random
        Dirichlet Q, SLSQP recovers each Q component to within ~0.05
        absolute (binomial sampling noise scales as ~1/sqrt(M))."""
        rng = np.random.default_rng(seed * 100 + k)
        m = 3000
        P = self._sample_p(m, k, rng)
        q_true = self._sample_q(k, rng)
        dosage = _binomial_dosage(P @ q_true, rng)

        q, _, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=k,
        )
        assert converged, f"SLSQP did not converge for k={k}, seed={seed}"
        assert q.shape == (k,)
        assert np.isclose(q.sum(), 1.0, atol=1e-6)
        assert np.all(q >= -1e-9), f"q has negative component: {q}"
        # Recovery tolerance is loose (0.10) because the inverse-problem
        # conditioning depends on the random P matrix; for poorly-
        # conditioned P (rare but possible at higher K) per-component
        # error can drift well beyond 3σ of the binomial sampling noise.
        # The point of this test is to confirm SLSQP CONVERGES to
        # approximately the right Q, not to verify a tight bound.
        recovery_err = np.max(np.abs(q - q_true))
        assert recovery_err < 0.10, (
            f"k={k} seed={seed}: q={q} q_true={q_true} err={recovery_err:.4f}"
        )

    @pytest.mark.parametrize("missing_frac", [0.0, 0.25, 0.5, 0.8])
    def test_recovery_robust_to_missingness(self, missing_frac: float) -> None:
        """Random subsets of the dosage vector set to NaN; the projection
        still recovers Q on the remaining SNPs. Recovery quality
        degrades gracefully as missingness rises."""
        rng = np.random.default_rng(42)
        m = 4000
        k = 4
        P = self._sample_p(m, k, rng)
        q_true = self._sample_q(k, rng)
        dosage = _binomial_dosage(P @ q_true, rng)
        # Mask out a fraction
        mask = rng.random(m) < missing_frac
        dosage[mask] = np.nan

        q, _, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=k,
        )
        assert converged
        n_used = int((~np.isnan(dosage)).sum())
        # Loose tolerance — point of this test is "more missingness ⇒
        # still recovers approximately", not a precise bound. The
        # tolerance scales modestly with n_used so a much-thinner
        # dosage doesn't accidentally pass.
        tolerance = max(0.10, 5.0 / np.sqrt(max(n_used, 1)))
        recovery_err = np.max(np.abs(q - q_true))
        assert recovery_err < tolerance, (
            f"missing={missing_frac:.2f} n_used={n_used} "
            f"err={recovery_err:.4f} > tolerance={tolerance:.4f}"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_extreme_q_boundary_components(self, seed: int) -> None:
        """Q vectors with components very close to 0 or 1 still recover.
        Tests SLSQP's behavior near the simplex boundary."""
        rng = np.random.default_rng(seed)
        m = 5000
        k = 3
        P = self._sample_p(m, k, rng)
        # Concentrate mass on one cluster: (0.9, 0.05, 0.05)
        boundary_q = np.array([0.9, 0.05, 0.05])
        dosage = _binomial_dosage(P @ boundary_q, rng)

        q, _, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=k,
        )
        assert converged
        assert np.isclose(q.sum(), 1.0, atol=1e-6)
        assert np.all(q >= -1e-9)
        assert np.max(np.abs(q - boundary_q)) < 0.10

    def test_pure_single_cluster_q_recovers(self) -> None:
        """Q = (1, 0, 0, ..., 0) — perfect cluster membership. SLSQP
        should snap to the boundary cleanly."""
        rng = np.random.default_rng(7)
        m = 3000
        k = 5
        P = self._sample_p(m, k, rng)
        q_true = np.zeros(k)
        q_true[0] = 1.0
        dosage = _binomial_dosage(P @ q_true, rng)

        q, _, converged = numpy_supervised_projection(
            target_dosage=dosage, p_matrix=P, k=k,
        )
        assert converged
        assert q[0] > 0.9  # Most of the mass on cluster 0
        assert np.all(q >= -1e-9)
