"""End-to-end integration test with real ADMIXTURE + plink2 binaries.

Runs the full library pipeline against the synthetic 3-cluster fixture
in `fixtures/`:

1. `build_panel_cache` invokes real ADMIXTURE (supervised mode, K=3)
   on the 90-sample × 2000-SNP panel, producing a real cached P + Q.
2. `project_target` invokes real plink2 for variant intersection +
   REF/ALT alignment + dosage extraction, then runs the NumPy SLSQP
   solver against the cached P.
3. The recovered Q vector for each of 4 held-out targets is compared
   against the truth in `fixtures/truth.json`.

This is what catches the next class of bugs the unit suite misses:

- ADMIXTURE binary format drift (Q/P file shape, log line format)
- plink2 BED / `--alt1-allele` / `--recode A` output conventions
- The library's own end-to-end correctness numbers (the docstring
  claim of ~1e-5 Q match vs. stock ADMIXTURE is otherwise unverified)

Skipped by default — pytest only collects the `integration` marker
when explicitly requested via `pytest -m integration`. CI runs a
dedicated Linux job that installs both binaries before invoking
this suite. macOS is unsupported (ADMIXTURE 1.4 has no macOS
binary; 1.3.0 SIGSEGV's on modern kernels but works on Mac, so the
test could in principle run there too — but ADMIXTURE 1.3.0 vs.
1.4.0 produce sufficiently different Q vectors at small panels
that asserting on truth would be fragile across the version split).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from admixture_cache import (
    SubprocessToolRunner,
    build_panel_cache,
    project_target,
)

# All tests in this module require the `integration` marker.
pytestmark = pytest.mark.integration


FIXTURES = Path(__file__).parent / "fixtures"
# Tolerance is loose: 2000 SNPs gives ~0.022 per-component binomial
# sampling noise (1/√2000); 0.10 absolute leaves ~4σ headroom for
# the worst component across all 4 targets. Empirically on the
# fixture, ADMIXTURE 1.3.0 + plink2 a.7.1 recovers Q to within
# ~0.03 absolute, so the 0.10 tolerance also covers small
# numerical drift between ADMIXTURE 1.3.0 (macOS) and 1.4.0 (Linux).
Q_RECOVERY_TOLERANCE = 0.10


def _binary_available(name: str) -> bool:
    return shutil.which(name) is not None


# ─── module-level skip conditions ────────────────────────────────────────

# Runtime requires both binaries on PATH. Production target is
# ADMIXTURE 1.4.0 (Linux only — see DEVELOPMENT.md) + plink2
# >= a.7.1. macOS dev environments can run the suite against
# ADMIXTURE 1.3.0 (the only macOS option); recovery numerics are
# close enough that the same tolerance holds.

if not _binary_available("admixture"):
    pytest.skip(
        "`admixture` not on PATH; download ADMIXTURE 1.4.0 Linux "
        "binary from https://dalexander.github.io/admixture/download.html "
        "(or 1.3.0 macOS for dev), and ensure it's on PATH as `admixture`.",
        allow_module_level=True,
    )

if not _binary_available("plink2"):
    pytest.skip(
        "`plink2` not on PATH; install from "
        "https://www.cog-genomics.org/plink/2.0/",
        allow_module_level=True,
    )

# Marker for tests that ONLY make sense against ADMIXTURE 1.4
# (e.g., a hypothetical assertion on a 1.4-specific output line).
# Currently no such tests exist; if you add one, gate it with
# `@pytest.mark.skipif(_admixture_version() != "1.4.0", reason=...)`.
del sys  # unused if no version-specific skips below


# ─── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def truth() -> dict:
    """Load the per-target known Q vectors written by
    `_generate_fixtures.py`."""
    return json.loads((FIXTURES / "truth.json").read_text())


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the panel cache once per test module.

    Uses K=3, seeds=[1] (single restart — no multimodality validation
    because the panel is small and the labels are exhaustive). The
    cache itself ends up at <tmp>/cache/.
    """
    work_root = tmp_path_factory.mktemp("e2e")
    cache = work_root / "cache"

    admixture_runner = SubprocessToolRunner("admixture")

    manifest = build_panel_cache(
        panel_bed=FIXTURES / "panel.bed",
        panel_pop_file=FIXTURES / "panel.pop",
        clusters_yaml=FIXTURES / "clusters.yaml",
        k=3,
        cache_dir=cache,
        admixture_runner=admixture_runner,
        track="regional",
        panel_id="integration_synth",
        panel_version="v1.0",
        admixture_version="1.4.0",
        seeds=[1],
        # Loose sd_threshold because we're running a SINGLE seed —
        # the multimodality check (SD across restarts) is degenerate
        # at N=1. The check still fires for code-path coverage; it
        # passes trivially because std of one value is zero.
        sd_threshold=10.0,
        threads=2,
        max_parallel_restarts=1,
        per_restart_timeout_seconds=300,
    )

    # Sanity-check the manifest reflects the build.
    assert manifest.k == 3
    assert manifest.cluster_order == ["A", "B", "C"]
    assert manifest.best_seed == 1
    assert manifest.best_loglikelihood < 0  # real LL is negative
    return cache


# ─── tests ───────────────────────────────────────────────────────────────


class TestBuildPanelCache:
    """End-to-end build verification — does ADMIXTURE actually produce
    a usable cache through our wrapper?"""

    def test_cache_directory_layout(self, cache_dir: Path) -> None:
        """Every file the documented cache contract names must be on disk."""
        assert (cache_dir / "manifest.json").is_file()
        assert (cache_dir / "panel.3.P").is_file()
        assert (cache_dir / "panel.3.Q").is_file()
        assert (cache_dir / "panel.bim").is_file()
        assert (cache_dir / "restart_sd.json").is_file()
        assert (cache_dir / "cluster_order.json").is_file()
        assert (cache_dir / "build_logs").is_dir()

    def test_cached_p_shape_matches_panel(self, cache_dir: Path) -> None:
        """Cached P is (M_snps, K) — 2000 × 3 for our fixture."""
        p_matrix = np.loadtxt(cache_dir / "panel.3.P")
        assert p_matrix.shape == (2000, 3)
        # All entries in (0, 1) — they're allele frequencies.
        assert np.all((p_matrix > 0.0) & (p_matrix < 1.0))

    def test_cached_q_shape_matches_panel(self, cache_dir: Path) -> None:
        """Cached Q is (N_samples, K) — 90 × 3 for our fixture.

        Supervised mode means every panel sample's Q should
        concentrate on its cluster — A samples have Q[A] ≈ 1, etc."""
        q_matrix = np.loadtxt(cache_dir / "panel.3.Q")
        assert q_matrix.shape == (90, 3)
        # Each row sums to 1.
        np.testing.assert_allclose(q_matrix.sum(axis=1), 1.0, atol=1e-6)
        # The first 30 samples are cluster A — expect Q[:,0] ≈ 1.
        np.testing.assert_array_less(0.95, q_matrix[:30, 0].mean())
        # The middle 30 are cluster B.
        np.testing.assert_array_less(0.95, q_matrix[30:60, 1].mean())
        # The last 30 are cluster C.
        np.testing.assert_array_less(0.95, q_matrix[60:90, 2].mean())

    def test_cluster_order_matches_pop_file(self, cache_dir: Path) -> None:
        """`_derive_cluster_order_from_pop_file` must match what
        ADMIXTURE actually used (first-appearance order)."""
        co = json.loads((cache_dir / "cluster_order.json").read_text())
        assert co["cluster_order"] == ["A", "B", "C"]


class TestProjectTarget:
    """End-to-end projection — does the NumPy SLSQP path recover Q
    against the cached P?"""

    @pytest.mark.parametrize("target_name", [
        "target_pure_A",
        "target_pure_C",
        "target_AB_5050",
        "target_three",
    ])
    def test_recovers_known_q_within_tolerance(
        self,
        target_name: str,
        cache_dir: Path,
        truth: dict,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """For each held-out target with a known Q, the projection
        recovers it within ~0.10 absolute per-component."""
        plink2_runner = SubprocessToolRunner("plink2")
        work_dir = Path(tmp_path) / f"proj_{target_name}"
        result = project_target(
            target_bed=FIXTURES / f"{target_name}.bed",
            cache_dir=cache_dir,
            plink2_runner=plink2_runner,
            work_dir=work_dir,
        )
        assert result.converged
        assert result.cluster_order == ["A", "B", "C"]
        np.testing.assert_allclose(result.target_q.sum(), 1.0, atol=1e-6)
        q_true = np.array(truth["targets"][target_name])
        max_err = float(np.max(np.abs(result.target_q - q_true)))
        assert max_err < Q_RECOVERY_TOLERANCE, (
            f"{target_name}: recovered Q={result.target_q.tolist()} "
            f"differs from truth Q={q_true.tolist()} by {max_err:.4f} "
            f"(tolerance {Q_RECOVERY_TOLERANCE})"
        )

    def test_n_snps_used_matches_panel_size(
        self,
        cache_dir: Path,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """All 2000 fixture SNPs are non-missing → projection uses
        all of them."""
        plink2_runner = SubprocessToolRunner("plink2")
        work_dir = Path(tmp_path) / "proj_nsnps"
        result = project_target(
            target_bed=FIXTURES / "target_pure_A.bed",
            cache_dir=cache_dir,
            plink2_runner=plink2_runner,
            work_dir=work_dir,
        )
        assert result.n_snps_used == 2000

    def test_recovers_q_when_target_missing_panel_snps(
        self,
        cache_dir: Path,
        truth: dict,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Regression guard for the v1.4.1 fix (target∩panel < panel).

        The other fixture targets carry 100% of the panel's SNPs, so they
        never exercised the common real-world case where the target is missing
        some panel variants — and that's exactly the case the old code got
        wrong (``--extract panel.bim`` yields a dosage shorter than P, in the
        target's order, so project_target aborted with "cached P has N SNPs but
        aligned target dosage has M", or — at coincidental equal length —
        silently mis-aligned the dosage row-for-row against P).

        Here we drop 100 of target_pure_A's 2000 variants with ``plink2
        --exclude`` and confirm the projection still reindexes to the full
        panel order (NaN-filling the 100 gaps), uses only the 1900 present
        SNPs, and recovers the known Q. This test FAILS on pre-1.4.1 code."""
        plink2_runner = SubprocessToolRunner("plink2")
        work_dir = Path(tmp_path) / "proj_partial"
        (work_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Drop every 20th panel SNP (100 of 2000) from the target.
        panel_ids = [
            line.split("\t")[1]
            for line in (cache_dir / "panel.bim").read_text().splitlines()
            if line.strip()
        ]
        dropped = panel_ids[::20]
        drop_file = work_dir / "drop.txt"
        drop_file.write_text("\n".join(dropped) + "\n")
        reduced = work_dir / "target_reduced"
        plink2_runner.run(
            args=[
                "--bfile", str(FIXTURES / "target_pure_A"),
                "--exclude", str(drop_file),
                "--make-bed", "--out", str(reduced),
            ],
            cwd=work_dir,
            log_dir=work_dir / "logs",
        )

        result = project_target(
            target_bed=reduced.with_suffix(".bed"),
            cache_dir=cache_dir,
            plink2_runner=plink2_runner,
            work_dir=work_dir / "proj",
        )
        assert result.converged
        # Only the SNPs the target still carries are used (NaN-filled gaps
        # are dropped by the projection's non-missing mask).
        assert result.n_snps_used == 2000 - len(dropped)
        q_true = np.array(truth["targets"]["target_pure_A"])
        max_err = float(np.max(np.abs(result.target_q - q_true)))
        assert max_err < Q_RECOVERY_TOLERANCE, (
            f"partial-overlap target recovered Q={result.target_q.tolist()} "
            f"differs from truth {q_true.tolist()} by {max_err:.4f}"
        )


class TestCacheReuse:
    """Idempotency — second `build_panel_cache` call with matching SHAs
    must skip ADMIXTURE and return the existing manifest."""

    def test_rebuild_is_idempotent(self, cache_dir: Path) -> None:
        """Pre-existing cache + matching panel/yaml SHAs → no rebuild."""
        # Record the manifest content + mtime; rebuild; verify
        # neither changed (i.e., the manifest file wasn't rewritten).
        manifest_path = cache_dir / "manifest.json"
        original_bytes = manifest_path.read_bytes()
        original_mtime = manifest_path.stat().st_mtime

        admixture_runner = SubprocessToolRunner("admixture")
        manifest = build_panel_cache(
            panel_bed=FIXTURES / "panel.bed",
            panel_pop_file=FIXTURES / "panel.pop",
            clusters_yaml=FIXTURES / "clusters.yaml",
            k=3,
            cache_dir=cache_dir,
            admixture_runner=admixture_runner,
            track="regional",
            panel_id="integration_synth",
            panel_version="v1.0",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
            threads=2,
            max_parallel_restarts=1,
            per_restart_timeout_seconds=300,
        )
        assert manifest.k == 3
        assert manifest_path.read_bytes() == original_bytes
        assert manifest_path.stat().st_mtime == original_mtime
