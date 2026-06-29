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
    align_target_to_panel_bim,
    build_panel_cache,
    extract_target_dosage_via_plink2,
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
            min_overlap_snps=0,  # 2000-SNP test fixture; disable the floor
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
            min_overlap_snps=0,  # 2000-SNP test fixture; disable the floor
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
            min_overlap_snps=0,  # 2000-SNP test fixture; disable the floor
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


# ─── D11: strand-ambiguous SNP inversion (real plink2) ────────────────────


# PLINK 1 BED 2-bit genotype codes (see _generate_fixtures.py for the full
# convention). We only need two fixed payloads here; their absolute
# orientation is irrelevant — the test asserts what changes when the SAME
# payload is read against a panel-matching vs. a strand-flipped .bim.
_BED_HOM = 0b00   # a homozygous call
_BED_HET = 0b10   # a heterozygous call

# Panel for the D11 test: one A/T ambiguous SNP + one A/G non-ambiguous
# control. The control keeps the variant set non-empty after exclusion so
# the alignment still runs (and lets us prove the control is unaffected).
_D11_PANEL: list[tuple[str, int, str, str, int]] = [
    ("rsAMB", 1000, "A", "T", _BED_HOM),   # ambiguous, homozygous call
    ("rsCTRL", 2000, "A", "G", _BED_HET),  # control, heterozygous call
]


def _write_single_sample_bed(
    prefix: Path, snps: list[tuple[str, int, str, str, int]],
) -> Path:
    """Write a 1-sample BED triplet. ``snps`` is a list of
    ``(variant_id, bp, allele1, allele2, bed_code)`` rows; each SNP is one
    BED byte (the sample occupies bits 0-1). Returns the ``.bed`` path."""
    prefix.with_suffix(".bim").write_text(
        "\n".join(
            f"1\t{vid}\t0\t{bp}\t{a1}\t{a2}" for vid, bp, a1, a2, _ in snps
        ) + "\n",
    )
    prefix.with_suffix(".fam").write_text("F\tt\t0\t0\t0\t-9\n")
    with prefix.with_suffix(".bed").open("wb") as f:
        f.write(b"\x6c\x1b\x01")
        f.write(bytes(code for *_, code in snps))
    return prefix.with_suffix(".bed")


class TestStrandAmbiguousInversionRealPlink2:
    """D11 regression with the real plink2 binary — the empirically-confirmed
    behavior the unit suite can only assert indirectly (it checks that
    ``--exclude`` is *constructed*, not that plink2 actually inverts without
    it). Runs at the align + dosage-extract seam where the bug lives; needs
    no ADMIXTURE.

    An A/T (strand-ambiguous) SNP's allele set is invariant under a strand
    flip, so ``--alt1-allele <panel.bim>`` (which matches by allele LETTER,
    not strand) silently inverts a homozygous opposite-strand target
    (0 <-> 2). A control A/G SNP is immune (its flipped letters C/T do not
    match the panel, so the forcing is skipped). The build/projection guard
    drops the ambiguous SNP by default so no inversion can occur.
    """

    def _panel_bim(self, tmp_path: Path) -> Path:
        _write_single_sample_bed(tmp_path / "panel", _D11_PANEL)
        return tmp_path / "panel.bim"

    def _aligned_dosage(
        self, *, target_bed: Path, panel_bim: Path, work: Path,
        exclude_strand_ambiguous: bool,
    ) -> dict[str, float]:
        """Run the real align + dosage-extract pipeline and return an
        ``{variant_id: dosage}`` map (order-independent)."""
        runner = SubprocessToolRunner("plink2")
        aligned = align_target_to_panel_bim(
            target_bed=target_bed, panel_bim=panel_bim,
            output_prefix=work / "aligned", plink2_runner=runner,
            log_dir=work / "logs",
            exclude_strand_ambiguous=exclude_strand_ambiguous,
        )
        ids = [
            line.split()[1]
            for line in aligned.with_suffix(".bim").read_text().splitlines()
            if line.strip()
        ]
        dosage = extract_target_dosage_via_plink2(
            target_bed=aligned, output_prefix=work / "dosage",
            plink2_runner=runner, log_dir=work / "logs",
        )
        return dict(zip(ids, [float(x) for x in dosage], strict=True))

    def test_kept_ambiguous_snp_inverts_opposite_strand_homozygote(
        self, tmp_path: Path,
    ) -> None:
        """With the SNP kept (exclude_strand_ambiguous=False), the SAME
        genotype payload read against a strand-flipped A/T .bim yields the
        INVERTED homozygous dosage, while the control A/G SNP is unchanged —
        the silent corruption D11 prevents."""
        panel_bim = self._panel_bim(tmp_path)

        # Same payload, two allele orderings of the A/T SNP. "panel" matches
        # the panel's A/T order; "flip" is the opposite-strand encoding (T/A).
        # The control A/G SNP is identical in both.
        panel_order = _write_single_sample_bed(
            tmp_path / "t_panel",
            [("rsAMB", 1000, "A", "T", _BED_HOM),
             ("rsCTRL", 2000, "A", "G", _BED_HET)],
        )
        flipped = _write_single_sample_bed(
            tmp_path / "t_flip",
            [("rsAMB", 1000, "T", "A", _BED_HOM),
             ("rsCTRL", 2000, "A", "G", _BED_HET)],
        )

        d_panel = self._aligned_dosage(
            target_bed=panel_order, panel_bim=panel_bim,
            work=tmp_path / "w_panel", exclude_strand_ambiguous=False,
        )
        d_flip = self._aligned_dosage(
            target_bed=flipped, panel_bim=panel_bim,
            work=tmp_path / "w_flip", exclude_strand_ambiguous=False,
        )

        # Control A/G SNP: strand flip cannot affect it (letters don't match).
        assert d_flip["rsCTRL"] == d_panel["rsCTRL"]
        # Ambiguous A/T SNP: the homozygote is silently inverted (0 <-> 2).
        assert d_panel["rsAMB"] in (0.0, 2.0)
        assert d_flip["rsAMB"] == 2.0 - d_panel["rsAMB"]
        assert d_flip["rsAMB"] != d_panel["rsAMB"]

    def test_default_exclusion_drops_ambiguous_snp(
        self, tmp_path: Path,
    ) -> None:
        """With the default guard (exclude_strand_ambiguous=True), the A/T
        SNP is dropped from the alignment entirely, so no inversion is
        possible — the control A/G SNP survives and is scored normally."""
        panel_bim = self._panel_bim(tmp_path)
        flipped = _write_single_sample_bed(
            tmp_path / "t_flip",
            [("rsAMB", 1000, "T", "A", _BED_HOM),
             ("rsCTRL", 2000, "A", "G", _BED_HET)],
        )

        dosage = self._aligned_dosage(
            target_bed=flipped, panel_bim=panel_bim,
            work=tmp_path / "w", exclude_strand_ambiguous=True,
        )

        assert "rsAMB" not in dosage  # ambiguous SNP excluded, cannot invert
        assert "rsCTRL" in dosage     # control retained and scored
