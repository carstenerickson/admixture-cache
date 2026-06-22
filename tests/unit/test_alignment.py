"""Smoke tests for target-to-panel alignment + dosage extraction.

plink2 is mocked: we assert the args plink2 would receive and the
post-call error handling for missing-output cases. Real plink2 is
exercised in integration tests, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from admixture_cache import (
    PanelCacheError,
    align_target_to_panel_bim,
    extract_target_dosage_via_plink2,
)
from admixture_cache.alignment import (
    is_strand_ambiguous,
    reindex_dosage_to_panel_order,
    strand_ambiguous_variant_ids,
)


class _MockRunner:
    """Minimal ToolRunner that records args and optionally writes a stub
    output BED/raw next to output_prefix to keep downstream code happy."""

    def __init__(
        self,
        *,
        emit_bed: bool = False,
        emit_raw: bool = False,
        raw_content: str | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.emit_bed = emit_bed
        self.emit_raw = emit_raw
        self.raw_content = raw_content

    def run(
        self,
        *,
        args: list[str],
        cwd: Path,
        log_dir: Path,
        timeout_seconds: int = 600,
        log_name: str | None = None,
    ) -> object:
        self.calls.append({
            "args": list(args),
            "cwd": cwd,
            "log_dir": log_dir,
            "timeout_seconds": timeout_seconds,
        })
        # Find --out and emit stub files for downstream existence checks.
        # Real plink2 --make-bed emits the full BED triplet; mirror that
        # so v1.1.1's triplet-completeness check in
        # `align_target_to_panel_bim` doesn't trip on the stub output.
        if "--out" in args:
            out_prefix = Path(args[args.index("--out") + 1])
            if self.emit_bed:
                out_prefix.with_suffix(".bed").touch()
                out_prefix.with_suffix(".bim").touch()
                out_prefix.with_suffix(".fam").touch()
            if self.emit_raw and self.raw_content is not None:
                out_prefix.with_suffix(".raw").write_text(self.raw_content)
        return None


def _write_bed_triplet(tmp_path: Path, stem: str = "target") -> Path:
    """Create a minimal BED triplet (.bed/.bim/.fam) so the new
    sibling-file validation in `_detect_target_format` accepts the
    path. Returns the .bed path."""
    bed = tmp_path / f"{stem}.bed"
    bed.write_bytes(b"\x6c\x1b\x01")
    (tmp_path / f"{stem}.bim").write_text("1\trs1\t0\t1\tA\tC\n")
    (tmp_path / f"{stem}.fam").write_text("F\tI\t0\t0\t0\t-9\n")
    return bed


def _write_panel_bim(tmp_path: Path, name: str = "panel.bim") -> Path:
    """Write a minimal NON-ambiguous panel .bim so
    `align_target_to_panel_bim`'s strand-ambiguous scan (which reads
    panel.bim) finds nothing to exclude. Returns the .bim path."""
    path = tmp_path / name
    path.write_text("1\trs1\t0\t1\tA\tC\n1\trs2\t0\t2\tA\tG\n")
    return path


class TestAlignTargetToPanelBim:
    @pytest.fixture(autouse=True)
    def _ensure_target_triplet(self, tmp_path: Path) -> None:
        """v1.1.1 added sibling-file validation in
        `_detect_target_format`; tests that pass `target.bed` paths
        now need the full BED triplet on disk. Autouse fixture
        creates one per-test so each test_* method's `tmp_path /
        'target.bed'` resolves to a valid BED triplet. Also writes a
        non-ambiguous panel.bim, which the strand-ambiguous scan now
        reads on every alignment call."""
        _write_bed_triplet(tmp_path)
        _write_panel_bim(tmp_path)

    def test_args_constructed_correctly(self, tmp_path: Path) -> None:
        runner = _MockRunner(emit_bed=True)
        out_prefix = tmp_path / "aligned"

        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=out_prefix,
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )

        assert len(runner.calls) == 1
        args = runner.calls[0]["args"]

        # --bfile target prefix (no .bed suffix)
        assert "--bfile" in args
        bfile_idx = args.index("--bfile")
        assert args[bfile_idx + 1] == str(tmp_path / "target")

        # --extract panel.bim
        assert "--extract" in args
        ext_idx = args.index("--extract")
        assert args[ext_idx + 1] == str(tmp_path / "panel.bim")

        # --alt1-allele points at panel.bim with cols 5 (ALT) and 2 (ID)
        assert "--alt1-allele" in args
        aa_idx = args.index("--alt1-allele")
        assert args[aa_idx + 1] == str(tmp_path / "panel.bim")
        assert args[aa_idx + 2] == "5"
        assert args[aa_idx + 3] == "2"

        # --make-bed asserted
        assert "--make-bed" in args

        # --out prefix matches
        assert "--out" in args
        out_idx = args.index("--out")
        assert args[out_idx + 1] == str(out_prefix)

    def test_returns_path_to_emitted_bed(self, tmp_path: Path) -> None:
        runner = _MockRunner(emit_bed=True)
        out_prefix = tmp_path / "aligned"
        result = align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=out_prefix,
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        assert result == out_prefix.with_suffix(".bed")
        assert result.exists()

    def test_missing_bed_after_run_raises(self, tmp_path: Path) -> None:
        """If plink2 'succeeded' (runner returned) but the aligned BED
        triplet didn't appear, we raise a clear PanelCacheError naming
        the missing siblings."""
        runner = _MockRunner(emit_bed=False)
        with pytest.raises(PanelCacheError, match="incomplete BED triplet"):
            align_target_to_panel_bim(
                target_bed=tmp_path / "target.bed",
                panel_bim=tmp_path / "panel.bim",
                output_prefix=tmp_path / "aligned",
                plink2_runner=runner,
                log_dir=tmp_path / "logs",
            )

    def test_log_dir_created(self, tmp_path: Path) -> None:
        runner = _MockRunner(emit_bed=True)
        log_dir = tmp_path / "deep" / "logs"
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=log_dir,
        )
        assert log_dir.is_dir()

    def test_output_dir_created(self, tmp_path: Path) -> None:
        runner = _MockRunner(emit_bed=True)
        out_prefix = tmp_path / "nested" / "subdir" / "aligned"
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=out_prefix,
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        assert out_prefix.parent.is_dir()

    def test_timeout_default_propagated(self, tmp_path: Path) -> None:
        runner = _MockRunner(emit_bed=True)
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        assert runner.calls[0]["timeout_seconds"] == 600

    def test_timeout_override_propagated(self, tmp_path: Path) -> None:
        runner = _MockRunner(emit_bed=True)
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
            timeout_seconds=1234,
        )
        assert runner.calls[0]["timeout_seconds"] == 1234


class TestAlignmentFullTripletValidation:
    """v1.1.1 second-pass: `align_target_to_panel_bim` must validate
    the full BED triplet, not just `.bed`. A partial plink2 output
    surfacing as a `extract_target_dosage_via_plink2` failure later
    would mis-attribute the cause."""

    def test_partial_output_raises_with_actionable_message(
        self, tmp_path: Path,
    ) -> None:
        """plink2 returned success but produced only `.bed` (e.g. disk
        full mid-write). Function must raise PanelCacheError naming
        which sibling files are missing."""
        _write_bed_triplet(tmp_path)
        _write_panel_bim(tmp_path)

        class _BedOnlyRunner:
            """Emits the .bed but not .bim/.fam to simulate a
            truncated plink2 write."""

            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 600,
                log_name: str | None = None,
            ) -> object:
                out_idx = args.index("--out")
                out_prefix = Path(args[out_idx + 1])
                out_prefix.with_suffix(".bed").touch()
                # Deliberately NOT writing .bim / .fam
                return None

        with pytest.raises(PanelCacheError) as exc_info:
            align_target_to_panel_bim(
                target_bed=tmp_path / "target.bed",
                panel_bim=tmp_path / "panel.bim",
                output_prefix=tmp_path / "aligned",
                plink2_runner=_BedOnlyRunner(),
                log_dir=tmp_path / "logs",
            )
        msg = str(exc_info.value)
        assert "incomplete BED triplet" in msg
        assert "aligned.bim" in msg
        assert "aligned.fam" in msg


class TestLdPrunePanelDottedPrefixRegression:
    """v1.1.1 second-pass: `ld_prune_panel`'s sibling probing must use
    APPEND semantics (not `Path.with_suffix`) so caller-supplied
    `output_prefix` with dots in the stem (e.g. `aadr.v66.pruned`)
    don't silently mis-probe."""

    def test_dotted_output_prefix_detects_plink2_output_correctly(
        self, tmp_path: Path,
    ) -> None:
        """With `output_prefix=cohort.v2`, plink2 produces
        `cohort.v2.prune.in` and `cohort.v2.bed`. The function must
        probe the correct paths — pre-fix v1.1.1 would have looked
        for `cohort.prune.in` (stripping `.v2`)."""
        from admixture_cache.builder import ld_prune_panel

        # Create a minimal panel.bed/.bim/.fam.
        panel_bed = tmp_path / "panel.bed"
        panel_bed.write_bytes(b"\x6c\x1b\x01")
        (tmp_path / "panel.bim").write_text("1\trs0\t0\t1\tA\tC\n")
        (tmp_path / "panel.fam").write_text("F\tI\t0\t0\t0\t-9\n")

        class _DottedStemRunner:
            """Mirrors plink2: writes outputs at
            `<output_prefix><suffix>` (APPEND), not REPLACE."""

            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 3600,
                log_name: str | None = None,
            ) -> object:
                from admixture_cache._paths import append_suffix

                out_idx = args.index("--out")
                out_prefix = Path(args[out_idx + 1])
                if "--indep-pairwise" in args:
                    append_suffix(out_prefix, ".prune.in").write_text("rs0\n")
                if "--extract" in args:
                    append_suffix(out_prefix, ".bed").write_bytes(b"\x6c\x1b\x01")
                    append_suffix(out_prefix, ".bim").write_text(
                        "1\trs0\t0\t1\tA\tC\n",
                    )
                    append_suffix(out_prefix, ".fam").write_text(
                        "F\tI\t0\t0\t0\t-9\n",
                    )
                return None

        result = ld_prune_panel(
            panel_bed=panel_bed,
            output_prefix=tmp_path / "cohort.v2",  # dotted stem
            plink2_runner=_DottedStemRunner(),
            log_dir=tmp_path / "logs",
        )
        assert result == tmp_path / "cohort.v2.bed"
        assert result.exists()


class TestTargetFormatDottedStemRegression:
    """v1.1.1 regression: `_detect_target_format` on a suffixless input
    with a dotted stem (e.g. `cohort.v2`) must NOT replace the trailing
    `.v2` segment when probing sibling files. Earlier `Path.with_suffix`
    behavior would silently probe `cohort.pgen` (replacing `.v2`)
    instead of `cohort.v2.pgen` (appending)."""

    def test_dotted_stem_probes_appended_not_replaced(self, tmp_path: Path) -> None:
        from admixture_cache.alignment import _detect_target_format

        # Lay down a BED triplet at the dotted stem; the FAKE wrong
        # stem (without `.v2`) is deliberately absent so the test
        # fails if `with_suffix` semantics sneak back in.
        bed = tmp_path / "cohort.v2.bed"
        bed.write_bytes(b"\x6c\x1b\x01")
        (tmp_path / "cohort.v2.bim").write_text("1\trs1\t0\t1\tA\tC\n")
        (tmp_path / "cohort.v2.fam").write_text("F\tI\t0\t0\t0\t-9\n")

        # Pass a suffixless dotted path to force the no-suffix probe.
        flag, stem = _detect_target_format(tmp_path / "cohort.v2")
        assert flag == "--bfile"
        assert stem == tmp_path / "cohort.v2"

    def test_dotted_stem_doesnt_collide_with_unrelated_short_stem(
        self, tmp_path: Path,
    ) -> None:
        """If an unrelated `cohort.pgen` happens to exist nearby, the
        detector for `cohort.v2` must NOT pick it up (the v1.1.0 bug
        would silently return --pfile against the wrong cohort)."""
        from admixture_cache.alignment import _detect_target_format

        # Drop a misleading sibling
        (tmp_path / "cohort.pgen").write_bytes(b"\x6c\x1b\x10\x00")
        (tmp_path / "cohort.psam").write_text("#FID\tIID\nF\tI\n")
        (tmp_path / "cohort.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t1\trs1\tA\tC\n")

        # cohort.v2 has no triplet → must raise, NOT silently return
        # --pfile with `cohort` as stem.
        with pytest.raises(PanelCacheError, match="not found as either"):
            _detect_target_format(tmp_path / "cohort.v2")


class TestTargetFormatSiblingValidation:
    """v1.1.1: explicit `.bed` / `.pgen` suffix branches must validate
    that all three triplet siblings exist, not just the named file."""

    def test_bed_with_missing_bim_raises(self, tmp_path: Path) -> None:
        from admixture_cache.alignment import _detect_target_format

        (tmp_path / "target.bed").write_bytes(b"\x6c\x1b\x01")
        # Deliberately omit .bim
        (tmp_path / "target.fam").write_text("F\tI\t0\t0\t0\t-9\n")
        with pytest.raises(PanelCacheError, match=r"BED triplet.*incomplete.*\.bim"):
            _detect_target_format(tmp_path / "target.bed")

    def test_pgen_with_missing_psam_raises(self, tmp_path: Path) -> None:
        from admixture_cache.alignment import _detect_target_format

        (tmp_path / "target.pgen").write_bytes(b"\x6c\x1b\x10\x00")
        # Omit .psam
        (tmp_path / "target.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n")
        with pytest.raises(PanelCacheError, match=r"PGEN triplet.*incomplete.*\.psam"):
            _detect_target_format(tmp_path / "target.pgen")


class TestTargetFormatDetection:
    """`_detect_target_format` returns the right plink2 input flag
    based on the target's extension or sibling-file presence."""

    def _make_bed_triplet(self, tmp_path: Path, stem: str = "target") -> Path:
        bed = tmp_path / f"{stem}.bed"
        bed.write_bytes(b"\x6c\x1b\x01")
        (tmp_path / f"{stem}.bim").write_text("1\trs1\t0\t1\tA\tC\n")
        (tmp_path / f"{stem}.fam").write_text("F\tI\t0\t0\t0\t-9\n")
        return bed

    def _make_pgen_triplet(self, tmp_path: Path, stem: str = "target") -> Path:
        pgen = tmp_path / f"{stem}.pgen"
        pgen.write_bytes(b"\x6c\x1b\x10\x00")  # any non-empty content
        (tmp_path / f"{stem}.psam").write_text("#FID\tIID\nF\tI\n")
        (tmp_path / f"{stem}.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t1\trs1\tA\tC\n")
        return pgen

    def test_bed_suffix_detected(self, tmp_path: Path) -> None:
        from admixture_cache.alignment import _detect_target_format

        bed = self._make_bed_triplet(tmp_path)
        flag, stem = _detect_target_format(bed)
        assert flag == "--bfile"
        assert stem == tmp_path / "target"

    def test_pgen_suffix_detected(self, tmp_path: Path) -> None:
        from admixture_cache.alignment import _detect_target_format

        pgen = self._make_pgen_triplet(tmp_path)
        flag, stem = _detect_target_format(pgen)
        assert flag == "--pfile"
        assert stem == tmp_path / "target"

    def test_suffixless_prefers_pgen(self, tmp_path: Path) -> None:
        """When both BED and PGEN siblings exist and the user passes a
        no-suffix stem, PGEN wins (it's the more modern format)."""
        from admixture_cache.alignment import _detect_target_format

        self._make_bed_triplet(tmp_path)
        self._make_pgen_triplet(tmp_path)
        flag, _stem = _detect_target_format(tmp_path / "target")
        assert flag == "--pfile"

    def test_suffixless_falls_back_to_bed(self, tmp_path: Path) -> None:
        """No-suffix stem with only BED siblings → --bfile."""
        from admixture_cache.alignment import _detect_target_format

        self._make_bed_triplet(tmp_path)
        flag, _stem = _detect_target_format(tmp_path / "target")
        assert flag == "--bfile"

    def test_missing_raises_actionable_error(self, tmp_path: Path) -> None:
        from admixture_cache.alignment import _detect_target_format

        with pytest.raises(PanelCacheError, match="not found as either"):
            _detect_target_format(tmp_path / "nonexistent")


class TestAlignWithPgenInput:
    """When target_bed points at a PGEN, align_target_to_panel_bim
    swaps --bfile for --pfile in the plink2 call."""

    def test_pgen_input_uses_pfile_flag(self, tmp_path: Path) -> None:
        pgen = tmp_path / "target.pgen"
        pgen.write_bytes(b"\x6c\x1b\x10\x00")
        (tmp_path / "target.psam").write_text("#FID\tIID\nF\tI\n")
        (tmp_path / "target.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n")
        (tmp_path / "panel.bim").write_text("")
        runner = _MockRunner(emit_bed=True)
        align_target_to_panel_bim(
            target_bed=pgen,  # PGEN path passed in BED-named parameter
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        args = runner.calls[0]["args"]
        assert "--pfile" in args
        assert "--bfile" not in args
        # The stem (path without suffix) follows --pfile.
        pfile_idx = args.index("--pfile")
        assert args[pfile_idx + 1] == str(tmp_path / "target")

    def test_bed_input_still_uses_bfile_flag(self, tmp_path: Path) -> None:
        """Sanity: BED input keeps the existing --bfile behavior."""
        bed = tmp_path / "target.bed"
        bed.write_bytes(b"\x6c\x1b\x01")
        (tmp_path / "target.bim").write_text("")
        (tmp_path / "target.fam").write_text("")
        (tmp_path / "panel.bim").write_text("")
        runner = _MockRunner(emit_bed=True)
        align_target_to_panel_bim(
            target_bed=bed,
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        args = runner.calls[0]["args"]
        assert "--bfile" in args
        assert "--pfile" not in args


class TestExtractTargetDosageViaPlink2:
    @pytest.fixture(autouse=True)
    def _ensure_target_triplet(self, tmp_path: Path) -> None:
        _write_bed_triplet(tmp_path)

    def _raw_for(self, dosages: list[str]) -> str:
        """Build a plink2 --recode A .raw file with one sample row."""
        header = "FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE\t" + "\t".join(
            f"snp{i}_A" for i in range(len(dosages))
        )
        row = "FID1\tIID1\t0\t0\t1\t-9\t" + "\t".join(dosages)
        return header + "\n" + row + "\n"

    def test_parses_raw_dosage_correctly(self, tmp_path: Path) -> None:
        raw = self._raw_for(["0", "1", "2", "NA", "0"])
        runner = _MockRunner(emit_raw=True, raw_content=raw)
        out_prefix = tmp_path / "dosage"

        dosage = extract_target_dosage_via_plink2(
            target_bed=tmp_path / "target.bed",
            output_prefix=out_prefix,
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        assert dosage.shape == (5,)
        assert dosage.dtype == np.float64
        np.testing.assert_array_equal(
            np.isnan(dosage), np.array([False, False, False, True, False]),
        )
        assert dosage[0] == 0.0
        assert dosage[1] == 1.0
        assert dosage[2] == 2.0

    def test_args_constructed_correctly(self, tmp_path: Path) -> None:
        raw = self._raw_for(["0", "1"])
        runner = _MockRunner(emit_raw=True, raw_content=raw)
        out_prefix = tmp_path / "dosage"

        extract_target_dosage_via_plink2(
            target_bed=tmp_path / "target.bed",
            output_prefix=out_prefix,
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        args = runner.calls[0]["args"]
        assert "--bfile" in args
        assert args[args.index("--bfile") + 1] == str(tmp_path / "target")
        assert "--recode" in args
        assert args[args.index("--recode") + 1] == "A"
        assert "--out" in args
        assert args[args.index("--out") + 1] == str(out_prefix)

    def test_missing_raw_after_run_raises(self, tmp_path: Path) -> None:
        runner = _MockRunner(emit_raw=False)
        with pytest.raises(PanelCacheError, match="not produced"):
            extract_target_dosage_via_plink2(
                target_bed=tmp_path / "target.bed",
                output_prefix=tmp_path / "dosage",
                plink2_runner=runner,
                log_dir=tmp_path / "logs",
            )

    def test_multi_sample_raw_rejected(self, tmp_path: Path) -> None:
        """The function is for single-target projection only."""
        header = "FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE\tsnp0_A"
        rows = "F1\tI1\t0\t0\t1\t-9\t1\nF2\tI2\t0\t0\t1\t-9\t0\n"
        runner = _MockRunner(emit_raw=True, raw_content=header + "\n" + rows)

        with pytest.raises(PanelCacheError, match="expected 1 sample"):
            extract_target_dosage_via_plink2(
                target_bed=tmp_path / "target.bed",
                output_prefix=tmp_path / "dosage",
                plink2_runner=runner,
                log_dir=tmp_path / "logs",
            )


class TestProtocolConformance:
    """The Protocol allows duck-typed runners; verify our function works
    with both a class instance and a MagicMock that supports keyword
    `run(args=..., cwd=..., log_dir=..., timeout_seconds=...)`."""

    @pytest.fixture(autouse=True)
    def _ensure_target_triplet(self, tmp_path: Path) -> None:
        _write_bed_triplet(tmp_path)
        _write_panel_bim(tmp_path)

    def test_magicmock_runner_works(self, tmp_path: Path) -> None:
        runner = MagicMock()

        def fake_run(**kwargs: Any) -> None:
            # Emit the full BED triplet — v1.1.1 validates that all
            # three sibling files exist post-plink2 (not just .bed).
            out_idx = kwargs["args"].index("--out")
            out_prefix = Path(kwargs["args"][out_idx + 1])
            for s in (".bed", ".bim", ".fam"):
                out_prefix.with_suffix(s).touch()

        runner.run.side_effect = fake_run
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        runner.run.assert_called_once()


class TestStrandAmbiguousPredicate:
    """D11: detecting strand-ambiguous (A/T, C/G) SNPs."""

    @pytest.mark.parametrize(
        "a1,a2",
        [("A", "T"), ("T", "A"), ("C", "G"), ("G", "C"), ("a", "t"), ("c", "g")],
    )
    def test_ambiguous_pairs(self, a1: str, a2: str) -> None:
        assert is_strand_ambiguous(a1, a2)

    @pytest.mark.parametrize(
        "a1,a2",
        [("A", "G"), ("A", "C"), ("C", "T"), ("G", "T"), ("A", "A"), ("A", "0")],
    )
    def test_non_ambiguous_pairs(self, a1: str, a2: str) -> None:
        assert not is_strand_ambiguous(a1, a2)

    def test_strand_ambiguous_variant_ids_reads_bim(self, tmp_path: Path) -> None:
        bim = tmp_path / "p.bim"
        bim.write_text(
            "1\tk1\t0\t1\tA\tC\n"   # ok
            "1\tk2\t0\t2\tT\tA\n"   # ambiguous
            "1\tk3\t0\t3\tC\tG\n"   # ambiguous
            "1\tk4\t0\t4\tA\tG\n"   # ok
        )
        assert strand_ambiguous_variant_ids(bim) == ["k2", "k3"]


class TestStrandAmbiguousExclusion:
    """D11: strand-ambiguous (A/T, C/G) panel SNPs are excluded from the
    projection alignment by default; --alt1-allele cannot strand-harmonize
    them, so an opposite-strand target would be silently inverted."""

    @pytest.fixture(autouse=True)
    def _ensure_target_triplet(self, tmp_path: Path) -> None:
        _write_bed_triplet(tmp_path)

    def _write_mixed_panel_bim(self, tmp_path: Path) -> Path:
        # rs1 A/C ok, rs2 A/T ambiguous, rs3 A/G ok, rs4 C/G ambiguous
        path = tmp_path / "panel.bim"
        path.write_text(
            "1\trs1\t0\t1\tA\tC\n"
            "1\trs2\t0\t2\tA\tT\n"
            "1\trs3\t0\t3\tA\tG\n"
            "1\trs4\t0\t4\tC\tG\n"
        )
        return path

    def test_excludes_ambiguous_by_default(self, tmp_path: Path) -> None:
        self._write_mixed_panel_bim(tmp_path)
        runner = _MockRunner(emit_bed=True)
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        args = runner.calls[0]["args"]
        assert "--exclude" in args
        exclude_path = Path(args[args.index("--exclude") + 1])
        assert exclude_path.exists()
        assert sorted(exclude_path.read_text().split()) == ["rs2", "rs4"]
        # --extract panel.bim is still present; --exclude is layered on top.
        assert "--extract" in args
        assert "--alt1-allele" in args

    def test_keeps_ambiguous_when_disabled(self, tmp_path: Path) -> None:
        self._write_mixed_panel_bim(tmp_path)
        runner = _MockRunner(emit_bed=True)
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
            exclude_strand_ambiguous=False,
        )
        assert "--exclude" not in runner.calls[0]["args"]

    def test_no_exclude_when_panel_has_no_ambiguous(self, tmp_path: Path) -> None:
        _write_panel_bim(tmp_path)  # A/C + A/G only
        runner = _MockRunner(emit_bed=True)
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        assert "--exclude" not in runner.calls[0]["args"]


class TestReindexDosageToPanelOrder:
    """The aligned dosage (target∩panel, in TARGET order) must be reindexed
    to the full panel.bim order (= cached P's row order), NaN-filling panel
    SNPs the target lacks — otherwise project_target's length check fails, or
    (at coincidentally-equal length) the dosage is mis-aligned row-for-row
    against P and produces a silently-wrong Q vector."""

    def _write_bim(self, path: Path, ids: list[str]) -> None:
        # PLINK .bim: chr id cm bp a1 a2 — only the ID column matters here.
        path.write_text(
            "\n".join(f"1\t{vid}\t0\t{n + 1}\tA\tC" for n, vid in enumerate(ids))
            + "\n",
        )

    def test_fills_missing_with_nan_and_reorders_by_id(self, tmp_path: Path) -> None:
        # Panel order rs1..rs5; the target carries only rs3, rs1, rs4 — and in
        # a DIFFERENT order than the panel, to prove ID-keyed placement.
        self._write_bim(tmp_path / "panel.bim", ["rs1", "rs2", "rs3", "rs4", "rs5"])
        aligned_bed = tmp_path / "target_aligned.bed"
        aligned_bed.touch()
        self._write_bim(tmp_path / "target_aligned.bim", ["rs3", "rs1", "rs4"])
        dosage = np.array([2.0, 0.0, 1.0])  # rs3=2, rs1=0, rs4=1 (target order)

        full = reindex_dosage_to_panel_order(
            dosage=dosage, aligned_bed=aligned_bed,
            panel_bim=tmp_path / "panel.bim",
        )

        assert full.shape == (5,)
        assert full[0] == 0.0          # rs1
        assert np.isnan(full[1])       # rs2 absent from target → NaN
        assert full[2] == 2.0          # rs3
        assert full[3] == 1.0          # rs4
        assert np.isnan(full[4])       # rs5 absent from target → NaN

    def test_full_overlap_preserves_values_in_panel_order(self, tmp_path: Path) -> None:
        # Target has every panel SNP (no NaN) but in reverse order; output must
        # be in PANEL order, not target order.
        self._write_bim(tmp_path / "panel.bim", ["rs1", "rs2", "rs3"])
        aligned_bed = tmp_path / "aligned.bed"
        aligned_bed.touch()
        self._write_bim(tmp_path / "aligned.bim", ["rs3", "rs2", "rs1"])
        dosage = np.array([2.0, 1.0, 0.0])  # rs3=2, rs2=1, rs1=0

        full = reindex_dosage_to_panel_order(
            dosage=dosage, aligned_bed=aligned_bed,
            panel_bim=tmp_path / "panel.bim",
        )
        np.testing.assert_array_equal(full, np.array([0.0, 1.0, 2.0]))  # panel order

    def test_dotted_aligned_prefix_resolves_bim(self, tmp_path: Path) -> None:
        # Regression (repo's append-vs-with_suffix history): a dotted aligned
        # stem 'x.v2.bed' must resolve its bim as 'x.v2.bim', not 'x.bim'.
        self._write_bim(tmp_path / "panel.bim", ["rs1", "rs2"])
        aligned_bed = tmp_path / "x.v2.bed"
        aligned_bed.touch()
        self._write_bim(tmp_path / "x.v2.bim", ["rs2"])
        full = reindex_dosage_to_panel_order(
            dosage=np.array([1.0]), aligned_bed=aligned_bed,
            panel_bim=tmp_path / "panel.bim",
        )
        assert np.isnan(full[0]) and full[1] == 1.0

    def test_dosage_bim_length_mismatch_raises(self, tmp_path: Path) -> None:
        self._write_bim(tmp_path / "panel.bim", ["rs1"])
        aligned_bed = tmp_path / "aligned.bed"
        aligned_bed.touch()
        self._write_bim(tmp_path / "aligned.bim", ["rs1", "rs2"])  # 2 ids
        with pytest.raises(PanelCacheError, match="out of sync"):
            reindex_dosage_to_panel_order(
                dosage=np.array([1.0]),  # only 1 dosage
                aligned_bed=aligned_bed, panel_bim=tmp_path / "panel.bim",
            )
