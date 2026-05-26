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
    ) -> object:
        self.calls.append({
            "args": list(args),
            "cwd": cwd,
            "log_dir": log_dir,
            "timeout_seconds": timeout_seconds,
        })
        # Find --out and emit stub files for downstream existence checks.
        if "--out" in args:
            out_prefix = Path(args[args.index("--out") + 1])
            if self.emit_bed:
                out_prefix.with_suffix(".bed").touch()
            if self.emit_raw and self.raw_content is not None:
                out_prefix.with_suffix(".raw").write_text(self.raw_content)
        return None


class TestAlignTargetToPanelBim:
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
        """If plink2 'succeeded' (runner returned) but no .bed appeared,
        we raise a clear PanelCacheError."""
        runner = _MockRunner(emit_bed=False)
        with pytest.raises(PanelCacheError, match="not produced"):
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

    def test_magicmock_runner_works(self, tmp_path: Path) -> None:
        runner = MagicMock()

        def fake_run(**kwargs: Any) -> None:
            out_idx = kwargs["args"].index("--out")
            Path(kwargs["args"][out_idx + 1]).with_suffix(".bed").touch()

        runner.run.side_effect = fake_run
        align_target_to_panel_bim(
            target_bed=tmp_path / "target.bed",
            panel_bim=tmp_path / "panel.bim",
            output_prefix=tmp_path / "aligned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        runner.run.assert_called_once()
