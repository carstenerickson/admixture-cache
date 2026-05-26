"""Target-to-panel alignment and dosage extraction (plink2-backed).

Per-target work that runs every projection: filter the target BED to
the cached panel.bim variant set, flip REF/ALT axes to match the
panel via ``plink2 --alt1-allele``, and extract genotype dosage as a
NumPy 1D array.

REF/ALT axis mismatch silently produces wrong Q vectors (the binomial
likelihood inverts every affected SNP's allele count), so axis
alignment is mandatory — never let the caller skip it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from admixture_cache.errors import PanelCacheError

if TYPE_CHECKING:
    from admixture_cache.runner import ToolRunner

logger = logging.getLogger(__name__)


def align_target_to_panel_bim(
    *, target_bed: Path, panel_bim: Path,
    output_prefix: Path, plink2_runner: ToolRunner,
    log_dir: Path,
    timeout_seconds: int = 600,
) -> Path:
    """Filter target.bed to cached panel.bim variant set + align REF/ALT
    axes via plink2 --alt1-allele.

    REF/ALT axis mismatch between target and reference panel silently
    produces wrong Q vectors (the binomial likelihood inverts every
    affected SNP's allele count). --alt1-allele forces the target's
    ALT1 column to match the panel's ALT1 column at every overlapping
    variant, flipping dosages where needed.

    Returns the path to the aligned target .bed file.
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    plink2_runner.run(
        args=[
            "--bfile", str(target_bed.with_suffix("")),
            "--extract", str(panel_bim),
            # --alt1-allele <bim_file> <alt-col> <id-col>
            # bim_file columns are 1-based: 2=ID, 5=ALT, 6=REF
            "--alt1-allele", str(panel_bim), "5", "2",
            "--make-bed",
            "--out", str(output_prefix),
        ],
        cwd=output_prefix.parent,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
    )

    aligned_bed = output_prefix.with_suffix(".bed")
    if not aligned_bed.exists():
        raise PanelCacheError(
            f"align_target_to_panel_bim: plink2 succeeded but "
            f"{aligned_bed} not produced",
        )
    return aligned_bed


def extract_target_dosage_via_plink2(
    *, target_bed: Path, output_prefix: Path,
    plink2_runner: ToolRunner, log_dir: Path,
    timeout_seconds: int = 600,
) -> np.ndarray:
    """Extract target dosage via ``plink2 --recode A`` (text format),
    then parse to a NumPy 1D array of len M (M = SNPs in target.bim,
    NaN for missing).

    For a single target, this is acceptable (~28 sec on 850K SNPs).
    A future optimization is to replace with ``bed-reader`` library
    for direct binary BED reading (~30× faster).

    Returns dosage as float64 1D array.
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    plink2_runner.run(
        args=[
            "--bfile", str(target_bed.with_suffix("")),
            "--recode", "A",
            "--out", str(output_prefix),
        ],
        cwd=output_prefix.parent,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
    )

    raw_path = output_prefix.with_suffix(".raw")
    if not raw_path.exists():
        raise PanelCacheError(
            f"extract_target_dosage_via_plink2: {raw_path} not produced",
        )

    import pandas as pd

    raw = pd.read_csv(raw_path, sep="\t", na_values=["NA"])
    if raw.shape[0] != 1:
        raise PanelCacheError(
            f"extract_target_dosage_via_plink2: expected 1 sample in "
            f"{raw_path}, got {raw.shape[0]}",
        )
    # First 6 columns are FID IID PAT MAT SEX PHENOTYPE; rest are dosages.
    # np.asarray() guarantees a typed ndarray result even when pandas
    # types resolve to Any (strict mypy in CI doesn't ship pandas-stubs).
    return np.asarray(raw.iloc[0, 6:].to_numpy()).astype(np.float64)


__all__ = ["align_target_to_panel_bim", "extract_target_dosage_via_plink2"]
