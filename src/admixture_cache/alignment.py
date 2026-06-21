"""Target-to-panel alignment and dosage extraction (plink2-backed).

Per-target work that runs every projection: filter the target genotypes
to the cached panel.bim variant set, flip REF/ALT axes to match the
panel via ``plink2 --alt1-allele``, and extract genotype dosage as a
NumPy 1D array.

REF/ALT axis mismatch silently produces wrong Q vectors (the binomial
likelihood inverts every affected SNP's allele count), so axis
alignment is mandatory — never let the caller skip it.

Target format
-------------

Accepts both PLINK 1 BED (``.bed`` + ``.bim`` + ``.fam``) and PLINK 2
PGEN (``.pgen`` + ``.psam`` + ``.pvar``) inputs; plink2 handles both
natively, so the alignment output is always a BED triplet regardless
of input format. Pass the path to whichever genotype file you have
(``target.bed`` or ``target.pgen``); the helper detects the format
from the extension or sibling-file presence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from admixture_cache._dispatch import _call_runner
from admixture_cache._paths import BED_SIBLINGS, PGEN_SIBLINGS, append_suffix
from admixture_cache.errors import PanelCacheError

if TYPE_CHECKING:
    from admixture_cache.runner import ToolRunner

logger = logging.getLogger(__name__)


# Local aliases preserved for grep-friendliness; canonical source
# of truth is admixture_cache._paths.
_BED_SIBLINGS = BED_SIBLINGS
_PGEN_SIBLINGS = PGEN_SIBLINGS
_append_suffix = append_suffix


# Strand-ambiguous SNP detection. A SNP whose two alleles are
# complementary (A/T or C/G) has a base-pair representation that is
# invariant under a strand flip, so REF/ALT harmonization by variant ID
# cannot tell whether the target is on the same strand as the panel:
# ``plink2 --alt1-allele`` "succeeds" by matching the allele LETTER while
# silently leaving an opposite-strand target's dosage inverted
# (homozygotes flip 0<->2; heterozygotes are unaffected). Non-ambiguous
# SNPs are safe because their flipped letters do not match, so the
# forcing is skipped. The only robust fix is to drop the ambiguous SNPs;
# see SCIENCE.md D11.
_STRAND_AMBIGUOUS_PAIRS = frozenset({
    frozenset({"A", "T"}),
    frozenset({"C", "G"}),
})


def is_strand_ambiguous(allele1: str, allele2: str) -> bool:
    """Return True if a SNP's two alleles are strand-complementary
    (A/T or C/G), so the allele set is invariant under a strand flip and
    REF/ALT harmonization by ID cannot detect an opposite-strand target.
    Case-insensitive; non-SNP / multi-character alleles return False."""
    return (
        frozenset({allele1.upper(), allele2.upper()})
        in _STRAND_AMBIGUOUS_PAIRS
    )


def strand_ambiguous_variant_ids(bim_path: Path) -> list[str]:
    """Return the variant IDs (column 2) of every strand-ambiguous
    (A/T, C/G) SNP in a PLINK ``.bim`` (alleles in columns 5 and 6),
    in file order."""
    ids: list[str] = []
    with bim_path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 6 and is_strand_ambiguous(parts[4], parts[5]):
                ids.append(parts[1])
    return ids


def _detect_target_format(target_path: Path) -> tuple[str, Path]:
    """Determine the plink2 input flag and stem path from a target path.

    Returns ``("--bfile" | "--pfile", stem)`` where ``stem`` is the
    path without the format suffix. plink2 expands the stem to
    ``stem.bed/.bim/.fam`` for ``--bfile`` and
    ``stem.pgen/.psam/.pvar`` for ``--pfile``.

    Detection precedence:

    1. Explicit ``.bed`` or ``.pgen`` suffix on the input path; all
       three sibling files must be present.
    2. No plink suffix → probe the filesystem; PGEN preferred if both
       complete triplets exist.

    Raises :class:`PanelCacheError` if neither format is found OR the
    explicit format is missing a sibling file (e.g. ``.bed`` present
    but ``.bim`` missing).
    """
    # Explicit-suffix branches: caller named the format explicitly.
    # Require all three sibling files; raise a clear error otherwise so
    # incomplete triplets surface here rather than as opaque plink2
    # downstream messages.
    if target_path.suffix == ".bed":
        stem = target_path.with_suffix("")
        missing = [
            s for s in _BED_SIBLINGS
            if not _append_suffix(stem, s).exists()
        ]
        if missing:
            raise PanelCacheError(
                f"_detect_target_format: BED triplet for {target_path} "
                f"is incomplete; missing sibling file(s): "
                f"{', '.join(missing)}",
            )
        return "--bfile", stem
    if target_path.suffix == ".pgen":
        stem = target_path.with_suffix("")
        missing = [
            s for s in _PGEN_SIBLINGS
            if not _append_suffix(stem, s).exists()
        ]
        if missing:
            raise PanelCacheError(
                f"_detect_target_format: PGEN triplet for {target_path} "
                f"is incomplete; missing sibling file(s): "
                f"{', '.join(missing)}",
            )
        return "--pfile", stem

    # No plink suffix — probe the filesystem. Use APPEND semantics
    # (`stem.name + ".pgen"`) rather than `with_suffix` so that a stem
    # like `cohort.v2` doesn't get its trailing `.v2` replaced
    # (which would silently probe the wrong path).
    stem = target_path
    if _append_suffix(stem, ".pgen").exists():
        return "--pfile", stem
    if _append_suffix(stem, ".bed").exists():
        return "--bfile", stem
    raise PanelCacheError(
        f"_detect_target_format: target {target_path} not found as "
        f"either a PLINK 1 BED triplet (.bed/.bim/.fam) or a PLINK 2 "
        f"PGEN triplet (.pgen/.psam/.pvar)",
    )


def align_target_to_panel_bim(
    *, target_bed: Path, panel_bim: Path,
    output_prefix: Path, plink2_runner: ToolRunner,
    log_dir: Path,
    timeout_seconds: int = 600,
    exclude_strand_ambiguous: bool = True,
) -> Path:
    """Filter target genotypes to cached panel.bim variant set + align
    REF/ALT axes via plink2 --alt1-allele.

    REF/ALT axis mismatch between target and reference panel silently
    produces wrong Q vectors (the binomial likelihood inverts every
    affected SNP's allele count). --alt1-allele forces the target's
    ALT1 column to match the panel's ALT1 column at every overlapping
    variant, flipping dosages where needed.

    ``--alt1-allele`` matches by allele LETTER, so it cannot fix a
    strand-ambiguous (A/T, C/G) SNP whose target is on the opposite
    strand: the allele set is identical under complement, the forcing
    "succeeds", and the dosage is silently inverted (see SCIENCE.md
    D11). When ``exclude_strand_ambiguous`` is True (the default), such
    panel SNPs are dropped from this projection via ``plink2 --exclude``
    and a warning is logged. Caches built with build_panel_cache's
    default guard contain no ambiguous SNPs, so this is a no-op for
    them; it protects legacy caches that still contain them. Pass
    False to keep them (only safe when target and panel are guaranteed
    to share a strand convention, e.g. both reference-forward).

    The ``target_bed`` parameter accepts either a BED path (``.bed``
    + ``.bim`` + ``.fam`` triplet) or a PGEN path (``.pgen`` + ``.psam``
    + ``.pvar`` triplet). plink2 handles both via ``--bfile`` /
    ``--pfile``. The kwarg name is BED-specific for backward
    compatibility; PGEN support added in v1.1.

    Returns the path to the aligned target .bed file (output is always
    BED regardless of input format).
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # plink2 reads panel_bim for both --extract and --alt1-allele, and the
    # strand-ambiguous scan below reads it directly; surface a missing
    # reference as a PanelCacheError (the package's error contract) rather
    # than a bare FileNotFoundError from the scan.
    if not panel_bim.exists():
        raise PanelCacheError(
            f"align_target_to_panel_bim: panel .bim missing at {panel_bim}",
        )

    input_flag, input_stem = _detect_target_format(target_bed)

    # Strand-ambiguous (A/T, C/G) panel SNPs cannot be safely
    # REF/ALT-harmonized by --alt1-allele (their allele set is invariant
    # under strand flip, so an opposite-strand target is silently
    # inverted). Exclude them from this projection by default; see
    # SCIENCE.md D11. New caches built with build_panel_cache's default
    # guard contain none, so this is a no-op for them.
    exclude_args: list[str] = []
    if exclude_strand_ambiguous:
        ambiguous_ids = strand_ambiguous_variant_ids(panel_bim)
        if ambiguous_ids:
            exclude_path = append_suffix(
                output_prefix, ".strand_ambiguous_exclude.txt",
            )
            exclude_path.write_text("\n".join(ambiguous_ids) + "\n")
            exclude_args = ["--exclude", str(exclude_path)]
            logger.warning(
                "align_target_to_panel_bim: excluding %d strand-ambiguous "
                "(A/T, C/G) panel SNP(s) from this projection to avoid "
                "silent strand inversion; pass exclude_strand_ambiguous="
                "False to keep them (see SCIENCE.md D11).",
                len(ambiguous_ids),
            )

    # Route through _call_runner so log_name (and pid_callback if a
    # future feature needs it) are forwarded to runners that support
    # them. Distinct per-call log name keeps logs collision-free if
    # multiple project_target calls share a log_dir.
    _call_runner(
        plink2_runner,
        args=[
            input_flag, str(input_stem),
            "--extract", str(panel_bim),
            *exclude_args,
            # --alt1-allele <bim_file> <alt-col> <id-col>
            # bim_file columns are 1-based: 2=ID, 5=ALT, 6=REF
            "--alt1-allele", str(panel_bim), "5", "2",
            "--make-bed",
            "--out", str(output_prefix),
        ],
        cwd=output_prefix.parent,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
        log_name=f"align_{output_prefix.name}.out",
    )

    # Validate that plink2 produced the FULL aligned BED triplet, not
    # just the `.bed`. A truncated `.bim`/`.fam` (disk-full, killed
    # subprocess, FS error) would otherwise surface downstream as a
    # confusing "BED triplet incomplete" inside
    # `extract_target_dosage_via_plink2`, which would mis-attribute
    # the failure to dosage extraction rather than alignment.
    aligned_bed = append_suffix(output_prefix, ".bed")
    missing = [
        append_suffix(output_prefix, s)
        for s in BED_SIBLINGS
        if not append_suffix(output_prefix, s).exists()
    ]
    if missing:
        raise PanelCacheError(
            f"align_target_to_panel_bim: plink2 succeeded but produced "
            f"an incomplete BED triplet at {output_prefix}; missing "
            f"sibling file(s): {[p.name for p in missing]}",
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

    The ``target_bed`` parameter accepts either a BED path or a PGEN
    path (same as :func:`align_target_to_panel_bim`); plink2 handles
    both natively via ``--bfile`` / ``--pfile``. The kwarg name is
    BED-specific for backward compatibility; PGEN acceptance added in
    v1.1.1.

    For a single target, this is acceptable (~28 sec on 850K SNPs).
    A future optimization is to replace with ``bed-reader`` library
    for direct binary BED reading (~30× faster).

    Returns dosage as float64 1D array.
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    input_flag, input_stem = _detect_target_format(target_bed)

    _call_runner(
        plink2_runner,
        args=[
            input_flag, str(input_stem),
            "--recode", "A",
            "--out", str(output_prefix),
        ],
        cwd=output_prefix.parent,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
        log_name=f"dosage_{output_prefix.name}.out",
    )

    raw_path = append_suffix(output_prefix, ".raw")
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


def _read_bim_variant_ids(bim_path: Path) -> list[str]:
    """Return the variant IDs (column 2) of a PLINK ``.bim``, in file order.

    IDs are assumed unique — plink2 ``--extract`` (the upstream step that
    produces the aligned target) already requires a unique-ID variant set,
    and :func:`reindex_dosage_to_panel_order` builds an ID→index map that
    would collapse duplicates. Panels exported with placeholder IDs (e.g.
    ``.`` for unnamed variants) violate this and must be re-ID'd before
    caching.
    """
    ids: list[str] = []
    with bim_path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                ids.append(parts[1])
    return ids


def reindex_dosage_to_panel_order(
    *, dosage: np.ndarray, aligned_bed: Path, panel_bim: Path,
) -> np.ndarray:
    """Reindex a target dosage vector to the cached panel's variant order.

    :func:`align_target_to_panel_bim` runs ``plink2 --extract panel.bim``,
    which (a) keeps only the target∩panel variants — so the dosage is SHORTER
    than the panel when the target is missing any panel SNP — and (b) preserves
    the *target's* variant order, not the panel's. The cached ``P`` matrix is
    in ``panel.bim`` order, so :func:`project_target` must reindex the dosage to
    that order before projecting; otherwise the length check fails, or — worse,
    when lengths coincidentally match — the dosage is silently mis-aligned
    row-for-row against ``P`` and every offset SNP's allele count is projected
    against the wrong cluster frequencies.

    Builds a length-``len(panel)`` vector, ``NaN`` everywhere the target lacks
    a panel variant (the SLSQP projection treats ``NaN`` as missing), and places
    each target dosage at its panel-order index by matching variant ID.

    Returns the reindexed float64 vector (``len == panel.bim`` variant count).
    """
    # `with_suffix` replaces only the final extension, so it handles
    # dotted stems correctly: target_aligned.bed → .bim, and the
    # repo's historical edge case x.v2.bed → x.v2.bim (NOT x.bim).
    aligned_bim = aligned_bed.with_suffix(".bim")
    target_ids = _read_bim_variant_ids(aligned_bim)
    if len(target_ids) != int(dosage.shape[0]):
        raise PanelCacheError(
            f"reindex_dosage_to_panel_order: aligned .bim {aligned_bim} has "
            f"{len(target_ids)} variants but the dosage vector has "
            f"{int(dosage.shape[0])} — dosage/bim are out of sync",
        )
    panel_ids = _read_bim_variant_ids(panel_bim)
    panel_index = {vid: i for i, vid in enumerate(panel_ids)}
    full = np.full(len(panel_ids), np.nan, dtype=np.float64)
    n_placed = 0
    # strict=True: target_ids/dosage equal length is guaranteed by the
    # check above, so this never raises — but it locks the invariant in
    # case that guard is ever refactored away.
    for vid, value in zip(target_ids, dosage, strict=True):
        j = panel_index.get(vid)
        if j is not None:
            full[j] = value
            n_placed += 1
    logger.info(
        "reindex_dosage_to_panel_order: placed %d/%d target variants into the "
        "%d-variant panel order (%d panel variants missing from target → NaN)",
        n_placed, len(target_ids), len(panel_ids), len(panel_ids) - n_placed,
    )
    return full


__all__ = [
    "align_target_to_panel_bim",
    "extract_target_dosage_via_plink2",
    "is_strand_ambiguous",
    "reindex_dosage_to_panel_order",
    "strand_ambiguous_variant_ids",
]
