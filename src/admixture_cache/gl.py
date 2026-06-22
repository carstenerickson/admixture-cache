"""Genotype-likelihood (GL) input for low-coverage / ancient-DNA targets.

The hard-call projection path (alignment.py + projection.numpy_supervised_projection)
collapses each site to a single 0/1/2 dosage. For genuinely low-coverage data that
throws away per-site uncertainty: a site covered by one error-prone read is treated
as confidently as a site covered by thirty. The genotype-likelihood path keeps that
uncertainty by carrying, per site, the three genotype likelihoods GL(g) = P(reads |
genotype g) and marginalizing over the unknown genotype under a Hardy-Weinberg prior
(the NGSadmix / fastNGSadmix model; Skotte, Korneliussen & Albrechtsen 2013,
doi:10.1534/genetics.113.154138; Bansal & Libiger 2015, doi:10.1186/s12859-014-0418-7).
fastNGSadmix is precisely this fixed-reference projection setting.

Input format: an ANGSD-style **beagle GL** text file (tab-separated) for a single
target individual::

    marker      allele1 allele2 Ind0    Ind0    Ind0
    rs0001      A       G       0.94    0.05    0.01
    ...

``allele1`` is the major allele, ``allele2`` the minor; the three per-individual
columns are the genotype likelihoods of (allele1/allele1, allele1/allele2,
allele2/allele2), i.e. indexed by the count of ``allele2``. Alleles may be given as
letters (A/C/G/T) or ANGSD numeric codes (0=A, 1=C, 2=G, 3=T). Per-site
normalization is irrelevant: a per-site constant factors out of the product over
sites and cannot move the argmax over Q.

``marker`` must match the cached ``panel.bim`` variant IDs; alignment to the panel
(variant matching, REF/ALT orientation to the panel's allele-1 axis, strand-ambiguous
SNP exclusion) is done here in pure Python, so no plink2 is needed for the GL path.
Mapping / reference bias is NOT corrected (it persists even with GLs; see
SCIENCE.md D17).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from admixture_cache.alignment import is_strand_ambiguous
from admixture_cache.errors import PanelCacheError

logger = logging.getLogger(__name__)

# ANGSD beagle alleles are 0=A,1=C,2=G,3=T; letters are also accepted.
_ALLELE_DECODE = {
    "0": "A", "1": "C", "2": "G", "3": "T",
    "A": "A", "C": "C", "G": "G", "T": "T",
}
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


@dataclass(frozen=True)
class BeagleGL:
    """Parsed beagle genotype-likelihood file for one target individual."""

    marker_ids: list[str]
    allele1: list[str]  # major, decoded to A/C/G/T (or raw if undecodable)
    allele2: list[str]  # minor
    gl: np.ndarray  # shape (M, 3): P(reads | allele1-hom, het, allele2-hom)


def _decode_allele(raw: object) -> str | None:
    """Decode a beagle allele (numeric code or letter) to A/C/G/T, or None
    if it is not a recognized single nucleotide (e.g. an indel)."""
    return _ALLELE_DECODE.get(str(raw).strip().upper())


def read_beagle_gl(path: Path) -> BeagleGL:
    """Read an ANGSD-style beagle GL file for a SINGLE target individual.

    Expects a header line and columns ``marker allele1 allele2`` followed by
    exactly three genotype-likelihood columns (one individual). Raises
    :class:`PanelCacheError` on a missing file, the wrong number of GL columns
    (i.e. not exactly one individual), or non-numeric GL values.
    """
    if not path.exists():
        raise PanelCacheError(f"read_beagle_gl: beagle file not found: {path}")

    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    if df.shape[1] != 6:
        raise PanelCacheError(
            f"read_beagle_gl: expected 6 columns (marker, allele1, allele2, and "
            f"3 genotype-likelihood columns for ONE individual) in {path}, got "
            f"{df.shape[1]}. The GL path projects a single target at a time.",
        )
    if df.shape[0] == 0:
        raise PanelCacheError(f"read_beagle_gl: {path} has no data rows")

    marker_ids = [str(m) for m in df.iloc[:, 0].tolist()]
    allele1 = [str(a) for a in df.iloc[:, 1].tolist()]
    allele2 = [str(a) for a in df.iloc[:, 2].tolist()]
    try:
        gl = np.asarray(df.iloc[:, 3:6].to_numpy()).astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise PanelCacheError(
            f"read_beagle_gl: genotype-likelihood columns in {path} are not "
            f"numeric: {exc}",
        ) from exc
    if not np.all(np.isfinite(gl)):
        raise PanelCacheError(
            f"read_beagle_gl: {path} contains non-finite genotype likelihoods",
        )
    return BeagleGL(
        marker_ids=marker_ids, allele1=allele1, allele2=allele2, gl=gl,
    )


def _read_panel_bim_alleles(panel_bim: Path) -> list[tuple[str, str, str]]:
    """Return ``(variant_id, allele1, allele2)`` per PLINK ``.bim`` line in
    file order. allele1 is column 5 (the panel's allele-1 / ALT axis, the
    allele whose frequency the cached P stores); allele2 is column 6."""
    out: list[tuple[str, str, str]] = []
    with panel_bim.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 6:
                out.append((parts[1], parts[4], parts[5]))
    return out


def _orient_gl_to_panel_a1(
    panel_a1: str, panel_a2: str, b_a1: str, b_a2: str, triple: np.ndarray,
) -> np.ndarray | None:
    """Re-index a beagle GL triple to the count of the panel's allele 1.

    The beagle triple is indexed by the count of ``b_a2`` (the minor allele):
    ``[P(0 copies), P(1), P(2)]``. The projection likelihood needs the triple
    indexed by the count of ``panel_a1`` (the allele whose frequency P stores).
    Matches the panel/beagle allele pair directly or under strand complement;
    returns None when the alleles are incompatible (different SNP, indel, etc.).
    """
    pa1, pa2 = panel_a1.upper(), panel_a2.upper()
    for a1, a2, _flipped in (
        (b_a1, b_a2, False),
        (_COMPLEMENT.get(b_a1), _COMPLEMENT.get(b_a2), True),
    ):
        if a1 is None or a2 is None:
            continue
        if {a1, a2} != {pa1, pa2}:
            continue
        if pa1 == a2:  # panel allele 1 IS the beagle minor -> identity
            return triple
        if pa1 == a1:  # panel allele 1 is the beagle major -> reverse the triple
            return triple[::-1]
        return None
    return None


def align_gl_to_panel(
    *, beagle: BeagleGL, panel_bim: Path, exclude_strand_ambiguous: bool = True,
) -> np.ndarray:
    """Align a target's beagle GLs to the cached panel order + allele-1 axis.

    Returns an ``(M_panel, 3)`` array of GL triples indexed by the count of the
    panel's allele 1, in ``panel.bim`` order. Panel SNPs absent from the beagle
    file, with incompatible alleles, or (when ``exclude_strand_ambiguous``)
    strand-ambiguous (A/T, C/G) are left as a row of NaN, which the GL
    projection masks out as missing. This is the GL analog of
    ``align_target_to_panel_bim`` + ``reindex_dosage_to_panel_order``, done in
    pure Python.
    """
    panel = _read_panel_bim_alleles(panel_bim)
    beagle_index = {m: i for i, m in enumerate(beagle.marker_ids)}
    out = np.full((len(panel), 3), np.nan, dtype=np.float64)

    n_placed = n_missing = n_ambiguous = n_allele_mismatch = 0
    for j, (vid, pa1, pa2) in enumerate(panel):
        if exclude_strand_ambiguous and is_strand_ambiguous(pa1, pa2):
            n_ambiguous += 1
            continue
        i = beagle_index.get(vid)
        if i is None:
            n_missing += 1
            continue
        b_a1 = _decode_allele(beagle.allele1[i])
        b_a2 = _decode_allele(beagle.allele2[i])
        if b_a1 is None or b_a2 is None:
            n_allele_mismatch += 1
            continue
        oriented = _orient_gl_to_panel_a1(pa1, pa2, b_a1, b_a2, beagle.gl[i])
        if oriented is None:
            n_allele_mismatch += 1
            continue
        out[j] = oriented
        n_placed += 1

    logger.info(
        "align_gl_to_panel: placed %d/%d panel SNPs from the beagle file "
        "(%d panel SNPs missing from target, %d strand-ambiguous excluded, "
        "%d allele-incompatible)",
        n_placed, len(panel), n_missing, n_ambiguous, n_allele_mismatch,
    )
    if n_placed == 0:
        raise PanelCacheError(
            "align_gl_to_panel: no panel SNP could be aligned to the beagle "
            "file (check that the beagle 'marker' column uses the same variant "
            "IDs as panel.bim).",
        )
    return out


__all__ = [
    "BeagleGL",
    "align_gl_to_panel",
    "read_beagle_gl",
]
