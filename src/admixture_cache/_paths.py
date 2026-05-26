"""Path-manipulation helpers shared across the library.

These exist because :meth:`pathlib.Path.with_suffix` REPLACES an
existing extension rather than appending — so a caller-supplied
stem like ``cohort.v2`` gets its trailing ``.v2`` silently stripped
when we try to probe sibling files like ``cohort.v2.pgen``. The
result is misleading "not found" errors or, worse, silently reading
an unrelated file with the same probed name.

The fix is a one-liner — concatenate the suffix to the name as a
raw string — but it needs to be applied at every callsite that
probes / constructs sibling paths around a user-supplied stem.
"""

from __future__ import annotations

from pathlib import Path

# PLINK 1 and PLINK 2 binary genotype triplets. Module-level
# constants so the same definition is shared between
# `_detect_target_format` and any future caller that wants to
# validate a complete triplet.
BED_SIBLINGS = (".bed", ".bim", ".fam")
PGEN_SIBLINGS = (".pgen", ".psam", ".pvar")


def append_suffix(stem: Path, suffix: str) -> Path:
    """Append ``suffix`` to ``stem``'s last path component.

    Use this instead of :meth:`pathlib.Path.with_suffix` whenever
    the stem may contain dots that aren't plink-recognized
    extensions. ``Path.with_suffix`` REPLACES the existing extension
    (e.g., ``Path("cohort.v2").with_suffix(".pgen") == Path("cohort.pgen")``,
    NOT ``Path("cohort.v2.pgen")``); this helper APPENDS.

    >>> append_suffix(Path("/data/cohort.v2"), ".pgen")
    PosixPath('/data/cohort.v2.pgen')
    >>> append_suffix(Path("target"), ".bed")
    PosixPath('target.bed')
    """
    return stem.parent / (stem.name + suffix)


__all__ = ["BED_SIBLINGS", "PGEN_SIBLINGS", "append_suffix"]
