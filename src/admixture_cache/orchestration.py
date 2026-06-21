"""End-to-end per-target projection.

Wraps the steps every projection runs through: load + verify manifest,
align target to cached panel (variant set + REF/ALT axes), extract
target dosage, load cached P, run the NumPy SLSQP solver, return a
:class:`ProjectionResult` with the Q vector + provenance metadata
from the manifest.

Total wallclock ~2 sec end-to-end on a typical 850K-SNP panel.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from admixture_cache.alignment import (
    align_target_to_panel_bim,
    extract_target_dosage_via_plink2,
    reindex_dosage_to_panel_order,
)
from admixture_cache.errors import PanelCacheError
from admixture_cache.io import load_cache_manifest, load_cached_p
from admixture_cache.projection import (
    ProjectionResult,
    numpy_supervised_projection,
)

if TYPE_CHECKING:
    from admixture_cache.runner import ToolRunner

logger = logging.getLogger(__name__)


def project_target(
    *, target_bed: Path,
    cache_dir: Path,
    plink2_runner: ToolRunner,
    work_dir: Path,
    exclude_strand_ambiguous: bool = True,
) -> ProjectionResult:
    """End-to-end per-target projection:
    1. Validate cache exists + load manifest
    2. Align target.bed to cached panel.bim (variant set + REF/ALT axes)
    3. Extract aligned target dosage as NumPy array
    4. Load cached P matrix
    5. Run NumPy SLSQP projection
    6. Return ProjectionResult with Q vector + metadata

    ``exclude_strand_ambiguous`` (default True) drops strand-ambiguous
    (A/T, C/G) panel SNPs from the alignment, which cannot be safely
    REF/ALT-harmonized and are silently strand-inverted for an
    opposite-strand target (see SCIENCE.md D11). It is a no-op against a
    cache built with build_panel_cache's default guard (which contains
    none) and protects legacy caches that still contain them. Pass False
    only when target and panel are guaranteed to share a strand
    convention.

    Total wallclock: ~2 sec end-to-end on a typical 850K-SNP panel
    (excluding the 28-sec pandas .raw load that currently dominates;
    can be reduced further with bed-reader).

    The ``work_dir`` may be reused across calls — this function creates
    a unique per-call subdirectory (``work_dir/<target-stem>-<uuid>/``)
    so concurrent or back-to-back invocations don't collide on
    intermediate files or log names.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # Per-call subdir keyed by target stem + short uuid. Without
    # this, two project_target calls sharing a work_dir would
    # overwrite each other's intermediate files (target_aligned.bed,
    # target_dosage.raw) AND their log files (align_target_aligned.out,
    # dosage_target_dosage.out), and SubprocessToolRunner's
    # one-generation .prev rotation would drop the older debug
    # history silently. The subdir guarantees per-call isolation.
    call_id = uuid.uuid4().hex[:8]
    call_dir = work_dir / f"{target_bed.stem}-{call_id}"
    call_dir.mkdir(parents=True, exist_ok=False)

    t0 = time.time()
    manifest = load_cache_manifest(cache_dir)
    logger.info(
        "project_target: loaded cache manifest (track=%s panel=%s K=%d, "
        "built %s, wallclock %.0fs)",
        manifest.track, manifest.panel_id, manifest.k,
        # Explicit .isoformat() keeps the ISO-8601 `T` separator
        # stable; bare %s on a datetime renders with a space and
        # would silently break log scrapers grepping the prior format.
        manifest.build_timestamp.isoformat(),
        manifest.build_wallclock_seconds,
    )

    panel_bim = cache_dir / "panel.bim"
    if not panel_bim.exists():
        raise PanelCacheError(
            f"project_target: cache missing panel.bim at {panel_bim}",
        )

    # Step 2: align target to panel variant set + axes
    aligned_prefix = call_dir / "target_aligned"
    aligned_bed = align_target_to_panel_bim(
        target_bed=target_bed,
        panel_bim=panel_bim,
        output_prefix=aligned_prefix,
        plink2_runner=plink2_runner,
        log_dir=call_dir / "logs",
        exclude_strand_ambiguous=exclude_strand_ambiguous,
    )

    # Step 3: extract dosage as NumPy array
    dosage_prefix = call_dir / "target_dosage"
    dosage = extract_target_dosage_via_plink2(
        target_bed=aligned_bed,
        output_prefix=dosage_prefix,
        plink2_runner=plink2_runner,
        log_dir=call_dir / "logs",
    )

    # Step 3b: reindex the dosage to the FULL panel.bim variant order (= cached
    # P's row order), NaN-filling panel SNPs the target lacks. plink2 --extract
    # in align_target_to_panel_bim yields only target∩panel variants in the
    # TARGET's order, so the raw dosage is both shorter than P (whenever the
    # target misses any panel SNP — the common case) and, even at equal length,
    # mis-aligned row-for-row against P. The SLSQP projection treats NaN as
    # missing (see n_obs below).
    dosage = reindex_dosage_to_panel_order(
        dosage=dosage, aligned_bed=aligned_bed, panel_bim=panel_bim,
    )

    # Step 4: load cached P
    P = load_cached_p(cache_dir, manifest.k)
    if P.shape[0] != dosage.shape[0]:
        raise PanelCacheError(
            f"project_target: cached P has {P.shape[0]} SNPs but the "
            f"panel-reindexed target dosage has {dosage.shape[0]} — the cache "
            f"is internally inconsistent (P row count != panel.bim variant "
            f"count); rebuild the cache",
        )

    # Step 5: NumPy projection
    n_obs = int((~np.isnan(dosage)).sum())
    logger.info(
        "project_target: projecting target on %d non-missing SNPs (of %d in P)",
        n_obs, P.shape[0],
    )
    q, n_iter, converged = numpy_supervised_projection(
        target_dosage=dosage, p_matrix=P, k=manifest.k,
    )
    logger.info(
        "project_target: SLSQP %s in %d iters; Q = %s; total wallclock %.1fs",
        "converged" if converged else "DID NOT CONVERGE", n_iter,
        np.round(q, 6).tolist(), time.time() - t0,
    )

    return ProjectionResult(
        target_q=q,
        cluster_order=manifest.cluster_order,
        panel_stability_max_sd=manifest.restart_sd_max,
        n_snps_used=n_obs,
        optimization_iterations=n_iter,
        converged=converged,
    )


__all__ = ["project_target"]
