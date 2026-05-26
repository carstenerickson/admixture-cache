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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from admixture_cache.alignment import (
    align_target_to_panel_bim,
    extract_target_dosage_via_plink2,
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
) -> ProjectionResult:
    """End-to-end per-target projection:
    1. Validate cache exists + load manifest
    2. Align target.bed to cached panel.bim (variant set + REF/ALT axes)
    3. Extract aligned target dosage as NumPy array
    4. Load cached P matrix
    5. Run NumPy SLSQP projection
    6. Return ProjectionResult with Q vector + metadata

    Total wallclock: ~2 sec end-to-end on a typical 850K-SNP panel
    (excluding the 28-sec pandas .raw load that currently dominates;
    can be reduced further with bed-reader).
    """
    work_dir.mkdir(parents=True, exist_ok=True)

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
    aligned_prefix = work_dir / "target_aligned"
    aligned_bed = align_target_to_panel_bim(
        target_bed=target_bed,
        panel_bim=panel_bim,
        output_prefix=aligned_prefix,
        plink2_runner=plink2_runner,
        log_dir=work_dir / "logs",
    )

    # Step 3: extract dosage as NumPy array
    dosage_prefix = work_dir / "target_dosage"
    dosage = extract_target_dosage_via_plink2(
        target_bed=aligned_bed,
        output_prefix=dosage_prefix,
        plink2_runner=plink2_runner,
        log_dir=work_dir / "logs",
    )

    # Step 4: load cached P
    P = load_cached_p(cache_dir, manifest.k)
    if P.shape[0] != dosage.shape[0]:
        raise PanelCacheError(
            f"project_target: cached P has {P.shape[0]} SNPs but "
            f"aligned target dosage has {dosage.shape[0]} — alignment "
            f"step may have failed silently",
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
