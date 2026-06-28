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
import math
import time
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from admixture_cache.alignment import (
    align_target_to_panel_bim,
    extract_target_dosage_via_plink2,
    reindex_dosage_to_panel_order,
)
from admixture_cache.errors import PanelCacheError
from admixture_cache.gl import align_gl_to_panel, read_beagle_gl
from admixture_cache.io import load_cache_manifest, load_cached_p
from admixture_cache.projection import (
    ProjectionResult,
    numpy_supervised_projection,
    numpy_supervised_projection_gl,
)

if TYPE_CHECKING:
    from admixture_cache.runner import ToolRunner

logger = logging.getLogger(__name__)

# At/below this observed heterozygosity the target looks pseudo-haploid
# (every site homozygous) OR very low-coverage diploid; the two are not
# cleanly separable by heterozygosity alone (low-coverage diploid is also
# depressed, doi:10.1186/s12864-015-1219-8), so this only WARNS (D17).
_PSEUDOHAPLOID_HET_MAX = 0.005


def _warn_on_low_heterozygosity(het_rate: float, n_obs: int) -> None:
    """Warn when a target's observed heterozygosity is essentially zero,
    which indicates pseudo-haploid / haploidized data or a very low-coverage
    diploid sample (SCIENCE.md D17). Advisory only: it never changes the
    projection. Note the hard-call projection point estimate is the SAME for
    pseudo-haploid and diploid data (the diploid binomial likelihood is a
    constant multiple of the Bernoulli one, so the MLE argmax is identical);
    the caveat is that the diploid model overstates per-site information, and
    for low-coverage data a genotype-likelihood method should inform the
    estimate with per-site uncertainty."""
    if n_obs == 0 or math.isnan(het_rate) or het_rate > _PSEUDOHAPLOID_HET_MAX:
        return
    warnings.warn(
        f"project_target: observed heterozygosity is {het_rate:.4%} (near "
        f"zero). This indicates pseudo-haploid / haploidized data (one sampled "
        f"allele coded as homozygous) or a very low-coverage diploid sample. "
        f"The hard-call projection point estimate is unaffected (pseudo-haploid "
        f"and diploid yield the same Q here), but the diploid model overstates "
        f"per-site information; for low-coverage targets prefer the "
        f"genotype-likelihood path (project_target_gl / CLI --gl-beagle), which "
        f"downweights uncertain sites (SCIENCE.md D17).",
        UserWarning,
        stacklevel=3,
    )


def _resolve_exclude_strand_ambiguous(
    explicit: bool | None, manifest_decision: bool | None,
) -> bool:
    """Resolve the projection-time strand-ambiguous policy (D11).

    ``explicit`` is the caller's override; ``None`` (the default) decides
    from the cache's recorded build state. Projection defaults to the
    *protective* choice (exclude), because whether a target shares the
    panel's strand convention is a property of the (panel, target) pair
    decided here at projection time — not something the builder can assert
    for unknown downstream targets (and this library distributes caches
    across parties). The manifest's ``strand_ambiguous_excluded`` is used
    only to skip needless work, never to weaken the default:

    - ``True``  — the build certified the panel free of A/T,C/G SNPs, so
      there is nothing to exclude. Return False to skip the per-projection
      ``panel.bim`` scan (a no-op that would find nothing anyway).
    - ``False`` — the operator kept ambiguous SNPs at build time, so the
      panel still contains them; exclude them here protectively.
    - ``None``  — legacy cache that may still contain them; exclude
      protectively (scan + ``--exclude``).

    Keeping ambiguous SNPs at projection is a per-call opt-in via an
    explicit ``False`` (CLI ``--keep-strand-ambiguous``); it is
    deliberately NOT inherited from how the cache was built. An explicit
    ``True``/``False`` always overrides.
    """
    if explicit is not None:
        return explicit
    return manifest_decision is not True


def project_target(
    *, target_bed: Path,
    cache_dir: Path,
    plink2_runner: ToolRunner,
    work_dir: Path,
    exclude_strand_ambiguous: bool | None = None,
) -> ProjectionResult:
    """End-to-end per-target projection:
    1. Validate cache exists + load manifest
    2. Align target.bed to cached panel.bim (variant set + REF/ALT axes)
    3. Extract aligned target dosage as NumPy array
    4. Load cached P matrix
    5. Run NumPy SLSQP projection
    6. Return ProjectionResult with Q vector + metadata

    ``exclude_strand_ambiguous`` controls dropping strand-ambiguous
    (A/T, C/G) panel SNPs from the alignment — they cannot be safely
    REF/ALT-harmonized and are silently strand-inverted for an
    opposite-strand target (see SCIENCE.md D11). The default ``None``
    excludes them protectively, using the manifest only to skip needless
    work: a build-certified-clean cache skips the per-projection panel.bim
    scan, while a cache that may still contain them (operator kept them at
    build, or a legacy pre-D11 cache) has them excluded. Keeping them is a
    per-call opt-in via ``False`` (CLI ``--keep-strand-ambiguous``), only
    safe when this target shares the panel's strand convention; it is not
    inherited from the build. Pass ``True`` to force exclusion.

    The returned :class:`ProjectionResult` records the target's observed
    ``heterozygosity``; an essentially-zero rate emits a UserWarning that
    the target looks pseudo-haploid or very low coverage (advisory only, it
    never changes the result). The hard-call projection point estimate is
    the same for pseudo-haploid and diploid input; for low-coverage targets
    a genotype-likelihood method should be preferred (see SCIENCE.md D17).

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

    # Resolve the strand-ambiguous policy from the caller's override and
    # the build's recorded decision (default None -> follow the manifest;
    # see _resolve_exclude_strand_ambiguous). A certified-clean cache
    # resolves to False, skipping the per-projection panel.bim scan.
    effective_exclude = _resolve_exclude_strand_ambiguous(
        exclude_strand_ambiguous, manifest.strand_ambiguous_excluded,
    )

    # Step 2: align target to panel variant set + axes
    aligned_prefix = call_dir / "target_aligned"
    aligned_bed = align_target_to_panel_bim(
        target_bed=target_bed,
        panel_bim=panel_bim,
        output_prefix=aligned_prefix,
        plink2_runner=plink2_runner,
        log_dir=call_dir / "logs",
        exclude_strand_ambiguous=effective_exclude,
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
    obs_mask = ~np.isnan(dosage)
    n_obs = int(obs_mask.sum())
    # Observed heterozygosity (fraction of non-missing genotypes == 1).
    # Near-zero het flags likely pseudo-haploid or very low-coverage input (D17).
    het_rate = float(np.mean(dosage[obs_mask] == 1)) if n_obs else float("nan")
    _warn_on_low_heterozygosity(het_rate, n_obs)
    logger.info(
        "project_target: projecting target on %d non-missing SNPs (of %d in P), "
        "heterozygosity=%.4f",
        n_obs, P.shape[0], het_rate,
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
        heterozygosity=het_rate,
    )


def project_target_gl(
    *, target_gl_beagle: Path,
    cache_dir: Path,
    exclude_strand_ambiguous: bool | None = None,
) -> ProjectionResult:
    """Project a target from genotype likelihoods (beagle GL file) against a
    cached panel (SCIENCE.md D17).

    The genotype-likelihood analog of :func:`project_target`: instead of
    collapsing the target to hard 0/1/2 dosages, it carries per-site genotype
    likelihoods and marginalizes over the unknown genotype under a
    Hardy-Weinberg prior (the NGSadmix / fastNGSadmix model). This downweights
    low-confidence sites, so for genuinely low-coverage data it changes (and
    improves) the estimate relative to hard calls. No plink2 is needed: the
    beagle file is matched to ``panel.bim`` by variant ID and oriented to the
    panel's allele-1 axis in pure Python.

    ``exclude_strand_ambiguous`` follows the same policy as
    :func:`project_target` (default ``None`` excludes A/T,C/G SNPs protectively
    unless the cache is certified clean; see SCIENCE.md D11).

    The returned :class:`ProjectionResult` has ``heterozygosity`` = NaN (there
    are no hard genotype calls) and ``n_snps_used`` = the number of panel SNPs
    with usable GLs. Mapping / reference bias is not corrected (it persists even
    with genotype likelihoods).
    """
    t0 = time.time()
    manifest = load_cache_manifest(cache_dir)
    panel_bim = cache_dir / "panel.bim"
    if not panel_bim.exists():
        raise PanelCacheError(
            f"project_target_gl: cache missing panel.bim at {panel_bim}",
        )

    effective_exclude = _resolve_exclude_strand_ambiguous(
        exclude_strand_ambiguous, manifest.strand_ambiguous_excluded,
    )

    beagle = read_beagle_gl(target_gl_beagle)
    gl_panel = align_gl_to_panel(
        beagle=beagle, panel_bim=panel_bim,
        exclude_strand_ambiguous=effective_exclude,
    )

    P = load_cached_p(cache_dir, manifest.k)
    if P.shape[0] != gl_panel.shape[0]:
        raise PanelCacheError(
            f"project_target_gl: cached P has {P.shape[0]} SNPs but the "
            f"panel-aligned GL matrix has {gl_panel.shape[0]} rows; the cache "
            f"is internally inconsistent (P row count != panel.bim variant "
            f"count); rebuild the cache",
        )

    n_obs = int((~np.isnan(gl_panel).any(axis=1)).sum())
    logger.info(
        "project_target_gl: projecting target on %d usable GL sites (of %d in P)",
        n_obs, P.shape[0],
    )
    q, n_iter, converged = numpy_supervised_projection_gl(
        gl=gl_panel, p_matrix=P, k=manifest.k,
    )
    logger.info(
        "project_target_gl: SLSQP %s in %d iters; Q = %s; total wallclock %.1fs",
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
        heterozygosity=float("nan"),
    )


__all__ = ["project_target", "project_target_gl"]
