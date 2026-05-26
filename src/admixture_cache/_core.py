"""Precomputed-P admixture projection for the three ADMIXTURE tracks.

Splits each supervised-ADMIXTURE track into:

1. **Panel cache build** (one-time per panel × K × cluster-YAML combo,
   ~hours of compute): run stock ADMIXTURE × N restarts via ToolRunner,
   validate multimodality, cache the best-LL P matrix + non-target Q +
   panel.bim + manifest. Lives in
   ``data/<panel_id>/admixture_cache/<track>_<K>_<yaml_sha8>/``.

2. **Per-target projection** (every run, <2 sec): align target.bed
   to cached panel.bim variants + axes, load dosage as NumPy array,
   solve for target Q via scipy SLSQP under the standard binomial
   admixture likelihood. NO ADMIXTURE binary needed at projection
   time.

Phase 0b validated this approach: NumPy projection matches stock
ADMIXTURE Q values to 1e-5 absolute (50× tighter than the 0.5%
acceptance threshold) on real Carsten data. Wallclock 0.02 sec
for the SLSQP step itself; total ~2 sec end-to-end including
plink2-based target alignment + dosage load.

Design rationale: see workplan v0.2 at
``cs-wiki/projects/admixture-projection-cache-workplan.md``.

This module is consumed by:
- ``ancestry_pipeline.tracks.regional`` (K=22 supervised ADMIXTURE
  on AADR v66 HO)
- ``ancestry_pipeline.pop_automation.tracks.ContinentalAdmixtureTrack``
  (K=4 on HGDP+1kGP)
- ``ancestry_pipeline.pop_automation.tracks.AncestralClusterTrack``
  (K=4 on Bug-#45-geo-filtered AADR v66 1240K, per continent)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.optimize import minimize

from admixture_cache.errors import (
    PopAutomationConfigError,
)

if TYPE_CHECKING:
    from admixture_cache.runner import ToolRunner

logger = logging.getLogger(__name__)


# ─── Cache manifest schema ───────────────────────────────────────────────


class PanelCacheManifest(BaseModel):
    """Manifest written next to cached P + Q + bim. Validated at
    cache-load time; any SHA mismatch triggers cache miss → fall back
    to full run (or rebuild via build script).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    track: str  # "regional", "continental_admixture", "ancestral_cluster"
    continent: str | None = None  # only set for ancestral_cluster
    panel_id: str
    panel_version: str
    panel_bim_sha256: str
    clusters_yaml_sha256: str
    k: int
    admixture_version: str
    seeds_used: list[int]
    best_seed: int
    best_loglikelihood: float
    restart_sd_max: float
    cluster_order: list[str]
    geo_filter_yaml_shas: dict[str, str] = field(default_factory=dict)
    pgen_samplebind_version: str | None = None
    build_wallclock_seconds: float
    build_timestamp: str


@dataclass(frozen=True)
class ProjectionResult:
    """Per-target projection output. Q vector + cluster names from
    cached manifest. Panel-stability metric carried through from build
    time."""

    target_q: np.ndarray  # shape (K,)
    cluster_order: list[str]
    panel_stability_max_sd: float  # from cached restart_sd metadata
    n_snps_used: int  # non-missing SNPs after mask
    optimization_iterations: int
    converged: bool


# ─── NumPy supervised-ADMIXTURE projection (Phase 0b validated) ──────────


def numpy_supervised_projection(
    *, target_dosage: np.ndarray, p_matrix: np.ndarray, k: int,
    eps: float = 1e-9, maxiter: int = 200, ftol: float = 1e-9,
) -> tuple[np.ndarray, int, bool]:
    """Pure NumPy/scipy supervised-ADMIXTURE projection.

    Given target genotype dosage ``target_dosage`` (M-vector,
    values 0/1/2 with NaN for missing) and fixed allele-frequency
    matrix ``p_matrix`` (M × K, P[s,k] = freq of allele 1 in pop k
    at SNP s), compute the target's K-vector admixture proportions
    q via maximum-likelihood under the binomial model:

        L(q) = ∏_s Binomial(g_s; 2, q^T P_s)

    Subject to: sum(q) = 1, q_k >= 0.

    Validated in Phase 0b: matches stock ``admixture --supervised``
    Q to 1e-5 absolute on real Carsten data (50× tighter than
    workplan's 0.5% acceptance threshold). SLSQP converges in
    ~9 iterations / ~0.02 sec on 850K SNPs at K=4.

    Returns (q, n_iter, converged).
    """
    assert target_dosage.shape == (p_matrix.shape[0],), (
        f"dosage shape {target_dosage.shape} != P rows {p_matrix.shape[0]}"
    )
    assert p_matrix.shape[1] == k, (
        f"P has {p_matrix.shape[1]} columns but k={k}"
    )

    mask = ~np.isnan(target_dosage)
    g_obs = target_dosage[mask]
    P_obs = p_matrix[mask]

    if g_obs.size == 0:
        raise PopAutomationConfigError(
            "numpy_supervised_projection: target has zero non-missing "
            "SNPs after mask; cannot project (no data).",
        )

    def neg_log_lik(q: np.ndarray) -> float:
        f = np.clip(P_obs @ q, eps, 1 - eps)
        return -(g_obs * np.log(f) + (2 - g_obs) * np.log(1 - f)).sum()

    def grad_neg_log_lik(q: np.ndarray) -> np.ndarray:
        f = np.clip(P_obs @ q, eps, 1 - eps)
        score = g_obs / f - (2 - g_obs) / (1 - f)
        return -P_obs.T @ score

    result = minimize(
        neg_log_lik, np.ones(k) / k, jac=grad_neg_log_lik,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{
            "type": "eq",
            "fun": lambda q: q.sum() - 1.0,
            "jac": lambda q: np.ones(k),
        }],
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return result.x, result.nit, result.success


# ─── Cache build (one-time, slow) ────────────────────────────────────────


def build_panel_cache(
    *,
    panel_bed: Path,
    panel_pop_file: Path,
    clusters_yaml: Path,
    k: int,
    cache_dir: Path,
    admixture_runner: "ToolRunner",
    track: str,
    panel_id: str,
    panel_version: str,
    admixture_version: str,
    continent: str | None = None,
    geo_filter_yaml_shas: dict[str, str] | None = None,
    pgen_samplebind_version: str | None = None,
    seeds: list[int] | None = None,
    sd_threshold: float = 0.02,
    threads: int = 16,
    # Per-restart timeout is enforced inside admixture_runner.run; see
    # per_restart_timeout_seconds below. Parallel-restart config below.
    max_parallel_restarts: int = 1,
    # Default 24 hr. Empirical: K=21 regional cache on AADR v66 HO
    # (27K samples × 580K SNPs) needs ~12-14 hr per restart to reach
    # delta<0.0001 (each QN/Block iter ~25 min; ~25 iters to converge).
    # K=4 caches (CAT + AC) finish in <2 hr per restart. 24hr is
    # tolerant of the slowest case; one-time cost so wallclock matters
    # less than correctness.
    per_restart_timeout_seconds: int = 86400,
) -> PanelCacheManifest:
    """Build a panel-only admixture cache: stock ADMIXTURE × N restarts,
    multimodality validation, save best-LL P + manifest.

    The panel_bed must contain ONLY panel samples (anchors + clinal,
    no target). build_supervised_pop_file or equivalent must have
    produced the matching panel_pop_file labeling each row.

    Idempotent: if cache_dir/manifest.json exists and SHAs match
    (panel_bim_sha + clusters_yaml_sha + K + geo_filter_yaml_shas),
    skip rebuild and return the existing manifest.

    On multimodality failure (max per-cluster restart_sd > sd_threshold),
    raises PopAutomationConfigError after saving partial outputs for
    debugging. Cache is NOT marked valid (no manifest.json written).

    Returns the validated PanelCacheManifest.
    """
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = cache_dir / "build_logs"
    log_dir.mkdir(exist_ok=True)

    # Compute current config SHAs for idempotency + manifest
    panel_bim_path = panel_bed.with_suffix(".bim")
    if not panel_bim_path.exists():
        raise PopAutomationConfigError(
            f"build_panel_cache: panel .bim missing at {panel_bim_path}",
        )
    panel_bim_sha = sha256_file(panel_bim_path)
    clusters_yaml_sha = sha256_file(clusters_yaml)
    geo_shas = geo_filter_yaml_shas or {}

    # Idempotency check
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        matched, reason = verify_cache_matches_current_config(
            cache_dir=cache_dir,
            expected_panel_bim_sha256=panel_bim_sha,
            expected_clusters_yaml_sha256=clusters_yaml_sha,
            expected_k=k,
            expected_geo_filter_yaml_shas=geo_shas if geo_shas else None,
        )
        if matched:
            logger.info(
                "build_panel_cache: cache at %s matches current config; "
                "skipping rebuild (no-op).",
                cache_dir,
            )
            return load_cache_manifest(cache_dir)
        logger.info(
            "build_panel_cache: cache at %s exists but stale (%s); rebuilding.",
            cache_dir, reason,
        )

    # Run N restarts via ToolRunner. Sequential by default
    # (max_parallel_restarts=1); operator opts into parallel for ~N×
    # wallclock reduction at the cost of N× peak memory + CPU contention
    # (each ADMIXTURE process needs `threads` threads + ~16 GB RSS).
    t0 = time.time()
    effective_parallelism = max(1, min(max_parallel_restarts, len(seeds)))
    if effective_parallelism > 1:
        logger.info(
            "build_panel_cache: running %d restarts in parallel "
            "(max_parallel_restarts=%d, threads per restart=%d, "
            "total subprocess threads=%d)",
            effective_parallelism, max_parallel_restarts, threads,
            effective_parallelism * threads,
        )

    def _run_one_restart(seed: int) -> dict:
        return _run_one_admixture_restart(
            seed=seed,
            panel_bed=panel_bed,
            panel_pop_file=panel_pop_file,
            k=k,
            threads=threads,
            cache_dir=cache_dir,
            log_dir=log_dir,
            admixture_runner=admixture_runner,
            per_restart_timeout_seconds=per_restart_timeout_seconds,
        )

    per_restart_results: list[dict] = []
    if effective_parallelism > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # ThreadPool is the right choice — each restart spawns a
        # subprocess that releases the GIL (no Python-side contention).
        with ThreadPoolExecutor(max_workers=effective_parallelism) as ex:
            future_to_seed = {ex.submit(_run_one_restart, s): s for s in seeds}
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    per_restart_results.append(future.result())
                except Exception as exc:
                    # Cancel pending restarts on first failure
                    for other in future_to_seed:
                        if not other.done():
                            other.cancel()
                    raise PopAutomationConfigError(
                        f"build_panel_cache: parallel restart seed={seed} "
                        f"failed: {exc}",
                    ) from exc
    else:
        for seed in seeds:
            per_restart_results.append(_run_one_restart(seed))

    # Re-sort by seed for deterministic downstream ordering (parallel
    # completion order is nondeterministic; we want the .json manifest
    # + restart_sd.json to read out in seed order regardless).
    per_restart_results.sort(key=lambda r: r["seed"])

    # Pick best LL restart
    with_ll = [r for r in per_restart_results if r["ll"] is not None]
    if not with_ll:
        raise PopAutomationConfigError(
            "build_panel_cache: no restart produced a parseable "
            "loglikelihood; check build_logs/",
        )
    best = max(with_ll, key=lambda r: r["ll"])
    logger.info(
        "build_panel_cache: best restart seed=%d (LL=%.6e)",
        best["seed"], best["ll"],
    )

    # Compute multimodality SD across restarts on the Q matrices.
    # SD per (cluster, sample) over the restarts, then take the max.
    q_matrices = [np.loadtxt(r["q_path"]) for r in per_restart_results]
    if len(q_matrices) >= 2:
        # All Q matrices have shape (N_samples, K); SD per (sample, cluster)
        stacked = np.stack(q_matrices, axis=0)  # (N_restarts, N_samples, K)
        per_cell_sd = stacked.std(axis=0, ddof=1)  # (N_samples, K)
        restart_sd_max = float(per_cell_sd.max())
        logger.info(
            "build_panel_cache: max per-cluster restart_sd across %d "
            "restarts: %.4f", len(q_matrices), restart_sd_max,
        )
        # Multimodality validation
        if restart_sd_max > sd_threshold:
            raise PopAutomationConfigError(
                f"build_panel_cache: multimodality detected — max per-cluster "
                f"restart SD = {restart_sd_max:.4f} > threshold {sd_threshold}. "
                f"Cache NOT marked valid. Investigate: different seeds? "
                f"Cluster YAML curation? See {log_dir} for per-restart logs.",
            )
    else:
        # Single-restart build: no SD to compute. Used for quick
        # end-to-end validation; production builds should use >=2 seeds.
        per_cell_sd = np.zeros_like(q_matrices[0])
        restart_sd_max = 0.0
        logger.warning(
            "build_panel_cache: single-restart build (len(seeds)=1); "
            "multimodality check skipped. This is OK for validation but "
            "production caches should use seeds=[1,2,3,4,5].",
        )

    # Cluster order: derive from .pop file (ADMIXTURE supervised convention:
    # K columns of .Q follow the first-appearance order of non-'-' labels
    # in the .pop file).
    cluster_order = _derive_cluster_order_from_pop_file(
        panel_pop_file=panel_pop_file,
        expected_k=k,
    )

    # Copy best restart's outputs to the canonical cache locations
    import shutil as _shutil
    best_p_dest = cache_dir / f"panel.{k}.P"
    best_q_dest = cache_dir / f"panel.{k}.Q"
    _shutil.copy2(best["p_path"], best_p_dest)
    _shutil.copy2(best["q_path"], best_q_dest)
    # Also copy panel.bim for projection-time variant alignment
    _shutil.copy2(panel_bim_path, cache_dir / "panel.bim")

    # Write restart_sd.json (per-cluster max SD across non-anchor samples)
    restart_sd_per_cluster = {
        cluster_order[k_idx]: float(per_cell_sd[:, k_idx].max())
        for k_idx in range(k)
    }
    (cache_dir / "restart_sd.json").write_text(
        json.dumps({
            "per_cluster_max_sd": restart_sd_per_cluster,
            "overall_max_sd": restart_sd_max,
            "threshold": sd_threshold,
            "n_restarts": len(per_restart_results),
        }, indent=2)
    )
    (cache_dir / "cluster_order.json").write_text(
        json.dumps({"cluster_order": cluster_order}, indent=2)
    )

    # Write manifest LAST — its presence is the "cache is valid" signal
    manifest = PanelCacheManifest(
        schema_version=1,
        track=track,
        continent=continent,
        panel_id=panel_id,
        panel_version=panel_version,
        panel_bim_sha256=panel_bim_sha,
        clusters_yaml_sha256=clusters_yaml_sha,
        k=k,
        admixture_version=admixture_version,
        seeds_used=seeds,
        best_seed=best["seed"],
        best_loglikelihood=best["ll"],
        restart_sd_max=restart_sd_max,
        cluster_order=cluster_order,
        geo_filter_yaml_shas=geo_shas,
        pgen_samplebind_version=pgen_samplebind_version,
        build_wallclock_seconds=time.time() - t0,
        build_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    logger.info(
        "build_panel_cache: SUCCESS — cache at %s (total wallclock %.0fs)",
        cache_dir, manifest.build_wallclock_seconds,
    )
    return manifest


def _run_one_admixture_restart(
    *,
    seed: int,
    panel_bed: Path,
    panel_pop_file: Path,
    k: int,
    threads: int,
    cache_dir: Path,
    log_dir: Path,
    admixture_runner: "ToolRunner",
    per_restart_timeout_seconds: int,
) -> dict:
    """Run one supervised-ADMIXTURE restart at the given seed.

    Stages a private restart_dir (cache_dir/build_restart_<seed>/),
    copies panel.bed/bim/fam + panel.pop into it (so concurrent
    restarts don't clobber each other's output files), runs
    ``admixture --supervised -j<threads> -s<seed> panel.bed K``, and
    returns the per-restart result dict for the multimodality + best-LL
    selection downstream.

    Safe to call concurrently with different seeds: each restart has
    its own isolated working directory + log file.
    """
    import shutil as _shutil

    restart_dir = cache_dir / f"build_restart_{seed}"
    restart_dir.mkdir(exist_ok=True)

    # ADMIXTURE writes <bfile_stem>.<K>.{P,Q} in cwd. Stage the input
    # BED triplet + .pop file in restart_dir so the outputs land
    # alongside without clobbering between concurrent seeds.
    for suffix in (".bed", ".bim", ".fam"):
        src = panel_bed.with_suffix(suffix)
        dst = restart_dir / f"panel{suffix}"
        if not dst.exists():
            _shutil.copy2(src, dst)
    pop_dst = restart_dir / "panel.pop"
    if not pop_dst.exists():
        _shutil.copy2(panel_pop_file, pop_dst)

    log_file = log_dir / f"restart_{seed}.out"
    logger.info(
        "build_panel_cache: starting restart seed=%d (K=%d, %d threads)",
        seed, k, threads,
    )
    restart_t0 = time.time()
    admixture_runner.run(
        args=[
            "--supervised",
            f"-j{threads}",
            f"-s{seed}",
            "panel.bed",
            str(k),
        ],
        cwd=restart_dir,
        log_dir=log_dir,
        timeout_seconds=per_restart_timeout_seconds,
    )
    restart_elapsed = time.time() - restart_t0

    p_path = restart_dir / f"panel.{k}.P"
    q_path = restart_dir / f"panel.{k}.Q"
    if not p_path.exists() or not q_path.exists():
        raise PopAutomationConfigError(
            f"_run_one_admixture_restart: restart seed={seed} produced "
            f"no output files (looked for {p_path} and {q_path}); see "
            f"log_dir={log_dir}",
        )

    log_text = log_file.read_text() if log_file.exists() else ""
    ll = _parse_admixture_loglikelihood(log_text)
    logger.info(
        "build_panel_cache: restart seed=%d finished in %.1fs, LL=%s",
        seed, restart_elapsed, f"{ll:.3e}" if ll is not None else "?",
    )
    return {
        "seed": seed,
        "p_path": p_path,
        "q_path": q_path,
        "ll": ll,
        "wallclock_seconds": restart_elapsed,
    }


def _parse_admixture_loglikelihood(log_text: str) -> float | None:
    """Parse the final 'Loglikelihood: <val>' line from an ADMIXTURE
    stdout log. Returns None if no parseable line found.

    ADMIXTURE emits one Loglikelihood line per iteration; the last one
    is the converged value.
    """
    import re
    matches = re.findall(
        r"Loglikelihood:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        log_text,
    )
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _derive_cluster_order_from_pop_file(
    *, panel_pop_file: Path, expected_k: int,
) -> list[str]:
    """ADMIXTURE supervised mode orders the K columns of .Q by the
    first-appearance order of non-'-' labels in the .pop file.

    Returns the ordered list of cluster names; errors if the count
    doesn't match expected_k.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    with panel_pop_file.open() as f:
        for line in f:
            label = line.strip()
            if not label or label == "-":
                continue
            if label not in seen_set:
                seen.append(label)
                seen_set.add(label)
    if len(seen) != expected_k:
        raise PopAutomationConfigError(
            f"_derive_cluster_order_from_pop_file: {panel_pop_file.name} "
            f"has {len(seen)} distinct non-'-' labels but K={expected_k}. "
            f"Labels: {seen}",
        )
    return seen


# ─── Cache I/O ───────────────────────────────────────────────────────────


def load_cached_p(cache_dir: Path, k: int) -> np.ndarray:
    """Load cached panel.<K>.P matrix (M × K text format, ADMIXTURE
    convention)."""
    p_path = cache_dir / f"panel.{k}.P"
    if not p_path.exists():
        raise PopAutomationConfigError(
            f"load_cached_p: cache file missing: {p_path}; "
            f"run `ancestry-pipeline build-caches` to build it.",
        )
    P = np.loadtxt(p_path)
    if P.ndim != 2 or P.shape[1] != k:
        raise PopAutomationConfigError(
            f"load_cached_p: {p_path} has shape {P.shape}; expected "
            f"(M, {k})",
        )
    return P


def load_cache_manifest(cache_dir: Path) -> PanelCacheManifest:
    """Load + validate the cache manifest JSON."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise PopAutomationConfigError(
            f"load_cache_manifest: {manifest_path} missing; cache is "
            f"either incomplete or never built.",
        )
    return PanelCacheManifest.model_validate_json(manifest_path.read_text())


def verify_cache_matches_current_config(
    *, cache_dir: Path,
    expected_panel_bim_sha256: str,
    expected_clusters_yaml_sha256: str,
    expected_k: int,
    expected_geo_filter_yaml_shas: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Check whether cache_dir's manifest matches the current config.

    Returns (matched, reason). If matched is False, the reason string
    explains which SHA diverged (for actionable error messages /
    rebuild script logging).
    """
    try:
        manifest = load_cache_manifest(cache_dir)
    except PopAutomationConfigError as exc:
        return False, f"cache manifest unloadable: {exc}"

    if manifest.k != expected_k:
        return False, (
            f"K mismatch: cache has K={manifest.k}, current config "
            f"expects K={expected_k}"
        )
    if manifest.panel_bim_sha256 != expected_panel_bim_sha256:
        return False, "panel .bim changed (panel version bump?)"
    if manifest.clusters_yaml_sha256 != expected_clusters_yaml_sha256:
        return False, "clusters_yaml changed (curator edit?)"
    if expected_geo_filter_yaml_shas is not None:
        for yaml_name, expected_sha in expected_geo_filter_yaml_shas.items():
            cached_sha = manifest.geo_filter_yaml_shas.get(yaml_name)
            if cached_sha != expected_sha:
                return False, (
                    f"geo-filter YAML {yaml_name!r} changed "
                    f"({cached_sha[:8] if cached_sha else 'missing'} → "
                    f"{expected_sha[:8]})"
                )
    return True, "match"


# ─── Hashing utilities ───────────────────────────────────────────────────


def sha256_file(path: Path, *, chunk_size: int = 2**16) -> str:
    """Streaming sha256 of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ─── LD pruning (optional speedup before ADMIXTURE training) ─────────────


def ld_prune_panel(
    *,
    panel_bed: Path,
    output_prefix: Path,
    plink2_runner: "ToolRunner",
    window_kb: int = 50,
    step_size: int = 5,
    r2_threshold: float = 0.5,
    log_dir: Path,
    timeout_seconds: int = 3600,
) -> Path:
    """Apply LD-pruning to a panel BED via plink2 --indep-pairwise.

    Per Alexander et al. 2009 and the HLD Exp 2 measurements, LD-pruning
    is the dominant cost-cutter for supervised-ADMIXTURE training:
    LD-pruned SNPs are statistically more independent → ADMIXTURE
    converges in fewer iterations. Combined with the reduced SNP count
    per iter, a 50/5/0.5 prune typically yields 30-50% of variants and
    gives 3-5× total speedup.

    Two-step plink2 invocation:

    1. ``plink2 --bfile <panel> --indep-pairwise <window> <step> <r²>
       --out <output_prefix>``: identifies the LD-pruned variant subset,
       writes ``<output_prefix>.prune.in`` (variants to keep) and
       ``<output_prefix>.prune.out`` (variants to remove).
    2. ``plink2 --bfile <panel> --extract <output_prefix>.prune.in
       --make-bed --out <output_prefix>``: produces the pruned BED.

    Args:
        panel_bed: Path to the unpruned panel .bed (with sibling
            .bim/.fam).
        output_prefix: Prefix for plink2 outputs. The pruned BED lands
            at ``<output_prefix>.bed`` (with sibling .bim/.fam).
        plink2_runner: ToolRunner for plink2 invocations.
        window_kb: --indep-pairwise window size in kb (default 50).
        step_size: --indep-pairwise step size in variants (default 5).
        r2_threshold: --indep-pairwise r² threshold (default 0.5).
        log_dir: Where to write plink2 logs.
        timeout_seconds: Per-plink2-call timeout (default 1hr).

    Returns:
        Path to the pruned ``<output_prefix>.bed``.

    The .pop file is NOT carried through automatically — the caller
    must regenerate it for the pruned variant set. (In practice the
    .pop file lists per-sample labels, not per-variant data, so it
    stays valid unless the sample set also changed; the caller can
    just copy panel.pop next to the pruned output.)
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    panel_prefix = panel_bed.with_suffix("")

    # Step 1: identify LD-pruned variant subset
    plink2_runner.run(
        args=[
            "--bfile", str(panel_prefix),
            "--indep-pairwise",
            str(window_kb), str(step_size), str(r2_threshold),
            "--out", str(output_prefix),
        ],
        cwd=output_prefix.parent,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
    )

    prune_in = output_prefix.with_suffix(".prune.in")
    if not prune_in.exists():
        raise PopAutomationConfigError(
            f"ld_prune_panel: plink2 --indep-pairwise produced no "
            f"prune.in at {prune_in}; see {log_dir} for the plink2 log",
        )

    # Step 2: extract the pruned subset into a new BED
    plink2_runner.run(
        args=[
            "--bfile", str(panel_prefix),
            "--extract", str(prune_in),
            "--make-bed",
            "--out", str(output_prefix),
        ],
        cwd=output_prefix.parent,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
    )

    pruned_bed = output_prefix.with_suffix(".bed")
    if not pruned_bed.exists():
        raise PopAutomationConfigError(
            f"ld_prune_panel: plink2 --extract produced no output at "
            f"{pruned_bed}; see {log_dir} for the plink2 log",
        )

    # Diagnostics: count variants before/after for the operator
    pre_count = sum(1 for _ in panel_bed.with_suffix(".bim").open())
    post_count = sum(1 for _ in pruned_bed.with_suffix(".bim").open())
    logger.info(
        "ld_prune_panel: %s (%d variants) -> %s (%d retained, "
        "%.1f%% kept, %.2f× SNP reduction)",
        panel_bed.name, pre_count, pruned_bed.name, post_count,
        100.0 * post_count / max(pre_count, 1),
        pre_count / max(post_count, 1),
    )
    return pruned_bed


# ─── Target-to-panel alignment ───────────────────────────────────────────


def align_target_to_panel_bim(
    *, target_bed: Path, panel_bim: Path,
    output_prefix: Path, plink2_runner: "ToolRunner",
    log_dir: Path,
    timeout_seconds: int = 600,
) -> Path:
    """Filter target.bed to cached panel.bim variant set + align REF/ALT
    axes via plink2 --alt1-allele.

    Bug #38/#40 lesson: silent REF/ALT axis mismatches between target
    and reference panel produce wildly wrong projection results.
    --alt1-allele forces the target's ALT1 column to match the panel's
    ALT1 column at every overlapping variant, flipping dosages where
    needed.

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
        raise PopAutomationConfigError(
            f"align_target_to_panel_bim: plink2 succeeded but "
            f"{aligned_bed} not produced",
        )
    return aligned_bed


# ─── Target dosage loading ───────────────────────────────────────────────


def extract_target_dosage_via_plink2(
    *, target_bed: Path, output_prefix: Path,
    plink2_runner: "ToolRunner", log_dir: Path,
    timeout_seconds: int = 600,
) -> np.ndarray:
    """Extract target dosage via ``plink2 --recode A`` (text format),
    then parse to a NumPy 1D array of len M (M = SNPs in target.bim,
    NaN for missing).

    For a single target, this is acceptable (~28 sec on 850K SNPs).
    Phase 1 stretch goal: replace with ``bed-reader`` library for
    direct binary BED reading (~30× faster).

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
        raise PopAutomationConfigError(
            f"extract_target_dosage_via_plink2: {raw_path} not produced",
        )

    import pandas as pd

    raw = pd.read_csv(raw_path, sep="\t", na_values=["NA"])
    if raw.shape[0] != 1:
        raise PopAutomationConfigError(
            f"extract_target_dosage_via_plink2: expected 1 sample in "
            f"{raw_path}, got {raw.shape[0]}",
        )
    # First 6 columns are FID IID PAT MAT SEX PHENOTYPE; rest are dosages
    return raw.iloc[0, 6:].values.astype(np.float64)


# ─── Top-level projection orchestration ──────────────────────────────────


def project_target(
    *, target_bed: Path,
    cache_dir: Path,
    plink2_runner: "ToolRunner",
    work_dir: Path,
) -> ProjectionResult:
    """End-to-end per-target projection:
    1. Validate cache exists + load manifest
    2. Align target.bed to cached panel.bim (variant set + REF/ALT axes)
    3. Extract aligned target dosage as NumPy array
    4. Load cached P matrix
    5. Run NumPy SLSQP projection
    6. Return ProjectionResult with Q vector + metadata

    Total wallclock: ~2 sec end-to-end (validated in Phase 0b prototype,
    minus the 28-sec pandas .raw load which dominates; can be reduced
    further with bed-reader).
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    manifest = load_cache_manifest(cache_dir)
    logger.info(
        "project_target: loaded cache manifest (track=%s panel=%s K=%d, "
        "built %s, wallclock %.0fs)",
        manifest.track, manifest.panel_id, manifest.k,
        manifest.build_timestamp, manifest.build_wallclock_seconds,
    )

    panel_bim = cache_dir / "panel.bim"
    if not panel_bim.exists():
        raise PopAutomationConfigError(
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
        raise PopAutomationConfigError(
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
