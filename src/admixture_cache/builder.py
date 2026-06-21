"""Panel cache build (one-time, slow).

Runs stock supervised ADMIXTURE × N restarts via a caller-supplied
ToolRunner, validates multimodality across restarts (per-cluster SD
of Q must stay under the configured threshold), and writes the
best-LL P matrix + panel.bim + manifest to the cache directory.

Also exposes an optional LD-pruning helper to be run upstream of the
build: pruning typically retains 30–50% of variants and yields 3–5×
speedup at the ADMIXTURE training step.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import shutil
import signal
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from admixture_cache._dispatch import _call_runner, _runner_supports
from admixture_cache._paths import append_suffix
from admixture_cache.errors import PanelCacheError
from admixture_cache.io import (
    load_cache_manifest,
    sha256_file,
    verify_cache_matches_current_config,
)
from admixture_cache.manifest import PanelCacheManifest

if TYPE_CHECKING:
    from admixture_cache.runner import ToolRunner

logger = logging.getLogger(__name__)


class _RestartResult(TypedDict):
    seed: int
    p_path: Path
    q_path: Path
    ll: float | None
    wallclock_seconds: float


def _detect_numa_nodes() -> int:
    """Return the number of NUMA nodes on the current host.

    Reads ``/sys/devices/system/node/`` which Linux populates with one
    ``nodeN`` subdirectory per NUMA node (N is an integer). Returns 1
    on platforms where the path doesn't exist (macOS, Windows,
    single-node Linux without the sysfs entries).

    The kernel convention is strict: directory named ``node<N>`` where
    ``<N>`` is a non-negative integer. We filter accordingly so that
    sibling files like ``has_cpu`` / ``has_memory`` / ``online`` —
    and any future entries that happen to start with ``node`` but
    aren't node directories — don't inflate the count.
    """
    node_dir = Path("/sys/devices/system/node")
    if not node_dir.is_dir():
        return 1
    return sum(
        1 for p in node_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("node")
        and p.name[4:].isdigit()
    )


def _resolve_numa_nodes(*, enabled: bool, effective_parallelism: int) -> int:
    """Decide how many NUMA nodes to spread restarts across.

    Returns 1 (i.e. "no pinning") in any of these cases:
    - operator passed ``numa_node_per_restart=False`` (default)
    - sequential execution (pinning doesn't help with one process)
    - ``numactl`` not on PATH (macOS, minimal Linux containers)
    - the host has 1 NUMA node (single-socket box)

    Otherwise returns ``min(n_nodes, effective_parallelism)`` — capping
    at the actual parallelism so we don't reserve more nodes than we'll
    use.
    """
    if not enabled:
        return 1
    if effective_parallelism < 2:
        logger.info(
            "build_panel_cache: numa_node_per_restart=True but only one "
            "restart in flight; skipping NUMA pinning (no-op)",
        )
        return 1
    if shutil.which("numactl") is None:
        logger.warning(
            "build_panel_cache: numa_node_per_restart=True but `numactl` "
            "not on PATH; skipping NUMA pinning (install numactl or set "
            "the flag to False to silence this warning)",
        )
        return 1
    n_nodes = _detect_numa_nodes()
    if n_nodes < 2:
        logger.info(
            "build_panel_cache: numa_node_per_restart=True but host has "
            "%d NUMA node(s); skipping NUMA pinning (single-socket box)",
            n_nodes,
        )
        return 1
    use_n = min(n_nodes, effective_parallelism)
    logger.info(
        "build_panel_cache: NUMA pinning enabled — spreading %d restarts "
        "across %d node(s) (host has %d node(s) total)",
        effective_parallelism, use_n, n_nodes,
    )
    return use_n


def _auto_max_parallel_restarts(*, threads: int, n_seeds: int) -> int:
    """Default ``max_parallel_restarts`` from a memory-bandwidth heuristic.

    ADMIXTURE is memory-bandwidth-bound at typical panel sizes (>=10K
    samples × >=500K SNPs at K>=4). Beyond ~2-3 parallel restarts on a
    single-socket machine, additional parallelism mostly burns DRAM
    bandwidth without reducing wallclock. ``cores // (threads * 2)``
    biases toward fewer/fatter parallelism to stay below the DRAM
    bandwidth ceiling; ``max(1, …)`` keeps the value sane on tiny
    machines and ``min(n_seeds, …)`` avoids over-allocating beyond the
    actual workload.
    """
    cores = os.cpu_count() or 1
    return max(1, min(n_seeds, cores // max(threads * 2, 1)))


def build_panel_cache(
    *,
    panel_bed: Path,
    panel_pop_file: Path,
    clusters_yaml: Path,
    k: int,
    cache_dir: Path,
    admixture_runner: ToolRunner,
    track: str | None = None,
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
    # ``None`` triggers a memory-bandwidth heuristic
    # (`cores // (threads * 2)`, capped at len(seeds)); pass an
    # explicit positive integer to override.
    max_parallel_restarts: int | None = None,
    # NUMA pinning: when True (and Linux + numactl available + multi-
    # socket box), each parallel restart's subprocess is wrapped with
    # `numactl --membind=N --` where N is `(restart_index % n_nodes)`.
    # Pins memory allocation to the chosen NUMA node, avoiding the
    # ~2-3× cross-node latency penalty when ADMIXTURE's working set
    # is bigger than one node's local memory. Worth +10-30% on
    # multi-socket hardware (n2-standard-32+, c2-standard-30+);
    # no-op on single-socket boxes. Skipped with a warning if
    # numactl isn't on PATH.
    numa_node_per_restart: bool = False,
    # Default 24 hr. Empirical: K=21 regional cache on AADR v66 HO
    # (27K samples × 580K SNPs) needs ~12-14 hr per restart to reach
    # delta<0.0001 (each QN/Block iter ~25 min; ~25 iters to converge).
    # K=4 caches finish in <2 hr per restart. 24hr is tolerant of the
    # slowest case; one-time cost so wallclock matters less than
    # correctness.
    per_restart_timeout_seconds: int = 86400,
) -> PanelCacheManifest:
    """Build a panel-only admixture cache: stock ADMIXTURE × N restarts,
    multimodality validation, save best-LL P + manifest.

    The panel_bed must contain ONLY panel samples (anchors + clinal,
    no target). build_supervised_pop_file or equivalent must have
    produced the matching panel_pop_file labeling each row.

    Idempotent: if cache_dir/manifest.json exists and SHAs match
    (panel_bim_sha + panel_pop_sha + clusters_yaml_sha + K +
    geo_filter_yaml_shas), skip rebuild and return the existing manifest.
    The panel_pop_sha comparison is skipped for legacy caches whose
    manifest predates the field (panel_pop_sha256 is None), so an
    upgrade never triggers a spurious rebuild on its own.

    On multimodality failure (max per-cluster restart_sd > sd_threshold),
    raises PanelCacheError after saving partial outputs for
    debugging. Cache is NOT marked valid (no manifest.json written).

    Fully-labeled panels (no unlabeled '-' rows in panel_pop_file)
    -------------------------------------------------------------
    When every panel sample carries a cluster label, supervised
    ADMIXTURE has no free Q to estimate: Q is pinned by the labels and
    the build reduces to a near-closed-form per-cluster allele-frequency
    pass that converges in ~1 iteration. Two consequences worth knowing:

    - The build is fast and **seed-independent** — every restart computes
      the identical pinned P/Q, so ``restart_sd_max`` collapses to
      machine epsilon (~1e-16) and the multimodality check is
      structurally vacuous (it can never fail). A surprisingly-quick
      build (e.g. tens of seconds at K=21) on a fully-labeled panel is
      expected, not a sign the run short-circuited.
    - The ``seeds`` loop is therefore redundant work — all restarts are
      byte-identical. Passing ``seeds=[1]`` is sufficient for a
      fully-labeled panel and avoids the N× cost. (Multiple seeds only
      buy multimodality detection, which requires unlabeled samples /
      free Q to be meaningful.)

    Parameters of note
    ------------------
    threads
        Per-ADMIXTURE-process thread count (passed as ``-j<threads>``).
        Defaults to 16, matching common single-socket cloud SKUs
        (n2/e2-standard-16). ADMIXTURE's QN/Block step scales reasonably
        to that level.

    max_parallel_restarts
        How many ADMIXTURE restarts to run concurrently. ``None``
        triggers a memory-bandwidth-aware heuristic
        (``os.cpu_count() // (threads * 2)``, capped at ``len(seeds)``,
        floor 1). Pass a positive integer to override.

        **Important**: ADMIXTURE is memory-bandwidth-bound at typical
        panel sizes (≥10K samples × ≥500K SNPs at K≥4). Beyond ~2–3
        parallel restarts on a single-socket machine, additional
        parallelism mostly burns DRAM bandwidth without reducing
        wallclock. Empirically on a 16-core / 125 GiB cloud VM with a
        15K × 1.14M panel at K=4: ``5 × threads=3`` runs each ADMIXTURE
        process at ~155% CPU (of 300% available with ``-j3``), while
        ``2 × threads=8`` typically achieves closer to 700% per-process
        CPU and similar total wallclock with 2.5× lower peak memory.

        The default heuristic biases toward fewer/fatter parallelism
        for this reason. Override only when the workload is known to be
        compute-bound (small panels, low K) or the machine has unusually
        high DRAM bandwidth per core.

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
        raise PanelCacheError(
            f"build_panel_cache: panel .bim missing at {panel_bim_path}",
        )
    panel_bim_sha = sha256_file(panel_bim_path)
    # Hash the supervised-label .pop the same way (fast-fail on a missing
    # file, mirroring the .bim guard above, so we don't reach restart
    # staging before surfacing it). Recorded in the manifest and fed into
    # the idempotency check below so an off-pipeline panel.pop edit that
    # left every other hashed input untouched still forces a rebuild.
    if not panel_pop_file.exists():
        raise PanelCacheError(
            f"build_panel_cache: panel .pop missing at {panel_pop_file}",
        )
    panel_pop_sha = sha256_file(panel_pop_file)
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
            expected_panel_pop_sha256=panel_pop_sha,
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

    # Resolve the parallel-restart count. ``None`` triggers a
    # memory-bandwidth-aware default; any integer caller value is honored
    # verbatim (after the standard clamps).
    t0 = time.time()
    if max_parallel_restarts is None:
        resolved_parallelism = _auto_max_parallel_restarts(
            threads=threads, n_seeds=len(seeds),
        )
        logger.info(
            "build_panel_cache: auto-selected max_parallel_restarts=%d "
            "(cores=%d, threads=%d, n_seeds=%d)",
            resolved_parallelism, os.cpu_count() or 1, threads, len(seeds),
        )
    else:
        resolved_parallelism = max_parallel_restarts
    effective_parallelism = max(1, min(resolved_parallelism, len(seeds)))
    if effective_parallelism > 1:
        # Parallel mode requires BOTH:
        # - `log_name` so LL parsing can locate the right log per restart
        #   when N concurrent ADMIXTURE processes write to the same dir
        # - `pid_callback` so the failure path can SIGTERM in-flight
        #   restarts (otherwise a single failure waits up to
        #   per_restart_timeout_seconds × (N-1) for the others to finish)
        missing = [
            param for param in ("log_name", "pid_callback")
            if not _runner_supports(admixture_runner, param)
        ]
        if missing:
            raise PanelCacheError(
                "build_panel_cache: parallel restarts (effective "
                f"parallelism={effective_parallelism}) require a "
                f"ToolRunner that accepts the {', '.join(repr(p) for p in missing)} "
                "keyword(s) (Protocol extensions introduced in "
                "admixture-cache v1.0). Either upgrade the runner "
                "implementation, declare `**kwargs` on its `run` method "
                "to forward unknown keywords, or pass "
                "max_parallel_restarts=1.",
            )
        logger.info(
            "build_panel_cache: running %d restarts in parallel "
            "(max_parallel_restarts=%d, threads per restart=%d, "
            "total subprocess threads=%d)",
            effective_parallelism, resolved_parallelism, threads,
            effective_parallelism * threads,
        )

    # Resolve NUMA pinning. Skip silently on platforms / setups that
    # don't support it (no numactl on PATH, single-node machine, or
    # sequential execution where pinning doesn't help). Logs the
    # decision so operators can confirm at INFO level.
    numa_n_nodes = _resolve_numa_nodes(
        enabled=numa_node_per_restart,
        effective_parallelism=effective_parallelism,
    )

    # If pinning is enabled but the runner can't accept argv_prefix
    # (and isn't a **kwargs forwarder), the prefix would be silently
    # dropped. Surface this loudly so operators know NUMA isn't
    # actually pinning anything; degrade to non-pinned execution.
    if numa_n_nodes > 1 and not _runner_supports(admixture_runner, "argv_prefix"):
        logger.warning(
            "build_panel_cache: numa_node_per_restart=True but the "
            "supplied admixture_runner doesn't accept the "
            "`argv_prefix` kwarg (Protocol extension added in v1.1). "
            "NUMA pinning would be silently dropped — degrading to "
            "non-pinned execution. Upgrade the runner to accept "
            "argv_prefix (or declare **kwargs on its `run` method) "
            "to enable NUMA pinning.",
        )
        numa_n_nodes = 1

    # Slot-based NUMA assignment: one queue entry per PARALLEL SLOT
    # (not per NUMA node). When n_nodes >= effective_parallelism,
    # every worker gets a distinct node — full NUMA pinning. When
    # n_nodes < effective_parallelism, the queue holds the nodes
    # cycled (`i % numa_n_nodes`) so the extra workers share nodes
    # — partial pinning but no worker ever blocks on `get()`.
    #
    # Sizing the queue to numa_n_nodes (not effective_parallelism)
    # would block excess workers in `get()` BEFORE they registered
    # a PID; on first-failure, `_cancel_inflight` would no-op against
    # them, and after a peer released its slot the blocked worker
    # would unblock, claim the slot, and spawn a FRESH admixture
    # subprocess that the cancellation path can never reach. That
    # leaks up-to-24h orphan ADMIXTURE processes per blocked worker.
    # See https://github.com/carstenerickson/admixture-cache/issues
    # if you need to reconstruct the v1.1.0 → v1.1.1 history.
    numa_node_pool: queue.Queue[int] | None = None
    if numa_n_nodes > 1:
        numa_node_pool = queue.Queue()
        for i in range(effective_parallelism):
            numa_node_pool.put(i % numa_n_nodes)
        if numa_n_nodes < effective_parallelism:
            logger.warning(
                "build_panel_cache: numa_node_per_restart=True with "
                "%d NUMA node(s) but max_parallel_restarts=%d — %d "
                "worker(s) will share a node (partial pinning). For "
                "exclusive per-worker pinning, lower "
                "max_parallel_restarts to %d.",
                numa_n_nodes, effective_parallelism,
                effective_parallelism - numa_n_nodes, numa_n_nodes,
            )

    # PIDs of in-flight subprocesses, keyed by seed. The reference
    # SubprocessToolRunner reports its PID via pid_callback; the
    # parallel-mode guard above requires the runner support
    # pid_callback when effective_parallelism > 1, so in that branch
    # this dict is always populated. In sequential mode, runners that
    # don't accept pid_callback simply leave entries unset and the
    # cancellation path is moot (no concurrency to cancel).
    pids: dict[int, int] = {}
    pids_lock = threading.Lock()

    def _make_pid_callback(seed: int) -> Callable[[int], None]:
        def cb(pid: int) -> None:
            with pids_lock:
                pids[seed] = pid
        return cb

    def _run_one_restart(seed: int) -> _RestartResult:
        # Claim a NUMA node from the slot pool (if pinning enabled).
        # The queue is sized to `effective_parallelism` (not
        # numa_n_nodes), so a worker dispatched by the executor
        # always finds a slot immediately — no `get()` blocking. When
        # numa_n_nodes < effective_parallelism, the queue contains
        # `i % numa_n_nodes` entries, meaning some workers share a
        # node (partial pinning, see the queue-init comment above).
        claimed_node: int | None = None
        argv_prefix: list[str] | None = None
        if numa_node_pool is not None:
            # `get_nowait()` would raise queue.Empty if the queue is
            # ever exhausted; that should be impossible by construction
            # (one slot per worker), but `get()` is the safer call —
            # it would surface the contract violation as a hang rather
            # than a cryptic Empty exception.
            claimed_node = numa_node_pool.get()
            argv_prefix = ["numactl", f"--membind={claimed_node}", "--"]
        try:
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
                pid_callback=_make_pid_callback(seed),
                allow_log_scan_fallback=(effective_parallelism == 1),
                argv_prefix=argv_prefix,
            )
        finally:
            # Release the NUMA slot back to the pool before dropping
            # the PID — order doesn't matter functionally, but
            # releasing the scarcer resource first lets queued
            # workers pick it up immediately.
            if claimed_node is not None and numa_node_pool is not None:
                numa_node_pool.put(claimed_node)
            # Drop the PID once the runner has returned (subprocess is
            # reaped). The cancellation path skips entries already
            # popped from the map; the lock serializes the pop against
            # _cancel_inflight's read so we don't briefly observe a
            # stale-but-just-popped PID.
            with pids_lock:
                pids.pop(seed, None)

    def _cancel_inflight(active_seeds: list[int]) -> None:
        """Terminate any subprocess this build owns that hasn't yet
        completed.

        Signals the subprocess's *process group* rather than its bare
        PID. Runners that spawn subprocesses with `start_new_session=True`
        (the recommended pattern, used by SubprocessToolRunner) give
        each child its own pgid equal to its pid. Signaling the pgid
        instead of the bare pid:

        - Avoids the PID-recycle race when a subprocess exits between
          PID capture and the cancellation pass (recycled PIDs almost
          never reuse a process group id immediately).
        - Reaches any grandchildren the subprocess may have spawned
          (rare in our use case but cheap to handle correctly).

        Falls back to bare PID signal on platforms / runners that
        don't support process groups.
        """
        for s in active_seeds:
            with pids_lock:
                pid = pids.get(s)
            if pid is None:
                continue
            # Already-gone or out-of-our-control are expected during
            # racy cancellation; silently no-op.
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                try:
                    pgid = os.getpgid(pid)
                except (ProcessLookupError, OSError):
                    # Process already gone; nothing to signal.
                    continue
                # Refuse to signal a process group that includes our own
                # process — that would mean the runner did NOT use
                # start_new_session, and killpg would terminate us too.
                if pgid == os.getpgrp():
                    logger.warning(
                        "_cancel_inflight: seed=%d pid=%d shares this "
                        "process's group (runner did not use "
                        "start_new_session?); falling back to bare PID "
                        "signal", s, pid,
                    )
                    os.kill(pid, signal.SIGTERM)
                else:
                    os.killpg(pgid, signal.SIGTERM)

    per_restart_results: list[_RestartResult] = []
    if effective_parallelism > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # ThreadPool is the right choice — each restart spawns a
        # subprocess that releases the GIL (no Python-side contention).
        # The executor is managed manually (no `with` block) because the
        # context-manager exit calls `shutdown(wait=True)`, which would
        # block until every still-running worker thread returns. On the
        # failure path we want to bound that wait — the SIGTERM may not
        # land instantly, but we don't want to hang the operator's
        # shell for hours while D-state children unwind.
        ex = ThreadPoolExecutor(max_workers=effective_parallelism)
        future_to_seed = {ex.submit(_run_one_restart, s): s for s in seeds}
        try:
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    per_restart_results.append(future.result())
                except Exception as exc:
                    # Cancel pending + signal running futures on first
                    # failure. `cancel_futures=True` on shutdown(wait=False)
                    # marks not-yet-started workers cancelled; running
                    # workers receive SIGTERM via _cancel_inflight.
                    other_seeds = [
                        other_seed
                        for other_future, other_seed in future_to_seed.items()
                        if not other_future.done()
                    ]
                    _cancel_inflight(other_seeds)
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise PanelCacheError(
                        f"build_panel_cache: parallel restart seed={seed} "
                        f"failed: {exc}",
                    ) from exc
        finally:
            # NB: pass wait=False here too. On the success path every
            # worker has already returned (as_completed only yields done
            # futures), so wait=False is a no-op. On the exception path
            # we've already issued shutdown(wait=False) above; a second
            # shutdown(wait=True) WOULD still join lingering worker
            # threads — defeating the bounded-cancel intent. The
            # background threads that haven't honored SIGTERM yet will
            # be reaped at interpreter exit via the daemon-thread default.
            ex.shutdown(wait=False, cancel_futures=True)
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
        raise PanelCacheError(
            "build_panel_cache: no restart produced a parseable "
            "loglikelihood; check build_logs/",
        )
    best = max(with_ll, key=lambda r: cast(float, r["ll"]))
    best_ll: float = cast(float, best["ll"])
    logger.info(
        "build_panel_cache: best restart seed=%d (LL=%.6e)",
        best["seed"], best_ll,
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
            raise PanelCacheError(
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
    best_p_dest = cache_dir / f"panel.{k}.P"
    best_q_dest = cache_dir / f"panel.{k}.Q"
    shutil.copy2(best["p_path"], best_p_dest)
    shutil.copy2(best["q_path"], best_q_dest)
    # Also copy panel.bim for projection-time variant alignment
    shutil.copy2(panel_bim_path, cache_dir / "panel.bim")

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

    # Write manifest LAST — its presence is the "cache is valid"
    # signal. Use a tempfile + os.replace pair so the write is atomic:
    # a SIGKILL / power-loss / OS crash mid-write leaves either the
    # complete prior manifest (if one existed) or no manifest at all,
    # never a half-written JSON that load_cache_manifest reads as
    # "cache present but corrupt".
    manifest = PanelCacheManifest(
        schema_version=1,
        track=track,
        continent=continent,
        panel_id=panel_id,
        panel_version=panel_version,
        panel_bim_sha256=panel_bim_sha,
        panel_pop_sha256=panel_pop_sha,
        clusters_yaml_sha256=clusters_yaml_sha,
        k=k,
        admixture_version=admixture_version,
        seeds_used=seeds,
        best_seed=best["seed"],
        best_loglikelihood=best_ll,
        restart_sd_max=restart_sd_max,
        cluster_order=cluster_order,
        geo_filter_yaml_shas=geo_shas,
        pgen_samplebind_version=pgen_samplebind_version,
        build_wallclock_seconds=time.time() - t0,
        build_timestamp=datetime.now(UTC),
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=manifest_path.parent,
        prefix=".manifest-",
        suffix=".json.tmp",
        delete=False,
    ) as tmp:
        tmp.write(manifest.model_dump_json(indent=2))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, manifest_path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
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
    admixture_runner: ToolRunner,
    per_restart_timeout_seconds: int,
    pid_callback: Callable[[int], None] | None = None,
    allow_log_scan_fallback: bool = True,
    argv_prefix: list[str] | None = None,
) -> _RestartResult:
    """Run one supervised-ADMIXTURE restart at the given seed.

    Stages a private restart_dir (cache_dir/build_restart_<seed>/),
    copies panel.bed/bim/fam + panel.pop into it (so concurrent
    restarts don't clobber each other's output files), runs
    ``admixture --supervised -j<threads> -s<seed> panel.bed K``, and
    returns the per-restart result dict for the multimodality + best-LL
    selection downstream.

    Safe to call concurrently with different seeds: each restart has
    its own isolated working directory + log file.

    ``allow_log_scan_fallback``: only True under sequential execution
    — under parallel execution the snapshot-diff heuristic would race
    with sibling workers and could misattribute logs across seeds, so
    the caller (``build_panel_cache``) sets this False whenever
    ``effective_parallelism > 1``.
    """

    restart_dir = cache_dir / f"build_restart_{seed}"
    restart_dir.mkdir(exist_ok=True)

    # ADMIXTURE writes <bfile_stem>.<K>.{P,Q} in cwd. Stage the input
    # BED triplet + .pop file in restart_dir so the outputs land
    # alongside without clobbering between concurrent seeds.
    #
    # The .bed/.bim/.fam are linked (not copied) so concurrent restarts
    # share a single inode for the input — the OS page cache then
    # serves all N processes from one buffered copy of panel.bed
    # (~4-5 GiB for a regional panel × N restarts saved + meaningful
    # DRAM-bandwidth relief when N>=3).
    for suffix in (".bed", ".bim", ".fam"):
        src = panel_bed.with_suffix(suffix).resolve()
        dst = restart_dir / f"panel{suffix}"
        # Three cases to keep in lockstep with the current src:
        # 1) stale symlink (dangling target or pointing elsewhere)
        # 2) stale real file (left over from a v0.x cache that used
        #    shutil.copy2 instead of symlinks)
        # 3) up-to-date symlink → keep
        if dst.is_symlink():
            try:
                current_target = dst.resolve(strict=True)
            except (FileNotFoundError, OSError):
                current_target = None
            if current_target != src:
                dst.unlink()
                os.symlink(src, dst)
        elif dst.exists():
            # Real-file leftover (legacy build's shutil.copy2 result).
            # Replace with a symlink so subsequent rebuilds honor the
            # current source and we get OS page-cache dedupe.
            dst.unlink()
            os.symlink(src, dst)
        else:
            os.symlink(src, dst)
    # Keep .pop as a real copy: it's tiny (~33 KB for a 27K-sample
    # panel) and a writable file in restart_dir simplifies one-off
    # debugging without symlink-aware tooling. Always refresh from
    # source so a curator edit to clusters never silently re-trains
    # against a stale pop file from a prior build.
    pop_dst = restart_dir / "panel.pop"
    if pop_dst.exists():
        pop_dst.unlink()
    shutil.copy2(panel_pop_file, pop_dst)

    log_name = f"restart_{seed}.out"
    log_file = log_dir / log_name
    logger.info(
        "build_panel_cache: starting restart seed=%d (K=%d, %d threads)",
        seed, k, threads,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot log_dir contents (path + mtime) so we can find the log
    # even if the runner ignored our `log_name` request. Mtime-aware
    # so we catch the case where the runner overwrites an existing
    # path in place (or where SubprocessToolRunner-style rotation
    # leaves the canonical path with NEW content + a stale `.prev`).
    # This fallback is only safe under sequential execution — under
    # parallel, sibling workers write to the same dir concurrently and
    # the diff cannot be reliably attributed to any one seed. The
    # caller gates this via `allow_log_scan_fallback`.
    pre_call_snapshot: dict[Path, float] = (
        {
            p: p.stat().st_mtime
            for p in log_dir.glob("*") if p.is_file()
        }
        if allow_log_scan_fallback else {}
    )
    restart_t0 = time.time()
    _call_runner(
        admixture_runner,
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
        log_name=log_name,
        pid_callback=pid_callback,
        argv_prefix=argv_prefix,
    )
    restart_elapsed = time.time() - restart_t0

    p_path = restart_dir / f"panel.{k}.P"
    q_path = restart_dir / f"panel.{k}.Q"
    if not p_path.exists() or not q_path.exists():
        raise PanelCacheError(
            f"_run_one_admixture_restart: restart seed={seed} produced "
            f"no output files (looked for {p_path} and {q_path}); see "
            f"log_dir={log_dir}",
        )

    # Locate the log. Canonical path first; under sequential execution
    # fall back to whichever file in log_dir was created or modified
    # during this call (only one possible since no sibling worker can
    # be writing). Skip the fallback under parallel execution
    # (see comment above).
    if not log_file.exists() and allow_log_scan_fallback:
        # A file is "produced by this restart" if EITHER:
        # - it didn't exist pre-call (mtime not in snapshot), OR
        # - it existed but its mtime advanced (runner rewrote it)
        # Filter out rotated `.prev` files — those represent the
        # PREVIOUS attempt's content from a runner-side rotation, not
        # the current restart's log; picking one would silently parse
        # a stale loglikelihood. (See cli.py SubprocessToolRunner for
        # the rotation pattern that produces `.prev` files.)
        new_files: list[Path] = []
        for p in log_dir.glob("*"):
            if not p.is_file() or p.name.endswith(".prev"):
                continue
            prior_mtime = pre_call_snapshot.get(p)
            current_mtime = p.stat().st_mtime
            if prior_mtime is None or current_mtime > prior_mtime:
                new_files.append(p)
        if len(new_files) == 1:
            log_file = new_files[0]
            logger.debug(
                "build_panel_cache: seed=%d runner ignored log_name; "
                "discovered log at %s", seed, log_file,
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
        raise PanelCacheError(
            f"_derive_cluster_order_from_pop_file: {panel_pop_file.name} "
            f"has {len(seen)} distinct non-'-' labels but K={expected_k}. "
            f"Labels: {seen}",
        )
    return seen


def ld_prune_panel(
    *,
    panel_bed: Path,
    output_prefix: Path,
    plink2_runner: ToolRunner,
    window_size: int = 200,
    step_size: int = 25,
    r2_threshold: float = 0.4,
    log_dir: Path,
    timeout_seconds: int = 3600,
    # Deprecated misnomer for `window_size`. The value was always a
    # plink2 --indep-pairwise window in VARIANTS, never kb; honored with
    # a DeprecationWarning when passed so existing callers do not break.
    window_kb: int | None = None,
) -> Path:
    """Apply LD-pruning to a panel BED via plink2 --indep-pairwise.

    LD-pruning serves correctness first and speed second: ADMIXTURE
    assumes approximately unlinked markers, so correlated SNP blocks can
    inflate spurious structure, and LD-pruned SNPs also let ADMIXTURE
    converge in fewer iterations. This matters extra here because the
    resulting allele-frequency matrix P is cached and reused on every
    projection, so any LD-driven bias is frozen in.

    The defaults (200-variant window, 25-variant step, r²<0.4) match the
    dominant recipe in the human ancient-DNA ADMIXTURE literature: a
    survey of methods sections (AADR 1240K and Human Origins panels)
    finds variant-count windows used over kb windows roughly 17:1, with
    200/25/0.4 by far the most common single recipe (the Reich-lab /
    Human Origins house style, e.g. Cardial-LBK 2015
    doi:10.1093/molbev/msv181; Late Neolithic Switzerland 2020
    doi:10.1038/s41467-020-15560-x; Shimao 2025
    doi:10.1038/s41586-025-09799-x). The ADMIXTURE-manual recipe
    (50/10/0.1) is the main alternative. On a dense ~1.1M-SNP 1240K panel
    this retains roughly 450-600K SNPs. NOTE: a bare plink2
    ``--indep-pairwise`` window is a variant count, not kb (see
    ``window_size`` below).

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
        window_size: ``--indep-pairwise`` window size in VARIANTS
            (default 200). A bare plink2 ``--indep-pairwise`` window is a
            variant count, not kb: e.g. "200 25 0.4" is a 200-variant
            window with a 25-variant step. A kb window would need an
            explicit "kb" suffix AND a step of 1 (plink2 rejects a kb
            window with any other step), so this value has never been kb.
            The old ``window_kb`` keyword (a misnomer) is still accepted
            as a deprecated alias for this parameter.
        step_size: --indep-pairwise step size in variants (default 25).
        r2_threshold: --indep-pairwise r² threshold (default 0.4).
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
    # Back-compat: `window_kb` was a misnomer (the value is a variant
    # count, not kb). Map it onto `window_size` with a deprecation warning.
    if window_kb is not None:
        import warnings

        warnings.warn(
            "ld_prune_panel(window_kb=...) is a misnomer: the value is a "
            "plink2 --indep-pairwise window in VARIANTS, not kb. Pass "
            "window_size= instead; window_kb will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=2,
        )
        window_size = window_kb

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    panel_prefix = panel_bed.with_suffix("")

    # Step 1: identify LD-pruned variant subset. Route through
    # _call_runner so log_name (and pid_callback for cancellation) are
    # forwarded to runners that support them — keeps log files
    # collision-free if two ld_prune_panel invocations share the same
    # log_dir (e.g., pruning two panels in parallel).
    prune_tag = output_prefix.name
    _call_runner(
        plink2_runner,
        args=[
            "--bfile", str(panel_prefix),
            "--indep-pairwise",
            str(window_size), str(step_size), str(r2_threshold),
            "--out", str(output_prefix),
        ],
        cwd=output_prefix.parent,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
        log_name=f"ldprune_{prune_tag}_indep.out",
    )

    # output_prefix is a user-supplied stem (not a `.bed` path) — use
    # APPEND semantics, not `Path.with_suffix`'s REPLACE. A prefix
    # like `cohort.v2` would have its `.v2` segment silently stripped
    # by with_suffix, and the existence probe would look for
    # `cohort.prune.in` (wrong) while plink2 actually wrote
    # `cohort.v2.prune.in`. See `_paths.append_suffix`.
    prune_in = append_suffix(output_prefix, ".prune.in")
    if not prune_in.exists():
        raise PanelCacheError(
            f"ld_prune_panel: plink2 --indep-pairwise produced no "
            f"prune.in at {prune_in}; see {log_dir} for the plink2 log",
        )

    # Step 2: extract the pruned subset into a new BED
    _call_runner(
        plink2_runner,
        args=[
            "--bfile", str(panel_prefix),
            "--extract", str(prune_in),
            "--make-bed",
            "--out", str(output_prefix),
        ],
        cwd=output_prefix.parent,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
        log_name=f"ldprune_{prune_tag}_extract.out",
    )

    pruned_bed = append_suffix(output_prefix, ".bed")
    pruned_bim = append_suffix(output_prefix, ".bim")
    missing = [
        append_suffix(output_prefix, s)
        for s in (".bed", ".bim", ".fam")
        if not append_suffix(output_prefix, s).exists()
    ]
    if missing:
        raise PanelCacheError(
            f"ld_prune_panel: plink2 --extract produced an incomplete "
            f"BED triplet at {output_prefix}; missing sibling file(s): "
            f"{[p.name for p in missing]}; see {log_dir} for the plink2 "
            f"log",
        )

    # Diagnostics: count variants before/after for the operator.
    # `panel_bed.with_suffix(".bim")` is correct here because we
    # require panel_bed to carry the `.bed` extension per the
    # docstring contract (it's a path, not a stem) — with_suffix
    # then correctly REPLACES `.bed` with `.bim` even if the stem
    # itself has dots (e.g. `cohort.v2.bed` → `cohort.v2.bim`).
    pre_count = sum(1 for _ in panel_bed.with_suffix(".bim").open())
    post_count = sum(1 for _ in pruned_bim.open())
    logger.info(
        "ld_prune_panel: %s (%d variants) -> %s (%d retained, "
        "%.1f%% kept, %.2f× SNP reduction)",
        panel_bed.name, pre_count, pruned_bed.name, post_count,
        100.0 * post_count / max(pre_count, 1),
        pre_count / max(post_count, 1),
    )
    return pruned_bed


__all__ = ["build_panel_cache", "ld_prune_panel"]
