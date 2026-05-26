"""Console-script entry point for ``admixture-cache``.

Four subcommands:

- ``build`` — run stock supervised ADMIXTURE × N restarts and write a
  panel cache (slow, one-time per panel × K × clusters YAML).
- ``project`` — project one target against an existing cache (fast).
- ``verify`` — check whether a cache matches the current
  panel/YAML/K config; prints the SHA divergence reason on mismatch.
- ``download`` — placeholder; canonical published caches ship after
  the library hits v1.0.

A reference :class:`SubprocessToolRunner` (plain stdlib ``subprocess``)
is wired in by default; consumers can pass their own runner when
calling the library directly from Python.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from admixture_cache import (
    PanelCacheError,
    __version__,
    build_panel_cache,
    load_cache_manifest,
    project_target,
    sha256_file,
    verify_cache_matches_current_config,
)


class SubprocessToolRunner:
    """Default :class:`admixture_cache.ToolRunner` implementation.

    Spawns the binary via ``subprocess.run`` with the given args + cwd,
    captures stdout and stderr to a single log file under ``log_dir``
    named ``<binary>_<short-tag>.out``, and raises
    :class:`PanelCacheError` on non-zero exit or timeout.

    Construct with the absolute path (or name on PATH) of the binary to
    invoke; the same runner instance can be reused across calls.
    """

    def __init__(self, binary: str) -> None:
        self.binary = binary

    def run(
        self,
        *,
        args: list[str],
        cwd: Path,
        log_dir: Path,
        timeout_seconds: int = 600,
        log_name: str | None = None,
        pid_callback: Callable[[int], None] | None = None,
    ) -> object:
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_name is not None:
            log_path = log_dir / log_name
        else:
            log_path = log_dir / self._derive_log_name(args, self.binary)
        # If a prior attempt left a log at this path, rotate it to
        # `.prev` rather than clobbering. NOTE: only the *immediately
        # previous* attempt is preserved — repeated retries overwrite
        # `.prev` each time. For multi-attempt diagnostics, archive
        # `.prev` between retries externally.
        prev_path = log_path.with_suffix(log_path.suffix + ".prev")
        rotated = False
        if log_path.exists():
            log_path.replace(prev_path)
            rotated = True
        cmd = [self.binary, *args]
        try:
            log_file = log_path.open("w")
        except OSError as exc:
            # Restore the rotated file so operators don't lose the
            # last successful run's log to an unrelated open failure.
            if rotated:
                with contextlib.suppress(OSError):
                    prev_path.replace(log_path)
            raise PanelCacheError(
                f"SubprocessToolRunner: cannot open log {log_path}: {exc}",
            ) from exc

        # NB: `start_new_session=True` puts the child in its own
        # process group, so `_cancel_inflight` can signal the group
        # via the same id without racing PID recycle in the parent's
        # pgroup. Runners with their own subprocess management should
        # adopt the same pattern.
        proc: subprocess.Popen[bytes] | None = None
        returncode: int | None = None
        try:
            try:
                proc = subprocess.Popen(
                    cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                # Don't leave an empty log alongside a "binary missing"
                # error — clean up both the (empty) new log AND any
                # rotated .prev that we'd otherwise orphan from the
                # operator's diagnostic surface. Restore prior log on
                # rotation so the actual last-good attempt survives.
                log_file.close()
                log_path.unlink(missing_ok=True)
                if rotated:
                    with contextlib.suppress(OSError):
                        prev_path.replace(log_path)
                raise PanelCacheError(
                    f"SubprocessToolRunner: binary {self.binary!r} not "
                    f"found on PATH or at the given absolute path",
                ) from exc

            if pid_callback is not None:
                try:
                    pid_callback(proc.pid)
                except Exception:
                    # Callback failure must not orphan the subprocess.
                    # Bound the reap (D-state, NFS hang); suppress the
                    # secondary timeout so the operator sees the
                    # ORIGINAL callback exception, not a TimeoutExpired
                    # that obscures it.
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=30)
                    raise
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                # Bound the post-kill wait so a child stuck in
                # uninterruptible-sleep (D-state) doesn't block
                # cleanup forever. After 30 s the OS owns it.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=30)
                raise PanelCacheError(
                    f"SubprocessToolRunner: {self.binary} timed out "
                    f"after {timeout_seconds}s; log at {log_path}",
                ) from exc
        finally:
            # Reap the child if we leave the try-block on any path
            # (including a callback-raised exception). The kill+wait
            # above already attempted reaping for the common cases;
            # this finally is a defense-in-depth against unhandled
            # exception paths between Popen and the explicit kills.
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=30)
            log_file.close()
        if returncode is None or returncode != 0:
            raise PanelCacheError(
                f"SubprocessToolRunner: {self.binary} exited "
                f"{returncode}; log at {log_path}",
            )
        return None

    @staticmethod
    def _derive_log_name(args: list[str], binary: str) -> str:
        """Build a per-call log filename from the args.

        plink2 callers pass `--out <prefix>`; ADMIXTURE callers pass
        `-s<seed>`. Iterate ``enumerate(args)`` so duplicate flags
        produce distinct tags (the previous `args.index(a)` always
        returned the first match).
        """
        tag_parts: list[str] = []
        for i, a in enumerate(args):
            if a.startswith("-s") and a[2:].isdigit():
                tag_parts.append(f"seed{a[2:]}")
            elif a == "--out" and i + 1 < len(args):
                tag_parts.append(Path(args[i + 1]).name)
        tag = "_".join(tag_parts) or "run"
        return f"{Path(binary).name}_{tag}.out"


def _detect_admixture_version(binary: str = "admixture") -> str | None:
    """Return the ADMIXTURE version string from ``admixture --version``,
    or ``None`` if the binary is missing or unparseable."""
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or "") + (out.stderr or "")
    # ADMIXTURE prints "ADMIXTURE Version: 1.3.0" or "ADMIXTURE 1.3.0"
    for line in text.splitlines():
        if "ADMIXTURE" in line or "admixture" in line.lower():
            # Pull a token that looks like x.y[.z]
            for tok in line.replace(":", " ").split():
                if tok and tok[0].isdigit() and "." in tok:
                    return tok
    return None


def _parse_max_parallel_restarts(value: str) -> int | None:
    """argparse type for --max-parallel-restarts. Accepts ``auto`` (or
    the empty string) as a sentinel for "use the library's heuristic"
    and a positive integer otherwise.
    """
    if value.lower() in ("auto", ""):
        return None
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--max-parallel-restarts: expected 'auto' or a positive "
            f"integer, got {value!r}",
        ) from exc
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--max-parallel-restarts: must be >= 1, got {n}",
        )
    return n


def _parse_geo_filter_yamls(values: list[str]) -> dict[str, str]:
    """Parse ``--geo-filter-yaml name:/path/to/file`` repeated args
    into a {name → sha256} dict."""
    out: dict[str, str] = {}
    for entry in values:
        if ":" not in entry:
            raise SystemExit(
                f"--geo-filter-yaml expected 'name:/path/to/file', got {entry!r}"
            )
        name, _, path_str = entry.partition(":")
        if not name:
            raise SystemExit(
                f"--geo-filter-yaml: empty name in {entry!r} "
                f"(expected 'name:/path/to/file' with a non-empty name)",
            )
        if not path_str:
            raise SystemExit(
                f"--geo-filter-yaml: empty path in {entry!r}",
            )
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"--geo-filter-yaml: file not found: {path}")
        out[name] = sha256_file(path)
    return out


def _cmd_build(ns: argparse.Namespace) -> int:
    # Early track/continent validation. Surfacing the inconsistency
    # before launching ADMIXTURE saves ~hours of wasted compute that
    # would otherwise be undone by the manifest's model_validator at
    # write-time.
    if ns.track == "ancestral_cluster" and ns.continent is None:
        print(
            "error: --track=ancestral_cluster requires --continent",
            file=sys.stderr,
        )
        return 2
    if ns.track != "ancestral_cluster" and ns.continent is not None:
        print(
            f"error: --continent is only valid with --track=ancestral_cluster "
            f"(got --track={ns.track})",
            file=sys.stderr,
        )
        return 2

    admixture_version = ns.admixture_version
    if admixture_version is None:
        detected = _detect_admixture_version(ns.admixture_binary)
        if detected is None:
            print(
                "error: could not detect ADMIXTURE version; pass "
                "--admixture-version explicitly",
                file=sys.stderr,
            )
            return 2
        admixture_version = detected

    runner = SubprocessToolRunner(ns.admixture_binary)
    geo_shas: dict[str, str] = (
        _parse_geo_filter_yamls(ns.geo_filter_yaml or []) or {}
    )
    seeds = [int(s) for s in ns.seeds.split(",")] if ns.seeds else None

    manifest = build_panel_cache(
        panel_bed=ns.panel_bed,
        panel_pop_file=ns.panel_pop,
        clusters_yaml=ns.clusters_yaml,
        k=ns.k,
        cache_dir=ns.cache_dir,
        admixture_runner=runner,
        track=ns.track,
        panel_id=ns.panel_id,
        panel_version=ns.panel_version,
        admixture_version=admixture_version,
        continent=ns.continent,
        geo_filter_yaml_shas=geo_shas or None,
        seeds=seeds,
        sd_threshold=ns.sd_threshold,
        threads=ns.threads,
        max_parallel_restarts=ns.max_parallel_restarts,
        per_restart_timeout_seconds=ns.per_restart_timeout_seconds,
    )
    print(f"cache built at {ns.cache_dir} (best_seed={manifest.best_seed}, "
          f"restart_sd_max={manifest.restart_sd_max:.4f})")
    return 0


def _cmd_project(ns: argparse.Namespace) -> int:
    runner = SubprocessToolRunner(ns.plink2_binary)
    result = project_target(
        target_bed=ns.target_bed,
        cache_dir=ns.cache_dir,
        plink2_runner=runner,
        work_dir=ns.work_dir,
    )
    if ns.json:
        payload = {
            "target_q": result.target_q.tolist(),
            "cluster_order": result.cluster_order,
            "panel_stability_max_sd": result.panel_stability_max_sd,
            "n_snps_used": result.n_snps_used,
            "optimization_iterations": result.optimization_iterations,
            "converged": result.converged,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Converged: {result.converged}  iters: {result.optimization_iterations}")
        print(f"Non-missing SNPs used: {result.n_snps_used}")
        print(f"Panel stability max SD: {result.panel_stability_max_sd:.4f}")
        print("Q vector:")
        width = max(len(c) for c in result.cluster_order)
        for cluster, q in zip(result.cluster_order, result.target_q, strict=True):
            print(f"  {cluster:<{width}}  {q:.6f}")
    return 0 if result.converged else 1


def _cmd_verify(ns: argparse.Namespace) -> int:
    panel_bim_path = ns.panel_bed.with_suffix(".bim")
    if not panel_bim_path.exists():
        print(f"error: panel .bim missing at {panel_bim_path}", file=sys.stderr)
        return 2
    expected_panel_bim = sha256_file(panel_bim_path)
    expected_clusters_yaml = sha256_file(ns.clusters_yaml)
    geo_shas = _parse_geo_filter_yamls(ns.geo_filter_yaml or []) or None

    matched, reason = verify_cache_matches_current_config(
        cache_dir=ns.cache_dir,
        expected_panel_bim_sha256=expected_panel_bim,
        expected_clusters_yaml_sha256=expected_clusters_yaml,
        expected_k=ns.k,
        expected_geo_filter_yaml_shas=geo_shas,
    )
    if matched:
        manifest = load_cache_manifest(ns.cache_dir)
        print(f"match: cache at {ns.cache_dir} is current")
        print(f"  built: {manifest.build_timestamp.isoformat()}")
        print(f"  best_seed: {manifest.best_seed}")
        print(f"  restart_sd_max: {manifest.restart_sd_max:.4f}")
        return 0
    print(f"MISMATCH: {reason}", file=sys.stderr)
    return 1


def _cmd_download(ns: argparse.Namespace) -> int:
    print(
        "admixture-cache download: canonical published caches are not yet "
        "available. Track release progress at "
        "https://github.com/carstenerickson/admixture-cache/releases. "
        f"(Requested: {ns.name!r})",
        file=sys.stderr,
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="admixture-cache",
        description="Precomputed-P supervised-ADMIXTURE projection cache.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser(
        "build", help="build a panel cache (slow, one-time).",
    )
    p_build.add_argument("--panel-bed", type=Path, required=True)
    p_build.add_argument("--panel-pop", type=Path, required=True,
                         help="ADMIXTURE-format .pop file labeling panel samples")
    p_build.add_argument("--clusters-yaml", type=Path, required=True)
    p_build.add_argument("--k", type=int, required=True)
    p_build.add_argument("--cache-dir", type=Path, required=True)
    p_build.add_argument(
        "--track", required=True,
        choices=["regional", "continental_admixture", "ancestral_cluster"],
    )
    p_build.add_argument("--panel-id", required=True)
    p_build.add_argument("--panel-version", required=True)
    p_build.add_argument(
        "--continent", default=None,
        help="required when --track=ancestral_cluster",
    )
    p_build.add_argument(
        "--seeds", default=None,
        help="comma-separated seed list (default: 1,2,3,4,5)",
    )
    p_build.add_argument("--sd-threshold", type=float, default=0.02)
    p_build.add_argument("--threads", type=int, default=16)
    p_build.add_argument(
        "--max-parallel-restarts", type=_parse_max_parallel_restarts,
        default=None,
        help="how many ADMIXTURE restarts to run concurrently. Pass "
        "'auto' (or omit) to use the memory-bandwidth-aware heuristic "
        "(cores // (threads*2), capped at len(seeds)); pass a positive "
        "integer to override.",
    )
    p_build.add_argument(
        "--per-restart-timeout-seconds", type=int, default=86400,
    )
    p_build.add_argument(
        "--geo-filter-yaml", action="append", default=[],
        help="repeat as 'name:/path/to/file.yaml'; recorded SHA gates "
        "cache validity",
    )
    p_build.add_argument(
        "--admixture-binary", default="admixture",
        help="path to admixture binary (default: looked up on PATH)",
    )
    p_build.add_argument(
        "--admixture-version", default=None,
        help="override auto-detected ADMIXTURE version string",
    )
    p_build.set_defaults(func=_cmd_build)

    # project
    p_project = sub.add_parser(
        "project", help="project one target against an existing cache (fast).",
    )
    p_project.add_argument("--target-bed", type=Path, required=True)
    p_project.add_argument("--cache-dir", type=Path, required=True)
    p_project.add_argument(
        "--work-dir", type=Path, required=True,
        help="scratch dir for alignment + dosage intermediates",
    )
    p_project.add_argument(
        "--plink2-binary", default="plink2",
        help="path to plink2 binary (default: looked up on PATH)",
    )
    p_project.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON instead of human-readable text",
    )
    p_project.set_defaults(func=_cmd_project)

    # verify
    p_verify = sub.add_parser(
        "verify",
        help="report whether a cache matches the current "
        "panel/YAML/K config.",
    )
    p_verify.add_argument("--panel-bed", type=Path, required=True)
    p_verify.add_argument("--clusters-yaml", type=Path, required=True)
    p_verify.add_argument("--k", type=int, required=True)
    p_verify.add_argument("--cache-dir", type=Path, required=True)
    p_verify.add_argument(
        "--geo-filter-yaml", action="append", default=[],
        help="repeat as 'name:/path/to/file.yaml'",
    )
    p_verify.set_defaults(func=_cmd_verify)

    # download
    p_download = sub.add_parser(
        "download",
        help="placeholder for canonical published caches (post-v1.0).",
    )
    p_download.add_argument("name", help="cache name, e.g. 'regional-k21-aadr-v66-ho'")
    p_download.add_argument("--output-dir", type=Path, default=Path("."))
    p_download.set_defaults(func=_cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, ns.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(ns.func(ns))
    except PanelCacheError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cli() -> None:
    """Entry point referenced by ``[project.scripts]``."""
    sys.exit(main())


if __name__ == "__main__":
    cli()
