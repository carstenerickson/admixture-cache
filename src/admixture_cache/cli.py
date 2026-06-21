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
import json
import logging
import subprocess
import sys
from pathlib import Path

from admixture_cache import __version__
from admixture_cache._subprocess_runner import SubprocessToolRunner
from admixture_cache.builder import build_panel_cache
from admixture_cache.errors import PanelCacheError
from admixture_cache.io import (
    load_cache_manifest,
    sha256_file,
    verify_cache_matches_current_config,
)
from admixture_cache.orchestration import project_target


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
    # Pre-v1.4 had an early track/continent validation block here
    # mirroring the library's model_validator. Both are gone in v1.4:
    # `track` and `continent` are free-text provenance labels, the
    # library doesn't interpret them, and the CLI shouldn't either.
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
        pgen_samplebind_version=ns.pgen_samplebind_version,
        seeds=seeds,
        sd_threshold=ns.sd_threshold,
        exclude_strand_ambiguous=not ns.keep_strand_ambiguous,
        threads=ns.threads,
        max_parallel_restarts=ns.max_parallel_restarts,
        numa_node_per_restart=ns.numa_node_per_restart,
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
        exclude_strand_ambiguous=not ns.keep_strand_ambiguous,
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
    from admixture_cache.distribution import (
        CacheRelease,
        download_cache,
        list_available_caches,
    )

    if ns.list_caches:
        try:
            releases = list_available_caches(
                github_repo=ns.github_repo,
            )
        except PanelCacheError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not releases:
            print(
                f"No published caches at {ns.github_repo}",
                file=sys.stderr,
            )
            return 0
        # Group by name, list versions newest-first.
        by_name: dict[str, list[CacheRelease]] = {}
        for r in releases:
            by_name.setdefault(r.name, []).append(r)
        for name in sorted(by_name):
            entries = sorted(
                by_name[name],
                key=lambda r: r.version_number,
                reverse=True,
            )
            latest = entries[0]
            other_versions = [r.version for r in entries[1:]]
            mb = latest.size_bytes / (1024 * 1024)
            extra = (
                f" (also: {', '.join(other_versions)})"
                if other_versions else ""
            )
            print(
                f"{name}  {latest.version}  "
                f"{mb:.1f} MB  "
                f"{latest.published_at.date()}{extra}",
            )
            print(f"    {latest.html_url}")
        return 0

    if not ns.name:
        print(
            "error: cache name required (or pass --list to enumerate)",
            file=sys.stderr,
        )
        return 2

    def _progress_bar(downloaded: int, total: int) -> None:
        # Emit a single overwriting line to stderr.
        if total > 0:
            # Clamp to 100% — servers occasionally under-report
            # Content-Length, which would otherwise produce a
            # confusing "120.0%" display. SHA verification still
            # catches the actual integrity question.
            pct = min(100.0, 100.0 * downloaded / total)
            mb_now = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            sys.stderr.write(
                f"\r  {mb_now:6.1f} / {mb_total:6.1f} MB  {pct:5.1f}%",
            )
        else:
            mb_now = downloaded / (1024 * 1024)
            sys.stderr.write(f"\r  {mb_now:6.1f} MB downloaded")
        sys.stderr.flush()

    try:
        target = download_cache(
            ns.name,
            cache_root=ns.cache_root,
            github_repo=ns.github_repo,
            version=ns.cache_version,
            force=ns.force,
            progress=_progress_bar if not ns.quiet else None,
        )
    except PanelCacheError as exc:
        if not ns.quiet:
            sys.stderr.write("\n")  # newline after progress bar
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not ns.quiet:
        sys.stderr.write("\n")
    print(f"Installed {ns.name} → {target}")
    return 0


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
        "--track", default=None,
        help=(
            "free-text provenance label (e.g. 'regional', "
            "'continental_admixture', 'ancestral_cluster', or any "
            "string). Stored in the manifest, not interpreted by "
            "the library."
        ),
    )
    p_build.add_argument("--panel-id", required=True)
    p_build.add_argument("--panel-version", required=True)
    p_build.add_argument(
        "--continent", default=None,
        help=(
            "free-text provenance label paired with --track for "
            "consumers that want finer-grained categorization. "
            "Stored in the manifest, not interpreted."
        ),
    )
    p_build.add_argument(
        "--seeds", default=None,
        help="comma-separated seed list (default: 1,2,3,4,5)",
    )
    p_build.add_argument("--sd-threshold", type=float, default=0.02)
    p_build.add_argument(
        "--keep-strand-ambiguous", action="store_true",
        help="keep strand-ambiguous (A/T, C/G) SNPs instead of refusing "
        "to build from a panel that contains them. NOT recommended: such "
        "SNPs can be silently strand-inverted at projection time "
        "(SCIENCE.md D11). By default the build refuses an ambiguous "
        "panel; clean it first with strip_strand_ambiguous_snps.",
    )
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
        "--numa-node-per-restart", action="store_true",
        help="pin each parallel restart's memory to a distinct NUMA "
        "node via `numactl --membind=N` (Linux + multi-socket + "
        "numactl on PATH). +10-30%% on n2-standard-32+ class hardware; "
        "no-op on single-socket boxes (logs a warning).",
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
    p_build.add_argument(
        "--pgen-samplebind-version", default=None,
        help="optional version pin for callers that pre-process the "
        "panel via pgen-samplebind; recorded on the manifest.",
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
    p_project.add_argument(
        "--keep-strand-ambiguous", action="store_true",
        help="keep strand-ambiguous (A/T, C/G) panel SNPs in this "
        "projection. By default they are excluded because they cannot be "
        "safely REF/ALT-harmonized and are silently strand-inverted for "
        "an opposite-strand target (SCIENCE.md D11). Only pass this when "
        "the target shares the panel's strand convention.",
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
        help="Fetch a canonical published cache from GitHub Releases.",
        description=(
            "Download a published cache to a local cache root, with "
            "streaming SHA-256 verification. Caches install at "
            "<cache-root>/<name>/ ready to pass as `cache_dir=...` to "
            "`project_target`. See docs/PUBLISH_CACHE.md for the "
            "release format if you want to publish your own caches."
        ),
    )
    p_download.add_argument(
        "name", nargs="?",
        help=(
            "cache name as published, e.g. 'regional_k21_aadr_v66_ho'. "
            "Omit when passing --list."
        ),
    )
    p_download.add_argument(
        "--list", dest="list_caches", action="store_true",
        help="list available canonical caches and exit",
    )
    p_download.add_argument(
        "--cache-root", type=Path, default=None,
        help=(
            "where to install. Defaults to $ADMIXTURE_CACHE_ROOT, or "
            "~/.admixture-cache/caches/ if unset."
        ),
    )
    p_download.add_argument(
        "--github-repo",
        default="carstenerickson/admixture-cache",
        help="owner/repo to query for releases (default: %(default)s)",
    )
    p_download.add_argument(
        "--cache-version", default=None,
        help=(
            "specific version to install (e.g. 'v2'). Defaults to "
            "the latest published version."
        ),
    )
    p_download.add_argument(
        "--force", action="store_true",
        help="overwrite an existing cache at the target path",
    )
    p_download.add_argument(
        "--quiet", action="store_true",
        help="suppress the streaming progress display",
    )
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
