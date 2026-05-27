# admixture-cache

Precomputed-P supervised-ADMIXTURE projection cache. Build the slow training pass once per panel × K × clusters_yaml combo; project new targets in ~2 seconds.

## Why this exists

Supervised ADMIXTURE training on a real-world panel takes hours to days per restart (K=21 regional cache: ~12-14 hr × 5 restarts; K=4 ancestral_cluster: ~5-7 hr × 5 restarts). For consumer pipelines serving many users, re-running this training per target is wasteful — the P matrix is determined almost entirely by the panel, not the target.

`admixture-cache` splits the supervised-ADMIXTURE workflow into:

1. **Panel cache build** (operator, slow, one-time per panel update): stock ADMIXTURE × N restarts → cache best-LL P matrix + multimodality SD + manifest.
2. **Per-target projection** (consumer, fast, every run): align target.bed to cached panel variants + axes (plink2), load dosages, solve for Q via scipy SLSQP under the standard binomial admixture likelihood.

The projection math matches stock ADMIXTURE Q values to within ~1e-5 absolute on representative workloads (15K × 850K matrix at K=4).

## Install

```bash
pip install admixture-cache
```

Python 3.11 through 3.14 are supported. End-to-end paths require **ADMIXTURE** (for `build`) and **plink2** (for `project` / `verify`) on `PATH`. Pure-library use without those binaries is fine — only the build/projection orchestrators shell out.

## Quickstart — library

```python
from pathlib import Path
from admixture_cache import build_panel_cache, project_target

# One-time, slow (~hours per restart per cache)
manifest = build_panel_cache(
    panel_bed=Path("panel.bed"),
    panel_pop_file=Path("panel.pop"),
    clusters_yaml=Path("clusters.yaml"),
    k=21,
    cache_dir=Path("data/regional_k21_cache/"),
    admixture_runner=my_tool_runner,  # see ToolRunner Protocol below
    track="regional",
    panel_id="aadr_v66_ho",
    panel_version="v66.0",
    admixture_version="1.4.0",
    seeds=[1, 2, 3, 4, 5],
    sd_threshold=0.02,
)

# Per-target, fast (~2 seconds end-to-end)
result = project_target(
    target_bed=Path("target.bed"),
    cache_dir=Path("data/regional_k21_cache/"),
    plink2_runner=my_plink2_runner,
    work_dir=Path("scratch/projection/"),
)
print(result.target_q)               # K-vector
print(result.cluster_order)          # K names
print(result.panel_stability_max_sd) # cached panel restart_sd
```

## Quickstart — CLI

Installing the package registers the `admixture-cache` console script with four subcommands:

```bash
# 1. Build a panel cache (slow, one-time).
admixture-cache build \
    --panel-bed panel.bed \
    --panel-pop panel.pop \
    --clusters-yaml clusters.yaml \
    --k 21 \
    --cache-dir data/regional_k21_cache/ \
    --track regional \
    --panel-id aadr_v66_ho \
    --panel-version v66.0 \
    --seeds 1,2,3,4,5

# 2. Project a target against an existing cache (fast).
admixture-cache project \
    --target-bed target.bed \
    --cache-dir data/regional_k21_cache/ \
    --work-dir scratch/projection/

# 3. Check whether a cache matches the current panel/YAML/K config.
admixture-cache verify \
    --panel-bed panel.bed \
    --clusters-yaml clusters.yaml \
    --k 21 \
    --cache-dir data/regional_k21_cache/

# 4. Fetch a canonical published cache from GitHub Releases.
admixture-cache download --list                            # enumerate
admixture-cache download regional_k21_aadr_v66_ho          # install
admixture-cache download regional_k21_aadr_v66_ho \
    --cache-root ~/.admixture-cache/caches \
    --cache-version v2 \
    --force                                                # pin + overwrite
```

Caches install at `<cache-root>/<name>/` (default: `~/.admixture-cache/caches/`, or `$ADMIXTURE_CACHE_ROOT` if set). The downloader streams the tarball, verifies its SHA-256, validates the extracted manifest, and atomically renames into place — partial downloads never leave a half-installed cache.

Publishing your own canonical caches: see [docs/PUBLISH_CACHE.md](docs/PUBLISH_CACHE.md) for the tag convention + tarball format the discovery code expects.

The default `SubprocessToolRunner` runs the local `admixture` / `plink2` binaries on `PATH`; override with `--admixture-binary` / `--plink2-binary` to point at a specific build.

`build`, `project`, and `verify` all surface a non-zero exit code on failure with a descriptive `error: …` line on stderr. `project --json` emits machine-readable JSON instead of human-readable text.

## ToolRunner Protocol

When calling the library from Python (rather than via the CLI), pass any object satisfying the `ToolRunner` Protocol:

```python
from collections.abc import Callable
from pathlib import Path

class MyToolRunner:
    def run(
        self,
        *,
        args: list[str],
        cwd: Path,
        log_dir: Path,
        timeout_seconds: int = 600,
        # The two kwargs below are OPTIONAL but REQUIRED for
        # parallel `build_panel_cache` (max_parallel_restarts > 1):
        log_name: str | None = None,
        pid_callback: Callable[[int], None] | None = None,
    ) -> object:
        ...
```

- `log_name` — admixture-cache passes the per-restart canonical log filename (e.g. `restart_3.out`). Honor it when set; fall back to your own naming scheme when `None`. Required for parallel mode (concurrent restarts share `log_dir` and need disambiguated filenames).
- `pid_callback` — call with the subprocess PID immediately after spawning. admixture-cache uses this to SIGTERM in-flight restarts on first-failure cancellation. Required for parallel mode.
- Spawn subprocesses with `start_new_session=True` so each child gets its own process group. The cancellation path signals the pgid (via `os.killpg`) rather than the bare PID — avoids the classic UNIX PID-recycle race when a subprocess exits between PID capture and the cancellation pass.

Adapters that forward via `**kwargs` (e.g. `def run(self, **kwargs): return self._inner.run(**kwargs)`) are recognized as supporting both extensions — but the inner runner MUST actually honor them. A `**kwargs` forwarder that silently strips unknown kwargs will pass the parallel-mode guard but produce incoherent logs and broken cancellation.

For non-parallel use (`max_parallel_restarts=1`), both extensions are optional — only the four baseline kwargs are required.

## Cache directory layout

After `build_panel_cache` succeeds, `cache_dir` contains:

```
cache_dir/
├── panel.K.P              # Best-LL restart's allele freqs (M × K)
├── panel.K.Q              # Best-LL restart's non-target Q (N × K)
├── panel.bim              # Variant set + REF/ALT axes (alignment ref)
├── restart_sd.json        # Per-cluster SD across restarts
├── cluster_order.json     # K column → cluster name mapping
├── manifest.json          # Panel SHA + YAML SHA + K + version pins
└── build_logs/            # ADMIXTURE stdout/stderr per restart
```

Cache validity is determined by `manifest.json` SHAs matching the current config (panel.bim, clusters_yaml, K, optional geo-filter YAMLs). Any mismatch → consumer code can fall back to a full ADMIXTURE training pass or rebuild the cache.

## When to use this

- **Multi-user services**: cache once, project for every user (~5,000× per-target speedup at scale)
- **Reproducibility**: published canonical caches (forthcoming via GitHub Releases) give byte-identical P across consumers
- **CI/CD**: faster integration tests once you have a cache

## When NOT to use this

- **One-time analyses** with a custom panel that won't be reused — full ADMIXTURE is simpler
- **Novel methodologies** requiring per-target P refinement — the projection assumes P is fully determined by the panel

## Status

- **v1.0.0** — first PyPI release. Library + CLI surface frozen at this point; cache directory layout is stable at schema v1. Tracks numerical parity against stock ADMIXTURE; canonical published-cache artifacts to follow as separate GitHub releases.

See [CHANGELOG.md](CHANGELOG.md) for the per-release detail.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the three local validation gates (pytest / ruff / mypy), commit conventions, and the tag → OIDC PyPI release procedure. See [DEVELOPMENT.md](DEVELOPMENT.md) for the architecture map, design rationale, and module-level walkthroughs.

## Acknowledgments

This library was extracted from [ancestry-pipeline](https://github.com/carstenerickson/ancestry-pipeline)'s in-pipeline supervised-ADMIXTURE projection module (`pop_automation/admixture_projection.py`, ~744 LOC, validated against real-world workloads). The split lets sibling projects depend on the cache layer without pulling in the larger orchestrator.

## License

MIT. See [LICENSE](LICENSE).
