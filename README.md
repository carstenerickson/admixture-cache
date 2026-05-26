# admixture-cache

Precomputed-P supervised-ADMIXTURE projection cache. Build the slow
training pass once per panel × K × clusters_yaml combo; project new
targets in ~2 seconds.

## Why this exists

Supervised ADMIXTURE training on a real-world panel takes hours to
days per restart (K=21 regional cache: ~12-14 hr × 5 restarts;
K=4 ancestral_cluster: ~5-7 hr × 5 restarts). For consumer pipelines
serving many users, re-running this training per target is
wasteful — the P matrix is determined almost entirely by the panel,
not the target.

`admixture-cache` splits the supervised-ADMIXTURE workflow into:

1. **Panel cache build** (operator, slow, one-time per panel update):
   stock ADMIXTURE × N restarts → cache best-LL P matrix +
   multimodality SD + manifest.
2. **Per-target projection** (consumer, fast, every run):
   align target.bed to cached panel variants + axes (plink2), load
   dosages, solve for Q via scipy SLSQP under the standard binomial
   admixture likelihood.

The projection math matches stock ADMIXTURE Q values to within
~1e-5 absolute on real workloads (15K × 850K matrix at K=4).

## Quickstart

```python
from pathlib import Path
from admixture_cache import (
    build_panel_cache, project_target, PanelCacheManifest,
)

# One-time, slow (~hours per restart per cache)
manifest = build_panel_cache(
    panel_bed=Path("panel.bed"),
    panel_pop_file=Path("panel.pop"),
    clusters_yaml=Path("clusters.yaml"),
    k=21,
    cache_dir=Path("data/regional_k21_cache/"),
    admixture_runner=my_tool_runner,  # callable with .run() method
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
print(result.target_q)              # K-vector
print(result.cluster_order)         # K names
print(result.panel_stability_max_sd)  # cached panel restart_sd
```

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

Cache validity is determined by `manifest.json` SHAs matching the
current config (panel.bim, clusters_yaml, K, optional geo-filter
YAMLs). Any mismatch → consumer code can fall back to a full
ADMIXTURE training pass or rebuild the cache.

## When to use this

- **Multi-user services**: cache once, project for every user
  (~5,000× per-target speedup at scale)
- **Reproducibility**: published canonical caches (forthcoming via
  GitHub Releases) give byte-identical P across consumers
- **CI/CD**: faster integration tests once you have a cache

## When NOT to use this

- **One-time analyses** with a custom panel that won't be reused —
  full ADMIXTURE is simpler
- **Novel methodologies** requiring per-target P refinement — the
  projection assumes P is fully determined by the panel

## Status

- v0.1.0 — initial extraction from
  [ancestry-pipeline](https://github.com/carstenerickson/ancestry-pipeline)'s
  in-pipeline implementation (`admixture_projection.py`,
  ~744 LOC, validated against real-world workloads).
- Roadmap: parallel restart execution, optional LD-pruning flag,
  multi-threaded ADMIXTURE invocation, GitHub-released canonical
  caches.

## License

MIT. See [LICENSE](LICENSE).
