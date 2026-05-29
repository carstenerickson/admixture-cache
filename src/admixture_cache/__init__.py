"""admixture-cache — precomputed-P supervised-ADMIXTURE projection.

Split the slow supervised-ADMIXTURE training pass (panel-only,
~hours, one-time per panel × K × clusters_yaml combo) out of the
per-target hot path. After building, project a new target's K-vector
in <2 seconds via NumPy SLSQP against the cached P matrix.

Two phases, two APIs:

1. **Panel cache build** (operator-facing, slow):
   - :func:`build_panel_cache` runs stock ADMIXTURE × N restarts via
     an injected ToolRunner, validates multimodality, writes the
     canonical cached P + manifest.

2. **Per-target projection** (consumer-facing, fast):
   - :func:`project_target` aligns target.bed to cached panel.bim
     + axes (via plink2), reads the target as a dosage vector,
     solves for Q via scipy SLSQP under the binomial admixture
     likelihood.

The math is validated to <1e-5 absolute Q-vector match against stock
ADMIXTURE on representative panels (15K samples × 850K SNPs at K=4).
"""

from __future__ import annotations

from admixture_cache._subprocess_runner import SubprocessToolRunner
from admixture_cache.alignment import (
    align_target_to_panel_bim,
    extract_target_dosage_via_plink2,
)
from admixture_cache.builder import build_panel_cache, ld_prune_panel
from admixture_cache.distribution import (
    CacheRelease,
    download_cache,
    list_available_caches,
)
from admixture_cache.errors import PanelCacheError, PopAutomationConfigError
from admixture_cache.io import (
    load_cache_manifest,
    load_cached_p,
    sha256_file,
    verify_cache_matches_current_config,
)
from admixture_cache.manifest import PanelCacheManifest
from admixture_cache.orchestration import project_target
from admixture_cache.projection import (
    ProjectionResult,
    numpy_supervised_projection,
)
from admixture_cache.runner import ToolRunner

__version__ = "1.4.1"

__all__ = [
    # Public API — cache build (slow, one-time)
    "build_panel_cache",
    "ld_prune_panel",  # optional pre-step before build_panel_cache
    # Public API — per-target projection (fast)
    "project_target",
    "numpy_supervised_projection",
    # Public API — alignment + dosage I/O
    "align_target_to_panel_bim",
    "extract_target_dosage_via_plink2",
    # Public API — cache I/O + validation
    "load_cached_p",
    "load_cache_manifest",
    "verify_cache_matches_current_config",
    "sha256_file",
    # Public API — canonical cache distribution
    "download_cache",
    "list_available_caches",
    # Schemas
    "PanelCacheManifest",
    "ProjectionResult",
    "CacheRelease",
    # Error type
    "PanelCacheError",
    # Back-compat alias for the upstream source-of-extraction; kept
    # importable for callers mid-migration. Identical to
    # PanelCacheError; safe to delete once no consumer relies on it.
    "PopAutomationConfigError",
    # Runner Protocol (for consumers' type hints) + reference impl
    "ToolRunner",
    "SubprocessToolRunner",
    # Version
    "__version__",
]
