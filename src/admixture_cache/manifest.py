"""Cache manifest schema.

The manifest is the authoritative "this cache is valid" signal — it
records the SHAs of every input that should determine cache validity
(panel .bim, panel .pop, clusters YAML, optional geo-filter YAMLs, K),
plus the provenance fields needed to attribute results (best seed, LL,
restart SD bounds, cluster order, build wallclock, build timestamp).

Written last by :func:`admixture_cache.builder.build_panel_cache`; if
``manifest.json`` is missing the cache is treated as in-progress or
failed and ignored at load time.

Free-text provenance fields
---------------------------

``track`` and ``continent`` are free-text labels the library
**stores but does not interpret**. They exist so an operator
browsing a `cache_root/` directory months later can tell which
cache was built for which use case (e.g. ``track="regional"``,
``track="continental_admixture"``, ``track="my_polygenic_score_pipeline"``).
The library doesn't enforce an enum and doesn't enforce any
combination of the two — consumers should attach whatever semantics
they need at their own boundary.

Pre-v1.4 versions of this library enforced
``track ∈ {regional, continental_admixture, ancestral_cluster}``
plus a continent-required-only-for-ancestral_cluster rule. Those
constraints encoded a specific consumer's vocabulary (ancestry-pipeline's
three tracks) into the library schema; they're gone in v1.4. Existing
caches using those exact labels still load unchanged.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PanelCacheManifest(BaseModel):
    """Manifest written next to cached P + Q + bim. Validated at
    cache-load time; any SHA mismatch triggers cache miss → fall back
    to full run (or rebuild via build script).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    # Free-text provenance label (e.g. "regional", "continental_admixture",
    # "ancestral_cluster", or anything else the operator wants). Not
    # interpreted by the library; not part of the cache-validity gate.
    # Defaults to None so consumers that don't care about tagging can
    # omit it.
    track: str | None = None
    # Free-text label paired with `track` for consumers that need
    # finer-grained categorization (e.g. per-continent sub-models).
    # Same provenance-only semantics as `track`.
    continent: str | None = None
    panel_id: str
    panel_version: str
    panel_bim_sha256: str
    # SHA-256 of the supervised-label .pop file the cache was trained
    # against. Optional for back-compat: caches built before this field
    # existed have it as None, and verification skips the comparison
    # rather than forcing a rebuild of every legacy cache (see
    # verify_cache_matches_current_config). New builds always populate
    # it, which makes the builder's idempotency check sensitive to a
    # panel.pop edit that left every other hashed input unchanged.
    panel_pop_sha256: str | None = None
    clusters_yaml_sha256: str
    k: int
    admixture_version: str
    seeds_used: list[int]
    best_seed: int
    best_loglikelihood: float
    restart_sd_max: float
    cluster_order: list[str]
    geo_filter_yaml_shas: dict[str, str] = Field(default_factory=dict)
    pgen_samplebind_version: str | None = None
    build_wallclock_seconds: float
    build_timestamp: datetime


__all__ = ["PanelCacheManifest"]
