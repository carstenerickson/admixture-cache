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

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class PanelCacheManifest(BaseModel):
    """Manifest written next to cached P + Q + bim. Validated at
    cache-load time; any SHA mismatch triggers cache miss → fall back
    to full run (or rebuild via build script).
    """

    # Forward-compatible across library versions. A cache published by a
    # NEWER admixture-cache (whose manifest carries fields this version does
    # not know about) must still load on an OLDER consumer, so unknown keys
    # are ignored rather than rejected. This is load-bearing for the
    # cache-distribution path (download_cache), where the consumer's library
    # can be older than the builder's: with extra="forbid" every added
    # optional field (panel_pop_sha256, strand_ambiguous_excluded, ...)
    # silently broke older consumers, since schema_version stays 1 and the
    # serialized manifest always includes the new key. Known-field types are
    # still validated, and tarball integrity is covered by the SHA-256 check;
    # the manifest is machine-written, so rejecting unknown keys bought
    # little. Backward compatibility (new code reading old manifests) is
    # unaffected: missing optional fields fall back to their defaults.
    # Unknown keys are not dropped *silently*: _warn_on_unknown_fields logs
    # them so a future field or a typo is at least visible.
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _warn_on_unknown_fields(cls, data: Any) -> Any:
        """Surface unknown manifest keys as a warning before they are
        dropped. ``extra="ignore"`` buys forward compatibility (a manifest
        written by a newer library still loads) but would otherwise discard
        unrecognized keys with no signal at all — so a typo'd or stale field
        name would silently vanish. Logging it keeps the load non-fatal
        while making an unexpected key (a genuine future field OR a mistake)
        visible. Absent optional fields are not "unknown" and never warn."""
        if isinstance(data, dict):
            unknown = sorted(set(data) - set(cls.model_fields))
            if unknown:
                logger.warning(
                    "PanelCacheManifest: ignoring %d unrecognized manifest "
                    "field(s) %s — written by a newer admixture-cache, or a "
                    "typo. Known fields are still validated; unknown keys are "
                    "dropped for forward compatibility (extra='ignore').",
                    len(unknown), unknown,
                )
        return data

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
    # Spread (best minus worst) of the parseable per-restart final
    # loglikelihoods, a headline multimodality signal: ~0 means every
    # restart reached the same optimum, a large value means they did not
    # (SCIENCE.md D4). None when fewer than two restarts produced a
    # parseable loglikelihood (single-restart or legacy build). The full
    # per-seed loglikelihoods live in restart_sd.json. Provenance only;
    # not part of the cache-validity gate. Note the raw spread is in
    # loglikelihood units and so scales with panel size; it is a within-
    # cache diagnostic, not a cross-cache absolute threshold.
    loglikelihood_spread: float | None = None
    # Whether the panel had unlabeled (free-Q) rows, i.e. samples whose Q
    # ADMIXTURE estimated rather than pinning to a label (SCIENCE.md D4).
    # False: fully labeled, so restarts are deterministic (D15) and the
    # restart count / loglikelihood_spread carry no multimodality signal.
    # True: free Q, so seeds_used and loglikelihood_spread are meaningful
    # and 5 restarts may under-sample. None: legacy cache built before this
    # field existed. Provenance only; not part of the cache-validity gate.
    panel_has_free_q: bool | None = None
    cluster_order: list[str]
    geo_filter_yaml_shas: dict[str, str] = Field(default_factory=dict)
    pgen_samplebind_version: str | None = None
    # Whether the build certified the panel free of strand-ambiguous
    # (A/T, C/G) SNPs. True: the default build guard verified none were
    # present, so every projection against this cache is strand-safe by
    # construction. False: the operator opted to keep them
    # (exclude_strand_ambiguous=False); they may be silently
    # strand-inverted for an opposite-strand target (see SCIENCE.md D11).
    # None: legacy cache built before this field existed (the
    # projection-time guard still protects it). Provenance only; not
    # part of the cache-validity gate.
    strand_ambiguous_excluded: bool | None = None
    build_wallclock_seconds: float
    build_timestamp: datetime


__all__ = ["PanelCacheManifest"]
