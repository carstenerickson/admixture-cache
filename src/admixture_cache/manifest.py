"""Cache manifest schema.

The manifest is the authoritative "this cache is valid" signal — it
records the SHAs of every input that should determine cache validity
(panel .bim, clusters YAML, optional geo-filter YAMLs, K), plus the
provenance fields needed to attribute results (best seed, LL, restart
SD bounds, cluster order, build wallclock, build timestamp).

Written last by :func:`admixture_cache.builder.build_panel_cache`; if
``manifest.json`` is missing the cache is treated as in-progress or
failed and ignored at load time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_VALID_TRACKS = frozenset({"regional", "continental_admixture", "ancestral_cluster"})


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
    geo_filter_yaml_shas: dict[str, str] = Field(default_factory=dict)
    pgen_samplebind_version: str | None = None
    build_wallclock_seconds: float
    build_timestamp: datetime

    @model_validator(mode="after")
    def _validate_track_continent_consistency(self) -> Self:
        if self.track not in _VALID_TRACKS:
            raise ValueError(
                f"track={self.track!r} is not one of {sorted(_VALID_TRACKS)}"
            )
        if self.track == "ancestral_cluster" and self.continent is None:
            raise ValueError(
                "track='ancestral_cluster' requires continent to be set"
            )
        if self.track != "ancestral_cluster" and self.continent is not None:
            raise ValueError(
                f"track={self.track!r} must have continent=None "
                f"(got continent={self.continent!r}); continent is only "
                f"meaningful for ancestral_cluster"
            )
        return self


__all__ = ["PanelCacheManifest"]
