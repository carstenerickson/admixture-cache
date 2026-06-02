"""Cache load + verification helpers.

Read-side counterparts to :mod:`admixture_cache.builder`: load the
cached P matrix, load + validate the manifest JSON, and check whether
the cache matches the current config (panel SHA, clusters YAML SHA,
K, optional geo-filter YAMLs, optional panel.pop SHA).

Mismatch is reported as ``(False, reason)`` so callers can log the
specific SHA divergence rather than chasing a generic "cache invalid".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from admixture_cache.errors import PanelCacheError
from admixture_cache.manifest import PanelCacheManifest


def load_cached_p(cache_dir: Path, k: int) -> np.ndarray:
    """Load cached panel.<K>.P matrix (M × K text format, ADMIXTURE
    convention)."""
    p_path = cache_dir / f"panel.{k}.P"
    if not p_path.exists():
        raise PanelCacheError(
            f"load_cached_p: cache file missing: {p_path}; "
            f"build it via `admixture_cache.build_panel_cache`.",
        )
    P = np.loadtxt(p_path)
    if P.ndim != 2 or P.shape[1] != k:
        raise PanelCacheError(
            f"load_cached_p: {p_path} has shape {P.shape}; expected "
            f"(M, {k})",
        )
    return P


def load_cache_manifest(cache_dir: Path) -> PanelCacheManifest:
    """Load + validate the cache manifest JSON."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise PanelCacheError(
            f"load_cache_manifest: {manifest_path} missing; cache is "
            f"either incomplete or never built.",
        )
    try:
        return PanelCacheManifest.model_validate_json(manifest_path.read_text())
    except ValidationError as exc:
        raise PanelCacheError(
            f"load_cache_manifest: {manifest_path} failed schema validation: "
            f"{exc}",
        ) from exc


def verify_cache_matches_current_config(
    *, cache_dir: Path,
    expected_panel_bim_sha256: str,
    expected_clusters_yaml_sha256: str,
    expected_k: int,
    expected_geo_filter_yaml_shas: dict[str, str] | None = None,
    expected_panel_pop_sha256: str | None = None,
) -> tuple[bool, str]:
    """Check whether cache_dir's manifest matches the current config.

    Returns (matched, reason). If matched is False, the reason string
    explains which SHA diverged (for actionable error messages /
    rebuild script logging).

    ``expected_panel_pop_sha256`` is an optional direct guard on the
    supervised-label .pop file. It is compared **only when both** the
    caller supplies it and the cache recorded one — a legacy cache
    (``panel_pop_sha256 is None``) is never invalidated on this basis
    alone, and a caller that passes ``None`` opts out. See the inline
    comment below for why this is lenient where the geo-filter check
    is strict.
    """
    try:
        manifest = load_cache_manifest(cache_dir)
    except PanelCacheError as exc:
        return False, f"cache manifest unloadable: {exc}"

    if manifest.k != expected_k:
        return False, (
            f"K mismatch: cache has K={manifest.k}, current config "
            f"expects K={expected_k}"
        )
    if manifest.panel_bim_sha256 != expected_panel_bim_sha256:
        return False, "panel .bim changed (panel version bump?)"
    # panel.pop direct guard. Unlike the panel .bim / geo-filter checks
    # this is LENIENT on None: a cache built before panel_pop_sha256
    # existed records None, and we decline to force a (potentially
    # many-hour) rebuild of every legacy cache for a defense-in-depth
    # check it never opted into. We flag a mismatch only when BOTH the
    # cache pinned a sha AND the caller supplied one — exactly the case
    # this targets: a cache built WITH the field whose panel.pop was
    # later edited off-pipeline while panel.bim / clusters / K / geo all
    # stayed put. (The geo-filter check below is stricter because a
    # missing geo pin is itself a config signal — "built without that
    # filter" — whereas a missing panel.pop sha only means "predates the
    # field", not "no labels".)
    if (
        expected_panel_pop_sha256 is not None
        and manifest.panel_pop_sha256 is not None
        and manifest.panel_pop_sha256 != expected_panel_pop_sha256
    ):
        return False, "panel.pop changed (supervised labels edited?)"
    if manifest.clusters_yaml_sha256 != expected_clusters_yaml_sha256:
        return False, "clusters_yaml changed (curator edit?)"
    # Geo-filter SHA comparison is symmetric: both directions must
    # agree. A caller that omits an expected dict while the cache has
    # pins (or vice versa) is a real mismatch — silently treating it
    # as "match" lets stale-config caches escape detection.
    expected_geo = expected_geo_filter_yaml_shas or {}
    cached_geo = manifest.geo_filter_yaml_shas or {}
    all_names = set(expected_geo) | set(cached_geo)
    for yaml_name in sorted(all_names):
        expected_sha = expected_geo.get(yaml_name)
        cached_sha = cached_geo.get(yaml_name)
        if expected_sha != cached_sha:
            if expected_sha is None:
                return False, (
                    f"geo-filter YAML {yaml_name!r} pinned in cache "
                    f"({cached_sha[:8] if cached_sha else '?'}) but not "
                    f"supplied to verify"
                )
            if cached_sha is None:
                return False, (
                    f"geo-filter YAML {yaml_name!r} supplied to verify "
                    f"({expected_sha[:8]}) but not pinned in cache"
                )
            return False, (
                f"geo-filter YAML {yaml_name!r} changed "
                f"({cached_sha[:8]} → {expected_sha[:8]})"
            )
    return True, "match"


def sha256_file(path: Path, *, chunk_size: int = 2**16) -> str:
    """Streaming sha256 of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "load_cache_manifest",
    "load_cached_p",
    "sha256_file",
    "verify_cache_matches_current_config",
]
