"""PanelCacheManifest schema validation + JSON round-trip + cache
verification helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from admixture_cache import (
    PanelCacheError,
    PanelCacheManifest,
    verify_cache_matches_current_config,
)


def _good_manifest_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "track": "regional",
        "panel_id": "p1",
        "panel_version": "v1",
        "panel_bim_sha256": "a" * 64,
        "clusters_yaml_sha256": "b" * 64,
        "k": 4,
        "admixture_version": "1.4.0",
        "seeds_used": [1, 2, 3],
        "best_seed": 1,
        "best_loglikelihood": -1.0,
        "restart_sd_max": 0.01,
        "cluster_order": ["c1", "c2", "c3", "c4"],
        "build_wallclock_seconds": 10.0,
        "build_timestamp": datetime(2026, 1, 1, tzinfo=UTC),
    }
    base.update(overrides)
    return base


class TestManifestSchema:
    def test_minimal_valid_regional_manifest(self) -> None:
        m = PanelCacheManifest(**_good_manifest_kwargs())  # type: ignore[arg-type]
        assert m.track == "regional"
        assert m.continent is None
        assert isinstance(m.build_timestamp, datetime)

    def test_valid_ancestral_cluster_requires_continent(self) -> None:
        m = PanelCacheManifest(
            **_good_manifest_kwargs(track="ancestral_cluster", continent="Europe"),  # type: ignore[arg-type]
        )
        assert m.continent == "Europe"

    def test_ancestral_cluster_without_continent_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires continent"):
            PanelCacheManifest(
                **_good_manifest_kwargs(track="ancestral_cluster"),  # type: ignore[arg-type]
            )

    def test_non_ancestral_cluster_with_continent_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must have continent=None"):
            PanelCacheManifest(
                **_good_manifest_kwargs(track="regional", continent="Europe"),  # type: ignore[arg-type]
            )

    def test_unknown_track_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not one of"):
            PanelCacheManifest(**_good_manifest_kwargs(track="bogus"))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "track",
        ["regional", "continental_admixture"],
    )
    def test_valid_non_continent_tracks(self, track: str) -> None:
        m = PanelCacheManifest(**_good_manifest_kwargs(track=track))  # type: ignore[arg-type]
        assert m.track == track
        assert m.continent is None

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PanelCacheManifest(**_good_manifest_kwargs(extra_field="nope"))  # type: ignore[arg-type]

    def test_required_fields_enforced(self) -> None:
        kwargs = _good_manifest_kwargs()
        kwargs.pop("panel_id")
        with pytest.raises(ValidationError, match="panel_id"):
            PanelCacheManifest(**kwargs)  # type: ignore[arg-type]

    def test_json_round_trip_preserves_timestamp(self) -> None:
        m = PanelCacheManifest(**_good_manifest_kwargs())  # type: ignore[arg-type]
        text = m.model_dump_json()
        m2 = PanelCacheManifest.model_validate_json(text)
        assert m2.build_timestamp == m.build_timestamp
        assert m2.model_dump() == m.model_dump()

    def test_json_round_trip_preserves_lists_and_dicts(self) -> None:
        m = PanelCacheManifest(
            **_good_manifest_kwargs(
                seeds_used=[1, 5, 9],
                cluster_order=["A", "B", "C", "D"],
                geo_filter_yaml_shas={"foo.yaml": "c" * 64},
            ),  # type: ignore[arg-type]
        )
        m2 = PanelCacheManifest.model_validate_json(m.model_dump_json())
        assert m2.seeds_used == [1, 5, 9]
        assert m2.cluster_order == ["A", "B", "C", "D"]
        assert m2.geo_filter_yaml_shas == {"foo.yaml": "c" * 64}

    def test_geo_filter_yaml_shas_default_empty(self) -> None:
        m = PanelCacheManifest(**_good_manifest_kwargs())  # type: ignore[arg-type]
        assert m.geo_filter_yaml_shas == {}

    def test_pgen_samplebind_version_optional(self) -> None:
        m = PanelCacheManifest(**_good_manifest_kwargs())  # type: ignore[arg-type]
        assert m.pgen_samplebind_version is None
        m2 = PanelCacheManifest(
            **_good_manifest_kwargs(pgen_samplebind_version="0.4.0"),  # type: ignore[arg-type]
        )
        assert m2.pgen_samplebind_version == "0.4.0"


class TestVerifyCacheMatchesCurrentConfig:
    def _write_manifest(self, cache_dir: Path, **overrides: object) -> PanelCacheManifest:
        manifest = PanelCacheManifest(**_good_manifest_kwargs(**overrides))  # type: ignore[arg-type]
        (cache_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        return manifest

    def test_matches_when_shas_identical(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path)
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
        )
        assert matched is True
        assert reason == "match"

    def test_mismatch_when_k_differs(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path)
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=5,
        )
        assert matched is False
        assert "K mismatch" in reason

    def test_mismatch_when_panel_bim_differs(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path)
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="z" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
        )
        assert matched is False
        assert "panel .bim changed" in reason

    def test_mismatch_when_clusters_yaml_differs(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path)
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="z" * 64,
            expected_k=4,
        )
        assert matched is False
        assert "clusters_yaml" in reason

    def test_mismatch_when_geo_filter_yaml_differs(self, tmp_path: Path) -> None:
        self._write_manifest(
            tmp_path,
            geo_filter_yaml_shas={"region.yaml": "c" * 64},
        )
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
            expected_geo_filter_yaml_shas={"region.yaml": "d" * 64},
        )
        assert matched is False
        assert "region.yaml" in reason

    def test_match_when_geo_filter_yaml_unchanged(self, tmp_path: Path) -> None:
        self._write_manifest(
            tmp_path,
            geo_filter_yaml_shas={"region.yaml": "c" * 64},
        )
        matched, _ = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
            expected_geo_filter_yaml_shas={"region.yaml": "c" * 64},
        )
        assert matched is True

    def test_mismatch_when_cache_has_geo_pin_but_caller_omits(self, tmp_path: Path) -> None:
        """The cache pinned a geo-filter YAML; the verify call passed
        None (or an empty dict). The asymmetry was a real bug —
        previously this silently returned `match`."""
        self._write_manifest(
            tmp_path,
            geo_filter_yaml_shas={"region.yaml": "c" * 64},
        )
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
            # No geo-filter dict supplied — must still detect the cache pin.
        )
        assert matched is False
        assert "pinned in cache" in reason
        assert "region.yaml" in reason

    def test_mismatch_when_caller_supplies_geo_pin_not_in_cache(self, tmp_path: Path) -> None:
        """The cache has no geo-filter pin; caller passes one. The
        cache is not safe to reuse for a config that demands a pin
        the cache wasn't built against."""
        self._write_manifest(tmp_path)  # no geo_filter_yaml_shas
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
            expected_geo_filter_yaml_shas={"region.yaml": "d" * 64},
        )
        assert matched is False
        assert "not pinned in cache" in reason

    def test_missing_manifest_reported(self, tmp_path: Path) -> None:
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
        )
        assert matched is False
        assert "unloadable" in reason

    def test_corrupt_manifest_reported(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{not json")
        matched, _ = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
        )
        assert matched is False

    def test_returns_error_type_on_corrupt_load_path(self, tmp_path: Path) -> None:
        """load_cache_manifest raises; verify_cache_matches_current_config wraps."""
        from admixture_cache import load_cache_manifest

        # No file at all
        with pytest.raises(PanelCacheError):
            load_cache_manifest(tmp_path)


class TestManifestRequiredFields:
    @pytest.mark.parametrize(
        "missing",
        [
            "track", "panel_id", "panel_version", "panel_bim_sha256",
            "clusters_yaml_sha256", "k", "admixture_version", "seeds_used",
            "best_seed", "best_loglikelihood", "restart_sd_max",
            "cluster_order", "build_wallclock_seconds", "build_timestamp",
        ],
    )
    def test_missing_required_field_rejected(self, missing: str) -> None:
        kwargs = _good_manifest_kwargs()
        kwargs.pop(missing)
        with pytest.raises(ValidationError):
            PanelCacheManifest(**kwargs)  # type: ignore[arg-type]


class TestManifestSerializationFormat:
    def test_timestamp_serializes_as_iso_string(self) -> None:
        m = PanelCacheManifest(**_good_manifest_kwargs())  # type: ignore[arg-type]
        as_dict = json.loads(m.model_dump_json())
        ts = as_dict["build_timestamp"]
        assert isinstance(ts, str)
        # Should be ISO 8601 with timezone marker
        assert "T" in ts
        assert "2026" in ts


class TestLegacyManifestReparse:
    """v0.x manifests stored ``build_timestamp`` as a string. The v1.0
    schema declares ``datetime`` — pydantic transparently reparses
    ISO-8601 strings into ``datetime``, but the CHANGELOG's claim
    that 'old manifests still load' needs to be regression-tested
    against a hand-written legacy JSON, not just a round-trip of the
    new schema."""

    def _legacy_json(self, **overrides: object) -> str:
        # Build a JSON blob exactly as v0.3.1 would have written it:
        # build_timestamp is a quoted ISO-8601 string.
        payload: dict[str, object] = {
            "schema_version": 1,
            "track": "regional",
            "continent": None,
            "panel_id": "p1",
            "panel_version": "v1",
            "panel_bim_sha256": "a" * 64,
            "clusters_yaml_sha256": "b" * 64,
            "k": 4,
            "admixture_version": "1.4.0",
            "seeds_used": [1, 2, 3, 4, 5],
            "best_seed": 1,
            "best_loglikelihood": -1234567.89,
            "restart_sd_max": 0.0123,
            "cluster_order": ["c1", "c2", "c3", "c4"],
            "geo_filter_yaml_shas": {},
            "pgen_samplebind_version": None,
            "build_wallclock_seconds": 50432.1,
            "build_timestamp": "2026-04-01T12:34:56.789012+00:00",
        }
        payload.update(overrides)
        return json.dumps(payload)

    def test_legacy_iso_string_reparses_to_datetime(self) -> None:
        m = PanelCacheManifest.model_validate_json(self._legacy_json())
        assert isinstance(m.build_timestamp, datetime)
        assert m.build_timestamp == datetime(
            2026, 4, 1, 12, 34, 56, 789012, tzinfo=UTC,
        )

    def test_legacy_regional_with_null_continent_loads(self) -> None:
        """The most common v0.x case: track=regional, continent=None.
        Must continue to pass the new validator."""
        m = PanelCacheManifest.model_validate_json(self._legacy_json())
        assert m.track == "regional"
        assert m.continent is None

    def test_legacy_ancestral_cluster_with_continent_loads(self) -> None:
        m = PanelCacheManifest.model_validate_json(
            self._legacy_json(track="ancestral_cluster", continent="Europe"),
        )
        assert m.track == "ancestral_cluster"
        assert m.continent == "Europe"

    def test_legacy_zulu_z_suffix_iso_reparses(self) -> None:
        """Some ISO writers use `Z` instead of `+00:00` for UTC.
        Verify pydantic accepts both."""
        m = PanelCacheManifest.model_validate_json(
            self._legacy_json(build_timestamp="2026-04-01T12:34:56Z"),
        )
        assert m.build_timestamp.tzinfo is not None
