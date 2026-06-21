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

    @pytest.mark.parametrize("track", [
        # Legacy enum values that were enforced pre-v1.4.
        "regional",
        "continental_admixture",
        "ancestral_cluster",
        # Operator-chosen labels.
        "my_polygenic_score_pipeline",
        "anything_goes",
        # Boundary cases — pydantic `str | None` puts no constraint
        # on string contents, so all of these are valid free-text.
        "",                          # empty string
        "x" * 10000,                 # very long
        "日本語のラベル",                # non-ASCII Unicode
        "track-with-hyphens",        # hyphens
        "track with spaces",         # spaces
        "newline\nin\nlabel",        # control chars
        "'; DROP TABLE--",           # SQL-injection-style
        "../../etc/passwd",          # path-traversal-style
    ])
    def test_track_is_free_text_no_validator(self, track: str) -> None:
        """v1.4 dropped the enum constraint on `track`. ANY string is
        accepted; the library doesn't interpret it.

        The parametrize list covers the legacy enum values (for
        back-compat), conventional operator labels, and edge-case
        strings (empty, oversize, Unicode, control chars, SQLi-style,
        path-traversal-style) — none of which should raise. If a
        future regression adds a validator, it'll likely trip on at
        least one of these and fail the test."""
        m = PanelCacheManifest(**_good_manifest_kwargs(track=track))  # type: ignore[arg-type]
        assert m.track == track

    def test_track_optional_defaults_to_none(self) -> None:
        """Consumers that don't care about tagging can omit `track`."""
        kwargs = _good_manifest_kwargs()
        del kwargs["track"]
        m = PanelCacheManifest(**kwargs)  # type: ignore[arg-type]
        assert m.track is None
        assert m.continent is None

    def test_continent_no_longer_coupled_to_track(self) -> None:
        """v1.4 dropped the validator that tied continent to track.
        Any combination of the two free-text labels is valid."""
        combos = [
            ("regional", "Europe"),
            ("ancestral_cluster", None),
            ("regional", None),
            ("ancestral_cluster", "Asia"),
            (None, "Africa"),  # continent without track
            ("custom_label", "custom_continent"),
        ]
        for track, continent in combos:
            m = PanelCacheManifest(
                **_good_manifest_kwargs(track=track, continent=continent),  # type: ignore[arg-type]
            )
            assert m.track == track
            assert m.continent == continent

    def test_unknown_field_ignored_for_forward_compat(self) -> None:
        """A manifest written by a NEWER library version (carrying fields
        this version does not know) must still load: unknown keys are
        ignored, not rejected, so the cache-distribution path stays forward
        compatible. The real load path is `model_validate_json`
        (io.load_cache_manifest), so exercise that, not just kwargs."""
        base = PanelCacheManifest(**_good_manifest_kwargs())  # type: ignore[arg-type]
        blob = json.loads(base.model_dump_json())
        blob["future_field_from_v9"] = {"nested": "whatever"}
        m = PanelCacheManifest.model_validate_json(json.dumps(blob))
        assert not hasattr(m, "future_field_from_v9")
        assert m.panel_id == base.panel_id
        assert m.k == base.k

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

    def test_panel_pop_sha256_optional(self) -> None:
        """Defaults to None (back-compat with caches built before the
        field existed); round-trips when populated."""
        m = PanelCacheManifest(**_good_manifest_kwargs())  # type: ignore[arg-type]
        assert m.panel_pop_sha256 is None
        m2 = PanelCacheManifest(
            **_good_manifest_kwargs(panel_pop_sha256="e" * 64),  # type: ignore[arg-type]
        )
        assert m2.panel_pop_sha256 == "e" * 64
        # Survives a JSON round-trip alongside the other shas.
        m3 = PanelCacheManifest.model_validate_json(m2.model_dump_json())
        assert m3.panel_pop_sha256 == "e" * 64


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

    def test_mismatch_when_panel_pop_sha_differs(self, tmp_path: Path) -> None:
        """Cache pinned a panel.pop sha; caller supplies a different one
        (e.g. labels edited off-pipeline) → mismatch. This is the direct
        label guard the field exists for."""
        self._write_manifest(tmp_path, panel_pop_sha256="e" * 64)
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
            expected_panel_pop_sha256="f" * 64,
        )
        assert matched is False
        assert "panel .pop changed" in reason

    def test_clusters_yaml_reported_before_panel_pop_when_both_differ(
        self, tmp_path: Path,
    ) -> None:
        """When a curator edits clusters_yaml, the downstream panel.pop is
        regenerated, so BOTH shas diverge at once. The reported reason
        must attribute the rebuild to the upstream root cause
        (clusters_yaml), not the downstream panel.pop — i.e. clusters_yaml
        is checked first. Pins the precedence so the diagnostic message
        can't silently regress."""
        self._write_manifest(tmp_path, panel_pop_sha256="e" * 64)
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="DIFFERENT" + "b" * 55,  # 64 chars
            expected_k=4,
            expected_panel_pop_sha256="f" * 64,  # also differs
        )
        assert matched is False
        assert "clusters_yaml changed" in reason
        assert "panel .pop" not in reason

    def test_match_when_panel_pop_sha_identical(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path, panel_pop_sha256="e" * 64)
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
            expected_panel_pop_sha256="e" * 64,
        )
        assert matched is True
        assert reason == "match"

    def test_legacy_cache_without_pop_sha_not_invalidated(
        self, tmp_path: Path,
    ) -> None:
        """A cache built before panel_pop_sha256 existed records None.
        Even when the caller supplies a sha, the cache must NOT be
        invalidated on that basis alone — forcing a (potentially
        many-hour) rebuild of every legacy cache on upgrade is the wrong
        tradeoff for a defense-in-depth check. Leniency on None is the
        whole point of the field being optional."""
        self._write_manifest(tmp_path)  # no panel_pop_sha256 → None
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
            expected_panel_pop_sha256="f" * 64,
        )
        assert matched is True
        assert reason == "match"

    def test_caller_omitting_pop_sha_skips_check(self, tmp_path: Path) -> None:
        """Symmetric opt-out: a cache that pinned a sha is still reusable
        by a caller that doesn't supply one (default None). The guard is
        opt-in from both directions."""
        self._write_manifest(tmp_path, panel_pop_sha256="e" * 64)
        matched, reason = verify_cache_matches_current_config(
            cache_dir=tmp_path,
            expected_panel_bim_sha256="a" * 64,
            expected_clusters_yaml_sha256="b" * 64,
            expected_k=4,
            # expected_panel_pop_sha256 omitted → None
        )
        assert matched is True
        assert reason == "match"

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
        # `track` removed in v1.4 — now optional free-text provenance.
        # `continent` was already optional.
        [
            "panel_id", "panel_version", "panel_bim_sha256",
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

    def test_legacy_manifest_without_panel_pop_sha_loads_as_none(self) -> None:
        """The pre-field manifest JSON (no panel_pop_sha256 key) must
        load and surface the field as None so the verify leniency path
        engages. Absent optional fields fall back to their default
        regardless of the extra= policy (the model is now extra='ignore';
        forward-compat for unknown keys is covered by
        test_unknown_field_ignored_for_forward_compat)."""
        m = PanelCacheManifest.model_validate_json(self._legacy_json())
        assert m.panel_pop_sha256 is None

    def test_legacy_zulu_z_suffix_iso_reparses(self) -> None:
        """Some ISO writers use `Z` instead of `+00:00` for UTC.
        Verify pydantic accepts both."""
        m = PanelCacheManifest.model_validate_json(
            self._legacy_json(build_timestamp="2026-04-01T12:34:56Z"),
        )
        assert m.build_timestamp.tzinfo is not None
