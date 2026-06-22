"""Unit tests for orchestration-level policy resolution.

The end-to-end `project_target` pipeline is exercised in
tests/integration/test_e2e_build_and_project.py against real binaries;
here we cover the pure decision logic that does not need plink2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from admixture_cache import PanelCacheManifest
from admixture_cache import orchestration as orch
from admixture_cache.orchestration import (
    _resolve_exclude_strand_ambiguous,
    project_target,
)


class TestResolveExcludeStrandAmbiguous:
    """D11: the projection-time strand-ambiguous policy defaults to the
    PROTECTIVE choice (exclude) and only skips work when the build
    certified the panel clean; an explicit caller value always wins."""

    @pytest.mark.parametrize(
        "manifest_decision,expected",
        [
            # None default -> exclude unless the cache is certified clean.
            (None, True),    # legacy cache: may contain them -> exclude
            (True, False),   # build certified clean: nothing to exclude (skip scan)
            (False, True),   # operator kept them at build: panel still has them -> exclude
        ],
    )
    def test_auto_excludes_unless_certified_clean(
        self, manifest_decision: bool | None, expected: bool,
    ) -> None:
        assert (
            _resolve_exclude_strand_ambiguous(None, manifest_decision)
            is expected
        )

    @pytest.mark.parametrize("manifest_decision", [None, True, False])
    def test_explicit_true_overrides_manifest(
        self, manifest_decision: bool | None,
    ) -> None:
        assert (
            _resolve_exclude_strand_ambiguous(True, manifest_decision) is True
        )

    @pytest.mark.parametrize("manifest_decision", [None, True, False])
    def test_explicit_false_overrides_manifest(
        self, manifest_decision: bool | None,
    ) -> None:
        assert (
            _resolve_exclude_strand_ambiguous(False, manifest_decision)
            is False
        )


class _StopPipeline(Exception):
    """Halt project_target right after the align call so the wiring test
    need not mock dosage extraction + P loading downstream."""


# project_target reaches align (which is stubbed) before touching the
# runner, so a placeholder typed as Any satisfies the ToolRunner param.
_unused_runner: Any = object()


def _write_cache_dir(
    tmp_path: Path, *, strand_ambiguous_excluded: bool | None,
) -> Path:
    """Minimal cache dir whose manifest carries a given
    ``strand_ambiguous_excluded`` — enough for project_target to load the
    manifest and reach the align step (which the test stubs)."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "panel.bim").write_text("1\trs1\t0\t1\tA\tC\n")
    manifest = PanelCacheManifest(
        panel_id="p", panel_version="v", panel_bim_sha256="a" * 64,
        clusters_yaml_sha256="b" * 64, k=2, admixture_version="1.4.0",
        seeds_used=[1], best_seed=1, best_loglikelihood=-1.0,
        restart_sd_max=0.0, cluster_order=["c1", "c2"],
        strand_ambiguous_excluded=strand_ambiguous_excluded,
        build_wallclock_seconds=1.0,
        build_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    (cache / "manifest.json").write_text(manifest.model_dump_json())
    return cache


class TestProjectTargetWiresStrandAmbiguous:
    """D11: project_target resolves the strand-ambiguous policy and passes
    it to align_target_to_panel_bim. The plink2 ``--exclude`` construction
    itself is covered in test_alignment; this pins the project_target ->
    align wiring, the seam the policy actually lives at."""

    def _capture_exclude(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_align(**kwargs: Any) -> Path:
            captured["exclude"] = kwargs["exclude_strand_ambiguous"]
            raise _StopPipeline

        monkeypatch.setattr(orch, "align_target_to_panel_bim", fake_align)
        return captured

    @pytest.mark.parametrize(
        "manifest_value,expected_exclude",
        [
            (True, False),   # certified clean -> skip scan
            (False, True),   # kept at build, panel still has them -> exclude
            (None, True),    # legacy -> exclude protectively
        ],
    )
    def test_default_excludes_unless_certified_clean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        manifest_value: bool | None, expected_exclude: bool,
    ) -> None:
        cache = _write_cache_dir(
            tmp_path, strand_ambiguous_excluded=manifest_value,
        )
        captured = self._capture_exclude(monkeypatch)
        with pytest.raises(_StopPipeline):
            project_target(
                target_bed=tmp_path / "target.bed",
                cache_dir=cache,
                plink2_runner=_unused_runner,  # never reached; align is stubbed
                work_dir=tmp_path / "work",
            )
        assert captured["exclude"] is expected_exclude

    def test_explicit_keep_overrides_protective_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Legacy cache (excludes by default) but the caller forces keep:
        # the per-projection opt-out must win over the protective default.
        cache = _write_cache_dir(tmp_path, strand_ambiguous_excluded=None)
        captured = self._capture_exclude(monkeypatch)
        with pytest.raises(_StopPipeline):
            project_target(
                target_bed=tmp_path / "target.bed",
                cache_dir=cache,
                plink2_runner=_unused_runner,
                work_dir=tmp_path / "work",
                exclude_strand_ambiguous=False,
            )
        assert captured["exclude"] is False
