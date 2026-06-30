"""Builder tests: idempotency, ADMIXTURE log parsing, ld_prune_panel."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from admixture_cache import (
    PanelCacheError,
    PanelCacheManifest,
    build_panel_cache,
    ld_prune_panel,
    strip_strand_ambiguous_snps,
)
from admixture_cache._paths import append_suffix
from admixture_cache.builder import (
    _auto_max_parallel_restarts,
    _derive_cluster_order_from_pop_file,
    _parse_admixture_loglikelihood,
)
from admixture_cache.io import sha256_file


class TestParseAdmixtureLoglikelihood:
    def test_picks_last_loglikelihood_line(self) -> None:
        text = (
            "Iteration 1\nLoglikelihood: -1.234e6\n"
            "Iteration 2\nLoglikelihood: -1.111e6\n"
            "Iteration 3\nLoglikelihood: -1.000e6\n"
        )
        assert _parse_admixture_loglikelihood(text) == -1.000e6

    def test_no_loglikelihood_returns_none(self) -> None:
        assert _parse_admixture_loglikelihood("nothing here\nplain text\n") is None

    def test_empty_text_returns_none(self) -> None:
        assert _parse_admixture_loglikelihood("") is None

    def test_handles_plain_decimal(self) -> None:
        assert _parse_admixture_loglikelihood("Loglikelihood: -12345.6789\n") == -12345.6789

    def test_handles_positive_value(self) -> None:
        assert _parse_admixture_loglikelihood("Loglikelihood: 1.0\n") == 1.0

    def test_handles_integer_value(self) -> None:
        assert _parse_admixture_loglikelihood("Loglikelihood: -5\n") == -5.0


class _FakeAdmixtureRunner:
    """Fake admixture runner that writes synthetic P/Q files where
    ADMIXTURE would on a real run. Implements the modern Protocol
    (accepts `log_name` + `pid_callback`) so it works in parallel
    mode under the v1.0 guard."""

    def __init__(
        self,
        *,
        k: int,
        n_samples: int,
        n_snps: int,
        seed_to_ll: dict[int, float] | None = None,
    ) -> None:
        self.k = k
        self.n_samples = n_samples
        self.n_snps = n_snps
        # Each seed → final reported loglikelihood
        self.seed_to_ll = seed_to_ll or {}
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        args: list[str],
        cwd: Path,
        log_dir: Path,
        timeout_seconds: int = 86400,
        log_name: str | None = None,
        pid_callback: Any = None,
    ) -> object:
        self.calls.append({
            "args": list(args), "cwd": cwd, "log_name": log_name,
        })
        # Parse -s<seed> and K from args
        seed = None
        for a in args:
            if a.startswith("-s"):
                seed = int(a[2:])
        k = int(args[-1])
        bfile = args[-2]  # "panel.bed"
        bstem = Path(bfile).stem
        # Write synthetic P and Q deterministically per seed
        rng = np.random.default_rng(seed or 0)
        P = rng.uniform(0.05, 0.95, size=(self.n_snps, k))
        Q = rng.dirichlet(alpha=np.ones(k), size=self.n_samples)
        np.savetxt(cwd / f"{bstem}.{k}.P", P)
        np.savetxt(cwd / f"{bstem}.{k}.Q", Q)
        # Emit a fake log under log_dir at whatever name the builder
        # requested (modern runner contract).
        ll = self.seed_to_ll.get(seed or 0, -1.0e6)
        log_filename = log_name or f"restart_{seed}.out"
        (log_dir / log_filename).write_text(
            f"Iteration 1\nLoglikelihood: {ll}\n",
        )
        return None


def _write_panel_triplet(tmp_path: Path, n_samples: int, n_snps: int) -> Path:
    """Create empty .bed/.bim/.fam + .pop files for the panel."""
    bed = tmp_path / "panel.bed"
    bed.write_bytes(b"\x6c\x1b\x01")  # PLINK magic + mode bit
    # Minimal .bim: chrom snp cm pos a1 a2
    bim = tmp_path / "panel.bim"
    lines = []
    for i in range(n_snps):
        lines.append(f"1\trs{i}\t0\t{i+1000}\tA\tG")
    bim.write_text("\n".join(lines) + "\n")
    # Minimal .fam: FID IID PID MID SEX PHENOTYPE
    fam = tmp_path / "panel.fam"
    fam.write_text(
        "\n".join(f"F{i}\tI{i}\t0\t0\t1\t-9" for i in range(n_samples)) + "\n"
    )
    return bed


def _write_pop_file(tmp_path: Path, cluster_labels: list[str]) -> Path:
    pop = tmp_path / "panel.pop"
    pop.write_text("\n".join(cluster_labels) + "\n")
    return pop


def _write_clusters_yaml(tmp_path: Path, content: str = "k: 4\n") -> Path:
    p = tmp_path / "clusters.yaml"
    p.write_text(content)
    return p


class TestBuildPanelCacheIdempotency:
    def test_skip_rebuild_when_manifest_matches(self, tmp_path: Path) -> None:
        """If manifest.json already exists and SHAs match the current
        inputs, build_panel_cache must NOT call the runner."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=5)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Pre-write a manifest matching the current input SHAs
        manifest = PanelCacheManifest(
            track="regional",
            panel_id="p1", panel_version="v1",
            panel_bim_sha256=sha256_file(panel_bed.with_suffix(".bim")),
            clusters_yaml_sha256=sha256_file(yaml),
            k=2,
            admixture_version="1.4.0",
            seeds_used=[1, 2],
            best_seed=1,
            best_loglikelihood=-1.0,
            restart_sd_max=0.0,
            cluster_order=["A", "B"],
            build_wallclock_seconds=1.0,
            build_timestamp=datetime.now(UTC),
        )
        (cache_dir / "manifest.json").write_text(manifest.model_dump_json())

        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=5)
        result = build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1, 2],
            sd_threshold=0.02,
        )
        # Runner was NOT invoked
        assert runner.calls == []
        # Returned manifest matches the pre-existing one
        assert result.best_seed == 1
        assert result.k == 2

    def test_skip_rebuild_backfills_missing_panel_pop(
        self, tmp_path: Path,
    ) -> None:
        """gh #719: a cache that matches config but lacks panel.pop on disk
        (built by <= 1.5.1) is self-healed in place. build_panel_cache copies
        panel.pop in WITHOUT re-running ADMIXTURE, so the runtime validator
        stops rejecting the cache as stale. Mirrors
        test_skip_rebuild_when_manifest_matches with panel.pop absent."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=5)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        manifest = PanelCacheManifest(
            track="regional",
            panel_id="p1", panel_version="v1",
            panel_bim_sha256=sha256_file(panel_bed.with_suffix(".bim")),
            clusters_yaml_sha256=sha256_file(yaml),
            panel_pop_sha256=sha256_file(pop),
            k=2,
            admixture_version="1.4.0",
            seeds_used=[1, 2],
            best_seed=1,
            best_loglikelihood=-1.0,
            restart_sd_max=0.0,
            cluster_order=["A", "B"],
            build_wallclock_seconds=1.0,
            build_timestamp=datetime.now(UTC),
        )
        (cache_dir / "manifest.json").write_text(manifest.model_dump_json())
        # panel.pop deliberately absent from the cache dir (the <= 1.5.1 bug).
        assert not (cache_dir / "panel.pop").exists()

        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=5)
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1, 2],
            sd_threshold=0.02,
        )
        # No rebuild happened (the expensive ADMIXTURE pass was skipped) ...
        assert runner.calls == []
        # ... but panel.pop is now present and byte-identical to the source.
        cached_pop = cache_dir / "panel.pop"
        assert cached_pop.is_file()
        assert cached_pop.read_bytes() == pop.read_bytes()

    def test_rebuild_when_panel_bim_sha_changed(self, tmp_path: Path) -> None:
        """Stale cache (different panel_bim_sha) → rebuild."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=5)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Manifest with WRONG panel_bim_sha
        manifest = PanelCacheManifest(
            track="regional",
            panel_id="p1", panel_version="v1",
            panel_bim_sha256="z" * 64,  # not matching
            clusters_yaml_sha256=sha256_file(yaml),
            k=2,
            admixture_version="1.4.0",
            seeds_used=[1],
            best_seed=1, best_loglikelihood=-1.0, restart_sd_max=0.0,
            cluster_order=["A", "B"],
            build_wallclock_seconds=1.0,
            build_timestamp=datetime.now(UTC),
        )
        (cache_dir / "manifest.json").write_text(manifest.model_dump_json())

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=5, seed_to_ll={1: -100.0},
        )
        result = build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=1.0,  # accept any SD (single restart anyway)
        )
        assert len(runner.calls) == 1
        # Manifest has the correct SHA now
        assert result.panel_bim_sha256 == sha256_file(panel_bed.with_suffix(".bim"))

    def test_rebuild_when_panel_pop_sha_changed(self, tmp_path: Path) -> None:
        """Stale cache (panel.pop edited, every other hashed input
        unchanged) → rebuild. This is the off-pipeline-label-edit case
        the panel_pop_sha256 guard targets."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=5)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Correct bim + clusters shas but a WRONG panel_pop_sha256 —
        # isolates panel.pop as the sole rebuild trigger.
        manifest = PanelCacheManifest(
            track="regional",
            panel_id="p1", panel_version="v1",
            panel_bim_sha256=sha256_file(panel_bed.with_suffix(".bim")),
            panel_pop_sha256="z" * 64,  # not matching the real pop file
            clusters_yaml_sha256=sha256_file(yaml),
            k=2,
            admixture_version="1.4.0",
            seeds_used=[1],
            best_seed=1, best_loglikelihood=-1.0, restart_sd_max=0.0,
            cluster_order=["A", "B"],
            build_wallclock_seconds=1.0,
            build_timestamp=datetime.now(UTC),
        )
        (cache_dir / "manifest.json").write_text(manifest.model_dump_json())

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=5, seed_to_ll={1: -100.0},
        )
        result = build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=1.0,
        )
        assert len(runner.calls) == 1  # rebuilt
        assert result.panel_pop_sha256 == sha256_file(pop)

    def test_skip_rebuild_when_pop_sha_matches(self, tmp_path: Path) -> None:
        """Manifest pins the CORRECT panel.pop sha (plus all other
        inputs) → no rebuild. Exercises the populated-sha match path,
        not just the legacy-None leniency."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=5)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        manifest = PanelCacheManifest(
            track="regional",
            panel_id="p1", panel_version="v1",
            panel_bim_sha256=sha256_file(panel_bed.with_suffix(".bim")),
            panel_pop_sha256=sha256_file(pop),
            clusters_yaml_sha256=sha256_file(yaml),
            k=2,
            admixture_version="1.4.0",
            seeds_used=[1],
            best_seed=1, best_loglikelihood=-1.0, restart_sd_max=0.0,
            cluster_order=["A", "B"],
            build_wallclock_seconds=1.0,
            build_timestamp=datetime.now(UTC),
        )
        (cache_dir / "manifest.json").write_text(manifest.model_dump_json())

        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=5)
        build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=cache_dir, admixture_runner=runner,
            track="regional", panel_id="p1", panel_version="v1",
            admixture_version="1.4.0", seeds=[1], sd_threshold=1.0,
        )
        assert runner.calls == []  # skipped

    def test_legacy_manifest_without_pop_sha_still_skips(
        self, tmp_path: Path,
    ) -> None:
        """A pre-field manifest (panel_pop_sha256 omitted → None) whose
        other shas match must NOT rebuild on upgrade — the
        leniency-on-None path keeps legacy caches valid."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=5)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        manifest = PanelCacheManifest(  # no panel_pop_sha256 → None
            track="regional",
            panel_id="p1", panel_version="v1",
            panel_bim_sha256=sha256_file(panel_bed.with_suffix(".bim")),
            clusters_yaml_sha256=sha256_file(yaml),
            k=2,
            admixture_version="1.4.0",
            seeds_used=[1],
            best_seed=1, best_loglikelihood=-1.0, restart_sd_max=0.0,
            cluster_order=["A", "B"],
            build_wallclock_seconds=1.0,
            build_timestamp=datetime.now(UTC),
        )
        assert manifest.panel_pop_sha256 is None
        (cache_dir / "manifest.json").write_text(manifest.model_dump_json())

        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=5)
        build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=cache_dir, admixture_runner=runner,
            track="regional", panel_id="p1", panel_version="v1",
            admixture_version="1.4.0", seeds=[1], sd_threshold=1.0,
        )
        assert runner.calls == []  # legacy cache preserved, no rebuild

    def test_first_build_records_panel_pop_sha(self, tmp_path: Path) -> None:
        """A fresh build pins the sha of the supervised-label file it
        trained against, both on the returned object and on disk."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10, seed_to_ll={1: -100.0},
        )
        manifest = build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=cache_dir, admixture_runner=runner,
            track="regional", panel_id="p1", panel_version="v1",
            admixture_version="1.4.0", seeds=[1], sd_threshold=10.0,
        )
        assert manifest.panel_pop_sha256 == sha256_file(pop)
        reloaded = PanelCacheManifest.model_validate_json(
            (cache_dir / "manifest.json").read_text(),
        )
        assert reloaded.panel_pop_sha256 == sha256_file(pop)

    def test_first_build_runs_seeds_in_order(self, tmp_path: Path) -> None:
        """No existing cache → run all N seeds, pick best LL."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10,
            # Seed 3 has the best (highest) LL
            seed_to_ll={1: -200.0, 2: -150.0, 3: -100.0},
        )
        manifest = build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1, 2, 3],
            sd_threshold=10.0,  # tolerant for synthetic Q
        )
        assert manifest.best_seed == 3
        assert manifest.best_loglikelihood == -100.0
        assert manifest.seeds_used == [1, 2, 3]
        assert manifest.cluster_order == ["A", "B"]
        # Cache files emitted
        assert (cache_dir / "panel.2.P").exists()
        assert (cache_dir / "panel.2.Q").exists()
        assert (cache_dir / "panel.bim").exists()
        assert (cache_dir / "manifest.json").exists()
        assert (cache_dir / "restart_sd.json").exists()
        assert (cache_dir / "cluster_order.json").exists()

    def test_records_per_seed_loglikelihoods_and_spread(
        self, tmp_path: Path,
    ) -> None:
        """SCIENCE.md D4: the build records each restart's final LL, the
        best-minus-worst spread (manifest + restart_sd.json), and whether
        the panel had free Q, so multimodality is visible post-hoc."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])  # fully labeled
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10,
            seed_to_ll={1: -200.0, 2: -150.0, 3: -100.0},
        )
        manifest = build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=cache_dir, admixture_runner=runner,
            track="regional", panel_id="p1", panel_version="v1",
            admixture_version="1.4.0", seeds=[1, 2, 3],
            sd_threshold=10.0,  # tolerant for synthetic Q
        )
        # best (-100) minus worst (-200) = 100
        assert manifest.loglikelihood_spread == 100.0
        sd = json.loads((cache_dir / "restart_sd.json").read_text())
        assert sd["per_seed_loglikelihood"] == {
            "1": -200.0, "2": -150.0, "3": -100.0,
        }
        assert sd["best_seed"] == 3
        assert sd["loglikelihood_spread"] == 100.0
        assert sd["panel_has_free_q"] is False
        # Round-trips through the manifest schema (spread + free-Q flag).
        reloaded = PanelCacheManifest.model_validate_json(
            (cache_dir / "manifest.json").read_text(),
        )
        assert reloaded.loglikelihood_spread == 100.0
        assert reloaded.panel_has_free_q is False

    def test_single_restart_spread_is_none(self, tmp_path: Path) -> None:
        """SCIENCE.md D4: with fewer than two parseable restarts the spread
        is None (not 0.0), so a consumer can tell 'unknown' from 'all
        restarts agreed' (covers the manifest's documented None branch)."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10, seed_to_ll={1: -100.0},
        )
        manifest = build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=cache_dir, admixture_runner=runner,
            track="regional", panel_id="p1", panel_version="v1",
            admixture_version="1.4.0", seeds=[1], sd_threshold=10.0,
        )
        assert manifest.loglikelihood_spread is None
        sd = json.loads((cache_dir / "restart_sd.json").read_text())
        assert sd["loglikelihood_spread"] is None
        assert sd["per_seed_loglikelihood"] == {"1": -100.0}

    def test_free_q_panel_below_floor_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """SCIENCE.md D4: a free-Q panel ('-' rows) built with fewer than
        the recommended restarts emits an advisory warning but still
        produces a valid cache."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        # Two labeled rows (K=2) + two unlabeled '-' rows → free Q.
        pop = _write_pop_file(tmp_path, ["A", "B", "-", "-"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10,
            seed_to_ll={1: -200.0, 2: -150.0, 3: -100.0},
        )
        with caplog.at_level(logging.WARNING, logger="admixture_cache.builder"):
            manifest = build_panel_cache(
                panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
                k=2, cache_dir=cache_dir, admixture_runner=runner,
                track="regional", panel_id="p1", panel_version="v1",
                admixture_version="1.4.0", seeds=[1, 2, 3],
                sd_threshold=10.0,
            )
        assert manifest.k == 2  # cache still built
        assert "unlabeled" in caplog.text
        assert "restart" in caplog.text
        sd = json.loads((cache_dir / "restart_sd.json").read_text())
        assert sd["panel_has_free_q"] is True

    def test_fully_labeled_panel_does_not_warn_on_restart_count(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A fully-labeled panel is deterministic across restarts (D15),
        so the free-Q restart-count warning must NOT fire even with few
        seeds."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])  # no '-' rows
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10,
            seed_to_ll={1: -100.0, 2: -110.0, 3: -120.0},
        )
        with caplog.at_level(logging.WARNING, logger="admixture_cache.builder"):
            build_panel_cache(
                panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
                k=2, cache_dir=cache_dir, admixture_runner=runner,
                track="regional", panel_id="p1", panel_version="v1",
                admixture_version="1.4.0", seeds=[1, 2, 3],
                sd_threshold=10.0,
            )
        assert "unlabeled" not in caplog.text

    def test_derive_cluster_order_counts_unlabeled_rows(
        self, tmp_path: Path,
    ) -> None:
        """The single-pass parser returns the unlabeled-row count using the
        SAME blank-or-'-' predicate it uses to skip non-label rows, so a
        free-Q panel written with blank rows (not '-') is still detected
        (the gh #9 review gap)."""
        # Fully labeled: zero unlabeled.
        labeled = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        order, n_unlabeled = _derive_cluster_order_from_pop_file(
            panel_pop_file=labeled, expected_k=2,
        )
        assert order == ["A", "B"]
        assert n_unlabeled == 0
        # Mixed '-' and blank unlabeled rows: BOTH count as free Q.
        mixed = tmp_path / "mixed.pop"
        mixed.write_text("A\n-\nB\n\n-\n")  # 2 dash rows + 1 blank row
        order2, n_unlabeled2 = _derive_cluster_order_from_pop_file(
            panel_pop_file=mixed, expected_k=2,
        )
        assert order2 == ["A", "B"]
        assert n_unlabeled2 == 3

    def test_multimodality_failure_raises(self, tmp_path: Path) -> None:
        """If per-cluster restart SD exceeds threshold, no manifest is
        written and PanelCacheError is raised."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        # Two seeds produce different random Q matrices → guaranteed SD > 0;
        # threshold of 1e-9 ensures the check fails.
        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=10)
        with pytest.raises(PanelCacheError, match="multimodality detected"):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=runner,
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                sd_threshold=1e-9,
            )
        # Manifest NOT written
        assert not (cache_dir / "manifest.json").exists()

    def test_no_parseable_ll_raises(self, tmp_path: Path) -> None:
        """If every restart's log lacks a Loglikelihood: line, raise."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        class _NoLLRunner(_FakeAdmixtureRunner):
            def run(self, *, args: list[str], cwd: Path, log_dir: Path,
                    timeout_seconds: int = 86400) -> object:
                # Same P/Q output but log lacks Loglikelihood line
                super().run(args=args, cwd=cwd, log_dir=log_dir,
                            timeout_seconds=timeout_seconds)
                for f in log_dir.glob("restart_*.out"):
                    f.write_text("no ll here\n")
                return None

        runner = _NoLLRunner(k=2, n_samples=4, n_snps=10)
        # Pin sequential so the test stays valid on hosts where the
        # auto-heuristic resolves to >1 (e.g., 64-core CI runners). The
        # invariant under test is "no LL parsed → PanelCacheError", not
        # the heuristic's output.
        with pytest.raises(PanelCacheError, match="no restart produced"):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=runner,
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                sd_threshold=10.0,
                max_parallel_restarts=1,
            )

    def test_missing_panel_bim_raises(self, tmp_path: Path) -> None:
        panel_bed = tmp_path / "panel.bed"
        panel_bed.write_bytes(b"\x6c\x1b\x01")
        # No .bim!
        pop = _write_pop_file(tmp_path, ["A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        with pytest.raises(PanelCacheError, match=r"panel \.bim missing"):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=tmp_path / "cache",
                admixture_runner=_FakeAdmixtureRunner(k=2, n_samples=2, n_snps=5),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1],
            )

    def test_missing_panel_pop_raises(self, tmp_path: Path) -> None:
        """A missing .pop is surfaced up front (mirroring the .bim
        guard), not late during restart staging."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=2, n_snps=5)
        missing_pop = tmp_path / "panel.pop"  # never written
        yaml = _write_clusters_yaml(tmp_path)
        with pytest.raises(PanelCacheError, match=r"panel \.pop missing"):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=missing_pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=tmp_path / "cache",
                admixture_runner=_FakeAdmixtureRunner(k=2, n_samples=2, n_snps=5),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1],
            )


class TestRestartStagingSymlinks:
    """Each restart_dir's .bed/.bim/.fam should be symlinks pointing at
    the original panel triplet so the OS page cache dedupes across
    concurrent restart subprocesses. .pop stays a copy."""

    def test_bed_triplet_symlinked_not_copied(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10, seed_to_ll={1: -100.0},
        )
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
        )
        for suffix in (".bed", ".bim", ".fam"):
            staged = cache_dir / "build_restart_1" / f"panel{suffix}"
            assert staged.is_symlink(), f"{staged} should be a symlink"
            # And the symlink target is the source panel file
            target = staged.resolve()
            assert target == panel_bed.with_suffix(suffix).resolve()

    def test_pop_file_remains_real_copy(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10, seed_to_ll={1: -100.0},
        )
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
        )
        staged_pop = cache_dir / "build_restart_1" / "panel.pop"
        assert staged_pop.exists()
        assert not staged_pop.is_symlink()
        assert staged_pop.read_text() == pop.read_text()

    def test_log_name_routed_through_to_runner(self, tmp_path: Path) -> None:
        """Each restart calls the runner with log_name=restart_<seed>.out.
        The fake runner uses this when writing its synthetic log file."""

        class _LogNameAwareRunner(_FakeAdmixtureRunner):
            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
                pid_callback: object = None,
            ) -> object:
                self.calls.append({"args": list(args), "cwd": cwd,
                                   "log_name": log_name})
                # Emit P/Q like the base runner
                seed = None
                for a in args:
                    if a.startswith("-s"):
                        seed = int(a[2:])
                k = int(args[-1])
                rng = np.random.default_rng(seed or 0)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(self.n_snps, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=self.n_samples))
                # Use log_name if given (the path the library asked for)
                log_path = log_dir / (log_name or "fallback.out")
                log_path.write_text("Loglikelihood: -1.0\n")
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _LogNameAwareRunner(k=2, n_samples=4, n_snps=10)
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1, 2, 3],
            sd_threshold=10.0,
        )
        # All three calls received an explicit log_name
        log_names = sorted(c["log_name"] for c in runner.calls)
        assert log_names == ["restart_1.out", "restart_2.out", "restart_3.out"]
        # And the per-seed log files exist on disk
        log_dir = cache_dir / "build_logs"
        assert (log_dir / "restart_1.out").exists()
        assert (log_dir / "restart_2.out").exists()
        assert (log_dir / "restart_3.out").exists()

    def test_legacy_runner_uses_log_dir_scan_fallback(self, tmp_path: Path) -> None:
        """A sequential build with a runner that writes its log under
        a non-standard name (no `log_name` support) must still parse
        the LL — builder falls back to a snapshot-diff scan of log_dir."""

        class _NonCanonicalLogRunner:
            """Writes log under <binary>_<timestamp>.out, NOT
            restart_<seed>.out. Models a strict-typed legacy runner."""

            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
            ) -> object:
                seed = next(
                    int(a[2:]) for a in args if a.startswith("-s") and a[2:].isdigit()
                )
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                # Deliberately use a non-canonical log name.
                (log_dir / f"admixture_2026-01-01-seed{seed}.out").write_text(
                    "Loglikelihood: -42.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _NonCanonicalLogRunner()
        manifest = build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
        )
        # LL parsed via the fallback scan.
        assert manifest.best_loglikelihood == -42.0

    def test_parallel_with_legacy_runner_errors_early(self, tmp_path: Path) -> None:
        """A runner that doesn't support log_name and isn't a **kwargs
        forwarder cannot disambiguate concurrent restart logs. Builder
        must fail fast with a clear message rather than producing an
        incoherent build."""

        class _StrictLegacyRunner:
            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
            ) -> object:
                raise AssertionError("runner should not have been called")

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        with pytest.raises(PanelCacheError, match="parallel restarts"):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=_StrictLegacyRunner(),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                sd_threshold=10.0,
                max_parallel_restarts=2,
            )

    def test_kwargs_forwarder_runner_works_in_parallel(self, tmp_path: Path) -> None:
        """A runner declared with **kwargs is recognized as supporting
        log_name (and pid_callback), so parallel mode succeeds."""

        class _KwargsForwarder:
            """Idiomatic adapter — forwards anything via **kwargs."""

            def __init__(self) -> None:
                self.received_log_names: list[str] = []

            def run(self, **kwargs: Any) -> object:
                self.received_log_names.append(kwargs.get("log_name", ""))
                args = kwargs["args"]
                cwd = kwargs["cwd"]
                log_dir = kwargs["log_dir"]
                seed = next(
                    int(a[2:]) for a in args if a.startswith("-s") and a[2:].isdigit()
                )
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                ln = kwargs.get("log_name") or f"fallback_{seed}.out"
                (log_dir / ln).write_text("Loglikelihood: -10.0\n")
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _KwargsForwarder()
        manifest = build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1, 2],
            sd_threshold=10.0,
            max_parallel_restarts=2,
        )
        # Both restarts received their canonical log name (parallel
        # disambiguation worked).
        assert sorted(runner.received_log_names) == [
            "restart_1.out", "restart_2.out",
        ]
        assert manifest.best_loglikelihood == -10.0

    def test_dangling_symlink_refreshed(self, tmp_path: Path) -> None:
        """If a prior restart_dir contains a symlink whose target no
        longer exists (panel moved between builds), the rebuild must
        replace it — not silently keep the broken link."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        # Stage a dangling symlink mimicking a prior failed build
        # whose source path has since been removed.
        restart_dir = cache_dir / "build_restart_1"
        restart_dir.mkdir(parents=True)
        (cache_dir / "build_logs").mkdir()
        bogus_source = tmp_path / "gone_panel.bed"
        bogus_source.write_bytes(b"\x6c\x1b\x01")
        os.symlink(bogus_source, restart_dir / "panel.bed")
        bogus_source.unlink()  # link now dangling
        assert (restart_dir / "panel.bed").is_symlink()
        assert not (restart_dir / "panel.bed").exists()  # confirms dangling

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10, seed_to_ll={1: -50.0},
        )
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
        )
        # The symlink now points at the live panel_bed.
        staged = restart_dir / "panel.bed"
        assert staged.is_symlink()
        assert staged.resolve() == panel_bed.resolve()

    def test_symlink_to_different_source_replaced(self, tmp_path: Path) -> None:
        """A live symlink pointing at the *wrong* source (a different
        valid BED) must be replaced — same hazard as the dangling case
        but with a non-None target."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        wrong_panel = tmp_path / "wrong_panel.bed"
        wrong_panel.write_bytes(b"\x6c\x1b\x01")

        restart_dir = cache_dir / "build_restart_1"
        restart_dir.mkdir(parents=True)
        (cache_dir / "build_logs").mkdir()
        os.symlink(wrong_panel.resolve(), restart_dir / "panel.bed")

        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10, seed_to_ll={1: -50.0},
        )
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
        )
        staged = cache_dir / "build_restart_1" / "panel.bed"
        assert staged.resolve() == panel_bed.resolve()

    def test_runner_supports_recognizes_var_keyword(self) -> None:
        """Direct unit on the introspection helper itself."""
        from admixture_cache._dispatch import _runner_supports

        class _ExplicitKwarg:
            def run(self, *, args: list[str], cwd: Path, log_dir: Path,
                    timeout_seconds: int = 600,
                    log_name: str | None = None) -> None:
                ...

        class _Forwarder:
            def run(self, **kwargs: Any) -> None:
                ...

        class _Strict:
            def run(self, *, args: list[str], cwd: Path, log_dir: Path,
                    timeout_seconds: int = 600) -> None:
                ...

        assert _runner_supports(_ExplicitKwarg(), "log_name") is True
        assert _runner_supports(_Forwarder(), "log_name") is True
        assert _runner_supports(_Forwarder(), "anything_at_all") is True
        assert _runner_supports(_Strict(), "log_name") is False
        assert _runner_supports(_Strict(), "pid_callback") is False

    def test_concurrent_restarts_share_same_inode(self, tmp_path: Path) -> None:
        """All N restart_dirs' panel.bed symlinks resolve to the same
        underlying inode — that's what enables OS page-cache dedupe."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10,
            seed_to_ll={1: -100.0, 2: -110.0, 3: -120.0},
        )
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1, 2, 3],
            sd_threshold=10.0,
        )
        source_inode = panel_bed.stat().st_ino
        for seed in (1, 2, 3):
            staged = cache_dir / f"build_restart_{seed}" / "panel.bed"
            assert staged.stat().st_ino == source_inode

    def test_log_scan_fallback_ignores_dot_prev_files(self, tmp_path: Path) -> None:
        """If a prior run left a rotated `.prev` log in log_dir, the
        snapshot-diff fallback must NOT pick it as the current
        restart's log — that would parse a stale LL from the previous
        attempt and silently assign it to the new restart."""

        class _NonCanonicalLogRunner:
            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
            ) -> object:
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                # Simulate the SubprocessToolRunner rotation pattern:
                # rotate any prior log_dir/X to X.prev, then write the
                # new log. We do BOTH so the snapshot diff sees `.prev`
                # appear as "new" — the fallback must refuse it.
                live_log = log_dir / "admixture_run.out"
                if live_log.exists():
                    live_log.replace(live_log.with_suffix(".out.prev"))
                live_log.write_text(f"Loglikelihood: -{seed}.0\n")
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        log_dir = cache_dir / "build_logs"
        log_dir.mkdir(parents=True)
        # Pre-seed log_dir with a stale prior-attempt log that the
        # runner will rotate to .prev on first call.
        (log_dir / "admixture_run.out").write_text("Loglikelihood: -999.0\n")

        runner = _NonCanonicalLogRunner()
        manifest = build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
            max_parallel_restarts=1,
        )
        # Must have parsed the CURRENT run's LL (-1.0), not the stale
        # `.prev` rotation's LL (-999.0).
        assert manifest.best_loglikelihood == -1.0

    def test_legacy_real_file_refreshed_as_symlink(self, tmp_path: Path) -> None:
        """A v0.x cache_dir with REAL-file panel.bed/.bim/.fam (copied
        via shutil.copy2 in the v0 build) gets refreshed into symlinks
        on the next rebuild — otherwise the legacy data silently
        persists into the v1.x retrain."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        # Pre-populate restart_dir with REAL files (legacy state).
        restart_dir = cache_dir / "build_restart_1"
        restart_dir.mkdir(parents=True)
        import shutil as _sh
        for suffix in (".bed", ".bim", ".fam"):
            _sh.copy2(panel_bed.with_suffix(suffix),
                      restart_dir / f"panel{suffix}")
        # Sanity: these are real files (not symlinks).
        assert not (restart_dir / "panel.bed").is_symlink()

        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=10)
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
            max_parallel_restarts=1,
        )
        # After build: real files were unlinked and replaced with
        # symlinks pointing at the current panel source.
        for suffix in (".bed", ".bim", ".fam"):
            staged = restart_dir / f"panel{suffix}"
            assert staged.is_symlink(), (
                f"legacy real-file panel{suffix} not refreshed to symlink"
            )
            assert staged.resolve() == panel_bed.with_suffix(suffix).resolve()

    def test_legacy_pop_file_always_refreshed(self, tmp_path: Path) -> None:
        """An existing panel.pop in restart_dir is unconditionally
        replaced — a curator edit to the clusters file must not be
        silently masked by a stale copy from a prior build."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        # Pre-populate restart_dir with a STALE pop file
        # (different labels from the current `pop` source).
        restart_dir = cache_dir / "build_restart_1"
        restart_dir.mkdir(parents=True)
        stale_pop = restart_dir / "panel.pop"
        stale_pop.write_text("STALE\nDATA\nNOT\nCURRENT\n")

        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=10)
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1],
            sd_threshold=10.0,
            max_parallel_restarts=1,
        )
        assert stale_pop.read_text() == pop.read_text()


class TestAtomicManifestWrite:
    """Manifest write must be atomic so SIGKILL mid-write doesn't leave
    a partial JSON the next load_cache_manifest reads as corrupt."""

    def test_manifest_write_uses_tempfile_then_replace(
        self, tmp_path: Path,
    ) -> None:
        """Spy on os.replace to confirm the manifest is staged via a
        tempfile + atomic rename, not a direct write."""
        from unittest.mock import patch as _patch

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=10)

        original_replace = os.replace
        replace_calls: list[tuple[str, str]] = []

        def spy_replace(src: Any, dst: Any) -> None:
            replace_calls.append((str(src), str(dst)))
            original_replace(src, dst)

        with _patch("admixture_cache.builder.os.replace", side_effect=spy_replace):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=runner,
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1],
                sd_threshold=10.0,
                max_parallel_restarts=1,
            )

        # At least one os.replace call landed at manifest.json with a
        # tempfile source.
        manifest_replaces = [
            (s, d) for s, d in replace_calls
            if d.endswith("manifest.json")
        ]
        assert manifest_replaces, (
            f"manifest write did not go through os.replace; calls: "
            f"{replace_calls}"
        )
        src, _ = manifest_replaces[0]
        assert ".manifest-" in src and src.endswith(".json.tmp"), (
            f"manifest tempfile name unexpected: {src}"
        )


class _FakePlink2Runner:
    """Records args; optionally emits .prune.in / .bed stub files."""

    def __init__(
        self,
        *,
        emit_prune_in: bool = True,
        emit_pruned_bed: bool = True,
        kept_variants: list[str] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.emit_prune_in = emit_prune_in
        self.emit_pruned_bed = emit_pruned_bed
        self.kept_variants = kept_variants or ["rs0", "rs1"]

    def run(
        self,
        *,
        args: list[str],
        cwd: Path,
        log_dir: Path,
        timeout_seconds: int = 3600,
        log_name: str | None = None,
    ) -> object:
        from admixture_cache._paths import append_suffix

        self.calls.append(list(args))
        out_prefix = Path(args[args.index("--out") + 1])
        if "--indep-pairwise" in args and self.emit_prune_in:
            append_suffix(out_prefix, ".prune.in").write_text(
                "\n".join(self.kept_variants) + "\n",
            )
        if "--extract" in args and self.emit_pruned_bed:
            # Real plink2 --make-bed emits the full triplet
            # (.bed/.bim/.fam); mirror that so v1.1.1's triplet-
            # completeness check in ld_prune_panel doesn't trip.
            append_suffix(out_prefix, ".bed").write_bytes(b"\x6c\x1b\x01")
            append_suffix(out_prefix, ".bim").write_text(
                "\n".join(
                    f"1\t{v}\t0\t{i+1000}\tA\tG"
                    for i, v in enumerate(self.kept_variants)
                ) + "\n",
            )
            append_suffix(out_prefix, ".fam").write_text(
                "F\tI\t0\t0\t0\t-9\n",
            )
        return None


class TestLdPrunePanel:
    def test_emits_two_plink2_calls(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=5)
        runner = _FakePlink2Runner()
        out_prefix = tmp_path / "pruned"
        ld_prune_panel(
            panel_bed=panel_bed,
            output_prefix=out_prefix,
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        assert len(runner.calls) == 2
        # First call: --indep-pairwise
        assert "--indep-pairwise" in runner.calls[0]
        # Second call: --extract + --make-bed
        assert "--extract" in runner.calls[1]
        assert "--make-bed" in runner.calls[1]

    def test_default_parameters_in_args(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=5)
        runner = _FakePlink2Runner()
        ld_prune_panel(
            panel_bed=panel_bed,
            output_prefix=tmp_path / "pruned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        args = runner.calls[0]
        idx = args.index("--indep-pairwise")
        assert args[idx + 1] == "200"  # window_size default (variants)
        assert args[idx + 2] == "25"   # step_size default
        assert args[idx + 3] == "0.4"  # r2_threshold default

    def test_custom_parameters_passed_through(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=5)
        runner = _FakePlink2Runner()
        ld_prune_panel(
            panel_bed=panel_bed,
            output_prefix=tmp_path / "pruned",
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
            # Non-default values (default is 200/25/0.4) so each assertion
            # proves pass-through rather than coinciding with a default.
            window_size=123, step_size=10, r2_threshold=0.2,
        )
        args = runner.calls[0]
        idx = args.index("--indep-pairwise")
        assert args[idx + 1] == "123"
        assert args[idx + 2] == "10"
        assert args[idx + 3] == "0.2"

    def test_deprecated_window_kb_alias_maps_to_window_size(
        self, tmp_path: Path,
    ) -> None:
        """The old `window_kb` keyword (a misnomer: the value is a variant
        count, not kb) still works, mapping onto window_size with a
        DeprecationWarning, so existing callers do not break."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=5)
        runner = _FakePlink2Runner()
        # 77 is deliberately NOT the window_size default (200): if the
        # window_kb -> window_size mapping were dropped, window_size would
        # fall back to 200 and this assertion would catch it.
        with pytest.warns(DeprecationWarning, match="window_kb"):
            ld_prune_panel(
                panel_bed=panel_bed,
                output_prefix=tmp_path / "pruned",
                plink2_runner=runner,
                log_dir=tmp_path / "logs",
                window_kb=77,
            )
        args = runner.calls[0]
        idx = args.index("--indep-pairwise")
        assert args[idx + 1] == "77"  # alias value honored, not the 200 default

    def test_window_kb_and_window_size_both_raises(
        self, tmp_path: Path,
    ) -> None:
        """Passing both the deprecated `window_kb` and the canonical
        `window_size` is a caller mistake (they set the same
        --indep-pairwise window): raise TypeError rather than silently
        letting one win. Distinct values confirm the guard fires on
        "both passed", not on "values disagree". No plink2 call is made."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=5)
        runner = _FakePlink2Runner()
        with pytest.raises(TypeError, match="not both"):
            ld_prune_panel(
                panel_bed=panel_bed,
                output_prefix=tmp_path / "pruned",
                plink2_runner=runner,
                log_dir=tmp_path / "logs",
                window_size=200,
                window_kb=50,
            )
        assert runner.calls == []

    def test_missing_prune_in_raises(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=5)
        runner = _FakePlink2Runner(emit_prune_in=False)
        with pytest.raises(PanelCacheError, match=r"prune\.in"):
            ld_prune_panel(
                panel_bed=panel_bed,
                output_prefix=tmp_path / "pruned",
                plink2_runner=runner,
                log_dir=tmp_path / "logs",
            )

    def test_missing_pruned_bed_raises(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=5)
        runner = _FakePlink2Runner(emit_pruned_bed=False)
        # v1.1.1: the post-plink2 validation now checks the full BED
        # triplet, so the error message names the missing siblings.
        with pytest.raises(PanelCacheError, match="incomplete BED triplet"):
            ld_prune_panel(
                panel_bed=panel_bed,
                output_prefix=tmp_path / "pruned",
                plink2_runner=runner,
                log_dir=tmp_path / "logs",
            )

    def test_returns_pruned_bed_path(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=5)
        runner = _FakePlink2Runner(kept_variants=["rs0", "rs2"])
        out_prefix = tmp_path / "pruned"
        result = ld_prune_panel(
            panel_bed=panel_bed,
            output_prefix=out_prefix,
            plink2_runner=runner,
            log_dir=tmp_path / "logs",
        )
        assert result == out_prefix.with_suffix(".bed")
        assert result.exists()


def _write_panel_triplet_with_alleles(
    tmp_path: Path, allele_rows: list[tuple[str, str]], n_samples: int = 4,
) -> Path:
    """Like _write_panel_triplet but with caller-chosen (a1, a2) alleles
    per SNP, so a panel can include strand-ambiguous (A/T, C/G) SNPs."""
    bed = tmp_path / "panel.bed"
    bed.write_bytes(b"\x6c\x1b\x01")
    bim = tmp_path / "panel.bim"
    bim.write_text(
        "\n".join(
            f"1\trs{i}\t0\t{i+1000}\t{a1}\t{a2}"
            for i, (a1, a2) in enumerate(allele_rows)
        ) + "\n"
    )
    fam = tmp_path / "panel.fam"
    fam.write_text(
        "\n".join(f"F{i}\tI{i}\t0\t0\t1\t-9" for i in range(n_samples)) + "\n"
    )
    return bed


class TestBuildStrandAmbiguousGuard:
    """D11: build refuses a panel containing strand-ambiguous (A/T, C/G)
    SNPs by default and records the decision in the manifest."""

    def test_build_refuses_ambiguous_panel_by_default(
        self, tmp_path: Path,
    ) -> None:
        # rs0 A/G ok, rs1 A/T ambiguous, rs2 C/G ambiguous, rs3 A/G ok
        panel_bed = _write_panel_triplet_with_alleles(
            tmp_path, [("A", "G"), ("A", "T"), ("C", "G"), ("A", "G")],
        )
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=4)
        with pytest.raises(PanelCacheError, match="strand-ambiguous"):
            build_panel_cache(
                panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
                k=2, cache_dir=tmp_path / "cache", admixture_runner=runner,
                panel_id="p", panel_version="v", admixture_version="1.4.0",
                seeds=[1], sd_threshold=10.0,
            )
        # Guard fires BEFORE any training: runner is never invoked, no
        # manifest is written.
        assert runner.calls == []
        assert not (tmp_path / "cache" / "manifest.json").exists()

    def test_build_allows_ambiguous_when_disabled(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet_with_alleles(
            tmp_path, [("A", "G"), ("A", "T"), ("C", "G"), ("A", "G")],
        )
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=4)
        manifest = build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=tmp_path / "cache", admixture_runner=runner,
            panel_id="p", panel_version="v", admixture_version="1.4.0",
            seeds=[1], sd_threshold=10.0,
            exclude_strand_ambiguous=False,
        )
        assert manifest.strand_ambiguous_excluded is False

    def test_idempotent_rerun_of_kept_ambiguous_cache_is_noop(
        self, tmp_path: Path,
    ) -> None:
        """A cache built with exclude_strand_ambiguous=False (ambiguous
        SNPs retained) must still no-op on an idempotent re-run with the
        default (True). The guard runs AFTER the idempotency short-circuit,
        so a SHA-matching cache returns the existing manifest instead of
        hard-failing on the still-ambiguous panel."""
        panel_bed = _write_panel_triplet_with_alleles(
            tmp_path, [("A", "G"), ("A", "T"), ("C", "G"), ("A", "G")],
        )
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        # First build keeps ambiguous SNPs (would otherwise be refused).
        build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=cache_dir,
            admixture_runner=_FakeAdmixtureRunner(k=2, n_samples=4, n_snps=4),
            panel_id="p", panel_version="v", admixture_version="1.4.0",
            seeds=[1], sd_threshold=10.0, exclude_strand_ambiguous=False,
        )
        # Re-run with the DEFAULT (exclude=True) against the same valid
        # cache + unchanged ambiguous panel: must be a no-op, not a raise.
        runner2 = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=4)
        manifest = build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=cache_dir, admixture_runner=runner2,
            panel_id="p", panel_version="v", admixture_version="1.4.0",
            seeds=[1], sd_threshold=10.0,
        )
        assert runner2.calls == []  # idempotent short-circuit, no training
        assert manifest.strand_ambiguous_excluded is False  # original build's

    def test_build_records_excluded_true_on_clean_panel(
        self, tmp_path: Path,
    ) -> None:
        # _write_panel_triplet writes A/G (non-ambiguous) SNPs.
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=5)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        runner = _FakeAdmixtureRunner(k=2, n_samples=4, n_snps=5)
        manifest = build_panel_cache(
            panel_bed=panel_bed, panel_pop_file=pop, clusters_yaml=yaml,
            k=2, cache_dir=tmp_path / "cache", admixture_runner=runner,
            panel_id="p", panel_version="v", admixture_version="1.4.0",
            seeds=[1], sd_threshold=10.0,
        )
        assert manifest.strand_ambiguous_excluded is True


class TestStripStrandAmbiguousSnps:
    """D11 pre-build helper: drop A/T, C/G SNPs from a panel via plink2."""

    class _Runner:
        """Emits a full BED triplet on --make-bed (strip uses --make-bed
        with or without --exclude, never --extract)."""

        def __init__(self, retained: tuple[str, ...] = ("rs0", "rs3")) -> None:
            self.calls: list[list[str]] = []
            self.retained = retained

        def run(
            self, *, args: list[str], cwd: Path, log_dir: Path,
            timeout_seconds: int = 3600, log_name: str | None = None,
        ) -> object:
            self.calls.append(list(args))
            out_prefix = Path(args[args.index("--out") + 1])
            append_suffix(out_prefix, ".bed").write_bytes(b"\x6c\x1b\x01")
            append_suffix(out_prefix, ".bim").write_text(
                "\n".join(
                    f"1\t{v}\t0\t{i+1000}\tA\tG"
                    for i, v in enumerate(self.retained)
                ) + "\n"
            )
            append_suffix(out_prefix, ".fam").write_text("F\tI\t0\t0\t0\t-9\n")
            return None

    def test_excludes_ambiguous_snps(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet_with_alleles(
            tmp_path, [("A", "G"), ("A", "T"), ("C", "G"), ("A", "C")],
        )
        runner = self._Runner()
        out = tmp_path / "clean"
        result = strip_strand_ambiguous_snps(
            panel_bed=panel_bed, output_prefix=out,
            plink2_runner=runner, log_dir=tmp_path / "logs",
        )
        args = runner.calls[0]
        assert "--exclude" in args
        exclude_path = Path(args[args.index("--exclude") + 1])
        assert sorted(exclude_path.read_text().split()) == ["rs1", "rs2"]
        assert result == append_suffix(out, ".bed")
        assert result.exists()

    def test_copies_through_clean_panel(self, tmp_path: Path) -> None:
        panel_bed = _write_panel_triplet(tmp_path, n_samples=3, n_snps=4)  # A/G
        runner = self._Runner()
        strip_strand_ambiguous_snps(
            panel_bed=panel_bed, output_prefix=tmp_path / "clean",
            plink2_runner=runner, log_dir=tmp_path / "logs",
        )
        args = runner.calls[0]
        assert "--exclude" not in args
        assert "--make-bed" in args

    def test_missing_panel_bim_raises(self, tmp_path: Path) -> None:
        bed = tmp_path / "panel.bed"
        bed.write_bytes(b"\x6c\x1b\x01")  # no sibling .bim
        with pytest.raises(PanelCacheError, match=r"\.bim missing"):
            strip_strand_ambiguous_snps(
                panel_bed=bed, output_prefix=tmp_path / "clean",
                plink2_runner=self._Runner(), log_dir=tmp_path / "logs",
            )


class TestParallelRestartCancellation:
    """On first-failure during parallel restart execution, the in-flight
    subprocesses must receive SIGTERM (via reported PIDs) — Future.cancel
    alone leaves running children dangling for hours."""

    def test_sigterm_sent_to_inflight_children(self, tmp_path: Path) -> None:
        import subprocess
        import threading
        import time
        from collections.abc import Callable

        spawned: dict[int, subprocess.Popen[bytes]] = {}
        # Signal: the failure-injecting worker waits until at least one
        # other restart has reported its PID before raising. Otherwise
        # the failure can win the race and the cancel path has nothing
        # to SIGTERM.
        peer_pid_registered = threading.Event()

        class _SleepRunner:
            """Spawns `sleep 30` with start_new_session=True (so it gets
            its own pgid, matching the SubprocessToolRunner contract).
            Reports PID via pid_callback. On seed=2, raises (after peers
            have registered) to trigger the cancellation path."""

            def __init__(self) -> None:
                self.fail_on_seed = 2

            def run(
                self, *,
                args: list[str],
                cwd: Path,
                log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
                pid_callback: Callable[[int], None] | None = None,
            ) -> object:
                seed = next(
                    (int(a[2:]) for a in args if a.startswith("-s") and a[2:].isdigit()),
                    -1,
                )
                if seed == self.fail_on_seed:
                    # Wait for at least one peer to register a PID so
                    # the cancellation actually has something to do.
                    peer_pid_registered.wait(timeout=5.0)
                    raise RuntimeError(f"injected failure on seed {seed}")
                # Otherwise spawn a real long-running sleep; report PID.
                # start_new_session=True so the SUT's killpg path works
                # without falling back to bare PID kill.
                proc = subprocess.Popen(
                    ["sleep", "30"], start_new_session=True,
                )
                spawned[seed] = proc
                if pid_callback is not None:
                    pid_callback(proc.pid)
                peer_pid_registered.set()
                proc.wait()
                # Emit synthetic outputs so downstream code doesn't get
                # confused if we somehow exit cleanly.
                k = int(args[-1])
                np.savetxt(cwd / f"panel.{k}.P", np.full((10, k), 0.5))
                np.savetxt(cwd / f"panel.{k}.Q", np.full((4, k), 1.0 / k))
                if log_name:
                    (log_dir / log_name).write_text("Loglikelihood: -1.0\n")
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        # Force parallel execution with 2 seeds: seed=1 starts a real
        # sleep + reports its PID; seed=2 waits for seed=1's PID
        # registration before raising, so the cancel path is
        # deterministic. The try/finally guarantees we don't leak
        # `sleep 30` processes on assertion failure — orphaned sleeps
        # would slow down (or hang) the pytest worker for 30s each.
        try:
            t0 = time.time()
            with pytest.raises(PanelCacheError, match="seed=2"):
                build_panel_cache(
                    panel_bed=panel_bed,
                    panel_pop_file=pop,
                    clusters_yaml=yaml,
                    k=2,
                    cache_dir=cache_dir,
                    admixture_runner=_SleepRunner(),
                    track="regional",
                    panel_id="p1",
                    panel_version="v1",
                    admixture_version="1.4.0",
                    seeds=[1, 2],
                    threads=1,
                    sd_threshold=10.0,
                    max_parallel_restarts=2,
                )
            elapsed = time.time() - t0

            # The whole thing should take well under the 30-second sleep
            # window — SIGTERM must have killed seed=1's sleep.
            assert elapsed < 10, (
                f"build took {elapsed:.1f}s — SIGTERM may not have reached "
                f"the in-flight subprocesses"
            )
            # The spawned sleep should be terminated by now (poll up to
            # 5 s for the kernel to update process state on a loaded
            # CI runner — a sleep that DID receive SIGTERM may take a
            # moment to show up as poll()-non-None).
            for seed, proc in spawned.items():
                for _ in range(100):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.05)
                assert proc.poll() is not None, (
                    f"seed={seed} subprocess still running after raise"
                )
        finally:
            # Defense-in-depth: kill any sleep we spawned that's still
            # somehow alive (e.g., the cancellation under test regressed,
            # or we hit an unexpected exception path). Without this, a
            # broken cancellation would leak `sleep 30` per failed test,
            # accumulating zombies across pytest-xdist reruns.
            for proc in spawned.values():
                if proc.poll() is None:
                    with contextlib.suppress(Exception):
                        proc.kill()
                        proc.wait(timeout=5)


class TestAutoMaxParallelRestarts:
    """Heuristic ``cores // (threads * 2)`` capped at len(seeds), floor 1."""

    @pytest.mark.parametrize(
        "cpu_count,threads,n_seeds,expected",
        [
            # 1 core: always 1
            (1, 1, 5, 1),
            (1, 3, 5, 1),
            (1, 8, 5, 1),
            # 4 cores
            (4, 1, 5, 2),    # 4 // 2 = 2
            (4, 3, 5, 1),    # 4 // 6 = 0 → clamped to 1
            (4, 8, 5, 1),    # 4 // 16 = 0 → 1
            # 8 cores
            (8, 1, 5, 4),    # 8 // 2 = 4
            (8, 3, 5, 1),    # 8 // 6 = 1
            (8, 8, 5, 1),    # 8 // 16 = 0 → 1
            # 16 cores — the empirical sweet-spot bucket
            (16, 1, 5, 5),   # 16 // 2 = 8, capped at 5 seeds
            (16, 3, 5, 2),   # 16 // 6 = 2 — matches the empirical sweet spot on 16-core / K=4
            (16, 8, 5, 1),   # 16 // 16 = 1
            # 32 cores
            (32, 1, 5, 5),   # 32 // 2 = 16, capped
            (32, 3, 5, 5),   # 32 // 6 = 5, capped
            (32, 8, 5, 2),   # 32 // 16 = 2
            # n_seeds floor effect
            (32, 1, 3, 3),   # cap dominates
        ],
    )
    def test_heuristic_value(
        self, cpu_count: int, threads: int, n_seeds: int, expected: int,
    ) -> None:
        with patch("admixture_cache.builder.os.cpu_count", return_value=cpu_count):
            got = _auto_max_parallel_restarts(threads=threads, n_seeds=n_seeds)
        assert got == expected

    def test_cpu_count_none_treated_as_one(self) -> None:
        """os.cpu_count() can return None on hosts without a /proc/cpuinfo
        — must not divide by zero or crash."""
        with patch("admixture_cache.builder.os.cpu_count", return_value=None):
            got = _auto_max_parallel_restarts(threads=4, n_seeds=5)
        assert got == 1

    def test_threads_zero_treated_as_one(self) -> None:
        """Defensive: don't divide by zero if someone passes threads=0."""
        with patch("admixture_cache.builder.os.cpu_count", return_value=8):
            got = _auto_max_parallel_restarts(threads=0, n_seeds=5)
        assert got >= 1


class TestNumaPinning:
    """`numa_node_per_restart=True` makes build_panel_cache pass
    ``argv_prefix=["numactl", "--membind=N", "--"]`` to the runner for
    each parallel restart. Skipped (no-op) on platforms without
    numactl, single-socket boxes, or sequential execution."""

    def test_numa_disabled_by_default_no_argv_prefix(self, tmp_path: Path) -> None:
        """Default `numa_node_per_restart=False` → dispatcher must NOT
        forward the `argv_prefix` kwarg at all (not even as `None`).
        This tightens the assertion to verify the kwarg is omitted
        from the call, distinguishing 'kwarg dropped because None' from
        'kwarg dropped because runner doesn't support it'."""
        observed_kwargs: list[set[str]] = []

        class _KwargsCapturingRunner:
            """Captures the exact set of kwargs received, not just
            their values. A None forwarded explicitly would show up
            in the kwargs dict; an omitted kwarg would not."""

            def run(self, **kwargs: Any) -> object:
                observed_kwargs.append(set(kwargs.keys()))
                args = kwargs["args"]
                cwd = kwargs["cwd"]
                log_dir = kwargs["log_dir"]
                log_name = kwargs.get("log_name")
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                (log_dir / (log_name or f"restart_{seed}.out")).write_text(
                    f"Loglikelihood: -{seed}.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=_KwargsCapturingRunner(),
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1, 2],
            threads=1,
            sd_threshold=10.0,
            max_parallel_restarts=2,
        )
        # The kwarg `argv_prefix` must not appear in any of the
        # observed call signatures — the dispatcher's `if argv_prefix
        # is not None` guard means it's never explicitly passed.
        for call_kwargs in observed_kwargs:
            assert "argv_prefix" not in call_kwargs, (
                f"argv_prefix forwarded when None: {call_kwargs}"
            )

    def test_numa_enabled_no_numactl_falls_back_silently(
        self, tmp_path: Path,
    ) -> None:
        """When `numa_node_per_restart=True` but numactl isn't on PATH,
        the build proceeds without pinning (logs a warning)."""
        from unittest.mock import patch as _patch

        observed: list[list[str] | None] = []

        class _Runner:
            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
                pid_callback: Any = None,
                argv_prefix: list[str] | None = None,
            ) -> object:
                observed.append(argv_prefix)
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                (log_dir / (log_name or f"restart_{seed}.out")).write_text(
                    f"Loglikelihood: -{seed}.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        # Mock numactl missing.
        with _patch("admixture_cache.builder.shutil.which", return_value=None):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=_Runner(),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                threads=1,
                sd_threshold=10.0,
                max_parallel_restarts=2,
                numa_node_per_restart=True,
            )
        # numactl missing → no argv_prefix forwarded.
        assert observed == [None, None]

    def test_numa_enabled_multi_node_spreads_restarts(
        self, tmp_path: Path,
    ) -> None:
        """With numactl available and 2 NUMA nodes detected, two
        parallel restarts land on nodes 0 and 1 (one each)."""
        from unittest.mock import patch as _patch

        observed: dict[int, list[str] | None] = {}

        class _Runner:
            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
                pid_callback: Any = None,
                argv_prefix: list[str] | None = None,
            ) -> object:
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                observed[seed] = argv_prefix
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                (log_dir / (log_name or f"restart_{seed}.out")).write_text(
                    f"Loglikelihood: -{seed}.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        with _patch("admixture_cache.builder.shutil.which", return_value="/usr/bin/numactl"), \
             _patch("admixture_cache.builder._detect_numa_nodes", return_value=2):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=_Runner(),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                threads=1,
                sd_threshold=10.0,
                max_parallel_restarts=2,
                numa_node_per_restart=True,
            )
        # Seed 1 → node 0; seed 2 → node 1 (sorted seed order).
        assert observed[1] == ["numactl", "--membind=0", "--"]
        assert observed[2] == ["numactl", "--membind=1", "--"]

    def test_numa_enabled_sequential_skipped(self, tmp_path: Path) -> None:
        """Even with numactl + multi-node, sequential execution
        (max_parallel_restarts=1) is a no-op — pinning doesn't help
        one process at a time."""
        from unittest.mock import patch as _patch

        observed: list[list[str] | None] = []

        class _Runner:
            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
                pid_callback: Any = None,
                argv_prefix: list[str] | None = None,
            ) -> object:
                observed.append(argv_prefix)
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                (log_dir / (log_name or f"restart_{seed}.out")).write_text(
                    f"Loglikelihood: -{seed}.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        with _patch("admixture_cache.builder.shutil.which", return_value="/usr/bin/numactl"), \
             _patch("admixture_cache.builder._detect_numa_nodes", return_value=4):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=_Runner(),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                threads=1,
                sd_threshold=10.0,
                max_parallel_restarts=1,
                numa_node_per_restart=True,
            )
        assert observed == [None, None]

    def test_detect_numa_nodes_no_sysfs(self) -> None:
        """On macOS / containers without sysfs, _detect_numa_nodes returns 1."""
        from unittest.mock import patch as _patch

        from admixture_cache.builder import _detect_numa_nodes

        with _patch("admixture_cache.builder.Path") as path_cls:
            instance = path_cls.return_value
            instance.is_dir.return_value = False
            assert _detect_numa_nodes() == 1

    def test_detect_numa_nodes_ignores_non_dir_entries(
        self, tmp_path: Path,
    ) -> None:
        """v1.1.1 regression: `_detect_numa_nodes` must filter for
        DIRECTORIES named `nodeN` (where N is digits). Files / symlinks
        starting with 'node' (e.g., a `node_list` file in a kernel
        patch or container overlay) should NOT inflate the count."""
        from unittest.mock import patch as _patch

        from admixture_cache.builder import _detect_numa_nodes

        # Build a fake sysfs-like layout under tmp_path.
        sysfs = tmp_path / "node"
        sysfs.mkdir()
        (sysfs / "node0").mkdir()
        (sysfs / "node1").mkdir()
        # Decoys: file starting with 'node' + dir starting with 'node'
        # but not followed by digits.
        (sysfs / "node_list").write_text("0-1")
        (sysfs / "nodeinfo").mkdir()

        with _patch("admixture_cache.builder.Path", return_value=sysfs):
            assert _detect_numa_nodes() == 2

    def test_numa_pinning_warns_when_runner_lacks_argv_prefix(
        self, tmp_path: Path,
    ) -> None:
        """v1.1.1 regression: when `numa_node_per_restart=True` but the
        runner doesn't support `argv_prefix` (and isn't a **kwargs
        forwarder), the build must WARN and degrade to non-pinned
        execution — not silently advertise NUMA pinning while
        dropping the kwarg."""
        from unittest.mock import patch as _patch

        observed_kwargs: list[set[str]] = []

        class _V10EraRunner:
            """log_name + pid_callback present; argv_prefix absent.
            Models a custom runner upgraded for v1.0 but not v1.1."""

            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
                pid_callback: Any = None,
            ) -> object:
                # Record what we WERE called with (vs the v1.0 protocol).
                observed_kwargs.append({
                    "log_name", "pid_callback",  # always implied
                })
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                (log_dir / (log_name or f"restart_{seed}.out")).write_text(
                    f"Loglikelihood: -{seed}.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        with (
            _patch("admixture_cache.builder.shutil.which",
                   return_value="/usr/bin/numactl"),
            _patch("admixture_cache.builder._detect_numa_nodes",
                   return_value=2),
            self._capture_warning_log() as warnings_collected,
        ):
            build_panel_cache(
                    panel_bed=panel_bed,
                    panel_pop_file=pop,
                    clusters_yaml=yaml,
                    k=2,
                    cache_dir=cache_dir,
                    admixture_runner=_V10EraRunner(),
                    track="regional",
                    panel_id="p1",
                    panel_version="v1",
                    admixture_version="1.4.0",
                    seeds=[1, 2],
                    threads=1,
                    sd_threshold=10.0,
                    max_parallel_restarts=2,
                    numa_node_per_restart=True,
                )

        # The warning must have been emitted, naming argv_prefix.
        assert any("argv_prefix" in msg for msg in warnings_collected), (
            f"expected an argv_prefix warning; got: {warnings_collected}"
        )

    @contextlib.contextmanager
    def _capture_warning_log(self) -> Any:
        """Capture WARNING-level log messages emitted during the
        context body. Used to verify the NUMA-degradation warning."""
        import logging

        collected: list[str] = []
        handler = logging.Handler()

        def handle(record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                collected.append(record.getMessage())

        handler.emit = handle  # type: ignore[method-assign]
        root = logging.getLogger("admixture_cache.builder")
        root.addHandler(handler)
        prior_level = root.level
        root.setLevel(logging.WARNING)
        try:
            yield collected
        finally:
            root.removeHandler(handler)
            root.setLevel(prior_level)

    def test_numa_pinning_workers_dont_block_when_nodes_lt_parallelism(
        self, tmp_path: Path,
    ) -> None:
        """v1.1.1 second-pass regression: when n_nodes < effective_parallelism,
        no worker may block on `numa_node_pool.get()`. The pre-fix code
        sized the queue to `numa_n_nodes`, so excess workers blocked
        before registering their PID — and on first-failure could
        unblock AFTER cancellation and spawn unkillable subprocesses.

        Test: 2 NUMA nodes, max_parallel_restarts=4. All 4 workers must
        get a node immediately; some share. Build completes; warning
        is emitted naming the partial-pinning condition."""
        from unittest.mock import patch as _patch

        node_assignments: dict[int, int] = {}

        class _NodeRecordingRunner:
            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
                pid_callback: Any = None,
                argv_prefix: list[str] | None = None,
            ) -> object:
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                node = -1
                if argv_prefix is not None:
                    for tok in argv_prefix:
                        if tok.startswith("--membind="):
                            node = int(tok.split("=")[1])
                            break
                node_assignments[seed] = node
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                (log_dir / (log_name or f"restart_{seed}.out")).write_text(
                    f"Loglikelihood: -{seed}.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        with (
            _patch("admixture_cache.builder.shutil.which",
                   return_value="/usr/bin/numactl"),
            _patch("admixture_cache.builder._detect_numa_nodes",
                   return_value=2),
            self._capture_warning_log() as warnings_collected,
        ):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=_NodeRecordingRunner(),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2, 3, 4],
                threads=1,
                sd_threshold=10.0,
                max_parallel_restarts=4,
                numa_node_per_restart=True,
            )

        # All 4 seeds got a node assignment (no blocked workers).
        assert set(node_assignments) == {1, 2, 3, 4}
        for seed, node in node_assignments.items():
            assert node in (0, 1), f"seed {seed} got unexpected node {node}"

        # Warning emitted naming the partial-pinning condition.
        assert any(
            "partial pinning" in msg.lower() or "share a node" in msg.lower()
            for msg in warnings_collected
        ), f"expected partial-pinning warning; got: {warnings_collected}"

    def test_numa_pinning_slot_based_assignment(self, tmp_path: Path) -> None:
        """v1.1.1 regression: with seeds > effective_parallelism, the
        NUMA assignment must be slot-based (not seed-ordinal). Each
        in-flight worker holds exactly one node; finished worker
        releases the node before the next worker claims it. No two
        in-flight restarts share a node."""
        import threading
        import time
        from unittest.mock import patch as _patch

        node_history: list[tuple[int, str, int]] = []
        node_history_lock = threading.Lock()
        # Map seed → start barrier so we can synchronize concurrent
        # in-flight to verify uniqueness.
        live_seeds: set[int] = set()
        live_seeds_lock = threading.Lock()

        class _SlowRunner:
            """Records which NUMA node each restart claims. Stays in
            flight long enough that the test can verify all concurrent
            in-flight slots hold distinct nodes."""

            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
                pid_callback: Any = None,
                argv_prefix: list[str] | None = None,
            ) -> object:
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                # Parse the node from argv_prefix.
                node = -1
                if argv_prefix is not None:
                    for tok in argv_prefix:
                        if tok.startswith("--membind="):
                            node = int(tok.split("=")[1])
                            break
                # Record entry. Uniqueness check is performed after
                # the build by walking the history; doing it here
                # under the lock would race with other workers.
                with live_seeds_lock, node_history_lock:
                    node_history.append((seed, "claim", node))
                    live_seeds.add(seed)
                # Hold the slot briefly so peers race.
                time.sleep(0.05)
                with live_seeds_lock:
                    live_seeds.discard(seed)
                    with node_history_lock:
                        node_history.append((seed, "release", node))
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                (log_dir / (log_name or f"restart_{seed}.out")).write_text(
                    f"Loglikelihood: -{seed}.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"

        # 5 seeds × 2 parallel slots × 2 NUMA nodes — the exact
        # configuration where v1.1.0's static seed→node mapping
        # would have collided.
        with _patch("admixture_cache.builder.shutil.which",
                    return_value="/usr/bin/numactl"), \
             _patch("admixture_cache.builder._detect_numa_nodes",
                    return_value=2):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=_SlowRunner(),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2, 3, 4, 5],
                threads=1,
                sd_threshold=10.0,
                max_parallel_restarts=2,
                numa_node_per_restart=True,
            )

        # All seeds saw a node assignment (None means no pinning).
        claims = [(s, n) for s, evt, n in node_history if evt == "claim"]
        assert len(claims) == 5
        for s, n in claims:
            assert n in (0, 1), f"seed {s} got node {n}"

        # The strict assertion: at no point should two concurrent
        # claims have occupied the same node. Reconstruct the
        # in-flight set as we walk the history.
        in_flight: dict[int, int] = {}  # seed → node
        for s, evt, n in node_history:
            if evt == "claim":
                # Check no current in-flight seed holds this node.
                conflict = [
                    other_s for other_s, other_n in in_flight.items()
                    if other_n == n
                ]
                assert not conflict, (
                    f"seed {s} tried to claim node {n} while seeds "
                    f"{conflict} were already on it: {node_history}"
                )
                in_flight[s] = n
            elif evt == "release":
                in_flight.pop(s, None)


class TestBuildPanelCacheAutoDefault:
    def test_default_value_used_when_max_parallel_restarts_none(
        self, tmp_path: Path,
    ) -> None:
        """Passing max_parallel_restarts=None triggers the auto heuristic
        (verified by patching cpu_count to a known value)."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10, seed_to_ll={1: -100.0, 2: -110.0},
        )
        # 16 cores / threads=3 → heuristic gives 2 parallel restarts.
        # We don't directly observe the parallelism choice here, but
        # the build should still succeed.
        with patch("admixture_cache.builder.os.cpu_count", return_value=16):
            manifest = build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=runner,
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                threads=3,
                sd_threshold=10.0,
            )
        assert manifest.best_seed == 1

    def test_partial_modern_runner_no_pid_callback_rejected_in_parallel(
        self, tmp_path: Path,
    ) -> None:
        """A runner that supports `log_name` but NOT `pid_callback`
        is REJECTED at the parallel-mode guard — without pid_callback
        the failure path can't SIGTERM in-flight restarts and the
        whole build would hang up to per_restart_timeout_seconds × N.
        Better to error early with a clear message than silently lose
        cancellation."""

        class _LogNameOnly:
            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
            ) -> object:
                raise AssertionError("runner should not have been called")

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        with pytest.raises(PanelCacheError, match="pid_callback"):
            build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=_LogNameOnly(),
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                threads=1,
                sd_threshold=10.0,
                max_parallel_restarts=2,
            )

    def test_partial_modern_runner_no_pid_callback_works_sequentially(
        self, tmp_path: Path,
    ) -> None:
        """The same partial-modern runner DOES work in sequential mode
        — pid_callback support is only required when concurrency
        creates the need for cancellation."""

        class _LogNameOnly:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def run(
                self, *, args: list[str], cwd: Path, log_dir: Path,
                timeout_seconds: int = 86400,
                log_name: str | None = None,
            ) -> object:
                self.calls.append(log_name or "")
                seed = next(
                    int(a[2:]) for a in args
                    if a.startswith("-s") and a[2:].isdigit()
                )
                k = int(args[-1])
                rng = np.random.default_rng(seed)
                np.savetxt(cwd / f"panel.{k}.P",
                           rng.uniform(0.05, 0.95, size=(10, k)))
                np.savetxt(cwd / f"panel.{k}.Q",
                           rng.dirichlet(np.ones(k), size=4))
                (log_dir / (log_name or f"restart_{seed}.out")).write_text(
                    "Loglikelihood: -55.0\n",
                )
                return None

        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _LogNameOnly()
        manifest = build_panel_cache(
            panel_bed=panel_bed,
            panel_pop_file=pop,
            clusters_yaml=yaml,
            k=2,
            cache_dir=cache_dir,
            admixture_runner=runner,
            track="regional",
            panel_id="p1",
            panel_version="v1",
            admixture_version="1.4.0",
            seeds=[1, 2],
            threads=1,
            sd_threshold=10.0,
            max_parallel_restarts=1,  # explicit sequential
        )
        assert sorted(runner.calls) == ["restart_1.out", "restart_2.out"]
        assert manifest.best_loglikelihood == -55.0

    def test_explicit_integer_overrides_auto(self, tmp_path: Path) -> None:
        """Operator-provided integer is honored verbatim."""
        panel_bed = _write_panel_triplet(tmp_path, n_samples=4, n_snps=10)
        pop = _write_pop_file(tmp_path, ["A", "B", "A", "B"])
        yaml = _write_clusters_yaml(tmp_path)
        cache_dir = tmp_path / "cache"
        runner = _FakeAdmixtureRunner(
            k=2, n_samples=4, n_snps=10, seed_to_ll={1: -100.0, 2: -110.0},
        )
        # Even on a 1-core host, an explicit max_parallel_restarts=2 is
        # respected (clamped only by len(seeds)).
        with patch("admixture_cache.builder.os.cpu_count", return_value=1):
            manifest = build_panel_cache(
                panel_bed=panel_bed,
                panel_pop_file=pop,
                clusters_yaml=yaml,
                k=2,
                cache_dir=cache_dir,
                admixture_runner=runner,
                track="regional",
                panel_id="p1",
                panel_version="v1",
                admixture_version="1.4.0",
                seeds=[1, 2],
                threads=8,
                sd_threshold=10.0,
                max_parallel_restarts=2,
            )
        assert manifest.seeds_used == [1, 2]
