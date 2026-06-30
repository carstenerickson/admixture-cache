"""Cache I/O helpers: sha256_file streaming hash, load_cached_p shape
validation, cluster-order derivation from .pop files."""

from __future__ import annotations

import hashlib
from datetime import UTC
from pathlib import Path

import numpy as np
import pytest

from admixture_cache import PanelCacheError, load_cached_p, sha256_file
from admixture_cache.builder import _derive_cluster_order_from_pop_file


class TestSha256File:
    def test_matches_hashlib_on_small_file(self, tmp_path: Path) -> None:
        path = tmp_path / "hello.txt"
        payload = b"hello world\n"
        path.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()
        assert sha256_file(path) == expected

    def test_matches_hashlib_on_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert sha256_file(path) == expected

    def test_matches_hashlib_on_multi_chunk_file(self, tmp_path: Path) -> None:
        """Spans multiple chunks of default chunk_size to exercise the
        streaming loop."""
        path = tmp_path / "big.bin"
        payload = b"x" * (2**16 * 3 + 7)  # 3 chunks + remainder
        path.write_bytes(payload)
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

    def test_custom_chunk_size_yields_same_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "custom.txt"
        payload = b"abcdef" * 1000
        path.write_bytes(payload)
        assert sha256_file(path, chunk_size=7) == sha256_file(path, chunk_size=2**16)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sha256_file(tmp_path / "does_not_exist")


class TestLoadCachedP:
    def _write_p(self, path: Path, m: int, k: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        P = rng.uniform(0.05, 0.95, size=(m, k))
        np.savetxt(path, P)
        return P

    def test_loads_correctly_shaped_p(self, tmp_path: Path) -> None:
        P = self._write_p(tmp_path / "panel.4.P", m=50, k=4)
        loaded = load_cached_p(tmp_path, k=4)
        assert loaded.shape == (50, 4)
        np.testing.assert_allclose(loaded, P, atol=1e-10)

    def test_missing_p_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PanelCacheError, match="cache file missing"):
            load_cached_p(tmp_path, k=4)

    def test_error_message_points_at_builder(self, tmp_path: Path) -> None:
        """Error must direct the user to the public build API, not a
        consumer-specific CLI."""
        try:
            load_cached_p(tmp_path, k=4)
        except PanelCacheError as exc:
            msg = str(exc)
            assert "build_panel_cache" in msg
            # And NOT the legacy consumer command
            assert "ancestry-pipeline" not in msg

    def test_wrong_k_column_count_rejected(self, tmp_path: Path) -> None:
        self._write_p(tmp_path / "panel.4.P", m=20, k=4)
        # File has k=4 but caller asks k=5 — but load_cached_p looks at panel.5.P
        # so this is actually a missing-file case. Pre-load with mismatched name:
        rng = np.random.default_rng(0)
        # Save a 20x3 array under panel.4.P would conflict with above; use new dir
        bad_dir = tmp_path.parent / "bad"
        bad_dir.mkdir()
        bad = rng.uniform(0.1, 0.9, size=(20, 3))
        np.savetxt(bad_dir / "panel.4.P", bad)
        with pytest.raises(PanelCacheError, match=r"has shape"):
            load_cached_p(bad_dir, k=4)

    def test_one_dim_p_rejected(self, tmp_path: Path) -> None:
        np.savetxt(tmp_path / "panel.4.P", np.arange(8.0))
        with pytest.raises(PanelCacheError):
            load_cached_p(tmp_path, k=4)


class TestDeriveClusterOrderFromPopFile:
    def test_first_appearance_order_preserved(self, tmp_path: Path) -> None:
        pop = tmp_path / "panel.pop"
        # First-appearance order: B, A, C
        pop.write_text("B\nA\nB\nC\n-\nA\n")
        order, n_unlabeled = _derive_cluster_order_from_pop_file(
            panel_pop_file=pop, expected_k=3,
        )
        assert order == ["B", "A", "C"]
        assert n_unlabeled == 1  # one '-' row

    def test_dash_lines_ignored(self, tmp_path: Path) -> None:
        pop = tmp_path / "panel.pop"
        pop.write_text("-\n-\nA\n-\nB\n-\n")
        order, n_unlabeled = _derive_cluster_order_from_pop_file(
            panel_pop_file=pop, expected_k=2,
        )
        assert order == ["A", "B"]
        assert n_unlabeled == 4  # four '-' rows

    def test_empty_lines_ignored_for_order_but_counted_unlabeled(
        self, tmp_path: Path,
    ) -> None:
        pop = tmp_path / "panel.pop"
        pop.write_text("\n\nA\n\nB\n\n")
        order, n_unlabeled = _derive_cluster_order_from_pop_file(
            panel_pop_file=pop, expected_k=2,
        )
        assert order == ["A", "B"]
        # Blank lines contribute no label but ARE unlabeled (free-Q) rows,
        # matching how ADMIXTURE reads a positionally-aligned .pop (gh #9).
        assert n_unlabeled == 4

    def test_unexpected_k_count_raises(self, tmp_path: Path) -> None:
        pop = tmp_path / "panel.pop"
        pop.write_text("A\nB\nC\n")
        with pytest.raises(PanelCacheError, match="3 distinct"):
            _derive_cluster_order_from_pop_file(
                panel_pop_file=pop, expected_k=2,
            )

    def test_only_dash_lines_yields_zero_clusters(self, tmp_path: Path) -> None:
        pop = tmp_path / "panel.pop"
        pop.write_text("-\n-\n-\n")
        with pytest.raises(PanelCacheError, match="0 distinct"):
            _derive_cluster_order_from_pop_file(
                panel_pop_file=pop, expected_k=2,
            )

    def test_duplicates_collapsed(self, tmp_path: Path) -> None:
        pop = tmp_path / "panel.pop"
        pop.write_text("A\nA\nA\nB\nB\nA\nB\nC\nC\n")
        order, n_unlabeled = _derive_cluster_order_from_pop_file(
            panel_pop_file=pop, expected_k=3,
        )
        assert order == ["A", "B", "C"]
        assert n_unlabeled == 0


class TestLoadCachedPIntegration:
    """Cross-cutting: P shape matches manifest K and is round-trippable
    through the cache directory."""

    def test_p_load_with_k_matches_manifest(self, tmp_path: Path) -> None:
        from datetime import datetime

        from admixture_cache import PanelCacheManifest

        # Write a synthetic P matrix
        P = np.random.default_rng(0).uniform(0.05, 0.95, size=(30, 4))
        np.savetxt(tmp_path / "panel.4.P", P)
        # Write a matching manifest
        manifest = PanelCacheManifest(
            track="regional", panel_id="x", panel_version="v1",
            panel_bim_sha256="a"*64, clusters_yaml_sha256="b"*64,
            k=4, admixture_version="1.4.0", seeds_used=[1],
            best_seed=1, best_loglikelihood=-1.0, restart_sd_max=0.0,
            cluster_order=["c1","c2","c3","c4"],
            build_wallclock_seconds=1.0,
            build_timestamp=datetime.now(UTC),
        )
        (tmp_path / "manifest.json").write_text(manifest.model_dump_json())

        from admixture_cache import load_cache_manifest
        m = load_cache_manifest(tmp_path)
        loaded_p = load_cached_p(tmp_path, k=m.k)
        assert loaded_p.shape[1] == m.k
        np.testing.assert_allclose(loaded_p, P)
