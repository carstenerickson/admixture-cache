"""Tests for the published-cache distribution module.

Mocks urllib so we don't hit GitHub. A small synthetic tarball is
built inline to exercise the extract + validate path end-to-end
without checked-in binary fixtures.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import threading
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from admixture_cache import PanelCacheError
from admixture_cache.distribution import (
    DEFAULT_GITHUB_REPO,
    CacheRelease,
    _default_cache_root,
    _fetch_sha256_expected,
    _find_manifest_root,
    download_cache,
    list_available_caches,
)

# ─── fixture helpers ─────────────────────────────────────────────────────


def _make_synthetic_cache_dir(
    tmp_path: Path, name: str = "synth",
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    """Build a minimal valid cache directory with the structure
    admixture-cache produces. The manifest is intentionally small;
    every required field is set so `load_cache_manifest` succeeds.

    ``extra_manifest`` injects additional manifest keys, e.g. a field a
    newer library version would write that this version does not know,
    to exercise the forward-compatible (``extra='ignore'``) load path."""
    cache = tmp_path / name
    cache.mkdir(parents=True)
    # Numeric files — admixture-cache's load_cached_p reads these via
    # numpy.loadtxt, so they need to be well-formed numbers but not
    # necessarily realistic values.
    (cache / "panel.3.P").write_text("0.5 0.5 0.5\n0.5 0.5 0.5\n")
    (cache / "panel.3.Q").write_text("1.0 0.0 0.0\n")
    (cache / "panel.bim").write_text("1\trs1\t0\t1\tA\tG\n")
    (cache / "restart_sd.json").write_text('{"per_cluster_max_sd": {}, '
                                          '"overall_max_sd": 0.0, '
                                          '"threshold": 0.02, '
                                          '"n_restarts": 1}\n')
    (cache / "cluster_order.json").write_text('{"cluster_order": ["A","B","C"]}\n')
    (cache / "build_logs").mkdir()
    manifest = {
        "schema_version": 1,
        "track": "regional", "continent": None,
        "panel_id": "synth", "panel_version": "v0",
        "panel_bim_sha256": "0" * 64,
        "clusters_yaml_sha256": "0" * 64,
        "k": 3,
        "admixture_version": "1.4.0",
        "seeds_used": [1], "best_seed": 1,
        "best_loglikelihood": -100.0,
        "restart_sd_max": 0.0,
        "cluster_order": ["A", "B", "C"],
        "geo_filter_yaml_shas": {},
        "pgen_samplebind_version": None,
        "build_wallclock_seconds": 1.0,
        "build_timestamp": "2026-05-26T00:00:00+00:00",
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (cache / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return cache


def _make_tarball(cache_dir: Path, *, flat: bool) -> bytes:
    """Pack a cache directory into a .tar.gz blob.

    `flat=True`: members at the tarball top level (./panel.K.P etc).
    `flat=False`: wrapped in a single dir (./<name>/panel.K.P etc).
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz, tarfile.open(fileobj=gz, mode="w") as tf:
        for item in cache_dir.rglob("*"):
            arcname_rel = item.relative_to(cache_dir)
            arcname = (
                str(arcname_rel) if flat
                else f"{cache_dir.name}/{arcname_rel}"
            )
            tf.add(item, arcname=arcname, recursive=False)
    return buf.getvalue()


def _fake_releases_payload(
    name: str = "regional_k21_aadr_v66_ho",
    versions: list[str] | None = None,
    extra_releases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Mimic the GitHub Releases JSON list response shape."""
    if versions is None:
        versions = ["v1"]
    out: list[dict[str, Any]] = []
    for v in versions:
        out.append({
            "tag_name": f"cache-{name}-{v}",
            "html_url": f"https://example.com/release/{v}",
            "published_at": "2026-05-26T12:00:00Z",
            "body": "release notes",
            "assets": [
                {
                    "name": f"{name}.tar.gz",
                    "size": 12345,
                    "browser_download_url":
                        f"https://example.com/{name}-{v}.tar.gz",
                },
                {
                    "name": f"{name}.tar.gz.sha256",
                    "size": 80,
                    "browser_download_url":
                        f"https://example.com/{name}-{v}.tar.gz.sha256",
                },
            ],
        })
    if extra_releases:
        out.extend(extra_releases)
    return out


def _patch_urlopen_chain(monkeypatch: pytest.MonkeyPatch,
                          responses: dict[str, bytes]) -> None:
    """Patch urllib.request.urlopen to return canned bytes per URL.

    `responses` maps URL → bytes (or callable that returns bytes given
    the request URL). Unknown URLs raise URLError.

    URL matching is query-string-tolerant: a registered URL of
    ``https://api.github.com/.../releases`` matches a request to
    ``https://api.github.com/.../releases?per_page=100``. This way
    distribution.py can append API parameters without forcing every
    test to know about them.
    """
    def fake_urlopen(req_or_url: Any, *_args: Any, **_kw: Any) -> MagicMock:
        url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
        base_url = url.split("?", 1)[0]
        if url in responses:
            payload = responses[url]
        elif base_url in responses:
            payload = responses[base_url]
        else:
            raise urllib.error.URLError(f"unexpected URL {url}")
        mock = MagicMock()
        mock.__enter__ = lambda self: mock
        mock.__exit__ = lambda *a: None
        # Stateful reader so streaming `resp.read(chunk)` works.
        state = {"pos": 0}

        def read(size: int = -1) -> bytes:
            start = state["pos"]
            if size < 0 or size > len(payload) - start:
                chunk = payload[start:]
            else:
                chunk = payload[start:start + size]
            state["pos"] = start + len(chunk)
            return chunk

        mock.read.side_effect = read
        # `headers.get("Link", "")` returns empty so the paginator
        # terminates after one page. `Content-Length` reflects the
        # canned payload length so progress + Content-Length
        # validation work as the production code expects.
        mock.headers.get = lambda key, default=None: {
            "Content-Length": str(len(payload)),
            "Link": "",
        }.get(key, default)
        return mock

    monkeypatch.setattr(
        "admixture_cache.distribution.urllib.request.urlopen",
        fake_urlopen,
    )


# ─── _default_cache_root ─────────────────────────────────────────────────


class TestDefaultCacheRoot:
    def test_env_var_overrides_default(self,
                                        monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
        monkeypatch.setenv("ADMIXTURE_CACHE_ROOT", str(tmp_path / "custom"))
        assert _default_cache_root() == tmp_path / "custom"

    def test_unset_env_falls_back_to_home(self,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ADMIXTURE_CACHE_ROOT", raising=False)
        root = _default_cache_root()
        assert root.parts[-2:] == (".admixture-cache", "caches")


# ─── _find_manifest_root ─────────────────────────────────────────────────


class TestFindManifestRoot:
    def test_flat_layout(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}")
        assert _find_manifest_root(tmp_path) == tmp_path

    def test_wrapped_layout(self, tmp_path: Path) -> None:
        sub = tmp_path / "wrapper"
        sub.mkdir()
        (sub / "manifest.json").write_text("{}")
        assert _find_manifest_root(tmp_path) == sub

    def test_no_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PanelCacheError, match="cannot locate manifest"):
            _find_manifest_root(tmp_path)

    def test_two_wrapper_dirs_raises(self, tmp_path: Path) -> None:
        """Ambiguous layout — refuse rather than guess."""
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "manifest.json").write_text("{}")
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "manifest.json").write_text("{}")
        with pytest.raises(PanelCacheError, match="cannot locate manifest"):
            _find_manifest_root(tmp_path)


# ─── _fetch_sha256_expected ──────────────────────────────────────────────


class TestFetchSha256:
    def test_bare_digest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        digest = "a" * 64
        _patch_urlopen_chain(
            monkeypatch, {"https://example.com/sha": digest.encode("ascii")},
        )
        assert _fetch_sha256_expected(
            "https://example.com/sha", timeout=5.0,
        ) == digest

    def test_coreutils_sha256sum_format(self,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        digest = "b" * 64
        content = f"{digest}  file.tar.gz\n".encode("ascii")
        _patch_urlopen_chain(
            monkeypatch, {"https://example.com/sha": content},
        )
        assert _fetch_sha256_expected(
            "https://example.com/sha", timeout=5.0,
        ) == digest

    def test_invalid_content_raises(self,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_urlopen_chain(
            monkeypatch, {"https://example.com/sha": b"not a hex digest"},
        )
        with pytest.raises(PanelCacheError,
                            match="64-character hex sha256"):
            _fetch_sha256_expected("https://example.com/sha", timeout=5.0)

    def test_network_error_wrapped(self,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(*_args: Any, **_kw: Any) -> Any:
            raise urllib.error.URLError("connection refused")
        monkeypatch.setattr(
            "admixture_cache.distribution.urllib.request.urlopen",
            fake_urlopen,
        )
        with pytest.raises(PanelCacheError, match="cannot reach"):
            _fetch_sha256_expected("https://example.com/sha", timeout=5.0)


# ─── list_available_caches ───────────────────────────────────────────────


class TestListAvailableCaches:
    def test_returns_releases_matching_tag_convention(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = _fake_releases_payload(versions=["v1", "v2"])
        url = f"https://api.github.com/repos/{DEFAULT_GITHUB_REPO}/releases"
        _patch_urlopen_chain(monkeypatch, {url: json.dumps(payload).encode()})
        releases = list_available_caches()
        assert len(releases) == 2
        assert {r.version for r in releases} == {"v1", "v2"}
        assert all(r.name == "regional_k21_aadr_v66_ho" for r in releases)
        # version_number sort yields v2 first
        assert max(releases, key=lambda r: r.version_number).version == "v2"

    def test_accepts_uppercase_continent_in_cache_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ancestral_cluster cache keys fold the continent verbatim, and some
        continents carry uppercase (e.g. ac_W_Eurasia_k4_<sha>). The tag grammar
        must accept that casing or such a release is silently undiscoverable.
        GitHub tags are case-sensitive and the runtime looks the key up with the
        same casing, so an uppercase-name release round-trips exactly."""
        name = "ac_W_Eurasia_k4_28db6795"
        payload = _fake_releases_payload(name=name, versions=["v1"])
        url = f"https://api.github.com/repos/{DEFAULT_GITHUB_REPO}/releases"
        _patch_urlopen_chain(monkeypatch, {url: json.dumps(payload).encode()})
        releases = list_available_caches()
        assert len(releases) == 1
        assert releases[0].name == name
        assert releases[0].version == "v1"

    def test_skips_releases_without_matching_tag(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Releases not matching `cache-<name>-<version>` are ignored
        (e.g. library version tags v1.2.0)."""
        payload = _fake_releases_payload(versions=["v1"])
        payload.append({
            "tag_name": "v1.2.0",  # not a cache release
            "html_url": "", "published_at": "2026-01-01T00:00:00Z",
            "body": "", "assets": [],
        })
        url = f"https://api.github.com/repos/{DEFAULT_GITHUB_REPO}/releases"
        _patch_urlopen_chain(monkeypatch, {url: json.dumps(payload).encode()})
        releases = list_available_caches()
        assert len(releases) == 1
        assert releases[0].version == "v1"

    def test_skips_releases_missing_required_assets(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If a cache-tagged release is missing the .sha256 sibling,
        it's skipped (incomplete publish — not safe to download)."""
        payload = _fake_releases_payload(versions=["v1"])
        # Strip the .sha256 asset
        payload[0]["assets"] = [
            a for a in payload[0]["assets"]
            if not a["name"].endswith(".sha256")
        ]
        url = f"https://api.github.com/repos/{DEFAULT_GITHUB_REPO}/releases"
        _patch_urlopen_chain(monkeypatch, {url: json.dumps(payload).encode()})
        releases = list_available_caches()
        assert releases == []

    def test_http_error_wrapped_as_panel_cache_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_urlopen(*_args: Any, **_kw: Any) -> Any:
            raise urllib.error.HTTPError(
                url="https://api.github.com/...",
                code=404, msg="Not Found",
                hdrs=None, fp=None,  # type: ignore[arg-type]
            )
        monkeypatch.setattr(
            "admixture_cache.distribution.urllib.request.urlopen",
            fake_urlopen,
        )
        with pytest.raises(PanelCacheError, match="GitHub API returned 404"):
            list_available_caches(github_repo="bogus/repo")


# ─── download_cache (full pipeline) ──────────────────────────────────────


class TestDownloadCacheEndToEnd:
    """Exercise the full download → verify → extract → validate path
    with mocked urllib but a real on-disk tarball."""

    def _setup_mocks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cache_dir: Path,
        *,
        flat: bool = True,
        name: str = "synth",
        corrupt_sha: bool = False,
    ) -> None:
        tarball = _make_tarball(cache_dir, flat=flat)
        sha = hashlib.sha256(tarball).hexdigest()
        if corrupt_sha:
            sha = "0" * 64
        api_url = f"https://api.github.com/repos/{DEFAULT_GITHUB_REPO}/releases"
        tarball_url = f"https://example.com/{name}.tar.gz"
        sha_url = f"https://example.com/{name}.tar.gz.sha256"
        payload = [
            {
                "tag_name": f"cache-{name}-v1",
                "html_url": "https://example.com/release/v1",
                "published_at": "2026-05-26T12:00:00Z",
                "body": "",
                "assets": [
                    {"name": f"{name}.tar.gz", "size": len(tarball),
                     "browser_download_url": tarball_url},
                    {"name": f"{name}.tar.gz.sha256", "size": 80,
                     "browser_download_url": sha_url},
                ],
            },
        ]
        _patch_urlopen_chain(monkeypatch, {
            api_url: json.dumps(payload).encode(),
            tarball_url: tarball,
            sha_url: sha.encode("ascii"),
        })

    def test_happy_path_flat_tarball(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        cache = _make_synthetic_cache_dir(tmp_path / "src", name="synth")
        self._setup_mocks(monkeypatch, cache, flat=True)
        installed = download_cache(
            "synth", cache_root=tmp_path / "root",
        )
        assert installed == tmp_path / "root" / "synth"
        assert (installed / "manifest.json").is_file()
        assert (installed / "panel.3.P").is_file()

    def test_forward_compat_unknown_manifest_field(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """A cache published by a NEWER admixture-cache carries manifest
        fields this (older) consumer does not know. download_cache must
        still install it, because PanelCacheManifest is extra='ignore' and
        the post-extract load_cache_manifest tolerates the unknown key.
        Guards the exact regression the extra='ignore' change prevents on
        its load-bearing path (without this, extra='forbid' rejected the
        SHA-valid tarball at distribution.py's manifest-validation step)."""
        from admixture_cache import load_cache_manifest

        cache = _make_synthetic_cache_dir(
            tmp_path / "src", name="synth",
            extra_manifest={"future_field_from_v9": {"nested": True}},
        )
        self._setup_mocks(monkeypatch, cache, flat=True)
        installed = download_cache("synth", cache_root=tmp_path / "root")
        assert (installed / "manifest.json").is_file()
        # The installed manifest also loads cleanly via the library path.
        m = load_cache_manifest(installed)
        assert m.panel_id == "synth"
        assert not hasattr(m, "future_field_from_v9")

    def test_happy_path_wrapped_tarball(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Tarball packed with a wrapper dir is handled equivalently."""
        cache = _make_synthetic_cache_dir(tmp_path / "src", name="synth")
        self._setup_mocks(monkeypatch, cache, flat=False)
        installed = download_cache(
            "synth", cache_root=tmp_path / "root",
        )
        assert (installed / "manifest.json").is_file()

    def test_sha256_mismatch_raises_no_install(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        cache = _make_synthetic_cache_dir(tmp_path / "src", name="synth")
        self._setup_mocks(monkeypatch, cache, flat=True, corrupt_sha=True)
        with pytest.raises(PanelCacheError, match="SHA-256 mismatch"):
            download_cache("synth", cache_root=tmp_path / "root")
        # Target dir should NOT exist after the failure.
        assert not (tmp_path / "root" / "synth").exists()

    def test_existing_target_refused_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        existing = tmp_path / "root" / "synth"
        existing.mkdir(parents=True)
        (existing / "marker.txt").write_text("preserve me")

        cache = _make_synthetic_cache_dir(tmp_path / "src", name="synth")
        self._setup_mocks(monkeypatch, cache, flat=True)
        with pytest.raises(PanelCacheError, match="already exists"):
            download_cache("synth", cache_root=tmp_path / "root")
        # Existing untouched.
        assert (existing / "marker.txt").read_text() == "preserve me"

    def test_force_overwrites_existing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        existing = tmp_path / "root" / "synth"
        existing.mkdir(parents=True)
        (existing / "stale.txt").write_text("old")

        cache = _make_synthetic_cache_dir(tmp_path / "src", name="synth")
        self._setup_mocks(monkeypatch, cache, flat=True)
        installed = download_cache(
            "synth", cache_root=tmp_path / "root", force=True,
        )
        # New cache installed; stale file gone.
        assert (installed / "manifest.json").is_file()
        assert not (installed / "stale.txt").exists()

    def test_unknown_name_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        cache = _make_synthetic_cache_dir(tmp_path / "src", name="synth")
        self._setup_mocks(monkeypatch, cache, flat=True)
        with pytest.raises(PanelCacheError, match="no published cache"):
            download_cache("nonexistent", cache_root=tmp_path / "root")

    def test_progress_callback_invoked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        cache = _make_synthetic_cache_dir(tmp_path / "src", name="synth")
        self._setup_mocks(monkeypatch, cache, flat=True)
        observations: list[tuple[int, int]] = []
        download_cache(
            "synth", cache_root=tmp_path / "root",
            progress=lambda dl, total: observations.append((dl, total)),
        )
        assert observations, "progress callback never invoked"
        # Last observation should show full progress.
        last_dl, last_total = observations[-1]
        assert last_dl == last_total

    def test_pinned_version_selected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """`version="v1"` picks v1 even when v2 exists."""
        cache = _make_synthetic_cache_dir(tmp_path / "src", name="synth")
        tarball = _make_tarball(cache, flat=True)
        sha = hashlib.sha256(tarball).hexdigest()
        api_url = f"https://api.github.com/repos/{DEFAULT_GITHUB_REPO}/releases"
        v1_url = "https://example.com/synth-v1.tar.gz"
        v1_sha_url = "https://example.com/synth-v1.tar.gz.sha256"
        v2_url = "https://example.com/synth-v2.tar.gz"
        v2_sha_url = "https://example.com/synth-v2.tar.gz.sha256"
        payload = [
            {
                "tag_name": "cache-synth-v2",
                "html_url": "", "published_at": "2026-05-26T12:00:00Z",
                "body": "", "assets": [
                    {"name": "synth.tar.gz", "size": 0,
                     "browser_download_url": v2_url},
                    {"name": "synth.tar.gz.sha256", "size": 0,
                     "browser_download_url": v2_sha_url},
                ],
            },
            {
                "tag_name": "cache-synth-v1",
                "html_url": "", "published_at": "2026-04-01T12:00:00Z",
                "body": "", "assets": [
                    {"name": "synth.tar.gz", "size": len(tarball),
                     "browser_download_url": v1_url},
                    {"name": "synth.tar.gz.sha256", "size": 0,
                     "browser_download_url": v1_sha_url},
                ],
            },
        ]
        _patch_urlopen_chain(monkeypatch, {
            api_url: json.dumps(payload).encode(),
            v1_url: tarball, v1_sha_url: sha.encode("ascii"),
        })
        # Pinning to v1 only hits v1 URLs; v2 URLs aren't in the mock.
        installed = download_cache(
            "synth", cache_root=tmp_path / "root", version="v1",
        )
        assert installed.exists()


class TestCacheReleaseDataclass:
    def test_version_number_parses_v_prefix(self) -> None:
        r = CacheRelease(
            name="x", version="v17", tag="cache-x-v17",
            tarball_url="", sha256_url="", size_bytes=0,
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
            html_url="", notes="",
        )
        assert r.version_number == 17


class TestNameValidationGuard:
    """`download_cache(name=...)` from the Python API accepts arbitrary
    strings; the guard enforces flat-directory-identifier semantics."""

    @pytest.mark.parametrize("bad_name", [
        # `..` traversal (single or multi-segment).
        "../escapes",
        "../../etc/passwd",
        # Path separators that would create unwanted nesting.
        "subdir/../escape",
        "a/b/c",
        "a\\b\\c",
        # Reserved values.
        "",
        ".",
        "..",
        # Hidden — reserved for our tempfile/lockfile pattern.
        ".secret_cache",
        # Absolute path.
        "/etc/evil",
        "/tmp/somewhere",
    ])
    def test_invalid_names_rejected(
        self, monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path, bad_name: str,
    ) -> None:
        # No mocks: the guard fires BEFORE any network I/O.
        with pytest.raises(
            PanelCacheError,
            match=r"not a valid cache identifier|resolves outside",
        ):
            download_cache(bad_name, cache_root=tmp_path)


class TestFindManifestRootMacOSResourceFork:
    """`__MACOSX/` resource-fork siblings (from macOS Finder
    Compress→unzip→tar pipelines) shouldn't trip the 'ambiguous
    layout' branch."""

    def test_macosx_sibling_filtered(self, tmp_path: Path) -> None:
        from admixture_cache.distribution import _find_manifest_root

        # Real wrapper dir.
        wrapper = tmp_path / "regional_k21"
        wrapper.mkdir()
        (wrapper / "manifest.json").write_text("{}")
        # macOS resource-fork sibling — should be filtered out.
        macosx = tmp_path / "__MACOSX"
        macosx.mkdir()
        (macosx / "._regional_k21").write_text("metadata")

        root = _find_manifest_root(tmp_path)
        assert root == wrapper

    def test_dotfile_sibling_filtered(self, tmp_path: Path) -> None:
        """Hidden dot-named dirs (e.g. `.git/`, `.DS_Store/`) also
        filtered — they're not ours."""
        from admixture_cache.distribution import _find_manifest_root

        wrapper = tmp_path / "cache"
        wrapper.mkdir()
        (wrapper / "manifest.json").write_text("{}")
        hidden = tmp_path / ".hidden_dir"
        hidden.mkdir()

        root = _find_manifest_root(tmp_path)
        assert root == wrapper


class TestConcurrentInstallLock:
    """`_exclusive_lock` serializes concurrent `download_cache()` calls
    on the same name. Verified by attempting two acquisitions of the
    same lock file — the second blocks until the first releases."""

    def test_lock_serializes_holders(self, tmp_path: Path) -> None:
        from admixture_cache.distribution import _exclusive_lock

        lock_path = tmp_path / ".test.lock"
        order: list[str] = []

        def first_holder() -> None:
            with _exclusive_lock(lock_path):
                order.append("first_acquired")
                # Hold briefly so `second_holder` has time to block.
                time.sleep(0.1)
                order.append("first_released")

        def second_holder() -> None:
            # Start slightly after first_holder so it acquires first.
            time.sleep(0.02)
            with _exclusive_lock(lock_path):
                order.append("second_acquired")

        t1 = threading.Thread(target=first_holder)
        t2 = threading.Thread(target=second_holder)
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        # The lock guarantees first_released < second_acquired.
        assert order == [
            "first_acquired", "first_released", "second_acquired",
        ], f"unexpected order: {order}"


class TestSlowLorisBudget:
    """A server streaming bytes within the per-read timeout window
    but exceeding the total wall-clock budget must trigger a
    PanelCacheError, not hang forever."""

    def test_total_wall_clock_budget_enforced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        # 100KB tarball; budget 0.05s; chunk reads sleep 0.03s each
        # so >2 chunks exceeds budget.
        tarball = b"x" * (100 * 1024)
        sha = hashlib.sha256(tarball).hexdigest()
        api_url = f"https://api.github.com/repos/{DEFAULT_GITHUB_REPO}/releases"
        tarball_url = "https://example.com/foo.tar.gz"
        sha_url = "https://example.com/foo.tar.gz.sha256"
        payload = [{
            "tag_name": "cache-foo-v1",
            "html_url": "", "published_at": "2026-05-26T12:00:00Z",
            "body": "",
            "assets": [
                {"name": "foo.tar.gz", "size": len(tarball),
                 "browser_download_url": tarball_url},
                {"name": "foo.tar.gz.sha256", "size": 80,
                 "browser_download_url": sha_url},
            ],
        }]

        def slow_urlopen(req_or_url: Any, *_a: Any, **_kw: Any) -> MagicMock:
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            base = url.split("?", 1)[0]
            if base == api_url:
                pl: bytes = json.dumps(payload).encode()
            elif url == sha_url:
                pl = sha.encode("ascii")
            elif url == tarball_url:
                pl = tarball
            else:
                raise urllib.error.URLError(f"unexpected URL {url}")
            mock = MagicMock()
            mock.__enter__ = lambda self: mock
            mock.__exit__ = lambda *a: None
            state = {"pos": 0}

            def read(size: int = -1) -> bytes:
                # Slow loris: sleep before each chunk read.
                if url == tarball_url:
                    time.sleep(0.03)
                start = state["pos"]
                chunk = pl[start:start + size] if size > 0 else pl[start:]
                state["pos"] = start + len(chunk)
                return chunk

            mock.read.side_effect = read
            mock.headers.get = lambda key, default=None: {
                "Content-Length": str(len(pl)),
                "Link": "",
            }.get(key, default)
            return mock

        monkeypatch.setattr(
            "admixture_cache.distribution.urllib.request.urlopen",
            slow_urlopen,
        )
        monkeypatch.setenv(
            "ADMIXTURE_CACHE_DOWNLOAD_BUDGET_SECONDS", "0.05",
        )

        with pytest.raises(PanelCacheError, match="wall-clock budget"):
            download_cache(
                "foo", cache_root=tmp_path / "root",
            )
