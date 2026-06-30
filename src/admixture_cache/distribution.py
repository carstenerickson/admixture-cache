"""Canonical published-cache discovery + download.

Operators publish a cache as a GitHub Release of admixture-cache (or
a fork) using the tag convention ``cache-<name>-<version>``. Each
release attaches two assets:

- ``<name>.tar.gz`` — the cache directory contents (panel.K.P,
  panel.K.Q, panel.bim, manifest.json, restart_sd.json,
  cluster_order.json, build_logs/). The tarball may have either
  a flat layout (files at the top level) or a single wrapper dir;
  the downloader auto-detects via :func:`_find_manifest_root`.
- ``<name>.tar.gz.sha256`` — the hex SHA-256 digest of the tarball,
  on a single line (with or without a filename suffix). Verified
  end-to-end before the cache is installed into ``cache_root``.

Operators publishing a new canonical cache: see docs/PUBLISH_CACHE.md.

The downloader:

1. Queries GitHub Releases via the public, unauthenticated REST API
   (``GET /repos/<owner>/<repo>/releases``) — no token required.
2. Filters releases by tag prefix ``cache-`` to identify canonical
   caches.
3. For an exact-name match, downloads the tarball with streaming
   SHA-256 verification (memory bound by chunk size, not tarball
   size — caches can be many GB).
4. Extracts into a temp dir, validates the manifest by loading it,
   then atomic-renames the validated cache into
   ``<cache_root>/<name>/``. A partial download or extract failure
   never leaves a half-installed cache that consumers might read.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import http.client
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from admixture_cache.errors import PanelCacheError
from admixture_cache.io import load_cache_manifest

logger = logging.getLogger(__name__)


# Tag convention: cache-<name>-<version>
# - <name>: ASCII letters (upper OR lower), digits, underscores. Lowercase is
#   the convention for panel/track ids (e.g. `regional_k21_aadr_v66_ho`), but
#   ancestral_cluster cache keys fold the continent verbatim and some continents
#   carry uppercase (e.g. `ac_W_Eurasia_k4_<sha>`); the grammar accepts uppercase
#   so those caches stay discoverable. GitHub tags are case-sensitive and the
#   runtime looks the key up with the same casing, so the round-trip is exact.
# - <version>: `v` followed by one or more digits (`v1`, `v2`, …)
# Split on the LAST hyphen so names containing hyphens parse correctly
# (we recommend underscores to avoid this entirely).
# Use re.fullmatch via $ + \Z-equivalent anchors so trailing newlines
# in tag names don't accidentally match (defensive — git refuses
# newlines in refs but the regex should be strict).
_TAG_PATTERN = re.compile(
    r"\Acache-([A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*)-(v\d+)\Z",
)

# Default GitHub repo to query for releases. Operators publishing
# forked / private caches can override via the CLI flag or the
# Python API parameter.
DEFAULT_GITHUB_REPO = "carstenerickson/admixture-cache"

# Default download chunk size — 64 KiB balances syscall overhead with
# progress-callback granularity. Tested against ~1 GB tarballs.
_DOWNLOAD_CHUNK_SIZE = 64 * 1024

# Max releases per GitHub API request. 100 is GitHub's documented
# maximum; reduces the number of pages we need to follow for large
# release histories.
_GITHUB_API_PAGE_SIZE = 100

# RFC 5988 Link-header rel="next" pattern, e.g.:
#   <https://api.github.com/.../releases?page=2&per_page=100>; rel="next"
_LINK_NEXT_PATTERN = re.compile(
    r'<([^>]+)>;\s*rel="next"',
)


def _paginated_get(
    url: str, *, timeout: float,
) -> Iterator[list[dict[str, Any]]]:
    """Yield successive JSON pages from a paginated GitHub REST endpoint.

    Follows ``Link: <…>; rel="next"`` until exhausted. Each yielded
    value is the parsed JSON array of that page (typically up to
    ``per_page=100`` items).
    """
    next_url: str | None = url
    while next_url:
        try:
            req = urllib.request.Request(
                next_url, headers={"Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
                link_header = resp.headers.get("Link", "")
        except urllib.error.HTTPError as exc:
            raise PanelCacheError(
                f"_paginated_get: GitHub API returned {exc.code} for "
                f"{next_url}; check the repo name + your network",
            ) from exc
        except (urllib.error.URLError, TimeoutError,
                http.client.HTTPException) as exc:
            raise PanelCacheError(
                f"_paginated_get: cannot reach {next_url}: {exc}",
            ) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise PanelCacheError(
                f"_paginated_get: malformed JSON from {next_url}: {exc}",
            ) from exc
        yield payload if isinstance(payload, list) else []
        # Parse Link header to find the next page, if any.
        next_url = None
        for link in link_header.split(","):
            m = _LINK_NEXT_PATTERN.search(link)
            if m:
                next_url = m.group(1)
                break


@dataclass(frozen=True)
class CacheRelease:
    """One published cache version on GitHub Releases."""

    name: str
    version: str  # "v1", "v2", ...
    tag: str
    tarball_url: str
    sha256_url: str
    size_bytes: int
    published_at: datetime
    html_url: str
    notes: str

    @property
    def version_number(self) -> int:
        """Integer parse of ``self.version`` for max() / sort()."""
        return int(self.version.removeprefix("v"))


def _default_cache_root() -> Path:
    """Where caches install if the caller doesn't pass ``cache_root``.

    Precedence:
    1. ``ADMIXTURE_CACHE_ROOT`` environment variable
    2. ``~/.admixture-cache/caches/``
    """
    env = os.environ.get("ADMIXTURE_CACHE_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".admixture-cache" / "caches"


def list_available_caches(
    github_repo: str = DEFAULT_GITHUB_REPO,
    *,
    timeout: float = 30.0,
) -> list[CacheRelease]:
    """List canonical caches published as GitHub Releases.

    Queries ``GET /repos/<repo>/releases`` (public, unauthenticated)
    and returns one :class:`CacheRelease` per release whose tag
    matches ``cache-<name>-<version>`` AND whose assets include both
    ``<name>.tar.gz`` and ``<name>.tar.gz.sha256``. Releases that
    don't satisfy both filters are silently skipped — they're not
    canonical caches.

    Pagination is followed automatically (``per_page=100`` + ``Link``
    header rel="next"); repos with hundreds of releases work
    transparently.

    Multiple versions of the same name are returned as separate
    :class:`CacheRelease` entries. :func:`download_cache` resolves
    "latest" via :attr:`CacheRelease.version_number`.
    """
    url = (
        f"https://api.github.com/repos/{github_repo}/releases"
        f"?per_page={_GITHUB_API_PAGE_SIZE}"
    )
    out: list[CacheRelease] = []
    for page in _paginated_get(url, timeout=timeout):
        for rel in page:
            match = _TAG_PATTERN.fullmatch(rel.get("tag_name", ""))
            if not match:
                continue
            name, version = match.groups()
            tarball_url: str | None = None
            sha256_url: str | None = None
            size_bytes = 0
            for asset in rel.get("assets", []):
                asset_name = asset.get("name", "")
                if asset_name == f"{name}.tar.gz":
                    tarball_url = asset["browser_download_url"]
                    size_bytes = int(asset.get("size", 0))
                elif asset_name == f"{name}.tar.gz.sha256":
                    sha256_url = asset["browser_download_url"]
            if not (tarball_url and sha256_url):
                logger.debug(
                    "list_available_caches: skipping %s — missing "
                    "tarball or sha256 asset", rel.get("tag_name"),
                )
                continue
            published_str = rel.get("published_at", "")
            # GitHub returns ISO-8601 with a trailing Z; convert to a
            # timezone-aware datetime. Fallback uses tz=UTC so the
            # comparison between fallback + real-release datetimes
            # doesn't raise TypeError on mixed naive/aware.
            published = (
                datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                if published_str else datetime.fromtimestamp(0, tz=UTC)
            )
            out.append(CacheRelease(
                name=name, version=version, tag=rel["tag_name"],
                tarball_url=tarball_url, sha256_url=sha256_url,
                size_bytes=size_bytes, published_at=published,
                html_url=rel.get("html_url", ""),
                notes=rel.get("body", "") or "",
            ))
    return out


def _fetch_sha256_expected(url: str, *, timeout: float) -> str:
    """Download a ``.sha256`` companion file and return the hex digest.

    File format is permissive: we accept either a bare 64-char hex
    digest on the first line, OR the GNU coreutils ``sha256sum``
    output format (``<digest>  <filename>``). Whitespace tolerated.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            content = resp.read().decode("ascii", errors="strict")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PanelCacheError(
            f"_fetch_sha256_expected: cannot reach {url}: {exc}",
        ) from exc
    except UnicodeDecodeError as exc:
        raise PanelCacheError(
            f"_fetch_sha256_expected: {url} contains non-ASCII bytes; "
            f"expected a hex sha256 digest",
        ) from exc
    # Take the first whitespace-separated token of the first line.
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    digest = first_line.split()[0] if first_line else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise PanelCacheError(
            f"_fetch_sha256_expected: {url} content {content!r} doesn't "
            f"contain a 64-character hex sha256 digest on the first line",
        )
    return digest.lower()


def _find_manifest_root(extract_dir: Path) -> Path:
    """Locate the directory containing ``manifest.json`` inside the
    extracted tarball.

    Tarballs may be packed two ways:

    - **Flat**: cache files at the tarball top level (`./panel.K.P`,
      `./manifest.json`, …). `_find_manifest_root` returns
      `extract_dir`.
    - **Wrapped**: cache files inside a single subdirectory
      (`./regional_k21_aadr_v66_ho/manifest.json`). Common when
      packed with `tar -czf cache.tar.gz cache_dir/`.
      `_find_manifest_root` returns that subdirectory.

    Other layouts (manifest deeper than one level, multiple manifests)
    raise :class:`PanelCacheError`.
    """
    direct = extract_dir / "manifest.json"
    if direct.is_file():
        return extract_dir
    # Single child directory containing the manifest. Filter out
    # macOS resource-fork siblings (`__MACOSX/` from Finder
    # Compress→unzip→tar pipelines) and hidden directories that
    # aren't ours — neither would contain a real cache and counting
    # them would falsely trip the "ambiguous layout" branch below.
    children = [
        p for p in extract_dir.iterdir()
        if p.is_dir() and p.name != "__MACOSX" and not p.name.startswith(".")
    ]
    if len(children) == 1 and (children[0] / "manifest.json").is_file():
        return children[0]
    raise PanelCacheError(
        f"_find_manifest_root: cannot locate manifest.json in extracted "
        f"tarball at {extract_dir}; expected either a flat layout or a "
        f"single wrapper directory. Found children: "
        f"{sorted(p.name for p in extract_dir.iterdir())}",
    )


def _safe_extract_tarball(tar_path: Path, dest: Path) -> None:
    """Extract ``tar_path`` into ``dest`` using tarfile's data filter
    (Python 3.12+) to reject member paths containing absolute paths,
    `..` traversal, or non-data entries (symlinks, device nodes)."""
    with tarfile.open(tar_path, mode="r:*") as tf:
        # `filter="data"` is the strictest option available on 3.12+;
        # rejects absolute paths, parent-dir traversal, and any
        # non-regular-file entries that could escape `dest`.
        tf.extractall(dest, filter="data")


def download_cache(
    name: str,
    *,
    cache_root: Path | None = None,
    github_repo: str = DEFAULT_GITHUB_REPO,
    version: str | None = None,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
    timeout: float = 600.0,
) -> Path:
    """Download a canonical cache to ``<cache_root>/<name>/``.

    Parameters
    ----------
    name
        Cache name as published — see :func:`list_available_caches`.
    cache_root
        Where to install the cache. Defaults to
        ``$ADMIXTURE_CACHE_ROOT`` if set, else
        ``~/.admixture-cache/caches/``.
    github_repo
        ``owner/repo`` whose Releases to query. Defaults to
        ``carstenerickson/admixture-cache``.
    version
        Specific version to download (e.g. ``"v2"``). Defaults to
        the highest-numbered version available.
    force
        If ``True``, overwrite an existing cache at the target path.
        If ``False`` (default) and the target exists, raise
        :class:`PanelCacheError`.
    progress
        Optional callback ``(downloaded_bytes, total_bytes)``
        invoked after each chunk. Useful for progress bars.
        ``total_bytes`` may be 0 if the server doesn't report
        ``Content-Length``.
    timeout
        Per-request timeout for both the API call and the tarball
        download (seconds). Default 10 minutes — caches can be GB-sized.

    Returns
    -------
    Path
        Absolute path to the installed cache directory (suitable
        for passing as ``cache_dir=...`` to
        :func:`project_target`).

    Raises
    ------
    PanelCacheError
        On any of: cache name not found, version not found, network
        error, SHA-256 mismatch, malformed tarball, manifest
        validation failure post-extract, existing cache without
        ``force=True``.
    """
    cache_root = (cache_root or _default_cache_root()).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    # Cache `name` must be a single flat directory identifier. Reject
    # path separators, `..` traversal, and absolute paths — all three
    # break the "every cache is exactly one directory directly under
    # cache_root" invariant. From the CLI, `name` comes from a
    # regex-sanitized tag name; from the Python API, it's
    # caller-controlled. The guard fires before any network I/O.
    if (
        "/" in name
        or "\\" in name
        or name in ("", ".", "..")
        or name.startswith(".")  # hidden — reserved for tempfiles
        or Path(name).is_absolute()
    ):
        raise PanelCacheError(
            f"download_cache: name {name!r} is not a valid cache "
            f"identifier (must be a flat directory name with no path "
            f"separators, leading dot, or '..' segment).",
        )
    target_dir = (cache_root / name).resolve()
    if not target_dir.is_relative_to(cache_root):
        raise PanelCacheError(
            f"download_cache: name {name!r} resolves outside cache_root "
            f"{cache_root} (target would be {target_dir}); refusing.",
        )
    if target_dir.exists() and not force:
        raise PanelCacheError(
            f"download_cache: target {target_dir} already exists; pass "
            f"force=True to overwrite (or remove the directory first)",
        )

    # Acquire an exclusive flock on a per-name lockfile under
    # cache_root. Serializes concurrent `download_cache(name=X)` calls
    # on the same machine, preventing the TOCTOU between the
    # exists()-check above and the eventual rename. Cross-machine
    # concurrency (e.g. shared NFS cache_root) is not protected —
    # operators should arrange their own coordination there.
    lock_path = cache_root / f".{name}.lock"
    with _exclusive_lock(lock_path):
        # Re-check existence inside the lock; if another process just
        # installed the cache, we either return the existing path
        # (force=False) or proceed to overwrite (force=True).
        if target_dir.exists() and not force:
            raise PanelCacheError(
                f"download_cache: target {target_dir} appeared during "
                f"contention (concurrent install completed); pass "
                f"force=True to overwrite",
            )
        try:
            return _do_download_install(
                name=name, cache_root=cache_root, target_dir=target_dir,
                github_repo=github_repo, version=version,
                progress=progress, timeout=timeout,
            )
        except PanelCacheError:
            raise
        except (
            urllib.error.URLError, TimeoutError, http.client.HTTPException,
            tarfile.TarError, OSError, ValueError,
        ) as exc:
            # Catch the long tail of network / disk / format errors
            # that aren't already wrapped — present a single
            # PanelCacheError surface to the caller, preserving the
            # original exception in __cause__.
            raise PanelCacheError(
                f"download_cache: unexpected failure installing {name!r}: "
                f"{type(exc).__name__}: {exc}",
            ) from exc


@contextlib.contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Acquire an exclusive `fcntl.flock` on `lock_path` for the
    duration of the with-block. POSIX only — the library declares
    Linux + macOS support in `pyproject.toml [project] classifiers`."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # `O_CREAT | O_RDWR` so the file exists when we flock it; no
    # truncation so we don't fight other processes for the same fd.
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _do_download_install(
    *,
    name: str,
    cache_root: Path,
    target_dir: Path,
    github_repo: str,
    version: str | None,
    progress: Callable[[int, int], None] | None,
    timeout: float,
) -> Path:
    """Inner body of :func:`download_cache`, called under the per-name
    exclusive lock. Split out so the outer function can wrap the
    long-tail exception types into ``PanelCacheError``."""
    # Discover available releases for this name.
    available = list_available_caches(github_repo=github_repo, timeout=timeout)
    matching = [r for r in available if r.name == name]
    if not matching:
        names = sorted({r.name for r in available})
        raise PanelCacheError(
            f"download_cache: no published cache named {name!r} at "
            f"{github_repo}; available: {names or '(none)'}",
        )
    if version is not None:
        matching = [r for r in matching if r.version == version]
        if not matching:
            versions = sorted({r.version for r in available if r.name == name})
            raise PanelCacheError(
                f"download_cache: cache {name!r} version {version!r} not "
                f"found at {github_repo}; available versions: {versions}",
            )
    # Pick the highest-numbered version (default) or the
    # operator-pinned version (single-element matching).
    release = max(matching, key=lambda r: r.version_number)
    logger.info(
        "download_cache: resolved %s → %s (tarball %d bytes, published %s)",
        name, release.tag, release.size_bytes,
        release.published_at.isoformat(),
    )

    # Fetch the SHA-256 companion first — it's tiny (~80 bytes) so a
    # network failure here happens before we commit to a multi-GB
    # tarball download.
    expected_sha256 = _fetch_sha256_expected(release.sha256_url, timeout=timeout)

    # Stream the tarball into a tempfile within cache_root (same
    # filesystem as the target so the final rename is atomic), hashing
    # as we go. The temp file is unlinked on any failure (including
    # KeyboardInterrupt — see the BaseException-catching cleanup below).
    # ruff SIM115 wants a context manager here, but we need the tempfile
    # to OUTLIVE this block — we close it after streaming, then re-open
    # via `tar_path` for extraction. delete=False means we own the unlink.
    tmp_handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        dir=cache_root, prefix=f".{name}-",
        suffix=".tar.gz.tmp", delete=False,
    )
    tmp_path = Path(tmp_handle.name)
    hasher = hashlib.sha256()
    downloaded = 0
    extract_dir: Path | None = None
    try:
        # ── Streaming download with hashing + slow-loris budget ──
        # The urllib timeout applies PER-READ (socket recv()), not
        # per-total. Track wall-clock time separately so a server
        # that drips bytes within the per-read window can't hold
        # the download open indefinitely.
        wall_start = time.monotonic()
        # Budget caps total wall-clock at 10× the per-read timeout
        # by default. Tunable via env if operators have unusual
        # network conditions; defaults handle GitHub-class servers
        # comfortably (1 GB at 10 MB/s = 100s; 10× the 600s default
        # leaves headroom).
        wall_budget = float(
            os.environ.get("ADMIXTURE_CACHE_DOWNLOAD_BUDGET_SECONDS", "")
            or timeout * 10,
        )
        try:
            req = urllib.request.Request(
                release.tarball_url,
                headers={"Accept": "application/octet-stream"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # Some servers omit Content-Length on chunked transfer;
                # fall back to the API-reported size. A malformed value
                # (non-numeric, negative) also falls back.
                content_length = resp.headers.get("Content-Length")
                try:
                    total = int(content_length) if content_length else 0
                    if total <= 0:
                        total = release.size_bytes
                except ValueError:
                    total = release.size_bytes
                while True:
                    chunk = resp.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    elapsed = time.monotonic() - wall_start
                    if elapsed > wall_budget:
                        raise PanelCacheError(
                            f"download_cache: total wall-clock budget "
                            f"({wall_budget:.0f}s) exceeded after "
                            f"{downloaded} of {total} bytes — server is "
                            f"streaming too slowly. Override via "
                            f"ADMIXTURE_CACHE_DOWNLOAD_BUDGET_SECONDS.",
                        )
                    tmp_handle.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
        except (urllib.error.URLError, TimeoutError,
                http.client.HTTPException) as exc:
            raise PanelCacheError(
                f"download_cache: tarball download failed at "
                f"{downloaded} bytes: {exc}",
            ) from exc
        finally:
            tmp_handle.close()

        actual_sha256 = hasher.hexdigest()
        if actual_sha256 != expected_sha256:
            raise PanelCacheError(
                f"download_cache: SHA-256 mismatch on {name} tarball; "
                f"expected {expected_sha256}, got {actual_sha256}. "
                f"The published .sha256 file disagrees with the "
                f"downloaded bytes — re-download or contact the publisher.",
            )

        # ── Extract + validate + atomic-install ──
        # UUID suffix prevents collision with stale .extract dirs
        # left over from a prior aborted run.
        extract_dir = cache_root / f".{name}.extract-{uuid.uuid4().hex[:8]}"
        extract_dir.mkdir(parents=True)
        _safe_extract_tarball(tmp_path, extract_dir)
        cache_internal_root = _find_manifest_root(extract_dir)
        # Validate by loading the manifest — catches a corrupted
        # tarball that happened to have a matching SHA (impossible
        # in practice but cheap to confirm).
        try:
            load_cache_manifest(cache_internal_root)
        except PanelCacheError as exc:
            raise PanelCacheError(
                f"download_cache: extracted cache failed manifest "
                f"validation: {exc}",
            ) from exc

        # Atomic install. If target_dir exists (force=True path),
        # rename it aside first, then move the new content in.
        backup_dir: Path | None = None
        if target_dir.exists():
            backup_dir = cache_root / f".{name}.old-{uuid.uuid4().hex[:8]}"
            target_dir.rename(backup_dir)
        try:
            if cache_internal_root == extract_dir:
                extract_dir.rename(target_dir)
                extract_dir = None  # ownership transferred
            else:
                cache_internal_root.rename(target_dir)
        except OSError:
            # Restore the backup if the rename failed. If the restore
            # ALSO fails the user's previous cache is left at the
            # `.{name}.old-*` path — log loudly so they know where it
            # went rather than only suppressing the OSError.
            if backup_dir is not None:
                try:
                    backup_dir.rename(target_dir)
                except OSError as restore_exc:
                    logger.error(
                        "download_cache: install failed AND backup "
                        "restore failed; your previous cache is at %s "
                        "and the failed extract is at %s. Manual "
                        "recovery: rename %s back to %s. Restore error: %s",
                        backup_dir, extract_dir, backup_dir, target_dir,
                        restore_exc,
                    )
            raise
        # New cache installed; clean up the backup if there was one.
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    except BaseException:
        # BaseException covers KeyboardInterrupt + SystemExit too —
        # operators hitting Ctrl-C mid-extract shouldn't leave
        # `.{name}.extract-*` orphans cluttering cache_root.
        if extract_dir is not None and extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)
        # Final belt-and-suspenders: even if extract_dir was renamed
        # into place above (rare edge — the rename branch sets it to
        # None), tolerate the path being absent.
        if extract_dir is not None and extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)

    logger.info(
        "download_cache: %s v%s installed at %s",
        name, release.version_number, target_dir,
    )
    return target_dir


__all__ = [
    "CacheRelease",
    "DEFAULT_GITHUB_REPO",
    "download_cache",
    "list_available_caches",
]
