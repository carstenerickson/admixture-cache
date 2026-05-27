"""CLI smoke tests: parser shape, verify subcommand against synthetic
cache, download stub, SubprocessToolRunner error paths."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from admixture_cache import (
    PanelCacheError,
    PanelCacheManifest,
    sha256_file,
)
from admixture_cache.cli import (
    SubprocessToolRunner,
    _build_parser,
    _parse_geo_filter_yamls,
    _parse_max_parallel_restarts,
    main,
)


class TestParser:
    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit) as e:
            parser.parse_args(["--version"])
        assert e.value.code == 0
        from admixture_cache import __version__
        out = capsys.readouterr().out
        assert __version__ in out

    def test_no_subcommand_errors(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_all_four_subcommands_registered(self) -> None:
        parser = _build_parser()
        # Each subcommand should be parseable with its required args
        for argv in [
            ["build", "--panel-bed", "x", "--panel-pop", "y",
             "--clusters-yaml", "z", "--k", "4", "--cache-dir", "c",
             "--track", "regional", "--panel-id", "p", "--panel-version", "v"],
            ["project", "--target-bed", "t", "--cache-dir", "c", "--work-dir", "w"],
            ["verify", "--panel-bed", "p", "--clusters-yaml", "y",
             "--k", "4", "--cache-dir", "c"],
            ["download", "name"],
        ]:
            ns = parser.parse_args(argv)
            assert ns.command == argv[0]


class TestParseGeoFilterYamls:
    def test_single_entry(self, tmp_path: Path) -> None:
        p = tmp_path / "f.yaml"
        p.write_text("data\n")
        out = _parse_geo_filter_yamls([f"region:{p}"])
        assert out == {"region": sha256_file(p)}

    def test_multiple_entries(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.yaml"
        p1.write_text("a\n")
        p2 = tmp_path / "b.yaml"
        p2.write_text("b\n")
        out = _parse_geo_filter_yamls([f"alpha:{p1}", f"beta:{p2}"])
        assert set(out.keys()) == {"alpha", "beta"}
        assert out["alpha"] == sha256_file(p1)
        assert out["beta"] == sha256_file(p2)

    def test_missing_colon_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            _parse_geo_filter_yamls([str(tmp_path / "a.yaml")])

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            _parse_geo_filter_yamls([f"x:{tmp_path}/does_not_exist.yaml"])

    def test_empty_name_rejected(self, tmp_path: Path) -> None:
        """`:/path/to/file.yaml` (leading colon, no name) is a typo —
        reject it rather than silently storing under an empty key."""
        p = tmp_path / "valid.yaml"
        p.write_text("data\n")
        with pytest.raises(SystemExit, match="empty name"):
            _parse_geo_filter_yamls([f":{p}"])

    def test_empty_path_rejected(self, tmp_path: Path) -> None:
        """`name:` with no path is also a typo."""
        with pytest.raises(SystemExit, match="empty path"):
            _parse_geo_filter_yamls(["name:"])


class TestVerifyCommand:
    def _write_manifest_and_inputs(
        self, tmp_path: Path, *, k: int = 4,
    ) -> tuple[Path, Path, Path]:
        """Build a synthetic cache_dir + panel.bed + clusters.yaml such
        that the SHAs in the manifest match the current files."""
        panel_bed = tmp_path / "panel.bed"
        panel_bed.write_bytes(b"\x6c\x1b\x01")
        panel_bim = tmp_path / "panel.bim"
        panel_bim.write_text(
            "\n".join(
                f"1\trs{i}\t0\t{i+1000}\tA\tG" for i in range(10)
            ) + "\n",
        )
        clusters_yaml = tmp_path / "clusters.yaml"
        clusters_yaml.write_text("k: 4\n")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        manifest = PanelCacheManifest(
            track="regional", panel_id="p1", panel_version="v1",
            panel_bim_sha256=sha256_file(panel_bim),
            clusters_yaml_sha256=sha256_file(clusters_yaml),
            k=k, admixture_version="1.4.0", seeds_used=[1],
            best_seed=1, best_loglikelihood=-1.0, restart_sd_max=0.01,
            cluster_order=[f"c{i}" for i in range(k)],
            build_wallclock_seconds=1.0,
            build_timestamp=datetime.now(UTC),
        )
        (cache_dir / "manifest.json").write_text(manifest.model_dump_json())
        return panel_bed, clusters_yaml, cache_dir

    def test_verify_match_returns_0(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        panel_bed, clusters_yaml, cache_dir = self._write_manifest_and_inputs(
            tmp_path,
        )
        rc = main([
            "verify",
            "--panel-bed", str(panel_bed),
            "--clusters-yaml", str(clusters_yaml),
            "--k", "4",
            "--cache-dir", str(cache_dir),
        ])
        assert rc == 0
        assert "match" in capsys.readouterr().out

    def test_verify_mismatch_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        panel_bed, clusters_yaml, cache_dir = self._write_manifest_and_inputs(
            tmp_path,
        )
        rc = main([
            "verify",
            "--panel-bed", str(panel_bed),
            "--clusters-yaml", str(clusters_yaml),
            "--k", "5",  # K mismatch
            "--cache-dir", str(cache_dir),
        ])
        assert rc == 1
        assert "MISMATCH" in capsys.readouterr().err

    def test_verify_missing_panel_bim_returns_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([
            "verify",
            "--panel-bed", str(tmp_path / "missing.bed"),
            "--clusters-yaml", str(tmp_path / "missing.yaml"),
            "--k", "4",
            "--cache-dir", str(tmp_path / "cache"),
        ])
        assert rc == 2
        assert "panel .bim missing" in capsys.readouterr().err


class TestDownloadCommand:
    """End-to-end tests of the `download` subcommand mocking the
    underlying `download_cache` / `list_available_caches` so we
    don't hit the real GitHub API."""

    def test_download_no_name_no_list_returns_2(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["download"])
        assert rc == 2
        assert "name required" in capsys.readouterr().err

    def test_download_invokes_library_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`download <name>` calls `download_cache` and prints the
        installed path on success."""
        captured_kwargs: dict[str, object] = {}

        def fake_download_cache(name: str, **kwargs: object) -> Path:
            captured_kwargs["name"] = name
            captured_kwargs.update(kwargs)
            installed = tmp_path / "fake_root" / name
            installed.mkdir(parents=True)
            return installed

        monkeypatch.setattr(
            "admixture_cache.distribution.download_cache",
            fake_download_cache,
        )
        rc = main([
            "download", "regional_k21_aadr_v66_ho",
            "--cache-root", str(tmp_path / "fake_root"),
            "--quiet",
        ])
        assert rc == 0
        assert captured_kwargs["name"] == "regional_k21_aadr_v66_ho"
        assert captured_kwargs["force"] is False
        out = capsys.readouterr().out
        assert "Installed regional_k21_aadr_v66_ho" in out

    def test_download_list_prints_available_caches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`download --list` enumerates releases newest-first."""
        from datetime import UTC, datetime

        from admixture_cache.distribution import CacheRelease

        fake_releases = [
            CacheRelease(
                name="regional_k21_aadr_v66_ho",
                version="v2", tag="cache-regional_k21_aadr_v66_ho-v2",
                tarball_url="https://example.com/regional_k21_aadr_v66_ho.tar.gz",
                sha256_url="https://example.com/regional_k21_aadr_v66_ho.tar.gz.sha256",
                size_bytes=120_000_000,
                published_at=datetime(2026, 5, 26, tzinfo=UTC),
                html_url="https://example.com/release/v2",
                notes="",
            ),
            CacheRelease(
                name="regional_k21_aadr_v66_ho",
                version="v1", tag="cache-regional_k21_aadr_v66_ho-v1",
                tarball_url="https://example.com/old.tar.gz",
                sha256_url="https://example.com/old.tar.gz.sha256",
                size_bytes=100_000_000,
                published_at=datetime(2026, 4, 1, tzinfo=UTC),
                html_url="https://example.com/release/v1",
                notes="",
            ),
        ]
        monkeypatch.setattr(
            "admixture_cache.distribution.list_available_caches",
            lambda **_kw: fake_releases,
        )
        rc = main(["download", "--list"])
        assert rc == 0
        out = capsys.readouterr().out
        # Latest version listed; older versions noted parenthetically.
        assert "regional_k21_aadr_v66_ho  v2" in out
        assert "(also: v1)" in out
        assert "https://example.com/release/v2" in out

    def test_download_no_caches_published(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "admixture_cache.distribution.list_available_caches",
            lambda **_kw: [],
        )
        rc = main(["download", "--list"])
        assert rc == 0
        assert "No published caches" in capsys.readouterr().err

    def test_download_library_error_returns_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A PanelCacheError from download_cache surfaces as exit 1."""
        def boom(name: str, **_kw: object) -> Path:
            raise PanelCacheError("simulated network failure")

        monkeypatch.setattr(
            "admixture_cache.distribution.download_cache",
            boom,
        )
        rc = main(["download", "regional_k21", "--quiet"])
        assert rc == 1
        assert "simulated network failure" in capsys.readouterr().err


class TestSubprocessToolRunner:
    def test_runs_a_real_subprocess(self, tmp_path: Path) -> None:
        """Use /bin/echo as a stand-in for the actual binaries."""
        runner = SubprocessToolRunner("/bin/echo")
        runner.run(
            args=["hello"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
        )
        logs = list((tmp_path / "logs").glob("echo_*.out"))
        assert len(logs) == 1
        assert "hello" in logs[0].read_text()

    def test_nonzero_exit_raises_panel_cache_error(self, tmp_path: Path) -> None:
        runner = SubprocessToolRunner("/usr/bin/false")
        with pytest.raises(PanelCacheError, match="exited"):
            runner.run(
                args=[],
                cwd=tmp_path,
                log_dir=tmp_path / "logs",
            )

    def test_timeout_raises_panel_cache_error(self, tmp_path: Path) -> None:
        runner = SubprocessToolRunner("/bin/sleep")
        with pytest.raises(PanelCacheError, match="timed out"):
            runner.run(
                args=["5"],
                cwd=tmp_path,
                log_dir=tmp_path / "logs",
                timeout_seconds=1,
            )

    def test_missing_binary_raises_panel_cache_error(self, tmp_path: Path) -> None:
        runner = SubprocessToolRunner("/no/such/binary/anywhere_unique_xyz")
        with pytest.raises(PanelCacheError, match="not found"):
            runner.run(
                args=[],
                cwd=tmp_path,
                log_dir=tmp_path / "logs",
            )

    def test_log_file_named_by_tag(self, tmp_path: Path) -> None:
        runner = SubprocessToolRunner("/bin/echo")
        runner.run(
            args=["-s42", "--out", str(tmp_path / "myout")],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
        )
        # Tag should contain seed42 and myout
        logs = list((tmp_path / "logs").iterdir())
        assert any("seed42" in p.name and "myout" in p.name for p in logs)

    def test_explicit_log_name_honored(self, tmp_path: Path) -> None:
        """When log_name is given, output lands at that exact path
        with no auto-tagging applied."""
        runner = SubprocessToolRunner("/bin/echo")
        runner.run(
            args=["-s9", "hello"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            log_name="restart_9.out",
        )
        assert (tmp_path / "logs" / "restart_9.out").exists()
        # No auto-tagged file alongside
        assert list((tmp_path / "logs").iterdir()) == [
            tmp_path / "logs" / "restart_9.out",
        ]

    def test_pid_callback_invoked(self, tmp_path: Path) -> None:
        """pid_callback receives the spawned subprocess's PID."""
        seen: list[int] = []
        runner = SubprocessToolRunner("/bin/echo")
        runner.run(
            args=["hi"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            pid_callback=seen.append,
        )
        assert len(seen) == 1
        assert seen[0] > 0  # any real PID is positive

    def test_pid_callback_raise_does_not_orphan_subprocess(
        self, tmp_path: Path,
    ) -> None:
        """When pid_callback raises after Popen succeeds, the runner
        must kill + reap the child rather than leaving it running.
        Uses `sleep 60` so we can observe whether it was reaped."""
        import subprocess as _sp
        import time

        captured: dict[str, int] = {}

        def boom(pid: int) -> None:
            captured["pid"] = pid
            raise RuntimeError("user callback failed")

        runner = SubprocessToolRunner("/bin/sleep")
        t0 = time.time()
        with pytest.raises(RuntimeError, match="user callback failed"):
            runner.run(
                args=["60"],
                cwd=tmp_path,
                log_dir=tmp_path / "logs",
                pid_callback=boom,
            )
        elapsed = time.time() - t0
        # Should be effectively instant (well under the 60s sleep).
        assert elapsed < 5
        # And the spawned PID should no longer be alive.
        pid = captured["pid"]
        # poll for up to 1s for the child to be reaped
        for _ in range(20):
            try:
                _sp.run(
                    ["ps", "-p", str(pid)],
                    capture_output=True, check=True, timeout=1,
                )
                time.sleep(0.05)
            except _sp.CalledProcessError:
                break
        else:
            raise AssertionError(f"subprocess pid={pid} still alive after raise")

    def test_log_rotated_on_rerun(self, tmp_path: Path) -> None:
        """A second run with the same log_name moves the prior log to
        `.prev` rather than truncating — so the previous attempt's
        diagnostic survives one rerun."""
        runner = SubprocessToolRunner("/bin/echo")
        runner.run(
            args=["first"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            log_name="restart_1.out",
        )
        first_content = (tmp_path / "logs" / "restart_1.out").read_text()
        assert "first" in first_content

        runner.run(
            args=["second"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            log_name="restart_1.out",
        )
        # Live log has second; prior is preserved under .prev
        assert "second" in (tmp_path / "logs" / "restart_1.out").read_text()
        assert (tmp_path / "logs" / "restart_1.out.prev").read_text() == first_content

    def test_missing_binary_does_not_leave_empty_log(self, tmp_path: Path) -> None:
        """FileNotFoundError on Popen must clean up the just-created
        empty log so the operator's diagnostic surface isn't polluted
        with zero-byte files."""
        runner = SubprocessToolRunner("/nonexistent/binary/xyz")
        with pytest.raises(PanelCacheError, match="not found"):
            runner.run(
                args=["whatever"],
                cwd=tmp_path,
                log_dir=tmp_path / "logs",
                log_name="restart_1.out",
            )
        # Log directory may exist, but no stray empty files.
        log_files = list((tmp_path / "logs").iterdir())
        assert log_files == [], f"unexpected stray log files: {log_files}"

    def test_duplicate_out_flag_does_not_double_tag(self, tmp_path: Path) -> None:
        """Auto-tag derivation must use enumerate(args), not
        args.index, so duplicate --out flags produce distinct tag
        components (or at minimum don't crash)."""
        runner = SubprocessToolRunner("/bin/echo")
        runner.run(
            args=["--out", str(tmp_path / "first"),
                  "--out", str(tmp_path / "second")],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
        )
        logs = list((tmp_path / "logs").iterdir())
        # Tag should contain BOTH out-prefixes, not the first one twice.
        names = [p.name for p in logs]
        assert any("first" in n and "second" in n for n in names), names


class TestConsoleScriptInstallation:
    def test_admixture_cache_help_via_subprocess(self) -> None:
        """The installed console script renders --help."""
        out = subprocess.run(
            ["admixture-cache", "--help"], capture_output=True, text=True,
            timeout=10, check=True,
        )
        assert "admixture-cache" in out.stdout
        for cmd in ("build", "project", "verify", "download"):
            assert cmd in out.stdout


class TestParseMaxParallelRestarts:
    def test_auto_returns_none(self) -> None:
        assert _parse_max_parallel_restarts("auto") is None
        assert _parse_max_parallel_restarts("AUTO") is None
        assert _parse_max_parallel_restarts("") is None

    def test_positive_int_passthrough(self) -> None:
        assert _parse_max_parallel_restarts("1") == 1
        assert _parse_max_parallel_restarts("8") == 8

    def test_zero_rejected(self) -> None:
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_max_parallel_restarts("0")

    def test_negative_rejected(self) -> None:
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_max_parallel_restarts("-1")

    def test_non_numeric_rejected(self) -> None:
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_max_parallel_restarts("yes")

    def test_argparse_propagates_error(self) -> None:
        """When wired through argparse, ArgumentTypeError → SystemExit."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "build",
                "--panel-bed", "x", "--panel-pop", "y",
                "--clusters-yaml", "z", "--k", "4", "--cache-dir", "c",
                "--track", "regional",
                "--panel-id", "p", "--panel-version", "v",
                "--max-parallel-restarts", "bogus",
            ])

    def test_build_command_default_is_none(self) -> None:
        """The build subcommand's default for --max-parallel-restarts
        must be None (auto), not 1 — otherwise the auto-heuristic is
        unreachable."""
        parser = _build_parser()
        ns = parser.parse_args([
            "build",
            "--panel-bed", "x", "--panel-pop", "y",
            "--clusters-yaml", "z", "--k", "4", "--cache-dir", "c",
            "--track", "regional",
            "--panel-id", "p", "--panel-version", "v",
        ])
        assert ns.max_parallel_restarts is None


class TestBuildCommandTrackContinent:
    """Early CLI-side validation of --track / --continent before
    launching ADMIXTURE (which would otherwise discover the
    inconsistency after hours of work)."""

    def _common_args(self, tmp_path: Path) -> list[str]:
        # Construct args that point at non-existent files; the validator
        # will short-circuit before the build runs.
        return [
            "build",
            "--panel-bed", str(tmp_path / "panel.bed"),
            "--panel-pop", str(tmp_path / "panel.pop"),
            "--clusters-yaml", str(tmp_path / "c.yaml"),
            "--k", "4",
            "--cache-dir", str(tmp_path / "cache"),
            "--panel-id", "p",
            "--panel-version", "v",
            "--admixture-version", "1.4.0",
        ]

    def test_ancestral_cluster_without_continent_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(
            [*self._common_args(tmp_path), "--track", "ancestral_cluster"],
        )
        assert rc == 2
        assert "ancestral_cluster requires --continent" in capsys.readouterr().err

    def test_continent_without_ancestral_cluster_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(
            [*self._common_args(tmp_path),
             "--track", "regional", "--continent", "Europe"],
        )
        assert rc == 2
        assert "only valid with --track=ancestral_cluster" in capsys.readouterr().err


class TestPopAutomationConfigErrorReexport:
    def test_importable_from_package(self) -> None:
        """The CHANGELOG documents PopAutomationConfigError as a
        back-compat alias preserved for upstream consumers."""
        from admixture_cache import (
            PanelCacheError,
            PopAutomationConfigError,
        )
        assert PopAutomationConfigError is PanelCacheError
        # And listed in __all__ so `from admixture_cache import *` works.
        import admixture_cache
        assert "PopAutomationConfigError" in admixture_cache.__all__
