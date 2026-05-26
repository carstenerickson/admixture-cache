"""Reference :class:`ToolRunner` implementation built on stdlib subprocess.

Lives in its own module (not ``cli.py``) so the public
``admixture_cache.SubprocessToolRunner`` re-export can be wired
top-down through ``__init__.py`` without the partial-init circular
import that ``cli.py``-as-source would require (``cli.py`` itself
imports many things from ``admixture_cache``).

The class is the canonical reference for the
:class:`admixture_cache.runner.ToolRunner` Protocol: spawns the binary
via :class:`subprocess.Popen` with ``start_new_session=True`` so each
child gets its own process group, honors all v1.0+v1.1 optional
Protocol kwargs (``log_name``, ``pid_callback``, ``argv_prefix``), and
bounds every post-kill wait so a D-state child can't wedge the
runner.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Callable
from pathlib import Path

from admixture_cache.errors import PanelCacheError


class SubprocessToolRunner:
    """Default :class:`admixture_cache.runner.ToolRunner` implementation.

    Spawns the binary via :class:`subprocess.Popen` with the given args
    + cwd, captures stdout and stderr to a single log file under
    ``log_dir``, and raises :class:`PanelCacheError` on non-zero exit
    or timeout.

    Construct with the absolute path (or name on PATH) of the binary to
    invoke; the same runner instance can be reused across calls.

    Honors all optional ToolRunner Protocol extensions:

    - ``log_name`` (v1.0): explicit per-call log filename. When None,
      a per-call name is derived from the args via
      :meth:`_derive_log_name`.
    - ``pid_callback`` (v1.0): invoked with the spawned PID
      immediately after ``Popen`` so the caller can SIGTERM the
      process group on cancellation.
    - ``argv_prefix`` (v1.1): list of tokens prepended to the spawned
      argv before ``self.binary``. Used for ``numactl --membind=N --``
      wrapping; also works for ``taskset``, ``nice``, ``time``, etc.
    """

    def __init__(self, binary: str) -> None:
        self.binary = binary

    def run(
        self,
        *,
        args: list[str],
        cwd: Path,
        log_dir: Path,
        timeout_seconds: int = 600,
        log_name: str | None = None,
        pid_callback: Callable[[int], None] | None = None,
        argv_prefix: list[str] | None = None,
    ) -> object:
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_name is not None:
            log_path = log_dir / log_name
        else:
            log_path = log_dir / self._derive_log_name(args, self.binary)
        # If a prior attempt left a log at this path, rotate it to
        # `.prev` rather than clobbering. NOTE: only the *immediately
        # previous* attempt is preserved — repeated retries overwrite
        # `.prev` each time. For multi-attempt diagnostics, archive
        # `.prev` between retries externally.
        prev_path = log_path.with_suffix(log_path.suffix + ".prev")
        rotated = False
        if log_path.exists():
            log_path.replace(prev_path)
            rotated = True
        # argv_prefix wraps the binary call — e.g. ["numactl",
        # "--membind=0", "--"] pins the spawned process's memory
        # allocations to NUMA node 0. The prefix elements go BEFORE
        # self.binary in the spawned argv.
        cmd = [*(argv_prefix or []), self.binary, *args]
        try:
            log_file = log_path.open("w")
        except OSError as exc:
            # Restore the rotated file so operators don't lose the
            # last successful run's log to an unrelated open failure.
            if rotated:
                with contextlib.suppress(OSError):
                    prev_path.replace(log_path)
            raise PanelCacheError(
                f"SubprocessToolRunner: cannot open log {log_path}: {exc}",
            ) from exc

        # NB: `start_new_session=True` puts the child in its own
        # process group, so `_cancel_inflight` can signal the group
        # via the same id without racing PID recycle in the parent's
        # pgroup. Runners with their own subprocess management should
        # adopt the same pattern.
        proc: subprocess.Popen[bytes] | None = None
        returncode: int | None = None
        try:
            try:
                proc = subprocess.Popen(
                    cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                # Don't leave an empty log alongside a "binary missing"
                # error — clean up both the (empty) new log AND any
                # rotated .prev that we'd otherwise orphan from the
                # operator's diagnostic surface. Restore prior log on
                # rotation so the actual last-good attempt survives.
                log_file.close()
                log_path.unlink(missing_ok=True)
                if rotated:
                    with contextlib.suppress(OSError):
                        prev_path.replace(log_path)
                # The missing binary is cmd[0], which is either
                # self.binary (no argv_prefix) or argv_prefix[0]
                # (e.g. "numactl" when NUMA pinning is on). The OSError
                # filename attribute on Py3.10+ tells us which binary
                # exec() couldn't find; fall back to cmd[0] if not set.
                missing_bin = getattr(exc, "filename", None) or cmd[0]
                raise PanelCacheError(
                    f"SubprocessToolRunner: binary {missing_bin!r} not "
                    f"found on PATH or at the given absolute path "
                    f"(while invoking {' '.join(cmd[:3])}...)",
                ) from exc

            if pid_callback is not None:
                try:
                    pid_callback(proc.pid)
                except Exception:
                    # Callback failure must not orphan the subprocess.
                    # Bound the reap (D-state, NFS hang); suppress the
                    # secondary timeout so the operator sees the
                    # ORIGINAL callback exception, not a TimeoutExpired
                    # that obscures it.
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=30)
                    raise
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                # Bound the post-kill wait so a child stuck in
                # uninterruptible-sleep (D-state) doesn't block
                # cleanup forever. After 30 s the OS owns it.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=30)
                raise PanelCacheError(
                    f"SubprocessToolRunner: {self.binary} timed out "
                    f"after {timeout_seconds}s; log at {log_path}",
                ) from exc
        finally:
            # Reap the child if we leave the try-block on any path
            # (including a callback-raised exception). The kill+wait
            # above already attempted reaping for the common cases;
            # this finally is a defense-in-depth against unhandled
            # exception paths between Popen and the explicit kills.
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=30)
            log_file.close()
        if returncode is None or returncode != 0:
            raise PanelCacheError(
                f"SubprocessToolRunner: {self.binary} exited "
                f"{returncode}; log at {log_path}",
            )
        return None

    @staticmethod
    def _derive_log_name(args: list[str], binary: str) -> str:
        """Build a per-call log filename from the args.

        plink2 callers pass ``--out <prefix>``; ADMIXTURE callers pass
        ``-s<seed>``. Iterate ``enumerate(args)`` so duplicate flags
        produce distinct tags (the previous ``args.index(a)`` always
        returned the first match).
        """
        tag_parts: list[str] = []
        for i, a in enumerate(args):
            if a.startswith("-s") and a[2:].isdigit():
                tag_parts.append(f"seed{a[2:]}")
            elif a == "--out" and i + 1 < len(args):
                tag_parts.append(Path(args[i + 1]).name)
        tag = "_".join(tag_parts) or "run"
        return f"{Path(binary).name}_{tag}.out"


__all__ = ["SubprocessToolRunner"]
