"""Protocol type for external-binary runners (plink2, admixture).

admixture-cache invokes plink2 (alignment, format conversion) and
ADMIXTURE (cache build) as subprocesses. The library doesn't depend
on a specific orchestrator's tool-running framework — callers pass
any object satisfying the ``ToolRunner`` Protocol below.

This keeps admixture-cache decoupled from any host framework's
tool-runner abstraction while keeping subprocess invocation ergonomic.
A reference subprocess-based implementation ships in
:mod:`admixture_cache.cli`.

Optional capabilities
---------------------

The ``log_name`` and ``pid_callback`` parameters are optional
extensions. The library detects via :mod:`inspect` whether a given
runner's ``run`` accepts them and only passes them when supported, so
older runners that predate these additions continue working with no
code change. Implementations are encouraged to honor both for
diagnostics and clean cancellation in parallel-restart builds.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class ToolRunner(Protocol):
    """Minimal runner interface admixture-cache calls into.

    Any orchestrator's tool wrapper that exposes a ``run`` method with
    this shape can be passed straight through — no adapter required.
    """

    def run(
        self,
        *,
        args: list[str],
        cwd: Path,
        log_dir: Path,
        timeout_seconds: int = ...,
        log_name: str | None = ...,
        pid_callback: Callable[[int], None] | None = ...,
    ) -> object:
        """Execute the underlying binary with ``args``, capturing
        stdout/stderr under ``log_dir``. Block until completion or
        ``timeout_seconds`` elapsed.

        When ``log_name`` is given, capture output to
        ``log_dir/<log_name>`` exactly (no timestamp tag). When
        ``pid_callback`` is given, invoke it with the spawned
        subprocess's PID immediately after ``fork``/``Popen`` so the
        caller can SIGTERM the process group on later cancellation.

        Implementations may raise a runner-specific exception on
        nonzero exit / timeout; admixture-cache catches that
        exception and wraps it as :class:`PanelCacheError`.

        Both ``log_name`` and ``pid_callback`` are optional extensions
        added after the initial Protocol shipped; runners that don't
        accept them are detected and called without those kwargs.
        """
        ...


__all__ = ["ToolRunner"]
