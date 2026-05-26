"""Runtime dispatcher for ``ToolRunner.run`` calls.

The library extends the ``ToolRunner`` Protocol over time (``log_name``
and ``pid_callback`` shipped in v1.0; future extensions follow the same
pattern). Each new optional kwarg is detected at call time via
``inspect.signature`` — see :func:`_runner_supports` — and conditionally
forwarded by :func:`_call_runner`.

This module is the single place every internal subprocess invocation
should route through. Direct ``runner.run(...)`` calls bypass the
extension-detection logic and silently lose the Protocol extensions on
older runners, which is fine for single-call paths (one-shot plink2
extracts, etc.) but wrong for the parallel-restart hot path where
``log_name`` and ``pid_callback`` are load-bearing.

Lives in its own module (not ``builder.py``) so that every layer-1+
module can route through it without importing ``builder``, which would
violate the dependency layering documented in ``DEVELOPMENT.md``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from admixture_cache.runner import ToolRunner


def _runner_supports(runner: ToolRunner, param: str) -> bool:
    """Return True if ``runner.run`` accepts the named keyword
    parameter — either explicitly or via a ``**kwargs`` forwarder.
    Falls back to ``False`` on any inspection failure so older runners
    that predate Protocol extensions degrade gracefully.
    """
    try:
        params = inspect.signature(runner.run).parameters
    except (TypeError, ValueError):
        return False
    if param in params:
        return True
    # A `**kwargs` forwarder (the idiomatic adapter pattern) accepts
    # any keyword, including our optional Protocol extensions.
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call_runner(
    runner: ToolRunner,
    *,
    args: list[str],
    cwd: Path,
    log_dir: Path,
    timeout_seconds: int,
    log_name: str | None = None,
    pid_callback: Callable[[int], None] | None = None,
    argv_prefix: list[str] | None = None,
) -> object:
    """Invoke ``runner.run`` with the optional ``log_name``,
    ``pid_callback``, and ``argv_prefix`` extensions when the runner
    supports them, plain invocation otherwise."""
    kwargs: dict[str, Any] = {
        "args": args,
        "cwd": cwd,
        "log_dir": log_dir,
        "timeout_seconds": timeout_seconds,
    }
    if log_name is not None and _runner_supports(runner, "log_name"):
        kwargs["log_name"] = log_name
    if pid_callback is not None and _runner_supports(runner, "pid_callback"):
        kwargs["pid_callback"] = pid_callback
    if argv_prefix is not None and _runner_supports(runner, "argv_prefix"):
        kwargs["argv_prefix"] = argv_prefix
    return runner.run(**kwargs)


__all__ = ["_call_runner", "_runner_supports"]
