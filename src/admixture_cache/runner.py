"""Protocol type for external-binary runners (plink2, admixture).

admixture-cache invokes plink2 (alignment, format conversion) and
ADMIXTURE (cache build) as subprocesses. The library doesn't depend
on a specific orchestrator's tool-running framework — callers pass
any object satisfying the ``ToolRunner`` Protocol below.

This decouples admixture-cache from ancestry-pipeline's ``ToolRegistry``
(or any other host framework) while keeping subprocess invocation
ergonomic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ToolRunner(Protocol):
    """Minimal runner interface admixture-cache calls into.

    Compatible with ancestry-pipeline's ``ToolRunner``,
    pgen-samplebind's wrappers, and any plain subprocess-helper class
    that matches this shape.
    """

    def run(
        self,
        *,
        args: list[str],
        cwd: Path,
        log_dir: Path,
        timeout_seconds: int = ...,
    ) -> object:
        """Execute the underlying binary with ``args``, capturing
        stdout/stderr under ``log_dir``. Block until completion or
        ``timeout_seconds`` elapsed.

        Implementations may raise a runner-specific exception on
        nonzero exit / timeout; admixture-cache catches that
        exception and wraps it as :class:`PanelCacheError`.
        """
        ...


__all__ = ["ToolRunner"]
