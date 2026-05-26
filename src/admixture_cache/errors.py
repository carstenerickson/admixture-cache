"""Error types raised by admixture-cache.

Kept minimal: a single ``PanelCacheError`` class so consumers can
catch one exception type without inheriting a larger error hierarchy.
"""

from __future__ import annotations


class PanelCacheError(Exception):
    """Raised when a cache operation fails for a foreseeable reason.

    Covers:
    - Cache directory missing / unloadable manifest
    - SHA mismatch between cached and current config
    - Multimodality failure at cache-build time
    - File-format issues parsing PLINK .fam / .bim
    - ADMIXTURE subprocess failure during cache build
    """


# Backward-compat alias for an upstream consumer that catches the
# error under its original name during a migration window. Safe to
# delete once no caller relies on this name.
PopAutomationConfigError = PanelCacheError

__all__ = ["PanelCacheError", "PopAutomationConfigError"]
