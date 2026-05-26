"""Error types raised by admixture-cache.

Kept minimal: a single ``PanelCacheError`` class so consumers can
catch one exception type without depending on ancestry-pipeline's
larger error hierarchy.
"""

from __future__ import annotations


class PanelCacheError(Exception):
    """Raised when a cache operation fails for a foreseeable reason.

    Covers:
    - Cache directory missing / unloadable manifest
    - SHA mismatch between cached and current config
    - Multimodality failure at cache-build time
    - File-format issues parsing AADR .anno or PLINK .fam
    - ADMIXTURE subprocess failure during cache build
    """


# Backward-compat alias for the source-of-extraction error name.
# ancestry-pipeline catches PopAutomationConfigError today; once it
# migrates to admixture-cache, references will be updated. Keeping
# the alias makes the first wire-up phase a pure import-rewrite with
# no behavioral change.
PopAutomationConfigError = PanelCacheError

__all__ = ["PanelCacheError", "PopAutomationConfigError"]
