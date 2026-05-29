"""Packaging-consistency guards.

These catch the class of mistake where a release bumps one version
declaration but not the others — e.g. PR #1 (v1.4.1) bumped
``pyproject.toml`` but left ``admixture_cache.__version__`` at
``1.4.0``, which would have shipped a wheel PyPI labels ``1.4.1``
while ``admixture-cache --version`` printed ``1.4.0``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import admixture_cache

# tests/unit/test_packaging.py → repo root is two parents up from the
# tests/ dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    data = tomllib.loads(_PYPROJECT.read_text())
    return data["project"]["version"]


def test_dunder_version_matches_pyproject() -> None:
    """``admixture_cache.__version__`` must equal the wheel version in
    pyproject. A mismatch means `pip install admixture-cache==X` gets a
    package whose `__version__` / `--version` reports Y != X."""
    assert admixture_cache.__version__ == _pyproject_version(), (
        f"version drift: admixture_cache.__version__="
        f"{admixture_cache.__version__!r} but pyproject [project].version="
        f"{_pyproject_version()!r}. Bump both in lockstep."
    )


def test_version_is_pep440_release() -> None:
    """Sanity: the version is a plain N.N.N release (no stray suffix).

    Loose check — three dot-separated leading numeric components. Catches
    a fat-fingered value like '1.4.0-dev' or 'v1.4.1' slipping into the
    declaration."""
    parts = admixture_cache.__version__.split(".")
    assert len(parts) >= 3, admixture_cache.__version__
    assert all(p.isdigit() for p in parts[:3]), admixture_cache.__version__
