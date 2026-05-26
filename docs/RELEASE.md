# Release procedure

This document captures the one-time setup steps to publish
`admixture-cache` to PyPI, plus the per-release runbook.

## One-time PyPI Trusted Publishing setup

The release workflow (`.github/workflows/release.yml`) authenticates via
OIDC — no long-lived API token in repo secrets. Before the first tag
push, register the project at PyPI:

1. Sign in to [https://pypi.org/](https://pypi.org/) as the project owner.
2. Go to [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/).
3. Click "Add a new pending publisher" and fill in:

   | Field | Value |
   |---|---|
   | PyPI project name | `admixture-cache` |
   | Owner | `carstenerickson` |
   | Repository | `admixture-cache` |
   | Workflow filename | `release.yml` |
   | Environment name | `pypi` |

4. Save. The pending publisher will activate on the first successful
   publish.

GitHub side — in the repo Settings → Environments, create a `pypi`
environment (no protection rules needed for now; the OIDC-only flow
means a leaked PAT can't accidentally publish from a branch).

## Per-release runbook

```bash
# 1. Make sure the working tree is clean and CI is green on main.
git status
gh run list --branch main --workflow CI -L 3

# 2. Bump the version in TWO places.
#    pyproject.toml: version = "X.Y.Z"
#    src/admixture_cache/__init__.py: __version__ = "X.Y.Z"
$EDITOR pyproject.toml src/admixture_cache/__init__.py

# 3. Close [Unreleased] → [X.Y.Z] - YYYY-MM-DD in CHANGELOG.md.
#    Update the bottom-of-file diff links.
$EDITOR CHANGELOG.md

# 4. Local sanity gates.
ruff check src/ tests/
mypy src/
pytest

# 5. Commit + push.
git add pyproject.toml src/admixture_cache/__init__.py CHANGELOG.md
git commit -m "chore(release): vX.Y.Z — version bump + CHANGELOG finalize"
git push origin main

# 6. Wait for CI to go green on the release commit.
gh run watch --workflow CI

# 7. Tag + push the tag.
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z

# 8. Watch the release workflow.
gh run watch --workflow Release

# 9. Verify on PyPI in a fresh venv.
python3.11 -m venv /tmp/verify-vXYZ
source /tmp/verify-vXYZ/bin/activate
pip install admixture-cache==X.Y.Z
admixture-cache --version       # should print X.Y.Z
python -c "import admixture_cache; print(admixture_cache.__version__)"
deactivate && rm -rf /tmp/verify-vXYZ
```

## Dry-running the release path

The release workflow has `workflow_dispatch:` enabled, so you can run
the full build + smoke-test path on any branch without publishing:

```bash
gh workflow run release.yml --ref my-branch
```

The `publish` job only fires for tag pushes, so a manual dispatch is a
safe rehearsal.

## What v1.0 means

The 0.x line predates publication and tracked the in-monorepo source
tool's API. v1.0 freezes the public API surface and the cache directory
layout. After v1.0:

- Additive changes (new optional kwargs, new CLI flags, new exports)
  bump the minor.
- Bug fixes and internal refactors bump the patch.
- Anything that breaks the public API or the cache layout bumps the
  major and ships a migration note in the changelog.

Canonical published-cache release artifacts (regional / continental /
ancestral-cluster tarballs) are versioned independently and ship as
separate GitHub releases (one per cache) — they share the library's
semver only insofar as the manifest schema is forward-compatible.
