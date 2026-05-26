# Contributing to admixture-cache

The project is small and opinionated. Substantive scope changes (alternative solvers, multi-target batch projection, dosage formats beyond plink2 `--recode A`) should start with a design discussion on a [GitHub issue](https://github.com/carstenerickson/admixture-cache/issues) before implementation.

For bug reports, small fixes, additive improvements, or test coverage, a PR is the right next step.

## Dev setup

```bash
git clone https://github.com/carstenerickson/admixture-cache.git
cd admixture-cache
python3.11 -m venv .venv          # or 3.12 / 3.13 / 3.14
source .venv/bin/activate
pip install -e '.[dev]'
```

The `[dev]` extra pulls in `pytest`, `pytest-cov`, `ruff`, `mypy`, `build`, and `twine`.

End-to-end projection paths require **ADMIXTURE** (for `build`) and **plink2** (for `project` / `verify`) on `PATH`. Unit tests use mocked runners + synthetic fixtures, so you don't need either tool installed just to run the suite locally.

## Running tests

```bash
pytest                            # the full suite, ~3 s
pytest tests/unit/test_projection.py -v
pytest -k "projection or manifest"
```

The test layout under `tests/unit/`:

| File | What it covers |
|---|---|
| `test_projection.py` | NumPy SLSQP recovery on synthetic K=2/3/4/5/8 panels, missing-dosage handling, simplex constraint, shape checks. |
| `test_manifest.py` | `PanelCacheManifest` schema (required fields, track/continent validator), JSON round-trip, `verify_cache_matches_current_config` match/mismatch reasons. |
| `test_io.py` | `sha256_file` streaming hash matches stdlib `hashlib`, `load_cached_p` shape validation, `_derive_cluster_order_from_pop_file` first-appearance ordering. |
| `test_alignment.py` | `align_target_to_panel_bim` + `extract_target_dosage_via_plink2` arg construction + post-run file checks via a recording mock runner. |
| `test_builder.py` | `build_panel_cache` idempotency on matching SHA, rebuild on stale SHA, multimodality failure, best-LL selection, ADMIXTURE log parsing, `ld_prune_panel` arg construction. |
| `test_cli.py` | Parser shape, all four subcommands registered, `verify` end-to-end against a synthetic on-disk cache, `SubprocessToolRunner` real-subprocess + timeout + missing-binary paths, installed console script smoke check. |

## Code quality

All three gates must pass locally before pushing; CI enforces them on every PR:

```bash
ruff check src/ tests/
mypy src/
pytest
```

- **ruff** — linting (E/F/W/I/B/UP/SIM/RUF rule sets per `pyproject.toml`). The scientific Unicode characters `×` and `–` are intentional in docstrings and comments (clearer than ASCII fallbacks in a math-heavy library), so the corresponding `RUF001/002/003` checks are disabled. `__all__` is grouped semantically rather than sorted, so `RUF022` is disabled too. Everything else is on.
- **mypy** — strict mode over `src/admixture_cache` only. `scipy.optimize` and `pandas` are marked `ignore_missing_imports` (no first-party stubs for the bits we use). The strict gate is intentionally narrow to keep test-writing friction low; new tests should still type-annotate fixtures, but they aren't gated by CI.
- **pytest** — 114 tests pass in ~3 s on a developer laptop; CI runs the same set on the 8-cell Python 3.11/3.12/3.13/3.14 × ubuntu/macos matrix.

## Commit + PR conventions

- **One logical change per commit** when feasible. The pre-publication structural pass that split `_core.py` into six modules landed as a sequence of small commits, not one mega-commit, so each step could be reverted independently.
- **Commit subject** uses imperative present tense: "add CLI download stub", "switch build_timestamp to datetime", "split _core.py into focused modules".
- **Body explains the why**, not just the what — what bug it fixes, why this approach, what alternatives were ruled out. Future maintainers (often us in six months) read commit bodies, not blame lines.
- **PR title = top commit subject**; PR body summarizes the change set, links any related issue, and lists the verification path (tests added, gates green, projection numerical-parity check verified if math changed).

## Release process

`admixture-cache` ships to PyPI via OIDC trusted publishing on tag push:

1. Land all changes on `main` via PR. Ensure CI is green.
2. Bump the version in `pyproject.toml` and `src/admixture_cache/__init__.py` (`__version__ = "X.Y.Z"`).
3. Update `CHANGELOG.md`: close `[Unreleased]` → open `[X.Y.Z] - YYYY-MM-DD` with the change set. Update the bottom-of-file diff links.
4. Commit + push: `chore(release): vX.Y.Z — version bump + CHANGELOG finalize`.
5. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.

The tag push fires `.github/workflows/release.yml`, which:

- Builds sdist + wheel via `python -m build`.
- Validates metadata with `twine check --strict`.
- Smoke-tests the built wheel across the full 8-cell matrix (Python 3.11/3.12/3.13/3.14 × ubuntu/macos): installs the wheel into a clean venv, runs `admixture-cache --help`, runs `pytest tests/unit/`.
- Publishes to PyPI via OIDC trusted publishing — no API token in repo secrets.

Manual workflow dispatch (without a tag) runs the build + smoke-test path without publishing — useful for dry-running the release before tagging.

### One-time PyPI Trusted Publishing setup

Before the first tag push, the project must be configured at [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/) with:

| Field | Value |
|---|---|
| PyPI project name | `admixture-cache` |
| Owner | `carstenerickson` |
| Repository | `admixture-cache` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

Once registered, no token lives in GitHub secrets; PyPI verifies the GitHub Actions OIDC claim from the matching environment on each publish.

## Design philosophy

Two principles, in tension order:

1. **Numerical correctness first.** The projection math matching stock `admixture --supervised` Q to ~1e-5 absolute is the regression bar. Every feature commit must keep that property; ergonomics and performance matter only after that.
2. **Bounded scope.** The library does precomputed-P projection. Not retraining-on-the-fly, not joint inference of P and Q from one target, not k-means-style initialization heuristics. Scope creep here adds maintenance debt that doesn't differentiate the project from an end-to-end ADMIXTURE rerun.

When the right call is ambiguous, file an issue first — admixture-cache errs toward shipping a smaller surface than the user's first ask, because the project's value is in being a well-tested single step rather than several plausible ones.
