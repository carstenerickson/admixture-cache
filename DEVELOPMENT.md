# Development guide

The "why" behind the code. For setup, validation gates, and PR conventions, see `CONTRIBUTING.md`. For the release runbook, see `docs/RELEASE.md`. For user-facing usage, see `README.md`.

## Architectural shape in one paragraph

`admixture-cache` splits the supervised-ADMIXTURE workflow into two phases so the per-target hot path doesn't pay the panel-training cost. **Phase 1 — `build_panel_cache`** runs stock ADMIXTURE × N restarts via an injected `ToolRunner`, validates that the per-cluster restart standard deviation stays under a threshold (multimodality check), picks the best-LL P matrix, and writes a sealed cache directory whose `manifest.json` SHA-pins every input. **Phase 2 — `project_target`** loads the cached P, aligns target.bed to the cached panel.bim (variant set + REF/ALT axes via plink2), reads the target as a dosage vector, and solves for Q via `scipy.optimize.minimize(method="SLSQP")` against the binomial admixture likelihood. The math matches stock `admixture --supervised` Q to ~1e-5 absolute on real workloads; the wallclock split is ~hours-per-restart for phase 1 vs ~2 seconds end-to-end for phase 2.

## Module map

```
src/admixture_cache/
├── __init__.py        # public API re-exports (~15 symbols)
├── projection.py      # numpy_supervised_projection + ProjectionResult
├── builder.py         # build_panel_cache, _run_one_admixture_restart, ld_prune_panel
├── manifest.py        # PanelCacheManifest pydantic model + track/continent validator
├── alignment.py       # align_target_to_panel_bim, extract_target_dosage_via_plink2
├── io.py              # load_cached_p, load_cache_manifest, verify_..., sha256_file
├── orchestration.py   # project_target end-to-end wrapper
├── cli.py             # admixture-cache console script + SubprocessToolRunner
├── runner.py          # ToolRunner Protocol
├── errors.py          # PanelCacheError + PopAutomationConfigError alias
└── py.typed           # PEP 561 marker (consumers get our type info)
```

| Module | Lines | Imports allowed from |
|---|---|---|
| `projection.py` | ~100 | stdlib, numpy, scipy, errors |
| `manifest.py` | ~70 | stdlib, pydantic |
| `runner.py` | ~35 | stdlib only (`Protocol`, `Path`) |
| `errors.py` | ~15 | stdlib only |
| `io.py` | ~130 | stdlib, numpy, errors, manifest |
| `alignment.py` | ~120 | stdlib, numpy, pandas (inline), errors, runner |
| `builder.py` | ~880 | stdlib, numpy, errors, manifest, io, runner |
| `orchestration.py` | ~130 | stdlib, numpy, errors, alignment, io, projection, runner |
| `cli.py` | ~515 | everything (it's the integration point) |

The dependency graph is acyclic: `errors → manifest → io → alignment, projection → orchestration → cli`. `runner` is pulled in everywhere as a typing-only `Protocol`. `builder` sits next to `orchestration`; both depend on `io` + `manifest` but not on each other.

## Two-phase data flow

```
                  ┌─────────────────────────────────────────┐
                  │  Phase 1 (slow, one-time per cache)     │
                  │  build_panel_cache(panel, K, ...)       │
                  │   │                                     │
                  │   ├─ N × _run_one_admixture_restart     │
                  │   │   └─ admixture_runner.run([...])    │
                  │   ├─ multimodality validation           │
                  │   ├─ best-LL selection                  │
                  │   └─ atomic manifest write              │
                  │                                         │
                  │       ↓ writes ↓                        │
                  │                                         │
                  │  cache_dir/                             │
                  │  ├── panel.K.P                          │
                  │  ├── panel.K.Q                          │
                  │  ├── panel.bim                          │
                  │  ├── restart_sd.json                    │
                  │  ├── cluster_order.json                 │
                  │  ├── manifest.json   ← sealing manifest │
                  │  └── build_logs/                        │
                  └─────────────────────────────────────────┘
                              │
                              │ cache reused unchanged
                              ↓
                  ┌─────────────────────────────────────────┐
                  │  Phase 2 (fast, every target)           │
                  │  project_target(target, cache_dir, ...) │
                  │   │                                     │
                  │   ├─ load_cache_manifest                │
                  │   ├─ align_target_to_panel_bim          │
                  │   │   (plink2 --extract + --alt1-allele)│
                  │   ├─ extract_target_dosage_via_plink2   │
                  │   │   (plink2 --recode A + pandas)      │
                  │   ├─ load_cached_p                      │
                  │   └─ numpy_supervised_projection        │
                  │       (scipy SLSQP on binomial L)       │
                  │                                         │
                  │       ↓ returns ↓                       │
                  │                                         │
                  │  ProjectionResult                       │
                  │  ├── target_q          (shape (K,))     │
                  │  ├── cluster_order     (K names)        │
                  │  ├── panel_stability_max_sd             │
                  │  ├── n_snps_used                        │
                  │  ├── optimization_iterations            │
                  │  └── converged                          │
                  └─────────────────────────────────────────┘
```

## The cache contract (sealed by `manifest.json`)

`manifest.json` is the single source of truth for "is this cache valid?". The write is **atomic** (`tempfile + os.fsync + os.replace`) so a SIGKILL / power-loss / OS crash mid-write leaves either the prior complete manifest or no manifest at all — never a half-written JSON. The presence of `manifest.json` is the cache-valid signal; all other artifacts in `cache_dir/` (`panel.K.P`, `panel.K.Q`, `panel.bim`, `restart_sd.json`, `cluster_order.json`, `build_logs/`) are written BEFORE the manifest, so a manifest implies the rest are present.

A cache is "valid for the current config" iff `manifest.panel_bim_sha256` + `manifest.clusters_yaml_sha256` + `manifest.k` + `manifest.geo_filter_yaml_shas` all match the caller's current values. The check is symmetric: a cache with pinned geo-filter YAMLs reports mismatch against a caller who omits them, AND vice versa. See `io.verify_cache_matches_current_config` for the full check matrix.

The `PanelCacheManifest` schema (`manifest.py`) is forward-extensible via `schema_version`. Bumping the schema is a breaking change for existing on-disk caches and warrants a major version. The `track/continent` consistency validator (`model_validator(mode="after")`) enforces that `ancestral_cluster` builds carry a continent and the other two tracks don't.

## Key data structures

### `PanelCacheManifest` (`manifest.py`)

Pydantic model. Every field SHA-pins or version-pins an input or output. `geo_filter_yaml_shas` is a `dict[str, str]` because operators may pin multiple YAMLs (one per filter category). `build_timestamp` is a `datetime` (since v1.0.0; was `str` in v0.x — pydantic re-parses old ISO-8601 strings transparently). The `model_config = ConfigDict(extra="forbid")` is intentional: an unknown field in a manifest means the cache was written by a newer library, and silently ignoring would let a v1 consumer accept a v2 cache and pull subtly-wrong data.

### `ProjectionResult` (`projection.py`)

`@dataclass(frozen=True)`. The Q vector lives here alongside its provenance: which cluster each Q[k] corresponds to, the panel's restart-SD bound (so consumers can decide whether the build was tight enough for their use case), the number of non-missing SNPs the projection used, SLSQP iteration count, and a `converged` boolean. Immutable so it's safe to pass around without defensive copies.

### `ToolRunner` Protocol (`runner.py`)

Structural interface for the subprocess plumbing. The library doesn't ship a fixed runner — callers pass any object satisfying the Protocol. Why a Protocol over a fixed class:

- Most consumers already have their own subprocess wrapper (with timeout / logging / retry / metrics conventions). Forcing them to use ours would be a worse fit than letting them adapt theirs.
- A Protocol gives static type-checking ("does this object satisfy `ToolRunner`?") without forcing inheritance.
- Adding new optional Protocol parameters (`log_name`, `pid_callback`) is API-additive — runners that predate the extension still satisfy the Protocol, and the library detects support via `inspect.signature`.

The Protocol is documented in `README.md` under "ToolRunner Protocol". The reference implementation lives in `cli.SubprocessToolRunner`.

## The runner-extension dispatch story

The library extends the `ToolRunner` Protocol over time. Each new optional kwarg (`log_name` in v1.0.0, `pid_callback` in v1.0.0, potentially more later) is detected at call time via `inspect.signature` (`builder._runner_supports`). If the runner accepts the kwarg directly OR has a `VAR_KEYWORD` (`**kwargs`) parameter, the dispatcher forwards the kwarg; otherwise it's silently dropped.

Two gotchas worth knowing about this design:

1. **`**kwargs` forwarders are recognized as supporting any kwarg**, but the library can only inspect the signature — it can't verify the runner's body actually FORWARDS what it receives. An adapter that explicitly enumerates downstream args (e.g., `def run(self, **kwargs): return self._inner.run(args=kwargs["args"], cwd=kwargs["cwd"], ...)`) will pass the introspection check but silently strip `log_name`/`pid_callback`. The parallel-mode guard in `build_panel_cache` would then admit such a runner into parallel mode, produce incoherent logs, and break cancellation. **README documents this caveat loudly**; the contract is "if you declare `**kwargs`, you promise to forward what you receive."

2. **Parallel mode REQUIRES both `log_name` and `pid_callback`** (not just one). Without `log_name`, concurrent restarts can't disambiguate their log files; without `pid_callback`, the failure path can't SIGTERM in-flight subprocesses and `ThreadPoolExecutor.shutdown` would block up to `per_restart_timeout_seconds × (N-1)` (default ≈ 24 h per worker) for naturally-completing peers. The guard surfaces missing support at build start instead of via a multi-hour hang.

For runners that lack both kwargs, sequential mode (`max_parallel_restarts=1`) still works. `_run_one_admixture_restart` has a sequential-only log-discovery fallback (`allow_log_scan_fallback`) that snapshots `log_dir` before each call and identifies the new/modified-during-call file (mtime-aware, excludes `.prev` rotation artifacts).

## The memory-bandwidth heuristic

`max_parallel_restarts` defaults to `None`, which triggers `_auto_max_parallel_restarts(threads, n_seeds)`:

```python
cores = os.cpu_count() or 1
return max(1, min(n_seeds, cores // max(threads * 2, 1)))
```

The `(threads * 2)` denominator (vs. naive `cores // threads`) accounts for the fact that ADMIXTURE is memory-bandwidth-bound at typical panel sizes (≥10K samples × ≥500K SNPs at K≥4). Empirical observation on a 16-core / 125 GiB cloud VM with a 15K × 1.14M panel at K=4:

- 5 restarts at `-j3`: each ADMIXTURE process ran at ~155% CPU (of 300% available with `-j3`) and saturated DRAM bandwidth. ~7 cores stalled waiting on memory at any time.
- 2 restarts at `-j8`: each process ran at ~700% CPU. Similar total wallclock with 2.5× lower peak memory.

The heuristic biases toward fewer/fatter parallelism for this reason. Operators who know their hardware better can override by passing an explicit integer. Tested across {1, 4, 8, 16, 32, 64}-core × {1, 3, 8}-thread combinations (`test_builder.TestAutoMaxParallelRestarts`).

## Cancellation contract

When one restart in a parallel build fails, the other in-flight subprocesses must be terminated promptly — otherwise the build hangs for `per_restart_timeout_seconds` (default 24 h) waiting for them to finish naturally. The flow:

1. `SubprocessToolRunner` spawns the child with `start_new_session=True`, giving each subprocess its own process group (`pgid == pid`).
2. The runner's `pid_callback(proc.pid)` registers the PID in the build's `pids` dict (mutex-protected for thread safety).
3. On first-failure, `_cancel_inflight` reads the registered PIDs and sends `SIGTERM` to each via `os.killpg(os.getpgid(pid))`.
4. The executor shuts down with `wait=False, cancel_futures=True` so not-yet-started workers are cancelled and running workers receive their SIGTERM.

Why pgid signaling instead of bare `os.kill(pid)`:

- **PID-recycle safety**: between a subprocess exiting and `_cancel_inflight` reading the registered PID, the kernel might recycle the PID to an unrelated process. Bare `os.kill(old_pid)` would then signal the wrong process. Process group IDs recycle much more slowly (and `start_new_session=True` ensures the pgid is OUR pgid, not the parent's).
- **Grandchild cleanup**: if the spawned subprocess itself spawns children, `killpg` reaches the whole tree.

A safety check refuses to `killpg` if the discovered pgid equals our own pgid (which would mean the runner didn't use `start_new_session` and we'd be signaling ourselves). In that case we fall back to a bare `os.kill(pid)` with a warning log.

For runners that don't use `start_new_session`, the cancellation is best-effort but the library still does its best.

## Validation gates

Three local gates, each gating each commit:

1. **`pytest`** — 175 unit tests under `tests/unit/`. Tests exercise:
   - NumPy SLSQP math on synthetic 100-SNP panels where the true Q is analytically known
   - Pydantic schema validation + JSON round-trip + legacy `Z`-suffix reparse
   - Build idempotency, multimodality failure, best-LL selection
   - Parallel-restart cancellation via real `sleep 30` subprocesses
   - 15-cell heuristic parametrization
   - BED-triplet symlink staging + legacy-real-file refresh
   - Snapshot-diff log fallback (mtime-aware, `.prev`-excluding)
   - Atomic manifest write (via `os.replace` spy)
   - SubprocessToolRunner end-to-end with mock binaries
   - CLI argparse, type-validators, exit codes
   - Console-script smoke (`admixture-cache --help`)

2. **`ruff check src/ tests/`** — line length 88, target Python 3.11+. The library's own code stays minimal-import (e.g. no `from typing import *`).

3. **`mypy src/`** — runs with `strict = true` (`pyproject.toml [tool.mypy]`). The two pandas overrides allow pandas imports to be `Any` if `pandas-stubs` isn't installed. Local dev installs pandas-stubs via `[dev]`; CI installs the same, so local and CI typecheck match exactly.

All three must pass before a commit lands on `main`. The pre-tag dry-run additionally requires `twine check --strict dist/*` to pass.

## CI matrix

`.github/workflows/ci.yml` and `.github/workflows/release.yml` both use the same 8-cell matrix: `{ubuntu-latest, macos-latest} × {3.11, 3.12, 3.13, 3.14}`. The matrix matches the Python versions declared in `pyproject.toml [project] requires-python = ">=3.11,<3.15"` and the classifiers in `[project] classifiers`. Adding a Python version means updating all three places.

The release workflow's `smoke-test-wheel` job installs the built wheel from the build job's artifact (not from source) into a clean venv on each matrix cell and runs the unit tests. This catches packaging issues that source-tree CI can't (e.g., a missing file in `[tool.setuptools.package-data]`, an incorrect entry-point declaration, an exclude pattern that drops a needed file).

## Testing strategy

- **Real subprocess**: `tests/unit/test_cli.py` exercises `SubprocessToolRunner` with `/bin/sleep`, `/bin/false`, and a fake binary that writes deterministic output. The `test_sigterm_sent_to_inflight_children` test in `test_builder.py` spawns a real `sleep 30` and asserts the cancellation path terminates it within 10 s; a `try/finally` finalizer kills any survivors so test failures don't leak zombies.
- **Mock runners**: `_FakeAdmixtureRunner` and `_FakePlink2Runner` in the test files emit the same output shape as the real binaries (synthetic P/Q matrices, `Loglikelihood:` lines, `.prune.in` files) so the builder can exercise its multimodality / best-LL / idempotency logic without an ADMIXTURE install.
- **Synthetic fixtures**: `_write_panel_triplet` writes a minimal valid PLINK BED triplet (magic bytes + tab-separated bim/fam). Real PLINK files would be 100K+; the synthetic shape is enough to exercise file-handling code paths.
- **Hand-written legacy JSON**: `TestLegacyManifestReparse` (test_manifest.py) feeds a hand-written v0.3.x-shape manifest (with string `build_timestamp`) into `PanelCacheManifest.model_validate_json` to lock in the forward-compat claim.

## Error handling

Everything that can foreseeably fail raises `PanelCacheError` (`errors.py`). It's a single exception type so consumers can `try: ... except PanelCacheError:` and catch the whole library's foreseeable-failure surface at once. The library does NOT swallow unforeseen exceptions (e.g. `KeyboardInterrupt`, `MemoryError`, runner-internal `OSError`); those propagate as-is.

Pydantic `ValidationError` is caught at the `load_cache_manifest` boundary and rewrapped as `PanelCacheError("manifest schema validation failed: ...") from exc`. The original ValidationError lives in the chain for diagnostic purposes; consumers only ever see `PanelCacheError`.

`PopAutomationConfigError` is a back-compat alias (`PopAutomationConfigError = PanelCacheError`). It's exported from `admixture_cache.errors` and re-exported on `admixture_cache.__all__` so callers of older library versions can do `from admixture_cache import PopAutomationConfigError` and keep working. The alias is just `=`, not a subclass, so `isinstance(exc, PopAutomationConfigError)` and `isinstance(exc, PanelCacheError)` are the same check.

## Why pandas (a heavyweight runtime dep)

`extract_target_dosage_via_plink2` parses plink2 `--recode A` text output via `pd.read_csv`. The function imports pandas inline (not at module load) so only the projection hot path pays the import cost. Pandas was an implicit dep in v0.x — the source project's environment provided it transitively. Made explicit in v1.0.0 so `pip install admixture-cache` installs everything the default `project_target` flow needs.

A future optimization is to replace `pd.read_csv` with `numpy.genfromtxt` (or a binary BED reader like `bed-reader`) and drop pandas. The text-parsing step currently dominates per-target wallclock (~28 s out of ~30 s total); a binary reader would ~30× the throughput. Scoped as a v1.x stretch goal, not blocking.

## Release flow

See `docs/RELEASE.md` for the per-release runbook. Highlights:

- Tag pattern `v*` triggers `release.yml` which builds sdist + wheel, runs the 8-cell smoke matrix on the built wheel, then publishes to PyPI via OIDC trusted publishing (no API token in repo secrets).
- Tagging happens after `main` CI is green on the release commit. The release workflow re-runs the matrix but on the wheel itself, catching packaging issues source-tree CI can't.
- `workflow_dispatch` is enabled for dry-runs: build + smoke without publish.

## When in doubt

- Reading the code: start with `cli.py:cli()`, follow the subcommand into `_cmd_build` or `_cmd_project`, that's the operator-side flow. For library-side, start with `builder.py:build_panel_cache` then `orchestration.py:project_target`.
- Tracking a behavior to its test: most tests are named after the behavior they're checking (`test_legacy_real_file_refreshed_as_symlink`, `test_sigterm_sent_to_inflight_children`, etc.). `grep -rn "def test_" tests/` to enumerate.
- Adding a Protocol kwarg: add to `runner.py:ToolRunner.run` with a default, plumb through `builder._call_runner`'s conditional dispatch, update `SubprocessToolRunner.run` to honor it, document the requirement in README and the parallel-mode guard if the kwarg is required for parallel execution. Tests for the kwarg should cover (a) modern runner that accepts it, (b) `**kwargs` forwarder, (c) strict legacy runner without it (graceful degradation or clear-error path).
- File an issue rather than working around something opaque: <https://github.com/carstenerickson/admixture-cache/issues>.
