# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-05-26

Additive minor release. API-additive only; no public API breaks
relative to v1.0.0. All v1.0 callers continue to work unchanged.

### Added

- **`SubprocessToolRunner` is now part of the public API** — exported
  from `admixture_cache.__all__` and importable as
  `from admixture_cache import SubprocessToolRunner`. Previously
  required reaching into the implementation module (`from
  admixture_cache.cli import SubprocessToolRunner`).
- **`build_panel_cache(numa_node_per_restart=True)`** — opt-in NUMA
  pinning for parallel restarts on multi-socket Linux hosts. Each
  restart's subprocess is wrapped with `numactl --membind=N --` where
  N cycles through detected NUMA nodes, avoiding the ~2-3× cross-node
  memory-latency penalty when ADMIXTURE's working set exceeds one
  node's local memory. Worth +10-30% on n2-standard-32+ /
  c2-standard-30+ class hardware. No-op on single-socket boxes,
  macOS, or any environment without `numactl` on PATH (logs a warning
  and proceeds with non-pinned execution).
- **`ToolRunner.run(argv_prefix=...)` optional Protocol kwarg.** A
  general-purpose hook for wrapping the spawned argv with a command
  prefix (`numactl`, `taskset`, `nice`, `time`, etc.). When provided,
  the spawned process is `[*argv_prefix, <binary>, *args]` instead of
  `[<binary>, *args]`. Used internally for NUMA pinning; available to
  callers for arbitrary process-wrapping needs. Detected via
  `inspect.signature` — older runners that predate this kwarg
  continue working unchanged.
- **PGEN format support in `align_target_to_panel_bim`.** The function
  now accepts either PLINK 1 BED (`.bed`/`.bim`/`.fam`) or PLINK 2
  PGEN (`.pgen`/`.psam`/`.pvar`) target genotypes; plink2 handles both
  natively via `--bfile`/`--pfile`. Detection is by file extension
  (`.pgen` → PGEN, `.bed` → BED) or sibling-file presence for
  suffixless paths (PGEN preferred when both exist). The output is
  always a BED triplet so downstream dosage extraction is unchanged.
  The `target_bed` parameter name is kept for backward compatibility;
  it accepts PGEN paths despite the BED-flavored name.
- **Hypothesis-driven property tests for `numpy_supervised_projection`.**
  ~65 new property-based test cases covering random panels (K=2..10),
  random Dirichlet Q vectors, dosage missingness up to 80%, boundary
  Q vectors (near-pure single-cluster membership), and recovery
  tolerance scaling with sample size. Catches regressions the
  hand-written analytic-Q cases wouldn't surface.

### Changed — internal layout

- **`_call_runner` + `_runner_supports` moved to `admixture_cache._dispatch`.**
  Previously lived in `builder.py`; relocated so other layer-1 modules
  (notably `alignment.py`) can route their `runner.run()` calls
  through the dispatcher without introducing a layering violation
  (alignment shouldn't depend on builder). `alignment.py` now routes
  its plink2 calls through `_call_runner` — same behavior as before,
  but Protocol-extension forwarding (log_name, pid_callback,
  argv_prefix) now works for alignment too.
- **Module layering is now enforced by `import-linter`** (CI gate). The
  layered contract in `pyproject.toml [tool.importlinter]` mirrors the
  dependency convention documented in `DEVELOPMENT.md`. A new module
  that imports from a lower layer fails CI immediately.

### Fixed

- (none — release is purely additive)

## [1.0.0] - 2026-05-26

First PyPI release. Bundles the v1.0 publication-readiness pass with
the parallel-restart polish work that surfaced from running real
production builds on cloud VMs. API-additive only; no public API
breaks relative to v0.3.1.

### Added — CLI + packaging

- **`admixture-cache` CLI** — new `admixture_cache.cli` module exposes a console script (`admixture-cache build/project/verify/download`) backed by argparse and a default `SubprocessToolRunner`. No third-party CLI framework dependency. `build` wraps `build_panel_cache`, `project` wraps `project_target` (text or JSON output), `verify` reports SHA divergence reasons, `download` is a placeholder for canonical published caches (post-v1.0). Pyproject `[project.scripts]` entry registers `admixture-cache = "admixture_cache.cli:cli"`.
- **CI workflows.** `.github/workflows/ci.yml` runs the pytest + ruff + mypy gates on Python 3.11/3.12/3.13/3.14 × ubuntu-latest/macos-latest. `.github/workflows/release.yml` builds sdist + wheel on tag push, smoke-tests the built wheel across the same 8-cell matrix (installs into a clean venv and runs unit tests + `admixture-cache --help`), and publishes to PyPI via OIDC trusted publishing (no API token in repo secrets).
- **`CHANGELOG.md` + `CONTRIBUTING.md` + `docs/RELEASE.md`.** Changelog adopts Keep-a-Changelog formatting; CONTRIBUTING covers dev setup, the three local validation gates (pytest / ruff / mypy), commit + PR conventions, and the tag → OIDC PyPI release procedure; RELEASE.md documents the one-time PyPI Trusted Publishing setup + per-release runbook.

### Added — schema + tests

- **`PanelCacheManifest` track/continent consistency validator.** A pydantic `model_validator(mode="after")` enforces `track ∈ {regional, continental_admixture, ancestral_cluster}` and the track-specific continent constraint (`ancestral_cluster` requires a continent; the other two must have `continent=None`). Surfaces schema bugs at manifest-write time rather than as silent mis-categorization at consumer-load time.
- **Test scaffolding under `tests/unit/`** — 143 tests covering NumPy SLSQP math on synthetic panels, schema validation + JSON round-trip, sha256 streaming hash + cache-load shape, mock-driven plink2 + ADMIXTURE runner smoke tests, build idempotency + multimodality failure + best-LL selection, the parallel-restart heuristic across 15 (cpu × threads × n_seeds) cells, real-subprocess SIGTERM cancellation on first failure, BED-triplet symlink staging, per-seed log file naming, `SubprocessToolRunner` behavior end-to-end, and installed-console-script smoke.

### Added — parallel-restart polish

- **`build_panel_cache(max_parallel_restarts=None)` triggers a memory-bandwidth-aware heuristic** (`os.cpu_count() // (threads * 2)`, capped at `len(seeds)`, floor 1). Empirically on a 16-core / 125 GiB cloud VM with a 15K × 1.14M panel at K=4, running 5 restarts at `-j3` each ran ADMIXTURE at ~155% CPU (of 300% available) and saturated DRAM bandwidth; 2 restarts at `-j8` typically hit ~700% per-process CPU at similar total wallclock with 2.5× lower peak memory. The default biases toward fewer/fatter parallelism for that reason. Passing an explicit positive integer overrides the heuristic verbatim.
- **`ToolRunner.run(log_name=...)` optional parameter.** Per-restart log filenames (`restart_<seed>.out`) are now passed explicitly to the runner instead of derived from a second-precision timestamp tag. Eliminates the log-collision case where 5 restarts launched in the same second wrote to the same log filename. Build-side detection via `inspect.signature` means runners that predate this addition continue to work without modification.
- **`ToolRunner.run(pid_callback=...)` optional parameter.** Runners report the spawned subprocess's PID via the callback; the build's first-failure handler then `SIGTERM`s any tracked PID that hasn't completed. `Future.cancel()` alone is a no-op against an already-running ADMIXTURE worker — the SIGTERM is what actually frees the cores. Tested end-to-end with a `sleep 30` subprocess that completes in <10 s when a peer raises mid-batch.

### Changed — internal layout + cache build mechanics

- **`PanelCacheManifest.build_timestamp` is now `datetime` (was `str`).** Manifests written by `build_panel_cache` produce ISO-8601 strings on JSON serialization (pydantic's native format), and consumers can compute "how old is this cache?" without a re-parse step. Existing on-disk manifests written by v0.1.0–v0.3.1 carried a string in this field; pydantic re-parses ISO-8601 strings into `datetime` transparently, so old manifests still load.
- **`_core.py` split into six focused modules.** The 935-LOC `_core.py` extracted from the source project has been split into `projection.py` (NumPy SLSQP), `builder.py` (`build_panel_cache` + `_run_one_admixture_restart` + LD-pruning), `manifest.py` (the `PanelCacheManifest` pydantic model), `alignment.py` (target-to-panel plink2 helpers), `io.py` (SHA + manifest load + cache verification), and `orchestration.py` (`project_target` end-to-end wrapper). Public API surface unchanged — all 14 symbols still importable as `from admixture_cache import …`.
- **`_run_one_admixture_restart` stages the BED triplet as symlinks.** Each restart_dir's `panel.bed` / `panel.bim` / `panel.fam` is a symlink to the source file rather than a physical copy. Concurrent restarts now share a single inode for the input, so the OS page cache serves all N processes from one buffered copy of `panel.bed` — meaningful DRAM-bandwidth relief at N≥3 plus ~17 GiB disk saved on a 5-restart regional build. The `.pop` file stays a real copy (it's tiny and a writable file simplifies one-off debugging).
- **`load_cache_manifest` wraps pydantic `ValidationError` as `PanelCacheError`.** A corrupt or schema-stale `manifest.json` previously leaked `pydantic.ValidationError` past the documented exception type; consumers now see a single error class. Bottom-of-stack pydantic message preserved in the exception chain (`from exc`).
- **Library no longer references the source project in docstrings, error messages, or comments.** `_core.py`'s extraction-history annotations (validation-history markers, source-data references, internal bug-tracker IDs, consumer module paths) have been replaced with neutral technical descriptions. Error messages that pointed at the source project's CLI now point at `admixture_cache.build_panel_cache`.
- **`build_panel_cache` docstring documents the memory-bandwidth tradeoff explicitly.** The empirical "5 × threads=3 ≈ 2 × threads=8 in wallclock but 2.5× more peak memory" observation is now captured next to the `max_parallel_restarts` parameter, so operators picking a value have the numbers without having to spelunk a separate workplan.

### Fixed

- **`load_cached_p` error message no longer references a specific consumer.** Was `"run \`ancestry-pipeline build-caches\` to build it"`; now `"build it via \`admixture_cache.build_panel_cache\`"` — generic and correct for any user of the library.
- **Parallel-mode guard requires both `log_name` AND `pid_callback`.** Without `pid_callback`, the failure path can't SIGTERM in-flight restarts and `ThreadPoolExecutor.shutdown` would block for up to `per_restart_timeout_seconds × (N-1)` (default 24 h per worker) waiting for the others to naturally complete. Surface the dependency at build start rather than via a multi-hour hang.
- **`SubprocessToolRunner` no longer relies on `Popen.__exit__` for cleanup.** `Popen.__exit__` calls `self.wait()` with no timeout; a child in uninterruptible-sleep (D-state, NFS hang) after `SIGKILL` would wedge the runner indefinitely, defeating the 30 s post-kill bound. Replaced with an explicit `try/finally` that bounds the wait. The post-kill wait in the `pid_callback`-raised branch is now wrapped in `contextlib.suppress(TimeoutExpired)` so the original callback exception isn't masked by a secondary timeout.
- **`SubprocessToolRunner` spawns with `start_new_session=True`.** Children get their own process group; `_cancel_inflight` signals the pgid via `os.killpg` rather than the bare PID. Mitigates the classic UNIX PID-recycle race where a subprocess exits between PID capture and the cancellation pass and the kernel reassigns the PID to an unrelated process. Falls back to bare-PID signaling (with a warning log) for runners that don't follow the recommended pattern.
- **Manifest write is atomic.** `manifest.json` writes via `tempfile.NamedTemporaryFile` + `os.fsync` + `os.replace` instead of in-place `write_text`. A SIGKILL / power-loss / OS crash between open and final byte write previously left a half-written JSON that `load_cache_manifest` reported as "cache corrupt"; the atomic write preserves the "manifest exists ⇒ cache is valid" invariant.
- **Restart staging refreshes leftover real-file inputs.** A v0.x cache_dir whose `build_restart_<seed>/panel.{bed,bim,fam}` are real files (legacy `shutil.copy2` result) gets them replaced with symlinks on the next v1.x rebuild — previously the `is_symlink() or exists()` branch silently kept the legacy bytes even when the source panel had updated. `panel.pop` is now unconditionally refreshed from source too, so a curator edit can't be masked by a stale copy.
- **Log-discovery snapshot diff is mtime-aware and skips `.prev` files.** The fallback that locates the runner's log when `log_name` was ignored now (a) considers a path "new" if its mtime advanced during the call, not just if the path didn't exist before — handles the `SubprocessToolRunner` rotation pattern that creates a new file at the canonical path; and (b) explicitly excludes any `.prev` rotated artifact so a stale LL value from a prior attempt can't be misattributed to the current restart.
- **Snapshot fallback is gated to sequential mode.** Under parallel execution, sibling workers writing into the same `log_dir` race the snapshot diff and could misattribute logs across seeds. The fallback is now skipped whenever `effective_parallelism > 1` (the stricter parallel-mode guard requires `log_name` support anyway, so the canonical path is reliable in that branch).
- **`ld_prune_panel` routes plink2 calls through `_call_runner`.** Two `ld_prune_panel` invocations sharing a `log_dir` could collide on auto-derived log filenames; the runner now receives a per-call `log_name` (`ldprune_<prefix>_indep.out` / `ldprune_<prefix>_extract.out`) so log files stay disambiguated, and cancellation works on the prune step too.
- **`_runner_supports` recognizes `**kwargs` forwarders.** Adapter-pattern runners declared as `def run(self, **kwargs): ...` now pass the introspection check for any Protocol extension (the previous string-equality returned `False` because the `VAR_KEYWORD` parameter is named `'kwargs'`, not `'log_name'`). Documented in README: a `**kwargs` runner that silently strips unknown kwargs will pass the guard but produce incoherent logs — implementers must actually honor what they forward.
- **Executor finally now uses `wait=False`.** Was `shutdown(wait=True)` which still joined worker threads after a prior `shutdown(wait=False)` on the failure path — defeating the bounded-cancel intent. Background threads honoring SIGTERM finish quickly; threads on D-state children are no longer load-bearing for `build_panel_cache` to return control to the operator.
- **Dropped the unused `PyYAML` runtime dependency.** Inherited from the source project's extraction; the library reads YAML SHAs via `sha256_file` but never parses YAML. Removing it shrinks the install footprint without changing any public behavior.
- **Declared `pandas>=2.0,<3` as a runtime dependency.** `extract_target_dosage_via_plink2` parses plink2 `--recode A` text output via `pd.read_csv` (imported inline). Was previously an implicit dep — the source project's environment provided it transitively. Now explicit so `pip install admixture-cache` installs everything the default `project_target` flow needs. Added `pandas-stubs>=2.0` to the dev group so strict mypy in CI matches local typecheck behavior.

## [0.3.1] - 2026-05-26

Pre-publication code-review fixes (`Option A` minimal-touch pass).

### Changed

- **`PanelCacheManifest.geo_filter_yaml_shas` switched to pydantic's `Field(default_factory=dict)`** instead of `dataclasses.field`. Pydantic v2 happened to duck-type the latter, but mixing dataclasses + pydantic factories on a BaseModel was visually confusing; canonical pydantic API now used. No on-disk format change. Dropped `field` from the `dataclasses` import.
- **Library code uses `PanelCacheError` directly throughout** instead of the back-compat `PopAutomationConfigError` alias. The alias remains in `errors.py` so any pre-extraction caller still works; new code in the library is on the canonical name.

### Fixed

- **`extract_target_dosage_via_plink2`: `.values` → `.to_numpy()`.** Pandas `.values` emits `DeprecationWarning` on pandas ≥ 2.0; `.to_numpy()` is the supported API for getting a NumPy array out of a pandas DataFrame.

## [0.3.0] - 2026-05-26

Parallel restart execution lands.

### Added

- **`build_panel_cache(max_parallel_restarts: int = 1)`.** Operator opts into running multiple ADMIXTURE restarts concurrently via a `concurrent.futures.ThreadPoolExecutor`. ThreadPool (not ProcessPool): each restart spawns an ADMIXTURE subprocess that releases the GIL, so Python-side contention is a non-issue. Effective parallelism clamped to `min(max_parallel_restarts, len(seeds))`. Sequential default preserves prior behavior with no resource surprises; users explicitly opt in by passing a larger value.
- **Per-restart isolation under `cache_dir/build_restart_<seed>/`.** Each restart copies the panel triplet + `.pop` into its own staging directory so concurrent ADMIXTURE invocations don't collide on output file names (`<bfile_stem>.<K>.{P,Q}`).
- **Deterministic post-run ordering.** `per_restart_results.sort(key=seed)` after all futures complete, so the `manifest.json` / `restart_sd.json` always read out in seed order regardless of parallel completion order.

### Changed

- **Parallel-restart failure semantics** — first failed future raises `PanelCacheError` wrapping the underlying exception; pending futures are cancelled. Successful restarts that completed before the failure remain on disk for debugging; cache is NOT marked valid (manifest.json not written) so consumers correctly see "cache absent".

### Performance

- 5-seed K=21 regional builds shrink from ~60–70 hr (sequential) to ~12–14 hr (5-way parallel on a machine sized to host `threads × max_parallel_restarts` workers comfortably). Smaller K=4 builds shrink proportionally.

## [0.2.0] - 2026-05-26

LD-pruning helper + thread-count default bump.

### Added

- **`ld_prune_panel(panel_bed, output_prefix, plink2_runner, …)`.** Two-step plink2 invocation: `--indep-pairwise` identifies the LD-pruned variant subset; `--extract --make-bed` produces the pruned BED. Per Alexander et al. 2009 + internal speedup measurements, LD-pruning is the dominant cost-cutter for supervised-ADMIXTURE training — pruned SNPs are statistically more independent so ADMIXTURE converges in fewer iterations, and the reduced SNP count cuts per-iter cost. A 50 kb / step 5 / r² 0.5 prune typically retains 30–50% of variants and yields 3–5× total speedup. The `.pop` file from the original panel stays valid (per-sample, not per-variant) so callers just copy or re-emit it next to the pruned output.

### Changed

- **`build_panel_cache(threads=16)`** — default thread count bumped from 8 to 16. Standard cloud-VM SKUs (n2-standard-16 / e2-standard-16) have 16 vCPUs; ADMIXTURE's QN/Block step scales reasonably to that level. Expected ~1.5–1.8× wallclock improvement on the regional K=21 case (independent of LD-pruning).

## [0.1.0] - 2026-05-26

Initial release. Extracted from a private monorepo's
`pop_automation/admixture_projection.py` (~744 LOC, validated against real
multi-thousand-sample workloads).

### Added

- **`build_panel_cache`** — stock ADMIXTURE × N restarts, multimodality validation across restarts (max per-cluster SD must stay under `sd_threshold`), best-LL P-matrix selection, atomic manifest write. Idempotent: re-running with a matched SHA on `panel.bim` + clusters YAML + K + geo-filter YAMLs is a no-op that returns the existing manifest.
- **`project_target`** — end-to-end per-target projection: load + verify manifest, align target.bed to cached panel.bim (variant set + REF/ALT axes via plink2 `--alt1-allele`), extract dosage as NumPy array via plink2 `--recode A`, load cached P, run NumPy SLSQP solver, return a `ProjectionResult` with the Q vector + provenance metadata.
- **`numpy_supervised_projection`** — pure NumPy/scipy supervised-ADMIXTURE projection. Maximizes the binomial admixture likelihood subject to the simplex constraint (`sum(q)=1`, `q≥0`) via scipy SLSQP. Matches stock `admixture --supervised` Q to within ~1e-5 absolute on representative panels (15K samples × 850K SNPs at K=4). Converges in ~9 iterations / ~0.02 s.
- **`PanelCacheManifest`** — pydantic schema for the canonical cache contract (SHA pins on panel + clusters YAML + optional geo-filter YAMLs, K, ADMIXTURE version, seeds used, best seed + LL, restart SD bounds, cluster order, build wallclock + timestamp).
- **`PanelCacheError`** — single-exception surface so consumers can catch one type. `PopAutomationConfigError` is shipped as a back-compat alias.
- **`ToolRunner` Protocol** — minimal `run(args, cwd, log_dir, timeout_seconds)` interface; admixture-cache invokes plink2 + ADMIXTURE through it, with no host-framework dependency.
- **Cache I/O + verification helpers** — `load_cached_p`, `load_cache_manifest`, `verify_cache_matches_current_config`, `sha256_file`. The verification helper returns `(matched, reason)` so callers can log the specific SHA divergence rather than chasing a generic "cache invalid".

[Unreleased]: https://github.com/carstenerickson/admixture-cache/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/carstenerickson/admixture-cache/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/carstenerickson/admixture-cache/compare/v0.3.1...v1.0.0
[0.3.1]: https://github.com/carstenerickson/admixture-cache/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/carstenerickson/admixture-cache/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/carstenerickson/admixture-cache/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/carstenerickson/admixture-cache/releases/tag/v0.1.0
