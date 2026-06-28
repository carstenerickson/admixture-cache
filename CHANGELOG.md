# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Heterozygosity warning for pseudo-haploid / low-coverage targets
  (SCIENCE.md D17).** `project_target` now computes the target's observed
  heterozygosity, reports it on `ProjectionResult.heterozygosity`, and emits a
  `UserWarning` when it is essentially zero (the signature of pseudo-haploid /
  haploidized data, or a very low-coverage diploid sample, which heterozygosity
  alone cannot distinguish). The warning is advisory and never changes the
  projection.
  - **Honest framing (verified).** Contrary to the naive reading of D17, feeding
    pseudo-haploid hard calls (g in {0,2}) into the diploid Binomial(2, f)
    likelihood does NOT bias this tool's projection point estimate: that
    likelihood is exactly twice the correct Bernoulli (n=1) one at every site, so
    the MLE argmax is identical (confirmed numerically, max |Q difference| about
    4e-6). The large pseudo-haploid bias in the literature is in ADMIXTURE's
    joint P+Q build with small panels, which admixture-cache delegates to stock
    ADMIXTURE, not in fixed-P projection. A `ploidy` parameter was therefore
    deliberately not added (it would be an inert knob). A genotype-likelihood
    projection path, which genuinely improves low-coverage estimates, is being
    added separately.

- **Strand-ambiguous (A/T, C/G) SNP handling (SCIENCE.md D11).** REF/ALT
  harmonization via `plink2 --alt1-allele` matches by allele letter, so it
  cannot fix a strand-ambiguous SNP whose target is on the opposite
  strand: the allele set is invariant under complement, the forcing
  "succeeds", and homozygous dosages are silently inverted (0<->2),
  biasing Q. Confirmed empirically against plink2 v2.0.0. The fix is
  on by default, with an opt-out at both stages:
  - **Build-time guard.** `build_panel_cache` now refuses to build from a
    panel that contains A/T or C/G SNPs (`exclude_strand_ambiguous=True`,
    the default), so a cache is strand-safe by construction. Clean the
    panel first with the new `strip_strand_ambiguous_snps` plink2 helper
    (analogous to `ld_prune_panel`). Pass `exclude_strand_ambiguous=False`
    (CLI `--keep-strand-ambiguous`) to keep them.
  - **Projection-time guard.** `align_target_to_panel_bim` /
    `project_target` now drop strand-ambiguous panel SNPs from the
    alignment via `plink2 --exclude`, protecting caches built before this
    release (which may still contain them) with no rebuild. By default
    (`exclude_strand_ambiguous=None`) `project_target` excludes them
    protectively, using the manifest only to skip needless work: a
    build-certified-clean cache (`strand_ambiguous_excluded=True`) skips
    the per-projection `panel.bim` scan, while a cache that may still
    contain them (operator kept them at build, or a legacy pre-D11 cache)
    has them excluded. Whether a target shares the panel's strand
    convention is a per-(panel, target) property decided at projection,
    not bakeable at build, so keeping ambiguous SNPs is a per-projection
    opt-in (CLI `--keep-strand-ambiguous`, or `exclude_strand_ambiguous=
    False`), deliberately not inherited from the build. Pass `True` to
    force exclusion.
  - **Manifest.** New optional `strand_ambiguous_excluded: bool | None`
    field records the build's decision (`True` certified clean, `False`
    opted to keep, `None` legacy). `schema_version` stays `1`; the field
    is provenance only and not part of the cache-validity gate.
  - New public helpers: `strip_strand_ambiguous_snps`,
    `is_strand_ambiguous`, `strand_ambiguous_variant_ids`.

### Changed

- **`PanelCacheManifest` now tolerates unknown fields
  (`extra="ignore"`, was `extra="forbid"`).** Every optional field added
  over the project's history (`panel_pop_sha256`,
  `strand_ambiguous_excluded`, ...) kept `schema_version` at 1, and the
  serialized manifest always includes the new key, so a cache published by
  a newer library version was rejected on load by any older consumer
  (the `download_cache` and `project_target` paths both call
  `load_cache_manifest`, which `model_validate_json`s the manifest). With
  `extra="ignore"`, an older consumer ignores fields it does not recognize
  and loads the cache. Known-field types are still validated and tarball
  integrity is still covered by the SHA-256 check; the manifest is
  machine-written, so rejecting unknown keys bought little. Backward
  compatibility (new code reading old manifests) is unchanged: absent
  optional fields fall back to their defaults. Unknown keys are not
  dropped *silently*: an unrecognized field (a newer library's field, or a
  typo) is logged as a warning naming it, so a stale/mistyped key cannot
  vanish without a trace.

- **`ld_prune_panel` default raised to `--indep-pairwise 200 25 0.4`
  (was `50 5 0.5`), grounded in the literature (SCIENCE.md D7).** A
  survey of human ancient-DNA ADMIXTURE methods sections (AADR 1240K and
  Human Origins panels) found variant-count windows used over kb windows
  roughly 17:1, with `200 25 0.4` by far the most common single recipe
  (the Reich-lab / Human Origins house style, e.g. Cardial-LBK 2015
  doi:10.1093/molbev/msv181, Late Neolithic Switzerland 2020
  doi:10.1038/s41467-020-15560-x, Shimao 2025
  doi:10.1038/s41586-025-09799-x; the ADMIXTURE-manual `50 10 0.1` is the
  main alternative). The previous `50 5 0.5` default was valid but light:
  a smaller window and the most permissive r² in common use. The new
  default prunes more thoroughly on both window and r², matching the bulk
  of the published reference literature, which matters because the
  resulting P matrix is cached and reused on every projection. On a dense
  ~1.1M-SNP 1240K panel it retains roughly 450-600K SNPs. Callers who
  passed explicit `window_size`/`step_size`/`r2_threshold` (or the
  deprecated `window_kb`) are unaffected; only the defaults changed.
  `ld_prune_panel` is an optional pre-build helper, so this does not alter
  any existing cache.

### Fixed

- **`ld_prune_panel` parameter `window_kb` was a misnomer (SCIENCE.md
  D7).** The value is passed straight to `plink2 --indep-pairwise` as a
  bare integer, which plink2 interprets as a window in VARIANTS, not kb
  (a kb window needs an explicit "kb" suffix AND a step of 1; plink2
  rejects a kb window with any other step, so the documented "kb"
  reading was never even valid alongside the default step of 5). The
  conventional bare-integer prune the helper runs is a variant-count
  window with a variant-count step, which is the intended and standard
  behavior. Confirmed against plink2 v2.0.0. The rename itself is
  behavior-preserving (a caller passing explicit parameters gets an
  identical plink2 command): the parameter is renamed `window_size`, the
  docstring and help now say "variants", and the old `window_kb` keyword
  is still accepted as a deprecated alias (with a `DeprecationWarning`)
  so existing callers do not break. The default values also change; see
  Changed above. Passing **both** `window_size` and the deprecated
  `window_kb` now raises `TypeError` (they set the same window, so
  silently letting one win would hide a caller mistake); `window_size`
  defaults to `None` internally and resolves to a module-private default
  of 200 so an explicit value is distinguishable from the default.

## [1.5.0] - 2026-06-09

### Added

- **`PanelCacheManifest.panel_pop_sha256` — a direct guard on the
  supervised-label `.pop` file (admixture-cache #2).** The cache-validity
  gate previously hashed `panel.bim`, `clusters_yaml`, the geo-filter
  YAMLs, and `K`, but not `panel.pop`. Those inputs capture a
  config-driven label change transitively (`panel.pop` is deterministic
  from `clusters_yaml` + the panel sample set), so the only gap was a
  *non-config* path that rewrites `panel.pop` while leaving every hashed
  input untouched — e.g. a hand-edit between builds into the same
  `cache_dir`. `build_panel_cache` now records `sha256(panel_pop_file)`
  in the manifest and feeds it into its own idempotency check, so such an
  edit forces a rebuild instead of being silently skipped.
  - **Optional / back-compat.** The field defaults to `None`.
    `verify_cache_matches_current_config` gained an
    `expected_panel_pop_sha256` parameter and compares it **only when
    both** the caller supplies one and the cache recorded one. A legacy
    cache (`panel_pop_sha256 is None`) is never invalidated on this basis
    alone, so upgrading the library does not trigger a spurious
    (potentially many-hour) rebuild of existing caches. `schema_version`
    stays `1`, matching how prior optional fields
    (`pgen_samplebind_version`, `track`/`continent`) were added. The
    geo-filter check stays strict on `None` because a missing geo pin is
    itself a config signal; a missing `panel.pop` sha only means "this
    cache predates the field".
  - **Scope: a build-time guard.** `panel.pop` is a build *input*, not a
    distributed cache artifact — projection never reads it and it is not
    in the release tarball — so there is no consumer-side
    `cache_dir/panel.pop` to hash at projection time. Defense-in-depth:
    no uncovered `panel.pop` mutation path is currently known, and the
    guard can only ever invalidate more aggressively, never produce a
    false cache hit.
  - `build_panel_cache` now fast-fails with a clear `panel .pop missing`
    error when `panel_pop_file` is absent, mirroring the existing `.bim`
    check, instead of surfacing a raw error later during restart staging.

## [1.4.2] - 2026-05-29

### Fixed

- **`numpy_supervised_projection` could return a confidently-wrong Q
  with `converged=True` on large panels — CRITICAL, present in every
  release ≤ 1.4.1.** The projection minimized the *summed* negative
  log-likelihood, whose gradient scales with the SNP count (~1e6 at
  1.1M SNPs). SLSQP doesn't auto-scale, so against the O(1) sum-to-1
  constraint Jacobian the QP subproblem is badly conditioned and the
  optimizer stalls at a simplex corner while reporting success. On the
  real 1.14M-SNP W_Eurasia AADR panel (K=4, cluster correlations
  ~0.85), projecting an interior mixture with true Q `[0.20, 0.50,
  0.25, 0.05]` returned **`[0, 0, 1, 0]`** (max error 0.75) — a
  silently garbage result, no error raised. The fix normalizes the
  objective (and gradient) by the observed-SNP count — the **MEAN**
  per-SNP NLL instead of the sum. The argmax is identical (scaling by
  the constant 1/M can't move the optimum) but the gradient becomes
  O(1) regardless of panel size, so SLSQP's tolerances behave the same
  at 100 SNPs and 1.14M. Post-fix the same projection recovers Q to
  within **0.0024**.

  - **Why it escaped every prior release:** all unit/integration
    fixtures used 100–2000-SNP synthetic panels, where the summed
    gradient stays small enough that SLSQP works. The stall is an
    emergent property of the full-scale real P — it does not reproduce
    on any contiguous slice of the real matrix (≤500K rows are fine)
    nor on synthetic panels up to 1.5M SNPs. It surfaced only when the
    published 1.4.1 wheel was run against the production caches.
  - **Impact:** any consumer projecting real targets against a
    real ~10⁵–10⁶-SNP panel could receive a wrong Q with no failure
    signal — most damaging for interior (admixed) targets, which are
    precisely what an ancestral-cluster track exists to measure.
    Single-ancestry targets near a simplex corner were more likely to
    project correctly by luck. **Re-run any projections made with
    ≤1.4.1 against large panels.**
  - **Regression guard:** a white-box test
    (`TestObjectiveNormalization`) intercepts the objective handed to
    SLSQP and asserts it is the mean (÷M) NLL, not the sum — it fails
    on ≤1.4.1 and needs no real data (the emergent full-scale stall
    isn't cheaply reproducible). Plus large correlated-cluster panel
    coverage in `TestLargePanelConditioning`.

### Documentation

- **`build_panel_cache` docstring documents the fully-labeled-panel
  fast path.** When `panel_pop_file` has no unlabeled (`-`) rows,
  supervised ADMIXTURE has no free Q to estimate and reduces to a
  near-closed-form one-iteration per-cluster allele-frequency pass:
  fast (tens of seconds even at K=21) and seed-independent
  (`restart_sd_max` ≈ 1e-16, multimodality check structurally
  vacuous). `seeds=[1]` suffices for such panels; extra seeds are
  redundant. Surfaced while forensically confirming an 88 s K=21
  regional cache build was legitimate, not a short-circuit.

## [1.4.1] - 2026-05-29

### Fixed

- **`project_target`: reindex the target dosage to the full panel variant
  order before projection.** `align_target_to_panel_bim` runs
  `plink2 --extract panel.bim`, which keeps only the target∩panel variants in
  the *target's* order. Whenever the target was missing any panel SNP (the
  common case for a real sample vs an 850K-SNP panel) the extracted dosage was
  shorter than the cached `P` matrix and `project_target` aborted with
  `cached P has N SNPs but aligned target dosage has M`. Even at
  coincidentally-equal length the dosage was mis-aligned row-for-row against
  `P`, silently projecting each offset SNP's allele count against the wrong
  cluster frequencies. `project_target` now reindexes the dosage to the full
  `panel.bim` order by variant ID (new `reindex_dosage_to_panel_order`),
  NaN-filling panel SNPs absent from the target — the SLSQP projection already
  treats NaN as missing. Surfaced projecting HG00096 (≈579.3K of 579.7K panel
  SNPs) against a regional cache.

## [1.4.0] - 2026-05-27

Two coordinated changes shipped together:

1. **Schema cleanup.** `track` and `continent` were enforced as a
   three-value enum + a continent-required-only-for-ancestral_cluster
   rule — vocabulary borrowed from one specific consumer
   (ancestry-pipeline). They're now free-text provenance labels.
2. **Distribution-layer hardening.** Code review of v1.3 + v1.4
   surfaced 15 findings (1 Python-version floor, 1 missing GitHub
   API pagination, 4 medium-severity download_cache correctness
   issues, 9 polish items). All addressed.

API surface unchanged; manifest schema unchanged. Migration is
zero-op for existing callers + existing caches.

### Changed (schema cleanup, non-breaking)

- **`PanelCacheManifest.track` is now optional + free-text.** Was
  `track: str` with a `model_validator` enforcing
  `track ∈ {regional, continental_admixture, ancestral_cluster}`;
  now `track: str | None = None` with no validator. Any string
  (including `None` and `""`) is accepted. Existing manifests
  using the three legacy values still load — they're just no longer
  constrained.
- **`PanelCacheManifest.continent` no longer coupled to `track`.**
  The `_validate_track_continent_consistency` model_validator that
  required `continent` to be set iff `track == "ancestral_cluster"`
  is gone. Either field can be set or omitted independently.
- **CLI `admixture-cache build --track` argparse `choices=[...]`
  constraint dropped.** Was rejecting unknown strings at parse time;
  now accepts any string. The early track/continent validation block
  in `_cmd_build` (which mirrored the library's now-removed
  model_validator) is also gone.
- **`build_panel_cache` keyword arg `track: str` → `track: str | None
  = None`.** Callers can omit it entirely.

#### Why

The three named tracks (`regional`, `continental_admixture`,
`ancestral_cluster`) are ancestry-pipeline's routing categories,
not library-fundamental concepts. The cache itself is just
"supervised ADMIXTURE on a specific panel × K × clusters_yaml"
— it doesn't care what the consumer plans to call its outputs.
By dropping the validator, the library stays purely about the
caching mechanics; consumers attach their own semantics at their
boundary.

#### Migration

- **No-op for existing caches.** Every cache built under
  v1.0–v1.3 has `track ∈ {regional, continental_admixture,
  ancestral_cluster}` — all three remain valid free-text values.
  Loading old manifests is unchanged.
- **No-op for existing callers.** `build_panel_cache(track="regional", ...)`
  still works. `admixture-cache build --track regional` still works.
- **New flexibility.** Custom track labels (`track="my_pgs_pipeline"`,
  `track=None`) now accepted by both the library and the CLI.
- **Schema version unchanged** (still `schema_version=1`). No
  forward-compat hazard: old code reading new manifests sees the
  same field types, just with a wider value space.

#### Out-of-band consumer note

Anyone who relied on the v1.0–v1.3 validator catching typos like
`track="regoinal"` should add that validation at their own
boundary (a typed enum, an `Annotated[str, AfterValidator(...)]`
field on their own model, etc.). The library no longer offers that
service.

### Fixed (distribution-layer hardening)

#### Correctness

- **Python floor bumped to `>=3.11.4`** (was `>=3.11`).
  `distribution.py:_safe_extract_tarball` uses PEP-706 `filter="data"`
  which was backported to 3.11.4. pip now refuses to install on
  3.11.0–3.11.3 instead of letting users hit a confusing `TypeError`
  mid-extract.
- **`list_available_caches` follows GitHub Releases pagination**
  via `?per_page=100` + `Link: rel="next"` header parsing. Repos
  with >30 releases no longer silently hide older cache versions.
- **`download_cache` acquires `fcntl.flock`** on
  `<cache_root>/.<name>.lock` for the install duration, serializing
  concurrent calls for the same name. Two terminals running
  `admixture-cache download foo` no longer race on the final rename.
- **`download_cache(name=...)` validates `name` as a flat directory
  identifier** — rejects `/`, `\`, `..`, leading dots, absolute
  paths, empty string before any network I/O. Closes the Python-API
  path-traversal escape (`name="../evil"` would write outside
  `cache_root`).
- **Long-tail exception wrapping.** `download_cache`'s outer
  except now catches `http.client.HTTPException`, `tarfile.TarError`,
  `OSError`, `ValueError` and re-raises as `PanelCacheError`. Raw
  tracebacks no longer leak past the CLI's `except PanelCacheError`
  on edge failures (disk full, corrupt tarball, malformed
  Content-Length).
- **Total wall-clock budget on `download_cache`.** Default
  10× the per-read `timeout`, override via
  `$ADMIXTURE_CACHE_DOWNLOAD_BUDGET_SECONDS`. A slow-loris server
  drip-feeding bytes within the per-read timeout can no longer hold
  the download open indefinitely.

#### Robustness

- **`_find_manifest_root` filters `__MACOSX/`** (macOS Finder
  Compress artifacts) and hidden `.*` directories. Tarballs packed
  via Finder no longer trip the "ambiguous layout" branch.
- **tz-aware datetime fallback.** `list_available_caches` returns
  `datetime.fromtimestamp(0, tz=UTC)` (was naive) in the
  published_at-missing path. Mixing aware/naive datetimes in
  `sorted(releases, key=lambda r: r.published_at)` no longer raises
  `TypeError`.
- **`except BaseException` cleanup.** `download_cache`'s
  extract-dir cleanup catches `BaseException` (with re-raise) so
  `Ctrl-C` during a multi-GB extract removes the orphan
  `.{name}.extract-*` dir before propagating.
- **Backup-restore-failure logging.** When `download_cache` fails
  to install AND fails to restore the prior cache from `.old-*`,
  the failure is now logged at ERROR level pointing operators at
  the recoverable path (was silently `contextlib.suppress`-ed).

#### Polish

- **CLI `_progress_bar` clamps to 100%.** Under-reporting
  Content-Length headers no longer render "120.0%".
- **`_TAG_PATTERN` uses `re.fullmatch`** via `\A...\Z` anchors;
  rejects pedantic edge cases like trailing newlines.
- **README "Status" section rewritten** to list v1.0 → v1.4 with
  one line each (was frozen at "v1.0.0 — first PyPI release").
- **`docs/PUBLISH_CACHE.md` clarifies** that cache names are
  operator-chosen free-text (the three legacy `track` values are
  suggestions, not requirements).

### Test coverage

- 13-cell `test_track_is_free_text_no_validator` covers empty,
  oversize (10K chars), Unicode, control chars,
  SQL-injection-style, path-traversal-style strings — proves the
  "ANY string accepted" docstring claim.
- 6-cell `test_continent_no_longer_coupled_to_track`.
- 4-cell `test_track_accepts_any_string_no_enum_constraint` +
  `test_track_and_continent_independent_no_constraint`.
- New `TestNameValidationGuard` (10 cells) — every flavor of
  invalid `name` rejected pre-network.
- New `TestFindManifestRootMacOSResourceFork` (2 cells) —
  `__MACOSX/` + hidden-dir siblings filtered.
- New `TestConcurrentInstallLock` — two threads contending for
  `_exclusive_lock` serialize correctly (timed lock-order
  assertion).
- New `TestSlowLorisBudget` — slow-streaming server exceeds the
  wall-clock budget and raises `PanelCacheError` instead of
  hanging.

Total: 312 unit tests pass (was 285); 10 integration tests pass
against ADMIXTURE 1.4 + plink2; ruff + mypy strict + import-linter
all green; `twine check` PASSED for sdist + wheel.

## [1.3.0] - 2026-05-26

Adds the panel-cache distribution layer — `admixture-cache download`
is no longer a placeholder. Library + CLI fetch canonical caches
from GitHub Releases with streaming SHA-256 verification and atomic
on-disk install. Publisher-side runbook in `docs/PUBLISH_CACHE.md`.

### Added

- **`admixture_cache.distribution` module** — new public API:
  - `download_cache(name, ...) -> Path` fetches a published cache
    by name and installs at `<cache_root>/<name>/`, returning the
    cache_dir path suitable for `project_target(cache_dir=...)`.
  - `list_available_caches() -> list[CacheRelease]` queries the
    GitHub Releases REST API and returns one `CacheRelease` per
    release matching the `cache-<name>-<version>` tag convention
    with both `<name>.tar.gz` and `<name>.tar.gz.sha256` assets.
  - `CacheRelease` dataclass surfacing tarball URL, sha256 URL,
    size, publish date, version number, release page URL.
- **`admixture-cache download` CLI is now functional** (was a v1.0
  placeholder returning exit 2 with a "not yet available" message):
  - `admixture-cache download <name>` — install the latest version
    of a named cache to `<cache-root>/<name>/`.
  - `admixture-cache download --list` — enumerate available caches
    on the configured GitHub repo, newest version per name first.
  - `--cache-root PATH` — override the install root (default:
    `$ADMIXTURE_CACHE_ROOT` or `~/.admixture-cache/caches/`).
  - `--cache-version vN` — pin a specific version instead of latest.
  - `--github-repo OWNER/REPO` — query a fork instead of
    `carstenerickson/admixture-cache`.
  - `--force` — overwrite an existing install at the target path.
  - `--quiet` — suppress the streaming progress bar.
- **Streaming download with on-the-fly SHA-256 verification.**
  The tarball is hashed as it's read; memory footprint is bound by
  the 64 KiB chunk size, not the tarball size (caches can be many
  GB). A mismatched sha256 is detected before extraction and the
  partial tempfile is unlinked.
- **Atomic install pattern.** Downloads land in a tempfile inside
  `cache_root`; extraction goes into a sibling
  `.<name>.extract-<uuid>` dir; manifest is loaded for validation;
  ONLY THEN is the directory renamed into the target. A
  pre-existing install at the target (with `--force`) is renamed
  aside first, then removed after the new install succeeds. No
  partial state survives a mid-download crash.
- **`docs/PUBLISH_CACHE.md`** — operator runbook for cutting a
  canonical-cache release. Covers tag convention
  (`cache-<name>-<version>`), tarball layout (flat or wrapped, both
  supported), `.sha256` file format (bare hex or sha256sum-style),
  release notes content guidance, plus an optional GitHub Actions
  snippet for automated publish-on-tag.

### Test coverage

- 23 new unit tests in `tests/unit/test_distribution.py`. Covers:
  default cache-root resolution + `$ADMIXTURE_CACHE_ROOT` override;
  flat vs. wrapped tarball extraction; manifest-root detection
  (single-wrapper vs. ambiguous); sha256 file parsing (bare hex +
  sha256sum format); release filtering (tag mismatch, missing
  asset); HTTP error wrapping; force-overwrite semantics; pinned
  version selection; progress callback invocation; partial-install
  cleanup on failure.
- 6 new CLI tests in `tests/unit/test_cli.py` (replaced the v1.0
  placeholder): no-name-no-list rejection; `--list` rendering;
  `--list` empty-result handling; `download_cache` invocation with
  forwarded kwargs; PanelCacheError → exit 1 mapping.
- Total test count: 289 unit (default `pytest`) + 10 integration
  (opt-in) = 299 (up from 272 in v1.2.0).

### Public API surface

- 3 new exports from `admixture_cache.__all__`: `download_cache`,
  `list_available_caches`, `CacheRelease`.
- No changes to existing exports; no library code paths changed.

## [1.2.0] - 2026-05-26

End-to-end integration testing against real ADMIXTURE + plink2
binaries. No library code changes; all unit tests + the integration
suite green.

### Added

- **`tests/integration/` suite — end-to-end pipeline test against
  real ADMIXTURE 1.4 + plink2 binaries.** Runs the full
  `build_panel_cache` → `project_target` pipeline on a synthetic
  3-cluster panel (90 samples × 2000 SNPs, K=3) with 4 held-out
  targets carrying known Q vectors. Verifies recovery within 0.10
  absolute per-component (empirical max error on the fixtures:
  ~0.03 against ADMIXTURE 1.3.0; the 0.10 tolerance leaves
  generous headroom for 1.3.0 ↔ 1.4.0 drift).
- **`pytest -m integration`** opt-in marker. Default `pytest` run
  excludes the integration suite (markers in `[tool.pytest.ini_options]`
  default the addopts to `-m 'not integration'`). Skipped cleanly
  when ADMIXTURE / plink2 aren't on PATH.
- **Deterministic fixture generator** at
  `tests/integration/_generate_fixtures.py` — writes the panel + 4
  target BED triplets via a pure-Python PLINK BED encoder (no
  third-party generator dep). Seeded with `SEED = 20260526`;
  byte-deterministic across runs. The generated `fixtures/` tree
  (~280 KB) is checked in so the test doesn't depend on the
  generator at run time.
- **CI integration job** at `.github/workflows/ci.yml` —
  downloads ADMIXTURE 1.4.0 Linux from
  `dalexander.github.io/admixture` and plink2 alpha7 Linux x86 from
  cog-genomics.org S3, then runs `pytest -m integration`. Job
  depends on the unit-test matrix (only runs when the matrix is
  green). Linux-only.

### Test coverage

- 10 new integration tests covering: cache directory layout, P/Q
  matrix shapes, cluster_order vs .pop file consistency, Q-recovery
  on each of 4 known-truth targets (pure-A, pure-C, 50/50 A+B,
  three-way 40/40/20), full SNP count usage, and build idempotency
  on the second `build_panel_cache` call.
- Total test count: 262 unit (default `pytest`) + 10 integration
  (opt-in `pytest -m integration`) = 272.

### Documentation

- `DEVELOPMENT.md` "Testing strategy" section expanded with the
  new integration suite + local instructions for running it.
- `CONTRIBUTING.md` "Local dev setup" notes that the integration
  suite is optional.

## [1.1.1] - 2026-05-26

Post-release hotfix addressing the 14 confirmed findings from the
v1.1.0 max-effort code review. All fixes are backward-compatible
relative to v1.1.0; no public API or manifest schema changes.

### Fixed — correctness

- **NUMA slot pool sizing — workers no longer block; cancellation cannot orphan ADMIXTURE subprocesses.** The v1.1.0 / first-pass v1.1.1 implementation sized the `queue.Queue` to `numa_n_nodes` (== `min(n_nodes, effective_parallelism)`); when `n_nodes < effective_parallelism` (e.g. dual-socket box with `max_parallel_restarts=4`), excess workers blocked in `queue.get()` BEFORE registering a PID. On first-failure, `_cancel_inflight` would no-op against blocked workers; after a peer released its slot, the blocked worker would unblock, claim the slot, and spawn a fresh ADMIXTURE subprocess that the cancellation path could never reach — leaving up-to-24h orphans. The queue is now sized to `effective_parallelism`, with entries cycled via `i % numa_n_nodes`. Partial pinning (some workers share a node) is documented + logged as a WARNING; no worker ever blocks.
- **`Path.with_suffix` dotted-stem bug fixed at all remaining call sites, not just `_detect_target_format`.** v1.1.1's first pass introduced `_append_suffix` for `alignment._detect_target_format` only — but `ld_prune_panel` had the same bug at its prune.in / .bed existence checks (caller-supplied `output_prefix` with a dotted stem like `aadr.v66.pruned` would mis-probe to `aadr.prune.in`). Lifted the helper to `admixture_cache._paths.append_suffix` + applied at every user-controlled callsite. Library-controlled with_suffix calls (where the path is guaranteed to carry the expected extension by docstring contract) are unchanged.
- **`align_target_to_panel_bim` validates the FULL aligned BED triplet, not just `.bed`.** Previously a partial plink2 output (`.bed` present but `.bim` truncated) returned success; downstream `extract_target_dosage_via_plink2` raised "BED triplet incomplete" — confusingly attributing the failure to the wrong step. Now the alignment helper itself raises with the missing-sibling list.
- **`ld_prune_panel` validates the full pruned BED triplet** with the same `append_suffix` + missing-sibling-list pattern.
- **`_detect_target_format` no longer mangles paths with dotted stems.** When the input had no plink suffix but contained a dot in its name (e.g. `cohort.v2`), `Path.with_suffix(".pgen")` REPLACED the `.v2` segment instead of appending — the function probed `cohort.pgen` rather than `cohort.v2.pgen` and either raised a false "not found" or silently read an unrelated file with the same probed name. Fix uses raw name concatenation (`stem.parent / (stem.name + suffix)`) for sibling probing.
- **NUMA pinning uses a slot-based free-list, not a static seed→node map.** v1.1.0 precomputed `numa_node_for_seed = {s: i % n_nodes for i, s in enumerate(sorted(seeds))}` at submit time; for `len(seeds) > effective_parallelism` (the common case — 5 seeds × 2 parallel slots), once one restart finished and the next started while a peer was still running, two concurrent restarts could land on the same node. Replaced with a `queue.Queue` of free nodes: workers claim on entry, release on exit; at most `min(n_nodes, parallelism)` workers are in flight and each holds a distinct node.
- **NUMA pinning warns + degrades to non-pinned when the runner doesn't support `argv_prefix`.** v1.1.0 silently dropped the kwarg on v1.0-era runners (no warning, INFO log still announced "NUMA pinning enabled"). Now logs a clear warning naming the missing kwarg and points at the v1.1 Protocol extension.
- **`_detect_numa_nodes` now filters for directories with the `nodeN` naming convention.** v1.1.0's `p.name.startswith("node")` would count any file/symlink starting with "node" (e.g. a future `node_list` sysfs entry, container-overlay quirks). Now requires `p.is_dir() and p.name[4:].isdigit()`.
- **`_detect_target_format` validates sibling files in the explicit-suffix branches.** v1.1.0 only checked sibling existence in the suffixless-probe fallback; passing `target.bed` with a missing `.bim`/`.fam` got an opaque plink2 downstream error instead of the actionable `PanelCacheError` the helper is designed to produce. Now raises `PanelCacheError` with the missing-sibling list in every branch.
- **`SubprocessToolRunner` FileNotFoundError attributes the correct binary.** When `argv_prefix=["numactl", "--membind=N", "--"]` was set and numactl wasn't on PATH, the error message said `binary {self.binary!r} not found` — but `self.binary` is the ADMIXTURE/plink2 binary, not numactl. Now uses `exc.filename` (or `cmd[0]`) and includes the invocation prefix in the message.

### Fixed — UX / operator surface

- **`admixture-cache build --numa-node-per-restart` flag exposed via CLI.** v1.1.0 added the kwarg to `build_panel_cache` but the CLI didn't forward it; the headline NUMA feature was only reachable via the Python API. Now a store_true flag with help text.
- **`admixture-cache build --pgen-samplebind-version` flag exposed via CLI.** Same gap as above for the optional provenance field on the manifest.
- **`extract_target_dosage_via_plink2` accepts PGEN target paths.** v1.1.0 added PGEN support to `align_target_to_panel_bim` but kept the dosage extractor hardcoded to `--bfile`. Now routes through `_detect_target_format` like its sibling. (No-op on the standard `project_target` orchestration path where input is always BED post-alignment, but matters for direct API callers.)
- **`project_target` creates a unique per-call subdirectory under `work_dir`.** Format: `work_dir/<target-stem>-<uuid8>/`. Avoids intermediate-file + log collisions when callers reuse a `work_dir` across multiple targets (batch projection scripts, test fixtures). The `SubprocessToolRunner` log rotation (`.prev`) only kept one generation, so collisions silently dropped debug history.

### Fixed — test / docs hygiene

- **`test_numa_disabled_by_default_no_argv_prefix` strengthened to verify kwarg ABSENCE.** v1.1.0's assertion `observed == [None, None]` couldn't distinguish "kwarg omitted" from "kwarg forwarded as None"; a regression in the dispatcher's `if argv_prefix is not None` guard would have passed the test. Now captures `set(kwargs.keys())` and asserts `argv_prefix not in` it.
- **`SubprocessToolRunner` moved out of `cli.py`** into a dedicated `_subprocess_runner.py` module. v1.1.0's bottom-of-`__init__.py` re-export created a `__init__ → cli → __init__` circular import that worked but was order-sensitive and fragile to future re-exports. The new layout has `__init__.py` and `cli.py` BOTH importing top-down from `_subprocess_runner`. Import-linter contract updated to enforce the new layering.
- **`DEVELOPMENT.md` stale references to `builder._call_runner` updated to `_dispatch._call_runner`** (the v1.1.0 lift was documented in the diff but not in the prose); module map + dependency-graph mermaid now include `_dispatch.py` and `_subprocess_runner.py`.
- **CHANGELOG documents the v1.1.0 log filename rename** (`plink2_<tag>.out` → `align_<name>.out` / `dosage_<name>.out` for the alignment/dosage steps). Log-scraping pipelines watching for the old prefix should be updated.

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

[Unreleased]: https://github.com/carstenerickson/admixture-cache/compare/v1.4.2...HEAD
[1.4.2]: https://github.com/carstenerickson/admixture-cache/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/carstenerickson/admixture-cache/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/carstenerickson/admixture-cache/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/carstenerickson/admixture-cache/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/carstenerickson/admixture-cache/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/carstenerickson/admixture-cache/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/carstenerickson/admixture-cache/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/carstenerickson/admixture-cache/compare/v0.3.1...v1.0.0
[0.3.1]: https://github.com/carstenerickson/admixture-cache/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/carstenerickson/admixture-cache/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/carstenerickson/admixture-cache/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/carstenerickson/admixture-cache/releases/tag/v0.1.0
