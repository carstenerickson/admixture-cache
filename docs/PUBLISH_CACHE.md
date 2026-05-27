# Publishing a canonical panel cache

This is the operator runbook for cutting a new GitHub Release that the
`admixture-cache download` command can fetch. The format below is
load-bearing — the downloader's discovery + verify code (in
`admixture_cache.distribution`) depends on it.

For the consumer-side view (`admixture-cache download <name>`), see
the README.

## When to publish

Each published cache is a heavyweight commitment — once a release is
out, consumers will pin to it for months or years. Publish when:

- The panel + clusters_yaml + K combination is stable and you've
  validated multimodality (per-cluster restart SD ≤ threshold).
- The build was produced from a panel + clusters_yaml that's
  reproducible (the SHA-pins in the manifest let consumers detect
  if their local config diverges).
- You're prepared to maintain backward-compat for that exact name +
  version combination indefinitely (new versions are additive; old
  versions stay around).

A cache being slow to build (multi-day) is a STRONG signal it's worth
publishing — that's exactly the cost the cache-distribution system
exists to amortize.

## Naming convention

Pick a cache name matching `[a-z0-9_]+(-[a-z0-9_]+)*` — lowercase
letters, digits, underscores. Hyphens are tolerated but underscores
are preferred (`cache-<name>-<version>` splits on the *last* hyphen,
so name-internal hyphens parse but look messy).

The name is **operator-chosen and free-text** — admixture-cache doesn't interpret it. A naming scheme that humans can grep through six months from now is what matters. Suggested format:

1. **Track or purpose label** — `regional`, `continental_admixture`, `ancestral_cluster`, or any string that fits your pipeline's vocabulary (e.g. `my_polygenic_score_pipeline`). The manifest's `track` field is free-text since v1.4.
2. **K value** (e.g. `k21`, `k4`)
3. **Panel source** (e.g. `aadr_v66_ho`, `hgdp_1kgp`)
4. **Scope** — anything finer-grained you need (continent, region, cohort filter, etc.)

Examples that follow the convention:

- `regional_k21_aadr_v66_ho`
- `continental_admixture_k4_hgdp_1kgp`
- `ancestral_cluster_k4_aadr_v66_w_eurasia`
- `my_pgs_k8_internal_panel_v3` — any operator-meaningful string works

## Version numbers

Format: `v1`, `v2`, `v3`, …

Bump the version when:

- The panel.bim SHA changes (different variant set or REF/ALT
  alignment).
- The clusters.yaml SHA changes (curation update).
- The K value changes.
- The ADMIXTURE binary version changes (rare; only when a new
  ADMIXTURE release lands).

Don't reuse a version — a published `v1` is permanent. If you find a
bug after publishing, fix it in `v2`. The downloader picks the
highest version by default; consumers who need the older version
pass `--cache-version v1`.

## Release procedure

### 1. Build the cache

```bash
admixture-cache build \
    --panel-bed panel.bed \
    --panel-pop panel.pop \
    --clusters-yaml clusters.yaml \
    --k 21 \
    --cache-dir /tmp/cache_to_publish \
    --track regional \
    --panel-id aadr_v66_ho \
    --panel-version v66.0 \
    --seeds 1,2,3,4,5
```

Verify the cache loads and the manifest is well-formed:

```bash
admixture-cache verify \
    --panel-bed panel.bed \
    --clusters-yaml clusters.yaml \
    --k 21 \
    --cache-dir /tmp/cache_to_publish
```

### 2. Pack the tarball

The downloader auto-detects between "flat" (files at top level) and
"wrapped" (single subdir) layouts. **Wrapped is recommended** because
it lets users `tar -xzf cache.tar.gz` outside their cache root
without polluting the cwd:

```bash
NAME=regional_k21_aadr_v66_ho
cd /tmp
# `cache_to_publish/` will be the wrapper directory
cp -r cache_to_publish $NAME
tar -czf $NAME.tar.gz $NAME
sha256sum $NAME.tar.gz > $NAME.tar.gz.sha256
```

The `.sha256` file format is either bare hex digest on a single line
OR the GNU coreutils `sha256sum` format (`<digest>  <filename>`) —
the downloader accepts both.

### 3. Tag + create the release

The tag MUST match `cache-<name>-<version>`:

```bash
git tag cache-$NAME-v1
git push origin cache-$NAME-v1
```

Then create the GitHub Release with both assets:

```bash
gh release create cache-$NAME-v1 \
    --title "cache-$NAME-v1" \
    --notes-file release_notes.md \
    $NAME.tar.gz $NAME.tar.gz.sha256
```

The `release_notes.md` should include:

- Panel build provenance (sample count, SNP count, K value).
- Source dataset version (e.g. AADR v66 HO release date).
- Multimodality summary (per-cluster restart SD, threshold used).
- Wallclock + which hardware it was built on.
- Any caveats (e.g. "this cache was built against ADMIXTURE 1.4.0;
  Q values differ from 1.3.0 builds at ~1e-3 absolute").

### 4. Verify the published cache is discoverable

```bash
admixture-cache download --list
# Should show the new entry.

admixture-cache download $NAME --cache-root /tmp/test_install
# Should succeed and install the cache; verify by loading it.
admixture-cache verify \
    --panel-bed panel.bed --clusters-yaml clusters.yaml --k 21 \
    --cache-dir /tmp/test_install/$NAME
```

## What goes wrong

- **`No published cache named X`** — caller's `<name>` doesn't match
  any release. Check `--list`; misspellings + wrong-case names are
  the common cause.
- **`SHA-256 mismatch`** — the `<name>.tar.gz.sha256` file disagrees
  with the bytes of `<name>.tar.gz`. Re-pack and re-upload; this
  usually means the tarball was uploaded WITHOUT the sha256 being
  regenerated.
- **`cannot locate manifest.json in extracted tarball`** — the
  tarball doesn't have `manifest.json` at either the top level OR
  in a single wrapper dir. Common cause: packed with `tar -czf
  cache.tar.gz cache_to_publish/*` (the `/*` flattens but creates
  no wrapper). Re-pack with `tar -czf cache.tar.gz cache_to_publish`
  (no trailing slash, no glob).
- **`extracted cache failed manifest validation`** — `manifest.json`
  is malformed or doesn't satisfy the pydantic schema. Rebuild the
  cache from scratch and re-pack; this shouldn't happen with a cache
  produced by `admixture-cache build`.

## Discovery (consumer side)

The downloader queries `GET /repos/<owner>/<repo>/releases`
unauthenticated. Public repos work without a token. For
private-repo distributions, consumers need a `GITHUB_TOKEN` env var
honored by `urllib.request` — not currently supported; file an
issue if you need it.

## CI hook (optional)

If you have many caches to publish, automate via a GitHub Actions
workflow on tag push matching `cache-*`. Use `gh release create`
inside the workflow with `secrets.GITHUB_TOKEN`. Sketch:

```yaml
on:
  push:
    tags: ["cache-*"]
jobs:
  release:
    runs-on: ubuntu-latest
    permissions: { contents: write }
    steps:
      - uses: actions/checkout@v5
      - run: |
          # Build the cache (or fetch from artifact storage)
          # Pack tarball + sha256
          gh release create "$GITHUB_REF_NAME" \
              --notes-file notes.md \
              <name>.tar.gz <name>.tar.gz.sha256
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```
