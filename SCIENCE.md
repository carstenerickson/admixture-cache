# SCIENCE.md

Every place in `admixture-cache` where a decision is made that affects the
scientific output: the inferred per-target ancestry proportions Q, the cached
panel allele-frequency matrix P, or the validity and reproducibility of either.
For each decision this document records what the code does (with a file and line
reference), a verdict on whether the choice is methodologically sound, and the
supporting or contradicting literature.

## How to read this

Verdict labels:

- **Sound**: standard, well supported practice; no material concern.
- **Sound with caveats**: the core choice is correct and conventional, but it
  carries documented risks the code does not fully guard against.
- **Caution**: defensible but with a parameter or default that sits outside the
  usual published range, or that is only conditionally safe.
- **Flag**: a specific behavior that can silently corrupt the science under
  inputs this project is likely to receive. Flags appear as caveats inside
  otherwise sound sections; they are called out explicitly.

### How the literature was consulted

Each decision was researched against an ancestry-genetics literature corpus
(roughly 3,190 papers) using two layers: passage-level search for grounding
citations, and a grounded synthesis ("ask") layer for the central
methodological questions. Citations are real corpus papers with DOIs; none are
invented. The foundational sections D1 through D5 are grounded primarily through
the synthesis layer, because an infrastructure outage blocked the corpus during
the first automated pass over those points. Every verdict also rests on a direct
read of the source.

### Priority gaps (most actionable)

If you only act on a few items, these are the ones that can change a result
silently:

1. **No minimum overlapping-SNP floor** at projection (D10, D20): a sparse
   target can return a confident, converged-looking Q that is meaningless.
2. **Strand-ambiguous A/T and C/G SNPs** (D11): RESOLVED (Unreleased).
   Previously ID-based `--alt1-allele` forcing could leave these on the
   wrong strand, silently inverting allele counts. Now excluded by default
   at both build time (`strip_strand_ambiguous_snps` plus a build guard)
   and projection time (`plink2 --exclude`), with a `--keep-strand-ambiguous`
   opt-out.
3. **No pseudo-haploid path or detection** (D17): the dominant ancient-DNA
   genotype format violates the n=2 binomial assumption and is accepted
   silently.
4. **Restart default of 5** (D4): below the common 10 to 100 range for
   panels that contain unlabeled (free-Q) samples.
5. **LD-pruning window units** (D7): RESOLVED (Unreleased). The `window_kb`
   parameter mislabeled a plink2 `--indep-pairwise` variant-count window as kb.
   Renamed to `window_size` (documented as variants), `window_kb` kept as a
   deprecated alias (passing both raises `TypeError`), and the default raised
   from `50 5 0.5` to the field-standard `200 25 0.4`.

## Summary table

| ID | Decision | Location | Verdict |
|----|----------|----------|---------|
| D1 | Hold panel P fixed, solve only target Q (projection) | projection.py:37; orchestration.py:124 | Sound with caveats |
| D2 | Binomial / Hardy-Weinberg admixture likelihood | projection.py:95 | Sound with caveats |
| D3 | Supervised ADMIXTURE with fixed `.pop` labels | builder.py:802 | Sound with caveats |
| D4 | Number of random restarts (default 5) | builder.py:253 | Caution |
| D5 | Best-loglikelihood restart selected as canonical | builder.py:569 | Sound with caveats |
| D6 | Multimodality gate: max per-cell Q SD > 0.02 fails | builder.py:159, 596 | Sound with caveats |
| D7 | Optional LD pruning (`--indep-pairwise 200 25 0.4`) | builder.py:970 | Sound (units fixed, default raised) |
| D8 | Mean (not sum) per-SNP NLL objective | projection.py:90 | Sound |
| D9 | SLSQP simplex optimization, single 1/K start | projection.py:107 | Sound |
| D10 | Mask missing SNPs, score observed sites only | projection.py:80; alignment.py:263 | Sound with caveats |
| D11 | REF/ALT harmonization via `--alt1-allele` | alignment.py:114 | Sound (ambiguous-SNP gap fixed) |
| D12 | Restrict to target-intersect-panel SNP set | alignment.py:147 | Sound with caveats |
| D13 | Clip expected frequency to [1e-9, 1-1e-9] | projection.py:96 | Sound with caveats |
| D14 | Stock ADMIXTURE convergence, 24h per-restart timeout | builder.py:183 | Sound with caveats |
| D15 | Fully-labeled panel: near-closed-form P, seed-independent | builder.py:204 | Sound with caveats |
| D16 | Cache validity gated on input SHA-256 | manifest.py; io.py:59 | Sound |
| D17 | Diploid additive 0/1/2 dosage, no pseudo-haploid path | alignment.py:185 | Sound with caveats |
| D18 | K operator-specified, no CV diagnostic | builder.py:148 | Sound with caveats |
| D19 | Point-estimate Q only, no uncertainty | projection.py:107 | Sound with caveats |
| D20 | No MAF / QC inside projection | projection.py; alignment.py | Sound with caveats |
| D21 | Multimodality SD computed over all panel samples | builder.py:589 | Sound with caveats |
| D22 | SLSQP `ftol=1e-9` | projection.py:39, 116 | Sound |
| D23 | SLSQP `maxiter=200` | projection.py:39, 116 | Sound with caveats |
| D24 | float64 throughout | alignment.py:204; io.py:33 | Sound |

---

## D1. Hold the panel P fixed and solve only the target Q

**What we do.** P (M by K allele frequencies) is estimated once from the
reference panel by stock supervised ADMIXTURE and cached. For each new target P
is held fixed and only the K-vector Q is solved by maximum likelihood under the
binomial admixture model (`projection.py:37`, `orchestration.py:124-142`).

**Verdict: Sound with caveats.**

**Literature.** This "fix P, solve Q" step is exactly the supervised projection
mode that ADMIXTURE-family methods support, and the grounded synthesis confirms
it is statistically valid: supervised estimates track the true ancestry values
with minimal bias across a range of Fst, whereas unsupervised estimation is
upward biased at low ancestry fractions and downward biased at high ones,
especially at low Fst (Alexander and Lange, ADMIXTURE enhancements, 2011,
doi:10.1186/1471-2105-12-246). The projection estimator that holds a fixed set
of allele frequencies and solves each individual's coefficients is an
established design (Bansal and Libiger, 2015, doi:10.1186/s12859-014-0418-7).

**Caveats and flags.**

- **Asymmetric treatment / reference-panel bias.** Reference individuals
  contribute to P; the target does not. The target is therefore expressed
  strictly in the basis of panel-defined clusters, and a target whose true
  ancestry is absent from the panel is forced onto the nearest available
  combination with no in-likelihood signal that the panel is inadequate (North
  Pontic supplement, 2025, doi:10.1038/s41586-024-08372-2; reference-panel
  misspecification bias in Conomos et al., 2016, doi:10.1016/j.ajhg.2015.11.022).
  The code reports no per-target goodness-of-fit residual; adding one (final NLL
  per SNP, or binomial deviance) would let a poorly-modeled target be detected.
- **Small panel plus haploid data.** The synthesis surfaced a specific failure
  mode directly relevant to ancient DNA: small reference panels combined with
  haploid (pseudo-haploid) data produce severe nonlinear bias and step-function
  artifacts that push intermediate ancestry toward 0 or 1; diploid data shows
  minimal bias regardless of panel size (reply to Lazaridis and Reich, 2017,
  doi:10.1073/pnas.1704442114). See D17.
- **Low-overlap instability.** With few non-missing SNPs the likelihood is
  nearly flat and the solve is poorly determined. The code logs the observed
  count but enforces no minimum (see D10, D20).

## D2. Binomial / Hardy-Weinberg admixture likelihood

**What we do.** Target dosage g in {0,1,2} is modeled as Binomial(2, f) with
f = q^T P_s, and per-SNP log terms are summed, i.e. SNPs treated as independent
(`projection.py:95-99`).

**Verdict: Sound with caveats.**

**Literature.** This is the exact genotype likelihood underlying STRUCTURE and
ADMIXTURE. The synthesis confirms the shared generative model: individual-level
frequency p = sum_k q_k f_k, genotype drawn Binomial(2, p), with the log
likelihood summed over individuals and loci (Pritchard, Stephens and Donnelly,
2000, doi:10.1093/genetics/155.2.945; Alexander and Lange, 2011,
doi:10.1186/1471-2105-12-246; sNMF, Frichot and Francois, 2013,
arXiv:1309.6208). Holding P fixed and maximizing this likelihood over q is
precisely what stock `admixture --supervised` solves for a new sample, and the
code notes it matches stock supervised Q to about 1e-3.

**Caveats and flags.** The model's three standard assumptions are structural and
unguarded:

- **Hardy-Weinberg within clusters.** Binomial(2, f) encodes HWE proportions
  inside each ancestral cluster. If a cached cluster is not panmictic (cryptic
  substructure, a mixed group), genotype probabilities are misspecified and Q is
  biased. Model-fit work shows this assumption can and should be checked
  (evaluation of model fit, 2019, doi:10.1101/708883).
- **Linkage equilibrium.** The product over SNPs assumes independence; on dense
  AADR-style panels LD makes neighbouring SNPs correlated, overstating effective
  information. Standard practice is to LD-prune before fitting (see D7).
- **Additive 0/1/2 dosage.** Assumes no dominance and genuinely diploid calling
  (see D17).

## D3. Supervised ADMIXTURE with fixed `.pop` labels

**What we do.** The build runs `admixture --supervised`, pinning reference
individuals to predefined cluster labels from the `.pop` file; the K-column
order is recovered from the first-appearance order of non-`-` labels and the
build fails if the distinct-label count is not exactly K
(`builder.py:802-817`, `builder.py:891-916`).

**Verdict: Sound with caveats.**

**Literature.** Learning a fixed P from a curated, labeled panel is a legitimate
and common use of ADMIXTURE, and is the right choice given the downstream
projection holds P fixed. The synthesis documents the trade: supervised mode
reduces target bias but assumes the reference labels are correct and exhaustive
(Alexander and Lange, 2011, doi:10.1186/1471-2105-12-246).

**Caveats and flags.**

- **Forced assignment into predefined sources.** Every target Q lies in the
  simplex spanned by the K labeled clusters; a target outside that simplex is
  silently decomposed into the nearest mixture, with no residual diagnostic to
  flag it (North Pontic supplement, 2025, doi:10.1038/s41586-024-08372-2).
- **Inability to detect ghost / unmodeled ancestry.** Any ancestry absent from
  the reference set has no column to load onto. This is the central weakness of
  fixed-source methods and is not mitigated anywhere in the pipeline.
- **Sensitivity to panel composition and labeling.** The result is conditioned
  on which populations are chosen and how individuals are labeled; the
  count==K check cannot validate that clusters are genuinely distinct or that
  within-label samples are homogeneous. Small or poorly-composed panels produce
  systematically biased assignments (reply to Lazaridis and Reich, 2017,
  doi:10.1073/pnas.1704442114; Pemberton et al., 2013,
  doi:10.3389/fgene.2012.00322). The `panel_pop_sha256` manifest guard (D16)
  is the right mitigation against silent relabeling.
- **Over-interpretation risk.** Supervised bar plots are easy to read as ground
  truth; an admixture model is almost always wrong and should be checked for
  lack of fit (Lawson, van Dorp and Falush, 2018,
  doi:10.1038/s41467-018-05257-7).

## D4. Number of random restarts (default 5)

**What we do.** The build runs N random-seed restarts of supervised ADMIXTURE
(default `seeds=[1,2,3,4,5]`) and keeps the best (`builder.py:253-254`).

**Verdict: Caution.**

**Literature.** Multiple restarts to escape local optima of a multimodal
likelihood is the correct mitigation in principle (fastSTRUCTURE, Raj et al.,
2013, doi:10.1101/001073). The synthesis, however, finds typical practice runs
10 to 100 replicates, with 50 to 100 most common when formal multimodality
detection is the goal, and replicate Q matrices are aligned and summarized with
tools such as pong and CLUMPP rather than a single best run (pong, Behr et al.,
2016, doi:10.1093/bioinformatics/btw327; CLUMPP usage in Jakobsson et al.,
2008, doi:10.1038/nature06742). ADMIXTURE's `-B` flag provides bootstrap
standard errors.

**Caveats and flags.**

- **5 is on the low end.** Whether 5 restarts suffice is panel- and K-dependent;
  for larger K and more-structured panels it may under-sample the optimum basin.
  There is no K-aware scaling and no adaptive top-up when seeds disagree.
- **Important mitigation.** For a fully-labeled panel the restarts are
  byte-identical and seed-independent (D15), so 5 (or even 1) is harmless there.
  The concern bites only when the panel contains unlabeled `-` rows so Q is free.
- Seeds are fixed `[1..5]`, which is good for reproducibility but always samples
  the same five basins.

## D5. Best-loglikelihood restart selected as canonical

**What we do.** Among restarts with a parseable loglikelihood, the single
highest-LL restart's P and Q are cached (`builder.py:569-581`).

**Verdict: Sound with caveats.**

**Literature.** Choosing the maximum-likelihood replicate is the conventional
way to handle local optima in ADMIXTURE and STRUCTURE; some studies additionally
require the top runs to agree within a small LL margin (Di Cristofaro et al.,
2013, doi:10.1371/journal.pone.0076748). The foundational optimizer is
non-convex when P and Q are estimated jointly, which is why multi-restart
selection exists (ADMIXTURE, Alexander, Novembre and Lange, 2009,
doi:10.1101/gr.094052.109).

**Caveats and flags.**

- **Best-of-N is not global.** With few restarts on a hard surface, all may share
  a non-global optimum and the LL spread gives no signal it was missed. The
  multimodality gate (D6) catches disagreement, not consistent agreement on a
  wrong optimum.
- **No replicate alignment.** Cross-restart SD is computed on raw stacked Q with
  no CLUMPP-style alignment. This is safe here only because supervised mode
  anchors columns to the `.pop` label order; if reused for unsupervised
  ADMIXTURE, label switching would invalidate both the unaligned SD and the
  column-wise best-LL copy (CLUMPP, in Jakobsson et al., 2008,
  doi:10.1038/nature06742).
- **No explicit convergence check before LL comparison**; only `None` LLs are
  filtered (see D14).

## D6. Multimodality gate: max per-cell Q SD over restarts > 0.02 fails

**What we do.** Per-(sample, cluster) SD of Q across restarts is computed; if the
maximum exceeds 0.02 the build fails and no manifest is written
(`builder.py:159`, `builder.py:583-602`).

**Verdict: Sound with caveats.**

**Literature.** Checking that restarts converge to the same solution before
trusting P is sound and matches the documented non-convexity of these models
(fastSTRUCTURE, 2013, doi:10.1101/001073). But the field quantifies replicate
concordance with alignment-plus-similarity metrics, not a raw per-cell SD:
CLUMPP symmetric similarity coefficient above 0.9 (Verdu et al., 2014,
doi:10.1371/journal.pgen.1004530; Wang et al., 2007,
doi:10.1371/journal.pgen.0030185), pong average pairwise similarity (Behr et
al., 2016, doi:10.1093/bioinformatics/btw327), or sNMF RMSE and squared
correlation between Q matrices (Frichot and Francois, 2013, arXiv:1309.6208).

**Caveats and flags.**

- **The 0.02 absolute threshold has no literature support.** Recognized criteria
  are scale-free similarity or correlation cutoffs that adapt to K, sample size,
  and structure; a fixed per-cell SD does not, so 0.02 reads as arbitrary. For
  reference, raw ancestry-proportion SDs of 0.08 to 0.15 are reported elsewhere
  (Tucker et al., 2014, doi:10.1371/journal.pgen.1004445), making 0.02 strict.
- **Max-over-cells is fragile.** It is dominated by a single worst
  sample-cluster, so one unstable individual can fail an otherwise concordant
  cache, the opposite of whole-run similarity metrics. With only 2 to 5 restarts
  the SD estimate (ddof=1) is itself noisy.
- **Stability is necessary, not sufficient.** A converged-yet-biased P passes
  the gate; replicate agreement does not prove correctness (Lawson, van Dorp and
  Falush, 2018, doi:10.1038/s41467-018-05257-7). The only corpus precedent for
  an SD-over-replicates statistic applies it to a posterior-predictive
  discrepancy, not raw Q with a hard cutoff (Mimno et al., 2015,
  doi:10.1073/pnas.1412301112).

## D7. Optional LD pruning before training (`--indep-pairwise 200 25 0.4`)

**What we do.** An optional helper LD-prunes the panel via plink2
`--indep-pairwise` with window 200, step 25, r-squared 0.4
(`ld_prune_panel`, `builder.py`). The window is a variant count, not kb.

**Verdict: Sound (units clarified and default raised to match the field
standard; Unreleased).** Originally `50 5 0.5` with a `window_kb` parameter
that mislabeled the variant-count window as kb.

**Literature.** The unlinked-marker assumption is fundamental to ADMIXTURE, and
pruning before ADMIXTURE or PCA is standard practice; the ADMIXTURE paper itself
notes dense marker sets should be pruned to mitigate background LD (Alexander,
Novembre and Lange, 2009, doi:10.1101/gr.094052.109). The synthesis records
r-squared thresholds of 0.1 to 0.5 in common use. It also adds nuance: LD-aware
methods (linked fineSTRUCTURE) can resolve structure that pruning-plus-ADMIXTURE
misses, so pruning discards real information, and direct pruned-versus-unpruned
ADMIXTURE accuracy benchmarks are scarce (the justification is theoretical).

**Caveats and flags.**

- **Window units (RESOLVED, Unreleased).** The original code passed
  `str(window_kb)` = "50" to plink2 with no `kb` suffix, so plink2 read it as a
  50-variant window, not the 50 kb the parameter name and docstring implied
  (confirmed against plink2 v2.0.0: a `kb` window also requires step 1, which the
  default step of 5 violated, so the "kb" reading was never even valid). The
  parameter is now `window_size`, documented as variants; `window_kb` remains a
  deprecated alias, and passing both raises `TypeError`.
- **r-squared default raised to 0.4 (RESOLVED, Unreleased).** The old `0.5` sat
  at the lenient end of the published range; the default is now `0.4`, the modal
  value in the human ancient-DNA ADMIXTURE literature (window 200, step 25). A
  corpus methods survey found variant-count windows used over kb roughly 17:1
  for this use case, with `200 25 0.4` the most common single recipe.
- **Pruning removes only background LD, not admixture LD**, which can span tens
  of megabases in recently admixed panels, so the unlinked assumption is only
  partially satisfied. Information loss is real and is baked permanently into the
  cached P, because P is then frozen for all targets (Lawson, van Dorp and
  Falush, 2018, doi:10.1038/s41467-018-05257-7). Pruning can also reduce
  between-population Fst and blur closely related clusters.
- The "3 to 5x speedup, 30 to 50% of variants" claims are project measurements,
  not established literature results; keep the assumption justification separate
  from the engineering claim.

## D8. Mean (not sum) per-SNP negative loglikelihood objective

**What we do.** The SLSQP objective is the mean per-SNP NLL (scaled by 1/M)
rather than the sum, to keep the gradient O(1) so the tolerance behaves
identically at 100 versus 1.1M SNPs (`projection.py:90-118`).

**Verdict: Sound.**

**Literature.** This is a pure numerical-conditioning choice: scaling the
objective and gradient by a positive constant cannot move the argmax, so the
statistical estimand is unchanged. sNMF writes the identical binomial likelihood
and notes it holds up to a constant that does not influence estimation (Frichot
and Francois, 2013, arXiv:1309.6208). The fixed-P Q-subproblem is concave and
has a unique optimum (ADMIXTURE, 2009, doi:10.1101/gr.094052.109; convergence to
the same solution regardless of initialization in Bansal and Libiger, 2015,
doi:10.1186/s12859-014-0418-7), so the reported summed-form stall at a wrong
corner is a conditioning artifact, not multimodality. SLSQP ill-conditioning of
the constrained QP subproblem is a recognized concern (SciPy 1.0, 2020,
doi:10.1038/s41592-019-0686-2).

**Caveats and flags.**

- The exact reported failure mode (summed objective returns `success=True` at a
  wrong vertex) is grounded only in the project's own test; it is theoretically
  plausible but not independently corroborated in the corpus.
- Mean-scaling fixes magnitude, not the geometric degeneracy of near-collinear
  (low-Fst) panel columns, which can still leave Q ill-determined (Alexander and
  Lange, 2011, doi:10.1186/1471-2105-12-246).

## D9. SLSQP simplex optimization, single 1/K start

**What we do.** Q is constrained to the probability simplex (sum to 1,
nonnegativity, box bounds) and optimized by SLSQP from the uniform start 1/K,
with P fixed (`projection.py:107-118`).

**Verdict: Sound.**

**Literature.** Simplex constraints are the standard parameterization used by
ADMIXTURE, STRUCTURE, and sNMF (STRUCTURE, 2000, doi:10.1093/genetics/155.2.945;
sNMF, 2013, arXiv:1309.6208). The single uniform start is justified because the
fixed-P Q-update is a convex subproblem with a unique optimum (ADMIXTURE, 2009;
OpenADMIXTURE describes it as a simplified convex problem, 2023,
doi:10.1016/j.ajhg.2022.12.008), so the multi-restart need that applies to joint
P and Q estimation does not apply here.

**Caveats and flags.**

- The convexity that makes a single start safe should be stated explicitly in
  the code, since it is the load-bearing reason no multi-start is used.
- Unregularized boundary solutions (q_k exactly 0 or 1) are expected MLE
  behavior and provide no shrinkage; tiny components should not be
  over-interpreted at low SNP counts (Alexander and Lange, 2011,
  doi:10.1186/1471-2105-12-246).
- `result.success` means the KKT tolerance was met, not that the answer was
  validated (see D8).

## D10. Mask missing SNPs, score observed sites only

**What we do.** Panel SNPs absent from the target are NaN-filled by
`reindex_dosage_to_panel_order` and masked out; the likelihood is evaluated only
over observed sites. The only guard is a non-empty mask
(`projection.py:80-88`, `alignment.py:263-312`).

**Verdict: Sound with caveats.**

**Literature.** Per-individual exclusion of missing sites under
missing-at-random is the standard, defensible behavior for likelihood-based
ancestry methods, and the supervised fixed-P design is the variant the
literature flags as least biased (NGSadmix, Skotte et al., 2013,
doi:10.1534/genetics.113.154138; the synthesis confirms likelihood-over-observed
is standard).

**Caveats and flags.**

- **No minimum-overlap floor (Flag).** The code raises only at zero observed
  SNPs. Ancestry proportions are documented as highly unstable below roughly
  10,000 to 15,000 SNPs (Flegontov et al., 2020,
  doi:10.1101/2020.01.06.885103), and real aDNA pipelines hard-code floors of
  20,000 (Sirak et al., 2021, doi:10.1038/s41467-021-27356-8) and higher. A
  target overlapping a few hundred SNPs returns a confident, wrong Q. Recommend
  a configurable floor (order 10k to 20k) that warns or refuses.
- **Missing-at-random is violated by aDNA capture.** Which SNPs are observed is
  correlated with the assay (1240K poorly enriches about 28% of its own SNPs:
  Rohland et al., 2022, doi:10.1101/gr.276728.122) and with allele identity
  (allelic bias at about 62% of 1240K SNPs: Davidson et al., 2023,
  doi:10.1101/2023.07.04.547445). Mapping bias alone shifts proportions by up to
  about 4% (Nielsen et al., 2024, doi:10.1101/2024.07.01.601500). The observed
  subset is then a biased subsample.
- **Panel-versus-target assay mismatch is unguarded** (genotype array versus
  1240K versus shotgun); batch effects inflate genetic similarity (Davidson et
  al., 2023).

## D11. REF/ALT harmonization via `--alt1-allele`

**What we do.** Before dosage extraction, plink2 is run with `--extract
panel.bim` and `--alt1-allele panel.bim 5 2`, forcing each overlapping SNP's
ALT1 to match the panel by variant ID (`alignment.py:114-162`).

**Verdict: Sound with caveats.**

**Literature.** REF/ALT and strand harmonization before any
allele-frequency-based ancestry estimation is mandatory: a swap silently inverts
the allele count and produces a wrong Q under the binomial likelihood (because P
is fixed on the panel axis). The synthesis confirms variant matching, strand
alignment, and removal of ambiguous alleles are prerequisites (best-practices
PCA toolkit, 2020, doi:10.1093/bioinformatics/btaa520; MESA, 2012,
doi:10.1371/journal.pgen.1002640).

**Status: the strand-ambiguous gap below is FIXED (Unreleased).** Empirically
confirmed against plink2 v2.0.0 that an opposite-strand A/T or C/G target was
silently inverted (homozygotes flip 0 to 2) while a non-ambiguous SNP was not.
The fix excludes strand-ambiguous SNPs by default at build time
(`strip_strand_ambiguous_snps` plus a `build_panel_cache` guard) and at
projection time (`plink2 --exclude`), with a `--keep-strand-ambiguous` opt-out.

**Caveats and flags.**

- **Strand-ambiguous A/T and C/G SNPs (FIXED).** ID-based ALT1 forcing matches
  by variant ID and can succeed while leaving an A/T or C/G SNP on the wrong
  strand, silently inverting its allele count. The corpus consensus is to
  exclude these SNPs entirely, not to ID-force them (MESA, 2012; PGG.
  Population, 2017, doi:10.1093/nar/gkx1032; Calabrian Greeks, 2021,
  doi:10.1038/s41598-021-82591-9). This is now what the code does: such SNPs are
  dropped from the panel at build time and from each projection by default.
- **No allele-pair validation.** Forcing the ALT column does not verify the
  target's allele pair equals the panel's pair; a shared rsID whose alleles match
  neither panel allele can pass through mis-coded (the mistyped class filtered in
  the Saudi AADR merge, 2025, doi:10.1101/2025.01.10.632500).
- **Silent partial application.** plink2 may only warn when a requested allele is
  absent; the code validates the output triplet exists but does not record how
  many SNPs were realigned versus skipped. Parsing the plink2 log would close
  this.
- **aDNA compounding.** Deamination concentrates errors at C/T and G/A
  transitions, worsening the ambiguous-SNP problem and arguing for removal over
  ID-forcing on low-coverage targets.

## D12. Restrict to target-intersect-panel SNP set

**What we do.** Only variants in `panel.bim` are kept (`--extract`);
target-private SNPs are dropped, panel SNPs missing from the target become NaN,
and the usable marker set is fixed to the cached panel (`alignment.py:147-162`).

**Verdict: Sound with caveats.**

**Literature.** Intersecting a sample to a fixed reference SNP set is the
universal standard in human aDNA work (the AADR 1240K and Human Origins merges),
so the mechanics are idiomatic (Bansal and Libiger, 2015,
doi:10.1186/s12859-014-0418-7).

**Caveats and flags.**

- **Closed-world ancestry.** Fixing P and intersecting to `panel.bim` forces a
  target with out-of-panel ancestry onto the available clusters, with no residual
  diagnostic (North Pontic supplement, 2025, doi:10.1038/s41586-024-08372-2).
- **Cross-target non-comparability.** Each target keeps a different NaN subset, so
  Q vectors are not estimated on a common marker set; the code logs placement
  counts but enforces no floor or warning (see D10).
- **Panel-inherited ascertainment bias.** P and the marker set live entirely
  within the ascertained panel, so SNP-chip ascertainment skew is baked in
  (Albrechtsen et al., 2010, doi:10.1093/molbev/msq148; array-design bias, 2021,
  doi:10.1371/journal.pone.0245178). Affinities to discovery populations may be
  inflated and rare target variation is invisible.

## D13. Clip expected frequency to [1e-9, 1-1e-9]

**What we do.** The expected admixed frequency f = q^T P is clipped to
[1e-9, 1-1e-9] before taking log(f) and log(1-f) (`projection.py:96, 102`).

**Verdict: Sound with caveats.**

**Literature.** Clamping probabilities away from 0 and 1 to operate in log-space
is a standard numerical safeguard, and the closest comparable method (a binomial
NLL minimized with SciPy SLSQP) does exactly this (Mathieson et al., 2022,
doi:10.1101/2022.08.24.505188). The statistical bias of a 1e-9 clip is
negligible because it activates only at degenerate boundary cases.

**Caveats and flags.**

- **1e-9 is far tighter than documented bounds.** The analogous method bounds
  frequency at 0.01 and 0.99. At 1e-9 the score can reach about 1e9, producing
  badly-scaled gradients that the mean-NLL normalization only partly tames,
  exactly the conditioning that the docstring warns can stall SLSQP. A clip near
  1e-6 to 1e-4 would be more standard.
- **It masks rather than prevents a degenerate panel.** If a P column is exactly
  0 or 1 at a SNP (a monomorphic cluster), the clip converts a legitimately
  near-infinite penalty into a finite term, which can let an implausible Q
  survive. The field removes monomorphic SNPs upstream; the projection path does
  not. Recommend filtering monomorphic panel SNPs at cache-build time.

## D14. Stock convergence and 24h per-restart timeout

**What we do.** Each restart uses stock ADMIXTURE convergence (delta-LL < 1e-4
per the docstring) capped at 24h; a timed-out or non-converged restart
contributes no usable LL (`builder.py:178-183`).

**Verdict: Sound with caveats.**

**Literature.** Relying on the stock tolerance is the safe direction: 1e-4 is
the deliberately strict ADMIXTURE default (Alexander, Novembre and Lange, 2009,
doi:10.1101/gr.094052.109; Alexander and Lange, 2011,
doi:10.1186/1471-2105-12-246).

**Caveats and flags.**

- **Per-run tolerance is not global-optimum convergence.** The field judges
  convergence by agreement of the best LL across many seeds (commonly 10 to 100),
  not the program's internal delta (Raghavan et al., 2015,
  doi:10.1126/science.aab3884). A restart can satisfy delta < 1e-4 at a local
  optimum and still be selected (see D4, D5).
- **Timeout creates a silent selection bias.** If the slowest seeds on a large
  high-K panel are the ones still climbing toward the true optimum, dropping them
  biases the cached P toward faster, shallower optima. Large high-K ADMIXTURE
  often fails to converge even in 100 runs (fast admixture inference, 2022,
  doi:10.1038/s41437-022-00535-z). Recommend recording per-restart convergence
  status and the spread of best LLs in the manifest.
- The specific 86400s value has no literature basis; the field terminates on
  LL/iteration criteria, not wall-clock.

## D15. Fully-labeled panel: near-closed-form P, seed-independent

**What we do.** When every panel sample is labeled, supervised ADMIXTURE has no
free Q; P reduces to a per-cluster allele-frequency pass converging in about one
iteration, identical across seeds, so the multimodality check is structurally
vacuous (`builder.py:204-220`).

**Verdict: Sound with caveats.**

**Literature.** With all reference individuals labeled, the supervised estimate
of P is effectively the per-cluster sample allele frequencies, which is the
correct and expected behavior (consistent with the fixed-allele-frequency
projection design in Bansal and Libiger, 2015,
doi:10.1186/s12859-014-0418-7).

**Caveats and flags.**

- **No prior or shrinkage on P.** The cached P is unsmoothed within-cluster
  frequencies; small-cluster sampling error then biases every fixed-P projection.
  Joint estimation would regularize this; the closed-form pass does not.
- Boundary frequencies can destabilize the Q-solve (see D13), and within-source
  heterogeneity is forced to zero.
- The multimodality gate (D6) certifies nothing in this regime; a quick build is
  expected, not a sign of a short-circuit.

## D16. Cache validity gated on input SHA-256

**What we do.** A cache is reused only if the SHA-256 of every declared
scientific input (panel `.bim`, panel `.pop`, clusters YAML, K, geo-filter
YAMLs) matches the manifest; any divergence forces a rebuild
(`manifest.py`, `io.py:59-145`).

**Verdict: Sound.**

**Literature.** Content-addressed invalidation operationalizes recognized
reproducibility norms: pinning input versions and binding an output to the exact
inputs that produced it (AADR, 2023, doi:10.1038/s41597-024-03031-7;
nf-core/eager, 2021, doi:10.7717/peerj.10947).

**Caveats and flags.**

- **Software provenance is recorded but not gated.** `admixture_version` and
  `pgen_samplebind_version` are stored but not compared, so a cache built with a
  different ADMIXTURE or plink2 build is silently reused; reviewers note
  different versions can produce different output (Jomon analysis, 2020,
  doi:10.1038/s42003-020-01162-2). Consider comparing solver versions.
- **The panel.pop SHA guard is lenient on `None`** for back-compat, so a
  supervised-label edit can go undetected for pre-field caches (the input whose
  change most directly alters P; see D3).
- The restart budget is load-bearing (ADMIXTURE is multimodal) but is recorded
  for provenance only, not part of the validity key.

## D17. Diploid additive 0/1/2 dosage, no pseudo-haploid path

**What we do.** The target is read as plink2 additive 0/1/2 dosages and fed to a
Binomial(g; 2, f) likelihood that hard-codes n=2 (`alignment.py:185-241`,
`projection.py:95-99`). There is no pseudo-haploid path.

**Verdict: Sound with caveats.**

**Literature.** For genuinely diploid, well-covered inputs (arrays,
high-coverage WGS, imputed calls) this is the standard, correct model.

**Caveats and flags.**

- **No pseudo-haploid path or detection (Flag).** The dominant representation for
  low-coverage and ancient DNA is pseudo-haploid (one sampled read, coded as
  homozygous). Treating it as diploid violates the n=2 assumption and drives
  nonlinear ADMIXTURE bias, especially with small or structured reference
  clusters (reply to Lazaridis and Reich, 2017, doi:10.1073/pnas.1704442114). A
  pseudo-haploid sample has near-zero heterozygous sites, which the code could
  cheaply detect and warn on, but does not.
- **No reference-bias guard.** Pseudo-haploid aDNA is skewed toward the reference
  allele, pulling Q toward the reference-closest cluster; mapping bias alone
  shifts proportions by up to about 4% (reference bias, 2019,
  doi:10.1371/journal.pgen.1008302; Nielsen et al., 2024,
  doi:10.1101/2024.07.01.601500).
- **No genotype-likelihood path.** The preferred route for true low-coverage
  data is genotype-likelihood admixture (NGSadmix / fastNGSadmix), which uses
  read counts and uncertainty instead of hard calls (Skotte et al., 2013,
  doi:10.1534/genetics.113.154138). The docstring should state that input must be
  genuinely diploid, since AADR users routinely have pseudo-haploid data.

## D18. K operator-specified, no CV diagnostic

**What we do.** K is supplied by the operator (pinned to the labeled reference
cluster count); the tool computes no cross-validation error or K model-selection
diagnostic (`builder.py:148`).

**Verdict: Sound with caveats.**

**Literature.** In supervised projection, K is not a free knob: it is the number
of labeled sources, so the operator is choosing a panel design rather than
estimating an unknown K, which matches published supervised pipelines. The
literature documents the risks of a mis-specified K and that CV itself is an
unreliable K estimator (Evanno et al., 2005,
doi:10.1111/j.1365-294x.2005.02553.x; SHIPS, 2012,
doi:10.1371/journal.pone.0045685; Wang parsimony estimator, 2019,
doi:10.1111/1755-0998.13000).

**Caveats and flags.**

- **No diagnostic against alternative K.** Nothing signals that two reference
  clusters are not genetically distinct (redundant K) or that a cluster should
  split; CV-based estimators fail silently under low differentiation (SHIPS,
  2012). A mis-curated panel yields confident-looking Q with no flag (Lawson, van
  Dorp and Falush, 2018, doi:10.1038/s41467-018-05257-7).
- A too-small K (missing a real source) cannot be detected at projection time and
  produces a plausible but wrong ancestry vector (North Pontic supplement, 2025,
  doi:10.1038/s41586-024-08372-2).

## D19. Point-estimate Q only, no uncertainty

**What we do.** `project_target` returns a single SLSQP point estimate plus
convergence flags; no SE, bootstrap CI, or posterior (`projection.py:107-121`).

**Verdict: Sound with caveats.**

**Literature.** The point estimate is standard, but ADMIXTURE, STRUCTURE, and
qpAdm each ship a companion uncertainty estimate, and reporting bare proportions
with no precision is treated as a real shortcoming. It is acute here because
targets are single, often low-coverage aDNA samples where intervals are widest
(ADMIXTURE bootstrap `-B`, 2009, doi:10.1101/gr.094052.109; qpAdm performance,
2020, doi:10.1101/2020.04.09.032664).

**Caveats and flags.**

- A consumer cannot distinguish a confident component from a barely identified
  one. `panel_stability_max_sd` is a build-time restart metric for P, not
  per-target uncertainty for Q.
- A naive per-SNP bootstrap would understate SEs because of LD; a block or
  LD-pruned bootstrap is the principled choice (ADMIXTURE, 2009). Holding P fixed
  also means any target SE ignores uncertainty in P.
- This is a reporting-completeness gap, not a correctness bug.

## D20. No MAF / QC inside projection

**What we do.** Projection applies no MAF filter, no per-SNP or per-sample
missingness QC, and no contamination, relatedness, or batch check; all QC is
delegated to the panel build and upstream pipeline (`projection.py`,
`alignment.py`).

**Verdict: Sound with caveats.**

**Literature.** Delegating per-SNP QC to the cache build is largely necessary,
because P and the scored SNP set are frozen at build time, so the panel SNPs are
an already-curated set. The genuine gap is target-side QC.

**Caveats and flags.**

- **No minimum overlapping-SNP floor (Flag).** Same gap as D10: ancestry is
  unstable below roughly 10,000 to 15,000 SNPs (Flegontov et al., 2020,
  doi:10.1101/2020.01.06.885103). Recommend gating on `n_snps_used`.
- **No contamination flag for ancient targets.** Standard aDNA practice excludes
  samples above about 5% contamination; contamination corrupts the target dosage
  and biases Q toward the contaminant, and the code cannot detect it. If QC is
  delegated upstream, that assumption should be documented and ideally enforced
  as a required input flag.
- **No platform/batch guard.** Co-analyzing a target assayed on a different
  platform than the panel (capture versus shotgun versus array) is documented as
  unreliable and is normally handled with bias-corrected panels or batch-effect
  SNP removal. Aligning REF/ALT and intersecting SNPs does not ensure assay
  compatibility.
- Relatedness/duplicate QC is the lowest-risk omission here: with fixed P and
  single-target projection, a related target cannot distort clusters as it would
  in a joint run.

## D21. Multimodality SD computed over all panel samples

**What we do.** The multimodality gate computes per-cell Q SD across all panel
samples, including both labeled anchors and any unlabeled clinal samples, then
takes the global max (`builder.py:589-590, 632`).

**Verdict: Sound with caveats.**

**Why it matters.** This sets the denominator for the D6 gate. Labeled anchors
have Q pinned by their labels, so their cross-seed SD is essentially zero;
including them can only dilute the max-SD statistic, never inflate it, so the
gate effectively measures the unlabeled (free-Q) subset, which is the right
subset to monitor. The caveat is that this coupling is implicit: if a future
change made anchor Q non-trivial, the all-samples denominator would mask
instability. The choice should be documented. See D6 for the threshold critique.

## D22. SLSQP `ftol=1e-9`

**What we do.** The SLSQP function-value tolerance is `ftol=1e-9`
(`projection.py:39, 116`).

**Verdict: Sound.**

**Why it matters.** `ftol` sets when the optimizer stops and thus the precision
of Q. Because the objective is the mean (not sum) NLL (D8), `ftol` is
scale-independent across panel sizes, so 1e-9 behaves the same at 100 and 1.1M
SNPs. The cross-check against stock `admixture --supervised` (about 1e-3
agreement) confirms 1e-9 does not cause spurious non-convergence; keep that check
as the guard if the objective scaling is ever changed.

## D23. SLSQP `maxiter=200`

**What we do.** The SLSQP iteration cap is `maxiter=200` (`projection.py:39,
116`).

**Verdict: Sound with caveats.**

**Why it matters.** If the cap is hit before convergence, Q is left at a
suboptimal point and `result.success` is False. The docstring reports
convergence in about 10 to 15 iterations on real panels, so 200 is generously
above the working regime. The caveat is that `converged` is surfaced in
`ProjectionResult` but the orchestration path does not refuse a non-converged Q;
a caller should check the flag, and a hard refuse-or-warn on
`converged=False` would be safer.

## D24. float64 throughout

**What we do.** Dosage, P, and Q are stored and computed in float64 (dosage cast
explicitly, P via `np.loadtxt`, Q via numpy default) (`alignment.py:204, 241,
297`, `io.py:33`).

**Verdict: Sound.**

**Why it matters.** Double precision is the correct default for accumulating the
binomial objective and gradient over up to about 1.1M SNPs; float32 would
accumulate rounding error in the sum and could perturb Q near the boundary. No
concern; recorded for completeness because the precision choice is a (benign)
decision affecting the numerical output.

---

## References

Corpus papers cited above, by DOI. Years are publication or preprint years as
recorded in the corpus.

- Pritchard, Stephens and Donnelly, 2000. Inference of Population Structure
  Using Multilocus Genotype Data (STRUCTURE). doi:10.1093/genetics/155.2.945
- Evanno, Regnaut and Goudet, 2005. Detecting the number of clusters of
  individuals using STRUCTURE. doi:10.1111/j.1365-294x.2005.02553.x
- Wang et al., 2007. Genetic Variation and Population Structure in Native
  Americans. doi:10.1371/journal.pgen.0030185
- Jakobsson et al., 2008. Genotype, haplotype and copy-number variation in
  worldwide human populations (carries the CLUMPP citation).
  doi:10.1038/nature06742
- Alexander, Novembre and Lange, 2009. Fast model-based estimation of ancestry
  in unrelated individuals (ADMIXTURE). doi:10.1101/gr.094052.109
- Albrechtsen, Nielsen and Nielsen, 2010. Ascertainment biases in SNP chips
  affect measures of population divergence. doi:10.1093/molbev/msq148
- Alexander and Lange, 2011. Enhancements to the ADMIXTURE algorithm for
  individual ancestry estimation. doi:10.1186/1471-2105-12-246
- Conomos et al., 2012. Population Structure of Hispanics in the United States
  (MESA). doi:10.1371/journal.pgen.1002640
- SHIPS, 2012. Spectral Hierarchical clustering for the Inference of Population
  Structure. doi:10.1371/journal.pone.0045685
- Di Cristofaro et al., 2013. Afghan Hindu Kush: Where Eurasian Sub-Continent
  Gene Flows Converge. doi:10.1371/journal.pone.0076748
- Pemberton et al., 2013. A Method for Inferring an Individual's Genetic Ancestry
  and Degree of Admixture. doi:10.3389/fgene.2012.00322
- Raj, Stephens and Pritchard, 2013. fastSTRUCTURE: Variational Inference of
  Population Structure. doi:10.1101/001073
- Frichot and Francois, 2013. Fast Inference of Admixture Coefficients Using
  Sparse Non-negative Matrix Factorization (sNMF). arXiv:1309.6208
- Skotte, Korneliussen and Albrechtsen, 2013. Estimating Individual Admixture
  Proportions from Next Generation Sequencing Data (NGSadmix).
  doi:10.1534/genetics.113.154138
- Verdu et al., 2014. Patterns of Admixture and Population Structure in Native
  Populations. doi:10.1371/journal.pgen.1004530
- Tucker et al., 2014. Comparison of Methods to Account for Relatedness in GWAS.
  doi:10.1371/journal.pgen.1004445
- Bansal and Libiger, 2015. Fast individual ancestry inference from DNA sequence
  data leveraging allele frequencies. doi:10.1186/s12859-014-0418-7
- Raghavan et al., 2015. Genomic evidence for the Pleistocene and recent
  population history of Native Americans. doi:10.1126/science.aab3884
- Mimno, Blei and Engelhardt, 2015. Posterior predictive checks to quantify
  lack of fit in admixture models. doi:10.1073/pnas.1412301112
- Behr et al., 2016. pong: fast analysis and visualization of latent clusters.
  doi:10.1093/bioinformatics/btw327
- Conomos, Reiner, Weir and Thornton, 2016. Model-free Estimation of Recent
  Genetic Relatedness. doi:10.1016/j.ajhg.2015.11.022
- Lu et al., 2017. PGG.Population: a database for genomic diversity.
  doi:10.1093/nar/gkx1032
- Sikora et al. (reply to Lazaridis and Reich), 2017. Robust model-based
  inference of male-biased admixture. doi:10.1073/pnas.1704442114
- Günther and Nettelblad, 2019. The presence and impact of reference bias on
  population genomic studies. doi:10.1371/journal.pgen.1008302
- Evaluation of Model Fit of Inferred Admixture Proportions, 2019.
  doi:10.1101/708883
- Wang, 2019. A parsimony estimator of the number of populations from a
  STRUCTURE-like analysis. doi:10.1111/1755-0998.13000
- Flegontov et al., 2020. Diverse genetic origins of medieval steppe nomad
  conquerors. doi:10.1101/2020.01.06.885103
- Privé et al., 2020. Efficient toolkit implementing best practices for PCA of
  population genetic data. doi:10.1093/bioinformatics/btaa520
- Virtanen et al., 2020. SciPy 1.0: fundamental algorithms for scientific
  computing in Python. doi:10.1038/s41592-019-0686-2
- Assessing the Performance of qpAdm, 2020. doi:10.1101/2020.04.09.032664
- Model-based genotype and ancestry estimation for potential hybrids, 2020.
  doi:10.1101/2020.07.31.231514
- Sirak et al., 2021. Social stratification without genetic differentiation at
  Kulubnarti. doi:10.1038/s41467-021-27356-8
- array-design SNP ascertainment bias, 2021. doi:10.1371/journal.pone.0245178
- Calabrian Greeks genetic history, 2021. doi:10.1038/s41598-021-82591-9
- Rohland et al., 2022. Three assays for in-solution enrichment of ancient human
  DNA. doi:10.1101/gr.276728.122
- Mathieson et al., 2022. 1,000 ancient genomes uncover 10,000 years of natural
  selection in Europe. doi:10.1101/2022.08.24.505188
- Fast and accurate population admixture inference, 2022.
  doi:10.1038/s41437-022-00535-z
- Lawson, van Dorp and Falush, 2018. A tutorial on how not to over-interpret
  STRUCTURE and ADMIXTURE bar plots. doi:10.1038/s41467-018-05257-7
- OpenADMIXTURE (unsupervised discovery of ancestry-informative markers), 2023.
  doi:10.1016/j.ajhg.2022.12.008
- Davidson et al., 2023. Allelic bias when performing in-solution enrichment of
  ancient human DNA. doi:10.1101/2023.07.04.547445
- AADR: the Allen Ancient DNA Resource, 2023. doi:10.1038/s41597-024-03031-7
- nf-core/eager: reproducible ancient genome reconstruction, 2021.
  doi:10.7717/peerj.10947
- Jomon genome analysis, 2020. doi:10.1038/s42003-020-01162-2
- Nielsen et al., 2024. Estimating allele frequencies, ancestry proportions and
  genotype likelihoods in the presence of mapping bias.
  doi:10.1101/2024.07.01.601500
- A genomic history of the North Pontic Region, 2025.
  doi:10.1038/s41586-024-08372-2
- Patterns of population structure within the Saudi Arabian population, 2025.
  doi:10.1101/2025.01.10.632500

---

*This document was assembled by cataloguing the scientific decision points
directly from the source, then grounding each against an ancestry-genetics
literature corpus via passage search and grounded synthesis. Verdicts reflect
both the literature and a direct read of the code. Citations are real corpus
papers; where the corpus was thin on a precise point (for example the exact
mean-versus-sum NLL anecdote in D8, or the 0.02 threshold in D6), that is stated
rather than papered over.*
