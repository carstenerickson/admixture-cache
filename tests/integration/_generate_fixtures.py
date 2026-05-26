"""Generate synthetic PLINK BED fixtures for the integration test.

Produces a small but biologically meaningful 3-cluster panel plus a
handful of held-out targets with KNOWN admixture proportions. The
checked-in `fixtures/` directory is the result of running this
script with the seed below; re-run only if the fixture format
changes. (The byproducts are small — ~50 KB total — so they're
checked in for reproducibility rather than regenerated at test
time.)

Design:

- K = 3 ancestral clusters (A, B, C).
- M = 2000 SNPs. Each cluster has its own allele-1 frequency vector
  drawn from a Beta(0.5, 0.5) (mimicking real human-population
  allele-frequency spectra: U-shaped, most variants near
  fixation in at least one population).
- N_panel = 90 panel samples (30 per cluster, labeled in
  `panel.pop`). Genotypes drawn binomial(2, p_cluster).
- N_target = 4 held-out targets with known Q vectors:
    - target_pure_A:   q = (1.0, 0.0, 0.0)
    - target_pure_C:   q = (0.0, 0.0, 1.0)
    - target_AB_5050:  q = (0.5, 0.5, 0.0)
    - target_three:    q = (0.4, 0.4, 0.2)
  Genotypes drawn binomial(2, q^T @ P) per SNP.
- All RNG via `numpy.random.default_rng(SEED)` so the fixtures are
  byte-deterministic. Same Python+numpy version → same bytes.

PLINK 1 BED format (https://www.cog-genomics.org/plink/1.9/formats#bed):
- 3 magic bytes: 0x6c 0x1b 0x01 (third byte = 1 → SNP-major mode).
- Then for each SNP: ceil(N_samples / 4) bytes of 2-bit genotype codes.
  Each byte packs 4 sample genotypes, LSB-first (sample 0 in bits 0-1,
  sample 1 in bits 2-3, ...).
- Genotype encoding (per the 2-bit values, NOT homozygous-major):
    00 = homozygous A1 (ALT)
    01 = missing
    10 = heterozygous
    11 = homozygous A2 (REF)
  We use dosage=2 for hom-ALT (00), dosage=1 for het (10), dosage=0
  for hom-REF (11). All our synthetic data has zero missingness.

Run:
    cd tests/integration
    python _generate_fixtures.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

SEED = 20260526  # Deterministic; matches the date of fixture creation.
K = 3
N_SAMPLES_PER_CLUSTER = 30
N_PANEL_SAMPLES = K * N_SAMPLES_PER_CLUSTER  # 90
N_SNPS = 2000
CLUSTER_NAMES = ["A", "B", "C"]

# Held-out targets with known Q.
TARGETS: list[tuple[str, np.ndarray]] = [
    ("target_pure_A",  np.array([1.0, 0.0, 0.0])),
    ("target_pure_C",  np.array([0.0, 0.0, 1.0])),
    ("target_AB_5050", np.array([0.5, 0.5, 0.0])),
    ("target_three",   np.array([0.4, 0.4, 0.2])),
]


def _encode_bed_byte(dosages: list[int]) -> int:
    """Pack 4 sample dosages into one BED byte.

    Per PLINK BED convention:
        dosage 0 (hom-REF) → 0b11
        dosage 1 (het)     → 0b10
        dosage 2 (hom-ALT) → 0b00
        missing            → 0b01
    """
    code = 0
    for slot, d in enumerate(dosages):
        if d == 0:
            bits = 0b11
        elif d == 1:
            bits = 0b10
        elif d == 2:
            bits = 0b00
        else:
            bits = 0b01  # missing
        code |= bits << (slot * 2)
    return code


def write_bed(out_prefix: Path, genotypes: np.ndarray) -> None:
    """Write a PLINK BED file given an (M_snps, N_samples) dosage matrix.

    Genotypes are integer 0/1/2 (no missing; the fixture is dense)."""
    n_snps, n_samples = genotypes.shape
    bytes_per_snp = (n_samples + 3) // 4
    with out_prefix.with_suffix(".bed").open("wb") as f:
        f.write(b"\x6c\x1b\x01")
        for snp in range(n_snps):
            for byte_idx in range(bytes_per_snp):
                start = byte_idx * 4
                chunk = list(genotypes[snp, start:start + 4])
                while len(chunk) < 4:
                    # Pad final byte with missing (won't be read)
                    chunk.append(-1)
                f.write(struct.pack("B", _encode_bed_byte(chunk)))


def write_bim(out_prefix: Path, n_snps: int) -> None:
    """Write a PLINK BIM file: chr, ID, cM, bp, ALT, REF (tab-separated)."""
    lines = []
    for i in range(n_snps):
        # Synthetic chr 1, evenly-spaced positions, all biallelic A/G.
        lines.append(f"1\trs{i:04d}\t0\t{(i + 1) * 1000}\tA\tG")
    out_prefix.with_suffix(".bim").write_text("\n".join(lines) + "\n")


def write_fam(out_prefix: Path, sample_ids: list[str]) -> None:
    """Write a PLINK FAM file: FID, IID, PAT, MAT, SEX, PHENO."""
    lines = [f"F{i}\t{sid}\t0\t0\t0\t-9" for i, sid in enumerate(sample_ids)]
    out_prefix.with_suffix(".fam").write_text("\n".join(lines) + "\n")


def write_pop(out_prefix: Path, labels: list[str]) -> None:
    """Write the ADMIXTURE supervised `.pop` file.

    One label per sample (same order as the .fam); `-` means
    'unlabeled / project onto the K clusters'. Our integration test
    uses an all-labeled panel (no unlabeled samples).
    """
    out_prefix.with_suffix(".pop").write_text("\n".join(labels) + "\n")


def write_clusters_yaml(out_path: Path) -> None:
    """Minimal YAML opaque to admixture-cache (it just SHA-hashes it).

    The library never parses this — it's the operator's record of
    which sample IDs map to which cluster. We write a deterministic
    string so the cache manifest's SHA is stable across regenerations.
    """
    lines = ["# Synthetic integration-test clusters (admixture-cache)"]
    for cluster, members in zip(CLUSTER_NAMES,
                                _cluster_assignments_for_yaml(),
                                strict=True):
        lines.append(f"{cluster}:")
        for sid in members:
            lines.append(f"  - {sid}")
    out_path.write_text("\n".join(lines) + "\n")


def _cluster_assignments_for_yaml() -> list[list[str]]:
    """Per-cluster sample ID lists matching the .fam ordering."""
    out: list[list[str]] = [[] for _ in range(K)]
    for cluster_idx, cluster_name in enumerate(CLUSTER_NAMES):
        for i in range(N_SAMPLES_PER_CLUSTER):
            sid = f"sample_{cluster_name}_{i:02d}"
            out[cluster_idx].append(sid)
    return out


def main() -> None:
    here = Path(__file__).parent
    fixtures = here / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Cluster allele frequencies — Beta(0.5, 0.5) gives a U-shaped
    # distribution mimicking real human SNP allele-frequency spectra.
    # Clip to [0.05, 0.95] to keep the binomial likelihood numerically
    # stable (boundary frequencies cause logarithm blow-ups in SLSQP).
    P = rng.beta(0.5, 0.5, size=(N_SNPS, K))
    P = np.clip(P, 0.05, 0.95)

    # Panel samples: 30 per cluster.
    panel_genotypes = np.zeros((N_SNPS, N_PANEL_SAMPLES), dtype=np.int8)
    panel_ids: list[str] = []
    panel_labels: list[str] = []
    col = 0
    for cluster_idx, cluster_name in enumerate(CLUSTER_NAMES):
        for i in range(N_SAMPLES_PER_CLUSTER):
            for snp in range(N_SNPS):
                panel_genotypes[snp, col] = rng.binomial(
                    2, P[snp, cluster_idx],
                )
            panel_ids.append(f"sample_{cluster_name}_{i:02d}")
            panel_labels.append(cluster_name)
            col += 1

    panel_prefix = fixtures / "panel"
    write_bed(panel_prefix, panel_genotypes)
    write_bim(panel_prefix, N_SNPS)
    write_fam(panel_prefix, panel_ids)
    write_pop(panel_prefix, panel_labels)
    write_clusters_yaml(fixtures / "clusters.yaml")
    print(f"wrote panel: {panel_prefix}.bed/.bim/.fam/.pop "
          f"({N_PANEL_SAMPLES} samples × {N_SNPS} SNPs, K={K})")

    # Held-out targets. Each target's genotype is sampled from
    # binomial(2, q^T @ P) — i.e., as if it were itself an admixed
    # individual with the documented Q vector. The integration test
    # asserts that admixture-cache recovers this Q within tolerance.
    truth = {"clusters": CLUSTER_NAMES, "targets": {}}
    for target_name, q_true in TARGETS:
        target_freqs = P @ q_true  # (M_snps,)
        target_geno = rng.binomial(
            2, target_freqs[:, None], size=(N_SNPS, 1),
        ).astype(np.int8)
        prefix = fixtures / target_name
        write_bed(prefix, target_geno)
        write_bim(prefix, N_SNPS)
        write_fam(prefix, [target_name])
        truth["targets"][target_name] = q_true.tolist()
        print(f"wrote target: {prefix}.bed/.bim/.fam  q_true={q_true}")

    (fixtures / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(f"wrote truth: {fixtures / 'truth.json'}")
    print(f"\nFixture seed: {SEED}")
    print(f"Total samples: panel={N_PANEL_SAMPLES} target={len(TARGETS)}")


if __name__ == "__main__":
    main()
