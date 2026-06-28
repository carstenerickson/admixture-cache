"""Genotype-likelihood (beagle) projection path tests (SCIENCE.md D17).

Covers the beagle reader, panel alignment + REF/ALT orientation, the
marginalized-HWE GL likelihood, and an end-to-end project_target_gl run that
needs no external binaries (the GL path matches by variant ID in pure Python).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from admixture_cache import PanelCacheError, PanelCacheManifest
from admixture_cache.gl import (
    BeagleGL,
    align_gl_to_panel,
    read_beagle_gl,
)
from admixture_cache.orchestration import project_target_gl
from admixture_cache.projection import (
    numpy_supervised_projection,
    numpy_supervised_projection_gl,
)

# ─── fixtures / helpers ──────────────────────────────────────────────────


def _write_beagle(
    path: Path,
    markers: list[str],
    allele1: list[str],
    allele2: list[str],
    gl: np.ndarray,
) -> Path:
    """Write a single-individual beagle GL file (marker, allele1, allele2,
    then 3 GL columns)."""
    lines = ["marker\tallele1\tallele2\tInd0\tInd0\tInd0"]
    for m, a1, a2, row in zip(markers, allele1, allele2, gl, strict=True):
        lines.append(f"{m}\t{a1}\t{a2}\t{row[0]}\t{row[1]}\t{row[2]}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_panel_bim(path: Path, markers: list[str], a1a2: list[tuple[str, str]]) -> None:
    lines = [
        f"1\t{m}\t0\t{i + 1}\t{a1}\t{a2}"
        for i, (m, (a1, a2)) in enumerate(zip(markers, a1a2, strict=True))
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_gl_cache(tmp_path: Path, p_matrix: np.ndarray, markers: list[str]) -> Path:
    """Minimal cache dir for the GL path: panel.bim (alleles A/G), panel.K.P,
    and a manifest. No plink2 needed."""
    cache = tmp_path / "cache"
    cache.mkdir()
    k = p_matrix.shape[1]
    _write_panel_bim(cache / "panel.bim", markers, [("A", "G")] * len(markers))
    np.savetxt(cache / f"panel.{k}.P", p_matrix)
    manifest = PanelCacheManifest(
        panel_id="p", panel_version="v", panel_bim_sha256="a" * 64,
        clusters_yaml_sha256="b" * 64, k=k, admixture_version="1.4.0",
        seeds_used=[1], best_seed=1, best_loglikelihood=-1.0,
        restart_sd_max=0.0, cluster_order=[f"c{i}" for i in range(k)],
        strand_ambiguous_excluded=True,
        build_wallclock_seconds=1.0,
        build_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    (cache / "manifest.json").write_text(manifest.model_dump_json())
    return cache


# ─── read_beagle_gl ──────────────────────────────────────────────────────


class TestReadBeagleGL:
    def test_parses_letters(self, tmp_path: Path) -> None:
        p = _write_beagle(
            tmp_path / "t.beagle",
            ["rs1", "rs2"], ["A", "C"], ["G", "T"],
            np.array([[0.9, 0.08, 0.02], [0.1, 0.2, 0.7]]),
        )
        b = read_beagle_gl(p)
        assert b.marker_ids == ["rs1", "rs2"]
        assert b.allele1 == ["A", "C"]
        assert b.gl.shape == (2, 3)
        assert np.isclose(b.gl[0, 0], 0.9)

    def test_parses_numeric_alleles(self, tmp_path: Path) -> None:
        # ANGSD numeric coding: 0=A,1=C,2=G,3=T (decoded at alignment time).
        p = _write_beagle(
            tmp_path / "t.beagle", ["rs1"], ["0"], ["2"],
            np.array([[0.8, 0.15, 0.05]]),
        )
        b = read_beagle_gl(p)
        assert b.allele1 == ["0"] and b.allele2 == ["2"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PanelCacheError, match="not found"):
            read_beagle_gl(tmp_path / "nope.beagle")

    def test_multi_individual_rejected(self, tmp_path: Path) -> None:
        # 9 columns = 2 individuals; the GL path projects one at a time.
        path = tmp_path / "multi.beagle"
        path.write_text(
            "marker\tallele1\tallele2\tInd0\tInd0\tInd0\tInd1\tInd1\tInd1\n"
            "rs1\tA\tG\t0.9\t0.1\t0.0\t0.5\t0.4\t0.1\n",
        )
        with pytest.raises(PanelCacheError, match="6 columns"):
            read_beagle_gl(path)

    def test_non_numeric_gl_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.beagle"
        path.write_text(
            "marker\tallele1\tallele2\tInd0\tInd0\tInd0\n"
            "rs1\tA\tG\t0.9\tNOTNUM\t0.0\n",
        )
        with pytest.raises(PanelCacheError, match=r"not numeric|non-finite"):
            read_beagle_gl(path)

    def test_empty_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.beagle"
        path.write_text("marker\tallele1\tallele2\tInd0\tInd0\tInd0\n")
        with pytest.raises(PanelCacheError, match="no data"):
            read_beagle_gl(path)

    def test_negative_gl_rejected(self, tmp_path: Path) -> None:
        # Negative values mean log/phred-scaled GLs, which this path does not
        # support (it expects linear probabilities).
        path = tmp_path / "neg.beagle"
        path.write_text(
            "marker\tallele1\tallele2\tInd0\tInd0\tInd0\n"
            "rs1\tA\tG\t-0.1\t0.5\t0.6\n",
        )
        with pytest.raises(PanelCacheError, match="negative"):
            read_beagle_gl(path)

    def test_numeric_markers_preserved_as_strings(self, tmp_path: Path) -> None:
        # Numeric markers must stay verbatim ("100", not float-coerced "100.0"),
        # else they would never match panel.bim variant IDs.
        p = _write_beagle(
            tmp_path / "num.beagle", ["100", "200"], ["A", "A"], ["G", "G"],
            np.array([[0.9, 0.08, 0.02], [0.1, 0.2, 0.7]]),
        )
        b = read_beagle_gl(p)
        assert b.marker_ids == ["100", "200"]


# ─── align_gl_to_panel ───────────────────────────────────────────────────


class TestAlignGLToPanel:
    def _panel(self, tmp_path: Path, a1a2: list[tuple[str, str]]) -> Path:
        bim = tmp_path / "panel.bim"
        _write_panel_bim(bim, [f"rs{i}" for i in range(len(a1a2))], a1a2)
        return bim

    def test_identity_orientation(self, tmp_path: Path) -> None:
        # panel allele1 == beagle allele2 (minor) -> GL triple kept as-is
        # (both indexed by count of the panel's allele 1).
        bim = self._panel(tmp_path, [("A", "G")])
        beagle = BeagleGL(["rs0"], ["G"], ["A"], np.array([[0.7, 0.2, 0.1]]))
        out = align_gl_to_panel(beagle=beagle, panel_bim=bim)
        np.testing.assert_allclose(out[0], [0.7, 0.2, 0.1])

    def test_reversed_orientation(self, tmp_path: Path) -> None:
        # panel allele1 == beagle allele1 (major) -> triple reversed.
        bim = self._panel(tmp_path, [("A", "G")])
        beagle = BeagleGL(["rs0"], ["A"], ["G"], np.array([[0.7, 0.2, 0.1]]))
        out = align_gl_to_panel(beagle=beagle, panel_bim=bim)
        np.testing.assert_allclose(out[0], [0.1, 0.2, 0.7])

    def test_strand_flip_complement_match(self, tmp_path: Path) -> None:
        # panel A/G, beagle T/C (the complement) -> matched via complement.
        # beagle allele2=C complements to G == panel allele1 -> identity.
        bim = self._panel(tmp_path, [("A", "G")])
        beagle = BeagleGL(["rs0"], ["C"], ["T"], np.array([[0.6, 0.3, 0.1]]))
        out = align_gl_to_panel(beagle=beagle, panel_bim=bim)
        # beagle alleles complement to G/A: panel A1=A == complemented allele2 (A)
        # -> identity.
        np.testing.assert_allclose(out[0], [0.6, 0.3, 0.1])

    def test_strand_ambiguous_dropped(self, tmp_path: Path) -> None:
        bim = self._panel(tmp_path, [("A", "T"), ("A", "G")])
        beagle = BeagleGL(
            ["rs0", "rs1"], ["A", "G"], ["T", "A"],
            np.array([[0.9, 0.1, 0.0], [0.5, 0.3, 0.2]]),
        )
        out = align_gl_to_panel(beagle=beagle, panel_bim=bim)
        assert np.isnan(out[0]).all()       # A/T ambiguous -> dropped
        assert not np.isnan(out[1]).any()   # A/G kept

    def test_missing_marker_is_nan(self, tmp_path: Path) -> None:
        bim = self._panel(tmp_path, [("A", "G"), ("A", "G")])
        beagle = BeagleGL(["rs0"], ["G"], ["A"], np.array([[0.7, 0.2, 0.1]]))
        out = align_gl_to_panel(beagle=beagle, panel_bim=bim)
        assert not np.isnan(out[0]).any()
        assert np.isnan(out[1]).all()       # rs1 absent from beagle

    def test_allele_mismatch_is_nan(self, tmp_path: Path) -> None:
        # Two panel SNPs so something places (align raises only if NOTHING
        # aligns). rs0's beagle alleles A/C neither match nor complement the
        # panel's A/G; rs1 matches.
        bim = self._panel(tmp_path, [("A", "G"), ("A", "G")])
        beagle = BeagleGL(
            ["rs0", "rs1"], ["A", "G"], ["C", "A"],
            np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]),
        )
        out = align_gl_to_panel(beagle=beagle, panel_bim=bim)
        assert np.isnan(out[0]).all()       # A/C incompatible with A/G
        assert not np.isnan(out[1]).any()   # rs1 placed

    def test_no_overlap_raises(self, tmp_path: Path) -> None:
        bim = self._panel(tmp_path, [("A", "G")])
        beagle = BeagleGL(["other"], ["G"], ["A"], np.array([[0.7, 0.2, 0.1]]))
        with pytest.raises(PanelCacheError, match="no panel SNP"):
            align_gl_to_panel(beagle=beagle, panel_bim=bim)


# ─── numpy_supervised_projection_gl ──────────────────────────────────────


class TestGLProjectionMath:
    def test_point_mass_gl_equals_hard_call(self) -> None:
        """A GL with all mass on the true genotype must give exactly the
        hard-call result (the marginal reduces to the binomial pmf)."""
        rng = np.random.default_rng(0)
        m, k = 2000, 3
        p = rng.uniform(0.05, 0.95, size=(m, k))
        q_true = rng.dirichlet(np.ones(k))
        g = rng.binomial(2, p @ q_true)
        gl = np.zeros((m, 3))
        gl[np.arange(m), g] = 1.0

        q_gl, _, _ = numpy_supervised_projection_gl(gl=gl, p_matrix=p, k=k)
        q_hc, _, _ = numpy_supervised_projection(
            target_dosage=g.astype(np.float64), p_matrix=p, k=k,
        )
        np.testing.assert_allclose(q_gl, q_hc, atol=1e-6)

    def test_recovers_known_q(self) -> None:
        rng = np.random.default_rng(3)
        m, k = 3000, 4
        p = rng.uniform(0.05, 0.95, size=(m, k))
        q_true = rng.dirichlet(np.ones(k))
        g = rng.binomial(2, p @ q_true)
        gl = np.full((m, 3), 0.05)
        gl[np.arange(m), g] = 0.9  # confident but not point-mass
        q, _, converged = numpy_supervised_projection_gl(gl=gl, p_matrix=p, k=k)
        assert converged
        assert np.isclose(q.sum(), 1.0, atol=1e-6)
        assert np.max(np.abs(q - q_true)) < 0.10

    def test_nan_rows_masked(self) -> None:
        rng = np.random.default_rng(1)
        m, k = 1000, 2
        p = rng.uniform(0.05, 0.95, size=(m, k))
        g = rng.binomial(2, p @ np.array([0.5, 0.5]))
        gl = np.zeros((m, 3))
        gl[np.arange(m), g] = 1.0
        gl[:300] = np.nan  # missing sites
        q, _, converged = numpy_supervised_projection_gl(gl=gl, p_matrix=p, k=k)
        assert converged
        assert np.isclose(q.sum(), 1.0, atol=1e-6)

    def test_all_missing_raises(self) -> None:
        p = np.column_stack([np.full(100, 0.9), np.full(100, 0.1)])
        gl = np.full((100, 3), np.nan)
        with pytest.raises(PanelCacheError, match="zero usable GL"):
            numpy_supervised_projection_gl(gl=gl, p_matrix=p, k=2)

    def test_shape_mismatch_raises(self) -> None:
        p = np.column_stack([np.full(100, 0.9), np.full(100, 0.1)])
        gl = np.zeros((99, 3))
        with pytest.raises(AssertionError):
            numpy_supervised_projection_gl(gl=gl, p_matrix=p, k=2)

    def test_unnormalized_gl_scale_invariant(self) -> None:
        """The solver normalizes each site, so a pure rescaling of all GLs must
        give the same Q. Regression for the eps-clip scale bug: at a tiny scale
        the pre-fix code floored every site and returned the uniform start."""
        rng = np.random.default_rng(11)
        m, k = 2000, 3
        p = rng.uniform(0.05, 0.95, size=(m, k))
        q_true = rng.dirichlet(np.ones(k))
        g = rng.binomial(2, p @ q_true)
        gl = np.full((m, 3), 0.1)
        gl[np.arange(m), g] = 0.8
        q_ref, _, _ = numpy_supervised_projection_gl(gl=gl, p_matrix=p, k=k)
        q_tiny, _, conv = numpy_supervised_projection_gl(
            gl=gl * 1e-11, p_matrix=p, k=k,
        )
        assert conv
        np.testing.assert_allclose(q_tiny, q_ref, atol=1e-6)
        # And it is not just pinned at the uniform start.
        assert np.max(np.abs(q_tiny - np.full(k, 1 / k))) > 0.05

    def test_zero_information_rows_masked(self) -> None:
        """All-zero GL rows carry no information and must be masked out (not
        floored to a flat term), leaving the informative sites to recover Q."""
        rng = np.random.default_rng(12)
        m, k = 2000, 2
        p = rng.uniform(0.05, 0.95, size=(m, k))
        q_true = rng.dirichlet(np.ones(k))
        g = rng.binomial(2, p @ q_true)
        gl = np.zeros((m, 3))
        gl[np.arange(m), g] = 1.0
        gl[:400] = 0.0  # zero-information rows
        q, _, converged = numpy_supervised_projection_gl(gl=gl, p_matrix=p, k=k)
        assert converged
        assert np.max(np.abs(q - q_true)) < 0.10


# ─── project_target_gl (end to end, no binaries) ─────────────────────────


class TestProjectTargetGLEndToEnd:
    def test_recovers_known_q_from_beagle(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(7)
        m, k = 3000, 3
        p = rng.uniform(0.05, 0.95, size=(m, k))
        q_true = rng.dirichlet(np.ones(k))
        markers = [f"rs{i}" for i in range(m)]
        # Diploid genotype = count of panel allele 1 (A); point-mass GL.
        g = rng.binomial(2, p @ q_true)
        gl = np.zeros((m, 3))
        gl[np.arange(m), g] = 1.0
        # beagle allele2 = A (== panel allele1) -> identity orientation.
        beagle = _write_beagle(
            tmp_path / "t.beagle", markers, ["G"] * m, ["A"] * m, gl,
        )
        cache = _write_gl_cache(tmp_path, p, markers)

        result = project_target_gl(target_gl_beagle=beagle, cache_dir=cache)
        assert result.converged
        assert result.n_snps_used == m
        assert np.isnan(result.heterozygosity)  # no hard genotypes in GL mode
        assert np.max(np.abs(result.target_q - q_true)) < 0.10
