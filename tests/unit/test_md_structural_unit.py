from pathlib import Path
import numpy as np

from mody_analyse.analyzers.kbi_analyzer import rebin_rdf
from mody_analyse.analyzers.md_structural import (
    build_local_frame_from_points,
    compute_enrichment_metrics,
    compute_hbond_autocorr,
    compute_hbond_network_metrics,
    detect_first_shell_cutoff,
    write_opendx_grid,
)


def test_detect_first_shell_cutoff_finds_first_minimum_after_peak():
    r = np.linspace(0.0, 2.0, 201)
    g = 1.0 + 1.4 * np.exp(-((r - 0.55) / 0.08) ** 2) - 0.35 * np.exp(-((r - 0.78) / 0.07) ** 2) + 0.3 * np.exp(-((r - 1.05) / 0.12) ** 2)
    g[r < 0.35] = 0.0
    rcut = detect_first_shell_cutoff(r, g, r_min=0.4, r_max=1.2)
    assert 0.70 <= rcut <= 0.85


def test_rebin_rdf_eliminates_fine_bin_cutoff_noise():
    r = np.arange(0.0, 2.0, 0.003)
    g = 1.0 + 1.3 * np.exp(-((r - 0.58) / 0.09) ** 2) - 0.25 * np.exp(-((r - 0.90) / 0.08) ** 2) + 0.15 * np.exp(-((r - 1.15) / 0.12) ** 2)
    g[r < 0.35] = 0.0
    rng = np.random.default_rng(12)
    g[r >= 0.40] += rng.normal(0.0, 0.18, size=int(np.sum(r >= 0.40)))
    r_rb, g_rb = rebin_rdf(r, g, bin_width_nm=0.025)
    rcut = detect_first_shell_cutoff(r_rb, g_rb, r_min=0.45, r_max=1.30)
    assert 0.80 <= rcut <= 1.00


def test_compute_enrichment_metrics_matches_blueprint_example():
    res = compute_enrichment_metrics(
        n_thy_shell=8.0,
        n_dod_shell=1.0,
        n_thy_bulk=370,
        n_dod_bulk=74,
        mw_thy=150.22,
        mw_dod=186.34,
        rho_thy_gcm3=0.950398,
        rho_dod_gcm3=0.81255,
    )
    assert np.isclose(res["x_thy_local"], 8.0 / 9.0, atol=1e-6)
    assert np.isclose(res["x_thy_bulk"], 5.0 / 6.0, atol=1e-6)
    assert np.isclose(res["enrichment_thy_molar"], (8.0 / 9.0) / (5.0 / 6.0), atol=1e-6)
    assert res["enrichment_thy_volume"] > 0.0


def test_compute_hbond_autocorr_recovers_decay():
    rng = np.random.default_rng(7)
    n_frames = 200
    presence = np.zeros((n_frames, 4), dtype=bool)
    for j in range(4):
        state = j % 2 == 0
        for t in range(n_frames):
            if state and rng.random() < 0.15:
                state = False
            elif (not state) and rng.random() < 0.05:
                state = True
            presence[t, j] = state
    tau_ps, acf, tau_rel_ps = compute_hbond_autocorr(presence, dt_ps=20.0, max_lag_frames=20)
    assert len(tau_ps) == len(acf)
    assert np.isclose(acf[0], 1.0, atol=1e-9)
    assert acf[-1] < acf[0]
    assert tau_rel_ps > 0.0


def test_build_local_frame_from_points_orthonormal_and_right_handed():
    origin = np.array([1.0, 2.0, 3.0])
    p_x = np.array([3.0, 2.0, 3.0])
    p_plane = np.array([1.0, 5.0, 3.0])
    rot = build_local_frame_from_points(origin, p_x, p_plane)
    assert rot.shape == (3, 3)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_write_opendx_grid_produces_valid_vmd_header(tmp_path: Path):
    grid = np.zeros((4, 4, 4), dtype=float)
    grid[1, 2, 3] = 2.5
    dx_path = tmp_path / "test.dx"
    write_opendx_grid(dx_path, grid, origin_a=np.array([-2.0, -2.0, -2.0]), spacing_a=1.0)
    txt = dx_path.read_text()
    assert "object 1 class gridpositions counts 4 4 4" in txt
    assert "origin -2.0000 -2.0000 -2.0000" in txt
    assert "delta 1.0000 0.0000 0.0000" in txt
    assert "2.5" in txt


def test_compute_hbond_network_metrics_chain_and_ring():
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)]
    node_types = {0: "THY", 1: "DOD", 2: "THY", 3: "THY", 4: "THY", 5: "DOD"}
    metrics = compute_hbond_network_metrics(edges, node_types)
    assert metrics["n_nodes"] == 6
    assert metrics["n_edges"] == 5
    assert metrics["n_components"] == 2
    assert metrics["largest_component_size"] == 3
    assert np.isclose(metrics["mean_degree"], 10.0 / 6.0, atol=1e-6)
    assert metrics["n_triangles"] == 1
    assert metrics["n_cycles"] == 1
    assert np.isclose(metrics["mean_component_size"], 3.0)
