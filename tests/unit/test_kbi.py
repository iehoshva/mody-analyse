import numpy as np
import pytest

from mody_analyse.analyzers.kbi_analyzer import (
    apply_gvdv_correction,
    compute_running_kbi,
    kruger_extrapolate,
    rebin_rdf,
    solve_ternary_kb_matrix,
)


def test_running_kbi_ideal_gas_is_zero():
    r = np.linspace(0.01, 2.4, 120)
    g = np.ones_like(r)
    G = compute_running_kbi(r, g)
    assert np.allclose(G, 0.0, atol=1e-12)


def test_running_kbi_square_well_matches_analytical_integral():
    r = np.linspace(0.0, 2.4, 2401)
    g = np.ones_like(r)
    mask = (r >= 0.4) & (r <= 0.8)
    g[mask] = 2.5
    G = compute_running_kbi(r, g)
    expected_nm3 = (4.0 * np.pi / 3.0) * (2.5 - 1.0) * (0.8**3 - 0.4**3)
    expected_cm3_mol = expected_nm3 * 1e-21 * 6.02214076e23
    assert np.isclose(G[-1], expected_cm3_mol, rtol=1e-2)


def test_gvdv_correction_removes_closed_box_depletion_divergence():
    r = np.linspace(0.02, 2.4, 120)
    # Closed-box depleted RDF for N=12 solutes (no real excess structure outside r=0.5 nm)
    g_raw = np.ones_like(r) * (1.0 - 1.0 / 12.0)
    g_raw[r < 0.3] = 0.0
    rho_j = 12.0 / 125.0  # molecules / nm^3
    g_corr = apply_gvdv_correction(r, g_raw, rho_j=rho_j, n_j=12, is_like_pair=True)
    # Tail beyond exclusion cavity should recover ~1.0 rather than ~0.9167
    assert np.allclose(g_corr[r > 0.6], 1.0, atol=0.03)
    G_raw = compute_running_kbi(r, g_raw)
    G_corr = compute_running_kbi(r, g_corr)
    assert abs(G_corr[-1]) < abs(G_raw[-1]) * 0.25


def test_kruger_extrapolate_with_kernel_recovers_square_well():
    r = np.linspace(0.01, 2.4, 120)
    g = np.ones_like(r)
    g[(r >= 0.3) & (r <= 0.6)] = 2.2
    expected_nm3 = (4.0 * np.pi / 3.0) * (2.2 - 1.0) * (0.6**3 - 0.3**3)
    expected_cm3 = expected_nm3 * 1e-21 * 6.02214076e23
    G_inf = kruger_extrapolate(r, g, r_fit_min=0.75, r_fit_max=1.15)
    assert np.isclose(G_inf, expected_cm3, rtol=0.08)


def test_solve_ternary_kb_matrix_symmetric_ideal_mixture():
    rho = np.array([2.96, 0.592, 0.096])  # molecules/nm^3
    G_cm3 = np.zeros((3, 3))
    out = solve_ternary_kb_matrix(G_cm3, rho)
    assert np.isclose(out["pseudo_binary_gamma_factor"], 1.0, atol=1e-6)
    assert out["ternary_gamma_matrix"].shape == (3, 3)
    assert np.allclose(out["ternary_gamma_matrix"], np.eye(3), atol=1e-6)
