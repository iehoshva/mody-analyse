"""Analytical & unit verification tests for all 3 waves of MoDy Analyse extended modules."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mody_analyse.analyzers.alchemical import (
    analyze_dhdl_window_set,
    compute_bar_pair,
    compute_mbar,
    compute_ti_free_energy,
)
from mody_analyse.analyzers.conformation_geometry import (
    BONDI_VDW_NM,
    compute_conformational_pca_clusters,
    compute_dihedral_deg,
    compute_dihedral_fes_1d,
    compute_periodic_voronoi,
    compute_radius_of_gyration,
    compute_sasa_shrake_rupley,
)
from mody_analyse.analyzers.electro_dynamics import (
    E_NM_TO_CM,
    EPS0_VAC,
    KB_J_K,
    compute_dielectric_constant,
    compute_glassy_metrics,
    compute_onsager_and_conductivity,
    compute_rotational_acf,
)
from mody_analyse.analyzers.interface_slab import (
    compute_surface_tension,
    compute_z_density_and_orientation_profiles,
    fit_slab_tanh_profile,
)
from mody_analyse.analyzers.kbi_analyzer import NM3_TO_CM3_MOL
from mody_analyse.analyzers.kbi_extended import (
    compute_kb_thermodynamics_full,
    compute_setschenow_constant,
    compute_structure_factor_sq,
    compute_subvolume_fluctuation_kbi,
    compute_temperature_series_gibbs_helmholtz,
    compute_waxs_atomistic_iq,
    cromer_mann_form_factor,
    dawass_extrapolate,
    integrate_composition_series_kbi,
    species_cromer_mann_form_factor,
)
from mody_analyse.core.pbc import (
    box_inscribed_radius_nm,
    box_matrix_from_dimensions,
    box_volume_nm3,
    minimum_image_displacement,
    wrap_positions,
)
from mody_analyse.analyzers.spectroscopy import THZ_TO_CM1, compute_ir_spectrum, compute_vdos_spectrum
from mody_analyse.analyzers.thermo_bulk import compute_npt_fluctuations
from mody_analyse.cli import build_arg_parser
from mody_analyse.core.registry import (
    ANALYSIS_PROFILES,
    evaluate_eligibility,
    export_catalog_payload,
    resolve_selected_analyses,
)


# ---------------------------------------------------------------------------
# 0. Registry, CLI & Physics Gatekeeper
# ---------------------------------------------------------------------------

def test_registry_profiles_and_gatekeeper_eligibility() -> None:
    sel_all = resolve_selected_analyses({"profile": "everything"})
    assert "kbi_full" in sel_all
    assert "alchemical_free_energy" in sel_all
    assert "spectroscopy_vdos_ir" in sel_all

    sel_custom = resolve_selected_analyses(
        {"profile": "electrolyte_des", "disabled": ["dielectric"]},
    )
    assert "electrochemistry_onsager" in sel_custom
    assert "dielectric" not in sel_custom

    # Verify Physics Gatekeeper skips ineligible modules with explicit reasons
    caps = {
        "n_runs": 2,
        "has_thermo": True,
        "has_stress": True,
        "has_traj": True,
        "has_trr": False,
        "has_rdf": True,
        "has_topol": True,
        "has_atomic_charges": True,
        "has_ionic_species": False,  # neutral liquid -> should skip electrochemistry_onsager
        "has_polyatomic": True,
        "has_torsion_candidate": True,
        "n_compositions": 1,  # single composition -> should skip kbi_composition_series
        "n_temperatures": 1,  # single T -> should skip kbi_temperature_series
        "dt_ps": 1.0,  # too coarse for VDOS/IR -> should skip spectroscopy_vdos_ir
        "n_frames": 100,
        "has_velocities": False,
        "is_slab_box": False,  # bulk isotropic -> should skip interface_slab
        "n_dhdl_files": 0,  # no FEP -> should skip alchemical_free_energy
        "prod_ns": 5.0,  # < 10 ns -> dielectric should RUN with warning
        "vdos_max_dt_ps": 0.05,
    }
    decisions, df_gate = evaluate_eligibility([], None, sel_all, caps_override=caps)
    assert not df_gate.empty
    assert decisions["thermo_fluctuations"].status == "RUN"
    assert decisions["kbi_full"].status == "RUN"
    assert decisions["electrochemistry_onsager"].status == "SKIPPED"
    assert decisions["kbi_composition_series"].status == "SKIPPED"
    assert decisions["kbi_temperature_series"].status == "SKIPPED"
    assert decisions["spectroscopy_vdos_ir"].status == "SKIPPED"
    assert "Nyquist" in decisions["spectroscopy_vdos_ir"].reason
    assert decisions["interface_slab"].status == "SKIPPED"
    assert decisions["alchemical_free_energy"].status == "SKIPPED"
    assert decisions["dielectric"].status == "RUN"
    assert any("10 ns" in w for w in decisions["dielectric"].warnings)

    parser = build_arg_parser()
    args = parser.parse_args(["--profile", "electrolyte_des", "--analyses", "equil,kbi_full"])
    assert args.profile == "electrolyte_des"
    assert args.analyses == "equil,kbi_full"
    cat = export_catalog_payload()
    assert set(cat["profiles"].keys()) == set(ANALYSIS_PROFILES.keys())
    assert len(cat["modules"]) >= 20


# ---------------------------------------------------------------------------
# 1. Wave 1: Full KBI Suite & Bulk NPT Fluctuations
# ---------------------------------------------------------------------------

def test_dawass_extrapolation_recovers_square_well_and_ideal_gas() -> None:
    r = np.linspace(0.0, 4.0, 2001)
    # Ideal gas: g(r) = 1 -> G_inf = 0
    res_ideal = dawass_extrapolate(r, np.ones_like(r))
    assert res_ideal["g_inf_cm3_mol"] == pytest.approx(0.0, abs=1e-6)
    assert res_ideal["convergence_status"] == "PASS"

    # Square-well: g=0 for r<0.3, g=2.0 for 0.3<=r<=0.6, g=1.0 for r>0.6
    g = np.ones_like(r)
    g[r < 0.3] = 0.0
    g[(r >= 0.3) & (r <= 0.6)] = 2.0
    exact_nm3 = (4.0 * math.pi / 3.0) * (-0.3 ** 3 + (0.6 ** 3 - 0.3 ** 3))
    exact_cm3 = exact_nm3 * NM3_TO_CM3_MOL
    res_sw = dawass_extrapolate(r, g, r_min=1.0, r_max=1.8, n_radii=12)
    assert res_sw["g_inf_cm3_mol"] == pytest.approx(exact_cm3, rel=0.03)
    assert res_sw["convergence_status"] == "PASS"


def test_subvolume_fluctuation_ideal_gas_near_zero() -> None:
    rng = np.random.default_rng(7)
    box = np.array([5.0, 5.0, 5.0])
    coms = {
        "A": rng.uniform(0.0, 5.0, size=(80, 120, 3)),
        "B": rng.uniform(0.0, 5.0, size=(80, 120, 3)),
    }
    df = compute_subvolume_fluctuation_kbi(coms, box, n_spheres=40, seed=7)
    assert len(df) == 3
    # For finite random samples, ideal gas G_ij should be small around 0
    assert np.all(np.abs(df["g_subvol_rmax_cm3_mol"].to_numpy()) < 80.0)


def test_kb_thermodynamics_full_and_sq_ideal_gas() -> None:
    rho = np.array([4.0, 6.0])  # nm^-3, total = 10 nm^-3
    G_zero = np.zeros((2, 2))
    T = 300.0
    res = compute_kb_thermodynamics_full(G_zero, rho, T, species_names=["A", "B"])
    expected_kappa_pa = 1.0 / (10.0 * 1e27 * KB_J_K * T)
    assert res["kappa_t_pa_inv"] == pytest.approx(expected_kappa_pa, rel=1e-6)
    assert res["pmv_cm3_mol"]["A"] == pytest.approx(NM3_TO_CM3_MOL / 10.0, rel=1e-6)
    assert res["gamma_binary"] == pytest.approx(1.0, rel=1e-6)
    assert res["S_NN_0"] == pytest.approx(1.0, rel=1e-6)
    assert res["S_CC_0"] == pytest.approx(0.4 * 0.6, rel=1e-6)
    assert res["S_NC_0"] == pytest.approx(0.0, abs=1e-10)

    r = np.linspace(0.0, 2.5, 251)
    q = np.linspace(0.0, 20.0, 41)
    g_pairs = {(0, 0): np.ones_like(r), (0, 1): np.ones_like(r), (1, 1): np.ones_like(r)}
    sq = compute_structure_factor_sq(r, g_pairs, rho, q, form_factors=np.array([10.0, 18.0]))
    assert np.allclose(sq["I_total"], 1.0, atol=1e-10)
    assert np.allclose(sq["S_NN"], 1.0, atol=1e-10)


def test_composition_series_regular_solution_and_gibbs_helmholtz() -> None:
    # Exact Regular Solution: GE/RT = A * x1 * (1 - x1) -> Gamma(x1) = 1 - 2 * A * x1 * (1 - x1)
    A = 1.2
    x1 = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    gamma_exact = 1.0 - 2.0 * A * x1 * (1.0 - x1)
    comp_df = pd.DataFrame({"x1": x1, "gamma_factor": gamma_exact})
    out = integrate_composition_series_kbi(comp_df, temperature_k=300.0, rk_order=2)
    assert np.allclose(out["ln_gamma_1"].to_numpy(), A * ((1.0 - x1) ** 2), atol=1e-5)
    assert np.allclose(out["ln_gamma_2"].to_numpy(), A * (x1 ** 2), atol=1e-5)
    rt_kj = 8.314462618e-3 * 300.0
    assert np.allclose(out["g_excess_kj_mol"].to_numpy(), A * rt_kj * x1 * (1.0 - x1), atol=1e-5)

    # Setschenow linear fit
    c = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    set_res = compute_setschenow_constant(c, 0.35 * c)
    assert set_res["K_setschenow_L_mol"] == pytest.approx(0.35, rel=1e-6)

    # Gibbs–Helmholtz: G^E(T) = H^E - T * S^E with H^E = 4.5 kJ/mol, S^E = 0.01 kJ/(mol K)
    T_arr = np.array([290.0, 310.0, 330.0])
    ge_arr = 4.5 - T_arr * 0.01
    gh_df = compute_temperature_series_gibbs_helmholtz(
        pd.DataFrame({"temperature_k": T_arr, "g_excess_kj_mol": ge_arr})
    )
    assert np.allclose(gh_df["h_excess_kj_mol"].to_numpy(), 4.5, atol=1e-5)
    assert np.allclose(gh_df["s_excess_j_mol_k"].to_numpy(), 10.0, atol=1e-3)


def test_npt_fluctuations_and_water_hildebrand_solubility() -> None:
    rng = np.random.default_rng(123)
    n = 20000
    n_mol = 500
    T_val = 298.15
    T = np.full(n, T_val)
    # Construct correlated (H, V) with exact target variances & covariance
    sigma_v = 0.12  # nm^3 around V_mean = 15.0 nm^3 (~30 cm^3/mol)
    sigma_h = 40.0  # kJ/mol-box
    corr = 0.5
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    V = 15.0 + sigma_v * z1
    H = -20000.0 + sigma_h * (corr * z1 + math.sqrt(1.0 - corr ** 2) * z2)

    # Water-like cohesive energy: 41.5 kJ/mol per molecule in V_1mol = 0.02989 nm^3 -> delta ≈ 48.0 MPa^(1/2)
    V_water = np.full(n, n_mol * 0.02989)
    pot_liq = np.full(n, -41.5 * n_mol)
    res_w = compute_npt_fluctuations(T, V_water, H, n_mol, potential_kj_mol=pot_liq, gas_energy_kj_mol=0.0)
    assert res_w["ced_mode"] == "gas_reference"
    assert res_w["hildebrand_delta_mpa_half"] == pytest.approx(48.0, abs=0.2)

    res = compute_npt_fluctuations(T, V, H, n_mol)
    expected_cp = ((sigma_h * 1000.0) ** 2) / (n_mol * 8.314462618 * T_val ** 2)
    expected_kappa_gpa = (sigma_v ** 2) * 1e-27 / (KB_J_K * T_val * 15.0) * 1e9
    assert res["cp_j_mol_k"] == pytest.approx(expected_cp, rel=0.05)
    assert res["kappa_t_gpa_inv"] == pytest.approx(expected_kappa_gpa, rel=0.05)
    assert res["cv_j_mol_k"] < res["cp_j_mol_k"]
    assert res["alpha_p_1e3_k_inv"] > 0.0


# ---------------------------------------------------------------------------
# 2. Wave 2: Electrochemistry, Rotational, Glassy & Dielectric
# ---------------------------------------------------------------------------

def test_onsager_conductivity_nernst_einstein_and_neutral_ion_pair_limits() -> None:
    rng = np.random.default_rng(42)
    n_frames, n_ions, dt = 250, 60, 1.0
    D_step = 0.05  # nm/sqrt(ps)
    steps_cat = rng.normal(0.0, D_step, size=(n_frames, n_ions, 3))
    steps_ani = rng.normal(0.0, D_step, size=(n_frames, n_ions, 3))
    r_cat = np.cumsum(steps_cat, axis=0)
    r_ani = np.cumsum(steps_ani, axis=0)

    # Uncorrelated independent ions: sigma_EH ~ sigma_NE, t+ + t- = 1
    res_free = compute_onsager_and_conductivity(
        {"CAT": r_cat, "ANI": r_ani},
        {"CAT": +1.0, "ANI": -1.0},
        dt_ps=dt,
        volume_nm3=64.0,
        temperature_k=300.0,
    )
    assert res_free["summary"]["sigma_ne_s_m"] > 0
    assert res_free["summary"]["sigma_eh_s_m"] > 0
    t_sum = sum(r["t_true_onsager"] for r in res_free["transference_rows"])
    assert t_sum == pytest.approx(1.0, abs=1e-6)

    # Perfectly bound neutral ion pairs (r_ani == r_cat): collective charge displacement = 0 -> sigma_EH = 0!
    res_paired = compute_onsager_and_conductivity(
        {"CAT": r_cat, "ANI": r_cat.copy()},
        {"CAT": +1.0, "ANI": -1.0},
        dt_ps=dt,
        volume_nm3=64.0,
        temperature_k=300.0,
    )
    assert res_paired["summary"]["sigma_eh_s_m"] == pytest.approx(0.0, abs=1e-10)
    assert res_paired["summary"]["ionicity"] == pytest.approx(0.0, abs=1e-10)


def test_rotational_debye_and_glassy_gaussian_and_dielectric_ideal() -> None:
    rng = np.random.default_rng(99)
    # Rotational diffusion on unit sphere: small random angular steps -> Debye ratio tau1 / tau2 ≈ 3.0
    n_frames, n_mol, dt = 300, 150, 0.5
    u = np.zeros((n_frames, n_mol, 3))
    v0 = rng.normal(size=(n_mol, 3))
    u[0] = v0 / np.linalg.norm(v0, axis=-1, keepdims=True)
    step_angle = 0.22
    for t in range(1, n_frames):
        perp = rng.normal(size=(n_mol, 3))
        perp -= np.sum(perp * u[t - 1], axis=-1, keepdims=True) * u[t - 1]
        perp /= np.linalg.norm(perp, axis=-1, keepdims=True)
        u_new = u[t - 1] + step_angle * perp
        u[t] = u_new / np.linalg.norm(u_new, axis=-1, keepdims=True)

    rot = compute_rotational_acf(u, dt_ps=dt)
    assert 2.4 <= rot["debye_ratio_tau1_tau2"] <= 3.6

    # Gaussian 3D Brownian walk -> alpha2(t) ≈ 0, Fs(q,t) = exp(-q^2 D t)
    D_nm2_ps = 0.005  # step variance per axis = 2 D dt
    sigma_step = math.sqrt(2.0 * D_nm2_ps * dt)
    pos = np.cumsum(rng.normal(0.0, sigma_step, size=(200, 200, 3)), axis=0)
    q_star = 5.0
    gl = compute_glassy_metrics(pos, dt_ps=dt, q_star_inv_nm=q_star)
    assert abs(gl["alpha2_max"]) < 0.2
    expected_tau_alpha = 1.0 / ((q_star ** 2) * D_nm2_ps)
    assert gl["tau_alpha_ps"] == pytest.approx(expected_tau_alpha, rel=0.25)

    # Ideal uncorrelated isotropic dipoles -> g_K ≈ 1.0 and exact epsilon_0
    n_f_d, n_m_d = 4000, 100
    mu_mag = 0.04  # e*nm (~1.92 D)
    dirs = rng.normal(size=(n_f_d, n_m_d, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dipoles = dirs * mu_mag
    vol_nm3, T = 27.0, 300.0
    diel = compute_dielectric_constant(dipoles, vol_nm3, T)
    assert diel["kirkwood_g_k"] == pytest.approx(1.0, rel=0.1)
    expected_eps0 = 1.0 + (n_m_d * (mu_mag * E_NM_TO_CM) ** 2) / (3.0 * EPS0_VAC * (vol_nm3 * 1e-27) * KB_J_K * T)
    assert diel["epsilon_0"] == pytest.approx(expected_eps0, rel=0.1)


# ---------------------------------------------------------------------------
# 3. Wave 3: Spectroscopy, Conformation/Voronoi, Interface/Slab & Alchemical
# ---------------------------------------------------------------------------

def test_spectroscopy_harmonic_oscillator_peak_frequency() -> None:
    dt_ps = 0.002  # Nyquist = 8339 cm^-1
    t = np.arange(1024) * dt_ps
    nu0_cm1 = 1000.0
    f0_thz = nu0_cm1 / THZ_TO_CM1
    v_osc = np.zeros((len(t), 2, 3))
    v_osc[:, 0, 0] = np.cos(2.0 * math.pi * f0_thz * t)
    v_osc[:, 1, 0] = -np.cos(2.0 * math.pi * f0_thz * t)

    vd = compute_vdos_spectrum(v_osc, np.array([12.0, 16.0]), dt_ps)
    assert vd["top_peaks_cm1"]
    assert abs(vd["top_peaks_cm1"][0] - nu0_cm1) < 15.0

    jm = np.zeros((len(t), 3))
    jm[:, 0] = np.cos(2.0 * math.pi * f0_thz * t)
    ir = compute_ir_spectrum(jm, dt_ps, temperature_k=300.0)
    assert ir["top_peaks_cm1"]
    assert abs(ir["top_peaks_cm1"][0] - nu0_cm1) < 15.0


def test_conformation_sasa_dihedral_pca_and_periodic_voronoi() -> None:
    # Single oxygen atom -> exact sphere SASA = 4 pi (r_O + r_probe)^2
    probe = 0.14
    s_single = compute_sasa_shrake_rupley(np.array([[0.0, 0.0, 0.0]]), ["O"], probe_radius_nm=probe, n_sphere_points=192)
    r_tot = BONDI_VDW_NM["O"] + probe
    assert s_single["sasa_total_nm2"] == pytest.approx(4.0 * math.pi * r_tot * r_tot, rel=1e-6)
    assert s_single["sasa_polar_nm2"] == pytest.approx(s_single["sasa_total_nm2"], rel=1e-6)

    # Overlapping dimer has strictly smaller SASA than 2 isolated atoms
    s_dimer = compute_sasa_shrake_rupley(np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]), ["O", "C"], probe_radius_nm=probe)
    r_c = BONDI_VDW_NM["C"] + probe
    assert s_dimer["sasa_total_nm2"] < 4.0 * math.pi * (r_tot ** 2 + r_c ** 2)

    # Exact trans (180 deg) and cis (0 deg) dihedral
    trans = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, -1.0, 0.0]]])
    cis = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    assert abs(abs(compute_dihedral_deg(trans, (0, 1, 2, 3))[0]) - 180.0) < 1e-6
    assert abs(compute_dihedral_deg(cis, (0, 1, 2, 3))[0]) < 1e-6
    assert compute_radius_of_gyration(trans[0]) > 0.0

    fes = compute_dihedral_fes_1d(np.concatenate([np.full(80, 175.0), np.full(20, 60.0)]), temperature_k=300.0)
    assert float(np.nanmin(fes["fes_kcal_mol"])) == pytest.approx(0.0, abs=1e-9)

    # PCA conformational clustering on 2 distinct conformations
    rng = np.random.default_rng(11)
    c_a = trans[0][None, :, :] + rng.normal(0.0, 0.01, size=(12, 4, 3))
    c_b = cis[0][None, :, :] + rng.normal(0.0, 0.01, size=(12, 4, 3))
    pca_res = compute_conformational_pca_clusters(np.vstack([c_a, c_b]), [0, 1, 2, 3], n_clusters=2)
    assert pca_res["pc1_var_ratio"] > 0.90

    # Exact 3x3x3 simple cubic lattice -> Voronoi volume = 1.0 nm^3, 6 neighbors each
    pts = np.array([[ix + 0.5, iy + 0.5, iz + 0.5] for ix in range(3) for iy in range(3) for iz in range(3)], dtype=float)
    vor = compute_periodic_voronoi(pts, np.array([3.0, 3.0, 3.0]), grid_n=12)
    assert vor["mean_voronoi_vol_nm3"] == pytest.approx(1.0, rel=1e-5)
    assert vor["mean_voronoi_neighbors"] == pytest.approx(6.0, rel=1e-5)


def test_interface_slab_surface_tension_and_tanh_fit() -> None:
    st = compute_surface_tension(np.full(50, -72.0), np.full(50, -72.0), np.full(50, 72.0), lz_nm=10.0)
    assert st["gamma_mn_m"] == pytest.approx(72.0, rel=1e-6)

    z = np.linspace(0.0, 10.0, 201)
    rho_true = 0.5 * 33.0 * (1.0 - np.tanh(2.0 * (np.abs(z - 5.0) - 2.0) / 0.4))
    fit = fit_slab_tanh_profile(z, rho_true)
    assert fit["rho_bulk_liquid"] == pytest.approx(33.0, rel=1e-3)
    assert fit["gibbs_half_thickness_nm"] == pytest.approx(2.0, rel=1e-3)
    assert fit["interfacial_width_nm"] == pytest.approx(0.4, rel=1e-3)
    assert fit["fit_r_squared"] > 0.999

    # Orientation profile: molecules aligned along z -> P2 = 1.0
    z_coms = np.full((10, 20), 5.0)
    u_z = np.tile(np.array([0.0, 0.0, 1.0]), (10, 20, 1))
    prof = compute_z_density_and_orientation_profiles(z_coms, u_z, np.tile([3.0, 3.0, 10.0], (10, 1)), n_bins=20)
    valid_p2 = prof["p2_orientation"].dropna()
    assert np.allclose(valid_p2.to_numpy(), 1.0, atol=1e-9)


def test_alchemical_ti_bar_mbar_exact_harmonic_oscillators(tmp_path: Path) -> None:
    """1D shifted harmonic oscillators U_k(x) = 0.5*(x - lam_k)^2 + c_k have exact ΔF_k = c_k - c_0."""
    rng = np.random.default_rng(2026)
    T = 300.0
    rt_kj = 8.314462618e-3 * T
    lambdas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    c_shifts_kt = np.array([0.0, 0.8, 1.9, 3.1, 4.5])  # exact ΔF = 4.5 kT
    n_per = 1500
    K = len(lambdas)

    xs = [rng.normal(loc=lam, scale=1.0, size=n_per) for lam in lambdas]
    x_all = np.concatenate(xs)
    u_kn = np.zeros((K, K * n_per), dtype=float)
    for k in range(K):
        u_kn[k] = 0.5 * ((x_all - lambdas[k]) ** 2) + c_shifts_kt[k]

    # Direct BAR on adjacent pair 0->1 (exact df = 0.8 kT)
    wf = u_kn[1, :n_per] - u_kn[0, :n_per]
    wr = u_kn[0, n_per : 2 * n_per] - u_kn[1, n_per : 2 * n_per]
    df_bar, err_bar = compute_bar_pair(wf, wr)
    assert df_bar == pytest.approx(0.8, abs=0.06)
    assert err_bar > 0.0

    # Global MBAR across all 5 states (exact df_total = 4.5 kT)
    mbar = compute_mbar(u_kn, np.full(K, n_per))
    assert mbar["converged"] is True
    assert mbar["df_total_kt"] == pytest.approx(4.5, abs=0.08)
    assert mbar["min_neighbor_overlap"] > 0.1

    # TI on exact polynomial <dH/dlam> = 10 * lam -> integral 0..1 = 5.0 kJ/mol
    ti = compute_ti_free_energy(lambdas, 10.0 * lambdas, np.full(K, 0.05))
    assert ti["dg_ti_trap_kj_mol"] == pytest.approx(5.0, abs=1e-6)
    assert ti["dg_ti_spline_kj_mol"] == pytest.approx(5.0, abs=1e-6)

    # End-to-end synthetic GROMACS dhdl.xvg parser & analyzer
    dhdl_paths = []
    for k, lam in enumerate(lambdas):
        wdir = tmp_path / f"lambda_{k:02d}"
        wdir.mkdir()
        p = wdir / "dhdl.xvg"
        lines = [
            f'@ subtitle "T = {T} (K), lambda = {lam:.2f}"',
            '@ s0 legend "dH/dlambda"',
        ]
        for j in range(K):
            lines.append(f'@ s{j + 1} legend "Delta H to lambda {j}"')
        xk = xs[k]
        u_self = 0.5 * ((xk - lam) ** 2) + c_shifts_kt[k]
        for idx_s in range(n_per):
            # dU/dlam for c(lam) ≈ 4.5*lam and 0.5*(x-lam)^2 -> -(x-lam) + 4.5 (in kT)
            dhdl_kj = (-(xk[idx_s] - lam) + 4.5) * rt_kj
            dhs = [(0.5 * ((xk[idx_s] - lambdas[j]) ** 2) + c_shifts_kt[j] - u_self[idx_s]) * rt_kj for j in range(K)]
            lines.append(f"{idx_s * 0.5:.2f} {dhdl_kj:.6f} " + " ".join(f"{v:.6f}" for v in dhs))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        dhdl_paths.append(p)

    fep_res = analyze_dhdl_window_set(dhdl_paths, temperature_k=T, equil_fraction=0.0)
    expected_kj = 4.5 * rt_kj
    assert fep_res["summary"]["dg_mbar_kj_mol"] == pytest.approx(expected_kj, rel=0.04)
    assert fep_res["summary"]["dg_bar_kj_mol"] == pytest.approx(expected_kj, rel=0.04)
    assert fep_res["summary"]["dg_ti_trap_kj_mol"] == pytest.approx(expected_kj, rel=0.05)
    assert fep_res["summary"]["overlap_qc_status"] == "PASS"


# ---------------------------------------------------------------------------
# 4. General Triclinic PBC & Cromer-Mann Atomistic WAXS/SAXS
# ---------------------------------------------------------------------------

def test_triclinic_pbc_rhombic_dodecahedron_voronoi_and_subvolume() -> None:
    # Standard GROMACS rhombic dodecahedron: a=b=c=d, alpha=60, beta=60, gamma=90 -> V = d^3 / sqrt(2)
    d = 4.0
    box_rd = np.array([d, d, d, 60.0, 60.0, 90.0])
    vol = box_volume_nm3(box_rd)
    assert vol == pytest.approx((d ** 3) / math.sqrt(2.0), rel=1e-6)
    # Inscribed sphere radius of rhombic dodecahedron with lattice vector length d is d / 2 * sqrt(2/3) ...
    # Wait: shortest perpendicular height between parallel (100)/(010)/(001) planes = d * sqrt(2/3) or d/sqrt(2) -> R_in = 0.5 * d / sqrt(2)
    r_in = box_inscribed_radius_nm(box_rd)
    assert r_in == pytest.approx(0.5 * d / math.sqrt(2.0), rel=1e-6)

    H = box_matrix_from_dimensions(box_rd)
    # A displacement equal to H[0] + H[2] + small epsilon [0.05, -0.04, 0.03] must reduce exactly to epsilon!
    eps = np.array([[0.05, -0.04, 0.03]])
    shifted = eps + H[0] + H[2]
    mic = minimum_image_displacement(shifted, box_rd)
    assert np.allclose(mic, eps, atol=1e-10)

    # Periodic Voronoi on a 2x2x2 triclinic sub-lattice (8 points in rhombic dodecahedron) -> exact volume = V / 8
    f1d = np.array([0.25, 0.75])
    fpts = np.stack(np.meshgrid(f1d, f1d, f1d, indexing="ij"), axis=-1).reshape(-1, 3)
    pts_triclinic = wrap_positions(fpts @ H, box_rd)
    vor = compute_periodic_voronoi(pts_triclinic, box_rd, grid_n=10)
    assert vor["mean_voronoi_vol_nm3"] == pytest.approx(vol / 8.0, rel=1e-5)

    # Subvolume fluctuation KBI runs cleanly on triclinic box without error
    rng = np.random.default_rng(5)
    frac_pos = rng.random((15, 40, 3))
    coms_tric = {"A": np.einsum("fmi,ij->fmj", frac_pos, H)}
    sv_df = compute_subvolume_fluctuation_kbi(coms_tric, box_rd, n_spheres=24, seed=5)
    assert len(sv_df) == 1
    assert np.isfinite(sv_df["g_subvol_rmax_cm3_mol"].iloc[0])


def test_cromer_mann_form_factors_and_waxs_iq() -> None:
    q = np.linspace(0.0, 60.0, 61)  # nm^-1
    expected_z = {"H": 1, "C": 6, "N": 7, "O": 8, "NA": 11, "CL": 17}
    for el, z in expected_z.items():
        f_q = cromer_mann_form_factor(el, q)
        assert f_q[0] == pytest.approx(z, abs=0.02)
        # X-ray form factor monotonically decreases with q
        assert np.all(np.diff(f_q) <= 1e-6)
        assert f_q[-1] < f_q[0]

    # Water molecule H2O has f(q=0) ≈ 10 electrons
    f_h2o = species_cromer_mann_form_factor(["O1", "H1", "H2"], q)
    assert f_h2o[0] == pytest.approx(10.0, abs=0.02)

    # Full atomistic WAXS with ideal gas g(r)=1 yields exact I_WAXS(q) = 1.0 at all q
    r = np.linspace(0.0, 2.5, 126)
    g_atom = {(0, 0): np.ones_like(r), (0, 1): np.ones_like(r), (1, 1): np.ones_like(r)}
    waxs = compute_waxs_atomistic_iq(r, g_atom, np.array([33.0, 66.0]), ["O", "H"], q)
    assert np.allclose(waxs["I_total"], 1.0, atol=1e-10)
