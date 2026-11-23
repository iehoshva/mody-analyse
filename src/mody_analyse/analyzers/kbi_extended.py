from __future__ import annotations

import math
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid

from mody_analyse.analyzers.kbi_analyzer import (
    AVOGADRO,
    NM3_TO_CM3_MOL,
    apply_gvdv_correction,
    compute_running_kbi,
    rebin_rdf,
)
from mody_analyse.core.config import Config
from mody_analyse.core.discovery import (
    RunRecord,
    compute_whole_coms_vectorized,
    equilibrated_frame_slice,
    read_xvg_dataframe,
)

KB_J_K = 1.380649e-23
R_KJ_MOL_K = 8.314462618e-3  # kJ / (mol K)
R_J_MOL_K = 8.314462618


def dawass_extrapolate(
    r_nm: np.ndarray,
    g_r: np.ndarray,
    r_min: float | None = None,
    r_max: float | None = None,
    n_radii: int = 12,
) -> dict[str, Any]:
    """Dawass et al. (2018/2020) finite-volume spherical KBI extrapolation: G^V(R) = G_inf + c / R.

    Uses the exact spherical subvolume weight w(x) = 1 - 1.5 x + 0.5 x^3 (x = r / (2R))
    and fits G^V(R) linearly against 1/R over R in [r_min, r_max], returning the intercept
    G_inf as 1/R -> 0 along with R^2 and convergence diagnostics.
    """
    r = np.asarray(r_nm, dtype=float)
    g = np.asarray(g_r, dtype=float)
    r_max_avail = float(r[-1]) if len(r) else 0.0
    if len(r) < 5 or r_max_avail <= 0:
        return {
            "g_inf_cm3_mol": float("nan"),
            "slope_surface_cm3_nm_mol": float("nan"),
            "r_squared": float("nan"),
            "std_err_cm3_mol": float("nan"),
            "convergence_status": "FAIL",
        }

    upper_r = min(float(r_max) if r_max is not None else 0.45 * r_max_avail, 0.49 * r_max_avail)
    lower_r = min(float(r_min) if r_min is not None else 0.25 * r_max_avail, upper_r * 0.7)
    if lower_r <= 0 or upper_r <= lower_r:
        g_run = compute_running_kbi(r, g)
        return {
            "g_inf_cm3_mol": float(g_run[-1]),
            "slope_surface_cm3_nm_mol": 0.0,
            "r_squared": 0.0,
            "std_err_cm3_mol": float("nan"),
            "convergence_status": "MARGINAL",
        }

    radii = np.linspace(lower_r, upper_r, max(4, int(n_radii)))
    gv_vals: list[float] = []
    inv_r_vals: list[float] = []
    for R in radii:
        mask = r <= 2.0 * R
        if int(np.sum(mask)) < 4:
            continue
        rm = r[mask]
        x = rm / (2.0 * R)
        w = 1.0 - 1.5 * x + 0.5 * (x ** 3)
        gv_nm3 = 4.0 * math.pi * float(trapezoid((g[mask] - 1.0) * (rm ** 2) * w, rm))
        gv_vals.append(gv_nm3 * NM3_TO_CM3_MOL)
        inv_r_vals.append(1.0 / R)

    if len(gv_vals) < 3:
        g_run = compute_running_kbi(r, g)
        return {
            "g_inf_cm3_mol": float(g_run[-1]),
            "slope_surface_cm3_nm_mol": 0.0,
            "r_squared": 0.0,
            "std_err_cm3_mol": float("nan"),
            "convergence_status": "FAIL",
        }

    x_arr = np.asarray(inv_r_vals, dtype=float)
    y_arr = np.asarray(gv_vals, dtype=float)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    y_pred = intercept + slope * x_arr
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 1.0
    dof = max(1, len(y_arr) - 2)
    s2 = ss_res / dof
    x_var = float(np.sum((x_arr - np.mean(x_arr)) ** 2))
    se_intercept = math.sqrt(s2 * (1.0 / len(y_arr) + (float(np.mean(x_arr)) ** 2) / max(x_var, 1e-18)))

    # Flat constant y (e.g., ideal gas G=0) has ss_tot ≈ 0 -> perfect convergence
    if ss_tot <= 1e-12 or (r2 >= 0.90 and se_intercept <= max(25.0, 0.20 * abs(float(intercept)) + 5.0)):
        status = "PASS"
    elif r2 >= 0.70:
        status = "MARGINAL"
    else:
        status = "FAIL"

    return {
        "g_inf_cm3_mol": float(intercept),
        "slope_surface_cm3_nm_mol": float(slope),
        "r_squared": float(r2),
        "std_err_cm3_mol": float(se_intercept),
        "convergence_status": status,
    }


def compute_subvolume_fluctuation_kbi(
    coms_by_species: dict[str, np.ndarray],
    box_nm: np.ndarray,
    *,
    n_spheres: int = 64,
    r_fracs: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35),
    seed: int = 42,
) -> pd.DataFrame:
    """Direct particle-number fluctuation KBI in spherical subvolumes (independent of RDF binning).

    coms_by_species: mapping species -> array of shape (n_frames, n_mol, 3) in nm.
    box_nm: array of shape (n_frames, 3) or (3,) in nm.
    """
    species = sorted(coms_by_species.keys())
    if not species:
        return pd.DataFrame()
    n_frames = min(coms_by_species[s].shape[0] for s in species)
    if n_frames == 0:
        return pd.DataFrame()

    box_arr = np.asarray(box_nm, dtype=float)
    if box_arr.ndim == 1:
        box_arr = np.broadcast_to(box_arr[None, :], (n_frames, 3))
    l_min = float(np.min(box_arr[:, :3]))
    radii = [f * l_min for f in r_fracs if 0.0 < f < 0.49]
    if not radii:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    # Sample uniform fractional sphere centers once per frame
    frac_centers = rng.random((n_frames, int(n_spheres), 3))

    # Collect N_i(frame, sphere) for each radius
    counts_by_r: dict[float, dict[str, np.ndarray]] = {R: {} for R in radii}
    for sp in species:
        pos = np.asarray(coms_by_species[sp][:n_frames], dtype=float)  # (F, M, 3)
        for R in radii:
            counts_by_r[R][sp] = np.zeros((n_frames, int(n_spheres)), dtype=float)
        for fi in range(n_frames):
            L = box_arr[fi, :3]
            centers = frac_centers[fi] * L[None, :]  # (S, 3)
            # Minimal-image distance between centers (S, 1, 3) and pos[fi] (1, M, 3)
            diff = pos[fi][None, :, :] - centers[:, None, :]
            diff -= L[None, None, :] * np.round(diff / L[None, None, :])
            d2 = np.sum(diff * diff, axis=-1)  # (S, M)
            for R in radii:
                counts_by_r[R][sp][fi] = np.sum(d2 <= (R * R), axis=-1)

    vol_box_mean = float(np.mean(box_arr[:, 0] * box_arr[:, 1] * box_arr[:, 2]))
    rows: list[dict[str, Any]] = []
    for i, sp_i in enumerate(species):
        n_mol_i = max(1, coms_by_species[sp_i].shape[1])
        for j in range(i, len(species)):
            sp_j = species[j]
            inv_r: list[float] = []
            g_vals: list[float] = []
            for R in radii:
                vol_r = (4.0 * math.pi / 3.0) * (R ** 3)
                ni = counts_by_r[R][sp_i].ravel()
                nj = counts_by_r[R][sp_j].ravel()
                mi = float(np.mean(ni))
                mj = float(np.mean(nj))
                if mi <= 1e-12 or mj <= 1e-12:
                    continue
                cov = float(np.mean((ni - mi) * (nj - mj)))
                delta_ij = 1.0 if sp_i == sp_j else 0.0
                g_raw_nm3 = vol_r * (cov / (mi * mj) - delta_ij / mi)
                # Closed-box canonical (NVT) finite-size correction (Cortes-Huerto / Schnell):
                # G_NVT(R) = G_inf * (1 - V_R/V_box) - (delta_ij / rho_i) * (V_R / V_box)
                frac_v = min(0.95, vol_r / max(vol_box_mean, 1e-12))
                g_nm3 = (g_raw_nm3 + delta_ij * (vol_r / n_mol_i)) / max(1.0 - frac_v, 1e-6)
                g_vals.append(g_nm3 * NM3_TO_CM3_MOL)
                inv_r.append(1.0 / R)
            if len(g_vals) >= 2:
                slope, intercept = np.polyfit(np.asarray(inv_r), np.asarray(g_vals), 1)
                g_inf = float(intercept)
            elif g_vals:
                slope, g_inf = 0.0, float(g_vals[-1])
            else:
                slope, g_inf = float("nan"), float("nan")
            rows.append(
                {
                    "species_i": sp_i,
                    "species_j": sp_j,
                    "g_subvol_inf_cm3_mol": g_inf,
                    "g_subvol_rmax_cm3_mol": float(g_vals[-1]) if g_vals else float("nan"),
                    "slope_inv_r": float(slope),
                    "n_radii": len(g_vals),
                }
            )
    return pd.DataFrame(rows)


def compute_kb_thermodynamics_full(
    G_cm3_mol: np.ndarray,
    rho_nm3: np.ndarray,
    temperature_k: float,
    species_names: list[str] | None = None,
) -> dict[str, Any]:
    """Exact N-component Kirkwood–Buff thermodynamic inversion, Bhatia–Thornton & preferential solvation.

    Uses B_ij = rho_i delta_ij + rho_i rho_j G_ij (nm^-3) and A = B^-1 (nm^3):
      - kappa_T = 1 / (k_B T * sum_{i,j} rho_i A_ij rho_j)
      - V_bar_i = (sum_j A_ij rho_j) / (sum_{k,l} rho_k A_kl rho_l)
      - S_ij(0) = delta_ij + sqrt(rho_i rho_j) G_ij
      - Bhatia–Thornton S_NN(0), S_CC(0), S_NC(0) for binary / primary pair projection
      - Preferential solvation Delta N_ij = rho_j G_ij
    """
    G_nm3 = np.asarray(G_cm3_mol, dtype=float) / NM3_TO_CM3_MOL
    rho = np.asarray(rho_nm3, dtype=float)
    n_comp = len(rho)
    names = species_names or [f"sp_{i}" for i in range(n_comp)]
    rho_tot = float(np.sum(rho))
    x = rho / max(rho_tot, 1e-30)

    B = np.diag(rho) + np.outer(rho, rho) * G_nm3
    A = np.linalg.pinv(B)
    eta = float(rho @ A @ rho)  # nm^-3
    # Convert eta (nm^-3) * k_B T (J) -> Pa: 1 nm^-3 = 1e27 m^-3
    denom_pa = eta * 1e27 * KB_J_K * float(temperature_k)
    kappa_t_pa_inv = 1.0 / denom_pa if denom_pa > 0 else float("nan")
    kappa_t_gpa_inv = kappa_t_pa_inv * 1e9

    v_bar_nm3 = (A @ rho) / eta if abs(eta) > 1e-30 else np.full(n_comp, float("nan"))
    v_bar_cm3_mol = v_bar_nm3 * NM3_TO_CM3_MOL

    # Constant-(T,P) thermodynamic factor matrix Gamma_ij = delta_ij + x_i (d ln gamma_i / d x_j)_{T,P}
    # Using exact Ben-Naim relation:
    # (1/kBT) (d mu_i / d N_j)_{T,P} * V = A_ij - (sum_k A_ik rho_k)(sum_l A_jl rho_l) / eta
    ar = A @ rho
    M = A - np.outer(ar, ar) / max(eta, 1e-30)  # nm^3
    # For independent composition variables (j = 0..n-2), d mu_i / d x_j = rho_tot * (M_ij - M_{i,n-1})
    gamma_mat = np.zeros((n_comp, n_comp), dtype=float)
    for i in range(n_comp):
        for j in range(n_comp):
            gamma_mat[i, j] = float(x[i] * rho_tot * M[i, j])

    # Partial static structure factor S_ij(q -> 0)
    sqrt_rho = np.sqrt(np.maximum(rho, 0.0))
    S0_partial = np.eye(n_comp) + np.outer(sqrt_rho, sqrt_rho) * G_nm3

    # Bhatia–Thornton q->0 (exact for binary, primary pair 0-1 projection for N>2)
    if n_comp >= 2:
        g11, g22, g12 = float(G_nm3[0, 0]), float(G_nm3[1, 1]), float(G_nm3[0, 1])
        x1, x2 = float(x[0]), float(x[1])
        gamma_binary = 1.0 / (1.0 + x1 * x2 * rho_tot * (g11 + g22 - 2.0 * g12)) if abs(1.0 + x1 * x2 * rho_tot * (g11 + g22 - 2.0 * g12)) > 1e-18 else float("nan")
        s_cc_0 = (x1 * x2) / gamma_binary if np.isfinite(gamma_binary) and abs(gamma_binary) > 1e-18 else float("nan")
        dilatation = rho_tot * float(v_bar_nm3[0] - v_bar_nm3[1])
        s_nc_0 = -dilatation * s_cc_0
        s_nn_0 = (rho_tot * 1e27 * KB_J_K * float(temperature_k) * kappa_t_pa_inv) + (dilatation ** 2) * s_cc_0
    else:
        gamma_binary = 1.0
        s_cc_0 = 0.0
        s_nc_0 = 0.0
        s_nn_0 = float(S0_partial[0, 0])

    # Excess coordination / preferential solvation Delta N_ij = rho_j G_ij
    delta_n = np.zeros((n_comp, n_comp), dtype=float)
    for i in range(n_comp):
        for j in range(n_comp):
            delta_n[i, j] = float(rho[j] * G_nm3[i, j])

    return {
        "species": names,
        "mole_fractions": x.tolist(),
        "kappa_t_pa_inv": float(kappa_t_pa_inv),
        "kappa_t_gpa_inv": float(kappa_t_gpa_inv),
        "pmv_cm3_mol": {names[i]: float(v_bar_cm3_mol[i]) for i in range(n_comp)},
        "gamma_matrix": gamma_mat,
        "gamma_binary": float(gamma_binary),
        "S0_partial": S0_partial,
        "S_NN_0": float(s_nn_0),
        "S_CC_0": float(s_cc_0),
        "S_NC_0": float(s_nc_0),
        "delta_n_matrix": delta_n,
    }


def compute_structure_factor_sq(
    r_nm: np.ndarray,
    g_pairs: dict[tuple[int, int], np.ndarray],
    rho_nm3: np.ndarray,
    q_inv_nm: np.ndarray,
    *,
    form_factors: np.ndarray | None = None,
    use_lorch: bool = True,
) -> dict[str, Any]:
    """Compute partial S_ij(q), Bhatia–Thornton S_NN/S_CC/S_NC(q), and total SAXS/SANS intensity I(q).

    S_ij(q) = delta_ij + 4 pi sqrt(rho_i rho_j) integral_0^{r_max} r^2 (g_ij(r) - 1) [sin(qr)/(qr)] W(r) dr
    where W(r) = sin(pi r / r_max) / (pi r / r_max) is the Lorch modification function.
    """
    r = np.asarray(r_nm, dtype=float)
    q = np.asarray(q_inv_nm, dtype=float)
    rho = np.asarray(rho_nm3, dtype=float)
    n_comp = len(rho)
    n_q = len(q)
    r_max = float(r[-1]) if len(r) else 1.0

    if use_lorch and r_max > 0:
        arg = math.pi * r / r_max
        lorch = np.where(arg > 1e-12, np.sin(arg) / np.where(arg > 1e-12, arg, 1.0), 1.0)
    else:
        lorch = np.ones_like(r)

    qr = np.outer(q, r)  # (n_q, n_r)
    sinc_qr = np.where(qr > 1e-12, np.sin(qr) / np.where(qr > 1e-12, qr, 1.0), 1.0)

    S_partial = np.zeros((n_comp, n_comp, n_q), dtype=float)
    for i in range(n_comp):
        for j in range(i, n_comp):
            g_ij = g_pairs.get((i, j))
            if g_ij is None:
                g_ij = g_pairs.get((j, i))
            if g_ij is None:
                g_ij = np.ones_like(r)
            pref = 4.0 * math.pi * math.sqrt(max(rho[i] * rho[j], 0.0))
            integrand = ((g_ij - 1.0) * (r ** 2) * lorch)[None, :] * sinc_qr
            val = (1.0 if i == j else 0.0) + pref * trapezoid(integrand, r, axis=1)
            S_partial[i, j, :] = val
            S_partial[j, i, :] = val

    # Total SAXS/SANS intensity I(q) = sum_{i,j} f_i f_j sqrt(x_i x_j) S_ij(q) / sum_i x_i f_i^2
    rho_tot = float(np.sum(rho))
    x = rho / max(rho_tot, 1e-30)
    f = np.ones(n_comp, dtype=float) if form_factors is None else np.asarray(form_factors, dtype=float)
    norm = float(np.sum(x * (f ** 2)))
    I_q = np.zeros(n_q, dtype=float)
    for i in range(n_comp):
        for j in range(n_comp):
            I_q += f[i] * f[j] * math.sqrt(max(x[i] * x[j], 0.0)) * S_partial[i, j, :]
    if norm > 0:
        I_q = I_q / norm

    # Bhatia–Thornton q-dependent for binary
    bt: dict[str, np.ndarray] = {}
    if n_comp >= 2:
        x1, x2 = float(x[0]), float(x[1])
        s11, s22, s12 = S_partial[0, 0], S_partial[1, 1], S_partial[0, 1]
        bt["S_NN"] = x1 * s11 + x2 * s22 + 2.0 * math.sqrt(x1 * x2) * s12
        bt["S_CC"] = x1 * x2 * (x2 * s11 + x1 * s22 - 2.0 * math.sqrt(x1 * x2) * s12)
        bt["S_NC"] = x1 * x2 * (s11 - s22 + (x2 - x1) / max(math.sqrt(x1 * x2), 1e-12) * s12)

    return {
        "q_inv_nm": q,
        "S_partial": S_partial,
        "I_total": I_q,
        **bt,
    }


def integrate_composition_series_kbi(
    comp_df: pd.DataFrame,
    *,
    temperature_k: float = 298.15,
    rk_order: int = 3,
) -> pd.DataFrame:
    """Integrate binary thermodynamic factor Gamma(x1) across compositions to obtain ln(gamma1), ln(gamma2), GE.

    Expects columns: 'x1' (mole fraction of component 1 in [0, 1]) and 'gamma_factor' (Gamma(x1)).
    Fits a Redlich–Kister expansion for GE/(RT) = x1(1-x1) * sum_{k=0}^{p-1} A_k (2x1 - 1)^k so that:
      Gamma(x1) = 1 + x1(1-x1) * d^2(GE/RT)/dx1^2
    which is automatically Gibbs–Duhem consistent and exact at x1=0 and x1=1!
    Also computes direct trapezoid quadrature of (Gamma - 1)/x1 and (1 - Gamma)/(1 - x1) for cross-check.
    """
    if comp_df.empty or len(comp_df) < 2:
        return pd.DataFrame()
    df = comp_df.sort_values("x1").reset_index(drop=True).copy()
    x1 = df["x1"].to_numpy(dtype=float)
    gam = df["gamma_factor"].to_numpy(dtype=float)
    rt_kj = R_KJ_MOL_K * float(temperature_k)

    # Basis functions for d^2/dx^2 [x(1-x)(2x-1)^k] evaluated numerically/analytically for Redlich-Kister fit
    interior = (x1 > 1e-6) & (x1 < 1.0 - 1e-6) & np.isfinite(gam)
    n_int = int(np.sum(interior))
    order = max(1, min(int(rk_order), n_int))

    def _rk_g_basis(x: np.ndarray, k: int) -> np.ndarray:
        return x * (1.0 - x) * ((2.0 * x - 1.0) ** k)

    def _rk_d2g_basis(x: np.ndarray, k: int) -> np.ndarray:
        # Exact second derivative of f_k(x) = x(1-x)(2x-1)^k using u = 2x - 1, x(1-x) = (1 - u^2)/4
        # df/du = (1/4)[ k u^{k-1} - (k+2) u^{k+1} ] * 2 -> d^2f/dx^2 = 4 d^2f/du^2
        # d^2[(1-u^2) u^k]/du^2 = k(k-1) u^{k-2} - (k+2)(k+1) u^k
        u = 2.0 * x - 1.0
        term1 = k * (k - 1) * (u ** (k - 2)) if k >= 2 else np.zeros_like(u)
        term2 = (k + 2) * (k + 1) * (u ** k)
        return term1 - term2

    def _rk_dg_basis(x: np.ndarray, k: int) -> np.ndarray:
        # df/dx = 2 * (1/4) * d[(1-u^2)u^k]/du = 0.5 * (k u^{k-1} - (k+2) u^{k+1})
        u = 2.0 * x - 1.0
        term1 = k * (u ** (k - 1)) if k >= 1 else np.zeros_like(u)
        term2 = (k + 2) * (u ** (k + 1))
        return 0.5 * (term1 - term2)

    if n_int >= 1:
        xi = x1[interior]
        yi = gam[interior] - 1.0
        design = np.column_stack([xi * (1.0 - xi) * _rk_d2g_basis(xi, k) for k in range(order)])
        coeffs, *_ = np.linalg.lstsq(design, yi, rcond=None)
    else:
        coeffs = np.zeros(order, dtype=float)

    ge_rt = np.zeros_like(x1)
    dge_rt_dx = np.zeros_like(x1)
    gamma_fit = np.ones_like(x1)
    for k, ak in enumerate(coeffs):
        ge_rt += ak * _rk_g_basis(x1, k)
        dge_rt_dx += ak * _rk_dg_basis(x1, k)
        gamma_fit += ak * x1 * (1.0 - x1) * _rk_d2g_basis(x1, k)

    # Exact partial molar excess properties:
    # ln gamma_1 = GE/RT + (1 - x1) d(GE/RT)/dx1
    # ln gamma_2 = GE/RT - x1 d(GE/RT)/dx1
    ln_gamma1 = ge_rt + (1.0 - x1) * dge_rt_dx
    ln_gamma2 = ge_rt - x1 * dge_rt_dx

    # Direct quadrature cross-check (anchored at endpoints 0 and 1 if covered, else nearest boundary)
    integrand1 = np.where(x1 > 1e-9, (gam - 1.0) / np.maximum(x1, 1e-9), 0.0)
    integrand2 = np.where(1.0 - x1 > 1e-9, (1.0 - gam) / np.maximum(1.0 - x1, 1e-9), 0.0)
    # Integral from x1=1 downwards for component 1, from x1=0 upwards for component 2
    ln_g1_quad = cumulative_trapezoid(integrand1, x1, initial=0.0)
    ln_g1_quad = ln_g1_quad - ln_g1_quad[-1]
    ln_g2_quad = cumulative_trapezoid(integrand2, x1, initial=0.0)
    gd_residual = float(np.sqrt(np.mean((gam - gamma_fit) ** 2)))

    out = df.copy()
    out["gamma_rk_fit"] = gamma_fit
    out["ln_gamma_1"] = ln_gamma1
    out["ln_gamma_2"] = ln_gamma2
    out["gamma_1"] = np.exp(ln_gamma1)
    out["gamma_2"] = np.exp(ln_gamma2)
    out["g_excess_kj_mol"] = ge_rt * rt_kj
    out["mu_ex_1_kj_mol"] = ln_gamma1 * rt_kj
    out["mu_ex_2_kj_mol"] = ln_gamma2 * rt_kj
    out["ln_gamma_1_quad"] = ln_g1_quad
    out["ln_gamma_2_quad"] = ln_g2_quad
    out["gibbs_duhem_fit_rmse"] = gd_residual
    return out


def compute_setschenow_constant(
    cosolvent_conc_mol_l: np.ndarray,
    ln_solubility_ratio: np.ndarray,
) -> dict[str, float]:
    """Fit Setschenow equation: ln(S_0 / S) = K_S * c_cosolvent (positive = salting-out, negative = cosolvency)."""
    c = np.asarray(cosolvent_conc_mol_l, dtype=float)
    y = np.asarray(ln_solubility_ratio, dtype=float)
    if len(c) < 2:
        return {"K_setschenow_L_mol": float("nan"), "intercept": float("nan"), "r_squared": float("nan")}
    slope, intercept = np.polyfit(c, y, 1)
    y_pred = intercept + slope * c
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 1.0
    return {"K_setschenow_L_mol": float(slope), "intercept": float(intercept), "r_squared": float(r2)}


def compute_temperature_series_gibbs_helmholtz(temp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute excess enthalpy HE and entropy TSE from G^E(T) across temperatures.

    Uses exact Gibbs–Helmholtz relation:
      H^E = d(G^E / T) / d(1/T)  and  T S^E = H^E - G^E
    Expects columns: 'temperature_k' and 'g_excess_kj_mol' (optional grouping column 'composition_key').
    """
    if temp_df.empty:
        return pd.DataFrame()
    df = temp_df.copy()
    if "composition_key" not in df.columns:
        df["composition_key"] = "default"
    out_rows: list[dict[str, Any]] = []
    for key, grp in df.groupby("composition_key", sort=True):
        g = grp.sort_values("temperature_k")
        T = g["temperature_k"].to_numpy(dtype=float)
        ge = g["g_excess_kj_mol"].to_numpy(dtype=float)
        if len(T) < 2:
            continue
        inv_t = 1.0 / T
        ge_over_t = ge / T
        if len(T) == 2:
            he_val = float((ge_over_t[1] - ge_over_t[0]) / (inv_t[1] - inv_t[0]))
            he_arr = np.full_like(T, he_val)
        else:
            # Fit linear or quadratic in (1/T) for smooth derivative
            deg = min(2, len(T) - 1)
            p = np.polyfit(inv_t, ge_over_t, deg)
            dp = np.polyder(p)
            he_arr = np.polyval(dp, inv_t)
        for ti, gei, hei in zip(T, ge, he_arr):
            out_rows.append(
                {
                    "composition_key": key,
                    "temperature_k": float(ti),
                    "g_excess_kj_mol": float(gei),
                    "h_excess_kj_mol": float(hei),
                    "ts_excess_kj_mol": float(hei - gei),
                    "s_excess_j_mol_k": float((hei - gei) * 1000.0 / ti),
                }
            )
    return pd.DataFrame(out_rows)


def _species_electron_counts(run: RunRecord) -> dict[str, float]:
    """Approximate Z_i (total electron count per molecule) for SAXS form factors at q->0."""
    z_map = {
        "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11, "MG": 12,
        "P": 15, "S": 16, "CL": 17, "K": 19, "CA": 20, "BR": 35, "I": 53,
    }
    out: dict[str, float] = {}
    for res, topo in run.topologies.items():
        z_tot = 0.0
        for name in topo.atom_names:
            sym = "".join(ch for ch in name if ch.isalpha()).upper()
            if sym[:2] in z_map:
                z_tot += z_map[sym[:2]]
            elif sym[:1] in z_map:
                z_tot += z_map[sym[:1]]
            else:
                z_tot += 6.0
        out[res] = max(z_tot, 1.0)
    return out


def _compute_pair_rdf_from_coms(
    coms_i: np.ndarray,
    coms_j: np.ndarray,
    boxes_nm: np.ndarray,
    r_max: float,
    dr: float,
    is_like: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast vectorized pair RDF from whole-molecule COMs (n_frames, n_mol, 3) in nm."""
    edges = np.arange(0.0, r_max + dr * 0.5, dr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(len(centers), dtype=float)
    n_frames = len(coms_i)
    ni, nj = coms_i.shape[1], coms_j.shape[1]
    if n_frames == 0 or ni == 0 or nj == 0:
        return centers, np.ones_like(centers)

    vol_sum = 0.0
    for fi in range(n_frames):
        L = boxes_nm[fi, :3]
        vol_sum += float(L[0] * L[1] * L[2])
        diff = coms_j[fi][None, :, :] - coms_i[fi][:, None, :]
        diff -= L[None, None, :] * np.round(diff / L[None, None, :])
        dist = np.linalg.norm(diff, axis=-1)
        if is_like:
            mask = np.triu(np.ones((ni, nj), dtype=bool), k=1)
            dvals = dist[mask]
            counts, _ = np.histogram(dvals, bins=edges)
            hist += 2.0 * counts
        else:
            counts, _ = np.histogram(dist.ravel(), bins=edges)
            hist += counts

    vol_avg = vol_sum / n_frames
    shell_vol = (4.0 * math.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho_j = (nj - (1 if is_like else 0)) / max(vol_avg, 1e-12)
    norm = n_frames * ni * rho_j * shell_vol
    g = np.divide(hist, norm, out=np.ones_like(hist), where=norm > 1e-12)
    return centers, g


def run_kbi_extended_analysis(
    runs: list[RunRecord],
    cfg: Config,
    base_kbi_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run Full KBI Suite: Dawass 1/R, sub-volume fluctuations, Bhatia–Thornton, S(q)/SAXS, block SEM."""
    opts = (cfg.analyses or {}).get("options") or {}
    n_spheres = int(opts.get("subvolume_n_spheres", 48))
    n_blocks = max(2, int(opts.get("n_blocks", 5)))
    q_max = float(opts.get("q_max_inv_nm", 25.0))
    q_grid = np.linspace(0.0, q_max, 126)

    dawass_rows: list[dict[str, Any]] = []
    subvol_rows: list[dict[str, Any]] = []
    bt_rows: list[dict[str, Any]] = []
    sq_rows: list[dict[str, Any]] = []
    pref_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []

    for run in runs:
        species = [sp for sp, cnt in sorted(run.mol_counts.items()) if cnt > 0]
        if not species:
            continue

        # Load trajectory COMs if available (enables block averaging, subvolume fluctuation, and on-the-fly RDFs)
        coms_by_sp: dict[str, np.ndarray] = {}
        boxes_nm = np.zeros((0, 3), dtype=float)
        if run.prod_tpr.exists() and run.prod_xtc.exists():
            try:
                u = mda.Universe(str(run.prod_tpr), str(run.prod_xtc))
                f_start, f_stop, f_step = equilibrated_frame_slice(
                    u, cfg.t_eq_ns, cfg.t_end_ns, stride=max(1, cfg.rdf_stride)
                )
                frames = list(range(f_start, f_stop, f_step))
                if frames:
                    box_list = []
                    com_lists: dict[str, list[np.ndarray]] = {sp: [] for sp in species}
                    for fi in frames:
                        u.trajectory[fi]
                        ts = u.trajectory.ts
                        box_nm = np.asarray(ts.dimensions[:3], dtype=float) / 10.0
                        box_list.append(box_nm)
                        for sp in species:
                            ag = u.select_atoms(f"resname {sp}")
                            topo = run.topologies.get(sp)
                            if topo and len(ag) > 0:
                                coms = compute_whole_coms_vectorized(ag, topo, ts.dimensions) / 10.0
                                com_lists[sp].append(coms)
                    boxes_nm = np.asarray(box_list, dtype=float)
                    for sp in species:
                        if com_lists[sp]:
                            coms_by_sp[sp] = np.stack(com_lists[sp], axis=0)
            except Exception:
                coms_by_sp = {}

        if boxes_nm.size > 0:
            avg_box = np.mean(boxes_nm, axis=0)
            vol_avg = float(avg_box[0] * avg_box[1] * avg_box[2])
        else:
            vol_avg = 125.0
            avg_box = np.array([5.0, 5.0, 5.0])

        rho_vec = np.array([run.mol_counts[sp] / max(vol_avg, 1e-12) for sp in species], dtype=float)
        r_max_eff = min(cfg.kbi_rmax_nm, 0.49 * float(np.min(avg_box)))
        n_comp = len(species)
        G_dawass_mat = np.zeros((n_comp, n_comp), dtype=float)
        g_pairs_for_sq: dict[tuple[int, int], np.ndarray] = {}
        r_common: np.ndarray | None = None

        for i, sp_i in enumerate(species):
            for j in range(i, n_comp):
                sp_j = species[j]
                r_arr: np.ndarray | None = None
                g_arr: np.ndarray | None = None
                if sp_i in coms_by_sp and sp_j in coms_by_sp and boxes_nm.size > 0:
                    r_arr, g_arr = _compute_pair_rdf_from_coms(
                        coms_by_sp[sp_i],
                        coms_by_sp[sp_j],
                        boxes_nm,
                        r_max_eff,
                        cfg.kbi_rebin_nm,
                        is_like=(sp_i == sp_j),
                    )
                else:
                    # Fallback to precomputed rdf_*.xvg if present
                    for key in (f"{sp_i}_{sp_j}", f"{sp_j}_{sp_i}", f"{sp_i.lower()}_{sp_j.lower()}"):
                        if key in run.rdf_xvgs and run.rdf_xvgs[key].exists():
                            df_rdf = read_xvg_dataframe(run.rdf_xvgs[key])
                            if df_rdf.shape[1] >= 2:
                                r_arr, g_arr = rebin_rdf(
                                    df_rdf.iloc[:, 0].to_numpy(dtype=float),
                                    df_rdf.iloc[:, 1].to_numpy(dtype=float),
                                    cfg.kbi_rebin_nm,
                                )
                            break
                if r_arr is None or g_arr is None:
                    continue

                g_corr = apply_gvdv_correction(
                    r_arr, g_arr, rho_vec[j], run.mol_counts[sp_j], is_like_pair=(sp_i == sp_j)
                )
                dw = dawass_extrapolate(r_arr, g_corr)
                G_dawass_mat[i, j] = dw["g_inf_cm3_mol"] if np.isfinite(dw["g_inf_cm3_mol"]) else 0.0
                G_dawass_mat[j, i] = G_dawass_mat[i, j]
                g_pairs_for_sq[(i, j)] = g_corr
                r_common = r_arr

                dawass_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "pair": f"{sp_i}-{sp_j}",
                        **dw,
                    }
                )

                # Block-averaged uncertainty
                if sp_i in coms_by_sp and sp_j in coms_by_sp and len(boxes_nm) >= n_blocks * 2:
                    splits = np.array_split(np.arange(len(boxes_nm)), n_blocks)
                    blk_vals = []
                    for idx_b in splits:
                        rb, gb = _compute_pair_rdf_from_coms(
                            coms_by_sp[sp_i][idx_b],
                            coms_by_sp[sp_j][idx_b],
                            boxes_nm[idx_b],
                            r_max_eff,
                            cfg.kbi_rebin_nm,
                            is_like=(sp_i == sp_j),
                        )
                        gb_c = apply_gvdv_correction(
                            rb, gb, rho_vec[j], run.mol_counts[sp_j], is_like_pair=(sp_i == sp_j)
                        )
                        b_dw = dawass_extrapolate(rb, gb_c)["g_inf_cm3_mol"]
                        if np.isfinite(b_dw):
                            blk_vals.append(b_dw)
                    if blk_vals:
                        arr_b = np.asarray(blk_vals, dtype=float)
                        block_rows.append(
                            {
                                "run_id": run.run_id,
                                "system_code": run.system_code,
                                "replica": run.replica,
                                "pair": f"{sp_i}-{sp_j}",
                                "n_blocks": len(arr_b),
                                "g_block_mean_cm3_mol": float(np.mean(arr_b)),
                                "g_block_sd_cm3_mol": float(np.std(arr_b, ddof=1)) if len(arr_b) > 1 else 0.0,
                                "g_block_sem_cm3_mol": float(np.std(arr_b, ddof=1) / math.sqrt(len(arr_b))) if len(arr_b) > 1 else 0.0,
                            }
                        )

        # Direct sub-volume fluctuation KBI
        if coms_by_sp and boxes_nm.size > 0:
            sv_df = compute_subvolume_fluctuation_kbi(coms_by_sp, boxes_nm, n_spheres=n_spheres)
            if not sv_df.empty:
                sv_df.insert(0, "replica", run.replica)
                sv_df.insert(0, "system_code", run.system_code)
                sv_df.insert(0, "run_id", run.run_id)
                subvol_rows.extend(sv_df.to_dict(orient="records"))

        # Full KB thermodynamic inversion & Bhatia-Thornton
        temp_k = float(run.metadata.get("temperatureK") or run.metadata.get("temperature_k") or cfg.temperature_k)
        kb_full = compute_kb_thermodynamics_full(G_dawass_mat, rho_vec, temp_k, species_names=species)
        bt_row: dict[str, Any] = {
            "run_id": run.run_id,
            "system_code": run.system_code,
            "replica": run.replica,
            "temperature_k": temp_k,
            "kappa_t_kb_gpa_inv": kb_full["kappa_t_gpa_inv"],
            "gamma_binary": kb_full["gamma_binary"],
            "S_NN_0": kb_full["S_NN_0"],
            "S_CC_0": kb_full["S_CC_0"],
            "S_NC_0": kb_full["S_NC_0"],
        }
        for sp, pmv in kb_full["pmv_cm3_mol"].items():
            bt_row[f"pmv_{sp}_cm3_mol"] = pmv
        bt_rows.append(bt_row)

        # Preferential solvation table
        delta_n = kb_full["delta_n_matrix"]
        for i, sp_i in enumerate(species):
            for j, sp_j in enumerate(species):
                pref_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "central_species": sp_i,
                        "neighbor_species": sp_j,
                        "delta_n_excess": float(delta_n[i, j]),
                        "S0_partial": float(kb_full["S0_partial"][i, j]),
                    }
                )

        # q-dependent S(q) & SAXS intensity
        if r_common is not None and g_pairs_for_sq:
            z_counts = _species_electron_counts(run)
            ff = np.array([z_counts.get(sp, 10.0) for sp in species], dtype=float)
            sq = compute_structure_factor_sq(r_common, g_pairs_for_sq, rho_vec, q_grid, form_factors=ff)
            for qi, qv in enumerate(sq["q_inv_nm"]):
                row_q: dict[str, Any] = {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "q_inv_nm": float(qv),
                    "I_saxs_norm": float(sq["I_total"][qi]),
                }
                if "S_NN" in sq:
                    row_q["S_NN_q"] = float(sq["S_NN"][qi])
                    row_q["S_CC_q"] = float(sq["S_CC"][qi])
                    row_q["S_NC_q"] = float(sq["S_NC"][qi])
                sq_rows.append(row_q)

    return {
        "kbi_dawass": pd.DataFrame(dawass_rows),
        "kbi_subvolume": pd.DataFrame(subvol_rows),
        "kbi_bhatia_thornton": pd.DataFrame(bt_rows),
        "kbi_preferential_solvation": pd.DataFrame(pref_rows),
        "kbi_structure_factor_sq": pd.DataFrame(sq_rows),
        "kbi_block_uncertainty": pd.DataFrame(block_rows),
    }


def run_kbi_composition_series(
    runs: list[RunRecord],
    cfg: Config,
    bt_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Integrate binary Γ(x) across a composition sweep to produce ln(γi) and G^E."""
    if bt_df is None or bt_df.empty:
        return {"kbi_activity_excess_gibbs": pd.DataFrame()}

    # Build x1 table from runs matched to bt_df
    by_run = {r.run_id: r for r in runs}
    rows: list[dict[str, Any]] = []
    primary_sp: str | None = None
    for _, r_bt in bt_df.iterrows():
        run = by_run.get(str(r_bt["run_id"]))
        if run is None:
            continue
        species = sorted(sp for sp, c in run.mol_counts.items() if c > 0)
        if len(species) < 2:
            continue
        if primary_sp is None:
            primary_sp = cfg.primary_solvent if cfg.primary_solvent in species else species[0]
        tot = sum(run.mol_counts.values())
        x1 = run.mol_counts.get(primary_sp, 0) / max(tot, 1)
        rows.append(
            {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "temperature_k": float(r_bt.get("temperature_k", cfg.temperature_k)),
                "component_1": primary_sp,
                "x1": float(x1),
                "gamma_factor": float(r_bt["gamma_binary"]),
            }
        )
    if not rows:
        return {"kbi_activity_excess_gibbs": pd.DataFrame()}

    df_in = pd.DataFrame(rows)
    out_frames: list[pd.DataFrame] = []
    for temp_k, grp in df_in.groupby("temperature_k", sort=True):
        agg = grp.groupby("x1", as_index=False).agg(
            gamma_factor=("gamma_factor", "mean"),
            n_replicates=("gamma_factor", "count"),
        )
        if len(agg) >= 3:
            res = integrate_composition_series_kbi(agg, temperature_k=float(temp_k))
            res.insert(0, "temperature_k", float(temp_k))
            res.insert(1, "component_1", primary_sp or "comp1")
            out_frames.append(res)

    return {
        "kbi_activity_excess_gibbs": pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    }


def run_kbi_temperature_series(
    activity_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute Gibbs–Helmholtz HE and TSE across temperatures when activity/GE series spans >=2 T."""
    if activity_df is None or activity_df.empty or "temperature_k" not in activity_df.columns:
        return {"kbi_gibbs_helmholtz": pd.DataFrame()}
    df = activity_df.copy()
    if "x1" in df.columns:
        df["composition_key"] = df["x1"].map(lambda v: f"x1={float(v):.3f}")
    return {"kbi_gibbs_helmholtz": compute_temperature_series_gibbs_helmholtz(df)}
