from __future__ import annotations

import math
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func

from mody_analyse.core.config import Config
from mody_analyse.core.discovery import (
    MoleculeTopology,
    RunRecord,
    compute_whole_coms_vectorized,
    equilibrated_frame_slice,
)
from mody_analyse.core.pbc import (
    box_volume_nm3,
    dimensions_angstrom_to_box_nm,
    minimum_image_displacement,
)

KB_J_K = 1.380649e-23
E_CHARGE_C = 1.602176634e-19
EPS0_VAC = 8.8541878128e-12  # F m^-1 = C^2 J^-1 m^-1
E_NM_TO_CM = E_CHARGE_C * 1e-9  # C m
E_NM_TO_DEBYE = E_NM_TO_CM / 3.33564095e-30  # ~48.03204 D
NM2_PER_PS_TO_M2_PER_S = 1e-6


def _unwrap_com_series(coms_nm: np.ndarray, boxes_nm: np.ndarray) -> np.ndarray:
    """Continuously unwrap molecular COM trajectory (n_frames, n_mol, 3) across orthorhombic or triclinic PBC."""
    pos = np.asarray(coms_nm, dtype=float).copy()
    if pos.shape[0] <= 1:
        return pos
    for fi in range(1, pos.shape[0]):
        delta = minimum_image_displacement(pos[fi] - pos[fi - 1], boxes_nm[fi])
        pos[fi] = pos[fi - 1] + delta
    return pos


def compute_onsager_and_conductivity(
    unwrapped_coms_nm: dict[str, np.ndarray],
    net_charges_e: dict[str, float],
    dt_ps: float,
    volume_nm3: float,
    temperature_k: float,
    *,
    fit_frac: tuple[float, float] = (0.2, 0.7),
) -> dict[str, Any]:
    """Compute full Onsager transport matrix L_ab, Einstein–Helfand & Nernst–Einstein conductivity,
    Haven ratio, ionicity, and true vs ideal transference numbers.

    unwrapped_coms_nm: species -> array (n_frames, n_mol, 3) in nm (already continuously unwrapped).
    """
    species = sorted(unwrapped_coms_nm.keys())
    if not species or dt_ps <= 0 or volume_nm3 <= 0:
        return {"summary": {}, "onsager_rows": [], "transference_rows": []}

    n_frames = min(unwrapped_coms_nm[s].shape[0] for s in species)
    if n_frames < 5:
        return {"summary": {}, "onsager_rows": [], "transference_rows": []}

    max_lag = max(2, n_frames // 2)
    lags = np.arange(1, max_lag + 1)
    t_lags_ps = lags * float(dt_ps)
    n_sp = len(species)

    # Precompute collective center-of-species sums R_alpha(t) = sum_{i in alpha} r_i(t)
    R_coll: dict[str, np.ndarray] = {
        sp: np.sum(unwrapped_coms_nm[sp][:n_frames], axis=1) for sp in species
    }

    # Self-diffusion MSDs and collective cross-displacement correlations
    msd_self: dict[str, np.ndarray] = {sp: np.zeros(len(lags), dtype=float) for sp in species}
    coll_corr = np.zeros((n_sp, n_sp, len(lags)), dtype=float)

    for li, lag in enumerate(lags):
        # Multiple time origins
        dR = {sp: R_coll[sp][lag:] - R_coll[sp][:-lag] for sp in species}
        for i, sp_i in enumerate(species):
            dr_i = unwrapped_coms_nm[sp_i][lag:n_frames] - unwrapped_coms_nm[sp_i][: n_frames - lag]
            # sum over molecules of <|dr_i|^2>
            msd_self[sp_i][li] = float(np.mean(np.sum(dr_i * dr_i, axis=-1))) * dr_i.shape[1]
            for j in range(i, n_sp):
                sp_j = species[j]
                val = float(np.mean(np.sum(dR[sp_i] * dR[sp_j], axis=-1)))
                coll_corr[i, j, li] = val
                coll_corr[j, i, li] = val

    # Linear fit window for diffusive regime
    i0 = max(0, int(fit_frac[0] * len(lags)))
    i1 = max(i0 + 2, int(fit_frac[1] * len(lags)))
    t_fit = t_lags_ps[i0:i1]

    v_m3 = float(volume_nm3) * 1e-27
    pref = NM2_PER_PS_TO_M2_PER_S / (6.0 * v_m3 * KB_J_K * float(temperature_k))  # converts nm^2/ps -> L_ab (J^-1 m^-1 s^-1)

    L_total = np.zeros((n_sp, n_sp), dtype=float)
    L_self = np.zeros(n_sp, dtype=float)
    D_self = np.zeros(n_sp, dtype=float)
    for i, sp_i in enumerate(species):
        n_mol_i = max(1, unwrapped_coms_nm[sp_i].shape[1])
        slope_self, _ = np.polyfit(t_fit, msd_self[sp_i][i0:i1], 1)
        L_self[i] = float(slope_self) * pref
        D_self[i] = float(slope_self / n_mol_i) * NM2_PER_PS_TO_M2_PER_S / 6.0  # m^2/s
        for j in range(i, n_sp):
            slope_ij, _ = np.polyfit(t_fit, coll_corr[i, j, i0:i1], 1)
            L_total[i, j] = float(slope_ij) * pref
            L_total[j, i] = L_total[i, j]

    z = np.array([float(net_charges_e.get(sp, 0.0)) for sp in species], dtype=float)
    sigma_eh_sm = float((E_CHARGE_C ** 2) * (z @ L_total @ z))
    sigma_ne_sm = float((E_CHARGE_C ** 2) * np.sum((z ** 2) * L_self))
    haven_ratio = sigma_ne_sm / sigma_eh_sm if abs(sigma_eh_sm) > 1e-18 else float("nan")
    ionicity = sigma_eh_sm / sigma_ne_sm if abs(sigma_ne_sm) > 1e-18 else float("nan")

    onsager_rows: list[dict[str, Any]] = []
    for i, sp_i in enumerate(species):
        for j in range(i, n_sp):
            sp_j = species[j]
            l_dist = L_total[i, j] - (L_self[i] if i == j else 0.0)
            onsager_rows.append(
                {
                    "species_i": sp_i,
                    "species_j": sp_j,
                    "z_i": float(z[i]),
                    "z_j": float(z[j]),
                    "L_total_per_J_m_s": float(L_total[i, j]),
                    "L_self_per_J_m_s": float(L_self[i] if i == j else 0.0),
                    "L_distinct_per_J_m_s": float(l_dist),
                }
            )

    transference_rows: list[dict[str, Any]] = []
    for i, sp_i in enumerate(species):
        if abs(z[i]) <= 1e-6:
            continue
        t_true = float((E_CHARGE_C ** 2) * z[i] * float(L_total[i] @ z) / sigma_eh_sm) if abs(sigma_eh_sm) > 1e-18 else float("nan")
        t_ideal = float((E_CHARGE_C ** 2) * (z[i] ** 2) * L_self[i] / sigma_ne_sm) if abs(sigma_ne_sm) > 1e-18 else float("nan")
        transference_rows.append(
            {
                "species": sp_i,
                "net_charge_e": float(z[i]),
                "D_self_1e9_m2_s": float(D_self[i] * 1e9),
                "t_true_onsager": t_true,
                "t_ideal_nernst_einstein": t_ideal,
            }
        )

    summary = {
        "sigma_eh_s_m": sigma_eh_sm,
        "sigma_eh_ms_cm": sigma_eh_sm * 10.0,
        "sigma_ne_s_m": sigma_ne_sm,
        "sigma_ne_ms_cm": sigma_ne_sm * 10.0,
        "haven_ratio": haven_ratio,
        "ionicity": ionicity,
    }
    return {"summary": summary, "onsager_rows": onsager_rows, "transference_rows": transference_rows}


def _fit_kww(t: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """Fit stretched exponential exp(-(t/tau0)^beta); return (tau_mean, tau0, beta)."""
    mask = (t >= 0) & (c > 0.02) & np.isfinite(c)
    if int(np.sum(mask)) < 4:
        tau_int = float(trapezoid(np.clip(c, 0.0, 1.0), t)) if len(t) >= 2 else float("nan")
        return tau_int, tau_int, 1.0
    tm, cm = t[mask], np.clip(c[mask], 1e-6, 1.0)
    tau_guess = max(float(trapezoid(cm, tm)), float(tm[1] - tm[0]) if len(tm) > 1 else 1.0)
    try:
        popt, _ = curve_fit(
            lambda x, tau0, beta: np.exp(-np.power(np.maximum(x, 0.0) / tau0, beta)),
            tm,
            cm,
            p0=[tau_guess, 0.85],
            bounds=([1e-6, 0.15], [1e9, 2.0]),
            maxfev=4000,
        )
        tau0, beta = float(popt[0]), float(popt[1])
        tau_mean = (tau0 / beta) * float(gamma_func(1.0 / beta))
        return tau_mean, tau0, beta
    except Exception:
        tau_int = float(trapezoid(np.clip(c, 0.0, 1.0), t))
        return tau_int, tau_int, 1.0


def compute_rotational_acf(unit_vectors: np.ndarray, dt_ps: float) -> dict[str, Any]:
    """Compute Legendre P1 and P2 orientational ACFs from unit_vectors of shape (n_frames, n_mol, 3)."""
    u = np.asarray(unit_vectors, dtype=float)
    n_frames = u.shape[0]
    if n_frames < 4 or dt_ps <= 0:
        return {
            "tau1_kww_ps": float("nan"),
            "tau2_kww_ps": float("nan"),
            "beta1_kww": float("nan"),
            "beta2_kww": float("nan"),
            "tau1_int_ps": float("nan"),
            "tau2_int_ps": float("nan"),
            "debye_ratio_tau1_tau2": float("nan"),
            "t_ps": np.array([]),
            "c1": np.array([]),
            "c2": np.array([]),
        }

    # Normalize safely
    norms = np.linalg.norm(u, axis=-1, keepdims=True)
    u = np.divide(u, norms, out=np.zeros_like(u), where=norms > 1e-12)

    max_lag = max(2, n_frames // 2)
    lags = np.arange(0, max_lag + 1)
    t_ps = lags * float(dt_ps)
    c1 = np.ones(len(lags), dtype=float)
    c2 = np.ones(len(lags), dtype=float)
    for li, lag in enumerate(lags):
        if lag == 0:
            continue
        dots = np.sum(u[:-lag] * u[lag:], axis=-1)
        c1[li] = float(np.mean(dots))
        c2[li] = float(np.mean(0.5 * (3.0 * dots * dots - 1.0)))

    tau1_kww, _, beta1 = _fit_kww(t_ps, c1)
    tau2_kww, _, beta2 = _fit_kww(t_ps, c2)
    tau1_int = float(trapezoid(np.clip(c1, 0.0, 1.0), t_ps))
    tau2_int = float(trapezoid(np.clip(c2, 0.0, 1.0), t_ps))
    ratio = tau1_kww / tau2_kww if (np.isfinite(tau2_kww) and tau2_kww > 1e-12) else float("nan")
    return {
        "tau1_kww_ps": tau1_kww,
        "tau2_kww_ps": tau2_kww,
        "beta1_kww": beta1,
        "beta2_kww": beta2,
        "tau1_int_ps": tau1_int,
        "tau2_int_ps": tau2_int,
        "debye_ratio_tau1_tau2": ratio,
        "t_ps": t_ps,
        "c1": c1,
        "c2": c2,
    }


def compute_glassy_metrics(
    unwrapped_coms_nm: np.ndarray,
    dt_ps: float,
    *,
    q_star_inv_nm: float = 14.0,
) -> dict[str, Any]:
    """Compute non-Gaussian parameter α2(t), self-intermediate scattering Fs(q*,t), τα, and van Hove Gs(r,t*)."""
    pos = np.asarray(unwrapped_coms_nm, dtype=float)
    n_frames = pos.shape[0]
    if n_frames < 5 or dt_ps <= 0:
        return {
            "alpha2_max": float("nan"),
            "t_star_ps": float("nan"),
            "tau_alpha_ps": float("nan"),
            "beta_alpha_kww": float("nan"),
            "q_star_inv_nm": float(q_star_inv_nm),
            "t_ps": np.array([]),
            "alpha2": np.array([]),
            "fs_qt": np.array([]),
            "vanhove_r_nm": np.array([]),
            "vanhove_gs": np.array([]),
        }

    max_lag = max(2, n_frames // 2)
    lags = np.arange(1, max_lag + 1)
    t_ps = lags * float(dt_ps)
    alpha2 = np.zeros(len(lags), dtype=float)
    fs_qt = np.zeros(len(lags), dtype=float)

    for li, lag in enumerate(lags):
        dr = pos[lag:] - pos[:-lag]
        r2 = np.sum(dr * dr, axis=-1)
        m2 = float(np.mean(r2))
        m4 = float(np.mean(r2 * r2))
        alpha2[li] = (3.0 * m4 / (5.0 * m2 * m2) - 1.0) if m2 > 1e-18 else 0.0
        r = np.sqrt(np.maximum(r2, 0.0))
        qr = float(q_star_inv_nm) * r
        sinc = np.where(qr > 1e-9, np.sin(qr) / np.where(qr > 1e-9, qr, 1.0), 1.0)
        fs_qt[li] = float(np.mean(sinc))

    idx_star = int(np.argmax(alpha2))
    alpha2_max = float(alpha2[idx_star])
    t_star_ps = float(t_ps[idx_star])

    # Prepend t=0, Fs=1 for KWW alpha-relaxation fit
    t_fs = np.concatenate(([0.0], t_ps))
    y_fs = np.concatenate(([1.0], fs_qt))
    tau_alpha, _, beta_alpha = _fit_kww(t_fs, y_fs)

    # Self van Hove 4 pi r^2 Gs(r, t*) histogram at t*
    lag_star = int(lags[idx_star])
    dr_star = pos[lag_star:] - pos[:-lag_star]
    r_star = np.linalg.norm(dr_star, axis=-1).ravel()
    r_hi = max(float(np.percentile(r_star, 99.5)), 0.5) if len(r_star) else 1.0
    hist, edges = np.histogram(r_star, bins=40, range=(0.0, r_hi), density=True)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    return {
        "alpha2_max": alpha2_max,
        "t_star_ps": t_star_ps,
        "tau_alpha_ps": tau_alpha,
        "beta_alpha_kww": beta_alpha,
        "q_star_inv_nm": float(q_star_inv_nm),
        "t_ps": t_ps,
        "alpha2": alpha2,
        "fs_qt": fs_qt,
        "vanhove_r_nm": r_centers,
        "vanhove_gs": hist,
    }


def compute_dielectric_constant(
    mol_dipoles_e_nm: np.ndarray,
    volume_nm3: float,
    temperature_k: float,
    *,
    is_ionic: bool = False,
) -> dict[str, Any]:
    """Compute static dielectric constant ε0 and Kirkwood gK from molecular dipoles (n_frames, n_mol, 3) in e*nm."""
    mu = np.asarray(mol_dipoles_e_nm, dtype=float)
    if mu.ndim != 3 or mu.shape[0] < 2 or volume_nm3 <= 0:
        return {
            "epsilon_0": float("nan"),
            "kirkwood_g_k": float("nan"),
            "mean_mol_dipole_debye": float("nan"),
            "is_ionic_rotational_only": is_ionic,
            "epsilon_running_last_half_drift": float("nan"),
        }

    M = np.sum(mu, axis=1)  # (n_frames, 3) in e*nm
    M2_enm = float(np.mean(np.sum(M * M, axis=-1)))
    M_mean_sq_enm = float(np.sum(np.mean(M, axis=0) ** 2))
    fluct_m2_enm = max(0.0, M2_enm - M_mean_sq_enm)
    fluct_m2_cm = fluct_m2_enm * (E_NM_TO_CM ** 2)

    v_m3 = float(volume_nm3) * 1e-27
    eps0 = 1.0 + fluct_m2_cm / (3.0 * EPS0_VAC * v_m3 * KB_J_K * float(temperature_k))

    sum_mu2_enm = float(np.mean(np.sum(mu * mu, axis=(1, 2))))
    g_k = fluct_m2_enm / sum_mu2_enm if sum_mu2_enm > 1e-18 else float("nan")
    mean_mu_debye = float(np.mean(np.linalg.norm(mu, axis=-1))) * E_NM_TO_DEBYE

    # Convergence check: compare full window vs first half
    half = max(2, mu.shape[0] // 2)
    M_h = M[:half]
    fluct_h_enm = max(0.0, float(np.mean(np.sum(M_h * M_h, axis=-1))) - float(np.sum(np.mean(M_h, axis=0) ** 2)))
    eps0_half = 1.0 + (fluct_h_enm * (E_NM_TO_CM ** 2)) / (3.0 * EPS0_VAC * v_m3 * KB_J_K * float(temperature_k))

    return {
        "epsilon_0": float(eps0),
        "kirkwood_g_k": float(g_k),
        "mean_mol_dipole_debye": mean_mu_debye,
        "is_ionic_rotational_only": bool(is_ionic),
        "epsilon_running_last_half_drift": float(abs(eps0 - eps0_half)),
    }


def _whole_molecule_positions_and_dipoles(
    ag: mda.AtomGroup,
    topo: MoleculeTopology,
    dimensions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap each molecule around atom 0 and return (coms_nm, unit_vecs, dipoles_e_nm)."""
    n_at = len(topo.atom_names)
    pos_nm = np.asarray(ag.positions, dtype=float) / 10.0
    n_mol = len(pos_nm) // n_at
    pos_nm = pos_nm[: n_mol * n_at].reshape(n_mol, n_at, 3)
    box_nm = dimensions_angstrom_to_box_nm(dimensions)

    # Unwrap atoms relative to atom 0 of each molecule (exact for orthorhombic or triclinic cells)
    ref = pos_nm[:, 0:1, :]
    rel = minimum_image_displacement(pos_nm - ref, box_nm)
    whole = ref + rel

    masses = np.asarray(topo.masses, dtype=float)
    if len(masses) != n_at or float(np.sum(masses)) <= 0:
        masses = np.ones(n_at, dtype=float)
    m_sum = float(np.sum(masses))
    coms = np.sum(whole * masses[None, :, None], axis=1) / m_sum  # (n_mol, 3)

    charges = np.asarray(topo.charges, dtype=float)
    if len(charges) != n_at:
        charges = np.zeros(n_at, dtype=float)
    rel_com = whole - coms[:, None, :]
    dipoles = np.sum(rel_com * charges[None, :, None], axis=1)  # (n_mol, 3) in e*nm

    # Orientation unit vector: dipole if polar (> 0.01 e*nm), else atom(0)->atom(furthest heavy)
    dip_norm = np.linalg.norm(dipoles, axis=-1, keepdims=True)
    if float(np.mean(dip_norm)) > 1e-3:
        uvec = np.divide(dipoles, dip_norm, out=np.zeros_like(dipoles), where=dip_norm > 1e-12)
    elif n_at >= 2:
        # Furthest atom from atom 0 in molecule 0
        d0 = np.linalg.norm(whole[0] - whole[0, 0:1], axis=-1)
        idx_far = int(np.argmax(d0))
        if idx_far == 0:
            idx_far = 1
        vec = whole[:, idx_far, :] - whole[:, 0, :]
        vn = np.linalg.norm(vec, axis=-1, keepdims=True)
        uvec = np.divide(vec, vn, out=np.zeros_like(vec), where=vn > 1e-12)
    else:
        uvec = np.zeros((n_mol, 3), dtype=float)

    return coms, uvec, dipoles


def _collect_run_dynamics_arrays(
    run: RunRecord,
    cfg: Config,
    stride: int = 1,
) -> dict[str, Any] | None:
    if not (run.tpr_path.exists() and run.xtc_path.exists()):
        return None
    u = mda.Universe(str(run.tpr_path), str(run.xtc_path))
    f_start, f_stop, f_step = equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, stride=max(1, int(stride)))
    frames = list(range(f_start, f_stop, f_step))
    if len(frames) < 4:
        return None

    species = [sp for sp, cnt in sorted(run.molecule_counts.items()) if cnt > 0 and sp in run.topologies]
    if not species:
        return None

    times_ps: list[float] = []
    boxes_nm: list[np.ndarray] = []
    coms_raw: dict[str, list[np.ndarray]] = {sp: [] for sp in species}
    uvecs: dict[str, list[np.ndarray]] = {sp: [] for sp in species}
    dipoles: dict[str, list[np.ndarray]] = {sp: [] for sp in species}

    ags = {sp: u.select_atoms(f"resname {sp}") for sp in species}
    for fi in frames:
        u.trajectory[fi]
        ts = u.trajectory.ts
        times_ps.append(float(ts.time))
        boxes_nm.append(dimensions_angstrom_to_box_nm(ts.dimensions))
        for sp in species:
            if len(ags[sp]) == 0:
                continue
            c, uv, dip = _whole_molecule_positions_and_dipoles(ags[sp], run.topologies[sp], ts.dimensions)
            coms_raw[sp].append(c)
            uvecs[sp].append(uv)
            dipoles[sp].append(dip)

    boxes_arr = np.asarray(boxes_nm, dtype=float)
    dt_ps = float(np.median(np.diff(times_ps))) if len(times_ps) >= 2 else 1.0
    vol_mean_nm3 = float(np.mean([box_volume_nm3(boxes_arr[fi]) for fi in range(len(boxes_arr))]))

    coms_unwrapped = {
        sp: _unwrap_com_series(np.stack(coms_raw[sp], axis=0), boxes_arr)
        for sp in species
        if coms_raw[sp]
    }
    uvecs_arr = {sp: np.stack(uvecs[sp], axis=0) for sp in species if uvecs[sp]}
    dipoles_arr = {sp: np.stack(dipoles[sp], axis=0) for sp in species if dipoles[sp]}
    net_charges = {sp: float(sum(run.topologies[sp].charges)) for sp in species}

    return {
        "dt_ps": dt_ps,
        "vol_mean_nm3": vol_mean_nm3,
        "boxes_nm": boxes_arr,
        "coms_unwrapped": coms_unwrapped,
        "uvecs": uvecs_arr,
        "dipoles": dipoles_arr,
        "net_charges": net_charges,
    }


def run_electrochemistry_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    onsager_rows: list[dict[str, Any]] = []
    trans_rows: list[dict[str, Any]] = []
    for run in runs:
        data = _collect_run_dynamics_arrays(run, cfg, stride=max(1, cfg.rdf_stride))
        if data is None:
            continue
        if not any(abs(q) > 1e-3 for q in data["net_charges"].values()):
            continue
        temp_k = float(run.metadata.get("temperatureK") or run.metadata.get("temperature_k") or cfg.temperature_k)
        res = compute_onsager_and_conductivity(
            data["coms_unwrapped"],
            data["net_charges"],
            data["dt_ps"],
            data["vol_mean_nm3"],
            temp_k,
        )
        if res["summary"]:
            summary_rows.append({"run_id": run.run_id, "system_code": run.system_code, "replica": run.replica, **res["summary"]})
        for r in res["onsager_rows"]:
            onsager_rows.append({"run_id": run.run_id, "system_code": run.system_code, "replica": run.replica, **r})
        for r in res["transference_rows"]:
            trans_rows.append({"run_id": run.run_id, "system_code": run.system_code, "replica": run.replica, **r})

    return {
        "electrochemistry_summary": pd.DataFrame(summary_rows),
        "onsager_matrix": pd.DataFrame(onsager_rows),
        "transference_numbers": pd.DataFrame(trans_rows),
    }


def run_rotational_dynamics_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        data = _collect_run_dynamics_arrays(run, cfg, stride=max(1, cfg.rdf_stride))
        if data is None:
            continue
        for sp, uv in data["uvecs"].items():
            if len(run.topologies[sp].atom_names) < 2:
                continue
            res = compute_rotational_acf(uv, data["dt_ps"])
            rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "species": sp,
                    "tau1_kww_ps": res["tau1_kww_ps"],
                    "tau2_kww_ps": res["tau2_kww_ps"],
                    "beta1_kww": res["beta1_kww"],
                    "beta2_kww": res["beta2_kww"],
                    "tau1_int_ps": res["tau1_int_ps"],
                    "tau2_int_ps": res["tau2_int_ps"],
                    "debye_ratio_tau1_tau2": res["debye_ratio_tau1_tau2"],
                }
            )
    return {"rotational_dynamics": pd.DataFrame(rows)}


def run_glassy_dynamics_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    vh_rows: list[dict[str, Any]] = []
    for run in runs:
        data = _collect_run_dynamics_arrays(run, cfg, stride=max(1, cfg.rdf_stride))
        if data is None:
            continue
        for sp, coms in data["coms_unwrapped"].items():
            # Estimate characteristic wavevector q* = 2*pi / (V / N_tot)^(1/3)
            n_tot = max(1, sum(run.molecule_counts.values()))
            d_mol = (data["vol_mean_nm3"] / n_tot) ** (1.0 / 3.0)
            q_star = 2.0 * math.pi / max(d_mol, 0.2)
            res = compute_glassy_metrics(coms, data["dt_ps"], q_star_inv_nm=q_star)
            rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "species": sp,
                    "q_star_inv_nm": res["q_star_inv_nm"],
                    "alpha2_max": res["alpha2_max"],
                    "t_star_ps": res["t_star_ps"],
                    "tau_alpha_ps": res["tau_alpha_ps"],
                    "beta_alpha_kww": res["beta_alpha_kww"],
                }
            )
            for rv, gv in zip(res["vanhove_r_nm"], res["vanhove_gs"]):
                vh_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "species": sp,
                        "t_star_ps": res["t_star_ps"],
                        "r_nm": float(rv),
                        "van_hove_pdf": float(gv),
                    }
                )
    return {
        "glassy_dynamics": pd.DataFrame(rows),
        "vanhove_profiles": pd.DataFrame(vh_rows),
    }


def run_dielectric_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        data = _collect_run_dynamics_arrays(run, cfg, stride=max(1, cfg.rdf_stride))
        if data is None or not data["dipoles"]:
            continue
        is_ionic = any(abs(q) > 1e-3 for q in data["net_charges"].values())
        all_mu = np.concatenate(list(data["dipoles"].values()), axis=1)
        temp_k = float(run.metadata.get("temperatureK") or run.metadata.get("temperature_k") or cfg.temperature_k)
        res = compute_dielectric_constant(all_mu, data["vol_mean_nm3"], temp_k, is_ionic=is_ionic)
        rows.append(
            {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "replica": run.replica,
                "temperature_k": temp_k,
                "volume_mean_nm3": data["vol_mean_nm3"],
                **res,
            }
        )
    return {"dielectric_properties": pd.DataFrame(rows)}
