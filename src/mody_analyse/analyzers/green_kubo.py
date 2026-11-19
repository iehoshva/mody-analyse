from __future__ import annotations

import math
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
import tidynamics
from scipy.integrate import cumulative_trapezoid

from mody_analyse.core.config import Config
from mody_analyse.core.discovery import RunRecord, compute_whole_coms_vectorized, equilibrated_frame_slice, read_xvg


KB_J_K = 1.380649e-23


def _prefactor_bar2_ps_to_mpa_s(volume_nm3: float, temp_K: float) -> float:
    return (volume_nm3 * 1e-27 * 1e10 * 1e-12 / (KB_J_K * temp_K)) * 1e3


def compute_stress_acf(
    stress_arr: np.ndarray,
    cols: list[str],
    t_eq_ps: float,
    t_end_ps: float,
    max_lag_ps: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    t_ps = stress_arr[:, cols.index("Time")]
    mask = (t_ps >= t_eq_ps) & (t_ps <= t_end_ps)
    sub = stress_arr[mask]
    if len(sub) < 4:
        sub = stress_arr
    dt_ps = float(np.median(np.diff(sub[:, cols.index("Time")])))

    vol_mean = float(np.mean(sub[:, cols.index("Volume")]))
    temp_mean = float(np.mean(sub[:, cols.index("Temperature")]))

    pxx = sub[:, cols.index("Pres-XX")]
    pyy = sub[:, cols.index("Pres-YY")]
    pzz = sub[:, cols.index("Pres-ZZ")]
    pxy = sub[:, cols.index("Pres-XY")]
    pxz = sub[:, cols.index("Pres-XZ")]
    pyz = sub[:, cols.index("Pres-YZ")]

    n_orth1 = 0.5 * (pxx - pyy)
    n_orth2 = (pyy + pzz - 2.0 * pxx) / (2.0 * math.sqrt(3.0))
    components = [pxy, pxz, pyz, n_orth1, n_orth2]

    max_lag_idx = max(1, min(int(round(max_lag_ps / max(dt_ps, 1e-9))), len(pxy) // 4))
    acfs = []
    for comp in components:
        c = tidynamics.acf(comp - np.mean(comp))[: max_lag_idx + 1]
        acfs.append(c)
    acf_mean = np.mean(acfs, axis=0)
    tau = np.arange(len(acf_mean), dtype=float) * dt_ps
    return tau, acf_mean, vol_mean, temp_mean


def integrate_green_kubo(
    tau_ps: np.ndarray,
    acf_bar2: np.ndarray,
    volume_nm3: float,
    temp_K: float,
    dt_ps: float,
) -> dict[str, Any]:
    pref = _prefactor_bar2_ps_to_mpa_s(volume_nm3, temp_K)
    running_full = pref * cumulative_trapezoid(acf_bar2, tau_ps, initial=0.0)
    naive_trap = float(running_full[-1]) if len(running_full) else 0.0

    if dt_ps >= 0.5:
        if len(tau_ps) > 2:
            tail_tau = tau_ps[1:]
            tail_acf = acf_bar2[1:]
            running_tail = pref * cumulative_trapezoid(tail_acf, tail_tau, initial=0.0)
            tail_window = running_tail[max(1, int(0.4 * len(running_tail))) :]
            eta_tail = float(np.mean(tail_window))
        else:
            tail_tau = tau_ps
            running_tail = running_full
            eta_tail = naive_trap
        return {
            "regime": "coarse_2ps_tail_diagnostic",
            "eta_plateau_mpa_s": float("nan"),
            "eta_tail_only_mpa_s": eta_tail,
            "eta_naive_trapezoid_mpa_s": naive_trap,
            "running_tau_ps": tail_tau,
            "running_eta_mpa_s": running_tail,
            "c0_bar2": float(acf_bar2[0]),
            "c_lag1_ratio": float(acf_bar2[1] / acf_bar2[0]) if len(acf_bar2) > 1 and acf_bar2[0] != 0 else float("nan"),
        }

    tail_start = max(1, int(0.6 * len(running_full)))
    eta_plat = float(np.mean(running_full[tail_start:]))
    return {
        "regime": "high_freq_full",
        "eta_plateau_mpa_s": eta_plat,
        "eta_tail_only_mpa_s": eta_plat,
        "eta_naive_trapezoid_mpa_s": naive_trap,
        "running_tau_ps": tau_ps,
        "running_eta_mpa_s": running_full,
        "c0_bar2": float(acf_bar2[0]),
        "c_lag1_ratio": float(acf_bar2[1] / acf_bar2[0]) if len(acf_bar2) > 1 and acf_bar2[0] != 0 else float("nan"),
    }


def stokes_einstein_viscosity_mpa_s(d_cm2_s: float, temp_K: float, r_h_nm: float = 0.38) -> float:
    """Estimate effective viscosity via Stokes-Einstein: eta = kB*T / (6*pi*D*r_H)."""
    d_m2_s = d_cm2_s * 1e-4
    r_h_m = r_h_nm * 1e-9
    eta_pa_s = (KB_J_K * temp_K) / (6.0 * math.pi * max(d_m2_s, 1e-18) * max(r_h_m, 1e-12))
    return float(eta_pa_s * 1e3)


def _resolve_rh_nm(run: RunRecord, sp: str) -> float:
    if sp in run.chemistry:
        return run.chemistry[sp].rh_nm
    return 0.38


def _compute_msd_and_curves(run: RunRecord, cfg: Config) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute unwrapped COM MSD and Einstein self-diffusion D_i (10^-5 cm^2/s) on equilibrated window."""
    u = mda.Universe(str(run.tpr_path), str(run.xtc_path))
    start_idx, stop_idx, step = equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, stride=1)

    species = [s for s in run.molecule_counts if s in run.topologies]
    sp_ag = {s: u.select_atoms(f"resname {s}") for s in species}
    times_ps: list[float] = []
    coms_by_sp: dict[str, list[np.ndarray]] = {s: [] for s in species}

    prev_coms: dict[str, np.ndarray] = {}
    unwrapped: dict[str, np.ndarray] = {}
    for ts in u.trajectory[start_idx:stop_idx:step]:
        box_a = ts.dimensions[:3]
        times_ps.append(float(ts.time))
        for s in species:
            topo = run.topologies[s]
            cur = compute_whole_coms_vectorized(
                sp_ag[s].positions,
                n_mol=run.molecule_counts[s],
                n_atoms=topo.n_atoms,
                masses=topo.masses,
                box_a=box_a,
                anchor_idx=topo.center_atom_0based,
            )
            if s not in prev_coms:
                prev_coms[s] = cur.copy()
                unwrapped[s] = cur.copy()
            else:
                delta = cur - prev_coms[s]
                delta -= box_a[None, :] * np.round(delta / box_a[None, :])
                unwrapped[s] = unwrapped[s] + delta
                prev_coms[s] = cur.copy()
            coms_by_sp[s].append(unwrapped[s].copy())

    t_arr = np.asarray(times_ps, dtype=float)
    dt_ps = float(np.median(np.diff(t_arr))) if len(t_arr) > 1 else 1.0
    rows = []
    curve_rows: list[dict[str, Any]] = []
    for s in species:
        traj_xyz = np.stack(coms_by_sp[s], axis=0)
        msd_mols = []
        for m in range(traj_xyz.shape[1]):
            msd_mols.append(tidynamics.msd(traj_xyz[:, m, :]))
        msd_a2 = np.mean(msd_mols, axis=0)
        lags_ps = np.arange(len(msd_a2), dtype=float) * dt_ps
        max_lag = min(100_000.0, lags_ps[-1] * 0.4)
        min_lag = min(20_000.0, max_lag * 0.25)
        fit_mask = (lags_ps >= min_lag) & (lags_ps <= max_lag) & (msd_a2 > 0)
        if np.sum(fit_mask) < 5:
            fit_mask = (lags_ps > 0) & (lags_ps <= lags_ps[-1] * 0.5) & (msd_a2 > 0)

        slope_a2_ps, _ = np.polyfit(lags_ps[fit_mask], msd_a2[fit_mask], 1)
        d_cm2_s = float(slope_a2_ps / 6.0 * 1e-4)
        beta_slope, _ = np.polyfit(np.log(lags_ps[fit_mask]), np.log(msd_a2[fit_mask]), 1)
        rh_sp = _resolve_rh_nm(run, s)

        rows.append(
            {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "replica": run.replica,
                "species": s,
                "is_solute": s == run.solute_resname,
                "r_h_nm": rh_sp,
                "d_cm2_s": d_cm2_s,
                "d_1e5_cm2_s": d_cm2_s * 1e5,
                "diffusive_beta": float(beta_slope),
                "stokes_einstein_eta_mpa_s": stokes_einstein_viscosity_mpa_s(d_cm2_s, cfg.temperature_k, r_h_nm=rh_sp),
                "stokes_einstein_eta_species_rh_mpa_s": stokes_einstein_viscosity_mpa_s(d_cm2_s, cfg.temperature_k, r_h_nm=rh_sp),
            }
        )
        max_plot_idx = min(len(lags_ps), int(round(150_000.0 / max(dt_ps, 1.0))))
        sample_idx = np.unique(np.linspace(1, max(1, max_plot_idx - 1), min(150, max(1, max_plot_idx - 1)), dtype=int))
        for li in sample_idx:
            curve_rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "species": s,
                    "is_solute": s == run.solute_resname,
                    "lag_ns": float(lags_ps[li] / 1000.0),
                    "msd_nm2": float(msd_a2[li] / 100.0),
                }
            )
    return pd.DataFrame(rows), curve_rows


def compute_msd_diffusion(run: RunRecord, cfg: Config) -> pd.DataFrame:
    df, _ = _compute_msd_and_curves(run, cfg)
    return df


def run_transport_analysis(runs: list[RunRecord], cfg: Config, compute_msd: bool = True) -> dict[str, pd.DataFrame]:
    gk_rows: list[dict[str, Any]] = []
    gk_curves: list[dict[str, Any]] = []
    msd_frames: list[pd.DataFrame] = []
    msd_curve_rows: list[dict[str, Any]] = []

    all_solvents: list[str] = []
    for run in runs:
        for s in run.solvent_resnames:
            if s not in all_solvents:
                all_solvents.append(s)

        src_path = run.stress_path if run.stress_path is not None else run.thermo_path
        cols, arr = read_xvg(src_path)
        t_eq_ps = cfg.t_eq_ns * 1000.0
        t_end_ps = cfg.t_end_ns * 1000.0
        span_ps = max(t_end_ps - t_eq_ps, float(arr[-1, cols.index("Time")] - arr[0, cols.index("Time")]))
        max_lag_ps = min(120.0, max(20.0, 0.25 * span_ps))
        tau, acf_mean, vol_mean, temp_mean = compute_stress_acf(arr, cols, t_eq_ps, t_end_ps, max_lag_ps=max_lag_ps)
        dt_ps = float(tau[1] - tau[0]) if len(tau) > 1 else 2.0
        gk_res = integrate_green_kubo(tau, acf_mean, vol_mean, temp_mean, dt_ps)

        n_blocks = max(2, min(8, int(span_ps // 10_000.0) or 4))
        blk_len = span_ps / n_blocks
        blk_tails = []
        for bi in range(n_blocks):
            b0 = t_eq_ps + bi * blk_len
            b1 = b0 + blk_len
            btau, bacf, bvol, btemp = compute_stress_acf(arr, cols, b0, b1, max_lag_ps=min(80.0, 0.25 * blk_len))
            bres = integrate_green_kubo(btau, bacf, bvol, btemp, dt_ps)
            blk_tails.append(bres["eta_tail_only_mpa_s"])
        tail_sd = float(np.std(blk_tails, ddof=1)) if len(blk_tails) > 1 else 0.0

        gk_rows.append(
            {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "replica": run.replica,
                "dt_ps": dt_ps,
                "regime": gk_res["regime"],
                "c0_bar2": gk_res["c0_bar2"],
                "c_lag1_ratio": gk_res["c_lag1_ratio"],
                "eta_tail_mpa_s": gk_res["eta_tail_only_mpa_s"],
                "eta_tail_block_sd_mpa_s": tail_sd,
                "eta_naive_trapezoid_mpa_s": gk_res["eta_naive_trapezoid_mpa_s"],
            }
        )
        for tt, ee in zip(gk_res["running_tau_ps"], gk_res["running_eta_mpa_s"]):
            gk_curves.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "tau_ps": float(tt),
                    "eta_running_tail_mpa_s": float(ee),
                }
            )

        if compute_msd:
            mdf, mcurves = _compute_msd_and_curves(run, cfg)
            msd_frames.append(mdf)
            msd_curve_rows.extend(mcurves)

    gk_df = pd.DataFrame(gk_rows)
    msd_df = pd.concat(msd_frames, ignore_index=True) if msd_frames else pd.DataFrame()

    species_summary_df = (
        msd_df.groupby(["system_code", "species", "is_solute"], as_index=False)
        .agg(
            n_rep=("replica", "count"),
            r_h_nm=("r_h_nm", "mean"),
            d_1e5_cm2_s=("d_1e5_cm2_s", "mean"),
            d_sd=("d_1e5_cm2_s", "std"),
            diffusive_beta=("diffusive_beta", "mean"),
            se_eta_mpa_s=("stokes_einstein_eta_species_rh_mpa_s", "mean"),
            se_eta_sd=("stokes_einstein_eta_species_rh_mpa_s", "std"),
        )
        if not msd_df.empty
        else pd.DataFrame()
    )

    sys_summary_rows = []
    for sys_code, grp in gk_df.groupby("system_code"):
        row: dict[str, Any] = {
            "system_code": sys_code,
            "n_rep": len(grp),
            "eta_tail_mpa_s": float(grp["eta_tail_mpa_s"].mean()),
            "eta_tail_sd": float(grp["eta_tail_mpa_s"].std(ddof=1)) if len(grp) > 1 else float("nan"),
        }
        if not msd_df.empty:
            m_sub = msd_df.loc[msd_df["system_code"] == sys_code]
            for sp in all_solvents:
                label = sp.lower()
                sp_sub = m_sub.loc[m_sub["species"] == sp]
                row[f"d_{label}_1e5_cm2_s"] = float(sp_sub["d_1e5_cm2_s"].mean()) if not sp_sub.empty else float("nan")
                row[f"d_{label}_sd"] = float(sp_sub["d_1e5_cm2_s"].std(ddof=1)) if len(sp_sub) > 1 else float("nan")
                row[f"d_{label}_beta"] = float(sp_sub["diffusive_beta"].mean()) if not sp_sub.empty else float("nan")
            sol_sub = m_sub.loc[m_sub["is_solute"]]
            if not sol_sub.empty:
                row["d_sol_1e5_cm2_s"] = float(sol_sub["d_1e5_cm2_s"].mean())
                row["d_sol_sd"] = float(sol_sub["d_1e5_cm2_s"].std(ddof=1)) if len(sol_sub) > 1 else float("nan")
                row["d_sol_beta"] = float(sol_sub["diffusive_beta"].mean())
                row["se_eta_from_sol_mpa_s"] = float(sol_sub["stokes_einstein_eta_mpa_s"].mean())
                row["se_eta_from_sol_sd"] = float(sol_sub["stokes_einstein_eta_mpa_s"].std(ddof=1)) if len(sol_sub) > 1 else float("nan")
            else:
                row["d_sol_1e5_cm2_s"] = float("nan")
                row["d_sol_sd"] = float("nan")
                row["d_sol_beta"] = float("nan")
                primary_sp = all_solvents[0] if all_solvents else None
                prim_sub = m_sub.loc[m_sub["species"] == primary_sp] if primary_sp else m_sub
                row["se_eta_from_sol_mpa_s"] = float(prim_sub["stokes_einstein_eta_mpa_s"].mean()) if not prim_sub.empty else float("nan")
                row["se_eta_from_sol_sd"] = float(prim_sub["stokes_einstein_eta_mpa_s"].std(ddof=1)) if len(prim_sub) > 1 else float("nan")
        sys_summary_rows.append(row)

    return {
        "gk_runs": gk_df,
        "gk_curves": pd.DataFrame(gk_curves),
        "msd_runs": msd_df,
        "msd_curves": pd.DataFrame(msd_curve_rows),
        "transport_species_summary": species_summary_df,
        "transport_summary": pd.DataFrame(sys_summary_rows),
    }
