from __future__ import annotations

import math
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from mody_analyse.analyzers.electro_dynamics import _whole_molecule_positions_and_dipoles
from mody_analyse.core.config import Config
from mody_analyse.core.discovery import RunRecord, equilibrated_frame_slice, read_xvg_dataframe

BAR_NM_TO_MN_M = 0.1  # 1 bar * 1 nm = 1e5 N/m^2 * 1e-9 m = 1e-4 N/m = 0.1 mN/m


def compute_surface_tension(
    pxx_bar: np.ndarray,
    pyy_bar: np.ndarray,
    pzz_bar: np.ndarray,
    lz_nm: float | np.ndarray,
    *,
    n_interfaces: int = 2,
    n_blocks: int = 5,
) -> dict[str, float]:
    """Compute surface tension γ (mN/m) from GROMACS pressure tensor components (bar) and Lz (nm).

    γ = (Lz / n_interfaces) * <Pzz - 0.5*(Pxx + Pyy)> * 0.1  [mN/m]
    """
    pxx = np.asarray(pxx_bar, dtype=float)
    pyy = np.asarray(pyy_bar, dtype=float)
    pzz = np.asarray(pzz_bar, dtype=float)
    n = min(len(pxx), len(pyy), len(pzz))
    if n < 2:
        return {"gamma_mn_m": float("nan"), "gamma_sem_mn_m": float("nan"), "delta_p_bar": float("nan")}

    dp = pzz[:n] - 0.5 * (pxx[:n] + pyy[:n])
    lz_arr = np.broadcast_to(np.asarray(lz_nm, dtype=float), (n,))
    gamma_t = (lz_arr / float(max(1, n_interfaces))) * dp * BAR_NM_TO_MN_M
    gamma_mean = float(np.mean(gamma_t))

    nb = min(max(2, int(n_blocks)), n // 2) if n >= 4 else 1
    if nb >= 2:
        splits = np.array_split(gamma_t, nb)
        blk = [float(np.mean(s)) for s in splits if len(s) > 0]
        sem = float(np.std(blk, ddof=1) / math.sqrt(len(blk))) if len(blk) > 1 else 0.0
    else:
        sem = 0.0

    return {
        "gamma_mn_m": gamma_mean,
        "gamma_sem_mn_m": sem,
        "delta_p_bar": float(np.mean(dp)),
    }


def fit_slab_tanh_profile(
    z_nm: np.ndarray,
    rho_z: np.ndarray,
) -> dict[str, float]:
    """Fit symmetric slab density profile centered at z0 to extract bulk density, Gibbs dividing surface, and width.

    rho(z) = 0.5 * (rho_liq + rho_vap) - 0.5 * (rho_liq - rho_vap) * tanh(2 * (|z - z0| - z_gibbs) / width_d)
    """
    z = np.asarray(z_nm, dtype=float)
    rho = np.asarray(rho_z, dtype=float)
    if len(z) < 6 or float(np.max(rho)) <= 1e-12:
        return {
            "z_center_nm": float("nan"),
            "rho_bulk_liquid": float("nan"),
            "rho_vapor": float("nan"),
            "gibbs_half_thickness_nm": float("nan"),
            "interfacial_width_nm": float("nan"),
            "fit_r_squared": float("nan"),
        }

    # Estimate slab center via circular/weighted center of mass
    w = np.maximum(rho, 0.0)
    z0_guess = float(np.sum(z * w) / max(float(np.sum(w)), 1e-12))
    rho_max = float(np.percentile(rho, 95))
    rho_min = float(np.percentile(rho, 5))
    dist_abs = np.abs(z - z0_guess)
    above_half = dist_abs[rho >= 0.5 * (rho_max + rho_min)]
    zg_guess = float(np.max(above_half)) if len(above_half) else 0.25 * float(z[-1] - z[0])
    d_guess = max(0.2, 0.1 * zg_guess)

    def _model(z_in: np.ndarray, rho_l: float, rho_v: float, z0: float, z_g: float, d_w: float) -> np.ndarray:
        return 0.5 * (rho_l + rho_v) - 0.5 * (rho_l - rho_v) * np.tanh(2.0 * (np.abs(z_in - z0) - z_g) / d_w)

    try:
        popt, _ = curve_fit(
            _model,
            z,
            rho,
            p0=[rho_max, rho_min, z0_guess, zg_guess, d_guess],
            bounds=(
                [0.0, 0.0, float(z[0]), 0.05, 0.02],
                [rho_max * 2.0 + 1.0, rho_max + 1.0, float(z[-1]), float(z[-1] - z[0]), float(z[-1] - z[0])],
            ),
            maxfev=5000,
        )
        rho_l, rho_v, z0, z_g, d_w = [float(v) for v in popt]
        pred = _model(z, rho_l, rho_v, z0, z_g, d_w)
        ss_res = float(np.sum((rho - pred) ** 2))
        ss_tot = float(np.sum((rho - np.mean(rho)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 1.0
    except Exception:
        rho_l, rho_v, z0, z_g, d_w, r2 = rho_max, rho_min, z0_guess, zg_guess, d_guess, float("nan")

    return {
        "z_center_nm": z0,
        "rho_bulk_liquid": rho_l,
        "rho_vapor": rho_v,
        "gibbs_half_thickness_nm": z_g,
        "interfacial_width_nm": d_w,
        "fit_r_squared": r2,
    }


def compute_z_density_and_orientation_profiles(
    z_coms_nm: np.ndarray,
    unit_vecs: np.ndarray,
    box_nm: np.ndarray,
    *,
    n_bins: int = 60,
) -> pd.DataFrame:
    """Compute 1D z number density (nm^-3) and P2(cos theta) = 0.5*(3*uz^2 - 1) binned along z.

    z_coms_nm: (n_frames, n_mol), unit_vecs: (n_frames, n_mol, 3), box_nm: (n_frames, 3).
    """
    z = np.asarray(z_coms_nm, dtype=float)
    u = np.asarray(unit_vecs, dtype=float)
    b = np.asarray(box_nm, dtype=float)
    n_frames = z.shape[0]
    if n_frames == 0:
        return pd.DataFrame()

    lz_mean = float(np.mean(b[:, 2]))
    area_xy = float(np.mean(b[:, 0] * b[:, 1]))
    edges = np.linspace(0.0, lz_mean, int(n_bins) + 1)
    dz = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    slab_vol = area_xy * dz

    z_wrapped = np.mod(z, lz_mean).ravel()
    uz = u[..., 2].ravel()
    p2_vals = 0.5 * (3.0 * uz * uz - 1.0)

    counts, _ = np.histogram(z_wrapped, bins=edges)
    p2_sum, _ = np.histogram(z_wrapped, bins=edges, weights=p2_vals)
    rho_num = counts / max(n_frames * slab_vol, 1e-12)
    p2_mean = np.divide(p2_sum, counts, out=np.full_like(p2_sum, np.nan), where=counts > 0)

    return pd.DataFrame(
        {
            "z_nm": centers,
            "number_density_nm3": rho_num,
            "p2_orientation": p2_mean,
            "count": counts,
        }
    )


def _find_pressure_cols(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    def _pick(keys: tuple[str, ...]) -> str | None:
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in keys):
                return str(c)
        return None

    return (
        _pick(("pres-xx", "pxx", "p_xx")),
        _pick(("pres-yy", "pyy", "p_yy")),
        _pick(("pres-zz", "pzz", "p_zz")),
    )


def run_interface_slab_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Run surface tension, z-density Gibbs dividing surface fit, and P2 orientation profile for slab runs."""
    opts = (cfg.analyses or {}).get("options") or {}
    force_slab = bool(opts.get("force_slab", False))
    n_blocks = int(opts.get("n_blocks", 5))
    t_eq_ps = float(cfg.t_eq_ns) * 1000.0

    summary_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for run in runs:
        lz_mean_nm = float("nan")
        is_slab = force_slab
        species_profiles: dict[str, pd.DataFrame] = {}

        if run.tpr_path.exists() and run.xtc_path.exists():
            u = mda.Universe(str(run.tpr_path), str(run.xtc_path))
            f_start, f_stop, f_step = equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, stride=max(1, cfg.rdf_stride))
            frames = list(range(f_start, f_stop, f_step))[:200]
            if frames:
                u.trajectory[frames[0]]
                dims0 = np.asarray(u.trajectory.ts.dimensions[:3], dtype=float) / 10.0
                lz_mean_nm = float(dims0[2])
                if max(dims0[0], dims0[1]) > 0 and dims0[2] / max(dims0[0], dims0[1]) >= 1.5:
                    is_slab = True

                if is_slab:
                    species = [sp for sp, cnt in sorted(run.molecule_counts.items()) if cnt > 0 and sp in run.topologies]
                    ags = {sp: u.select_atoms(f"resname {sp}") for sp in species}
                    boxes = []
                    z_map: dict[str, list[np.ndarray]] = {sp: [] for sp in species}
                    u_map: dict[str, list[np.ndarray]] = {sp: [] for sp in species}
                    for fi in frames:
                        u.trajectory[fi]
                        ts = u.trajectory.ts
                        boxes.append(np.asarray(ts.dimensions[:3], dtype=float) / 10.0)
                        for sp in species:
                            if len(ags[sp]) > 0:
                                coms, uv, _ = _whole_molecule_positions_and_dipoles(ags[sp], run.topologies[sp], ts.dimensions)
                                z_map[sp].append(coms[:, 2])
                                u_map[sp].append(uv)
                    boxes_arr = np.asarray(boxes, dtype=float)
                    lz_mean_nm = float(np.mean(boxes_arr[:, 2]))
                    for sp in species:
                        if z_map[sp]:
                            prof = compute_z_density_and_orientation_profiles(
                                np.stack(z_map[sp], axis=0), np.stack(u_map[sp], axis=0), boxes_arr
                            )
                            species_profiles[sp] = prof
                            for _, pr in prof.iterrows():
                                profile_rows.append(
                                    {
                                        "run_id": run.run_id,
                                        "system_code": run.system_code,
                                        "replica": run.replica,
                                        "species": sp,
                                        **pr.to_dict(),
                                    }
                                )

        if not is_slab:
            continue

        # Surface tension from stress.xvg or thermo.xvg
        st_res = {"gamma_mn_m": float("nan"), "gamma_sem_mn_m": float("nan"), "delta_p_bar": float("nan")}
        for xvg_cand in (run.stress_path, run.thermo_path):
            if xvg_cand is not None and xvg_cand.exists():
                df_p = read_xvg_dataframe(xvg_cand)
                if df_p.empty:
                    continue
                eq_p = df_p[df_p.iloc[:, 0] >= t_eq_ps]
                if len(eq_p) < 4:
                    eq_p = df_p
                cxx, cyy, czz = _find_pressure_cols(eq_p)
                if cxx and cyy and czz and np.isfinite(lz_mean_nm):
                    st_res = compute_surface_tension(
                        eq_p[cxx].to_numpy(dtype=float),
                        eq_p[cyy].to_numpy(dtype=float),
                        eq_p[czz].to_numpy(dtype=float),
                        lz_mean_nm,
                        n_blocks=n_blocks,
                    )
                    break

        # Fit total number density profile to tanh slab model
        fit_res = {
            "z_center_nm": float("nan"),
            "rho_bulk_liquid": float("nan"),
            "rho_vapor": float("nan"),
            "gibbs_half_thickness_nm": float("nan"),
            "interfacial_width_nm": float("nan"),
            "fit_r_squared": float("nan"),
        }
        if species_profiles:
            first_prof = next(iter(species_profiles.values()))
            z_grid = first_prof["z_nm"].to_numpy(dtype=float)
            rho_tot = np.zeros_like(z_grid)
            for prof in species_profiles.values():
                rho_tot += prof["number_density_nm3"].to_numpy(dtype=float)
            fit_res = fit_slab_tanh_profile(z_grid, rho_tot)

        summary_rows.append(
            {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "replica": run.replica,
                "lz_mean_nm": lz_mean_nm,
                **st_res,
                **fit_res,
            }
        )

    return {
        "interface_slab_summary": pd.DataFrame(summary_rows),
        "interface_z_profiles": pd.DataFrame(profile_rows),
    }
