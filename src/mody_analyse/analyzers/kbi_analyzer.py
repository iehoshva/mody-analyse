from __future__ import annotations

import math
import re
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid

from mody_analyse.core.config import Config
from mody_analyse.core.discovery import RunRecord, compute_whole_coms_vectorized, equilibrated_frame_slice, read_xvg


AVOGADRO = 6.02214076e23
NM3_TO_CM3_MOL = 1e-21 * AVOGADRO  # ~602.214076


def rebin_rdf(r: np.ndarray, g: np.ndarray, bin_width_nm: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    if bin_width_nm <= 0 or len(r) < 2:
        return r.copy(), g.copy()
    r_max = float(r[-1])
    edges = np.arange(0.0, r_max + bin_width_nm * 0.5, bin_width_nm)
    if len(edges) < 2:
        return r.copy(), g.copy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    g_out = np.zeros_like(centers)
    w = np.maximum(r, 1e-9) ** 2
    for idx in range(len(centers)):
        mask = (r >= edges[idx]) & (r < edges[idx + 1])
        if np.any(mask):
            denom = np.sum(w[mask])
            g_out[idx] = float(np.sum(g[mask] * w[mask]) / denom) if denom > 0 else float(np.mean(g[mask]))
        else:
            g_out[idx] = float(np.interp(centers[idx], r, g))
    return centers, g_out


def apply_gvdv_correction(
    r: np.ndarray,
    g_raw: np.ndarray,
    rho_j: float,
    n_j: int,
    is_like_pair: bool,
) -> np.ndarray:
    """Ganguly & van der Vegt (2013) closed-box finite-size RDF tail correction."""
    if n_j <= 1 or rho_j <= 0:
        return g_raw.copy()
    v_box = n_j / rho_j
    v_sph = (4.0 * math.pi / 3.0) * (r ** 3)
    integrand = 4.0 * math.pi * rho_j * g_raw * (r ** 2)
    delta_n = cumulative_trapezoid(integrand, r, initial=0.0)
    delta_ij = 1.0 if is_like_pair else 0.0
    denom = np.maximum(n_j - delta_ij - delta_n, 1e-6)
    numer = n_j * np.maximum(1.0 - v_sph / v_box, 1e-6)
    return g_raw * (numer / denom)


def compute_running_kbi(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    integrand = 4.0 * math.pi * (g - 1.0) * (r ** 2)
    g_nm3 = cumulative_trapezoid(integrand, r, initial=0.0)
    return g_nm3 * NM3_TO_CM3_MOL


def kruger_extrapolate(
    r: np.ndarray,
    g: np.ndarray,
    r_fit_min: float = 0.75,
    r_fit_max: float = 1.15,
) -> float:
    """Kruger et al. (2013) spherical-subvolume kernel w(x) = 1 - 1.5x + 0.5x^3 over r <= 2R."""
    r_max_avail = float(r[-1])
    upper_r = min(r_fit_max, 0.49 * r_max_avail)
    lower_r = min(r_fit_min, upper_r * 0.7)
    R_vals = np.linspace(lower_r, upper_r, 9)
    gv_list = []
    for R in R_vals:
        mask = r <= 2.0 * R
        if np.sum(mask) < 3:
            continue
        rm = r[mask]
        x = rm / (2.0 * R)
        w = 1.0 - 1.5 * x + 0.5 * (x ** 3)
        gv_nm3 = 4.0 * math.pi * trapezoid((g[mask] - 1.0) * (rm ** 2) * w, rm)
        gv_list.append(gv_nm3 * NM3_TO_CM3_MOL)
    if len(gv_list) < 3:
        G_run = compute_running_kbi(r, g)
        return float(G_run[-1])
    R_used = R_vals[: len(gv_list)]
    slope, _ = np.polyfit(R_used, R_used * np.asarray(gv_list), 1)
    return float(slope)


def solve_ternary_kb_matrix(G_cm3_mol: np.ndarray, rho_nm3: np.ndarray) -> dict[str, Any]:
    """Compute N-component KB susceptibility matrix, constant-(T,P) Gamma, PMVs, and pseudo-binary Gamma."""
    G_nm3 = np.asarray(G_cm3_mol, dtype=float) / NM3_TO_CM3_MOL
    rho = np.asarray(rho_nm3, dtype=float)
    n_comp = len(rho)
    B = np.diag(rho) + np.outer(rho, rho) * G_nm3
    cond = float(np.linalg.cond(B))
    A = np.linalg.pinv(B) if cond > 1e10 else np.linalg.inv(B)

    gamma_tv = np.diag(rho) @ A
    denom_pmv = float(rho @ A @ rho)
    pmv_nm3 = (A @ rho) / denom_pmv if abs(denom_pmv) > 1e-18 else np.full(n_comp, float("nan"))
    pmv_cm3_mol = pmv_nm3 * NM3_TO_CM3_MOL

    if n_comp < 2:
        return {
            "ternary_gamma_matrix": gamma_tv,
            "ternary_tp_gamma_2x2": np.empty((0, 0), dtype=float),
            "partial_molar_volumes_cm3_mol": pmv_cm3_mol,
            "pseudo_binary_gamma_factor": 1.0,
            "matrix_condition_number": cond,
        }

    rho_tot = float(np.sum(rho))
    x = rho / rho_tot
    gamma_tp = np.zeros((n_comp - 1, n_comp - 1), dtype=float)
    for j in range(n_comp - 1):
        drho_dxj = np.zeros(n_comp, dtype=float)
        for k in range(n_comp):
            d_kj = 1.0 if k == j else 0.0
            d_kK = 1.0 if k == (n_comp - 1) else 0.0
            drho_dxj[k] = rho_tot * ((d_kj - d_kK) - rho[k] * (pmv_nm3[j] - pmv_nm3[-1]))
        dmu_dxj = A @ drho_dxj
        for i in range(n_comp - 1):
            gamma_tp[i, j] = x[j] * dmu_dxj[i]

    if n_comp >= 3:
        # Treat components 0..-2 as pooled solvent pseudocomponent vs last component (or solute)
        x_des = float(np.sum(x[:-1]))
        x_sol = float(x[-1])
        w = x[:-1] / max(x_des, 1e-12)
        G_dd = float(w @ G_nm3[:-1, :-1] @ w)
        G_ds = float(w @ G_nm3[:-1, -1])
        G_ss = float(G_nm3[-1, -1])
        delta_12 = G_dd + G_ss - 2.0 * G_ds
        pseudo_gamma = 1.0 / max(1.0 + x_des * x_sol * rho_tot * delta_12, 1e-6)
    else:
        delta_12 = G_nm3[0, 0] + G_nm3[1, 1] - 2.0 * G_nm3[0, 1]
        pseudo_gamma = 1.0 / max(1.0 + x[0] * x[1] * rho_tot * delta_12, 1e-6)

    return {
        "ternary_gamma_matrix": gamma_tv,
        "ternary_tp_gamma_2x2": gamma_tp,
        "partial_molar_volumes_cm3_mol": pmv_cm3_mol,
        "pseudo_binary_gamma_factor": float(pseudo_gamma),
        "matrix_condition_number": cond,
    }


def _pair_species_from_stem(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    return parts[1], parts[2]


def _verify_rdf_with_mdanalysis(run: RunRecord, cfg: Config) -> pd.DataFrame:
    """Fast equilibrated whole-molecule COM RDF calculation/verification on trajectory."""
    u = mda.Universe(str(run.tpr_path), str(run.xtc_path))
    start_idx, stop_idx, _ = equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, stride=1)
    rows = []
    species = [s for s in run.molecule_counts if s in run.topologies]
    if not species or stop_idx <= start_idx:
        return pd.DataFrame()
    step = max(1, (stop_idx - start_idx) // 400)
    r_edges = np.linspace(0.0, cfg.kbi_rmax_nm, 121)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    pair_list = [(species[i], species[j]) for i in range(len(species)) for j in range(i, len(species))]
    hists = {p: np.zeros(len(r_centers), dtype=float) for p in pair_list}
    vol_sum = 0.0
    n_used = 0
    sp_ag = {s: u.select_atoms(f"resname {s}") for s in species}

    for ts in u.trajectory[start_idx:stop_idx:step]:
        box_a = ts.dimensions[:3]
        box_nm = box_a / 10.0
        vol_sum += float(np.prod(box_nm))
        n_used += 1
        coms_nm = {
            s: compute_whole_coms_vectorized(
                sp_ag[s].positions,
                n_mol=run.molecule_counts[s],
                n_atoms=run.topologies[s].n_atoms,
                masses=run.topologies[s].masses,
                box_a=box_a,
                anchor_idx=run.topologies[s].center_atom_0based,
            ) / 10.0
            for s in species
        }
        for s1, s2 in pair_list:
            c1 = coms_nm[s1]
            c2 = coms_nm[s2]
            diff = c1[:, None, :] - c2[None, :, :]
            diff -= box_nm[None, None, :] * np.round(diff / box_nm[None, None, :])
            dists = np.linalg.norm(diff, axis=-1)
            if s1 == s2:
                iu = np.triu_indices(len(c1), k=1)
                d_vals = dists[iu]
            else:
                d_vals = dists.ravel()
            counts, _ = np.histogram(d_vals, bins=r_edges)
            hists[(s1, s2)] += counts * (2.0 if s1 == s2 else 1.0)

    v_mean = vol_sum / max(n_used, 1)
    shell_vol = (4.0 * math.pi / 3.0) * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    for s1, s2 in pair_list:
        n1 = run.molecule_counts[s1]
        n2 = run.molecule_counts[s2]
        norm = n_used * n1 * n2 * (shell_vol / v_mean)
        g_mda = hists[(s1, s2)] / np.maximum(norm, 1e-12)
        rows.append({"run_id": run.run_id, "pair": f"{s1}_{s2}", "r_nm": r_centers, "g_mda": g_mda})
    return pd.DataFrame(rows)


def _load_post_eq_block_rdfs(run: RunRecord, stem: str, t_eq_ns: float) -> list[np.ndarray]:
    blocks: list[tuple[float, np.ndarray]] = []
    if not run.block_rdf_paths:
        return []
    pat = re.compile(rf"^{re.escape(stem)}_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)ns$")
    for bp in run.block_rdf_paths:
        m = pat.match(bp.stem)
        if not m:
            continue
        t0 = float(m.group(1))
        if t0 >= t_eq_ns - 1e-6:
            _, b_arr = read_xvg(bp)
            if b_arr.size > 0:
                blocks.append((t0, b_arr[:, 1]))
    blocks.sort(key=lambda x: x[0])
    return [b for _, b in blocks]


def run_kbi_analysis(runs: list[RunRecord], cfg: Config, verify_mda: bool = True) -> dict[str, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    sol_solv_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for run in runs:
        t_cols, t_arr = read_xvg(run.thermo_path)
        t_ps = t_arr[:, t_cols.index("Time")]
        vol = t_arr[:, t_cols.index("Volume")]
        eq_mask = (t_ps >= cfg.t_eq_ns * 1000.0) & (t_ps <= cfg.t_end_ns * 1000.0)
        if not np.any(eq_mask):
            eq_mask = t_ps >= float(np.min(t_ps))
        vol_mean_nm3 = float(np.mean(vol[eq_mask]))
        rho_map = {sp: cnt / vol_mean_nm3 for sp, cnt in run.molecule_counts.items()}

        need_mda = verify_mda or not run.rdf_paths
        mda_df = _verify_rdf_with_mdanalysis(run, cfg) if need_mda else pd.DataFrame()
        species = list(run.molecule_counts.keys())
        sp_idx = {sp: idx for idx, sp in enumerate(species)}
        G_mat = np.zeros((len(species), len(species)), dtype=float)

        # Gather pairs from pre-extracted rdf_*.xvg or auto-fallback to MDAnalysis RDFs
        pair_sources: list[tuple[str, str, str, np.ndarray, np.ndarray, list[np.ndarray]]] = []
        if run.rdf_paths:
            for stem, rdf_path in sorted(run.rdf_paths.items()):
                s1, s2 = _pair_species_from_stem(stem)
                if s1 not in rho_map or s2 not in rho_map:
                    continue
                _, raw_arr = read_xvg(rdf_path)
                r_raw, g_raw = raw_arr[:, 0], raw_arr[:, 1]
                eq_blocks = _load_post_eq_block_rdfs(run, stem, cfg.t_eq_ns)
                g_eq_mean = np.mean(eq_blocks, axis=0) if eq_blocks else g_raw
                pair_sources.append((stem, s1, s2, r_raw, g_eq_mean, eq_blocks))
        elif not mda_df.empty:
            for _, mrow in mda_df.iterrows():
                s1, s2 = str(mrow["pair"]).split("_", 1)
                pair_sources.append((f"rdf_{s1}_{s2}", s1, s2, np.asarray(mrow["r_nm"]), np.asarray(mrow["g_mda"]), []))

        for stem, s1, s2, r_raw, g_eq_mean, eq_blocks in pair_sources:
            r_reb, g_reb = rebin_rdf(r_raw, g_eq_mean, bin_width_nm=cfg.kbi_rebin_nm)
            g_corr = apply_gvdv_correction(
                r_reb,
                g_reb,
                rho_j=rho_map[s2],
                n_j=run.molecule_counts[s2],
                is_like_pair=(s1 == s2),
            )
            G_raw_run = compute_running_kbi(r_reb, g_reb)
            G_corr_run = compute_running_kbi(r_reb, g_corr)

            r_min, r_max = cfg.kbi_plateau_range_nm
            plat_mask = (r_reb >= r_min) & (r_reb <= r_max)
            if not np.any(plat_mask):
                plat_mask = r_reb >= 0.6 * float(r_reb[-1])
            G_plat_corr = float(np.mean(G_corr_run[plat_mask]))
            G_plat_raw = float(np.mean(G_raw_run[plat_mask]))
            G_kruger = kruger_extrapolate(r_reb, g_corr, 0.75, min(1.15, float(r_reb[-1]) * 0.48))

            blk_Gs = []
            for g_b in eq_blocks:
                r_b, g_b_reb = rebin_rdf(r_raw, g_b, bin_width_nm=cfg.kbi_rebin_nm)
                g_b_corr = apply_gvdv_correction(r_b, g_b_reb, rho_map[s2], run.molecule_counts[s2], s1 == s2)
                blk_Gs.append(kruger_extrapolate(r_b, g_b_corr, 0.75, min(1.15, float(r_b[-1]) * 0.48)))
            G_blk_sd = float(np.std(blk_Gs, ddof=1)) if len(blk_Gs) > 1 else 0.0

            mda_l2 = float("nan")
            if verify_mda and not mda_df.empty and run.rdf_paths:
                m_sub = mda_df.loc[mda_df["pair"].isin([f"{s1}_{s2}", f"{s2}_{s1}"])]
                if not m_sub.empty:
                    g_mda_interp = np.interp(r_reb, m_sub.iloc[0]["r_nm"], m_sub.iloc[0]["g_mda"])
                    mda_l2 = float(np.sqrt(np.mean((g_mda_interp - g_reb) ** 2)))

            i1, i2 = sp_idx[s1], sp_idx[s2]
            G_mat[i1, i2] = G_kruger
            G_mat[i2, i1] = G_kruger

            delta_N_12 = rho_map[s2] * (G_kruger / NM3_TO_CM3_MOL)
            delta_N_21 = rho_map[s1] * (G_kruger / NM3_TO_CM3_MOL)

            pair_rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "pair": f"{s1}-{s2}",
                    "species_1": s1,
                    "species_2": s2,
                    "tail_raw_1p6_2p4": float(np.mean(g_reb[r_reb >= min(1.6, 0.75 * float(r_reb[-1]))])),
                    "tail_gvdv_1p6_2p4": float(np.mean(g_corr[r_reb >= min(1.6, 0.75 * float(r_reb[-1]))])),
                    "G_raw_plateau_cm3_mol": G_plat_raw,
                    "G_gvdv_plateau_cm3_mol": G_plat_corr,
                    "G_kruger_inf_cm3_mol": G_kruger,
                    "G_block_sd_cm3_mol": G_blk_sd,
                    "delta_N_12": float(delta_N_12),
                    "delta_N_21": float(delta_N_21),
                    "mda_vs_xvg_l2": mda_l2,
                }
            )
            for rr, gr, gc, Gr, Gc in zip(r_reb, g_reb, g_corr, G_raw_run, G_corr_run):
                curve_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "pair": f"{s1}-{s2}",
                        "r_nm": float(rr),
                        "g_raw": float(gr),
                        "g_gvdv": float(gc),
                        "G_raw_cm3_mol": float(Gr),
                        "G_gvdv_cm3_mol": float(Gc),
                    }
                )

        # Ensure solute appears last when building constant-(T,P) matrix for pseudo-binary diagnostics
        ordered_species = list(run.solvent_resnames) + ([run.solute_resname] if run.solute_resname and run.solute_resname in species else [])
        for sp in species:
            if sp not in ordered_species:
                ordered_species.append(sp)
        perm = [sp_idx[s] for s in ordered_species]
        rho_vec = np.asarray([rho_map[s] for s in ordered_species], dtype=float)
        G_ord = G_mat[np.ix_(perm, perm)]
        kb_sol = solve_ternary_kb_matrix(G_ord, rho_vec)

        pmvs = kb_sol["partial_molar_volumes_cm3_mol"]
        pmv_map = {sp: float(pmvs[i]) for i, sp in enumerate(ordered_species)}

        sys_entry: dict[str, Any] = {
            "run_id": run.run_id,
            "system_code": run.system_code,
            "replica": run.replica,
            "solute": run.solute_resname or "NONE",
            "primary_solvent": run.solvent_resnames[0] if run.solvent_resnames else "NONE",
            "vol_mean_nm3": vol_mean_nm3,
            "pmv_SOL_cm3_mol": pmv_map.get(run.solute_resname, float("nan")) if run.solute_resname else float("nan"),
            "pseudo_binary_gamma_factor": kb_sol["pseudo_binary_gamma_factor"],
            "matrix_cond": kb_sol["matrix_condition_number"],
        }
        for sp, pmv_v in pmv_map.items():
            sys_entry[f"pmv_{sp}_cm3_mol"] = pmv_v

        delta_s_primary = float("nan")
        if run.solute_resname is not None and run.solute_resname in sp_idx:
            si = sp_idx[run.solute_resname]
            solvs = list(run.solvent_resnames)
            for s_idx, solv in enumerate(solvs):
                if solv in sp_idx:
                    g_val = float(G_mat[si, sp_idx[solv]])
                    sys_entry[f"G_S_{solv}_cm3_mol"] = g_val
                    if s_idx == 0:
                        sys_entry["G_S_primary_cm3_mol"] = g_val
                    elif s_idx == 1:
                        sys_entry["G_S_secondary_cm3_mol"] = g_val
                    sol_solv_rows.append(
                        {
                            "run_id": run.run_id,
                            "system_code": run.system_code,
                            "replica": run.replica,
                            "solute": run.solute_resname,
                            "solvent": solv,
                            "G_sol_solv_cm3_mol": g_val,
                            "delta_N_sol_solv": float(rho_map[solv] * (g_val / NM3_TO_CM3_MOL)),
                        }
                    )
            if len(solvs) >= 2 and solvs[0] in sp_idx and solvs[1] in sp_idx:
                p_solv = cfg.primary_solvent if (cfg.primary_solvent in sp_idx and cfg.primary_solvent in solvs) else solvs[0]
                sec_solv = next(s for s in solvs if s != p_solv)
                g_s1 = float(G_mat[si, sp_idx[p_solv]])
                g_s2 = float(G_mat[si, sp_idx[sec_solv]])
                tot_solv_cnt = sum(run.molecule_counts[s] for s in solvs)
                x1 = run.molecule_counts[p_solv] / max(tot_solv_cnt, 1)
                x2 = 1.0 - x1
                delta_g_nm3 = (g_s1 - g_s2) / NM3_TO_CM3_MOL
                v_cor_nm3 = (4.0 * math.pi / 3.0) * (0.8**3)
                delta_s_primary = float(
                    x1 * x2 * delta_g_nm3 / (v_cor_nm3 + x1 * (g_s1 / NM3_TO_CM3_MOL) + x2 * (g_s2 / NM3_TO_CM3_MOL))
                )
                sys_entry[f"delta_S_{p_solv}"] = delta_s_primary

        sys_entry["delta_S_primary"] = delta_s_primary
        system_rows.append(sys_entry)

    pairs_df = pd.DataFrame(pair_rows)
    sys_df = pd.DataFrame(system_rows)
    curves_df = pd.DataFrame(curve_rows)
    sol_solv_df = pd.DataFrame(sol_solv_rows)

    # Aggregate dynamic numeric columns across replicates cleanly
    summary_rows: list[dict[str, Any]] = []
    if not sys_df.empty:
        g_cols = [c for c in sys_df.columns if c.startswith("G_S_") or c.startswith("delta_S_") or c == "pseudo_binary_gamma_factor"]
        for sys_code, grp in sys_df.groupby("system_code"):
            srow: dict[str, Any] = {"system_code": sys_code, "n_rep": len(grp)}
            for col in g_cols:
                vals = grp[col].dropna()
                srow[col] = float(vals.mean()) if not vals.empty else float("nan")
                if col.endswith("_cm3_mol"):
                    sd_col = col.replace("_cm3_mol", "_sd")
                else:
                    sd_col = f"{col}_sd"
                srow[sd_col] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            summary_rows.append(srow)
    summary_df = pd.DataFrame(summary_rows)

    pairs_summary_df = (
        pairs_df.groupby(["system_code", "pair"], as_index=False)
        .agg(
            n_rep=("replica", "count"),
            tail_raw_mean=("tail_raw_1p6_2p4", "mean"),
            tail_gvdv_mean=("tail_gvdv_1p6_2p4", "mean"),
            G_raw_plateau_cm3_mol=("G_raw_plateau_cm3_mol", "mean"),
            G_gvdv_plateau_cm3_mol=("G_gvdv_plateau_cm3_mol", "mean"),
            G_gvdv_plateau_sd=("G_gvdv_plateau_cm3_mol", "std"),
            G_kruger_inf_cm3_mol=("G_kruger_inf_cm3_mol", "mean"),
            G_kruger_inf_sd=("G_kruger_inf_cm3_mol", "std"),
            delta_N_12=("delta_N_12", "mean"),
            delta_N_21=("delta_N_21", "mean"),
            mda_vs_xvg_l2_mean=("mda_vs_xvg_l2", "mean"),
        )
        if not pairs_df.empty
        else pd.DataFrame()
    )
    return {
        "kbi_pairs": pairs_df,
        "kbi_pairs_summary": pairs_summary_df,
        "kbi_solute_solvent": sol_solv_df,
        "kbi_systems": sys_df,
        "kbi_summary": summary_df,
        "kbi_curves": curves_df,
    }
