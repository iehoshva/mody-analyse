from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import networkx as nx
import numpy as np
import pandas as pd

from mody_analyse.analyzers.kbi_analyzer import rebin_rdf
from mody_analyse.core.config import Config
from mody_analyse.core.discovery import (
    MoleculeTopology,
    RunRecord,
    compute_whole_coms_vectorized,
    equilibrated_frame_slice,
    read_xvg,
)


def load_unwrapped_universe(run: RunRecord) -> mda.Universe:
    return mda.Universe(str(run.tpr_path), str(run.xtc_path))


def _equilibrated_frame_slice(u: mda.Universe, t_eq_ns: float, t_end_ns: float, stride: int) -> tuple[int, int, int]:
    return equilibrated_frame_slice(u, t_eq_ns, t_end_ns, stride)


def _min_image(vecs: np.ndarray, box_a: np.ndarray) -> np.ndarray:
    return vecs - box_a[None, :] * np.round(vecs / box_a[None, :])


def detect_first_shell_cutoff(
    r_nm: np.ndarray,
    g: np.ndarray,
    r_min: float = 0.35,
    r_max: float = 1.25,
) -> float:
    mask = (r_nm >= r_min) & (r_nm <= r_max)
    r_sub = r_nm[mask]
    g_sub = g[mask]
    if len(r_sub) < 5:
        return 0.75
    if len(g_sub) >= 5:
        g_sm = g_sub.copy()
        g_sm[1:-1] = 0.25 * g_sub[:-2] + 0.50 * g_sub[1:-1] + 0.25 * g_sub[2:]
    else:
        g_sm = g_sub
    peak_idx = int(np.argmax(g_sm))
    if peak_idx >= len(r_sub) - 2:
        return float(r_sub[-1])
    g_peak = float(g_sm[peak_idx])
    post_peak_g = g_sm[peak_idx + 1 :]
    post_peak_r = r_sub[peak_idx + 1 :]
    depth_thresh = 1.0 + 0.30 * max(g_peak - 1.0, 0.0)
    for i in range(1, len(post_peak_g) - 1):
        if (
            post_peak_g[i] <= post_peak_g[i - 1]
            and post_peak_g[i] <= post_peak_g[i + 1]
            and post_peak_g[i] <= depth_thresh
        ):
            return float(post_peak_r[i])
    min_idx = int(np.argmin(post_peak_g))
    return float(post_peak_r[min_idx])


def compute_multispecies_enrichment(
    n_shell_by_sp: dict[str, float],
    n_bulk_by_sp: dict[str, int],
    mol_vols_cm3_mol: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute local composition, molar enrichment, and volume-weighted enrichment for arbitrary N solvents."""
    solvents = list(n_bulk_by_sp.keys())
    tot_shell = sum(n_shell_by_sp.get(s, 0.0) for s in solvents)
    tot_bulk = sum(n_bulk_by_sp.values())
    vols = mol_vols_cm3_mol or {s: 100.0 for s in solvents}
    v_shell_tot = sum(n_shell_by_sp.get(s, 0.0) * vols.get(s, 100.0) for s in solvents)
    v_bulk_tot = sum(n_bulk_by_sp[s] * vols.get(s, 100.0) for s in solvents)

    out: dict[str, dict[str, float]] = {}
    for s in solvents:
        n_sh = float(n_shell_by_sp.get(s, 0.0))
        x_loc = n_sh / tot_shell if tot_shell > 0 else float("nan")
        x_blk = n_bulk_by_sp[s] / tot_bulk if tot_bulk > 0 else float("nan")
        e_mol = x_loc / x_blk if (x_blk and np.isfinite(x_loc)) else float("nan")
        phi_loc = (n_sh * vols.get(s, 100.0)) / max(v_shell_tot, 1e-12) if tot_shell > 0 else float("nan")
        phi_blk = (n_bulk_by_sp[s] * vols.get(s, 100.0)) / max(v_bulk_tot, 1e-12)
        e_vol = phi_loc / phi_blk if (phi_blk > 0 and np.isfinite(phi_loc)) else float("nan")
        out[s] = {
            "x_local": float(x_loc),
            "x_bulk": float(x_blk),
            "enrichment_molar": float(e_mol),
            "phi_local": float(phi_loc),
            "phi_bulk": float(phi_blk),
            "enrichment_volume": float(e_vol),
        }
    return out


def compute_enrichment_metrics(
    n_thy_shell: float,
    n_dod_shell: float,
    n_thy_bulk: int,
    n_dod_bulk: int,
    mw_thy: float = 150.22,
    mw_dod: float = 186.34,
    rho_thy_gcm3: float = 0.950398,
    rho_dod_gcm3: float = 0.81255,
) -> dict[str, float]:
    """Binary two-solvent local enrichment calculator (retained for direct binary API compatibility)."""
    m = compute_multispecies_enrichment(
        {"S1": n_thy_shell, "S2": n_dod_shell},
        {"S1": n_thy_bulk, "S2": n_dod_bulk},
        {"S1": mw_thy / rho_thy_gcm3, "S2": mw_dod / rho_dod_gcm3},
    )
    return {
        "x_thy_local": m["S1"]["x_local"],
        "x_dod_local": m["S2"]["x_local"],
        "x_thy_bulk": m["S1"]["x_bulk"],
        "enrichment_thy_molar": m["S1"]["enrichment_molar"],
        "enrichment_dod_molar": m["S2"]["enrichment_molar"],
        "phi_thy_local": m["S1"]["phi_local"],
        "phi_thy_bulk": m["S1"]["phi_bulk"],
        "enrichment_thy_volume": m["S1"]["enrichment_volume"],
    }


def _compute_auto_com_rdf(u: mda.Universe, run: RunRecord, sol: str, solv: str, start: int, stop: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    r_edges = np.linspace(0.0, 1.4, 57)  # 0.025 nm bins
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(len(r_centers), dtype=float)
    sol_ag = u.select_atoms(f"resname {sol}")
    solv_ag = u.select_atoms(f"resname {solv}")
    sol_tp, solv_tp = run.topologies[sol], run.topologies[solv]
    n_s, n_v = run.molecule_counts[sol], run.molecule_counts[solv]
    vol_sum = 0.0
    n_fr = 0
    step = max(stride, (stop - start) // 150)
    for ts in u.trajectory[start:stop:step]:
        box_nm = ts.dimensions[:3] / 10.0
        vol_sum += float(np.prod(box_nm))
        n_fr += 1
        c1 = compute_whole_coms_vectorized(sol_ag.positions, n_s, sol_tp.n_atoms, sol_tp.masses, ts.dimensions[:3], sol_tp.center_atom_0based) / 10.0
        c2 = compute_whole_coms_vectorized(solv_ag.positions, n_v, solv_tp.n_atoms, solv_tp.masses, ts.dimensions[:3], solv_tp.center_atom_0based) / 10.0
        diff = _min_image((c1[:, None, :] - c2[None, :, :]).reshape(-1, 3), box_nm)
        cnt, _ = np.histogram(np.linalg.norm(diff, axis=-1), bins=r_edges)
        hist += cnt
    v_mean = vol_sum / max(n_fr, 1)
    shell_vol = (4.0 * math.pi / 3.0) * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    norm = max(n_fr * n_s * n_v, 1) * (shell_vol / v_mean)
    return r_centers, hist / np.maximum(norm, 1e-12)


def analyze_solvation_shells(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Analysis #1: First solvation shell counts, local enrichment (N solvents), and site RDFs."""
    run_rows: list[dict[str, Any]] = []
    pair_enr_rows: list[dict[str, Any]] = []
    sens_rows: list[dict[str, Any]] = []
    site_rdf_rows: list[dict[str, Any]] = []

    site_edges_a = np.linspace(0.0, 14.0, 71)
    site_centers_nm = 0.5 * (site_edges_a[:-1] + site_edges_a[1:]) / 10.0
    site_shell_vol_a3 = (4.0 * math.pi / 3.0) * (site_edges_a[1:] ** 3 - site_edges_a[:-1] ** 3)

    for run in runs:
        if run.solute_resname is None:
            continue
        sol = run.solute_resname
        solvents = [s for s in run.solvent_resnames if s in run.topologies and run.molecule_counts.get(s, 0) > 0]
        if not solvents:
            continue

        u = load_unwrapped_universe(run)
        start, stop, stride = _equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, cfg.rdf_stride)

        rcuts_nm: dict[str, float] = {}
        for solv in solvents:
            key = f"rdf_{solv}_{sol}" if f"rdf_{solv}_{sol}" in run.rdf_paths else f"rdf_{sol}_{solv}"
            if key in run.rdf_paths:
                _, arr = read_xvg(run.rdf_paths[key])
                r_raw, g_raw = arr[:, 0], arr[:, 1]
                r_reb, g_reb = rebin_rdf(r_raw, g_raw, bin_width_nm=0.025)
                rcuts_nm[solv] = detect_first_shell_cutoff(r_reb, g_reb, r_min=0.35, r_max=1.15)
            else:
                r_reb, g_reb = _compute_auto_com_rdf(u, run, sol, solv, start, stop, stride)
                rcuts_nm[solv] = detect_first_shell_cutoff(r_reb, g_reb, r_min=0.35, r_max=1.15)

        rcut_common_nm = float(np.mean(list(rcuts_nm.values())))
        sol_ag = u.select_atoms(f"resname {sol}")
        sol_topo = run.topologies[sol]
        sol_hmask = np.asarray(sol_topo.elements) != "H"
        solv_ags = {s: u.select_atoms(f"resname {s}") for s in solvents}
        solv_topos = {s: run.topologies[s] for s in solvents}
        solv_hmasks = {s: (np.asarray(solv_topos[s].elements) != "H") for s in solvents}

        n_per_sol: dict[str, list[float]] = {s: [] for s in solvents}
        n_common: dict[str, list[float]] = {s: [] for s in solvents}
        n_065: dict[str, list[float]] = {s: [] for s in solvents}
        n_075: dict[str, list[float]] = {s: [] for s in solvents}
        n_085: dict[str, list[float]] = {s: [] for s in solvents}
        n_contact: dict[str, list[float]] = {s: [] for s in solvents}

        rcut_common_a = rcut_common_nm * 10.0
        max_com_cut_a = max([rcuts_nm[s] * 10.0 for s in solvents] + [rcut_common_a, 8.5])
        contact_cutoff_a = 4.5

        # Auto site selections via chemistry metadata + optional config overrides
        sol_chem = run.chemistry.get(sol)
        sol_polar_names = " ".join(sol_chem.polar_atoms) if sol_chem and sol_chem.polar_atoms else "all"
        sol_core_names = " ".join(sol_chem.ring_atoms) if sol_chem and sol_chem.ring_atoms else (
            " ".join([a for a, e in zip(sol_topo.atom_names, sol_topo.elements) if e != "H"][:6]) or "all"
        )
        sol_polar_ag = u.select_atoms(f"resname {sol} and name {sol_polar_names}")
        sol_core_ag = u.select_atoms(f"resname {sol} and name {sol_core_names}")

        site_pairs: list[tuple[str, Any, Any]] = []
        for s in solvents:
            s_chem = run.chemistry.get(s)
            if s_chem and s_chem.polar_atoms:
                p_sel = u.select_atoms(f"resname {s} and name {' '.join(s_chem.polar_atoms)}")
                site_pairs.append((f"SOL_head-{s}_{s_chem.polar_atoms[0]}", sol_polar_ag, p_sel))
            if s_chem and s_chem.ring_atoms:
                r_sel = u.select_atoms(f"resname {s} and name {' '.join(s_chem.ring_atoms)}")
                site_pairs.append((f"SOL_core-{s}_ring", sol_core_ag, r_sel))
            if s_chem and s_chem.tail_atoms:
                t_sel = u.select_atoms(f"resname {s} and name {' '.join(s_chem.tail_atoms)}")
                site_pairs.append((f"SOL_core-{s}_tail", sol_core_ag, t_sel))

        site_hists = {lbl: np.zeros(len(site_centers_nm), dtype=float) for lbl, _, _ in site_pairs}
        vol_a3_sum = 0.0
        n_frames_used = 0

        all_sol_heavy = u.select_atoms(f"resname {sol} and prop mass > 2.0")
        if len(all_sol_heavy) == 0:
            all_sol_heavy = sol_ag
        sol_atom_to_res = all_sol_heavy.resindices - all_sol_heavy.resindices.min()
        all_solv_heavy: dict[str, Any] = {}
        solv_atom_to_res: dict[str, np.ndarray] = {}
        for s in solvents:
            hg = u.select_atoms(f"resname {s} and prop mass > 2.0")
            if len(hg) == 0:
                hg = solv_ags[s]
            all_solv_heavy[s] = hg
            solv_atom_to_res[s] = hg.resindices - hg.resindices.min()

        n_s = run.molecule_counts[sol]
        for ts in u.trajectory[start:stop:stride]:
            box = ts.dimensions
            box_a = box[:3]
            vol_a3_sum += float(np.prod(box_a))
            n_frames_used += 1
            c_sol = compute_whole_coms_vectorized(
                sol_ag.positions, n_s, sol_topo.n_atoms, sol_topo.masses, box_a, sol_topo.center_atom_0based, sol_hmask
            ).astype(np.float32)

            for s in solvents:
                s_tp = solv_topos[s]
                c_s = compute_whole_coms_vectorized(
                    solv_ags[s].positions,
                    run.molecule_counts[s],
                    s_tp.n_atoms,
                    s_tp.masses,
                    box_a,
                    s_tp.center_atom_0based,
                    solv_hmasks[s],
                ).astype(np.float32)
                _, dists = capped_distance(c_sol, c_s, max_cutoff=max_com_cut_a, box=box, return_distances=True)
                n_per_sol[s].append(float(np.sum(dists <= rcuts_nm[s] * 10.0)) / n_s)
                n_common[s].append(float(np.sum(dists <= rcut_common_a)) / n_s)
                n_065[s].append(float(np.sum(dists <= 6.5)) / n_s)
                n_075[s].append(float(np.sum(dists <= 7.5)) / n_s)
                n_085[s].append(float(np.sum(dists <= 8.5)) / n_s)

                p_h = capped_distance(all_sol_heavy.positions, all_solv_heavy[s].positions, max_cutoff=contact_cutoff_a, box=box, return_distances=False)
                if len(p_h):
                    uniq = set(zip(sol_atom_to_res[p_h[:, 0]], solv_atom_to_res[s][p_h[:, 1]]))
                    n_contact[s].append(len(uniq) / n_s)
                else:
                    n_contact[s].append(0.0)

            for lbl, ag1, ag2 in site_pairs:
                if len(ag1) == 0 or len(ag2) == 0:
                    continue
                _, sdists = capped_distance(ag1.positions, ag2.positions, max_cutoff=14.0, box=box, return_distances=True)
                if len(sdists):
                    cnt, _ = np.histogram(sdists, bins=site_edges_a)
                    site_hists[lbl] += cnt

        v_mean_a3 = vol_a3_sum / max(n_frames_used, 1)
        for lbl, ag1, ag2 in site_pairs:
            if len(ag1) == 0 or len(ag2) == 0:
                continue
            norm = max(n_frames_used * len(ag1) * len(ag2), 1) * (site_shell_vol_a3 / v_mean_a3)
            g_site = site_hists[lbl] / np.maximum(norm, 1e-12)
            for rr, gg in zip(site_centers_nm, g_site):
                site_rdf_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "site_pair": lbl,
                        "r_nm": float(rr),
                        "g_r": float(gg),
                    }
                )

        n_bulk_map = {s: run.molecule_counts[s] for s in solvents}
        mol_vol_map = {
            s: (run.chemistry[s].molar_volume_cm3_mol if s in run.chemistry else 100.0)
            for s in solvents
        }

        means_rdf = {s: float(np.mean(n_per_sol[s])) for s in solvents}
        means_common = {s: float(np.mean(n_common[s])) for s in solvents}
        means_065 = {s: float(np.mean(n_065[s])) for s in solvents}
        means_075 = {s: float(np.mean(n_075[s])) for s in solvents}
        means_085 = {s: float(np.mean(n_085[s])) for s in solvents}
        means_contact = {s: float(np.mean(n_contact[s])) for s in solvents}

        enr_rdf = compute_multispecies_enrichment(means_rdf, n_bulk_map, mol_vol_map)
        enr_common = compute_multispecies_enrichment(means_common, n_bulk_map, mol_vol_map)
        enr_065 = compute_multispecies_enrichment(means_065, n_bulk_map, mol_vol_map)
        enr_075 = compute_multispecies_enrichment(means_075, n_bulk_map, mol_vol_map)
        enr_085 = compute_multispecies_enrichment(means_085, n_bulk_map, mol_vol_map)
        enr_contact = compute_multispecies_enrichment(means_contact, n_bulk_map, mol_vol_map)

        primary = cfg.primary_solvent if (cfg.primary_solvent in solvents) else solvents[0]
        row_entry: dict[str, Any] = {
            "run_id": run.run_id,
            "system_code": run.system_code,
            "replica": run.replica,
            "primary_solvent": primary,
            "rcut_common_nm": rcut_common_nm,
            "x_primary_local": enr_rdf[primary]["x_local"],
            "e_primary_molar": enr_rdf[primary]["enrichment_molar"],
            "e_primary_volume": enr_rdf[primary]["enrichment_volume"],
            "e_primary_common_molar": enr_common[primary]["enrichment_molar"],
            "e_primary_075nm_molar": enr_075[primary]["enrichment_molar"],
            "e_primary_contact_molar": enr_contact[primary]["enrichment_molar"],
        }

        for s in solvents:
            sl = s.lower()
            sd_s = float(np.std(n_per_sol[s], ddof=1)) if len(n_per_sol[s]) > 1 else 0.0
            row_entry[f"rcut_{sl}_nm"] = rcuts_nm[s]
            row_entry[f"n_{sl}_shell"] = means_rdf[s]
            row_entry[f"n_{sl}_shell_sd"] = sd_s
            row_entry[f"x_{sl}_local"] = enr_rdf[s]["x_local"]
            row_entry[f"e_{sl}_molar"] = enr_rdf[s]["enrichment_molar"]
            row_entry[f"e_{sl}_volume"] = enr_rdf[s]["enrichment_volume"]
            row_entry[f"n_{sl}_common"] = means_common[s]
            row_entry[f"e_{sl}_common_molar"] = enr_common[s]["enrichment_molar"]
            row_entry[f"e_{sl}_065nm_molar"] = enr_065[s]["enrichment_molar"]
            row_entry[f"e_{sl}_075nm_molar"] = enr_075[s]["enrichment_molar"]
            row_entry[f"e_{sl}_085nm_molar"] = enr_085[s]["enrichment_molar"]
            row_entry[f"n_{sl}_contact_4p5A"] = means_contact[s]
            row_entry[f"e_{sl}_contact_molar"] = enr_contact[s]["enrichment_molar"]

            pair_enr_rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "solute": sol,
                    "solvent": s,
                    "rcut_nm": rcuts_nm[s],
                    "n_shell": means_rdf[s],
                    "n_shell_sd": sd_s,
                    "x_local": enr_rdf[s]["x_local"],
                    "x_bulk": enr_rdf[s]["x_bulk"],
                    "enrichment_molar": enr_rdf[s]["enrichment_molar"],
                    "enrichment_volume": enr_rdf[s]["enrichment_volume"],
                    "e_075nm_molar": enr_075[s]["enrichment_molar"],
                    "e_contact_molar": enr_contact[s]["enrichment_molar"],
                }
            )
            for label, d_map in [
                ("rdf_minima", enr_rdf),
                ("common_com", enr_common),
                ("fixed_0.65nm", enr_065),
                ("fixed_0.75nm", enr_075),
                ("fixed_0.85nm", enr_085),
                ("contact_4.5A", enr_contact),
            ]:
                sens_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "solvent": s,
                        "definition": label,
                        **d_map[s],
                    }
                )
        run_rows.append(row_entry)

    runs_df = pd.DataFrame(run_rows)
    sens_df = pd.DataFrame(sens_rows)
    site_rdf_df = pd.DataFrame(site_rdf_rows)
    pair_enr_df = pd.DataFrame(pair_enr_rows)

    summary_rows: list[dict[str, Any]] = []
    if not runs_df.empty:
        num_cols = [c for c in runs_df.columns if c not in {"run_id", "system_code", "replica", "primary_solvent"}]
        for sc, grp in runs_df.groupby("system_code"):
            srow: dict[str, Any] = {
                "system_code": sc,
                "primary_solvent": grp["primary_solvent"].iloc[0],
                "n_rep": len(grp),
            }
            for col in num_cols:
                vals = grp[col].dropna()
                mean_v = float(vals.mean()) if not vals.empty else float("nan")
                sd_v = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
                srow[col] = mean_v
                # Clean naming + backward aliases for n_<solv>_mean and e_<solv>_molar_mean
                if col.endswith("_shell"):
                    base = col.replace("_shell", "")
                    srow[f"{base}_mean"] = mean_v
                    srow[f"{base}_sd"] = sd_v
                elif col.endswith("_molar") or col.endswith("_volume"):
                    srow[f"{col}_mean"] = mean_v
                    srow[f"{col}_sd"] = sd_v
                else:
                    srow[f"{col}_sd"] = sd_v
            summary_rows.append(srow)

    pair_enr_summary = (
        pair_enr_df.groupby(["system_code", "solvent"], as_index=False)
        .agg(
            n_rep=("replica", "count"),
            rcut_nm=("rcut_nm", "mean"),
            n_shell_mean=("n_shell", "mean"),
            n_shell_sd=("n_shell", "std"),
            x_local_mean=("x_local", "mean"),
            x_bulk=("x_bulk", "first"),
            enrichment_molar_mean=("enrichment_molar", "mean"),
            enrichment_molar_sd=("enrichment_molar", "std"),
            enrichment_volume_mean=("enrichment_volume", "mean"),
            e_075nm_mean=("e_075nm_molar", "mean"),
            e_contact_mean=("e_contact_molar", "mean"),
        )
        if not pair_enr_df.empty
        else pd.DataFrame()
    )

    return {
        "enrichment_runs": runs_df,
        "enrichment_solvent_pairs": pair_enr_df,
        "enrichment_solvent_pairs_summary": pair_enr_summary,
        "enrichment_sensitivity": sens_df,
        "enrichment_summary": pd.DataFrame(summary_rows),
        "site_rdfs": site_rdf_df,
    }


def compute_hbond_autocorr(
    presence_matrix: np.ndarray,
    dt_ps: float,
    max_lag_frames: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute intermittent H-bond ACF C(t) = <h(0)h(t)>/<h(0)> and effective relaxation time."""
    if presence_matrix.size == 0 or presence_matrix.shape[0] < 4:
        return np.array([0.0]), np.array([1.0]), 0.0
    x = presence_matrix.astype(float)
    mean_h = float(np.mean(x))
    if mean_h <= 0:
        return np.array([0.0]), np.array([1.0]), 0.0
    n_t = x.shape[0]
    max_lag = min(max_lag_frames, n_t // 2)
    acf = np.zeros(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        num = float(np.mean(x[:-lag] * x[lag:]))
        acf[lag] = num / mean_h
    tau_ps = np.arange(max_lag + 1, dtype=float) * dt_ps
    mask = acf > 0.02
    tau_trapz = float(np.trapezoid(acf[mask], tau_ps[mask])) if np.sum(mask) >= 3 else (
        float(dt_ps * acf[1]) if len(acf) > 1 else float(dt_ps)
    )
    tau_rel = tau_trapz
    if len(tau_ps) >= 8 and float(acf[-1]) < 0.85 and float(acf[1]) > 0.05:
        try:
            from scipy.optimize import curve_fit

            def _biexp(t: np.ndarray, a: float, t1: float, t2: float) -> np.ndarray:
                return a * np.exp(-t / max(t1, 1e-6)) + (1.0 - a) * np.exp(-t / max(t2, 1e-6))

            t_span = max(float(tau_ps[-1]), dt_ps * 2.0)
            p0 = [0.5, max(dt_ps, 0.1 * tau_trapz), max(dt_ps * 2.0, 2.0 * max(tau_trapz, dt_ps))]
            popt, _ = curve_fit(
                _biexp,
                tau_ps,
                np.clip(acf, 0.0, 1.0),
                p0=p0,
                bounds=([0.0, dt_ps * 0.1, dt_ps * 0.1], [1.0, t_span * 20.0, t_span * 20.0]),
                maxfev=2000,
            )
            pred = _biexp(tau_ps, *popt)
            ss_res = float(np.sum((acf - pred) ** 2))
            ss_tot = float(np.sum((acf - np.mean(acf)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            tau_biexp = float(popt[0] * popt[1] + (1.0 - popt[0]) * popt[2])
            if np.isfinite(tau_biexp) and r2 >= 0.85 and tau_biexp > 0.0:
                tau_rel = tau_biexp
        except Exception:
            tau_rel = tau_trapz
    return tau_ps, acf, max(tau_rel, 0.0)


def _detect_frame_hbonds(
    donor_pos: np.ndarray,
    h_pos: np.ndarray,
    donor_residx: np.ndarray,
    donor_resnames: np.ndarray,
    acc_pos: np.ndarray,
    acc_residx: np.ndarray,
    acc_resnames: np.ndarray,
    box: np.ndarray,
    dist_cutoff_a: float,
    angle_cutoff_deg: float,
) -> list[tuple[int, int, str, str, float, float]]:
    if len(donor_pos) == 0 or len(acc_pos) == 0:
        return []
    pairs, dists = capped_distance(donor_pos, acc_pos, max_cutoff=dist_cutoff_a, box=box, return_distances=True)
    if len(pairs) == 0:
        return []
    d_idx = pairs[:, 0]
    a_idx = pairs[:, 1]
    not_self = donor_residx[d_idx] != acc_residx[a_idx]
    d_idx = d_idx[not_self]
    a_idx = a_idx[not_self]
    dists = dists[not_self]
    if len(d_idx) == 0:
        return []

    box_a = box[:3]
    v_hd = _min_image(donor_pos[d_idx] - h_pos[d_idx], box_a)
    v_ha = _min_image(acc_pos[a_idx] - h_pos[d_idx], box_a)
    norm_hd = np.linalg.norm(v_hd, axis=1)
    norm_ha = np.linalg.norm(v_ha, axis=1)
    cos_ang = np.clip(np.sum(v_hd * v_ha, axis=1) / np.maximum(norm_hd * norm_ha, 1e-9), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_ang))
    ok = angles >= angle_cutoff_deg

    out: list[tuple[int, int, str, str, float, float]] = []
    for di, ai, dist_val, ang_val in zip(d_idx[ok], a_idx[ok], dists[ok], angles[ok]):
        out.append(
            (
                int(donor_residx[di]),
                int(acc_residx[ai]),
                str(donor_resnames[di]),
                str(acc_resnames[ai]),
                float(dist_val),
                float(ang_val),
            )
        )
    return out


def _collect_donor_acceptor_indices(u: mda.Universe, topologies: dict[str, MoleculeTopology]):
    donor_idx, h_idx, d_residx, d_resnames = [], [], [], []
    acc_idx, a_residx, a_resnames = [], [], []
    for res in u.residues:
        topo = topologies.get(res.resname)
        if topo is None:
            continue
        name_to_idx = {a.name: a.index for a in res.atoms}
        for d_name, h_name in topo.donors:
            if d_name in name_to_idx and h_name in name_to_idx:
                donor_idx.append(name_to_idx[d_name])
                h_idx.append(name_to_idx[h_name])
                d_residx.append(res.resindex)
                d_resnames.append(res.resname)
        for a_name in topo.acceptors:
            if a_name in name_to_idx:
                acc_idx.append(name_to_idx[a_name])
                a_residx.append(res.resindex)
                a_resnames.append(res.resname)
    return (
        np.asarray(donor_idx, dtype=int),
        np.asarray(h_idx, dtype=int),
        np.asarray(d_residx, dtype=int),
        np.asarray(d_resnames, dtype=object),
        np.asarray(acc_idx, dtype=int),
        np.asarray(a_residx, dtype=int),
        np.asarray(a_resnames, dtype=object),
    )


def analyze_hbonds(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Analysis #2: H-bond population, occupancy, angle sensitivity, and ACF lifetime across any solvents."""
    pair_rows: list[dict[str, Any]] = []
    acf_rows: list[dict[str, Any]] = []
    sol_rows: list[dict[str, Any]] = []
    dist_a = cfg.hbond_dist_nm * 10.0

    for run in runs:
        u = load_unwrapped_universe(run)
        start, stop, stride = _equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, cfg.hbond_stride)
        d_idx, h_idx, d_res, d_names, a_idx, a_res, a_names = _collect_donor_acceptor_indices(u, run.topologies)

        dt_ps = float(u.trajectory.dt) * stride
        n_frames_used = 0
        pair_counts: dict[str, list[int]] = {}
        pair_counts_135: dict[str, int] = {}
        sol_any_hbond_frames = np.zeros(run.molecule_counts.get(run.solute_resname or "", 0), dtype=int)
        sol_res_indices = (
            [r.resindex for r in u.select_atoms(f"resname {run.solute_resname}").residues]
            if run.solute_resname
            else []
        )
        sol_idx_pos = {ri: i for i, ri in enumerate(sol_res_indices)}
        pair_history: dict[str, dict[tuple[int, int], list[int]]] = {}

        for frame_i, ts in enumerate(u.trajectory[start:stop:stride]):
            n_frames_used += 1
            pos = ts.positions
            box = ts.dimensions
            min_ang = min(cfg.hbond_angle_deg, cfg.hbond_angle_sensitivity_deg)
            hbs_all = _detect_frame_hbonds(
                pos[d_idx], pos[h_idx], d_res, d_names, pos[a_idx], a_res, a_names, box, dist_a, min_ang
            )

            frame_pair_cnt: dict[str, int] = {}
            seen_sol_in_frame: set[int] = set()
            for r1, r2, s1, s2, _, ang_val in hbs_all:
                pname = f"{s1}->{s2}"
                if ang_val >= cfg.hbond_angle_sensitivity_deg:
                    pair_counts_135[pname] = pair_counts_135.get(pname, 0) + 1
                if ang_val < cfg.hbond_angle_deg:
                    continue
                frame_pair_cnt[pname] = frame_pair_cnt.get(pname, 0) + 1
                if r1 in sol_idx_pos:
                    seen_sol_in_frame.add(sol_idx_pos[r1])
                if r2 in sol_idx_pos:
                    seen_sol_in_frame.add(sol_idx_pos[r2])
                if s1 == run.solute_resname or s2 == run.solute_resname:
                    pair_history.setdefault(pname, {}).setdefault((r1, r2), []).append(frame_i)
                elif frame_i < 2000:
                    ph = pair_history.setdefault(pname, {})
                    if (r1, r2) in ph or len(ph) < 2000:
                        ph.setdefault((r1, r2), []).append(frame_i)

            for s_pos in seen_sol_in_frame:
                sol_any_hbond_frames[s_pos] += 1

            for pname in set(pair_counts.keys()) | set(frame_pair_cnt.keys()):
                if pname not in pair_counts:
                    pair_counts[pname] = [0] * (n_frames_used - 1)
                pair_counts[pname].append(frame_pair_cnt.get(pname, 0))

        for pname, lst in pair_counts.items():
            if len(lst) < n_frames_used:
                lst.extend([0] * (n_frames_used - len(lst)))

        for pname, lst in sorted(pair_counts.items()):
            arr = np.asarray(lst, dtype=float)
            s1, s2 = pname.split("->")
            ph = pair_history.get(pname, {})
            if ph:
                mat = np.zeros((n_frames_used, len(ph)), dtype=bool)
                for j, f_list in enumerate(ph.values()):
                    mat[f_list, j] = True
                tau_ps, acf, tau_rel_ps = compute_hbond_autocorr(mat, dt_ps=dt_ps, max_lag_frames=100)
            else:
                tau_ps, acf, tau_rel_ps = np.array([0.0]), np.array([1.0]), 0.0

            n_src = run.molecule_counts.get(s1, 1)
            pair_rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "hb_pair": pname,
                    "donor_species": s1,
                    "acceptor_species": s2,
                    "mean_count_per_frame": float(np.mean(arr)),
                    "sd_count_per_frame": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "count_per_donor_mol": float(np.mean(arr) / n_src),
                    "mean_count_135deg": float(pair_counts_135.get(pname, 0) / max(n_frames_used, 1)),
                    "tau_hb_intermittent_ps": tau_rel_ps,
                    "dt_sampling_ps": dt_ps,
                }
            )
            for tt, cc in zip(tau_ps[:50], acf[:50]):
                acf_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "hb_pair": pname,
                        "tau_ps": float(tt),
                        "acf": float(cc),
                    }
                )

        if run.solute_resname is not None:
            sol = run.solute_resname
            n_sol = max(run.molecule_counts[sol], 1)
            sol_hb_tot = sum(
                np.mean(pair_counts[p])
                for p in pair_counts
                if p.startswith(f"{sol}->") or p.endswith(f"->{sol}")
            )
            sol_taus = [
                r["tau_hb_intermittent_ps"]
                for r in pair_rows
                if r["run_id"] == run.run_id and (r["donor_species"] == sol or r["acceptor_species"] == sol)
            ]
            s_entry: dict[str, Any] = {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "replica": run.replica,
                "sol_hbonds_per_mol": float(sol_hb_tot / n_sol),
                "sol_occupancy_any_pct": float(np.mean(sol_any_hbond_frames) / max(n_frames_used, 1) * 100.0),
                "tau_hb_ps": float(np.mean(sol_taus)) if sol_taus else 0.0,
            }
            for solv in run.solvent_resnames:
                cnt_solv = sum(
                    np.mean(pair_counts[p])
                    for p in pair_counts
                    if p in {f"{sol}->{solv}", f"{solv}->{sol}"}
                )
                s_entry[f"sol_{solv.lower()}_hbonds_per_mol"] = float(cnt_solv / n_sol)
            sol_rows.append(s_entry)

    pairs_df = pd.DataFrame(pair_rows)
    sol_df = pd.DataFrame(sol_rows)
    hbond_pairs_summary = (
        pairs_df.groupby(["system_code", "hb_pair", "donor_species", "acceptor_species"], as_index=False)
        .agg(
            n_rep=("replica", "count"),
            mean_count_per_frame=("mean_count_per_frame", "mean"),
            mean_count_per_frame_sd=("mean_count_per_frame", "std"),
            count_per_donor_mol=("count_per_donor_mol", "mean"),
            count_per_donor_mol_sd=("count_per_donor_mol", "std"),
            mean_count_135deg=("mean_count_135deg", "mean"),
            tau_hb_intermittent_ps=("tau_hb_intermittent_ps", "mean"),
            tau_hb_intermittent_ps_sd=("tau_hb_intermittent_ps", "std"),
        )
        if not pairs_df.empty
        else pd.DataFrame()
    )

    sol_summary_rows: list[dict[str, Any]] = []
    if not sol_df.empty:
        num_cols = [c for c in sol_df.columns if c not in {"run_id", "system_code", "replica"}]
        for sc, grp in sol_df.groupby("system_code"):
            row_s: dict[str, Any] = {"system_code": sc, "n_rep": len(grp)}
            for col in num_cols:
                vals = grp[col].dropna()
                m_v = float(vals.mean()) if not vals.empty else float("nan")
                sd_v = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
                if col == "sol_hbonds_per_mol":
                    row_s["sol_hbonds_per_mol_mean"] = m_v
                    row_s["sol_hbonds_per_mol_sd"] = sd_v
                else:
                    row_s[col] = m_v
                    row_s[f"{col}_sd"] = sd_v
            sol_summary_rows.append(row_s)

    return {
        "hbond_pairs": pairs_df,
        "hbond_pairs_summary": hbond_pairs_summary,
        "hbond_acf": pd.DataFrame(acf_rows),
        "hbond_solute_runs": sol_df,
        "hbond_summary": pd.DataFrame(sol_summary_rows),
    }


def build_local_frame_from_points(origin: np.ndarray, p_x: np.ndarray, p_plane: np.ndarray) -> np.ndarray:
    ex = p_x - origin
    ex = ex / max(float(np.linalg.norm(ex)), 1e-12)
    v2 = p_plane - origin
    ez = np.cross(ex, v2)
    if float(np.linalg.norm(ez)) < 1e-10:
        # Collinear fallback: pick orthogonal helper vector
        trial = np.array([0.0, 1.0, 0.0]) if abs(ex[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        ez = np.cross(ex, trial)
    ez = ez / max(float(np.linalg.norm(ez)), 1e-12)
    ey = np.cross(ez, ex)
    ey = ey / max(float(np.linalg.norm(ey)), 1e-12)
    return np.vstack([ex, ey, ez])


def write_opendx_grid(
    out_path: Path,
    grid: np.ndarray,
    origin_a: np.ndarray,
    spacing_a: float,
) -> None:
    nx_g, ny_g, nz_g = grid.shape
    lines = [
        f"object 1 class gridpositions counts {nx_g} {ny_g} {nz_g}",
        f"origin {origin_a[0]:.4f} {origin_a[1]:.4f} {origin_a[2]:.4f}",
        f"delta {spacing_a:.4f} 0.0000 0.0000",
        f"delta 0.0000 {spacing_a:.4f} 0.0000",
        f"delta 0.0000 0.0000 {spacing_a:.4f}",
        f"object 2 class gridconnections counts {nx_g} {ny_g} {nz_g}",
        f"object 3 class array type double rank 0 items {grid.size} data follows",
    ]
    flat = grid.ravel(order="C")
    for i in range(0, len(flat), 3):
        lines.append(" ".join(f"{val:.6e}" for val in flat[i : i + 3]))
    lines.append('attribute "dep" string "positions"')
    lines.append('object "regular positions regular connections" class field')
    lines.append('component "positions" value 1')
    lines.append('component "connections" value 2')
    lines.append('component "data" value 3')
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_vmd_sdf_script_multi(vmd_path: Path, ref_xyz: Path, dx_paths: list[Path]) -> None:
    lines = [
        "# VMD visualization script for 3D Spatial Distribution Function (SDF)",
        f"mol new {ref_xyz.name} type xyz waitfor all",
        "mol modstyle 0 0 Licorice 0.15 12.0 12.0",
    ]
    colors = [1, 0, 7, 3, 4, 10]
    for vol_i, dx_p in enumerate(dx_paths):
        lines.append(f"mol addfile {dx_p.name} type dx waitfor all")
        lines.append("mol addrep 0")
        lines.append(f"mol modstyle {vol_i + 1} 0 Isosurface 2.5 {vol_i} 0 0 1 1")
        lines.append(f"mol modcolor {vol_i + 1} 0 ColorID {colors[vol_i % len(colors)]}")
    lines.extend(["display projection Orthographic", "axes location Off"])
    vmd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _accumulate_3d_grid(grid: np.ndarray, pts: np.ndarray, hw: float, sp: float, n_bins: int) -> None:
    in_box = np.all(np.abs(pts) < hw, axis=1)
    if not np.any(in_box):
        return
    idx = np.clip(np.floor((pts[in_box] + hw) / sp).astype(np.int64), 0, n_bins - 1)
    flat = (idx[:, 0] * n_bins + idx[:, 1]) * n_bins + idx[:, 2]
    np.add.at(grid.ravel(), flat, 1.0)


def _select_solvent_orientation_atoms(u: mda.Universe, run: RunRecord, solv: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (polar_center_indices, vector_end_indices, tail_indices) per molecule for arbitrary solvents."""
    topo = run.topologies[solv]
    chem = run.chemistry.get(solv)
    polar_name = chem.polar_atoms[0] if chem and chem.polar_atoms else topo.atom_names[0]
    polar_idx = u.select_atoms(f"resname {solv} and name {polar_name}").indices
    if len(polar_idx) == 0:
        polar_idx = u.select_atoms(f"resname {solv} and name {topo.atom_names[0]}").indices

    # Vector partner: donor H paired with polar_name if present, else bonded atom or different atom
    vec_end_name = None
    for d_n, h_n in topo.donors:
        if d_n == polar_name:
            vec_end_name = h_n
            break
    if vec_end_name is None and chem and chem.polar_h_atoms:
        vec_end_name = chem.polar_h_atoms[0]
    if vec_end_name is None:
        p_1b = topo.atom_names.index(polar_name) + 1 if polar_name in topo.atom_names else 1
        nbr = next((b if a == p_1b else a for a, b in topo.bonds_1based if a == p_1b or b == p_1b), None)
        if nbr is not None:
            vec_end_name = topo.atom_names[nbr - 1]
        else:
            vec_end_name = next((a for a in topo.atom_names if a != polar_name), polar_name)

    vec_end_idx = u.select_atoms(f"resname {solv} and name {vec_end_name}").indices
    if len(vec_end_idx) != len(polar_idx):
        vec_end_idx = polar_idx

    tail_idx = np.array([], dtype=int)
    if chem and chem.tail_atoms:
        tail_idx = u.select_atoms(f"resname {solv} and name {chem.tail_atoms[0]}").indices
    return polar_idx, vec_end_idx, tail_idx


def analyze_orientation_and_sdf(runs: list[RunRecord], cfg: Config) -> dict[str, Any]:
    """Analysis #3: Local-frame 3D SDF (.dx + VMD) and radial-angular distributions for arbitrary solvents."""
    sdf_dir = cfg.results_dir / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    orient_rows: list[dict[str, Any]] = []
    solvent_stat_rows: list[dict[str, Any]] = []
    hist2d_rows: list[dict[str, Any]] = []
    hist1d_rows: list[dict[str, Any]] = []

    hw = cfg.sdf_half_width_a
    sp = cfg.sdf_grid_spacing_a
    n_bins = int(round(2.0 * hw / sp))
    origin_a = np.array([-hw + 0.5 * sp] * 3)
    voxel_vol_a3 = sp**3

    for run in runs:
        if run.solute_resname is None:
            continue
        sol = run.solute_resname
        topo = run.topologies[sol]
        if topo.align_atoms is None:
            continue
        solvents = [s for s in run.solvent_resnames if s in run.topologies and run.molecule_counts.get(s, 0) > 0]
        if not solvents:
            continue

        a0_name, a1_name, a2_name = topo.align_atoms
        u = load_unwrapped_universe(run)
        start, stop, stride = _equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, cfg.sdf_stride)

        sol_res = u.select_atoms(f"resname {sol}").residues
        sol_triplets = [
            (
                r.atoms.select_atoms(f"name {a0_name}")[0].index,
                r.atoms.select_atoms(f"name {a1_name}")[0].index,
                r.atoms.select_atoms(f"name {a2_name}")[0].index,
            )
            for r in sol_res
        ]
        sol_chem = run.chemistry.get(sol)
        sol_axis_idx = sol_chem.sol_axis_index if sol_chem else (2 if sol == "LIM" else 0)

        solv_indices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        grids_polar: dict[str, np.ndarray] = {}
        grids_tail: dict[str, np.ndarray] = {}
        cos_shell: dict[str, list[float]] = {s: [] for s in solvents}
        r_shell: dict[str, list[float]] = {s: [] for s in solvents}

        for s in solvents:
            p_i, v_i, t_i = _select_solvent_orientation_atoms(u, run, s)
            solv_indices[s] = (p_i, v_i, t_i)
            grids_polar[s] = np.zeros((n_bins, n_bins, n_bins), dtype=float)
            if len(t_i) > 0:
                grids_tail[s] = np.zeros((n_bins, n_bins, n_bins), dtype=float)

        n_obs = 0
        vol_sum = 0.0
        ref_sol_local = None

        for ts in u.trajectory[start:stop:stride]:
            pos = ts.positions
            box_a = ts.dimensions[:3]
            vol_sum += float(np.prod(box_a))
            n_obs += 1

            solv_vecs: dict[str, np.ndarray] = {}
            for s in solvents:
                p_i, v_i, _ = solv_indices[s]
                vec = _min_image(pos[v_i] - pos[p_i], box_a)
                solv_vecs[s] = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)

            for r_i, (i0, i1, i2) in enumerate(sol_triplets):
                p0 = pos[i0]
                p1 = p0 + _min_image((pos[i1] - p0)[None, :], box_a)[0]
                p2 = p0 + _min_image((pos[i2] - p0)[None, :], box_a)[0]
                rot = build_local_frame_from_points(p0, p1, p2)
                sol_axis = rot[min(max(sol_axis_idx, 0), 2)]

                if ref_sol_local is None and r_i == 0:
                    ag_sol = sol_res[0].atoms
                    d_local = _min_image(ag_sol.positions - p0[None, :], box_a) @ rot.T
                    ref_sol_local = (list(topo.elements), d_local)

                for s in solvents:
                    p_i, _, t_i = solv_indices[s]
                    if len(p_i) > 0:
                        d_s = _min_image(pos[p_i] - p0[None, :], box_a)
                        dist_s = np.linalg.norm(d_s, axis=1)
                        mask_s = dist_s <= hw * 1.5
                        if np.any(mask_s):
                            loc_s = d_s[mask_s] @ rot.T
                            _accumulate_3d_grid(grids_polar[s], loc_s, hw, sp, n_bins)
                            near_s = dist_s <= 7.5
                            if np.any(near_s):
                                c_vals = solv_vecs[s][near_s] @ sol_axis
                                cos_shell[s].extend(c_vals.tolist())
                                r_shell[s].extend((dist_s[near_s] / 10.0).tolist())
                    if s in grids_tail and len(t_i) > 0:
                        d_t = _min_image(pos[t_i] - p0[None, :], box_a)
                        loc_t = d_t @ rot.T
                        _accumulate_3d_grid(grids_tail[s], loc_t, hw, sp, n_bins)

        n_sol = max(len(sol_triplets), 1)
        v_mean_a3 = vol_sum / max(n_obs, 1)
        run_sdf_dir = sdf_dir / f"{run.system_code}_{run.replica}"
        run_sdf_dir.mkdir(parents=True, exist_ok=True)
        xyz_p = run_sdf_dir / f"{run.system_code}_{run.replica}_solute_frame.xyz"
        vmd_p = run_sdf_dir / f"view_sdf_{run.system_code}_{run.replica}.vmd"

        row_entry: dict[str, Any] = {
            "run_id": run.run_id,
            "system_code": run.system_code,
            "replica": run.replica,
            "vmd_script": str(vmd_p),
        }
        dx_written: list[Path] = []
        c_edges_1d = np.linspace(-1.0, 1.0, 21)

        for s_idx, s in enumerate(solvents):
            sl = s.lower()
            p_i = solv_indices[s][0]
            rho_bulk = max(len(p_i), 1) / max(v_mean_a3, 1e-9)
            norm = max(n_obs * n_sol * voxel_vol_a3 * rho_bulk, 1e-12)
            g3d_p = grids_polar[s] / norm
            tag = f"{s}_OH" if (run.topologies[s].donors) else f"{s}_polar"
            dx_p = run_sdf_dir / f"{run.system_code}_{run.replica}_{tag}.dx"
            write_opendx_grid(dx_p, g3d_p, origin_a, sp)
            dx_written.append(dx_p)

            sdf_max = float(np.max(g3d_p))
            c_mean = float(np.mean(cos_shell[s])) if cos_shell[s] else float("nan")
            c_abs = float(np.mean(np.abs(cos_shell[s]))) if cos_shell[s] else float("nan")

            row_entry[f"sdf_{sl}_oh_max"] = sdf_max
            row_entry[f"cos_theta_{sl}_oh_mean"] = c_mean
            row_entry[f"cos_theta_{sl}_abs_mean"] = c_abs
            row_entry[f"dx_{sl}_path"] = str(dx_p)
            if s_idx == 0:
                row_entry["sdf_primary_max"] = sdf_max
                row_entry["cos_theta_primary_mean"] = c_mean

            t_max = float("nan")
            if s in grids_tail:
                g3d_t = grids_tail[s] / norm
                t_max = float(np.max(g3d_t))
                dx_tail_p = run_sdf_dir / f"{run.system_code}_{run.replica}_{s}_tail.dx"
                write_opendx_grid(dx_tail_p, g3d_t, origin_a, sp)
                row_entry[f"sdf_{sl}_tail_max"] = t_max

            solvent_stat_rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "solvent": s,
                    "solvent_group": tag,
                    "sdf_polar_max": sdf_max,
                    "sdf_tail_max": t_max,
                    "cos_theta_mean": c_mean,
                    "cos_theta_abs_mean": c_abs,
                    "dx_path": str(dx_p),
                }
            )

            if r_shell[s]:
                r_edges = np.linspace(0.20, 0.75, 23)
                c_edges = np.linspace(-1.0, 1.0, 21)
                h2d, _, _ = np.histogram2d(r_shell[s], cos_shell[s], bins=[r_edges, c_edges], density=True)
                for ri in range(len(r_edges) - 1):
                    for ci in range(len(c_edges) - 1):
                        hist2d_rows.append(
                            {
                                "system_code": run.system_code,
                                "replica": run.replica,
                                "solvent": s,
                                "solvent_group": tag,
                                "r_nm": 0.5 * (r_edges[ri] + r_edges[ri + 1]),
                                "cos_theta": 0.5 * (c_edges[ci] + c_edges[ci + 1]),
                                "density": float(h2d[ri, ci]),
                            }
                        )
                h1d, _ = np.histogram(cos_shell[s], bins=c_edges_1d, density=True)
                for ci in range(len(c_edges_1d) - 1):
                    hist1d_rows.append(
                        {
                            "run_id": run.run_id,
                            "system_code": run.system_code,
                            "replica": run.replica,
                            "solvent": s,
                            "solvent_group": tag,
                            "cos_theta": float(0.5 * (c_edges_1d[ci] + c_edges_1d[ci + 1])),
                            "density": float(h1d[ci]),
                        }
                    )

        if ref_sol_local is not None:
            elems, coords = ref_sol_local
            xyz_lines = [str(len(elems)), f"{sol} local frame reference"]
            for el, (x, y, z) in zip(elems, coords):
                xyz_lines.append(f"{el:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
            xyz_p.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
            _write_vmd_sdf_script_multi(vmd_p, xyz_p, dx_written)

        orient_rows.append(row_entry)

    orient_df = pd.DataFrame(orient_rows)
    orient_1d_df = pd.DataFrame(hist1d_rows)
    orient_solvent_df = pd.DataFrame(solvent_stat_rows)

    summary_rows: list[dict[str, Any]] = []
    if not orient_df.empty:
        num_cols = [c for c in orient_df.columns if c.startswith("sdf_") or c.startswith("cos_theta_")]
        for sc, grp in orient_df.groupby("system_code"):
            srow: dict[str, Any] = {"system_code": sc, "n_rep": len(grp)}
            for col in num_cols:
                vals = grp[col].dropna()
                srow[col] = float(vals.mean()) if not vals.empty else float("nan")
                srow[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            summary_rows.append(srow)

    return {
        "orientation_runs": orient_df,
        "orientation_solvent_stats": orient_solvent_df,
        "orientation_summary": pd.DataFrame(summary_rows),
        "orientation_hist2d": pd.DataFrame(hist2d_rows),
        "orientation_1d": orient_1d_df,
    }


def compute_hbond_network_metrics(edges: list[tuple[int, int]], node_types: dict[int, str]) -> dict[str, Any]:
    g = nx.Graph()
    for nid, ntype in node_types.items():
        g.add_node(nid, species=ntype)
    for u, v in edges:
        if u != v:
            g.add_edge(u, v)

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    degrees = [d for _, d in g.degree()]
    comps = [len(c) for c in nx.connected_components(g)]
    n_components = len(comps)
    largest = max(comps) if comps else 0
    mean_comp_size = float(np.mean(comps)) if comps else 0.0
    triangles = sum(nx.triangles(g).values()) // 3
    n_cycles = max(int(n_edges - n_nodes + n_components), 0)
    deg_hist = {k: int(sum(1 for d in degrees if d == k)) for k in range(5)}
    deg_hist["5+"] = int(sum(1 for d in degrees if d >= 5))

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "n_components": n_components,
        "mean_component_size": mean_comp_size,
        "largest_component_size": largest,
        "giant_component_frac": float(largest / max(n_nodes, 1)),
        "mean_clustering": float(nx.average_clustering(g)) if n_nodes > 0 else 0.0,
        "n_triangles": int(triangles),
        "n_cycles": n_cycles,
        "deg_0_frac": float(deg_hist[0] / max(n_nodes, 1)),
        "deg_1_frac": float(deg_hist[1] / max(n_nodes, 1)),
        "deg_2_frac": float(deg_hist[2] / max(n_nodes, 1)),
        "deg_3plus_frac": float((n_nodes - deg_hist[0] - deg_hist[1] - deg_hist[2]) / max(n_nodes, 1)),
    }


def analyze_solvent_network(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Analysis #4: NetworkX H-bond topology across all molecules and first-shell vs bulk."""
    run_rows: list[dict[str, Any]] = []
    dist_a = cfg.hbond_dist_nm * 10.0

    for run in runs:
        u = load_unwrapped_universe(run)
        start, stop, stride = _equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, cfg.network_stride)
        d_idx, h_idx, d_res, d_names, a_idx, a_res, a_names = _collect_donor_acceptor_indices(u, run.topologies)
        node_types = {r.resindex: r.resname for r in u.residues}

        solvents = [s for s in run.solvent_resnames if s in run.topologies and run.molecule_counts.get(s, 0) > 0]
        sol_ag = u.select_atoms(f"resname {run.solute_resname}") if run.solute_resname else None
        solv_ags = {s: u.select_atoms(f"resname {s}") for s in solvents}
        solv_topos = {s: run.topologies[s] for s in solvents}
        solv_hmasks = {s: (np.asarray(solv_topos[s].elements) != "H") for s in solvents}
        solv_res_ids = np.asarray(
            [r.resindex for s in solvents for r in solv_ags[s].residues],
            dtype=int,
        )
        sol_topo = run.topologies[run.solute_resname] if run.solute_resname else None
        sol_hmask = np.asarray(sol_topo.elements) != "H" if sol_topo else None

        frame_metrics: list[dict[str, Any]] = []
        shell_deg_list, bulk_deg_list = [], []

        for ts in u.trajectory[start:stop:stride]:
            pos = ts.positions
            box = ts.dimensions
            box_a = box[:3]
            hbs = _detect_frame_hbonds(
                pos[d_idx], pos[h_idx], d_res, d_names, pos[a_idx], a_res, a_names, box, dist_a, cfg.hbond_angle_deg
            )
            edges = [(r1, r2) for r1, r2, *_ in hbs]
            m = compute_hbond_network_metrics(edges, node_types)
            frame_metrics.append(m)

            if run.solute_resname is not None and sol_ag is not None and sol_topo is not None and solvents:
                c_sol = compute_whole_coms_vectorized(
                    sol_ag.positions, run.molecule_counts[run.solute_resname], sol_topo.n_atoms, sol_topo.masses, box_a, sol_topo.center_atom_0based, sol_hmask
                ).astype(np.float32)
                c_solv_parts = [
                    compute_whole_coms_vectorized(
                        solv_ags[s].positions,
                        run.molecule_counts[s],
                        solv_topos[s].n_atoms,
                        solv_topos[s].masses,
                        box_a,
                        solv_topos[s].center_atom_0based,
                        solv_hmasks[s],
                    ).astype(np.float32)
                    for s in solvents
                ]
                c_solv = np.vstack(c_solv_parts)
                pairs, _ = capped_distance(c_sol, c_solv, max_cutoff=7.5, box=box, return_distances=True)
                shell_set = set(solv_res_ids[pairs[:, 1]].tolist()) if len(pairs) else set()
                deg_map: dict[int, int] = {}
                for u_n, v_n in set((min(a, b), max(a, b)) for a, b in edges if a != b):
                    deg_map[u_n] = deg_map.get(u_n, 0) + 1
                    deg_map[v_n] = deg_map.get(v_n, 0) + 1
                if shell_set:
                    shell_deg_list.append(float(np.mean([deg_map.get(nid, 0) for nid in shell_set])))
                bulk_set = set(solv_res_ids.tolist()) - shell_set
                if bulk_set:
                    bulk_deg_list.append(float(np.mean([deg_map.get(nid, 0) for nid in bulk_set])))

        df_m = pd.DataFrame(frame_metrics)
        row = {
            "run_id": run.run_id,
            "system_code": run.system_code,
            "replica": run.replica,
            "mean_degree": float(df_m["mean_degree"].mean()),
            "mean_degree_sd": float(df_m["mean_degree"].std(ddof=1)) if len(df_m) > 1 else 0.0,
            "n_components": float(df_m["n_components"].mean()),
            "mean_component_size": float(df_m["mean_component_size"].mean()),
            "largest_component_size": float(df_m["largest_component_size"].mean()),
            "giant_frac": float(df_m["giant_component_frac"].mean()),
            "mean_clustering": float(df_m["mean_clustering"].mean()),
            "n_triangles": float(df_m["n_triangles"].mean()),
            "n_cycles": float(df_m["n_cycles"].mean()),
            "deg_0_frac": float(df_m["deg_0_frac"].mean()),
            "deg_1_frac": float(df_m["deg_1_frac"].mean()),
            "deg_2_frac": float(df_m["deg_2_frac"].mean()),
            "deg_3plus_frac": float(df_m["deg_3plus_frac"].mean()),
            "shell_solvent_degree": float(np.mean(shell_deg_list)) if shell_deg_list else float("nan"),
            "bulk_solvent_degree": float(np.mean(bulk_deg_list)) if bulk_deg_list else float("nan"),
        }
        run_rows.append(row)

    runs_df = pd.DataFrame(run_rows)
    summary_df = (
        runs_df.groupby("system_code", as_index=False)
        .agg(
            n_rep=("replica", "count"),
            mean_degree=("mean_degree", "mean"),
            mean_degree_sd=("mean_degree", "std"),
            giant_frac=("giant_frac", "mean"),
            giant_frac_sd=("giant_frac", "std"),
            mean_component_size=("mean_component_size", "mean"),
            largest_component_size=("largest_component_size", "mean"),
            mean_clustering=("mean_clustering", "mean"),
            mean_clustering_sd=("mean_clustering", "std"),
            n_triangles=("n_triangles", "mean"),
            n_cycles=("n_cycles", "mean"),
            n_cycles_sd=("n_cycles", "std"),
            deg_0_frac=("deg_0_frac", "mean"),
            deg_1_frac=("deg_1_frac", "mean"),
            deg_2_frac=("deg_2_frac", "mean"),
            deg_3plus_frac=("deg_3plus_frac", "mean"),
            shell_solvent_degree=("shell_solvent_degree", "mean"),
            bulk_solvent_degree=("bulk_solvent_degree", "mean"),
        )
        if not runs_df.empty
        else pd.DataFrame()
    )
    return {"network_runs": runs_df, "network_summary": summary_df}
