from __future__ import annotations

from collections import deque
import itertools
import json
from pathlib import Path
from typing import Any

from MDAnalysis.lib.distances import capped_distance
import numpy as np
import pandas as pd

from mody_analyse.analyzers.md_structural import _equilibrated_frame_slice, _min_image, load_unwrapped_universe
from mody_analyse.core.config import Config
from mody_analyse.core.discovery import MoleculeTopology, RunRecord, compute_whole_coms_vectorized


def make_molecule_whole(coords: np.ndarray, topo: MoleculeTopology, box_a: np.ndarray) -> np.ndarray:
    """Reconstruct bonded molecule across PBC using BFS over topology bonds."""
    n = topo.n_atoms
    out = coords.copy()
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in topo.bonds_1based:
        adj[a - 1].append(b - 1)
        adj[b - 1].append(a - 1)
    visited = [False] * n
    visited[0] = True
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                diff = out[v] - out[u]
                diff -= box_a * np.round(diff / box_a)
                out[v] = out[u] + diff
                visited[v] = True
                q.append(v)
    return out


def unwrap_pair_minimum_image(
    ref_coords: np.ndarray,
    other_coords_list: list[np.ndarray],
    box_a: np.ndarray,
) -> list[np.ndarray]:
    """Shift each whole partner molecule so its centroid lies at minimum image to reference."""
    ref_center = np.mean(ref_coords, axis=0)
    out = [ref_coords.copy()]
    for oc in other_coords_list:
        oc_center = np.mean(oc, axis=0)
        shift = -box_a * np.round((oc_center - ref_center) / box_a)
        out.append(oc + shift[None, :])
    return out


def validate_cluster_bonds(coords: np.ndarray, topo_list: list[MoleculeTopology]) -> None:
    offset = 0
    for topo in topo_list:
        sub = coords[offset : offset + topo.n_atoms]
        for a, b in topo.bonds_1based:
            d = float(np.linalg.norm(sub[a - 1] - sub[b - 1]))
            if d > 2.0:
                raise ValueError(f"Broken covalent bond {a}-{b} in {topo.resname}: {d:.3f} A")
        offset += topo.n_atoms


def write_xyz(path: Path, elements: list[str], coords: np.ndarray, comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(elements)), comment]
    for el, (x, y, z) in zip(elements, coords):
        lines.append(f"{el:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dihedral_deg(coords: np.ndarray, idx_1based: tuple[int, int, int, int]) -> float:
    p0, p1, p2, p3 = [coords[i - 1] for i in idx_1based]
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= max(float(np.linalg.norm(b1)), 1e-12)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def extract_solute_snapshots(runs: list[RunRecord], cfg: Config) -> pd.DataFrame:
    out_dir = cfg.results_dir / "clusters/strain_xyz"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for run in runs:
        if run.solute_resname is None:
            continue
        sol = run.solute_resname
        topo = run.topologies[sol]
        u = load_unwrapped_universe(run)
        start, stop, _ = _equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, 1)
        sol_res = u.select_atoms(f"resname {sol}").residues
        n_target = cfg.strain_snapshots_per_rep
        frame_indices = np.linspace(start, stop - 1, n_target, dtype=int)

        for snap_i, fidx in enumerate(frame_indices):
            ts = u.trajectory[int(fidx)]
            box_a = ts.dimensions[:3]
            mol_idx = snap_i % len(sol_res)
            coords = make_molecule_whole(sol_res[mol_idx].atoms.positions, topo, box_a)
            validate_cluster_bonds(coords, [topo])
            cid = f"{run.system_code}_{run.replica}_strain_{snap_i:03d}"
            xyz_path = out_dir / f"{cid}.xyz"
            dih_vals = [_dihedral_deg(coords, d) for d in topo.rotatable_dihedrals_1based]
            write_xyz(xyz_path, topo.elements, coords, comment=f"{cid} t={ts.time/1000.0:.2f}ns")
            rows.append(
                {
                    "cluster_id": cid,
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "solute_resname": sol,
                    "frame_idx": int(fidx),
                    "time_ns": float(ts.time / 1000.0),
                    "mol_idx": int(mol_idx),
                    "xyz_path": str(xyz_path),
                    "n_atoms": topo.n_atoms,
                    "dihedrals_deg_json": json.dumps([round(x, 2) for x in dih_vals]),
                }
            )
    return pd.DataFrame(rows)


def _pair_is_hbonded(
    sol_coords: np.ndarray,
    sol_topo: MoleculeTopology,
    solv_coords: np.ndarray,
    solv_topo: MoleculeTopology,
) -> tuple[bool, float]:
    s_map = {n: i for i, n in enumerate(sol_topo.atom_names)}
    v_map = {n: i for i, n in enumerate(solv_topo.atom_names)}
    best_dist = 99.0
    hb = False
    for d_coords, d_topo, d_map, a_coords, a_topo, a_map in [
        (sol_coords, sol_topo, s_map, solv_coords, solv_topo, v_map),
        (solv_coords, solv_topo, v_map, sol_coords, sol_topo, s_map),
    ]:
        for d_name, h_name in d_topo.donors:
            dp = d_coords[d_map[d_name]]
            hp = d_coords[d_map[h_name]]
            for a_name in a_topo.acceptors:
                ap = a_coords[a_map[a_name]]
                dist = float(np.linalg.norm(ap - dp))
                best_dist = min(best_dist, dist)
                if dist <= 3.5:
                    v_hd = dp - hp
                    v_ha = ap - hp
                    cos_a = np.dot(v_hd, v_ha) / max(np.linalg.norm(v_hd) * np.linalg.norm(v_ha), 1e-9)
                    ang = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                    if ang >= 140.0:
                        hb = True
    return hb, best_dist


def extract_stratified_dimers(runs: list[RunRecord], cfg: Config) -> pd.DataFrame:
    out_dir = cfg.results_dir / "clusters/dimer_xyz"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for run in runs:
        if run.solute_resname is None:
            continue
        sol = run.solute_resname
        sol_topo = run.topologies[sol]
        solvents = [s for s in run.solvent_resnames if s in run.topologies and run.molecule_counts.get(s, 0) > 0]
        if not solvents:
            continue

        u = load_unwrapped_universe(run)
        start, stop, _ = _equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, 1)
        sol_res = u.select_atoms(f"resname {sol}").residues
        solv_groups = {s: u.select_atoms(f"resname {s}").residues for s in solvents}

        sample_frames = np.linspace(start, stop - 1, 25, dtype=int)
        candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
            s: {"hbond": [], "vdw": []} for s in solvents
        }

        sol_ag = u.select_atoms(f"resname {sol}")
        solv_ags = {s: u.select_atoms(f"resname {s}") for s in solvents}
        sol_hmask = np.asarray(sol_topo.elements) != "H"
        for fidx in sample_frames:
            ts = u.trajectory[int(fidx)]
            box = ts.dimensions
            box_a = box[:3]
            sol_heavy = compute_whole_coms_vectorized(
                sol_ag.positions, run.molecule_counts[sol], sol_topo.n_atoms, sol_topo.masses, box_a, sol_topo.center_atom_0based, sol_hmask
            ).astype(np.float32)

            for solv_sp in solvents:
                solv_topo = run.topologies[solv_sp]
                s_res = solv_groups[solv_sp]
                s_hmask = np.asarray(solv_topo.elements) != "H"
                solv_heavy = compute_whole_coms_vectorized(
                    solv_ags[solv_sp].positions, run.molecule_counts[solv_sp], solv_topo.n_atoms, solv_topo.masses, box_a, solv_topo.center_atom_0based, s_hmask
                ).astype(np.float32)
                pairs, dists = capped_distance(sol_heavy, solv_heavy, max_cutoff=7.5, box=box, return_distances=True)
                if len(pairs) == 0:
                    continue
                order = np.argsort(dists)
                for p_idx in order[:20]:
                    si, vi = int(pairs[p_idx, 0]), int(pairs[p_idx, 1])
                    sc = make_molecule_whole(sol_res[si].atoms.positions, sol_topo, box_a)
                    vc = make_molecule_whole(s_res[vi].atoms.positions, solv_topo, box_a)
                    merged = unwrap_pair_minimum_image(sc, [vc], box_a)
                    s_h = merged[0][np.array(sol_topo.elements) != "H"]
                    v_h = merged[1][np.array(solv_topo.elements) != "H"]
                    if len(s_h) == 0 or len(v_h) == 0:
                        s_h, v_h = merged[0], merged[1]
                    min_h = float(np.min(np.linalg.norm(s_h[:, None, :] - v_h[None, :, :], axis=-1)))
                    if min_h > 4.5:
                        continue
                    is_hb, _ = _pair_is_hbonded(merged[0], sol_topo, merged[1], solv_topo)
                    motif = "hbond" if is_hb else "vdw"
                    candidates[solv_sp][motif].append(
                        {
                            "fidx": int(fidx),
                            "time_ns": float(ts.time / 1000.0),
                            "sol_idx": si,
                            "solv_idx": vi,
                            "min_heavy_a": min_h,
                            "com_dist_a": float(dists[p_idx]),
                            "coords": np.vstack(merged),
                            "elements": sol_topo.elements + solv_topo.elements,
                            "topos": [sol_topo, solv_topo],
                        }
                    )

        target = cfg.dimers_per_pair_per_rep
        for solv_sp in solvents:
            hb_list = candidates[solv_sp]["hbond"]
            vdw_list = candidates[solv_sp]["vdw"]
            selected: list[tuple[str, dict[str, Any]]] = []
            if hb_list and vdw_list:
                n_hb = min(len(hb_list), target // 2)
                n_vdw = min(len(vdw_list), target - n_hb)
                if n_hb + n_vdw < target and len(hb_list) > n_hb:
                    n_hb = min(len(hb_list), target - n_vdw)
                for idx in np.linspace(0, len(hb_list) - 1, n_hb, dtype=int):
                    selected.append(("hbond", hb_list[int(idx)]))
                for idx in np.linspace(0, len(vdw_list) - 1, n_vdw, dtype=int):
                    selected.append(("vdw", vdw_list[int(idx)]))
            else:
                pool = hb_list if hb_list else vdw_list
                m_lbl = "hbond" if hb_list else "vdw"
                for idx in np.linspace(0, len(pool) - 1, min(len(pool), target), dtype=int):
                    selected.append((m_lbl, pool[int(idx)]))

            for k_i, (motif, item) in enumerate(selected):
                validate_cluster_bonds(item["coords"], item["topos"])
                cid = f"{run.system_code}_{solv_sp}_{run.replica}_d{k_i:03d}"
                xyz_path = out_dir / f"{cid}.xyz"
                write_xyz(xyz_path, item["elements"], item["coords"], comment=f"{cid} motif={motif} dmin={item['min_heavy_a']:.2f}A")
                rows.append(
                    {
                        "cluster_id": cid,
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "pair": f"{run.system_code}-{solv_sp}",
                        "solute_resname": sol,
                        "solvent_resname": solv_sp,
                        "motif": motif,
                        "frame_idx": item["fidx"],
                        "time_ns": item["time_ns"],
                        "min_heavy_dist_a": item["min_heavy_a"],
                        "com_dist_a": item["com_dist_a"],
                        "n_atoms_a": item["topos"][0].n_atoms,
                        "n_atoms_b": item["topos"][1].n_atoms,
                        "xyz_path": str(xyz_path),
                    }
                )
    return pd.DataFrame(rows)


def _trimer_motif_name(sp_b: str, sp_c: str) -> str:
    return f"S+2{sp_b}" if sp_b == sp_c else f"S+{sp_b}+{sp_c}"


def extract_trimers(runs: list[RunRecord], cfg: Config) -> pd.DataFrame:
    out_dir = cfg.results_dir / "clusters/trimer_xyz"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for run in runs:
        if run.solute_resname is None:
            continue
        sol = run.solute_resname
        sol_topo = run.topologies[sol]
        solvents = [s for s in run.solvent_resnames if s in run.topologies and run.molecule_counts.get(s, 0) > 0]
        if not solvents:
            continue

        u = load_unwrapped_universe(run)
        start, stop, _ = _equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, 1)
        sol_res = u.select_atoms(f"resname {sol}").residues
        solv_res_map = {s: u.select_atoms(f"resname {s}").residues for s in solvents}
        solv_ag_map = {s: u.select_atoms(f"resname {s}") for s in solvents}
        solv_topos = {s: run.topologies[s] for s in solvents}

        motif_pairs = list(itertools.combinations_with_replacement(solvents, 2))
        pools: dict[str, list[dict[str, Any]]] = {_trimer_motif_name(a, b): [] for a, b in motif_pairs}
        sample_frames = np.linspace(start, stop - 1, 20, dtype=int)

        for fidx in sample_frames:
            ts = u.trajectory[int(fidx)]
            box_a = ts.dimensions[:3]
            coms_cache = {
                s: compute_whole_coms_vectorized(
                    solv_ag_map[s].positions,
                    run.molecule_counts[s],
                    solv_topos[s].n_atoms,
                    solv_topos[s].masses,
                    box_a,
                    solv_topos[s].center_atom_0based,
                )
                for s in solvents
            }
            for si in range(len(sol_res)):
                sc = make_molecule_whole(sol_res[si].atoms.positions, sol_topo, box_a)
                s_h = sc[np.array(sol_topo.elements) != "H"]
                if len(s_h) == 0:
                    s_h = sc
                s_com = np.mean(s_h, axis=0)

                by_sp: dict[str, list[tuple[float, int, np.ndarray]]] = {s: [] for s in solvents}
                for sp in solvents:
                    coms = coms_cache[sp]
                    d_com = np.linalg.norm(_min_image(coms - s_com[None, :], box_a), axis=1)
                    s_grp, s_tp = solv_res_map[sp], solv_topos[sp]
                    for idx_cand in np.argsort(d_com)[:8]:
                        if d_com[idx_cand] > 9.5:
                            break
                        vc = make_molecule_whole(s_grp[int(idx_cand)].atoms.positions, s_tp, box_a)
                        vc_img = unwrap_pair_minimum_image(sc, [vc], box_a)[1]
                        v_h = vc_img[np.array(s_tp.elements) != "H"]
                        if len(v_h) == 0:
                            v_h = vc_img
                        d_min = float(np.min(np.linalg.norm(s_h[:, None, :] - v_h[None, :, :], axis=-1)))
                        if d_min <= 4.3:
                            by_sp[sp].append((d_min, int(idx_cand), vc_img))
                    by_sp[sp].sort(key=lambda x: x[0])

                for sp_b, sp_c in motif_pairs:
                    motif = _trimer_motif_name(sp_b, sp_c)
                    tp_b, tp_c = solv_topos[sp_b], solv_topos[sp_c]
                    if sp_b == sp_c and len(by_sp[sp_b]) >= 2:
                        c_b, c_c = by_sp[sp_b][0][2], by_sp[sp_b][1][2]
                    elif sp_b != sp_c and by_sp[sp_b] and by_sp[sp_c]:
                        c_b, c_c = by_sp[sp_b][0][2], by_sp[sp_c][0][2]
                    else:
                        continue
                    pools[motif].append(
                        {
                            "fidx": int(fidx),
                            "time_ns": float(ts.time / 1000.0),
                            "coords": np.vstack([sc, c_b, c_c]),
                            "elements": sol_topo.elements + tp_b.elements + tp_c.elements,
                            "topos": [sol_topo, tp_b, tp_c],
                            "sp_b": sp_b,
                            "sp_c": sp_c,
                        }
                    )

        target_per_motif = max(2, cfg.trimers_per_system_per_rep // max(len(pools), 1))
        t_counter = 0
        for motif, plist in pools.items():
            if not plist:
                continue
            n_pick = min(len(plist), target_per_motif)
            for idx in np.linspace(0, len(plist) - 1, n_pick, dtype=int):
                item = plist[int(idx)]
                validate_cluster_bonds(item["coords"], item["topos"])
                cid = f"{run.system_code}_{run.replica}_t{t_counter:03d}_{motif.replace('+', '_')}"
                t_counter += 1
                xyz_path = out_dir / f"{cid}.xyz"
                write_xyz(xyz_path, item["elements"], item["coords"], comment=f"{cid} motif={motif}")
                rows.append(
                    {
                        "cluster_id": cid,
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "replica": run.replica,
                        "motif": motif,
                        "frame_idx": item["fidx"],
                        "time_ns": item["time_ns"],
                        "n_atoms_a": item["topos"][0].n_atoms,
                        "n_atoms_b": item["topos"][1].n_atoms,
                        "n_atoms_c": item["topos"][2].n_atoms,
                        "sp_b": item["sp_b"],
                        "sp_c": item["sp_c"],
                        "xyz_path": str(xyz_path),
                    }
                )
    return pd.DataFrame(rows)
