from __future__ import annotations

import math
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Voronoi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from mody_analyse.core.config import Config
from mody_analyse.core.discovery import MoleculeTopology, RunRecord, compute_whole_coms_vectorized, equilibrated_frame_slice
from mody_analyse.core.pbc import (
    box_matrix_from_dimensions,
    box_volume_nm3,
    dimensions_angstrom_to_box_nm,
    minimum_image_displacement,
    wrap_positions,
)

R_KCAL_MOL_K = 1.98720425864083e-3
R_KJ_MOL_K = 8.314462618e-3

# Bondi van der Waals radii in nm
BONDI_VDW_NM: dict[str, float] = {
    "H": 0.120,
    "C": 0.170,
    "N": 0.155,
    "O": 0.152,
    "F": 0.147,
    "P": 0.180,
    "S": 0.180,
    "CL": 0.175,
    "BR": 0.185,
    "I": 0.198,
    "NA": 0.227,
    "K": 0.275,
    "MG": 0.173,
    "CA": 0.231,
}
POLAR_ELEMENTS = {"N", "O", "S", "F", "P"}


def _element_from_name(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha()).upper()
    if letters[:2] in BONDI_VDW_NM:
        return letters[:2]
    if letters[:1] in BONDI_VDW_NM:
        return letters[:1]
    return "C"


def _fibonacci_sphere(n_points: int = 92) -> np.ndarray:
    """Generate nearly uniform unit-sphere sample directions using the golden spiral."""
    n = max(12, int(n_points))
    idx = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (idx + 0.5) / n
    r_xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = idx * math.pi * (3.0 - math.sqrt(5.0))
    return np.column_stack([r_xy * np.cos(phi), r_xy * np.sin(phi), z])


def compute_sasa_shrake_rupley(
    coords_nm: np.ndarray,
    elements: list[str] | tuple[str, ...],
    *,
    probe_radius_nm: float = 0.14,
    n_sphere_points: int = 92,
    polar_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute molecular SASA (total, polar, apolar) in nm^2 using the Shrake–Rupley algorithm."""
    pts = np.asarray(coords_nm, dtype=float)
    n_at = len(pts)
    if n_at == 0:
        return {"sasa_total_nm2": 0.0, "sasa_polar_nm2": 0.0, "sasa_apolar_nm2": 0.0}

    radii = np.array([BONDI_VDW_NM.get(el.upper(), 0.170) + float(probe_radius_nm) for el in elements], dtype=float)
    if polar_mask is None:
        is_polar = np.array([el.upper() in POLAR_ELEMENTS for el in elements], dtype=bool)
    else:
        is_polar = np.asarray(polar_mask, dtype=bool)

    unit_pts = _fibonacci_sphere(n_sphere_points)  # (P, 3)
    # Pairwise atom distances to skip distant non-overlapping atoms
    diff_aa = pts[:, None, :] - pts[None, :, :]
    dist_aa = np.linalg.norm(diff_aa, axis=-1)
    r_sum = radii[:, None] + radii[None, :]

    atom_areas = np.zeros(n_at, dtype=float)
    for i in range(n_at):
        ri = radii[i]
        nbrs = np.where((dist_aa[i] < r_sum[i]) & (np.arange(n_at) != i))[0]
        if len(nbrs) == 0:
            atom_areas[i] = 4.0 * math.pi * ri * ri
            continue
        test_pts = pts[i][None, :] + ri * unit_pts  # (P, 3)
        # Check if each test point falls inside any neighbor sphere
        d_pn = test_pts[:, None, :] - pts[nbrs][None, :, :]  # (P, Nnbr, 3)
        d2_pn = np.sum(d_pn * d_pn, axis=-1)
        buried = np.any(d2_pn < (radii[nbrs][None, :] ** 2), axis=1)
        exposed_frac = float(np.mean(~buried))
        atom_areas[i] = 4.0 * math.pi * ri * ri * exposed_frac

    total = float(np.sum(atom_areas))
    polar = float(np.sum(atom_areas[is_polar]))
    return {
        "sasa_total_nm2": total,
        "sasa_polar_nm2": polar,
        "sasa_apolar_nm2": max(0.0, total - polar),
    }


def compute_radius_of_gyration(coords_nm: np.ndarray, masses: np.ndarray | None = None) -> float:
    """Compute mass-weighted radius of gyration Rg in nm."""
    pts = np.asarray(coords_nm, dtype=float)
    if len(pts) == 0:
        return 0.0
    m = np.ones(len(pts), dtype=float) if masses is None else np.asarray(masses, dtype=float)
    m_tot = max(float(np.sum(m)), 1e-12)
    com = np.sum(pts * m[:, None], axis=0) / m_tot
    r2 = np.sum((pts - com[None, :]) ** 2, axis=-1)
    return float(math.sqrt(float(np.sum(m * r2)) / m_tot))


def compute_dihedral_deg(coords: np.ndarray, idx4: tuple[int, int, int, int]) -> np.ndarray:
    """Vectorized dihedral angle in degrees [-180, 180] for coords of shape (n_samples, n_atoms, 3)."""
    i, j, k, l = idx4
    p0, p1, p2, p3 = coords[:, i], coords[:, j], coords[:, k], coords[:, l]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_u = np.divide(b1, b1_norm, out=np.zeros_like(b1), where=b1_norm > 1e-12)
    v = b0 - np.sum(b0 * b1_u, axis=-1, keepdims=True) * b1_u
    w = b2 - np.sum(b2 * b1_u, axis=-1, keepdims=True) * b1_u
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_u, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


def compute_dihedral_fes_1d(
    angles_deg: np.ndarray,
    temperature_k: float,
    *,
    n_bins: int = 36,
) -> pd.DataFrame:
    """Compute 1D dihedral Free Energy Surface F(phi) = -k_B T ln P(phi) shifted so min(F) = 0."""
    ang = np.asarray(angles_deg, dtype=float)
    if len(ang) == 0:
        return pd.DataFrame()
    hist, edges = np.histogram(ang, bins=int(n_bins), range=(-180.0, 180.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    prob = hist / max(float(np.sum(hist)), 1.0)
    rt_kcal = R_KCAL_MOL_K * float(temperature_k)
    rt_kj = R_KJ_MOL_K * float(temperature_k)
    with np.errstate(divide="ignore"):
        f_kcal = np.where(prob > 0, -rt_kcal * np.log(prob), np.nan)
        f_kj = np.where(prob > 0, -rt_kj * np.log(prob), np.nan)
    if np.any(np.isfinite(f_kcal)):
        f_kcal -= float(np.nanmin(f_kcal))
        f_kj -= float(np.nanmin(f_kj))
    return pd.DataFrame(
        {
            "angle_deg": centers,
            "probability": prob,
            "fes_kcal_mol": f_kcal,
            "fes_kj_mol": f_kj,
        }
    )


def find_rotatable_dihedrals(topo: MoleculeTopology, max_dihedrals: int = 4) -> list[tuple[int, int, int, int]]:
    """Identify heavy-atom rotatable torsions (i-j-k-l) from molecular bond graph."""
    n_at = len(topo.atom_names)
    if n_at < 4 or not topo.bonds:
        return []
    elements = [_element_from_name(name) for name in topo.atom_names]
    adj: dict[int, set[int]] = {i: set() for i in range(n_at)}
    for a, b in topo.bonds:
        if 0 <= a < n_at and 0 <= b < n_at:
            adj[a].add(b)
            adj[b].add(a)

    # Detect ring edges via simple BFS path check excluding edge (j, k)
    def _in_ring(u: int, v: int) -> bool:
        visited = {u}
        queue = [w for w in adj[u] if w != v]
        while queue:
            cur = queue.pop(0)
            if cur == v:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    queue.append(nxt)
        return False

    torsions: list[tuple[int, int, int, int]] = []
    seen_bonds: set[tuple[int, int]] = set()
    for j, k in topo.bonds:
        if elements[j] == "H" or elements[k] == "H":
            continue
        key = (min(j, k), max(j, k))
        if key in seen_bonds:
            continue
        seen_bonds.add(key)
        if _in_ring(j, k):
            continue
        nbr_j = [n for n in sorted(adj[j]) if n != k and elements[n] != "H"]
        nbr_k = [n for n in sorted(adj[k]) if n != j and elements[n] != "H"]
        if nbr_j and nbr_k:
            torsions.append((nbr_j[0], j, k, nbr_k[0]))
            if len(torsions) >= max_dihedrals:
                break
    return torsions


def compute_conformational_pca_clusters(
    coords_samples_nm: np.ndarray,
    heavy_indices: list[int],
    *,
    n_clusters: int = 4,
) -> dict[str, Any]:
    """Run PCA + K-Means clustering on internal heavy-atom pairwise distances of conformations."""
    n_samp = coords_samples_nm.shape[0]
    if n_samp < 6 or len(heavy_indices) < 3:
        return {
            "n_samples": n_samp,
            "pc1_var_ratio": float("nan"),
            "pc2_var_ratio": float("nan"),
            "n_clusters": 0,
            "top_cluster_fraction": float("nan"),
            "cluster_populations": [],
        }

    h_pos = coords_samples_nm[:, heavy_indices, :]  # (S, Nh, 3)
    nh = len(heavy_indices)
    iu = np.triu_indices(nh, k=1)
    feats = np.zeros((n_samp, len(iu[0])), dtype=float)
    for si in range(n_samp):
        diff = h_pos[si, iu[0]] - h_pos[si, iu[1]]
        feats[si] = np.linalg.norm(diff, axis=-1)

    n_comp = min(2, feats.shape[0], feats.shape[1])
    pca = PCA(n_components=n_comp)
    proj = pca.fit_transform(feats)
    var_r = pca.explained_variance_ratio_

    k = min(max(2, int(n_clusters)), n_samp // 2)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(proj)
    counts = np.bincount(labels, minlength=k)
    pops = sorted((counts / float(n_samp)).tolist(), reverse=True)

    return {
        "n_samples": n_samp,
        "pc1_var_ratio": float(var_r[0]) if len(var_r) >= 1 else float("nan"),
        "pc2_var_ratio": float(var_r[1]) if len(var_r) >= 2 else float("nan"),
        "n_clusters": k,
        "top_cluster_fraction": float(pops[0]) if pops else float("nan"),
        "cluster_populations": pops,
    }


def compute_periodic_voronoi(
    points_nm: np.ndarray,
    box_nm: np.ndarray,
    *,
    vdw_radii_nm: np.ndarray | None = None,
    grid_n: int = 24,
) -> dict[str, Any]:
    """Compute exact 3D periodic Voronoi cell volumes, asphericity, coordination, and grid free volume.

    points_nm: molecular COM coordinates (N, 3) wrapped in [0, L).
    box_nm: box edge lengths (3,) in nm.
    """
    pts = np.asarray(points_nm, dtype=float)
    box_arr = np.asarray(box_nm, dtype=float)
    n_pts = len(pts)
    vol_box = box_volume_nm3(box_arr) if n_pts >= 4 else 0.0
    if n_pts < 4 or vol_box <= 1e-12:
        return {
            "mean_voronoi_vol_nm3": float("nan"),
            "std_voronoi_vol_nm3": float("nan"),
            "mean_asphericity": float("nan"),
            "mean_voronoi_neighbors": float("nan"),
            "free_volume_fraction": float("nan"),
            "mean_cavity_radius_nm": float("nan"),
            "max_cavity_radius_nm": float("nan"),
            "cell_volumes_nm3": np.array([]),
        }

    H = box_matrix_from_dimensions(box_arr)
    pts_w = wrap_positions(pts, box_arr)
    # Build 3x3x3 periodic tiled images using general lattice vectors H (orthorhombic or triclinic)
    shifts = np.array([[ix, iy, iz] for ix in (-1, 0, 1) for iy in (-1, 0, 1) for iz in (-1, 0, 1)], dtype=float)
    zero_idx = int(np.where((shifts == 0).all(axis=1))[0][0])
    shifts[[0, zero_idx]] = shifts[[zero_idx, 0]]
    shift_vecs = shifts @ H  # (27, 3)
    tiled = np.vstack([pts_w + sv[None, :] for sv in shift_vecs])

    vor = Voronoi(tiled)
    volumes: list[float] = []
    asphericities: list[float] = []
    # Count Voronoi neighbors from ridge_points touching central particles
    nbr_counts = np.zeros(n_pts, dtype=int)
    for p1, p2 in vor.ridge_points:
        if p1 < n_pts:
            nbr_counts[p1] += 1
        if p2 < n_pts:
            nbr_counts[p2] += 1

    for i in range(n_pts):
        reg_idx = vor.point_region[i]
        verts_idx = vor.regions[reg_idx]
        if not verts_idx or -1 in verts_idx or len(verts_idx) < 4:
            continue
        verts = vor.vertices[verts_idx]
        try:
            hull = ConvexHull(verts)
            vol = float(hull.volume)
            area = float(hull.area)
            volumes.append(vol)
            # Isoperimetric asphericity alpha = A^3 / (36 pi V^2) >= 1 (1.0 for perfect sphere)
            if vol > 1e-12:
                asphericities.append((area ** 3) / (36.0 * math.pi * vol * vol))
        except Exception:
            continue

    # Grid-probed free volume & cavity radius distribution (uniform in fractional unit-cell coordinates)
    gn = max(8, int(grid_n))
    f1d = (np.arange(gn) + 0.5) / gn
    fgrid = np.stack(np.meshgrid(f1d, f1d, f1d, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = fgrid @ H  # (G, 3)

    if vdw_radii_nm is None:
        r_eff = np.full(n_pts, 0.35 * ((vol_box / n_pts) ** (1.0 / 3.0)), dtype=float)
    else:
        r_eff = np.asarray(vdw_radii_nm, dtype=float)

    # Compute clearance distance = min_i (dist(grid_point, p_i) - r_i) using exact PBC minimum-image
    clearances = np.empty(len(grid), dtype=float)
    chunk = 512
    for start in range(0, len(grid), chunk):
        g_sub = grid[start : start + chunk]
        diff = minimum_image_displacement(g_sub[:, None, :] - pts_w[None, :, :], box_arr)
        dist = np.linalg.norm(diff, axis=-1) - r_eff[None, :]
        clearances[start : start + chunk] = np.min(dist, axis=1)

    void_mask = clearances > 0.0
    free_vol_frac = float(np.mean(void_mask))
    mean_cav = float(np.mean(clearances[void_mask])) if np.any(void_mask) else 0.0
    max_cav = float(np.max(clearances[void_mask])) if np.any(void_mask) else 0.0

    vol_arr = np.asarray(volumes, dtype=float)
    return {
        "mean_voronoi_vol_nm3": float(np.mean(vol_arr)) if len(vol_arr) else float("nan"),
        "std_voronoi_vol_nm3": float(np.std(vol_arr, ddof=1)) if len(vol_arr) > 1 else 0.0,
        "mean_asphericity": float(np.mean(asphericities)) if asphericities else float("nan"),
        "mean_voronoi_neighbors": float(np.mean(nbr_counts)),
        "free_volume_fraction": free_vol_frac,
        "mean_cavity_radius_nm": mean_cav,
        "max_cavity_radius_nm": max_cav,
        "cell_volumes_nm3": vol_arr,
    }


def _unwrap_molecules_frame(ag: mda.AtomGroup, topo: MoleculeTopology, dimensions: np.ndarray) -> np.ndarray:
    """Return whole molecules of shape (n_mol, n_atoms, 3) in nm."""
    n_at = len(topo.atom_names)
    pos_nm = np.asarray(ag.positions, dtype=float) / 10.0
    n_mol = len(pos_nm) // n_at
    pos_nm = pos_nm[: n_mol * n_at].reshape(n_mol, n_at, 3)
    box_nm = dimensions_angstrom_to_box_nm(dimensions)
    ref = pos_nm[:, 0:1, :]
    rel = minimum_image_displacement(pos_nm - ref, box_nm)
    return ref + rel


def run_conformation_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Run SASA, Rg, rotatable dihedral FES, and PCA conformational clustering per target species."""
    opts = (cfg.analyses or {}).get("options") or {}
    probe_nm = float(opts.get("sasa_probe_nm", 0.14))
    n_pts = int(opts.get("sasa_n_points", 92))

    summary_rows: list[dict[str, Any]] = []
    fes_rows: list[dict[str, Any]] = []

    for run in runs:
        if not (run.prod_tpr.exists() and run.prod_xtc.exists()):
            continue
        u = mda.Universe(str(run.prod_tpr), str(run.prod_xtc))
        f_start, f_stop, f_step = equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, stride=max(1, cfg.rdf_stride))
        frames = list(range(f_start, f_stop, f_step))
        if not frames:
            continue
        frames = frames[:200]
        temp_k = float(run.metadata.get("temperatureK") or run.metadata.get("temperature_k") or cfg.temperature_k)

        # Target solute first; if pure system, analyze polyatomic solvents (up to 20 molecules per frame)
        targets = [run.solute_resname] if run.solute_resname else [
            sp for sp, topo in sorted(run.topologies.items()) if len(topo.atom_names) >= 3
        ]
        for sp in targets:
            if sp not in run.topologies:
                continue
            topo = run.topologies[sp]
            ag = u.select_atoms(f"resname {sp}")
            if len(ag) == 0 or len(topo.atom_names) < 2:
                continue
            elements = [_element_from_name(nm) for nm in topo.atom_names]
            masses = np.asarray(topo.atom_masses, dtype=float)
            heavy_idx = [idx for idx, el in enumerate(elements) if el != "H"]

            samples: list[np.ndarray] = []
            sasa_tot, sasa_pol, sasa_apol, rg_vals = [], [], [], []
            for fi in frames:
                u.trajectory[fi]
                mols = _unwrap_molecules_frame(ag, topo, u.trajectory.ts.dimensions)
                # Cap molecules per frame to keep SASA fast for bulk solvents
                mols_sub = mols[: min(len(mols), 10)]
                for m_coords in mols_sub:
                    samples.append(m_coords)
                    rg_vals.append(compute_radius_of_gyration(m_coords, masses))
                # Evaluate SASA on first up to 3 molecules per frame
                for m_coords in mols_sub[: min(len(mols_sub), 3)]:
                    s = compute_sasa_shrake_rupley(
                        m_coords, elements, probe_radius_nm=probe_nm, n_sphere_points=n_pts
                    )
                    sasa_tot.append(s["sasa_total_nm2"])
                    sasa_pol.append(s["sasa_polar_nm2"])
                    sasa_apol.append(s["sasa_apolar_nm2"])

            if not samples:
                continue
            samp_arr = np.stack(samples, axis=0)
            pca_res = compute_conformational_pca_clusters(samp_arr, heavy_idx)
            torsions = find_rotatable_dihedrals(topo, max_dihedrals=3)
            for t_i, idx4 in enumerate(torsions):
                ang = compute_dihedral_deg(samp_arr, idx4)
                fes_df = compute_dihedral_fes_1d(ang, temp_k)
                if not fes_df.empty:
                    label = "-".join(topo.atom_names[k] for k in idx4)
                    fes_df.insert(0, "dihedral_label", label)
                    fes_df.insert(0, "dihedral_index", t_i)
                    fes_df.insert(0, "species", sp)
                    fes_df.insert(0, "replica", run.replica)
                    fes_df.insert(0, "system_code", run.system_code)
                    fes_df.insert(0, "run_id", run.run_id)
                    fes_rows.extend(fes_df.to_dict(orient="records"))

            summary_rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "species": sp,
                    "n_conformations": len(samples),
                    "rg_mean_nm": float(np.mean(rg_vals)),
                    "rg_sd_nm": float(np.std(rg_vals, ddof=1)) if len(rg_vals) > 1 else 0.0,
                    "sasa_total_mean_nm2": float(np.mean(sasa_tot)),
                    "sasa_polar_mean_nm2": float(np.mean(sasa_pol)),
                    "sasa_apolar_mean_nm2": float(np.mean(sasa_apol)),
                    "n_rotatable_dihedrals": len(torsions),
                    "pca_pc1_var_ratio": pca_res["pc1_var_ratio"],
                    "pca_pc2_var_ratio": pca_res["pc2_var_ratio"],
                    "top_cluster_fraction": pca_res["top_cluster_fraction"],
                }
            )

    return {
        "conformation_summary": pd.DataFrame(summary_rows),
        "dihedral_fes": pd.DataFrame(fes_rows),
    }


def run_voronoi_free_volume_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Run periodic 3D Voronoi tessellation & cavity/free-volume analysis across runs."""
    opts = (cfg.analyses or {}).get("options") or {}
    v_stride = max(1, int(opts.get("voronoi_stride", max(10, cfg.network_stride))))
    rows: list[dict[str, Any]] = []

    for run in runs:
        if not (run.prod_tpr.exists() and run.prod_xtc.exists()):
            continue
        u = mda.Universe(str(run.prod_tpr), str(run.prod_xtc))
        f_start, f_stop, f_step = equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, stride=v_stride)
        frames = list(range(f_start, f_stop, f_step))[:25]
        if not frames:
            continue

        species = [sp for sp, cnt in sorted(run.mol_counts.items()) if cnt > 0 and sp in run.topologies]
        ags = {sp: u.select_atoms(f"resname {sp}") for sp in species}

        frame_metrics: list[dict[str, Any]] = []
        for fi in frames:
            u.trajectory[fi]
            ts = u.trajectory.ts
            box_nm = dimensions_angstrom_to_box_nm(ts.dimensions)
            coms_all: list[np.ndarray] = []
            for sp in species:
                if len(ags[sp]) > 0:
                    coms_all.append(compute_whole_coms_vectorized(ags[sp], run.topologies[sp], ts.dimensions) / 10.0)
            if not coms_all:
                continue
            pts = np.vstack(coms_all)
            res = compute_periodic_voronoi(pts, box_nm, grid_n=20)
            frame_metrics.append(res)

        if frame_metrics:
            rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "n_frames_sampled": len(frame_metrics),
                    "mean_voronoi_vol_nm3": float(np.nanmean([m["mean_voronoi_vol_nm3"] for m in frame_metrics])),
                    "std_voronoi_vol_nm3": float(np.nanmean([m["std_voronoi_vol_nm3"] for m in frame_metrics])),
                    "mean_asphericity": float(np.nanmean([m["mean_asphericity"] for m in frame_metrics])),
                    "mean_voronoi_neighbors": float(np.nanmean([m["mean_voronoi_neighbors"] for m in frame_metrics])),
                    "free_volume_fraction": float(np.nanmean([m["free_volume_fraction"] for m in frame_metrics])),
                    "mean_cavity_radius_nm": float(np.nanmean([m["mean_cavity_radius_nm"] for m in frame_metrics])),
                    "max_cavity_radius_nm": float(np.nanmax([m["max_cavity_radius_nm"] for m in frame_metrics])),
                }
            )

    return {"voronoi_free_volume": pd.DataFrame(rows)}
