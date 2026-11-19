from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from mody_analyse.core.config import Config


def write_multiwfn_nci_script(out_path: Path, grid_quality: int = 2) -> Path:
    """Write Multiwfn batch input for NCI/RDG scatter + sl2r/RDG cube export."""
    lines = ["20", "1", "1", str(grid_quality), "2", "3", "0", "0", "q"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_multiwfn_igmh_script(
    out_path: Path,
    frag1: tuple[int, int],
    frag2: tuple[int, int],
    grid_quality: int = 2,
    frag3: tuple[int, int] | None = None,
) -> Path:
    """Write Multiwfn batch input for Hirshfeld-partitioned IGMH scatter + cube export."""
    if frag3 is None:
        lines = [
            "20",
            "11",
            "2",
            f"{frag1[0]}-{frag1[1]}",
            f"{frag2[0]}-{frag2[1]}",
            str(grid_quality),
            "2",
            "3",
            "0",
            "0",
            "q",
        ]
    else:
        lines = [
            "20",
            "11",
            "3",
            f"{frag1[0]}-{frag1[1]}",
            f"{frag2[0]}-{frag2[1]}",
            f"{frag3[0]}-{frag3[1]}",
            str(grid_quality),
            "2",
            "3",
            "0",
            "0",
            "q",
        ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_vmd_nci_script(
    vmd_path: Path,
    mol_xyz: Path,
    func1_cub: Path,
    func2_cub: Path,
    iso_val: float = 0.005,
) -> Path:
    txt = f"""# VMD visualization script for Multiwfn NCI / IGMH isosurface
mol new {mol_xyz.name} type xyz waitfor all
mol modstyle 0 0 Licorice 0.14 12.0 12.0
mol addfile {func1_cub.name} type cube waitfor all
mol addfile {func2_cub.name} type cube waitfor all
mol addrep 0
mol modstyle 1 0 Isosurface {iso_val:.4f} 1 0 0 1 1
mol modcolor 1 0 Volume 0
mol scaleminmax 0 1 -0.04 0.02
color scale method BGR
display projection Orthographic
axes location Off
"""
    vmd_path.write_text(txt, encoding="utf-8")
    return vmd_path


def convert_gbw_to_molden(gbw_path: Path, orca_2mkl_bin: Path, openmpi_dir: Path | None = None) -> Path:
    molden_path = gbw_path.parent / f"{gbw_path.stem}.molden.input"
    if molden_path.exists() and molden_path.stat().st_size > 100:
        return molden_path
    env = os.environ.copy()
    orca_dir = str(orca_2mkl_bin.parent)
    path_parts = [orca_dir]
    lib_parts = [f"{orca_dir}/lib"]
    if openmpi_dir is not None:
        path_parts.append(str(openmpi_dir / "bin"))
        lib_parts.append(str(openmpi_dir / "lib"))
    if env.get("PATH"):
        path_parts.append(env["PATH"])
    if env.get("LD_LIBRARY_PATH"):
        lib_parts.append(env["LD_LIBRARY_PATH"])
    env["PATH"] = ":".join(path_parts)
    env["LD_LIBRARY_PATH"] = ":".join(lib_parts)
    proc = subprocess.run(
        [str(orca_2mkl_bin), gbw_path.stem, "-molden"],
        cwd=str(gbw_path.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not molden_path.exists():
        raise RuntimeError(f"orca_2mkl failed for {gbw_path}: {proc.stderr}")
    return molden_path


def _run_multiwfn_batch(molden_path: Path, batch_in: Path, workdir: Path, multiwfn_bin: Path) -> Path:
    molden_abs = Path(molden_path).resolve()
    env = os.environ.copy()
    env["KMP_STACKSIZE"] = "256M"
    env["OMP_NUM_THREADS"] = "4"
    env["Multiwfnpath"] = f"{multiwfn_bin.parent.resolve()}/"
    log_path = workdir / f"{batch_in.stem}.log"
    with batch_in.open("r", encoding="utf-8") as fin:
        proc = subprocess.run(
            ["bash", "-c", f'ulimit -s unlimited && "{multiwfn_bin.resolve()}" "{molden_abs}"'],
            cwd=str(workdir),
            stdin=fin,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Multiwfn failed in {workdir} (rc={proc.returncode})")
    return log_path


def parse_igmh_integral_au(log_path: Path) -> dict[str, float]:
    """Parse integrated delta-g, delta-g_inter, and delta-g_intra (a.u.) from Multiwfn IGMH log."""
    txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

    def _pick(pat: str) -> float:
        m = re.search(pat, txt)
        return float(m.group(1)) if m else float("nan")

    return {
        "integral_dg_total_au": _pick(r"Integral of delta-g over whole space:\s+([-\d.Ee+]+)"),
        "integral_dg_inter_au": _pick(r"Integral of delta-g_inter over whole space:\s+([-\d.Ee+]+)"),
        "integral_dg_intra_au": _pick(r"Integral of delta-g_intra over whole space:\s+([-\d.Ee+]+)"),
    }


def _summarize_scatter(output_txt: Path, y_thresh: float = 0.5, kind: str = "auto") -> dict[str, float]:
    """Summarize Multiwfn NCI (5-col: x,y,z,sl2r,RDG) or IGMH (4-col: dg_inter,dg_intra,dg,sl2r) scatter."""
    arr = np.loadtxt(output_txt)
    if arr.ndim == 1:
        arr = arr[None, :]
    mode = kind
    if mode == "auto":
        mode = "igmh" if arr.shape[1] == 4 else "nci"

    sl2r = arr[:, 3]
    if mode == "igmh":
        dg_inter = arr[:, 0]
        min_dg = 1e-3
        active = dg_inter >= min_dg
        sl_sub = sl2r[active] if np.any(active) else sl2r
    else:
        rdg = arr[:, 4]
        active = rdg <= y_thresh if np.min(rdg) < y_thresh else np.ones(len(rdg), dtype=bool)
        sl_sub = sl2r[active]

    return {
        "n_points_total": int(len(arr)),
        "n_points_active": int(np.sum(active)),
        "frac_attractive_hb": float(np.mean(sl_sub < -0.015)) if len(sl_sub) else 0.0,
        "frac_vdw": float(np.mean((sl_sub >= -0.015) & (sl_sub <= 0.010))) if len(sl_sub) else 0.0,
        "frac_steric": float(np.mean(sl_sub > 0.010)) if len(sl_sub) else 0.0,
        "min_sl2r": float(np.min(sl_sub)) if len(sl_sub) else 0.0,
    }


def run_multiwfn_nci_igmh(
    dft_dimers: pd.DataFrame,
    dft_trimers: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Analysis #8: Convert ORCA .gbw -> .molden.input and execute Multiwfn NCI & IGMH + VMD."""
    nci_dir = cfg.results_dir / "nci_igmh"
    nci_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Select up to 2 representative dimers per discovered solute system + 1 strongest trimer per system
    selected_items: list[dict[str, Any]] = []
    if not dft_dimers.empty:
        for sys_code in sorted(dft_dimers["system_code"].dropna().unique().tolist()):
            sub = dft_dimers.loc[dft_dimers["system_code"] == sys_code].sort_values("de_int_r2scan3c_kcal")
            if not sub.empty:
                selected_items.append({**sub.iloc[0].to_dict(), "kind": "dimer"})
            if len(sub) > 1:
                selected_items.append({**sub.iloc[-1].to_dict(), "kind": "dimer"})
    if not dft_trimers.empty:
        for sys_code in sorted(dft_trimers["system_code"].dropna().unique().tolist()):
            sub = dft_trimers.loc[dft_trimers["system_code"] == sys_code].sort_values("e_3body_r2scan3c_kcal")
            if not sub.empty:
                selected_items.append({**sub.iloc[0].to_dict(), "kind": "trimer"})

    selected_items = selected_items[: cfg.nci_clusters_total]

    for item in selected_items:
        cid = item["cluster_id"]
        gbw = Path(item["gbw_path"])
        if not gbw.exists():
            continue
        molden = convert_gbw_to_molden(gbw, cfg.orca_2mkl_bin, openmpi_dir=cfg.openmpi_dir)
        cdir = nci_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        local_xyz = cdir / f"{cid}.xyz"
        local_xyz.write_text(Path(item["xyz_path"]).read_text(encoding="utf-8"), encoding="utf-8")

        na = int(item["n_atoms_a"])
        nb = int(item["n_atoms_b"])
        frag1 = (1, na)
        frag2 = (na + 1, na + nb)
        frag3 = (na + nb + 1, na + nb + int(item["n_atoms_c"])) if item["kind"] == "trimer" else None

        # 1. NCI (RDG)
        nci_sub = cdir / "nci"
        nci_sub.mkdir(exist_ok=True)
        (nci_sub / local_xyz.name).write_text(local_xyz.read_text(encoding="utf-8"), encoding="utf-8")
        nci_in = write_multiwfn_nci_script(nci_sub / "nci.mwfn.in", grid_quality=2)
        if not (nci_sub / "output.txt").exists():
            _run_multiwfn_batch(molden, nci_in, nci_sub, cfg.multiwfn_bin)
        write_vmd_nci_script(nci_sub / "view_nci.vmd", local_xyz, nci_sub / "func1.cub", nci_sub / "func2.cub", iso_val=0.5)
        nci_stats = _summarize_scatter(nci_sub / "output.txt", y_thresh=0.5, kind="nci")

        # 2. IGMH
        igmh_sub = cdir / "igmh"
        igmh_sub.mkdir(exist_ok=True)
        (igmh_sub / local_xyz.name).write_text(local_xyz.read_text(encoding="utf-8"), encoding="utf-8")
        igmh_in = write_multiwfn_igmh_script(igmh_sub / "igmh.mwfn.in", frag1=frag1, frag2=frag2, grid_quality=2, frag3=frag3)
        igmh_log = igmh_sub / "igmh.mwfn.log"
        if not (igmh_sub / "output.txt").exists() or not igmh_log.exists():
            igmh_log = _run_multiwfn_batch(molden, igmh_in, igmh_sub, cfg.multiwfn_bin)
        write_vmd_nci_script(igmh_sub / "view_igmh.vmd", local_xyz, igmh_sub / "sl2r.cub", igmh_sub / "dg_inter.cub", iso_val=0.005)
        igmh_stats = _summarize_scatter(igmh_sub / "output.txt", kind="igmh")
        igmh_integ = parse_igmh_integral_au(igmh_log)

        rows.append(
            {
                "cluster_id": cid,
                "kind": item["kind"],
                "system_code": item["system_code"],
                "motif": item.get("motif", ""),
                "nci_scatter_path": str(nci_sub / "output.txt"),
                "igmh_scatter_path": str(igmh_sub / "output.txt"),
                "igmh_vmd_script": str(igmh_sub / "view_igmh.vmd"),
                "nci_vmd_script": str(nci_sub / "view_nci.vmd"),
                "igmh_frac_hb": igmh_stats["frac_attractive_hb"],
                "igmh_frac_vdw": igmh_stats["frac_vdw"],
                "igmh_frac_steric": igmh_stats["frac_steric"],
                "igmh_min_sl2r": igmh_stats["min_sl2r"],
                "nci_min_sl2r": nci_stats["min_sl2r"],
                "integral_dg_inter_au": igmh_integ["integral_dg_inter_au"],
                "integral_dg_intra_au": igmh_integ["integral_dg_intra_au"],
            }
        )

    return pd.DataFrame(rows)
