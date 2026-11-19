from __future__ import annotations

import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

from mody_analyse.core.config import Config
from mody_analyse.qm_runners.qm_xtb import HARTREE_TO_KCAL, _read_xyz


def parse_orca_sp_energy_eh(output_text: str) -> float:
    matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", output_text)
    if not matches:
        raise ValueError("FINAL SINGLE POINT ENERGY not found in ORCA output")
    return float(matches[-1])


def _write_if_changed(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return path
        except OSError:
            pass
    path.write_text(text, encoding="utf-8")
    return path


def write_orca_dft_input(
    inp_path: Path,
    coords: list[tuple[str, float, float, float]],
    method: str = "r2SCAN-3c",
    nprocs: int = 4,
    maxcore_mb: int = 1200,
    keep_gbw: bool = True,
    opt: bool = False,
    scf_tol: str = "TightSCF",
) -> Path:
    keywords = f"! {method} {scf_tol} {'Opt' if opt else 'SP'}"
    lines = [
        keywords,
        "%pal",
        f"  nprocs {nprocs}",
        "end",
        f"%maxcore {maxcore_mb}",
    ]
    if not keep_gbw:
        lines.append("%output PrintLevel Mini end")
    lines.append("* xyz 0 1")
    for el, x, y, z in coords:
        lines.append(f"  {el:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    lines.append("*")
    return _write_if_changed(inp_path, "\n".join(lines) + "\n")


def write_orca_cp_dimer_inputs(
    out_dir: Path,
    coords_a: list[tuple[str, float, float, float]],
    coords_b: list[tuple[str, float, float, float]],
    method: str = "wB97X-D3 def2-TZVP",
    nprocs: int = 4,
    maxcore_mb: int = 1200,
    scf_tol: str = "TightSCF",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_one(name: str, atoms: list[tuple[str, float, float, float, bool]]) -> Path:
        p = out_dir / f"{name}.inp"
        lines = [
            f"! {method} {scf_tol} SP",
            "%pal",
            f"  nprocs {nprocs}",
            "end",
            f"%maxcore {maxcore_mb}",
            "* xyz 0 1",
        ]
        for el, x, y, z, ghost in atoms:
            tag = f"{el} :" if ghost else el
            lines.append(f"  {tag:4s} {x:14.8f} {y:14.8f} {z:14.8f}")
        lines.append("*")
        return _write_if_changed(p, "\n".join(lines) + "\n")

    a_real = [(el, x, y, z, False) for el, x, y, z in coords_a]
    b_real = [(el, x, y, z, False) for el, x, y, z in coords_b]
    a_ghost = [(el, x, y, z, True) for el, x, y, z in coords_a]
    b_ghost = [(el, x, y, z, True) for el, x, y, z in coords_b]

    return {
        "ab": _write_one("AB", a_real + b_real),
        "a_ghost_b": _write_one("A_ghostB", a_real + b_ghost),
        "b_ghost_a": _write_one("B_ghostA", a_ghost + b_real),
        "a_only": _write_one("A_only", a_real),
        "b_only": _write_one("B_only", b_real),
    }


def write_orca_led_input(
    inp_path: Path,
    coords_a: list[tuple[str, float, float, float]],
    coords_b: list[tuple[str, float, float, float]],
    basis: str = "def2-TZVP",
    nprocs: int = 2,
    maxcore_mb: int = 1000,
) -> Path:
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    aux_c = f"{basis}/C"
    aux_j = "def2/J"
    lines = [
        f"! DLPNO-CCSD(T) {basis} {aux_c} {aux_j} RIJCOSX NormalPNO TightSCF LED",
        "%pal",
        f"  nprocs {nprocs}",
        "end",
        f"%maxcore {maxcore_mb}",
        "* xyz 0 1",
    ]
    for el, x, y, z in coords_a:
        lines.append(f"  {el}(1) {x:14.8f} {y:14.8f} {z:14.8f}")
    for el, x, y, z in coords_b:
        lines.append(f"  {el}(2) {x:14.8f} {y:14.8f} {z:14.8f}")
    lines.append("*")
    return _write_if_changed(inp_path, "\n".join(lines) + "\n")


def choose_led_execution_settings(
    peak_rss_gb: float,
    ram_limit_gb: float,
    current_nprocs: int,
    current_basis: str,
) -> dict[str, Any]:
    if peak_rss_gb <= ram_limit_gb:
        return {"nprocs": current_nprocs, "basis": current_basis, "maxcore_mb": 1000, "emit_hpc_deck": False}
    if current_nprocs > 1:
        return {"nprocs": 1, "basis": current_basis, "maxcore_mb": 1000, "emit_hpc_deck": False}
    return {"nprocs": 1, "basis": "def2-SVP", "maxcore_mb": 1000, "emit_hpc_deck": True}


def parse_orca_led_output(output_text: str) -> dict[str, float]:
    """Parse ORCA 6.1.1 LED components (plus legacy/test format aliases) in kcal/mol."""

    def _find_eh_or_kcal(pattern_list: list[str]) -> float:
        for pat in pattern_list:
            m = re.search(pat, output_text)
            if m:
                if m.lastindex and m.lastindex >= 2 and m.group(2) is not None:
                    return float(m.group(2))
                return float(m.group(1)) * HARTREE_TO_KCAL
        return 0.0

    e_elstat = _find_eh_or_kcal(
        [
            r"Electrostatics\s*\(REF\.\)\s+(-?\d+\.\d+)",
            r"Sum of electrostatics\s*=\s*(-?\d+\.\d+)\s*\(\s*(-?\d+\.\d+)\s*kcal/mol\)",
            r"Electrostatic interaction\s*\(Ref\.\)\s*:\s*(-?\d+\.\d+)\s*Eh\s+(-?\d+\.\d+)\s*kcal/mol",
        ]
    )
    e_exch = _find_eh_or_kcal(
        [
            r"Exchange\s*\(REF\.\)\s+(-?\d+\.\d+)",
            r"Two electron exchange\s*=\s*(-?\d+\.\d+)\s*\(\s*(-?\d+\.\d+)\s*kcal/mol\)",
            r"Exchange interaction\s*\(Ref\.\)\s*:\s*(-?\d+\.\d+)\s*Eh\s+(-?\d+\.\d+)\s*kcal/mol",
        ]
    )
    e_disp_strong = _find_eh_or_kcal(
        [
            r"Dispersion\s*\(strong pairs\)\s*:?\s*(-?\d+\.\d+)(?:\s*Eh\s+(-?\d+\.\d+)\s*kcal/mol)?",
        ]
    )
    e_disp_weak = _find_eh_or_kcal(
        [
            r"Dispersion\s*\(weak pairs\)\s*:?\s*(-?\d+\.\d+)(?:\s*Eh\s+(-?\d+\.\d+)\s*kcal/mol)?",
        ]
    )
    e_disp = e_disp_strong + e_disp_weak

    e_ct_12 = _find_eh_or_kcal([r"Charge Transfer 1 to 2\s+(-?\d+\.\d+)"])
    e_ct_21 = _find_eh_or_kcal([r"Charge Transfer 2 to 1\s+(-?\d+\.\d+)"])
    e_triples = _find_eh_or_kcal([r"Inter triples\s+(-?\d+\.\d+)"])
    e_inter_corr = _find_eh_or_kcal(
        [
            r"Interfragment correlation\s+(-?\d+\.\d+)\s*\(\s*(-?\d+\.\d+)\s*kcal/mol\)",
            r"Total interaction\s+(-?\d+\.\d+)\s*\(\s*(-?\d+\.\d+)\s*kcal/mol\)",
        ]
    )
    e_non_disp_corr = (e_inter_corr - e_disp) if e_inter_corr != 0.0 else (e_ct_12 + e_ct_21 + e_triples)
    e_prep_ref = _find_eh_or_kcal(
        [r"Electronic preparation\s*\(Ref\.\)\s*:\s*(-?\d+\.\d+)\s*Eh\s+(-?\d+\.\d+)\s*kcal/mol"]
    )
    e_intra_sum_eh = 0.0
    m_intra = re.search(r"Sum of INTRA-fragment total energies\s*=\s*(-?\d+\.\d+)", output_text)
    if m_intra:
        e_intra_sum_eh = float(m_intra.group(1))
    m_tot = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", output_text)
    e_total_eh = float(m_tot.group(1)) if m_tot else 0.0

    return {
        "e_elstat_kcal": float(e_elstat),
        "e_exch_kcal": float(e_exch),
        "e_disp_kcal": float(e_disp),
        "e_ct_12_kcal": float(e_ct_12),
        "e_ct_21_kcal": float(e_ct_21),
        "e_triples_inter_kcal": float(e_triples),
        "e_non_disp_corr_kcal": float(e_non_disp_corr),
        "e_prep_kcal": float(e_prep_ref),
        "e_intra_sum_eh": float(e_intra_sum_eh),
        "e_total_eh": float(e_total_eh),
    }


def _tree_rss_gb(root_pid: int) -> float:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,ppid,rss"], text=True)
    except Exception:
        return 0.0
    children: dict[int, list[int]] = {}
    rss_kb: dict[int, int] = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, r = int(parts[0]), int(parts[1]), int(parts[2])
        children.setdefault(ppid, []).append(pid)
        rss_kb[pid] = r
    total_kb = 0
    stack = [root_pid]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        total_kb += rss_kb.get(cur, 0)
        stack.extend(children.get(cur, []))
    return total_kb / (1024.0 * 1024.0)


def _set_unlimited_stack() -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        target = hard if hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
        if soft != target:
            resource.setrlimit(resource.RLIMIT_STACK, (target, hard))
    except Exception:
        pass


def _cleanup_orca_tmp(inp_path: Path) -> None:
    for stale in inp_path.parent.glob(f"{inp_path.stem}*.tmp*"):
        stale.unlink(missing_ok=True)


def _run_orca(
    inp_path: Path,
    orca_bin: Path,
    monitor_rss: bool = False,
    abort_rss_gb: float | None = None,
    openmpi_dir: Path | None = None,
) -> tuple[Path, float]:
    out_path = inp_path.with_suffix(".out")
    if out_path.exists() and out_path.stat().st_mtime >= inp_path.stat().st_mtime:
        txt = out_path.read_text(encoding="utf-8", errors="ignore")
        if "****ORCA TERMINATED NORMALLY****" in txt:
            return out_path, 0.0

    # Clean stale tmp files if retrying
    _cleanup_orca_tmp(inp_path)

    env = os.environ.copy()
    orca_dir = str(orca_bin.parent)
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

    ok = False
    peak_rss = 0.0
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                [str(orca_bin), inp_path.name],
                cwd=str(inp_path.parent),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_set_unlimited_stack,
            )
            while proc.poll() is None:
                if monitor_rss or abort_rss_gb is not None:
                    cur_rss = _tree_rss_gb(proc.pid)
                    peak_rss = max(peak_rss, cur_rss)
                    if abort_rss_gb is not None and cur_rss > abort_rss_gb:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                            proc.wait(timeout=3)
                        except Exception:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except OSError:
                                pass
                        raise MemoryError(f"ORCA RSS {cur_rss:.3f} GiB exceeded limit {abort_rss_gb:.3f} GiB on {inp_path}")
                time.sleep(0.5 if abort_rss_gb is not None else 1.5 if monitor_rss else 0.5)
            rc = proc.returncode

        txt = out_path.read_text(encoding="utf-8", errors="ignore")
        if rc != 0 or "****ORCA TERMINATED NORMALLY****" not in txt:
            raise RuntimeError(f"ORCA failed on {inp_path} (rc={rc}); check {out_path}")
        ok = True
        return out_path, peak_rss
    finally:
        if not ok:
            _cleanup_orca_tmp(inp_path)


def _xyz_to_atom_tuples(xyz_path: Path) -> list[tuple[str, float, float, float]]:
    elems, coords = _read_xyz(xyz_path)
    return [(el, float(c[0]), float(c[1]), float(c[2])) for el, c in zip(elems, coords)]


def select_representative_dft_clusters(
    xtb_dimers: pd.DataFrame,
    xtb_trimers: pd.DataFrame,
    xtb_strain: pd.DataFrame,
    cfg: Config,
) -> dict[str, pd.DataFrame]:
    # 1. Dimers: 3 representative per pair (strongest, median, weak/vdW) + 1 CP validation per pair
    sel_dimers = []
    sel_cp = []
    for pair, grp in xtb_dimers.groupby("pair"):
        g = grp.sort_values("de_int_kcal").reset_index(drop=True)
        n = len(g)
        idxs = sorted(set(np.linspace(0, n - 1, min(cfg.dft_dimers_per_pair, n), dtype=int).tolist()))
        sel_dimers.append(g.iloc[idxs])
        # Pick representative near Q1/median hbond (if present) for CP check
        hb = g.loc[g["motif"] == "hbond"]
        pick_cp = hb.iloc[len(hb) // 2 : len(hb) // 2 + 1] if not hb.empty else g.iloc[n // 2 : n // 2 + 1]
        sel_cp.append(pick_cp)

    # 2. Trimers: up to cfg.dft_trimers_per_solute per solute across available motifs
    sel_trimers = []
    for sys_code, grp in xtb_trimers.groupby("system_code"):
        g = grp.sort_values("e_3body_kcal").reset_index(drop=True)
        n = len(g)
        idxs = sorted(set(np.linspace(0, n - 1, min(cfg.dft_trimers_per_solute, n), dtype=int).tolist()))
        sel_trimers.append(g.iloc[idxs])

    # 3. Strain validation: 3 conformers per solute (min, median, max strain)
    sel_strain = []
    for sol, grp in xtb_strain.groupby("solute_resname"):
        g = grp.sort_values("e_strain_relaxed_kcal").reset_index(drop=True)
        n = len(g)
        idxs = sorted(set(np.linspace(0, n - 1, min(3, n), dtype=int).tolist()))
        sel_strain.append(g.iloc[idxs])

    return {
        "dft_dimers": pd.concat(sel_dimers, ignore_index=True) if sel_dimers else pd.DataFrame(),
        "cp_dimers": pd.concat(sel_cp, ignore_index=True) if sel_cp else pd.DataFrame(),
        "dft_trimers": pd.concat(sel_trimers, ignore_index=True) if sel_trimers else pd.DataFrame(),
        "dft_strain": pd.concat(sel_strain, ignore_index=True) if sel_strain else pd.DataFrame(),
    }


def run_orca_dft_dimers_and_trimers(
    selected: dict[str, pd.DataFrame],
    ref_conformers: pd.DataFrame,
    cfg: Config,
) -> dict[str, pd.DataFrame]:
    """Run dispersion-corrected ORCA r2SCAN-3c on dimers, trimers, strain subset + wB97X-D3 CP subset."""
    base_dir = cfg.results_dir / "orca"
    dimer_rows: list[dict[str, Any]] = []
    cp_rows: list[dict[str, Any]] = []
    trimer_rows: list[dict[str, Any]] = []
    strain_rows: list[dict[str, Any]] = []

    # 1. r2SCAN-3c dimers (keep .gbw for Analysis #8 NCI/IGMH!)
    dft_dimers = selected.get("dft_dimers", pd.DataFrame())
    for i_d, (_, row) in enumerate(dft_dimers.iterrows(), start=1):
        cid = row["cluster_id"]
        print(f"  [ORCA r2SCAN-3c dimer {i_d}/{len(dft_dimers)}] {cid}")
        na = int(row["n_atoms_a"])
        atoms = _xyz_to_atom_tuples(Path(row["xyz_path"]))
        cdir = base_dir / "dimers_r2scan3c" / cid
        e_map = {}
        gbw_ab = cdir / "AB.gbw"
        for tag, sub in [("AB", atoms), ("A", atoms[:na]), ("B", atoms[na:])]:
            inp = write_orca_dft_input(
                cdir / f"{tag}.inp",
                sub,
                method="r2SCAN-3c",
                nprocs=cfg.orca_nprocs,
                maxcore_mb=cfg.orca_maxcore_mb,
                keep_gbw=(tag == "AB"),
            )
            out_p, _ = _run_orca(inp, cfg.orca_bin, openmpi_dir=cfg.openmpi_dir)
            e_map[tag] = parse_orca_sp_energy_eh(out_p.read_text(encoding="utf-8", errors="ignore"))
        de_dft = (e_map["AB"] - e_map["A"] - e_map["B"]) * HARTREE_TO_KCAL
        dimer_rows.append(
            {
                **row.to_dict(),
                "de_int_r2scan3c_kcal": float(de_dft),
                "gbw_path": str(gbw_ab),
            }
        )

    # 2. wB97X-D3/def2-TZVP Counterpoise BSSE validation subset
    cp_dimers = selected.get("cp_dimers", pd.DataFrame())
    for i_c, (_, row) in enumerate(cp_dimers.iterrows(), start=1):
        cid = row["cluster_id"]
        print(f"  [ORCA wB97X-D3 CP {i_c}/{len(cp_dimers)}] {cid}")
        na = int(row["n_atoms_a"])
        atoms = _xyz_to_atom_tuples(Path(row["xyz_path"]))
        cdir = base_dir / "cp_wb97xd3" / cid
        inps = write_orca_cp_dimer_inputs(
            cdir,
            coords_a=atoms[:na],
            coords_b=atoms[na:],
            method="wB97X-D3 def2-TZVP def2/J RIJCOSX NoFinalGridX",
            nprocs=cfg.orca_nprocs,
            maxcore_mb=cfg.orca_maxcore_mb,
            scf_tol="NormalSCF",
        )
        e_cp = {}
        for tag, inp_p in inps.items():
            out_p, _ = _run_orca(inp_p, cfg.orca_bin, openmpi_dir=cfg.openmpi_dir)
            e_cp[tag] = parse_orca_sp_energy_eh(out_p.read_text(encoding="utf-8", errors="ignore"))
        de_raw = (e_cp["ab"] - e_cp["a_only"] - e_cp["b_only"]) * HARTREE_TO_KCAL
        de_cp = (e_cp["ab"] - e_cp["a_ghost_b"] - e_cp["b_ghost_a"]) * HARTREE_TO_KCAL
        bsse = de_cp - de_raw
        cp_rows.append(
            {
                "cluster_id": cid,
                "pair": row["pair"],
                "system_code": row["system_code"],
                "motif": row["motif"],
                "de_xtb_kcal": float(row["de_int_kcal"]),
                "de_wb97xd3_raw_kcal": float(de_raw),
                "de_wb97xd3_cp_kcal": float(de_cp),
                "bsse_kcal": float(bsse),
            }
        )

    # 3. r2SCAN-3c trimers (3-body cooperativity)
    dft_trimers = selected.get("dft_trimers", pd.DataFrame())
    for i_t, (_, row) in enumerate(dft_trimers.iterrows(), start=1):
        cid = row["cluster_id"]
        print(f"  [ORCA r2SCAN-3c trimer {i_t}/{len(dft_trimers)}] {cid}")
        na, nb = int(row["n_atoms_a"]), int(row["n_atoms_b"])
        atoms = _xyz_to_atom_tuples(Path(row["xyz_path"]))
        cA, cB, cC = atoms[:na], atoms[na : na + nb], atoms[na + nb :]
        cdir = base_dir / "trimers_r2scan3c" / cid
        e_t = {}
        for tag, sub in [
            ("ABC", atoms),
            ("AB", cA + cB),
            ("AC", cA + cC),
            ("BC", cB + cC),
            ("A", cA),
            ("B", cB),
            ("C", cC),
        ]:
            inp = write_orca_dft_input(
                cdir / f"{tag}.inp",
                sub,
                method="r2SCAN-3c",
                nprocs=cfg.orca_nprocs,
                maxcore_mb=cfg.orca_maxcore_mb,
                keep_gbw=(tag == "ABC"),
            )
            out_p, _ = _run_orca(inp, cfg.orca_bin, openmpi_dir=cfg.openmpi_dir)
            e_t[tag] = parse_orca_sp_energy_eh(out_p.read_text(encoding="utf-8", errors="ignore"))
        e_3body_dft = (e_t["ABC"] - e_t["AB"] - e_t["AC"] - e_t["BC"] + e_t["A"] + e_t["B"] + e_t["C"]) * HARTREE_TO_KCAL
        trimer_rows.append(
            {
                **row.to_dict(),
                "e_3body_r2scan3c_kcal": float(e_3body_dft),
                "gbw_path": str(cdir / "ABC.gbw"),
            }
        )

    # 4. r2SCAN-3c strain validation on conformer subset
    dft_strain = selected.get("dft_strain", pd.DataFrame())
    if not dft_strain.empty and not ref_conformers.empty:
        ref_orca_e: dict[str, float] = {}
        for _, rref in ref_conformers.iterrows():
            sol = rref["solute_resname"]
            print(f"  [ORCA r2SCAN-3c strain ref opt] {sol}")
            atoms_ref = _xyz_to_atom_tuples(Path(rref["best_ref_xyz"]))
            cdir = base_dir / "strain_r2scan3c" / f"{sol}_ref"
            inp = write_orca_dft_input(
                cdir / f"{sol}_ref.inp",
                atoms_ref,
                method="r2SCAN-3c",
                nprocs=cfg.orca_nprocs,
                maxcore_mb=cfg.orca_maxcore_mb,
                opt=True,
                keep_gbw=False,
            )
            out_p, _ = _run_orca(inp, cfg.orca_bin, openmpi_dir=cfg.openmpi_dir)
            ref_orca_e[sol] = parse_orca_sp_energy_eh(out_p.read_text(encoding="utf-8", errors="ignore"))

        for _, rs in dft_strain.iterrows():
            cid = rs["cluster_id"]
            sol = rs["solute_resname"]
            atoms_md = _xyz_to_atom_tuples(Path(rs["relaxed_xyz_path"]))
            cdir = base_dir / "strain_r2scan3c" / cid
            inp = write_orca_dft_input(
                cdir / f"{cid}.inp",
                atoms_md,
                method="r2SCAN-3c",
                nprocs=cfg.orca_nprocs,
                maxcore_mb=cfg.orca_maxcore_mb,
                opt=False,
                keep_gbw=False,
            )
            out_p, _ = _run_orca(inp, cfg.orca_bin, openmpi_dir=cfg.openmpi_dir)
            e_md = parse_orca_sp_energy_eh(out_p.read_text(encoding="utf-8", errors="ignore"))
            strain_rows.append(
                {
                    "cluster_id": cid,
                    "system_code": rs["system_code"],
                    "solute_resname": sol,
                    "e_strain_xtb_relaxed_kcal": float(rs["e_strain_relaxed_kcal"]),
                    "e_strain_r2scan3c_kcal": float((e_md - ref_orca_e[sol]) * HARTREE_TO_KCAL),
                }
            )

    return {
        "dft_dimers": pd.DataFrame(dimer_rows),
        "cp_validation": pd.DataFrame(cp_rows),
        "dft_trimers": pd.DataFrame(trimer_rows),
        "dft_strain": pd.DataFrame(strain_rows),
    }


def run_orca_led_eda(dft_dimers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Analysis #9: DLPNO-CCSD(T) LED with mandatory 1-dimer memory benchmark + fallback priority."""
    if dft_dimers.empty:
        return pd.DataFrame()
    led_dir = cfg.results_dir / "orca/led_eda"
    led_dir.mkdir(parents=True, exist_ok=True)

    # Pick strongest representative across every solute-solvent pair discovered,
    # then fill up to led_dimers_total across unique systems, ordered smallest-first so
    # the smallest complex acts as the initial RAM/runtime benchmark.
    picks = []
    seen_cids: set[str] = set()
    dft_work = dft_dimers.assign(
        _tot_atoms=dft_dimers["n_atoms_a"].astype(int) + dft_dimers["n_atoms_b"].astype(int)
    )
    pair_order = (
        dft_work.groupby("pair", as_index=False)["_tot_atoms"]
        .min()
        .sort_values(["_tot_atoms", "pair"])["pair"]
        .tolist()
    )
    for pair in pair_order:
        sub = dft_work.loc[dft_work["pair"] == pair].sort_values("de_int_r2scan3c_kcal")
        if not sub.empty:
            row0 = sub.iloc[0]
            picks.append(row0)
            seen_cids.add(str(row0["cluster_id"]))
    if len(picks) < cfg.led_dimers_total:
        for sys_code in sorted(dft_work["system_code"].dropna().unique().tolist()):
            sub = dft_work.loc[dft_work["system_code"] == sys_code].sort_values("de_int_r2scan3c_kcal")
            for _, r in sub.iterrows():
                if str(r["cluster_id"]) not in seen_cids and len(picks) < cfg.led_dimers_total:
                    picks.append(r)
                    seen_cids.add(str(r["cluster_id"]))
    cand_df = pd.DataFrame(picks)
    if cand_df.empty:
        return pd.DataFrame()
    cand_df["total_atoms"] = cand_df["n_atoms_a"].astype(int) + cand_df["n_atoms_b"].astype(int)
    cand_df = cand_df.sort_values("total_atoms").head(cfg.led_dimers_total).reset_index(drop=True)

    cur_nprocs = cfg.led_nprocs
    cur_basis = "def2-TZVP"
    cur_maxcore = cfg.led_maxcore_mb
    rows: list[dict[str, Any]] = []

    for idx, row in cand_df.iterrows():
        cid = row["cluster_id"]
        na = int(row["n_atoms_a"])
        atoms = _xyz_to_atom_tuples(Path(row["xyz_path"]))
        cdir = led_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)

        # Write external HPC def2-TZVP reference deck in all cases
        write_orca_led_input(cdir / f"{cid}_HPC_def2TZVP.inp", atoms[:na], atoms[na:], basis="def2-TZVP", nprocs=12, maxcore_mb=3000)

        ran_ok = False
        peak_rss = 0.0
        used_basis = cur_basis
        used_nprocs = cur_nprocs

        while not ran_ok:
            print(f"  [ORCA DLPNO-CCSD(T)/LED {idx + 1}/{len(cand_df)}] {cid} basis={used_basis} nprocs={used_nprocs}")
            ab_inp = write_orca_led_input(
                cdir / f"{cid}_AB_{used_basis}_p{used_nprocs}.inp",
                atoms[:na],
                atoms[na:],
                basis=used_basis,
                nprocs=used_nprocs,
                maxcore_mb=cur_maxcore,
            )
            try:
                out_ab, peak_rss = _run_orca(
                    ab_inp,
                    cfg.orca_bin,
                    monitor_rss=True,
                    abort_rss_gb=cfg.led_ram_limit_gb,
                    openmpi_dir=cfg.openmpi_dir,
                )
                if idx == 0 and peak_rss > cfg.led_ram_limit_gb:
                    nxt = choose_led_execution_settings(peak_rss, cfg.led_ram_limit_gb, used_nprocs, used_basis)
                    used_nprocs = nxt["nprocs"]
                    used_basis = nxt["basis"]
                    cur_nprocs, cur_basis = used_nprocs, used_basis
                ran_ok = True
            except Exception:
                nxt = choose_led_execution_settings(999.0, cfg.led_ram_limit_gb, used_nprocs, used_basis)
                if nxt["nprocs"] == used_nprocs and nxt["basis"] == used_basis:
                    raise
                used_nprocs = nxt["nprocs"]
                used_basis = nxt["basis"]
                cur_nprocs, cur_basis = used_nprocs, used_basis

        # Isolated monomer DLPNO-CCSD(T) energies at matched basis for E_prep and E_total
        # Compact monomers (<=30 atoms, e.g. THY/LIM/GER) start at nprocs<=2 to avoid ORCA 6.1.1 MPI SHARK bug
        aux_c = f"{used_basis}/C"
        e_iso = {}
        for tag, sub in [("A", atoms[:na]), ("B", atoms[na:])]:
            sub_start = min(used_nprocs, 2) if len(sub) <= 30 else used_nprocs
            monomer_procs = list(dict.fromkeys([p for p in (sub_start, 2, 1) if p <= used_nprocs]))
            last_err: Exception | None = None
            for m_np in monomer_procs:
                m_mc = max(cur_maxcore, 4000) if m_np == 1 else max(cur_maxcore, 3500) if m_np == 2 else cur_maxcore
                suffix = "" if m_np == used_nprocs else f"_p{m_np}"
                m_inp = write_orca_dft_input(
                    cdir / f"{cid}_{tag}_{used_basis}{suffix}.inp",
                    sub,
                    method=f"DLPNO-CCSD(T) {used_basis} {aux_c} def2/J RIJCOSX NormalPNO",
                    nprocs=m_np,
                    maxcore_mb=m_mc,
                    keep_gbw=False,
                )
                try:
                    m_out, _ = _run_orca(m_inp, cfg.orca_bin, openmpi_dir=cfg.openmpi_dir)
                    e_iso[tag] = parse_orca_sp_energy_eh(m_out.read_text(encoding="utf-8", errors="ignore"))
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    print(f"    [Retry monomer {cid}_{tag}] nprocs={m_np} failed ({exc}); falling back...")
            if last_err is not None:
                raise last_err

        led_parsed = parse_orca_led_output(out_ab.read_text(encoding="utf-8", errors="ignore"))
        e_total_int_kcal = (led_parsed["e_total_eh"] - e_iso["A"] - e_iso["B"]) * HARTREE_TO_KCAL
        e_prep_kcal = (
            (led_parsed["e_intra_sum_eh"] - e_iso["A"] - e_iso["B"]) * HARTREE_TO_KCAL
            if led_parsed["e_intra_sum_eh"] != 0.0
            else led_parsed["e_prep_kcal"]
        )
        rows.append(
            {
                "cluster_id": cid,
                "label": row["pair"],
                "system_code": row["system_code"],
                "motif": row["motif"],
                "basis": used_basis,
                "nprocs": used_nprocs,
                "peak_rss_gb": float(peak_rss),
                "e_elstat_kcal": led_parsed["e_elstat_kcal"],
                "e_exch_kcal": led_parsed["e_exch_kcal"],
                "e_disp_kcal": led_parsed["e_disp_kcal"],
                "e_non_disp_corr_kcal": led_parsed["e_non_disp_corr_kcal"],
                "e_prep_kcal": float(e_prep_kcal),
                "e_total_kcal": float(e_total_int_kcal),
                "de_dft_r2scan3c_kcal": float(row["de_int_r2scan3c_kcal"]),
            }
        )

    return pd.DataFrame(rows)
