from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from mody_analyse.md_to_qm.cluster_extractor import write_xyz
from mody_analyse.core.chemistry import topology_to_rdkit_mol
from mody_analyse.core.config import Config
from mody_analyse.core.discovery import MoleculeTopology


HARTREE_TO_KCAL = 627.5094740631


def parse_xtb_energy_eh(output_text: str) -> float:
    matches = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", output_text)
    if not matches:
        raise ValueError("Could not find TOTAL ENERGY in xtb output")
    return float(matches[-1])


def build_dihedral_constrain_file(
    out_path: Path,
    dihedrals_1based: list[tuple[int, int, int, int]],
    force_constant: float = 1.0,
) -> Path:
    lines = ["$constrain", f"  force constant={force_constant:.3f}"]
    for a, b, c, d in dihedrals_1based:
        lines.append(f"  dihedral: {a},{b},{c},{d},auto")
    lines.append("$end")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def compute_three_body_energy_kcal(
    e_abc: float,
    e_ab: float,
    e_ac: float,
    e_bc: float,
    e_a: float,
    e_b: float,
    e_c: float,
) -> float:
    return float((e_abc - e_ab - e_ac - e_bc + e_a + e_b + e_c) * HARTREE_TO_KCAL)


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    n = int(lines[0].strip())
    elems, coords = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elems, np.asarray(coords, dtype=float)


def _run_xtb_job(
    xyz_path: Path,
    xtb_bin: Path,
    opt: bool = False,
    opt_level: str = "normal",
    input_file: Path | None = None,
) -> tuple[float, Path]:
    workdir = xyz_path.parent
    out_file = workdir / f"{xyz_path.stem}{'_opt' if opt else '_sp'}.out"
    opt_xyz = workdir / f"{xyz_path.stem}_xtbopt.xyz"

    fresh_input = input_file is None or out_file.stat().st_mtime >= input_file.stat().st_mtime if out_file.exists() else False
    if out_file.exists() and out_file.stat().st_mtime >= xyz_path.stat().st_mtime and fresh_input:
        try:
            e = parse_xtb_energy_eh(out_file.read_text(encoding="utf-8"))
            if not opt or opt_xyz.exists():
                return e, (opt_xyz if opt else xyz_path)
        except Exception:
            pass

    cmd = [str(xtb_bin), xyz_path.name, "--gfn", "2", "--chrg", "0", "--uhf", "0"]
    if opt:
        cmd.extend(["--opt", opt_level])
    if input_file is not None:
        cmd.extend(["--input", input_file.name])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    # Run in isolated temp prefix inside workdir to avoid xtbopt.xyz collision
    job_dir = workdir / f"_run_{xyz_path.stem}_{'opt' if opt else 'sp'}"
    job_dir.mkdir(parents=True, exist_ok=True)
    local_xyz = job_dir / xyz_path.name
    local_xyz.write_text(xyz_path.read_text(encoding="utf-8"), encoding="utf-8")
    if input_file is not None:
        (job_dir / input_file.name).write_text(input_file.read_text(encoding="utf-8"), encoding="utf-8")

    proc = subprocess.run(cmd, cwd=str(job_dir), env=env, capture_output=True, text=True, check=False)
    out_text = proc.stdout + "\n" + proc.stderr
    out_file.write_text(out_text, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"xtb failed for {xyz_path}: rc={proc.returncode}")
    e_eh = parse_xtb_energy_eh(out_text)
    if opt:
        produced = job_dir / "xtbopt.xyz"
        if produced.exists():
            opt_xyz.write_text(produced.read_text(encoding="utf-8"), encoding="utf-8")
        return e_eh, opt_xyz
    return e_eh, xyz_path


def _topology_to_rdkit_mol(topo: MoleculeTopology, sample_coords: np.ndarray) -> Chem.Mol:
    return topology_to_rdkit_mol(topo, sample_coords)


def generate_xtb_reference_conformers(
    topologies: dict[str, MoleculeTopology],
    strain_manifest: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if strain_manifest.empty:
        return pd.DataFrame()
    ref_dir = cfg.results_dir / "xtb/reference_conformers"
    ref_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for sol, grp in strain_manifest.groupby("solute_resname"):
        topo = topologies[sol]
        sample_xyz = Path(grp.iloc[0]["xyz_path"])
        _, sample_coords = _read_xyz(sample_xyz)
        mol = _topology_to_rdkit_mol(topo, sample_coords)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=cfg.strain_ref_conformers, params=params))
        if not cids:
            # Fallback: include sample MD coords directly if RDKit embedding struggled
            conf_xyzs = [sample_xyz]
        else:
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
            conf_xyzs = []
            for ci in cids:
                conf = mol.GetConformer(ci)
                c_pos = np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(topo.n_atoms)])
                p = ref_dir / f"{sol}_conf_{ci:03d}.xyz"
                write_xyz(p, topo.elements, c_pos, comment=f"{sol} ref conf {ci}")
                conf_xyzs.append(p)
            # Also include an unconstrained opt of MD snapshot #0 to guarantee global min coverage
            p0 = ref_dir / f"{sol}_conf_md0.xyz"
            write_xyz(p0, topo.elements, sample_coords, comment=f"{sol} md0 unconstrained")
            conf_xyzs.append(p0)

        def _opt_one(p: Path):
            e, opt_p = _run_xtb_job(p, cfg.xtb_bin, opt=True, opt_level="tight")
            return p, e, opt_p

        with ThreadPoolExecutor(max_workers=cfg.xtb_workers) as pool:
            res = list(pool.map(_opt_one, conf_xyzs))
        best_p, best_e, best_opt_p = min(res, key=lambda x: x[1])
        rows.append(
            {
                "solute_resname": sol,
                "n_conformers_tested": len(res),
                "e_ref_eh": float(best_e),
                "best_ref_xyz": str(best_opt_p),
            }
        )
    return pd.DataFrame(rows)


def run_xtb_strain(
    strain_manifest: pd.DataFrame,
    ref_df: pd.DataFrame,
    topologies: dict[str, MoleculeTopology],
    cfg: Config,
) -> pd.DataFrame:
    """Analysis #5: Dihedral-constrained GFN2-xTB strain (primary) + frozen SP (diagnostic)."""
    if strain_manifest.empty or ref_df.empty:
        return pd.DataFrame()
    work_dir = cfg.results_dir / "xtb/strain"
    work_dir.mkdir(parents=True, exist_ok=True)
    ref_map = ref_df.set_index("solute_resname")["e_ref_eh"].to_dict()

    def _eval_row(row_dict: dict[str, Any]) -> dict[str, Any]:
        cid = row_dict["cluster_id"]
        sol = row_dict["solute_resname"]
        topo = topologies[sol]
        src_xyz = Path(row_dict["xyz_path"])
        cdir = work_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        local_xyz = cdir / f"{cid}.xyz"
        local_xyz.write_text(src_xyz.read_text(encoding="utf-8"), encoding="utf-8")

        e_frozen, _ = _run_xtb_job(local_xyz, cfg.xtb_bin, opt=False)
        if topo.rotatable_dihedrals_1based:
            c_inp = build_dihedral_constrain_file(cdir / "constrain.inp", topo.rotatable_dihedrals_1based)
            e_relaxed, opt_xyz = _run_xtb_job(local_xyz, cfg.xtb_bin, opt=True, opt_level="normal", input_file=c_inp)
        else:
            e_relaxed, opt_xyz = _run_xtb_job(local_xyz, cfg.xtb_bin, opt=True, opt_level="normal")

        e_ref = ref_map[sol]
        return {
            **row_dict,
            "e_ref_eh": float(e_ref),
            "e_frozen_eh": float(e_frozen),
            "e_relaxed_eh": float(e_relaxed),
            "e_strain_frozen_kcal": float((e_frozen - e_ref) * HARTREE_TO_KCAL),
            "e_strain_relaxed_kcal": float((e_relaxed - e_ref) * HARTREE_TO_KCAL),
            "relaxed_xyz_path": str(opt_xyz),
        }

    records = strain_manifest.to_dict("records")
    with ThreadPoolExecutor(max_workers=cfg.xtb_workers) as pool:
        out_rows = list(pool.map(_eval_row, records))
    return pd.DataFrame(out_rows)


def run_xtb_dimers(dimer_manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Analysis #6: GFN2-xTB dimer interaction energies at exact complex fragment geometries."""
    if dimer_manifest.empty:
        return pd.DataFrame()
    work_dir = cfg.results_dir / "xtb/dimers"
    work_dir.mkdir(parents=True, exist_ok=True)

    def _eval_dimer(row_dict: dict[str, Any]) -> dict[str, Any]:
        cid = row_dict["cluster_id"]
        na = int(row_dict["n_atoms_a"])
        elems, coords = _read_xyz(Path(row_dict["xyz_path"]))
        cdir = work_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        p_ab = cdir / f"{cid}_AB.xyz"
        p_a = cdir / f"{cid}_A.xyz"
        p_b = cdir / f"{cid}_B.xyz"
        write_xyz(p_ab, elems, coords, comment=f"{cid} AB")
        write_xyz(p_a, elems[:na], coords[:na], comment=f"{cid} A")
        write_xyz(p_b, elems[na:], coords[na:], comment=f"{cid} B")
        e_ab, _ = _run_xtb_job(p_ab, cfg.xtb_bin, opt=False)
        e_a, _ = _run_xtb_job(p_a, cfg.xtb_bin, opt=False)
        e_b, _ = _run_xtb_job(p_b, cfg.xtb_bin, opt=False)
        de_kcal = (e_ab - e_a - e_b) * HARTREE_TO_KCAL
        return {
            **row_dict,
            "e_ab_eh": float(e_ab),
            "e_a_eh": float(e_a),
            "e_b_eh": float(e_b),
            "de_int_kcal": float(de_kcal),
        }

    records = dimer_manifest.to_dict("records")
    with ThreadPoolExecutor(max_workers=cfg.xtb_workers) as pool:
        out_rows = list(pool.map(_eval_dimer, records))
    return pd.DataFrame(out_rows)


def run_xtb_trimers(trimer_manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Analysis #7: GFN2-xTB 3-body non-additive cooperativity energies."""
    if trimer_manifest.empty:
        return pd.DataFrame()
    work_dir = cfg.results_dir / "xtb/trimers"
    work_dir.mkdir(parents=True, exist_ok=True)

    def _eval_trimer(row_dict: dict[str, Any]) -> dict[str, Any]:
        cid = row_dict["cluster_id"]
        na = int(row_dict["n_atoms_a"])
        nb = int(row_dict["n_atoms_b"])
        elems, coords = _read_xyz(Path(row_dict["xyz_path"]))
        eA, cA = elems[:na], coords[:na]
        eB, cB = elems[na : na + nb], coords[na : na + nb]
        eC, cC = elems[na + nb :], coords[na + nb :]

        cdir = work_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        files = {
            "ABC": (elems, coords),
            "AB": (eA + eB, np.vstack([cA, cB])),
            "AC": (eA + eC, np.vstack([cA, cC])),
            "BC": (eB + eC, np.vstack([cB, cC])),
            "A": (eA, cA),
            "B": (eB, cB),
            "C": (eC, cC),
        }
        energies = {}
        for tag, (el, co) in files.items():
            p = cdir / f"{cid}_{tag}.xyz"
            write_xyz(p, el, co, comment=f"{cid} {tag}")
            energies[tag], _ = _run_xtb_job(p, cfg.xtb_bin, opt=False)

        e_3body = compute_three_body_energy_kcal(
            energies["ABC"],
            energies["AB"],
            energies["AC"],
            energies["BC"],
            energies["A"],
            energies["B"],
            energies["C"],
        )
        de_total = (energies["ABC"] - energies["A"] - energies["B"] - energies["C"]) * HARTREE_TO_KCAL
        de_2body_sum = (
            (energies["AB"] - energies["A"] - energies["B"])
            + (energies["AC"] - energies["A"] - energies["C"])
            + (energies["BC"] - energies["B"] - energies["C"])
        ) * HARTREE_TO_KCAL
        return {
            **row_dict,
            "e_3body_kcal": float(e_3body),
            "de_2body_sum_kcal": float(de_2body_sum),
            "de_total_kcal": float(de_total),
        }

    records = trimer_manifest.to_dict("records")
    with ThreadPoolExecutor(max_workers=cfg.xtb_workers) as pool:
        out_rows = list(pool.map(_eval_trimer, records))
    return pd.DataFrame(out_rows)
