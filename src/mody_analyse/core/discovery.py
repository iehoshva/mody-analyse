from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from mody_analyse.core.chemistry import SpeciesChemistry, infer_species_chemistry
from mody_analyse.core.config import Config


DEFAULT_ALIGN_ATOMS: dict[str, tuple[str, str, str]] = {}


@dataclass(frozen=True)
class MoleculeTopology:
    resname: str
    n_atoms: int
    atom_names: list[str]
    elements: list[str]
    charges: np.ndarray
    masses: np.ndarray
    bonds_1based: list[tuple[int, int]]
    donors: list[tuple[str, str]]
    acceptors: list[str]
    rotatable_dihedrals_1based: list[tuple[int, int, int, int]]
    align_atoms: tuple[str, str, str] | None
    center_atom_0based: int = 0


def compute_whole_coms_vectorized(
    positions_a: np.ndarray,
    n_mol: int,
    n_atoms: int,
    masses: np.ndarray,
    box_a: np.ndarray,
    anchor_idx: int = 0,
    atom_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute PBC-whole COMs for n_mol contiguous molecules of n_atoms in one vectorized pass."""
    pos = positions_a.reshape(n_mol, n_atoms, 3)
    ref = pos[:, anchor_idx : anchor_idx + 1, :]
    diff = pos - ref
    diff -= box_a[None, None, :] * np.round(diff / box_a[None, None, :])
    whole = ref + diff
    w = masses.astype(float).copy()
    if atom_mask is not None:
        w = w * atom_mask.astype(float)
    w_sum = float(np.sum(w))
    w = w / (w_sum if w_sum > 0 else float(len(w)))
    return (whole * w[None, :, None]).sum(axis=1)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    system_code: str
    replica: str
    solute_resname: str | None
    path: Path
    metadata: dict[str, Any]
    molecule_counts: dict[str, int]
    topologies: dict[str, MoleculeTopology]
    tpr_path: Path
    xtc_path: Path
    edr_path: Path
    log_path: Path
    thermo_path: Path
    stress_path: Path | None
    rdf_paths: dict[str, Path]
    block_rdf_paths: list[Path]
    solvent_resnames: tuple[str, ...] = ()
    chemistry: dict[str, SpeciesChemistry] = field(default_factory=dict)


def _guess_element(mass: float, atom_name: str) -> str:
    if abs(mass - 1.008) < 0.2:
        return "H"
    if abs(mass - 12.011) < 0.5:
        return "C"
    if abs(mass - 14.007) < 0.5:
        return "N"
    if abs(mass - 15.999) < 0.5:
        return "O"
    if abs(mass - 30.974) < 0.5:
        return "P"
    if abs(mass - 32.06) < 0.5:
        return "S"
    if abs(mass - 18.998) < 0.5:
        return "F"
    if abs(mass - 35.45) < 0.8:
        return "CL"
    if abs(mass - 79.904) < 0.8:
        return "BR"
    m = re.match(r"([A-Za-z]+)", atom_name)
    return m.group(1)[0].upper() if m else "X"


def _find_rotatable_heavy_dihedrals(
    n_atoms: int, elements: list[str], bonds_1based: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    g = nx.Graph()
    for i in range(1, n_atoms + 1):
        g.add_node(i, elem=elements[i - 1])
    for u, v in bonds_1based:
        g.add_edge(u, v)

    heavy_graph = g.subgraph([n for n, d in g.nodes(data=True) if d["elem"] != "H"]).copy()
    cycle_edges: set[tuple[int, int]] = set()
    for cycle in nx.cycle_basis(heavy_graph):
        for idx in range(len(cycle)):
            a, b = cycle[idx], cycle[(idx + 1) % len(cycle)]
            cycle_edges.add((min(a, b), max(a, b)))

    dihedrals: list[tuple[int, int, int, int]] = []
    for u, v in sorted(heavy_graph.edges()):
        if (min(u, v), max(u, v)) in cycle_edges:
            continue
        u_nbrs = sorted(n for n in heavy_graph.neighbors(u) if n != v)
        v_nbrs = sorted(n for n in heavy_graph.neighbors(v) if n != u)
        if not u_nbrs or not v_nbrs:
            continue
        dihedrals.append((u_nbrs[0], u, v, v_nbrs[0]))
    return dihedrals


def _build_molecule_topology(
    resname: str,
    atom_names: list[str],
    elements: list[str],
    charges: list[float],
    masses: list[float],
    bonds: list[tuple[int, int]],
    align_override: tuple[str, str, str] | None = None,
) -> MoleculeTopology:
    donors: list[tuple[str, str]] = []
    acceptors: list[str] = []
    for idx, elem in enumerate(elements, start=1):
        if elem in {"O", "N"}:
            acceptors.append(atom_names[idx - 1])
            for a, b in bonds:
                other = b if a == idx else (a if b == idx else None)
                if other is not None and elements[other - 1] == "H":
                    donors.append((atom_names[idx - 1], atom_names[other - 1]))

    rot_dih = _find_rotatable_heavy_dihedrals(len(atom_names), elements, bonds)
    g_heavy = nx.Graph()
    for i, e in enumerate(elements, start=1):
        if e != "H":
            g_heavy.add_node(i)
    for a, b in bonds:
        if elements[a - 1] != "H" and elements[b - 1] != "H":
            g_heavy.add_edge(a, b)
    center_0 = (nx.center(g_heavy)[0] - 1) if g_heavy.number_of_nodes() > 0 and nx.is_connected(g_heavy) else 0

    align = align_override or DEFAULT_ALIGN_ATOMS.get(resname)
    if align is None and len(atom_names) >= 3:
        heavy = [a for a, e in zip(atom_names, elements) if e != "H"]
        if len(heavy) >= 3:
            align = (heavy[0], heavy[1], heavy[2])
        else:
            align = (atom_names[0], atom_names[1], atom_names[2])

    return MoleculeTopology(
        resname=resname,
        n_atoms=len(atom_names),
        atom_names=atom_names,
        elements=elements,
        charges=np.asarray(charges, dtype=float),
        masses=np.asarray(masses, dtype=float),
        bonds_1based=bonds,
        donors=donors,
        acceptors=acceptors,
        rotatable_dihedrals_1based=rot_dih,
        align_atoms=align,
        center_atom_0based=center_0,
    )


def parse_itp_topologies(itp_path: Path) -> list[MoleculeTopology]:
    """Parse one or more [ moleculetype ] blocks from an .itp or topol.top file."""
    results: list[MoleculeTopology] = []
    section = None
    cur_resname: str | None = None
    atom_names: list[str] = []
    elements: list[str] = []
    charges: list[float] = []
    masses: list[float] = []
    bonds: list[tuple[int, int]] = []

    def _flush() -> None:
        if cur_resname is not None and atom_names:
            results.append(_build_molecule_topology(cur_resname, atom_names, elements, charges, masses, bonds))

    with itp_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.split(";")[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[] ").lower()
                if section == "moleculetype":
                    _flush()
                    cur_resname = None
                    atom_names, elements, charges, masses, bonds = [], [], [], [], []
                continue
            parts = line.split()
            if section == "moleculetype" and len(parts) >= 1:
                cur_resname = parts[0]
            elif section == "atoms" and len(parts) >= 8 and cur_resname is not None:
                aname = parts[4]
                q = float(parts[6])
                m = float(parts[7])
                atom_names.append(aname)
                charges.append(q)
                masses.append(m)
                elements.append(_guess_element(m, aname))
            elif section == "bonds" and len(parts) >= 2 and cur_resname is not None:
                bonds.append((int(parts[0]), int(parts[1])))

    _flush()
    return results


def parse_itp_topology(itp_path: Path) -> MoleculeTopology:
    topos = parse_itp_topologies(itp_path)
    if topos:
        return topos[0]
    return _build_molecule_topology(itp_path.stem.upper()[:3], [], [], [], [], [])


def _parse_topol(topol_path: Path) -> tuple[list[Path], dict[str, int]]:
    includes: list[Path] = []
    counts: dict[str, int] = {}
    section = None
    with topol_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            s = raw_line.strip()
            if s.startswith("#include"):
                m = re.search(r'"([^"]+)"', s)
                if m and m.group(1).endswith(".itp") and "forcefield" not in m.group(1):
                    cand = (topol_path.parent / m.group(1)).resolve()
                    if cand.exists():
                        includes.append(cand)
                continue
            clean = raw_line.split(";")[0].strip()
            if not clean:
                continue
            if clean.startswith("[") and clean.endswith("]"):
                section = clean.strip("[] ").lower()
                continue
            if section == "molecules":
                parts = clean.split()
                if len(parts) >= 2:
                    counts[parts[0]] = counts.get(parts[0], 0) + int(parts[1])
    return includes, counts


def equilibrated_frame_slice(
    u: Any,
    t_eq_ns: float,
    t_end_ns: float,
    stride: int = 1,
) -> tuple[int, int, int]:
    """Return safe [start, stop, stride] excluding truncated EOF frames."""
    n_total = len(u.trajectory)
    stride_out = max(1, int(stride))
    if n_total <= 0:
        return 0, 0, stride_out

    last_valid = n_total - 1
    last_t_ps = 0.0
    while last_valid >= 0:
        try:
            last_t_ps = float(u.trajectory[last_valid].time)
            break
        except OSError:
            last_valid -= 1
    if last_valid < 0:
        return 0, 0, stride_out

    t_eq_ps = t_eq_ns * 1000.0
    t_end_ps = t_end_ns * 1000.0
    dt_ps = float(getattr(u.trajectory, "dt", 0.0) or 0.0)

    if dt_ps > 0.0:
        start_idx = max(0, min(last_valid, int(round(t_eq_ps / dt_ps))))
        while start_idx > 0 and float(u.trajectory[start_idx - 1].time) >= t_eq_ps:
            start_idx -= 1
        while start_idx < last_valid and float(u.trajectory[start_idx].time) < t_eq_ps:
            start_idx += 1
    else:
        start_idx = 0
        for idx in range(last_valid + 1):
            if float(u.trajectory[idx].time) >= t_eq_ps:
                start_idx = idx
                break

    if last_t_ps <= t_end_ps + 1e-3:
        stop_idx = last_valid + 1
    elif dt_ps > 0.0:
        end_idx = max(start_idx, min(last_valid, int(round(t_end_ps / dt_ps))))
        while end_idx < last_valid and float(u.trajectory[end_idx + 1].time) <= t_end_ps + 1e-3:
            end_idx += 1
        while end_idx > start_idx and float(u.trajectory[end_idx].time) > t_end_ps + 1e-3:
            end_idx -= 1
        stop_idx = end_idx + 1
    else:
        stop_idx = start_idx + 1
        for idx in range(last_valid, start_idx - 1, -1):
            if float(u.trajectory[idx].time) <= t_end_ps + 1e-3:
                stop_idx = idx + 1
                break

    return start_idx, stop_idx, stride_out


def read_xvg(path: Path, max_rows: int | None = None) -> tuple[list[str], np.ndarray]:
    legends: list[str] = []
    rows: list[list[float]] = []
    arr = np.empty((0, 0), dtype=float)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            if line.startswith("@"):
                m = re.match(r'@\s+s\d+\s+legend\s+"([^"]+)"', line.strip())
                if m:
                    legends.append(m.group(1))
                continue
            if max_rows is not None and max_rows <= 0:
                break
            s = line.strip()
            if not s:
                continue
            if max_rows is None:
                first = np.fromstring(s.replace("\t", " "), sep=" ", dtype=float)
                rest = np.loadtxt(fh, comments=("#", "@"), dtype=float)
                if rest.size == 0:
                    arr = first[None, :]
                elif rest.ndim == 1:
                    arr = np.vstack([first, rest[None, :]])
                else:
                    arr = np.vstack([first, rest])
                break
            rows.append([float(tok) for tok in s.split()])
            if len(rows) >= max_rows:
                break
    if arr.size == 0 and rows:
        arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        return ["Time"] + legends, arr
    n_cols = arr.shape[1]
    if len(legends) == n_cols - 1:
        cols = ["Time"] + legends
    else:
        cols = ["Time"] + [f"col_{i}" for i in range(1, n_cols)]
    return cols, arr


def _find_first_existing(run_dir: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        if not rel:
            continue
        p = run_dir / rel
        if p.exists():
            return p
    return None


def _infer_code_and_replica(run_dir_name: str, meta: dict[str, Any], cfg: Config, solute_res: str | None) -> tuple[str, str]:
    sys_kind = str(meta.get("systemKind", "")).lower()
    if sys_kind in {"pure-des", "pure"}:
        system_code = cfg.pure_system_code
    elif meta.get("soluteAbbreviation"):
        system_code = str(meta["soluteAbbreviation"]).upper()
    elif meta.get("systemCode"):
        system_code = str(meta["systemCode"]).upper()
    else:
        m_rep = re.search(r"(?:^|_|-)[Rr]0*(\d+)(?:_|-|$)", run_dir_name)
        cleaned = re.sub(r"(?:^|_|-)[Rr]0*\d+(?:_|-|$)", "_", run_dir_name).strip("_")
        if solute_res:
            system_code = solute_res.upper()
        elif "PURE" in run_dir_name.upper():
            system_code = cfg.pure_system_code
        else:
            system_code = cleaned or run_dir_name
            if not m_rep:
                system_code = system_code[:16]

    if "replica" in meta and meta["replica"] is not None:
        replica = f"R{int(meta['replica']):02d}"
    else:
        m_r = re.search(r"(?:^|_|-)[Rr](?:ep)?0*(\d+)(?:_|-|$)", run_dir_name)
        replica = f"R{int(m_r.group(1)):02d}" if m_r else "R01"
    return system_code, replica


def _classify_species(
    counts: dict[str, int],
    meta: dict[str, Any],
    cfg: Config,
) -> tuple[tuple[str, ...], str | None]:
    species = list(counts.keys())
    if not species:
        return (), None

    sys_kind = str(meta.get("systemKind", "")).lower()
    # 1. Explicit configuration wins
    if cfg.solvents:
        solvs = tuple(s for s in cfg.solvents if s in counts)
        if not solvs:
            solvs = tuple(species)
        sol_cands = [s for s in species if s not in solvs]
        if cfg.solutes:
            explicit = [s for s in sol_cands if s in cfg.solutes]
            sol_cands = explicit or sol_cands
        if sys_kind in {"pure-des", "pure"}:
            return solvs, None
        return solvs, (sol_cands[0] if sol_cands else None)

    # 2. Metadata explicit pure run
    if sys_kind in {"pure-des", "pure"} or len(species) == 1:
        return tuple(species), None

    # 3. Auto-detect minority solute vs majority solvent(s)
    if cfg.solutes:
        sol_cands = [s for s in species if s in cfg.solutes]
        if sol_cands:
            sol = sol_cands[0]
            return tuple(s for s in species if s != sol), sol

    max_cnt = max(counts.values())
    minorities = [s for s, n in counts.items() if n < max_cnt and n <= 0.15 * sum(counts.values())]
    if len(minorities) == 1:
        sol = minorities[0]
        return tuple(s for s in species if s != sol), sol

    if meta.get("soluteResname") and str(meta["soluteResname"]) in counts:
        sol = str(meta["soluteResname"])
        return tuple(s for s in species if s != sol), sol

    # Equal-count multi-component liquid mixture (no minor solute)
    return tuple(species), None


def discover_runs(cfg: Config) -> list[RunRecord]:
    runs: list[RunRecord] = []
    run_glob = getattr(cfg, "run_glob", "*")
    if not cfg.project_root.exists():
        return runs

    path_hints = cfg.paths
    for run_dir in sorted(cfg.project_root.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "metadata.json"
        topol_path = _find_first_existing(
            run_dir,
            [path_hints.get("topol_relpath", ""), "00_input/topol.top", "topol.top", "toppar/topol.top"],
        )
        tpr_path = _find_first_existing(
            run_dir,
            [path_hints.get("tpr_relpath", ""), "04_prod/prod.tpr", "prod.tpr", "md.tpr", "npt.tpr"],
        )
        xtc_path = _find_first_existing(
            run_dir,
            [path_hints.get("xtc_relpath", ""), "04_prod/prod.xtc", "prod.xtc", "md.xtc", "traj_comp.xtc", "traj.xtc"],
        )
        thermo_path = _find_first_existing(
            run_dir,
            [path_hints.get("thermo_relpath", ""), "analysis/thermo.xvg", "thermo.xvg", "energy.xvg"],
        )
        if not (topol_path and tpr_path and xtc_path and thermo_path and thermo_path.stat().st_size > 0):
            continue

        edr_path = _find_first_existing(run_dir, ["04_prod/prod.edr", "prod.edr", "md.edr"]) or (
            run_dir / "04_prod/prod.edr"
        )
        log_path = _find_first_existing(run_dir, ["04_prod/prod.log", "prod.log", "md.log"]) or (
            run_dir / "04_prod/prod.log"
        )
        rdf_dir = _find_first_existing(run_dir, [path_hints.get("rdf_dir_relpath", ""), "analysis/rdf", "rdf"])
        rdf_paths = {p.stem: p for p in sorted(rdf_dir.glob("rdf_*.xvg"))} if rdf_dir and rdf_dir.is_dir() else {}

        meta: dict[str, Any] = {}
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)

        includes, counts = _parse_topol(topol_path)
        topologies: dict[str, MoleculeTopology] = {}
        for inline_topo in parse_itp_topologies(topol_path):
            topologies[inline_topo.resname] = inline_topo
        for itp in includes:
            for topo in parse_itp_topologies(itp):
                topologies[topo.resname] = topo

        solvent_resnames, solute_resname = _classify_species(counts, meta, cfg)
        system_code, replica = _infer_code_and_replica(run_dir.name, meta, cfg, solute_resname)

        chemistry: dict[str, SpeciesChemistry] = {}
        updated_topos: dict[str, MoleculeTopology] = {}
        for resname, topo in topologies.items():
            ov = cfg.species_properties.get(resname)
            chem = infer_species_chemistry(topo, override=ov)
            chemistry[resname] = chem
            if chem.align_atoms != topo.align_atoms:
                updated_topos[resname] = MoleculeTopology(
                    resname=topo.resname,
                    n_atoms=topo.n_atoms,
                    atom_names=topo.atom_names,
                    elements=topo.elements,
                    charges=topo.charges,
                    masses=topo.masses,
                    bonds_1based=topo.bonds_1based,
                    donors=topo.donors,
                    acceptors=topo.acceptors,
                    rotatable_dihedrals_1based=topo.rotatable_dihedrals_1based,
                    align_atoms=chem.align_atoms,
                    center_atom_0based=topo.center_atom_0based,
                )
            else:
                updated_topos[resname] = topo

        stress_path = _find_first_existing(
            run_dir, [path_hints.get("stress_relpath", ""), "analysis/stress.xvg", "stress.xvg"]
        )
        blocks_dir = _find_first_existing(run_dir, ["analysis/blocks", "blocks"])
        block_rdf_paths = sorted(blocks_dir.glob("rdf_*.xvg")) if blocks_dir and blocks_dir.is_dir() else []

        runs.append(
            RunRecord(
                run_id=run_dir.name,
                system_code=system_code,
                replica=replica,
                solute_resname=solute_resname,
                path=run_dir,
                metadata=meta,
                molecule_counts=counts,
                topologies=updated_topos,
                tpr_path=tpr_path,
                xtc_path=xtc_path,
                edr_path=edr_path,
                log_path=log_path,
                thermo_path=thermo_path,
                stress_path=stress_path,
                rdf_paths=rdf_paths,
                block_rdf_paths=block_rdf_paths,
                solvent_resnames=solvent_resnames,
                chemistry=chemistry,
            )
        )
    return runs
