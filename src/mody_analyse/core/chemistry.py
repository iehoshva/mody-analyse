from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from rdkit import Chem

if TYPE_CHECKING:
    from mody_analyse.core.discovery import MoleculeTopology


AVOGADRO = 6.02214076e23

# Bondi van der Waals radii (Angstrom) for fast deterministic molecular volume / r_H estimation
_VDW_RADII_A: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "B": 1.92,
    "SI": 2.10,
}


@dataclass(frozen=True)
class SpeciesChemistry:
    """Physicochemical and atomistic site descriptors inferred from topology + optional overrides."""

    resname: str
    mw_g_mol: float
    density_gcm3: float | None
    molar_volume_cm3_mol: float
    rh_nm: float
    polar_atoms: tuple[str, ...]
    polar_h_atoms: tuple[str, ...]
    ring_atoms: tuple[str, ...]
    tail_atoms: tuple[str, ...]
    align_atoms: tuple[str, str, str] | None
    sol_axis_index: int = 0


def topology_to_rdkit_mol(
    topo: MoleculeTopology,
    sample_coords: np.ndarray | None = None,
) -> Chem.Mol:
    """Build an RDKit Mol with deterministic bond orders inferred from .itp topology connectivity."""
    g = nx.Graph()
    for idx_1based in range(1, topo.n_atoms + 1):
        g.add_node(idx_1based, elem=topo.elements[idx_1based - 1])
    for a, b in topo.bonds_1based:
        g.add_edge(a, b)

    sp2_c = {n for n in g.nodes if topo.elements[n - 1] == "C" and g.degree(n) == 3}
    aromatic_edges: set[tuple[int, int]] = set()
    if len(sp2_c) >= 6:
        sub_c = g.subgraph(sp2_c)
        for cyc in nx.cycle_basis(sub_c):
            if len(cyc) == 6:
                for i in range(6):
                    u, v = cyc[i], cyc[(i + 1) % 6]
                    aromatic_edges.add((min(u, v), max(u, v)))

    rw = Chem.RWMol()
    for el in topo.elements:
        atom_sym = el.capitalize() if len(el) > 1 and el.upper() not in {"CL", "BR", "SI"} else (
            "Cl" if el.upper() == "CL" else ("Br" if el.upper() == "BR" else ("Si" if el.upper() == "SI" else el.upper()))
        )
        rw.AddAtom(Chem.Atom(atom_sym))

    coords_arr = np.asarray(sample_coords, dtype=float) if sample_coords is not None else None
    for a, b in topo.bonds_1based:
        ea, eb = topo.elements[a - 1].upper(), topo.elements[b - 1].upper()
        pair = (min(a, b), max(a, b))
        btype = Chem.BondType.SINGLE
        if pair in aromatic_edges:
            btype = Chem.BondType.AROMATIC
        elif (ea == "O" and g.degree(a) == 1 and eb == "C" and b in sp2_c) or (
            eb == "O" and g.degree(b) == 1 and ea == "C" and a in sp2_c
        ):
            btype = Chem.BondType.DOUBLE
        elif (ea == "N" and g.degree(a) == 1 and eb == "C" and g.degree(b) == 2) or (
            eb == "N" and g.degree(b) == 1 and ea == "C" and g.degree(a) == 2
        ):
            btype = Chem.BondType.TRIPLE
        elif a in sp2_c and b in sp2_c:
            btype = Chem.BondType.DOUBLE
        elif coords_arr is not None and len(topo.bonds_1based) > 0:
            d_a = float(np.linalg.norm(coords_arr[a - 1] - coords_arr[b - 1]))
            if ea == "C" and eb == "C" and d_a < 1.38 and (g.degree(a) < 4 and g.degree(b) < 4):
                btype = Chem.BondType.DOUBLE
        rw.AddBond(a - 1, b - 1, btype)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol = rw.GetMol()
        for b in mol.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            mol.UpdatePropertyCache(strict=False)

    if coords_arr is not None and len(coords_arr) == topo.n_atoms:
        conf = Chem.Conformer(topo.n_atoms)
        for i, (x, y, z) in enumerate(coords_arr):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
    return mol


def _estimate_vdw_volume_nm3(elements: list[str], n_bonds: int) -> float:
    """Fast Bondi sphere-sum estimate corrected for covalent bond overlap (~0.62 effective packing)."""
    v_spheres_a3 = sum((4.0 / 3.0) * math.pi * (_VDW_RADII_A.get(el.upper(), 1.70) ** 3) for el in elements)
    overlap_factor = max(0.55, 0.72 - 0.004 * max(n_bonds - 1, 0))
    v_eff_a3 = max(v_spheres_a3 * overlap_factor, 10.0)
    return float(v_eff_a3 * 1e-3)


def _auto_align_atoms(topo: MoleculeTopology, g: nx.Graph) -> tuple[str, str, str] | None:
    if topo.n_atoms < 3:
        return None
    heavy = [i for i in range(1, topo.n_atoms + 1) if topo.elements[i - 1].upper() != "H"]
    if len(heavy) < 3:
        return (topo.atom_names[0], topo.atom_names[1], topo.atom_names[2])

    c0 = topo.center_atom_0based + 1
    if c0 not in heavy:
        c0 = heavy[0]
    heavy_nbrs = sorted(n for n in g.neighbors(c0) if topo.elements[n - 1].upper() != "H")
    a1 = heavy_nbrs[0] if heavy_nbrs else next(n for n in heavy if n != c0)

    # Prefer a non-neighbor heavy atom attached to a1 or next neighbor for good non-collinear span
    candidates = [n for n in g.neighbors(a1) if n not in {c0, a1} and topo.elements[n - 1].upper() != "H"]
    if not candidates:
        candidates = [n for n in heavy if n not in {c0, a1}]
    a2 = candidates[0] if candidates else next(i for i in range(1, topo.n_atoms + 1) if i not in {c0, a1})
    return (topo.atom_names[c0 - 1], topo.atom_names[a1 - 1], topo.atom_names[a2 - 1])


def infer_species_chemistry(
    topo: MoleculeTopology,
    override: dict[str, Any] | None = None,
) -> SpeciesChemistry:
    """Infer molecular weight, hydrodynamic radius, molar volume, and functional sites automatically."""
    ov = override or {}
    g = nx.Graph()
    for i in range(1, topo.n_atoms + 1):
        g.add_node(i, elem=topo.elements[i - 1].upper())
    for a, b in topo.bonds_1based:
        g.add_edge(a, b)

    mass_sum = float(np.sum(topo.masses)) if len(topo.masses) else float(len(topo.elements) * 12.0)
    mw = float(ov.get("mw_g_mol", mass_sum))

    v_vdw_nm3 = _estimate_vdw_volume_nm3(topo.elements, len(topo.bonds_1based))
    r_sphere_nm = ((3.0 * v_vdw_nm3) / (4.0 * math.pi)) ** (1.0 / 3.0)
    rh_nm = float(ov.get("rh_nm", round(max(0.14, r_sphere_nm), 4)))

    dens = float(ov["density_gcm3"]) if ov.get("density_gcm3") is not None else None
    if dens is not None and dens > 0.0:
        mol_vol_cm3_mol = mw / dens
    else:
        # Liquid-packing scaling (~1.55x van der Waals volume) when bulk liquid density isn't given
        mol_vol_cm3_mol = v_vdw_nm3 * 1e-21 * AVOGADRO * 1.55

    # Polar heavy atoms & paired hydrogens
    if ov.get("polar_atoms") is not None:
        polar_atoms = tuple(str(x) for x in ov["polar_atoms"])
    else:
        hetero = [
            topo.atom_names[i - 1]
            for i in range(1, topo.n_atoms + 1)
            if topo.elements[i - 1].upper() in {"O", "N", "S", "P", "F", "CL", "BR"}
        ]
        if hetero:
            polar_atoms = tuple(hetero[:4])
        else:
            # Nonpolar unsaturated site fallback (e.g. alkene sp2 carbons in limonene)
            sp2 = [
                topo.atom_names[i - 1]
                for i in range(1, topo.n_atoms + 1)
                if topo.elements[i - 1].upper() == "C" and g.degree(i) == 3
            ]
            polar_atoms = tuple(sp2[:4] if sp2 else [a for a, e in zip(topo.atom_names, topo.elements) if e.upper() != "H"][:2])

    if ov.get("polar_h_atoms") is not None:
        polar_h = tuple(str(x) for x in ov["polar_h_atoms"])
    else:
        polar_h = tuple(h for _, h in topo.donors)

    # Ring heavy atoms
    if ov.get("ring_atoms") is not None:
        ring_atoms = tuple(str(x) for x in ov["ring_atoms"])
    else:
        heavy_g = g.subgraph([i for i in range(1, topo.n_atoms + 1) if topo.elements[i - 1].upper() != "H"])
        ring_ids = sorted({idx for cyc in nx.cycle_basis(heavy_g) for idx in cyc})
        ring_atoms = tuple(topo.atom_names[i - 1] for i in ring_ids)

    # Hydrophobic terminal tail atoms (terminal heavy atom farthest in graph hops from polar anchors)
    if ov.get("tail_atoms") is not None:
        tail_atoms = tuple(str(x) for x in ov["tail_atoms"])
    else:
        heavy_g = g.subgraph([i for i in range(1, topo.n_atoms + 1) if topo.elements[i - 1].upper() != "H"])
        name_to_1b = {n: i for i, n in enumerate(topo.atom_names, start=1)}
        sources = [name_to_1b[p] for p in polar_atoms if p in name_to_1b and p_in_graph(heavy_g, name_to_1b[p])]
        if not sources and heavy_g.number_of_nodes() > 0:
            sources = [topo.center_atom_0based + 1 if (topo.center_atom_0based + 1) in heavy_g else next(iter(heavy_g.nodes))]
        best_tail: tuple[int, int] | None = None
        if sources and nx.is_connected(heavy_g):
            dists = nx.multi_source_dijkstra_path_length(heavy_g, sources)
            leaves = [n for n in heavy_g.nodes if heavy_g.degree(n) == 1 and n not in sources]
            pool = leaves if leaves else [n for n in heavy_g.nodes if n not in sources]
            if pool:
                best_node = max(pool, key=lambda n: (dists.get(n, 0), -n))
                best_tail = (dists.get(best_node, 0), best_node)
        tail_atoms = (topo.atom_names[best_tail[1] - 1],) if best_tail and best_tail[0] >= 2 else ()

    if ov.get("align_atoms") is not None:
        aa = tuple(str(x) for x in ov["align_atoms"])
        align_atoms: tuple[str, str, str] | None = (aa[0], aa[1], aa[2]) if len(aa) >= 3 else _auto_align_atoms(topo, g)
    elif topo.align_atoms is not None:
        align_atoms = topo.align_atoms
    else:
        align_atoms = _auto_align_atoms(topo, g)

    # Default ring normal (axis 2) for flat ring solutes with no heteroatoms; C-X axis (axis 0) otherwise
    has_hetero = any(e.upper() in {"O", "N", "S"} for e in topo.elements)
    default_axis = 0 if has_hetero or len(ring_atoms) == 0 else 2
    sol_axis_index = int(ov.get("sol_axis_index", default_axis))

    return SpeciesChemistry(
        resname=topo.resname,
        mw_g_mol=mw,
        density_gcm3=dens,
        molar_volume_cm3_mol=float(mol_vol_cm3_mol),
        rh_nm=rh_nm,
        polar_atoms=polar_atoms,
        polar_h_atoms=polar_h,
        ring_atoms=ring_atoms,
        tail_atoms=tail_atoms,
        align_atoms=align_atoms,
        sol_axis_index=sol_axis_index,
    )


def p_in_graph(g: nx.Graph, node: int) -> bool:
    return node in g
