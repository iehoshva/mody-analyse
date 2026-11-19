from pathlib import Path
import numpy as np
from rdkit import Chem

from mody_analyse.core.discovery import MoleculeTopology
from mody_analyse.qm_runners.qm_xtb import (
    HARTREE_TO_KCAL,
    _topology_to_rdkit_mol,
    build_dihedral_constrain_file,
    compute_three_body_energy_kcal,
    parse_xtb_energy_eh,
)


def test_build_dihedral_constrain_file_formats_1based_indices(tmp_path: Path):
    out = tmp_path / "constrain.inp"
    build_dihedral_constrain_file(out, dihedrals_1based=[(1, 2, 3, 4), (2, 3, 4, 5)], force_constant=1.0)
    txt = out.read_text()
    assert "$constrain" in txt
    assert "dihedral: 1,2,3,4,auto" in txt
    assert "dihedral: 2,3,4,5,auto" in txt
    assert "$end" in txt


def test_parse_xtb_energy_and_three_body_formula():
    sample_out = """
          | TOTAL ENERGY               -10.145231456789 Eh   |
    """
    e = parse_xtb_energy_eh(sample_out)
    assert np.isclose(e, -10.145231456789, atol=1e-12)

    e3 = compute_three_body_energy_kcal(
        e_abc=-30.5,
        e_ab=-20.2,
        e_ac=-20.1,
        e_bc=-20.0,
        e_a=-10.0,
        e_b=-10.0,
        e_c=-10.0,
    )
    expected = (-30.5 + 20.2 + 20.1 + 20.0 - 30.0) * HARTREE_TO_KCAL
    assert np.isclose(e3, expected, atol=1e-6)


def test_topology_to_rdkit_mol_assigns_carbonyl_double_bond_from_itp():
    # Formaldehyde (CH2O): 4 atoms, valence detects C=O double bond without XYZ dependence
    topo = MoleculeTopology(
        resname="FMD",
        n_atoms=4,
        atom_names=["C1", "O1", "H1", "H2"],
        elements=["C", "O", "H", "H"],
        charges=np.zeros(4),
        masses=np.array([12.011, 15.999, 1.008, 1.008]),
        bonds_1based=[(1, 2), (1, 3), (1, 4)],
        donors=[],
        acceptors=["O1"],
        rotatable_dihedrals_1based=[],
        align_atoms=("C1", "O1", "H1"),
        center_atom_0based=0,
    )
    mol = _topology_to_rdkit_mol(topo, np.zeros((4, 3), dtype=float))
    assert mol.GetNumAtoms() == 4
    btypes = [b.GetBondType() for b in mol.GetBonds()]
    assert btypes.count(Chem.BondType.DOUBLE) == 1
