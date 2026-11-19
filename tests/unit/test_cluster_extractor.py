import numpy as np
import pytest

from mody_analyse.md_to_qm.cluster_extractor import unwrap_pair_minimum_image, write_xyz


def test_unwrap_pair_minimum_image_reconstructs_split_box_cluster(tmp_path):
    box = np.array([50.0, 50.0, 50.0], dtype=float)  # Angstrom
    # Solute near x = 1.0 A
    sol_pos = np.array([[1.0, 10.0, 10.0], [2.4, 10.0, 10.0]])
    # Solvent across boundary at x = 49.0 A and 50.2 A (wrapped to 0.2 A) -> only 2.0 A away!
    solv_pos = np.array([[49.0, 10.0, 10.0], [48.1, 10.0, 10.0]])
    merged = unwrap_pair_minimum_image(sol_pos, [solv_pos], box)
    assert len(merged) == 2
    dist = np.linalg.norm(merged[1][0] - merged[0][0])
    assert dist < 3.0  # properly imaged to -1.0 A -> 2.0 A apart, NOT 48 A!
    intramol = np.linalg.norm(merged[1][1] - merged[1][0])
    assert intramol < 2.0

    xyz_path = tmp_path / "pair.xyz"
    write_xyz(xyz_path, ["O", "C", "O", "H"], np.vstack(merged), comment="test pair")
    lines = xyz_path.read_text().strip().splitlines()
    assert lines[0] == "4"
    assert len(lines) == 6
