from pathlib import Path
import numpy as np

from mody_analyse.core.discovery import (
    compute_whole_coms_vectorized,
    equilibrated_frame_slice,
    parse_itp_topology,
    read_xvg,
)


def test_parse_itp_topology_synthetic_ethanol(tmp_path: Path):
    itp = tmp_path / "eth.itp"
    itp.write_text(
        """[ moleculetype ]
ETH 3

[ atoms ]
1 opls_135 1 ETH C1 1 -0.18 12.011
2 opls_157 1 ETH C2 2 0.145 12.011
3 opls_154 1 ETH O1 3 -0.683 15.999
4 opls_155 1 ETH H1 4 0.418 1.008

[ bonds ]
1 2 1
2 3 1
3 4 1
""",
        encoding="utf-8",
    )
    topo = parse_itp_topology(itp)
    assert topo.resname == "ETH"
    assert topo.n_atoms == 4
    assert topo.elements == ["C", "C", "O", "H"]
    assert topo.donors == [("O1", "H1")]
    assert topo.acceptors == ["O1"]


def test_read_xvg_synthetic_legends_and_max_rows(tmp_path: Path):
    xvg = tmp_path / "thermo.xvg"
    xvg.write_text(
        """@ s0 legend "Temperature"
@ s1 legend "Density"
0.0   300.0   950.0
2.0   301.0   951.0
4.0   299.5   949.5
""",
        encoding="utf-8",
    )
    cols, arr = read_xvg(xvg, max_rows=2)
    assert cols == ["Time", "Temperature", "Density"]
    assert arr.shape == (2, 3)
    assert np.isclose(arr[1, 1], 301.0)


def test_compute_whole_coms_vectorized_unwraps_split_molecules():
    pos = np.array([
        [10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [12.0, 10.0, 10.0],
        [49.5, 20.0, 20.0], [0.5, 20.0, 20.0], [1.5, 20.0, 20.0],
    ])
    masses = np.array([12.0, 12.0, 12.0])
    box = np.array([50.0, 50.0, 50.0])
    coms = compute_whole_coms_vectorized(pos, n_mol=2, n_atoms=3, masses=masses, box_a=box, anchor_idx=1)
    assert np.allclose(coms[0], [11.0, 10.0, 10.0], atol=1e-6)
    assert np.allclose(coms[1], [0.5, 20.0, 20.0], atol=1e-6)


def test_equilibrated_frame_slice_skips_corrupt_tail_frame():
    class _FakeTS:
        def __init__(self, time_ps: float):
            self.time = time_ps

    class _FakeTraj:
        dt = 20.0

        def __init__(self):
            self._times = [i * 20.0 for i in range(21013)]

        def __len__(self):
            return len(self._times)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return [self[i] for i in range(*idx.indices(len(self._times)))]
            if idx == 21012:
                raise OSError("XTC read error = compression")
            return _FakeTS(self._times[idx])

        def __iter__(self):
            for i in range(21012):
                yield _FakeTS(self._times[i])

    class _FakeUniverse:
        trajectory = _FakeTraj()

    start, stop, stride = equilibrated_frame_slice(_FakeUniverse(), t_eq_ns=100.0, t_end_ns=500.0, stride=1)
    assert start == 5000
    assert stop == 21012
    assert stride == 1
    assert _FakeUniverse.trajectory[stop - 1].time == 420220.0
