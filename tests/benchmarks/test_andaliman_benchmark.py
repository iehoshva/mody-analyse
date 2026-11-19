from dataclasses import replace
import os
from pathlib import Path
import numpy as np
import pytest
from rdkit import Chem

from mody_analyse.analyzers.equilibration import analyze_equilibration, analyze_equilibration_windows
from mody_analyse.analyzers.md_structural import analyze_orientation_and_sdf, analyze_solvation_shells
from mody_analyse.core.config import load_config
from mody_analyse.core.discovery import discover_runs, read_xvg
from mody_analyse.qm_runners.qm_xtb import _topology_to_rdkit_mol


_BENCH_CFG = os.environ.get("MODY_BENCHMARK_CONFIG")
pytestmark = pytest.mark.skipif(
    not _BENCH_CFG or not Path(_BENCH_CFG).exists(),
    reason="Set MODY_BENCHMARK_CONFIG to a local Andaliman dataset config.yaml to run trajectory benchmarks.",
)


def _cfg():
    assert _BENCH_CFG is not None
    return load_config(Path(_BENCH_CFG))


def test_discover_runs_finds_all_eight_completed_runs():
    runs = discover_runs(_cfg())
    assert len(runs) == 8
    by_code = {}
    for r in runs:
        by_code.setdefault(r.system_code, []).append(r.replica)
    assert set(by_code.keys()) == {"PURE", "GA", "GER", "LIM"}
    for reps in by_code.values():
        assert sorted(reps) == ["R01", "R02"]


def test_parse_itp_topology_identifies_donors_acceptors_and_bonds():
    runs = discover_runs(_cfg())
    ga_run = next(r for r in runs if r.system_code == "GA" and r.replica == "R01")
    assert ga_run.topologies["THY"].n_atoms == 25
    assert ga_run.topologies["DOD"].n_atoms == 39
    assert ga_run.topologies["GAC"].n_atoms == 34


def test_read_xvg_handles_varying_column_headers_dynamically():
    runs = discover_runs(_cfg())
    pure_r01 = next(r for r in runs if r.system_code == "PURE" and r.replica == "R01")
    t_cols, _ = read_xvg(pure_r01.thermo_path, max_rows=20)
    assert "Time" in t_cols
    assert pure_r01.stress_path is not None and pure_r01.stress_path.exists()


def test_equilibration_verifies_window_and_block_convergence():
    runs = discover_runs(_cfg())
    pure_r01 = next(r for r in runs if r.system_code == "PURE" and r.replica == "R01")
    eq_df = analyze_equilibration([pure_r01], _cfg())
    win_df = analyze_equilibration_windows([pure_r01], window_ns=50.0)
    assert len(eq_df) == 1
    assert len(win_df) == 10


def test_analyze_solvation_and_orientation_on_real_run(tmp_path: Path):
    cfg = _cfg()
    runs = discover_runs(cfg)
    ga_r01 = next(r for r in runs if r.system_code == "GA" and r.replica == "R01")
    fast_cfg = replace(cfg, results_dir=tmp_path, t_eq_ns=418.0, t_end_ns=420.0, rdf_stride=5, sdf_stride=5)
    res_solv = analyze_solvation_shells([ga_r01], fast_cfg)
    assert res_solv["enrichment_runs"].iloc[0]["n_thy_shell"] > 0.0
    res_ori = analyze_orientation_and_sdf([ga_r01], replace(fast_cfg, t_eq_ns=419.5))
    assert res_ori["orientation_runs"].iloc[0]["sdf_thy_oh_max"] > 0.0


def test_topology_to_rdkit_mol_assigns_bond_orders_from_real_itp():
    runs = discover_runs(_cfg())
    topos = {k: v for r in runs for k, v in r.topologies.items()}
    mol = _topology_to_rdkit_mol(topos["THY"], np.zeros((topos["THY"].n_atoms, 3), dtype=float))
    btypes = [b.GetBondType() for b in mol.GetBonds()]
    assert btypes.count(Chem.BondType.AROMATIC) == 6
