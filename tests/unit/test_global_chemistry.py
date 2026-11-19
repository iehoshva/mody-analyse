from pathlib import Path
import numpy as np
import pandas as pd

from mody_analyse.analyzers.md_structural import compute_multispecies_enrichment
from mody_analyse.core.chemistry import infer_species_chemistry
from mody_analyse.core.config import load_config
from mody_analyse.core.discovery import MoleculeTopology, discover_runs
from mody_analyse.reporting.plotting import generate_all_figures
from mody_analyse.reporting.synthesis import build_master_synthesis_table, load_benchmark_reference


def test_infer_species_chemistry_water_ring_and_alkyl_alcohol():
    wat = MoleculeTopology(
        resname="WAT",
        n_atoms=3,
        atom_names=["O1", "H1", "H2"],
        elements=["O", "H", "H"],
        charges=np.array([-0.82, 0.41, 0.41]),
        masses=np.array([15.999, 1.008, 1.008]),
        bonds_1based=[(1, 2), (1, 3)],
        donors=[("O1", "H1"), ("O1", "H2")],
        acceptors=["O1"],
        rotatable_dihedrals_1based=[],
        align_atoms=None,
        center_atom_0based=0,
    )
    ch_w = infer_species_chemistry(wat)
    assert np.isclose(ch_w.mw_g_mol, 18.015, atol=1e-3)
    assert 0.14 <= ch_w.rh_nm <= 0.25
    assert ch_w.polar_atoms == ("O1",)
    assert ch_w.align_atoms == ("O1", "H1", "H2")

    # Hexanol chain: O1 - C1 - C2 - C3 - C4 - C5 - C6 -> C6 should be inferred as hydrophobic tail
    hexanol = MoleculeTopology(
        resname="HEX",
        n_atoms=7,
        atom_names=["O1", "C1", "C2", "C3", "C4", "C5", "C6"],
        elements=["O", "C", "C", "C", "C", "C", "C"],
        charges=np.zeros(7),
        masses=np.array([15.999] + [12.011] * 6),
        bonds_1based=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
        donors=[],
        acceptors=["O1"],
        rotatable_dihedrals_1based=[],
        align_atoms=None,
        center_atom_0based=3,
    )
    ch_h = infer_species_chemistry(hexanol)
    assert ch_h.tail_atoms == ("C6",)
    assert ch_h.align_atoms is not None and len(ch_h.align_atoms) == 3


def test_discover_runs_supports_standard_flat_gromacs_without_metadata(tmp_path: Path):
    run_dir = tmp_path / "runs" / "IBU_RELINE_R02"
    run_dir.mkdir(parents=True)
    (run_dir / "topol.top").write_text(
        """[ moleculetype ]
CHCL 3
[ atoms ]
1 N1 1 CHCL N1 1 0.1 14.007
2 C1 1 CHCL C1 2 0.0 12.011
3 O1 1 CHCL O1 3 -0.5 15.999
[ bonds ]
1 2 1
2 3 1

[ moleculetype ]
IBU 3
[ atoms ]
1 C1 1 IBU C1 1 0.0 12.011
2 O1 1 IBU O1 2 -0.5 15.999
3 H1 1 IBU H1 3 0.4 1.008
[ bonds ]
1 2 1
2 3 1

[ molecules ]
CHCL 120
IBU 6
""",
        encoding="utf-8",
    )
    (run_dir / "prod.tpr").write_bytes(b"dummy")
    (run_dir / "prod.xtc").write_bytes(b"dummy")
    (run_dir / "thermo.xvg").write_text("0.0 300.0 1020.0 -100.0 50.0\n", encoding="utf-8")

    cfg_p = tmp_path / "cfg.yaml"
    cfg_p.write_text(
        f"""
project_root: "{tmp_path / 'runs'}"
workflow_root: "{tmp_path}"
results_dir: "{tmp_path / 'results'}"
binaries:
  conda_env: mody-analyse
  orca_bin: orca
  orca_2mkl_bin: orca_2mkl
  xtb_bin: xtb
  multiwfn_bin: Multiwfn_noGUI
  vmd_bin: vmd
compute:
  orca_nprocs: 2
  orca_maxcore_mb: 1000
  led_nprocs: 2
  led_maxcore_mb: 1500
  led_ram_limit_gb: 8.0
  xtb_workers: 2
md:
  temperature_k: 310.0
  t_eq_ns: 0.0
  t_end_ns: 10.0
  kbi_rebin_nm: 0.02
  kbi_rmax_nm: 2.0
  kbi_plateau_range_nm: [1.2, 1.8]
  hbond_dist_nm: 0.35
  hbond_angle_deg: 150.0
  hbond_angle_sensitivity_deg: 135.0
  rdf_stride: 1
  hbond_stride: 1
  network_stride: 1
  sdf_stride: 1
  sdf_grid_spacing_a: 0.5
  sdf_half_width_a: 10.0
qm_sampling:
  strain_snapshots_per_rep: 2
  strain_ref_conformers: 2
  dimers_per_pair_per_rep: 2
  trimers_per_system_per_rep: 2
  dft_dimers_per_pair: 1
  cp_validation_dimers_per_pair: 1
  dft_trimers_per_solute: 1
  nci_clusters_total: 2
  led_dimers_total: 1
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_p)
    runs = discover_runs(cfg)
    assert len(runs) == 1
    r = runs[0]
    assert r.replica == "R02"
    assert r.solvent_resnames == ("CHCL",)
    assert r.solute_resname == "IBU"
    assert r.system_code == "IBU"
    assert r.chemistry["CHCL"].align_atoms == ("C1", "N1", "O1") or len(r.chemistry["CHCL"].align_atoms) == 3


def test_multispecies_enrichment_supports_ternary_solvents():
    res = compute_multispecies_enrichment(
        {"CHCL": 4.0, "URE": 6.0, "WAT": 2.0},
        {"CHCL": 100, "URE": 200, "WAT": 100},
        {"CHCL": 110.0, "URE": 45.0, "WAT": 18.0},
    )
    assert np.isclose(res["CHCL"]["x_local"] + res["URE"]["x_local"] + res["WAT"]["x_local"], 1.0)
    assert res["CHCL"]["enrichment_molar"] > 1.0


def test_global_synthesis_and_plotting_without_cosmors(tmp_path: Path):
    dummy = {
        "enrichment_solvent_pairs_summary": pd.DataFrame(
            [
                {"system_code": "IBU", "solvent": "CHCL", "n_shell_mean": 3.1, "n_shell_sd": 0.2, "enrichment_molar_mean": 1.15, "enrichment_molar_sd": 0.04},
                {"system_code": "IBU", "solvent": "URE", "n_shell_mean": 5.4, "n_shell_sd": 0.3, "enrichment_molar_mean": 0.92, "enrichment_molar_sd": 0.03},
                {"system_code": "CAF", "solvent": "CHCL", "n_shell_mean": 2.5, "n_shell_sd": 0.1, "enrichment_molar_mean": 0.96, "enrichment_molar_sd": 0.02},
                {"system_code": "CAF", "solvent": "URE", "n_shell_mean": 6.1, "n_shell_sd": 0.2, "enrichment_molar_mean": 1.03, "enrichment_molar_sd": 0.02},
            ]
        ),
        "kbi_summary": pd.DataFrame(
            [
                {"system_code": "IBU", "G_S_CHCL_cm3_mol": -95.0, "G_S_CHCL_sd": 4.0, "G_S_URE_cm3_mol": -140.0, "G_S_URE_sd": 6.0},
                {"system_code": "CAF", "G_S_CHCL_cm3_mol": -120.0, "G_S_CHCL_sd": 5.0, "G_S_URE_cm3_mol": -80.0, "G_S_URE_sd": 3.0},
            ]
        ),
        "transport_summary": pd.DataFrame(
            [
                {"system_code": "PURE", "eta_tail_mpa_s": 42.0, "d_chcl_1e5_cm2_s": 0.05, "d_ure_1e5_cm2_s": 0.08, "d_sol_1e5_cm2_s": np.nan},
                {"system_code": "IBU", "eta_tail_mpa_s": 44.0, "d_chcl_1e5_cm2_s": 0.04, "d_ure_1e5_cm2_s": 0.07, "d_sol_1e5_cm2_s": 0.03},
                {"system_code": "CAF", "eta_tail_mpa_s": 43.0, "d_chcl_1e5_cm2_s": 0.05, "d_ure_1e5_cm2_s": 0.08, "d_sol_1e5_cm2_s": 0.04},
            ]
        ),
        "dimer_xtb": pd.DataFrame(
            [
                {"pair": "IBU-CHCL", "system_code": "IBU", "de_int_kcal": -11.2},
                {"pair": "IBU-URE", "system_code": "IBU", "de_int_kcal": -8.4},
                {"pair": "CAF-CHCL", "system_code": "CAF", "de_int_kcal": -7.5},
                {"pair": "CAF-URE", "system_code": "CAF", "de_int_kcal": -9.9},
            ]
        ),
    }
    master = build_master_synthesis_table(dummy)
    assert set(master["system_code"]) == {"IBU", "CAF"}
    assert "kbi_G_S_CHCL_cm3_mol" in master.columns
    assert "xtb_de_int_URE_mean_kcal" in master.columns
    figs = generate_all_figures(dummy, tmp_path / "figs")
    assert len(figs) >= 5
    for p in figs:
        assert p.exists() and p.stat().st_size > 1000


def test_load_generic_benchmark_csv(tmp_path: Path):
    csv_p = tmp_path / "exp.csv"
    pd.DataFrame(
        [
            {"solvent": "Reline", "solute": "IBU", "logS": -1.2},
            {"solvent": "Reline", "solute": "CAF", "logS": -0.5},
        ]
    ).to_csv(csv_p, index=False)
    cfg_p = tmp_path / "c.yaml"
    cfg_p.write_text(
        f"""
project_root: "{tmp_path}"
workflow_root: "{tmp_path}"
benchmark:
  csv_path: "{csv_p}"
  format: generic
  filters: {{solvent: Reline}}
  system_col: solute
  value_col: logS
binaries:
  conda_env: mody-analyse
  orca_bin: orca
  orca_2mkl_bin: orca_2mkl
  xtb_bin: xtb
  multiwfn_bin: Multiwfn_noGUI
  vmd_bin: vmd
compute: {{orca_nprocs: 2, orca_maxcore_mb: 1000, led_nprocs: 2, led_maxcore_mb: 1500, led_ram_limit_gb: 8.0, xtb_workers: 2}}
md: {{temperature_k: 298.15, t_eq_ns: 0.0, t_end_ns: 10.0, kbi_rebin_nm: 0.02, kbi_rmax_nm: 2.0, kbi_plateau_range_nm: [1.2, 1.8], hbond_dist_nm: 0.35, hbond_angle_deg: 150.0, hbond_angle_sensitivity_deg: 135.0, rdf_stride: 1, hbond_stride: 1, network_stride: 1, sdf_stride: 1, sdf_grid_spacing_a: 0.5, sdf_half_width_a: 10.0}}
qm_sampling: {{strain_snapshots_per_rep: 2, strain_ref_conformers: 2, dimers_per_pair_per_rep: 2, trimers_per_system_per_rep: 2, dft_dimers_per_pair: 1, cp_validation_dimers_per_pair: 1, dft_trimers_per_solute: 1, nci_clusters_total: 2, led_dimers_total: 1}}
""",
        encoding="utf-8",
    )
    ref = load_benchmark_reference(load_config(cfg_p))
    assert set(ref["solute_code"]) == {"IBU", "CAF"}
    assert np.isclose(ref.loc[ref["solute_code"] == "IBU", "ln_gamma_inf"].iloc[0], -1.2)
