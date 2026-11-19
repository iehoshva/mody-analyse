from pathlib import Path

from mody_analyse.core.config import Config, load_config


def test_load_config_template_parses_types_and_defaults():
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "config.template.yaml"
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert 1 <= cfg.orca_nprocs <= 4
    assert cfg.orca_maxcore_mb <= 1500
    assert cfg.t_eq_ns >= 0.0
    assert cfg.t_end_ns > cfg.t_eq_ns
    assert cfg.kbi_plateau_range_nm == (1.4, 2.0)


def test_load_config_custom_yaml_resolves_paths(tmp_path: Path):
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                f'project_root: "{tmp_path / "runs"}"',
                f'workflow_root: "{tmp_path}"',
                f'cosmors_csv: "{tmp_path / "cosmo.csv"}"',
                f'results_dir: "{tmp_path / "results"}"',
                'run_glob: "TEST_*"',
                "binaries:",
                '  conda_env: "mody-analyse"',
                '  orca_bin: "orca"',
                '  orca_2mkl_bin: "orca_2mkl"',
                '  openmpi_dir: null',
                '  xtb_bin: "xtb"',
                '  multiwfn_bin: "Multiwfn_noGUI"',
                '  vmd_bin: "vmd"',
                "compute:",
                "  orca_nprocs: 2",
                "  orca_maxcore_mb: 1200",
                "  led_nprocs: 2",
                "  led_maxcore_mb: 2000",
                "  led_ram_limit_gb: 16.0",
                "  xtb_workers: 4",
                "md:",
                "  temperature_k: 298.15",
                "  t_eq_ns: 10.0",
                "  t_end_ns: 50.0",
                "  kbi_rebin_nm: 0.02",
                "  kbi_rmax_nm: 2.0",
                "  kbi_plateau_range_nm: [1.2, 1.8]",
                "  hbond_dist_nm: 0.35",
                "  hbond_angle_deg: 150.0",
                "  hbond_angle_sensitivity_deg: 135.0",
                "  rdf_stride: 5",
                "  hbond_stride: 1",
                "  network_stride: 10",
                "  sdf_stride: 10",
                "  sdf_grid_spacing_a: 0.5",
                "  sdf_half_width_a: 10.0",
                "qm_sampling:",
                "  strain_snapshots_per_rep: 10",
                "  strain_ref_conformers: 20",
                "  dimers_per_pair_per_rep: 10",
                "  trimers_per_system_per_rep: 5",
                "  dft_dimers_per_pair: 2",
                "  cp_validation_dimers_per_pair: 1",
                "  dft_trimers_per_solute: 2",
                "  nci_clusters_total: 4",
                "  led_dimers_total: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.run_glob == "TEST_*"
    assert cfg.openmpi_dir is None
    assert cfg.temperature_k == 298.15
    assert cfg.project_root == (tmp_path / "runs").resolve()
