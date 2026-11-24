"""Unit tests for automatic EDR -> XVG extraction & synchronization."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

from mody_analyse.analyzers.thermo_bulk import run_thermo_bulk_analysis
from mody_analyse.core.config import Config
from mody_analyse.core.discovery import MoleculeTopology, RunRecord, read_xvg_dataframe
from mody_analyse.core.edr_extractor import (
    DEFAULT_STRESS_TERMS,
    DEFAULT_THERMO_TERMS,
    read_xvg_legends,
    sync_directory_xvg_from_edr,
    sync_single_xvg_from_edr,
    write_xvg_from_dataframe,
)
from mody_analyse.core.registry import evaluate_eligibility


def _synthetic_edr_df() -> pd.DataFrame:
    t = np.arange(0.0, 20.0, 1.0)
    rng = np.random.default_rng(12)
    return pd.DataFrame(
        {
            "Time": t,
            "Temperature": 298.15 + rng.normal(0.0, 1.5, len(t)),
            "Pressure": 1.0 + rng.normal(0.0, 15.0, len(t)),
            "Density": 1000.0 + rng.normal(0.0, 2.0, len(t)),
            "Volume": 27.0 + rng.normal(0.0, 0.08, len(t)),
            "Potential": -15000.0 + rng.normal(0.0, 25.0, len(t)),
            "Kinetic En.": 3000.0 + rng.normal(0.0, 15.0, len(t)),
            "Total Energy": -12000.0 + rng.normal(0.0, 30.0, len(t)),
            "Enthalpy": -11998.0 + rng.normal(0.0, 30.0, len(t)),
            "LJ (SR)": -2500.0 + rng.normal(0.0, 10.0, len(t)),
            "Coulomb (SR)": -12500.0 + rng.normal(0.0, 20.0, len(t)),
            "Pres-XX": rng.normal(1.0, 20.0, len(t)),
            "Pres-YY": rng.normal(1.0, 20.0, len(t)),
            "Pres-ZZ": rng.normal(1.0, 20.0, len(t)),
            "Pres-XY": rng.normal(0.0, 15.0, len(t)),
            "Pres-XZ": rng.normal(0.0, 15.0, len(t)),
            "Pres-YZ": rng.normal(0.0, 15.0, len(t)),
        }
    )


def _install_fake_panedr(monkeypatch, df: pd.DataFrame) -> None:
    fake_mod = types.SimpleNamespace(edr_to_df=lambda _path: df.copy())
    monkeypatch.setitem(sys.modules, "panedr", fake_mod)


def test_sync_creates_updates_skips_and_refreshes_stale_xvg(tmp_path: Path, monkeypatch) -> None:
    edr_df = _synthetic_edr_df()
    _install_fake_panedr(monkeypatch, edr_df)

    edr_path = tmp_path / "04_prod" / "prod.edr"
    edr_path.parent.mkdir(parents=True)
    edr_path.write_bytes(b"fake-edr")

    thermo_xvg = tmp_path / "analysis" / "thermo.xvg"

    # 1. Missing thermo.xvg -> created from EDR
    res1 = sync_single_xvg_from_edr(edr_path, thermo_xvg, DEFAULT_THERMO_TERMS, run_id="R01", target="thermo", backend="panedr")
    assert res1.status == "created"
    assert thermo_xvg.is_file()
    assert "Enthalpy" in res1.columns
    assert "LJ (SR)" in res1.columns
    df_read = read_xvg_dataframe(thermo_xvg)
    assert len(df_read) == len(edr_df)
    assert "Enthalpy" in df_read.columns

    # 2. Already complete & newer -> up_to_date (untouched)
    res2 = sync_single_xvg_from_edr(edr_path, thermo_xvg, ["Temperature", "Volume", "Enthalpy"], run_id="R01", target="thermo", backend="panedr")
    assert res2.status == "up_to_date"

    # 3. Incomplete XVG (only Temperature + Volume, missing Enthalpy & LJ (SR)) -> updated!
    partial = edr_df[["Time", "Temperature", "Volume"]].copy()
    write_xvg_from_dataframe(partial, thermo_xvg)
    assert read_xvg_legends(thermo_xvg) == ["Temperature", "Volume"]
    res3 = sync_single_xvg_from_edr(edr_path, thermo_xvg, DEFAULT_THERMO_TERMS, run_id="R01", target="thermo", backend="panedr")
    assert res3.status == "updated"
    assert "Enthalpy" in res3.added_columns
    assert "LJ (SR)" in res3.added_columns

    # 4. Stale XVG (EDR modified later than XVG) -> refreshed!
    old_mtime = thermo_xvg.stat().st_mtime - 60.0
    os.utime(thermo_xvg, (old_mtime, old_mtime))
    res4 = sync_single_xvg_from_edr(edr_path, thermo_xvg, ["Temperature", "Volume", "Enthalpy"], run_id="R01", target="thermo", backend="panedr")
    assert res4.status == "updated"


def test_sync_directory_and_real_runrecord_integration(tmp_path: Path, monkeypatch) -> None:
    edr_df = _synthetic_edr_df()
    _install_fake_panedr(monkeypatch, edr_df)

    run_dir = tmp_path / "runs" / "PURE_R01"
    (run_dir / "04_prod").mkdir(parents=True)
    edr_path = run_dir / "04_prod" / "prod.edr"
    edr_path.write_bytes(b"fake-edr")

    sync_df = sync_directory_xvg_from_edr(tmp_path / "runs", backend="panedr")
    assert len(sync_df) == 2
    assert set(sync_df["status"]) == {"created"}
    thermo_path = run_dir / "analysis" / "thermo.xvg"
    stress_path = run_dir / "analysis" / "stress.xvg"
    assert thermo_path.is_file()
    assert stress_path.is_file()
    assert set(DEFAULT_STRESS_TERMS) <= set(read_xvg_legends(stress_path))

    # Build a real RunRecord and verify Gatekeeper + thermo_bulk work end-to-end with zero attribute mismatches
    topo = MoleculeTopology(
        resname="SOL",
        n_atoms=3,
        atom_names=["OW", "HW1", "HW2"],
        elements=["O", "H", "H"],
        charges=np.array([-0.834, 0.417, 0.417]),
        masses=np.array([15.999, 1.008, 1.008]),
        bonds_1based=[(1, 2), (1, 3)],
        donors=[("OW", "HW1"), ("OW", "HW2")],
        acceptors=["OW"],
        rotatable_dihedrals_1based=[],
        align_atoms=("OW", "HW1", "HW2"),
    )
    run = RunRecord(
        run_id="PURE_R01",
        system_code="PURE",
        replica="R01",
        solute_resname=None,
        path=run_dir,
        metadata={"temperatureK": 298.15},
        molecule_counts={"SOL": 500},
        topologies={"SOL": topo},
        tpr_path=run_dir / "04_prod" / "prod.tpr",
        xtc_path=run_dir / "04_prod" / "prod.xtc",
        edr_path=edr_path,
        log_path=run_dir / "04_prod" / "prod.log",
        thermo_path=thermo_path,
        stress_path=stress_path,
        rdf_paths={},
        block_rdf_paths=[],
        solvent_resnames=("SOL",),
    )
    cfg = Config(
        project_root=tmp_path / "runs",
        workflow_root=tmp_path,
        cosmors_csv=None,
        results_dir=tmp_path / "results",
        conda_env="mody-analyse",
        orca_bin=Path("orca"),
        orca_2mkl_bin=Path("orca_2mkl"),
        xtb_bin=Path("xtb"),
        multiwfn_bin=Path("Multiwfn_noGUI"),
        vmd_bin=Path("vmd"),
        orca_nprocs=4,
        orca_maxcore_mb=1500,
        led_nprocs=4,
        led_maxcore_mb=2500,
        led_ram_limit_gb=28.0,
        xtb_workers=4,
        temperature_k=298.15,
        t_eq_ns=0.0,
        t_end_ns=1.0,
        kbi_rebin_nm=0.02,
        kbi_rmax_nm=2.0,
        kbi_plateau_range_nm=(1.0, 1.6),
        hbond_dist_nm=0.35,
        hbond_angle_deg=150.0,
        hbond_angle_sensitivity_deg=135.0,
        rdf_stride=1,
        hbond_stride=1,
        network_stride=1,
        sdf_stride=1,
        sdf_grid_spacing_a=0.5,
        sdf_half_width_a=10.0,
        strain_snapshots_per_rep=4,
        strain_ref_conformers=4,
        dimers_per_pair_per_rep=4,
        trimers_per_system_per_rep=2,
        dft_dimers_per_pair=1,
        cp_validation_dimers_per_pair=1,
        dft_trimers_per_solute=1,
        nci_clusters_total=1,
        led_dimers_total=1,
    )

    decisions, _ = evaluate_eligibility([run], cfg, ["thermo_fluctuations", "electrochemistry_onsager"])
    assert decisions["thermo_fluctuations"].status == "RUN"
    assert decisions["electrochemistry_onsager"].status == "SKIPPED"

    th = run_thermo_bulk_analysis([run], cfg)["thermo_fluctuations"]
    assert len(th) == 1
    assert np.isfinite(th["cp_j_mol_k"].iloc[0])
    assert np.isfinite(th["kappa_t_gpa_inv"].iloc[0])
    assert np.isfinite(th["hildebrand_delta_mpa_half"].iloc[0])
