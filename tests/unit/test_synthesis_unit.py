from pathlib import Path
import numpy as np
import pandas as pd

from mody_analyse.reporting.plotting import generate_all_figures
from mody_analyse.reporting.synthesis import (
    build_master_synthesis_table,
    build_replicate_synthesis_table,
    load_cosmors_reference,
)


def _write_synthetic_cosmors(csv_path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "HBA": "Thymol",
                "HBD": "Dodecanol",
                "HBA_Ratio": 5,
                "HBD_Ratio": 1,
                "Temperature_C": 50.0,
                "ln_gamma_Geranyl_Acetate": -1.62,
                "C_Geranyl_Acetate": 5.05,
                "ln_gamma_Geraniol": -0.87,
                "C_Geraniol": 2.38,
                "ln_gamma_Limonene": 0.53,
                "C_Limonene": 0.59,
                "S_GA_vs_Geraniol": 2.12,
                "S_GA_vs_Limonene": 8.56,
            }
        ]
    ).to_csv(csv_path, index=False)
    return csv_path


def test_load_cosmors_reference_extracts_entry(tmp_path: Path):
    csv_p = _write_synthetic_cosmors(tmp_path / "cosmo.csv")
    df = load_cosmors_reference(csv_p)
    assert set(df["solute_code"]) == {"GA", "GER", "LIM"}
    ga = df.loc[df["solute_code"] == "GA"].iloc[0]
    assert np.isclose(ga["ln_gamma_inf"], -1.62)
    assert np.isclose(ga["selectivity_ga_vs_ger"], 2.12)
    assert np.isclose(ga["selectivity_ga_vs_lim"], 8.56)


def test_build_master_synthesis_table_and_figures(tmp_path: Path):
    csv_p = _write_synthetic_cosmors(tmp_path / "cosmo.csv")
    df_cosmo = load_cosmors_reference(csv_p)
    dummy = {
        "cosmors": df_cosmo,
        "enrichment_summary": pd.DataFrame([
            {"system_code": "GA", "e_thy_molar_mean": 1.05, "e_thy_molar_sd": 0.02, "n_thy_mean": 7.5, "n_dod_mean": 1.1},
            {"system_code": "GER", "e_thy_molar_mean": 1.02, "e_thy_molar_sd": 0.02, "n_thy_mean": 7.2, "n_dod_mean": 1.3},
            {"system_code": "LIM", "e_thy_molar_mean": 0.98, "e_thy_molar_sd": 0.01, "n_thy_mean": 6.8, "n_dod_mean": 1.4},
        ]),
        "kbi_summary": pd.DataFrame([
            {"system_code": "GA", "G_S_THY_cm3_mol": -120.0, "G_S_DOD_cm3_mol": -180.0, "delta_S_THY": 0.04},
            {"system_code": "GER", "G_S_THY_cm3_mol": -110.0, "G_S_DOD_cm3_mol": -150.0, "delta_S_THY": 0.02},
            {"system_code": "LIM", "G_S_THY_cm3_mol": -210.0, "G_S_DOD_cm3_mol": -140.0, "delta_S_THY": -0.03},
        ]),
        "transport_summary": pd.DataFrame([
            {"system_code": "PURE", "d_thy_1e5_cm2_s": 0.30, "d_dod_1e5_cm2_s": 0.18, "d_sol_1e5_cm2_s": float("nan"), "eta_tail_mpa_s": 3.5},
            {"system_code": "GA", "d_thy_1e5_cm2_s": 0.29, "d_dod_1e5_cm2_s": 0.17, "d_sol_1e5_cm2_s": 0.22, "eta_tail_mpa_s": 3.6},
            {"system_code": "GER", "d_thy_1e5_cm2_s": 0.28, "d_dod_1e5_cm2_s": 0.17, "d_sol_1e5_cm2_s": 0.20, "eta_tail_mpa_s": 3.7},
            {"system_code": "LIM", "d_thy_1e5_cm2_s": 0.31, "d_dod_1e5_cm2_s": 0.19, "d_sol_1e5_cm2_s": 0.28, "eta_tail_mpa_s": 3.4},
        ]),
        "hbond_summary": pd.DataFrame([
            {"system_code": "GA", "sol_hbonds_per_mol_mean": 0.85, "tau_hb_ps": 120.0},
            {"system_code": "GER", "sol_hbonds_per_mol_mean": 1.35, "tau_hb_ps": 160.0},
            {"system_code": "LIM", "sol_hbonds_per_mol_mean": 0.0, "tau_hb_ps": 0.0},
        ]),
        "network_summary": pd.DataFrame([
            {"system_code": "PURE", "mean_degree": 1.45, "giant_frac": 0.35},
            {"system_code": "GA", "mean_degree": 1.41, "giant_frac": 0.32},
            {"system_code": "GER", "mean_degree": 1.48, "giant_frac": 0.37},
            {"system_code": "LIM", "mean_degree": 1.44, "giant_frac": 0.34},
        ]),
        "strain_summary": pd.DataFrame([
            {"system_code": "GA", "e_strain_relaxed_mean_kcal": 1.8},
            {"system_code": "GER", "e_strain_relaxed_mean_kcal": 1.5},
            {"system_code": "LIM", "e_strain_relaxed_mean_kcal": 0.4},
        ]),
        "dimer_xtb": pd.DataFrame([
            {"pair": "GA-THY", "system_code": "GA", "de_int_kcal": -8.2},
            {"pair": "GA-DOD", "system_code": "GA", "de_int_kcal": -6.1},
            {"pair": "GER-THY", "system_code": "GER", "de_int_kcal": -9.1},
            {"pair": "GER-DOD", "system_code": "GER", "de_int_kcal": -7.4},
            {"pair": "LIM-THY", "system_code": "LIM", "de_int_kcal": -4.2},
            {"pair": "LIM-DOD", "system_code": "LIM", "de_int_kcal": -4.6},
        ]),
        "trimer_xtb": pd.DataFrame([
            {"system_code": "GA", "motif": "S+2THY", "e_3body_kcal": -0.7},
            {"system_code": "GER", "motif": "S+2THY", "e_3body_kcal": -1.1},
            {"system_code": "LIM", "motif": "S+2THY", "e_3body_kcal": 0.1},
        ]),
        "led_eda": pd.DataFrame([
            {"label": "GA-THY", "system_code": "GA", "e_elstat_kcal": -10.2, "e_exch_kcal": -2.8, "e_disp_kcal": -4.1, "e_prep_kcal": 8.9, "e_total_kcal": -8.2},
            {"label": "GER-THY", "system_code": "GER", "e_elstat_kcal": -11.4, "e_exch_kcal": -3.1, "e_disp_kcal": -3.4, "e_prep_kcal": 8.7, "e_total_kcal": -9.2},
            {"label": "LIM-THY", "system_code": "LIM", "e_elstat_kcal": -1.5, "e_exch_kcal": -0.9, "e_disp_kcal": -4.4, "e_prep_kcal": 2.6, "e_total_kcal": -4.2},
        ]),
    }
    scatter_p = tmp_path / "igmh_4col.txt"
    np.savetxt(scatter_p, np.array([[0.0, 0.0, 0.0, 0.0], [0.02, 0.01, 0.03, -0.025], [0.008, 0.004, 0.012, -0.005]]))
    dummy["nci_igmh"] = pd.DataFrame([
        {
            "cluster_id": "GA_THY_R01_d001",
            "kind": "dimer",
            "system_code": "GA",
            "motif": "hbond",
            "igmh_scatter_path": str(scatter_p),
            "igmh_frac_hb": 0.5,
            "igmh_frac_vdw": 0.5,
            "igmh_frac_steric": 0.0,
            "igmh_min_sl2r": -0.025,
            "integral_dg_inter_au": 0.146,
        }
    ])
    master = build_master_synthesis_table(dummy)
    assert set(master["system_code"]) == {"GA", "GER", "LIM"}
    assert "led_elstat_THY_kcal" in master.columns
    assert np.isclose(master.loc[master["system_code"] == "GA", "led_elstat_THY_kcal"].iloc[0], -10.2)
    dummy["kbi_systems"] = pd.DataFrame([
        {"run_id": "GA_R01", "system_code": "GA", "replica": "R01", "G_S_THY_cm3_mol": -118.0, "G_S_DOD_cm3_mol": -182.0, "delta_S_THY": 0.04},
        {"run_id": "GA_R02", "system_code": "GA", "replica": "R02", "G_S_THY_cm3_mol": -122.0, "G_S_DOD_cm3_mol": -178.0, "delta_S_THY": 0.04},
    ])
    rep_tbl = build_replicate_synthesis_table(dummy)
    assert len(rep_tbl) == 2

    figs = generate_all_figures(dummy, tmp_path)
    assert len(figs) >= 5
    for p in figs:
        assert p.exists() and p.stat().st_size > 1000
