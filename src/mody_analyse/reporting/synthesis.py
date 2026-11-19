from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mody_analyse.core.config import Config


def load_cosmors_reference(csv_path: Path) -> pd.DataFrame:
    """Load Thymol:Dodecanol 5:1 COSMO-RS benchmark CSV when present."""
    df = pd.read_csv(csv_path)
    sub = df.loc[
        (df["HBA"].str.lower() == "thymol")
        & (df["HBD"].str.lower() == "dodecanol")
        & (df["HBA_Ratio"] == 5)
        & (df["HBD_Ratio"] == 1)
        & (np.isclose(df["Temperature_C"], 50.0))
    ]
    if sub.empty:
        raise ValueError(f"Thymol:dodecanol 5:1 at 50.0 C not found in {csv_path}")
    row = sub.iloc[0]
    s_ga_ger = float(row["S_GA_vs_Geraniol"])
    s_ga_lim = float(row["S_GA_vs_Limonene"])
    return pd.DataFrame(
        [
            {
                "solute_code": "GA",
                "solute_name": "Geranyl Acetate",
                "ln_gamma_inf": float(row["ln_gamma_Geranyl_Acetate"]),
                "capacity_c": float(row["C_Geranyl_Acetate"]),
                "selectivity_ga_vs_ger": s_ga_ger,
                "selectivity_ga_vs_lim": s_ga_lim,
            },
            {
                "solute_code": "GER",
                "solute_name": "Geraniol",
                "ln_gamma_inf": float(row["ln_gamma_Geraniol"]),
                "capacity_c": float(row["C_Geraniol"]),
                "selectivity_ga_vs_ger": s_ga_ger,
                "selectivity_ga_vs_lim": s_ga_lim,
            },
            {
                "solute_code": "LIM",
                "solute_name": "Limonene",
                "ln_gamma_inf": float(row["ln_gamma_Limonene"]),
                "capacity_c": float(row["C_Limonene"]),
                "selectivity_ga_vs_ger": s_ga_ger,
                "selectivity_ga_vs_lim": s_ga_lim,
            },
        ]
    )


def load_benchmark_reference(cfg: Config) -> pd.DataFrame:
    """Load optional external reference/benchmark CSV (COSMO-RS or generic experimental table)."""
    bench_cfg = cfg.benchmark or {}
    csv_path = cfg.cosmors_csv
    if csv_path is None and bench_cfg.get("csv_path"):
        csv_path = Path(str(bench_cfg["csv_path"]))
    if csv_path is None or not Path(csv_path).exists():
        return pd.DataFrame()

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()

    fmt = str(bench_cfg.get("format", "auto")).lower()
    if fmt == "andaliman_cosmors" or (
        fmt == "auto" and {"HBA", "HBD", "ln_gamma_Geranyl_Acetate"}.issubset(df.columns)
    ):
        return load_cosmors_reference(csv_path)

    filters = bench_cfg.get("filters") or {}
    for col, val in filters.items():
        if col in df.columns:
            df = df.loc[df[col] == val]
    if df.empty:
        return pd.DataFrame()

    sys_col = str(bench_cfg.get("system_col", "system_code" if "system_code" in df.columns else "solute_code"))
    val_col = bench_cfg.get("value_col")
    cap_col = bench_cfg.get("capacity_col")
    name_col = bench_cfg.get("name_col")
    out = pd.DataFrame()
    if sys_col in df.columns:
        out["solute_code"] = df[sys_col].astype(str)
    else:
        return pd.DataFrame()
    out["solute_name"] = df[name_col].astype(str) if name_col and name_col in df.columns else out["solute_code"]
    if val_col and val_col in df.columns:
        out["ln_gamma_inf"] = pd.to_numeric(df[val_col], errors="coerce")
    elif "ln_gamma_inf" in df.columns:
        out["ln_gamma_inf"] = pd.to_numeric(df["ln_gamma_inf"], errors="coerce")
    if cap_col and cap_col in df.columns:
        out["capacity_c"] = pd.to_numeric(df[cap_col], errors="coerce")
    elif "capacity_c" in df.columns:
        out["capacity_c"] = pd.to_numeric(df["capacity_c"], errors="coerce")
    return out.reset_index(drop=True)


def _discover_system_codes(results: dict[str, Any], cosmors: pd.DataFrame) -> list[str]:
    ordered: list[str] = []
    if not cosmors.empty and "solute_code" in cosmors.columns:
        for sc in cosmors["solute_code"].dropna().astype(str):
            if sc not in ordered:
                ordered.append(sc)

    solute_first_tables = (
        "enrichment_summary",
        "hbond_summary",
        "kbi_summary",
        "orientation_summary",
        "strain_summary",
        "dimer_xtb",
        "xtb_dimers",
        "dft_dimers",
        "led_eda",
        "nci_igmh",
    )
    all_codes: list[str] = []
    for key in (*solute_first_tables, "network_summary", "transport_summary"):
        df = results.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty and "system_code" in df.columns:
            for sc in df["system_code"].dropna().astype(str):
                if sc not in all_codes:
                    all_codes.append(sc)

    # Prefer non-PURE solute systems in master synthesis table; include PURE only if no solutes exist
    non_pure = [sc for sc in all_codes if sc.upper() != "PURE"]
    targets = non_pure if non_pure else all_codes
    for sc in targets:
        if sc not in ordered:
            ordered.append(sc)
    return ordered


def build_master_synthesis_table(results: dict[str, Any]) -> pd.DataFrame:
    cosmors = results.get("cosmors", pd.DataFrame())
    if not isinstance(cosmors, pd.DataFrame):
        cosmors = pd.DataFrame()

    enr = results.get("enrichment_summary", pd.DataFrame())
    kbi = results.get("kbi_summary", pd.DataFrame())
    trans = results.get("transport_summary", pd.DataFrame())
    hb = results.get("hbond_summary", pd.DataFrame())
    orient = results.get("orientation_summary", pd.DataFrame())
    net = results.get("network_summary", pd.DataFrame())
    strain_df = results.get("strain_summary", pd.DataFrame())
    if strain_df.empty and "xtb_strain" in results and not results["xtb_strain"].empty:
        strain_df = (
            results["xtb_strain"]
            .groupby("system_code", as_index=False)
            .agg(
                e_strain_relaxed_mean_kcal=("e_strain_relaxed_kcal", "mean"),
                e_strain_relaxed_sd=("e_strain_relaxed_kcal", "std"),
            )
        )
    dimer_df = results.get("dimer_xtb", results.get("xtb_dimers", pd.DataFrame()))
    trimer_df = results.get("trimer_xtb", results.get("xtb_trimers", pd.DataFrame()))
    dft_dimers = results.get("dft_dimers", pd.DataFrame())
    dft_trimers = results.get("dft_trimers", pd.DataFrame())
    dft_strain = results.get("dft_strain", pd.DataFrame())
    cp_val = results.get("cp_validation", pd.DataFrame())
    nci_df = results.get("nci_igmh", pd.DataFrame())
    led_df = results.get("led_eda", pd.DataFrame())

    pure_deg = float(net.loc[net["system_code"] == "PURE", "mean_degree"].mean()) if not net.empty and "PURE" in net["system_code"].values else float("nan")
    pure_giant = float(net.loc[net["system_code"] == "PURE", "giant_frac"].mean()) if not net.empty and "PURE" in net["system_code"].values else float("nan")

    system_codes = _discover_system_codes(results, cosmors)
    cosmo_map: dict[str, pd.Series] = {}
    if not cosmors.empty and "solute_code" in cosmors.columns:
        for _, crow in cosmors.iterrows():
            cosmo_map[str(crow["solute_code"])] = crow

    rows = []
    for sc in system_codes:
        entry: dict[str, Any] = {"system_code": sc}
        if sc in cosmo_map:
            crow = cosmo_map[sc]
            entry["solute_name"] = crow.get("solute_name", sc)
            if "ln_gamma_inf" in crow:
                entry["cosmors_ln_gamma_inf"] = float(crow["ln_gamma_inf"])
            if "capacity_c" in crow:
                entry["cosmors_capacity_c"] = float(crow["capacity_c"])
            for col in crow.index:
                if col not in {"solute_code", "solute_name", "ln_gamma_inf", "capacity_c"}:
                    entry[f"benchmark_{col}"] = crow[col]
            if "selectivity_ga_vs_ger" in crow:
                entry["cosmors_S_GA_GER"] = float(crow["selectivity_ga_vs_ger"])
            if "selectivity_ga_vs_lim" in crow:
                entry["cosmors_S_GA_LIM"] = float(crow["selectivity_ga_vs_lim"])

        def _pick(df: pd.DataFrame, col: str) -> float:
            if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns or "system_code" not in df.columns:
                return float("nan")
            sub = pd.to_numeric(df.loc[df["system_code"] == sc, col], errors="coerce").dropna()
            return float(sub.mean()) if not sub.empty else float("nan")

        # Dynamically flatten numeric summary columns from each MD/QM stage
        stage_sources = [
            (kbi, "kbi_"),
            (enr, "enrichment_"),
            (hb, "hbond_"),
            (orient, "orient_"),
            (net, "network_"),
            (trans, "transport_"),
        ]
        for df_src, prefix in stage_sources:
            if isinstance(df_src, pd.DataFrame) and not df_src.empty and "system_code" in df_src.columns:
                for col in df_src.columns:
                    if col in {"system_code", "n_rep", "primary_solvent", "replica", "run_id"}:
                        continue
                    if pd.api.types.is_numeric_dtype(df_src[col]):
                        val = _pick(df_src, col)
                        entry[f"{prefix}{col}"] = val

        # Clean aliases used across standard reports & legacy checks
        if "network_mean_degree" in entry:
            entry["network_delta_degree_vs_pure"] = entry["network_mean_degree"] - pure_deg if np.isfinite(pure_deg) else float("nan")
        if "network_giant_frac" in entry:
            entry["network_delta_giant_vs_pure"] = entry["network_giant_frac"] - pure_giant if np.isfinite(pure_giant) else float("nan")
        if not hb.empty and "sol_hbonds_per_mol_mean" in hb.columns:
            entry["hbond_per_solute"] = _pick(hb, "sol_hbonds_per_mol_mean")

        entry["strain_xtb_relaxed_kcal"] = _pick(strain_df, "e_strain_relaxed_mean_kcal")
        entry["strain_xtb_relaxed_sd"] = _pick(strain_df, "e_strain_relaxed_sd")
        entry["strain_dft_r2scan3c_kcal"] = _pick(dft_strain, "e_strain_r2scan3c_kcal")

        # Dynamic dimer interaction summary per solvent partner found in pairs
        if not dimer_df.empty and "system_code" in dimer_df.columns and "pair" in dimer_df.columns:
            sub_sys = dimer_df.loc[dimer_df["system_code"] == sc]
            for pair_val, grp in sub_sys.groupby("pair"):
                solv_tag = str(pair_val).split("-", 1)[1] if "-" in str(pair_val) else str(pair_val)
                vals = grp["de_int_kcal"].dropna()
                entry[f"xtb_de_int_{solv_tag}_mean_kcal"] = float(vals.mean()) if not vals.empty else float("nan")
                entry[f"xtb_de_int_{solv_tag}_sd_kcal"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            entry["xtb_de_int_mean_kcal"] = float(sub_sys["de_int_kcal"].mean()) if not sub_sys.empty else float("nan")

        if not dft_dimers.empty and "system_code" in dft_dimers.columns and "pair" in dft_dimers.columns:
            sub_sys = dft_dimers.loc[dft_dimers["system_code"] == sc]
            for pair_val, grp in sub_sys.groupby("pair"):
                solv_tag = str(pair_val).split("-", 1)[1] if "-" in str(pair_val) else str(pair_val)
                vals = grp["de_int_r2scan3c_kcal"].dropna()
                entry[f"dft_de_int_{solv_tag}_mean_kcal"] = float(vals.mean()) if not vals.empty else float("nan")
            entry["dft_de_int_mean_kcal"] = float(sub_sys["de_int_r2scan3c_kcal"].mean()) if not sub_sys.empty else float("nan")

        if not cp_val.empty and "system_code" in cp_val.columns:
            entry["cp_wb97xd3_de_kcal"] = _pick(cp_val, "de_wb97xd3_cp_kcal")
            entry["cp_bsse_kcal"] = _pick(cp_val, "bsse_kcal")

        if not trimer_df.empty and "system_code" in trimer_df.columns:
            t_sub = trimer_df.loc[trimer_df["system_code"] == sc, "e_3body_kcal"].dropna()
            entry["xtb_e_3body_mean_kcal"] = float(t_sub.mean()) if not t_sub.empty else float("nan")
            entry["xtb_e_3body_sd_kcal"] = float(t_sub.std(ddof=1)) if len(t_sub) > 1 else float("nan")
        if not dft_trimers.empty and "system_code" in dft_trimers.columns:
            t_sub = dft_trimers.loc[dft_trimers["system_code"] == sc, "e_3body_r2scan3c_kcal"].dropna()
            entry["dft_e_3body_mean_kcal"] = float(t_sub.mean()) if not t_sub.empty else float("nan")
            entry["dft_e_3body_sd_kcal"] = float(t_sub.std(ddof=1)) if len(t_sub) > 1 else float("nan")

        if not nci_df.empty and "system_code" in nci_df.columns:
            entry["igmh_frac_hb"] = _pick(nci_df, "igmh_frac_hb")
            entry["igmh_frac_vdw"] = _pick(nci_df, "igmh_frac_vdw")
            entry["igmh_frac_steric"] = _pick(nci_df, "igmh_frac_steric")
            entry["igmh_integral_dg_inter_au"] = _pick(nci_df, "integral_dg_inter_au")
            entry["igmh_integral_dg_intra_au"] = _pick(nci_df, "integral_dg_intra_au")

        if not led_df.empty and "system_code" in led_df.columns:
            entry["led_elstat_kcal"] = _pick(led_df, "e_elstat_kcal")
            entry["led_exch_kcal"] = _pick(led_df, "e_exch_kcal")
            entry["led_disp_kcal"] = _pick(led_df, "e_disp_kcal")
            entry["led_prep_kcal"] = _pick(led_df, "e_prep_kcal")
            entry["led_total_kcal"] = _pick(led_df, "e_total_kcal")
            if "label" in led_df.columns:
                l_sys = led_df.loc[led_df["system_code"] == sc]
                for lbl, lsub in l_sys.groupby("label"):
                    solv_tag = str(lbl).split("-", 1)[1] if "-" in str(lbl) else str(lbl)
                    entry[f"led_elstat_{solv_tag}_kcal"] = float(lsub["e_elstat_kcal"].mean())
                    entry[f"led_disp_{solv_tag}_kcal"] = float(lsub["e_disp_kcal"].mean())
                    entry[f"led_total_{solv_tag}_kcal"] = float(lsub["e_total_kcal"].mean())

        rows.append(entry)

    return pd.DataFrame(rows)


def build_replicate_synthesis_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build per-replicate (R01, R02, ...) MD/KBI/GK synthesis table dynamically across all systems."""
    run_tables = [
        "equilibration",
        "kbi_systems",
        "enrichment_runs",
        "hbond_solute_runs",
        "orientation_runs",
        "network_runs",
        "gk_runs",
    ]
    key_cols = ["run_id", "system_code", "replica"]
    base: pd.DataFrame | None = None

    for tbl_name in run_tables:
        df = results.get(tbl_name, pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty or not all(k in df.columns for k in key_cols):
            continue
        num_cols = [
            c
            for c in df.columns
            if c not in key_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not num_cols:
            continue
        sub = df[key_cols + num_cols].drop_duplicates(subset=key_cols)
        if base is None:
            base = sub
        else:
            overlap = [c for c in num_cols if c in base.columns]
            if overlap:
                sub = sub.drop(columns=overlap)
            base = base.merge(sub, on=key_cols, how="outer")

    # Add pivoted species MSD diffusion per replicate if available
    msd_df = results.get("msd_runs", pd.DataFrame())
    if isinstance(msd_df, pd.DataFrame) and not msd_df.empty and all(k in msd_df.columns for k in (*key_cols, "species", "d_1e5_cm2_s")):
        piv = msd_df.pivot_table(index=key_cols, columns="species", values="d_1e5_cm2_s", aggfunc="mean")
        piv.columns = [f"d_{str(sp).lower()}_1e5_cm2_s" for sp in piv.columns]
        piv = piv.reset_index()
        base = piv if base is None else base.merge(piv, on=key_cols, how="outer")

    if base is None or base.empty:
        return pd.DataFrame()
    return base.sort_values(["system_code", "replica"]).reset_index(drop=True)
