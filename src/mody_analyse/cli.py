from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import pandas as pd

from mody_analyse.analyzers.equilibration import analyze_equilibration, analyze_equilibration_windows
from mody_analyse.analyzers.green_kubo import run_transport_analysis
from mody_analyse.analyzers.kbi_analyzer import run_kbi_analysis
from mody_analyse.analyzers.md_structural import (
    analyze_hbonds,
    analyze_orientation_and_sdf,
    analyze_solvation_shells,
    analyze_solvent_network,
)
from mody_analyse.core.config import Config, load_config
from mody_analyse.core.discovery import discover_runs
from mody_analyse.md_to_qm.cluster_extractor import (
    extract_solute_snapshots,
    extract_stratified_dimers,
    extract_trimers,
)
from mody_analyse.qm_runners.nci_igmh import run_multiwfn_nci_igmh
from mody_analyse.qm_runners.qm_orca import (
    run_orca_dft_dimers_and_trimers,
    run_orca_led_eda,
    select_representative_dft_clusters,
)
from mody_analyse.qm_runners.qm_xtb import (
    generate_xtb_reference_conformers,
    run_xtb_dimers,
    run_xtb_strain,
    run_xtb_trimers,
)
from mody_analyse.reporting.synthesis import (
    build_master_synthesis_table,
    build_replicate_synthesis_table,
    load_benchmark_reference,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mody-analyse",
        description="MoDy Analyse: Modular MD-KBI-GK and MD-to-QM Analysis Workflow",
    )
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="Run fast pilot subset (first pure + first solute R01 run)")
    mode.add_argument("--all", action="store_true", help="Run full pipeline on all discovered replicates")
    p.add_argument(
        "--stage",
        default="all",
        choices=["all", "equil", "kbi", "gk", "md", "extract", "xtb", "orca", "nci", "synth"],
    )
    return p


def _save_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=False)


def _load_cached_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    if tables_dir.exists():
        for p in tables_dir.glob("*.csv"):
            loaded[p.stem] = pd.read_csv(p)
    return loaded


def execute_pipeline(cfg: Config, pilot: bool = False, stage: str = "all") -> dict[str, Any]:
    t0 = time.time()
    all_runs = discover_runs(cfg)
    if pilot:
        r01_runs = [r for r in all_runs if r.replica == "R01"] or all_runs
        picked: list[Any] = []
        pure_pick = next((r for r in r01_runs if r.solute_resname is None), None)
        if pure_pick is not None:
            picked.append(pure_pick)
        sol_pick = next((r for r in r01_runs if r.solute_resname is not None), None)
        if sol_pick is not None:
            picked.append(sol_pick)
        runs = picked or r01_runs[:2]
        cfg = Config(
            **{
                **cfg.__dict__,
                "results_dir": cfg.workflow_root / "results_pilot",
                "strain_snapshots_per_rep": 4,
                "strain_ref_conformers": 10,
                "dimers_per_pair_per_rep": 4,
                "trimers_per_system_per_rep": 3,
                "dft_dimers_per_pair": 1,
                "cp_validation_dimers_per_pair": 1,
                "dft_trimers_per_solute": 1,
                "nci_clusters_total": 2,
                "led_dimers_total": 1,
            }
        )
    else:
        runs = all_runs

    tables_dir = cfg.results_dir / "tables"
    results: dict[str, Any] = _load_cached_tables(tables_dir)
    bench_df = load_benchmark_reference(cfg)
    if not bench_df.empty:
        results["cosmors"] = bench_df
    elif "cosmors" not in results:
        results["cosmors"] = pd.DataFrame()
    all_topos = {k: v for r in runs for k, v in r.topologies.items()}

    def _run_stage(name: str) -> bool:
        return stage in {"all", name}

    if _run_stage("equil"):
        print("[Stage] Equilibration check...")
        results["equilibration"] = analyze_equilibration(runs, cfg)
        results["equilibration_windows"] = analyze_equilibration_windows(runs)
        _save_tables(
            {
                "equilibration": results["equilibration"],
                "equilibration_windows": results["equilibration_windows"],
            },
            tables_dir,
        )

    if _run_stage("kbi"):
        print("[Stage] KBI Analyzer (GvdV + Kruger)...")
        kbi_res = run_kbi_analysis(runs, cfg, verify_mda=True)
        results.update(kbi_res)
        _save_tables(kbi_res, tables_dir)

    if _run_stage("gk"):
        print("[Stage] Green-Kubo & MSD Transport Analyzer...")
        gk_res = run_transport_analysis(runs, cfg, compute_msd=True)
        results.update(gk_res)
        _save_tables(gk_res, tables_dir)

    if _run_stage("md"):
        print("[Stage] MD Structural Analyses #1-#4...")
        print("  -> #1 Solvation shell & local enrichment...")
        s1 = analyze_solvation_shells(runs, cfg)
        results.update(s1)
        _save_tables(s1, tables_dir)
        print("  -> #2 Hydrogen-bond population & lifetime...")
        s2 = analyze_hbonds(runs, cfg)
        results.update(s2)
        _save_tables(s2, tables_dir)
        print("  -> #3 Orientational & 3D spatial distribution...")
        s3 = analyze_orientation_and_sdf(runs, cfg)
        results.update(s3)
        _save_tables(s3, tables_dir)
        print("  -> #4 Solvent H-bond network rearrangement...")
        s4 = analyze_solvent_network(runs, cfg)
        results.update(s4)
        _save_tables(s4, tables_dir)

    if _run_stage("extract"):
        print("[Stage] PBC-Safe Cluster Extraction (#5-#7)...")
        sol_man = extract_solute_snapshots(runs, cfg)
        dim_man = extract_stratified_dimers(runs, cfg)
        tri_man = extract_trimers(runs, cfg)
        results.update({"strain_manifest": sol_man, "dimer_manifest": dim_man, "trimer_manifest": tri_man})
        _save_tables({"strain_manifest": sol_man, "dimer_manifest": dim_man, "trimer_manifest": tri_man}, tables_dir)

    if _run_stage("xtb"):
        print("[Stage] GFN2-xTB Screening (#5-#7)...")
        ref_df = generate_xtb_reference_conformers(all_topos, results["strain_manifest"], cfg)
        xtb_strain = run_xtb_strain(results["strain_manifest"], ref_df, all_topos, cfg)
        xtb_dimers = run_xtb_dimers(results["dimer_manifest"], cfg)
        xtb_trimers = run_xtb_trimers(results["trimer_manifest"], cfg)
        strain_summary = (
            xtb_strain.groupby("system_code", as_index=False)
            .agg(
                e_strain_relaxed_mean_kcal=("e_strain_relaxed_kcal", "mean"),
                e_strain_relaxed_sd=("e_strain_relaxed_kcal", "std"),
                e_strain_frozen_mean_kcal=("e_strain_frozen_kcal", "mean"),
            )
            if not xtb_strain.empty
            else pd.DataFrame()
        )
        results.update(
            {
                "xtb_ref_conformers": ref_df,
                "xtb_strain": xtb_strain,
                "strain_summary": strain_summary,
                "xtb_dimers": xtb_dimers,
                "dimer_xtb": xtb_dimers,
                "xtb_trimers": xtb_trimers,
                "trimer_xtb": xtb_trimers,
            }
        )
        _save_tables(
            {
                "xtb_ref_conformers": ref_df,
                "xtb_strain": xtb_strain,
                "strain_summary": strain_summary,
                "xtb_dimers": xtb_dimers,
                "xtb_trimers": xtb_trimers,
            },
            tables_dir,
        )

    if _run_stage("orca"):
        print("[Stage] ORCA DFT-D, CP & DLPNO-CCSD(T) LED (#6, #7, #9)...")
        sel = select_representative_dft_clusters(results["xtb_dimers"], results["xtb_trimers"], results["xtb_strain"], cfg)
        if pilot:
            if not sel["cp_dimers"].empty:
                sel["cp_dimers"] = (
                    sel["cp_dimers"]
                    .assign(_nat=sel["cp_dimers"]["n_atoms_a"] + sel["cp_dimers"]["n_atoms_b"])
                    .sort_values("_nat")
                    .drop(columns=["_nat"])
                    .head(1)
                )
            if not sel["dft_strain"].empty:
                sel["dft_strain"] = sel["dft_strain"].head(1)
        orca_res = run_orca_dft_dimers_and_trimers(sel, results["xtb_ref_conformers"], cfg)
        results.update(orca_res)
        _save_tables(orca_res, tables_dir)
        led_df = run_orca_led_eda(orca_res["dft_dimers"], cfg)
        orca_res["led_eda"] = led_df
        results["led_eda"] = led_df
        _save_tables({"led_eda": led_df}, tables_dir)

    if _run_stage("nci"):
        print("[Stage] Multiwfn NCI / IGMH & VMD (#8)...")
        nci_df = run_multiwfn_nci_igmh(results.get("dft_dimers", pd.DataFrame()), results.get("dft_trimers", pd.DataFrame()), cfg)
        results["nci_igmh"] = nci_df
        _save_tables({"nci_igmh": nci_df}, tables_dir)

    if _run_stage("synth"):
        from mody_analyse.reporting.plotting import generate_all_figures

        print("[Stage] Analysis #10 Synthesis & Publication Figures...")
        master_df = build_master_synthesis_table(results)
        rep_df = build_replicate_synthesis_table(results)
        results["master_synthesis"] = master_df
        results["replicate_synthesis"] = rep_df
        save_map: dict[str, pd.DataFrame] = {
            "master_synthesis": master_df,
            "replicate_synthesis": rep_df,
        }
        if isinstance(results.get("cosmors"), pd.DataFrame) and not results["cosmors"].empty:
            save_map["cosmors"] = results["cosmors"]
        _save_tables(save_map, tables_dir)
        fig_dir = cfg.results_dir / "figures"
        figs = generate_all_figures(results, fig_dir)
        print(f"Generated {len(figs)} figures in {fig_dir}")

    print(f"Done in {time.time() - t0:.1f}s")
    return results


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    execute_pipeline(cfg, pilot=args.pilot, stage=args.stage)


if __name__ == "__main__":
    main()
