from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from mody_analyse.analyzers.alchemical import run_alchemical_analysis
from mody_analyse.analyzers.conformation_geometry import (
    run_conformation_analysis,
    run_voronoi_free_volume_analysis,
)
from mody_analyse.analyzers.electro_dynamics import (
    run_dielectric_analysis,
    run_electrochemistry_analysis,
    run_glassy_dynamics_analysis,
    run_rotational_dynamics_analysis,
)
from mody_analyse.analyzers.equilibration import analyze_equilibration, analyze_equilibration_windows
from mody_analyse.analyzers.green_kubo import run_transport_analysis
from mody_analyse.analyzers.interface_slab import run_interface_slab_analysis
from mody_analyse.analyzers.kbi_analyzer import run_kbi_analysis
from mody_analyse.analyzers.kbi_extended import (
    run_kbi_composition_series,
    run_kbi_extended_analysis,
    run_kbi_temperature_series,
)
from mody_analyse.analyzers.md_structural import (
    analyze_hbonds,
    analyze_orientation_and_sdf,
    analyze_solvation_shells,
    analyze_solvent_network,
)
from mody_analyse.analyzers.spectroscopy import run_spectroscopy_analysis
from mody_analyse.analyzers.thermo_bulk import run_thermo_bulk_analysis
from mody_analyse.core.config import Config, load_config
from mody_analyse.core.discovery import discover_runs
from mody_analyse.core.registry import (
    ANALYSIS_PROFILES,
    evaluate_eligibility,
    export_catalog_payload,
    resolve_selected_analyses,
)
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


STAGE_CHOICES = [
    "all",
    "equil",
    "thermo",
    "kbi",
    "gk",
    "electrolyte",
    "dynamics",
    "spectroscopy",
    "md",
    "structure",
    "conformation",
    "interface",
    "alchemical",
    "extended",
    "extract",
    "xtb",
    "orca",
    "nci",
    "synth",
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mody-analyse",
        description="MoDy Analyse: Modular MD-KBI-GK, Thermo/Electrolyte/Spectroscopy/FEP, and MD-to-QM Workflow",
    )
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="Run fast pilot subset (first pure + first solute R01 run)")
    mode.add_argument("--all", action="store_true", help="Run full pipeline on all discovered replicates")
    p.add_argument(
        "--stage",
        default="all",
        choices=STAGE_CHOICES,
        help="Run a specific stage group (default: all active modules in profile/checklist)",
    )
    p.add_argument(
        "--profile",
        default=None,
        choices=sorted(ANALYSIS_PROFILES.keys()),
        help="Named analysis profile overriding analyses.profile in config.yaml",
    )
    p.add_argument(
        "--analyses",
        default=None,
        help="Comma-separated list of explicit analysis module IDs (e.g. equil,thermo_fluctuations,kbi_full)",
    )
    p.add_argument(
        "--list-analyses",
        action="store_true",
        help="Print JSON catalog of all available analysis modules, profiles, and physics prerequisites, then exit",
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


def execute_pipeline(
    cfg: Config,
    pilot: bool = False,
    stage: str = "all",
    profile: str | None = None,
    analyses: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
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
    tables_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = _load_cached_tables(tables_dir)
    bench_df = load_benchmark_reference(cfg)
    if not bench_df.empty:
        results["cosmors"] = bench_df
    elif "cosmors" not in results:
        results["cosmors"] = pd.DataFrame()
    all_topos = {k: v for r in runs for k, v in r.topologies.items()}

    selected_ids = resolve_selected_analyses(
        cfg.analyses,
        stage=stage,
        profile_override=profile,
        analyses_override=analyses,
    )
    decisions, gate_df = evaluate_eligibility(runs, cfg, selected_ids)
    results["analysis_gatekeeper_report"] = gate_df
    _save_tables({"analysis_gatekeeper_report": gate_df}, tables_dir)
    (tables_dir / "analysis_gatekeeper_report.json").write_text(
        json.dumps([d.to_row() for d in decisions.values()], indent=2),
        encoding="utf-8",
    )

    for mid in selected_ids:
        dec = decisions[mid]
        if dec.status == "SKIPPED":
            print(f"[Gatekeeper] SKIPPED {mid}: {dec.reason}")
        elif dec.warnings:
            print(f"[Gatekeeper] WARN {mid}: {'; '.join(dec.warnings)}")

    def _active(mid: str) -> bool:
        return mid in decisions and decisions[mid].status == "RUN"

    if _active("equil"):
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

    if _active("thermo_fluctuations"):
        print("[Stage] Bulk NPT Fluctuations (Cp, κT, αP, Cv) & Solubility Parameters...")
        th_res = run_thermo_bulk_analysis(runs, cfg)
        results.update(th_res)
        _save_tables(th_res, tables_dir)

    if _active("kbi"):
        print("[Stage] KBI Analyzer (GvdV + Kruger)...")
        kbi_res = run_kbi_analysis(runs, cfg, verify_mda=True)
        results.update(kbi_res)
        _save_tables(kbi_res, tables_dir)

    if _active("kbi_full"):
        print("[Stage] Full KBI Suite (Dawass 1/R, Sub-Volume, Bhatia–Thornton, S(q)/SAXS, Block SEM)...")
        kbi_ext = run_kbi_extended_analysis(runs, cfg, base_kbi_results=results)
        results.update(kbi_ext)
        _save_tables(kbi_ext, tables_dir)

    if _active("kbi_composition_series"):
        print("[Stage] Composition-Series KBI Integration (ln γi, GE, Δμex)...")
        comp_res = run_kbi_composition_series(runs, cfg, bt_df=results.get("kbi_bhatia_thornton"))
        results.update(comp_res)
        _save_tables(comp_res, tables_dir)

    if _active("kbi_temperature_series"):
        print("[Stage] Temperature-Series Gibbs–Helmholtz (HE, TSE)...")
        t_res = run_kbi_temperature_series(results.get("kbi_activity_excess_gibbs"))
        results.update(t_res)
        _save_tables(t_res, tables_dir)

    if _active("gk"):
        print("[Stage] Green-Kubo & MSD Transport Analyzer...")
        gk_res = run_transport_analysis(runs, cfg, compute_msd=True)
        results.update(gk_res)
        _save_tables(gk_res, tables_dir)

    if _active("electrochemistry_onsager"):
        print("[Stage] Electrochemistry & Onsager Transport (σ_EH, σ_NE, Haven, Lij, t+)...")
        el_res = run_electrochemistry_analysis(runs, cfg)
        results.update(el_res)
        _save_tables(el_res, tables_dir)

    if _active("rotational_dynamics"):
        print("[Stage] Rotational Reorientation Dynamics (C1, C2, KWW, Debye ratio)...")
        rot_res = run_rotational_dynamics_analysis(runs, cfg)
        results.update(rot_res)
        _save_tables(rot_res, tables_dir)

    if _active("glassy_dynamics"):
        print("[Stage] Glassy / Heterogeneous Dynamics (α2(t), Fs(q*,t), van Hove)...")
        gl_res = run_glassy_dynamics_analysis(runs, cfg)
        results.update(gl_res)
        _save_tables(gl_res, tables_dir)

    if _active("dielectric"):
        print("[Stage] Static Dielectric Constant (ε0) & Kirkwood gK...")
        diel_res = run_dielectric_analysis(runs, cfg)
        results.update(diel_res)
        _save_tables(diel_res, tables_dir)

    if _active("spectroscopy_vdos_ir"):
        print("[Stage] Spectroscopy (VDOS & IR / Far-IR / THz)...")
        sp_res = run_spectroscopy_analysis(runs, cfg)
        results.update(sp_res)
        _save_tables(sp_res, tables_dir)

    if _active("md_solvation"):
        print("[Stage] MD Structural #1: Solvation shell & local enrichment...")
        s1 = analyze_solvation_shells(runs, cfg)
        results.update(s1)
        _save_tables(s1, tables_dir)

    if _active("md_hbond"):
        print("[Stage] MD Structural #2: Hydrogen-bond population & lifetime...")
        s2 = analyze_hbonds(runs, cfg)
        results.update(s2)
        _save_tables(s2, tables_dir)

    if _active("md_sdf"):
        print("[Stage] MD Structural #3: Orientational & 3D spatial distribution...")
        s3 = analyze_orientation_and_sdf(runs, cfg)
        results.update(s3)
        _save_tables(s3, tables_dir)

    if _active("md_network"):
        print("[Stage] MD Structural #4: Solvent H-bond network rearrangement...")
        s4 = analyze_solvent_network(runs, cfg)
        results.update(s4)
        _save_tables(s4, tables_dir)

    if _active("voronoi_free_volume"):
        print("[Stage] Periodic 3D Voronoi Tessellation & Free Volume...")
        vor_res = run_voronoi_free_volume_analysis(runs, cfg)
        results.update(vor_res)
        _save_tables(vor_res, tables_dir)

    if _active("conformation_sasa_fes"):
        print("[Stage] Solute Conformation (SASA, Rg, Dihedral FES, PCA Clustering)...")
        cf_res = run_conformation_analysis(runs, cfg)
        results.update(cf_res)
        _save_tables(cf_res, tables_dir)

    if _active("interface_slab"):
        print("[Stage] Interface / Slab Analysis (Surface Tension γ, z-Profiles, P2)...")
        if_res = run_interface_slab_analysis(runs, cfg)
        results.update(if_res)
        _save_tables(if_res, tables_dir)

    if _active("alchemical_free_energy"):
        print("[Stage] Alchemical Free Energy (TI, BAR, MBAR + Overlap Matrix)...")
        al_res = run_alchemical_analysis(runs, cfg)
        results.update(al_res)
        _save_tables(al_res, tables_dir)

    if _active("extract"):
        print("[Stage] PBC-Safe Cluster Extraction (#5-#7)...")
        sol_man = extract_solute_snapshots(runs, cfg)
        dim_man = extract_stratified_dimers(runs, cfg)
        tri_man = extract_trimers(runs, cfg)
        results.update({"strain_manifest": sol_man, "dimer_manifest": dim_man, "trimer_manifest": tri_man})
        _save_tables({"strain_manifest": sol_man, "dimer_manifest": dim_man, "trimer_manifest": tri_man}, tables_dir)

    if _active("xtb"):
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

    if _active("orca"):
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

    if _active("nci"):
        print("[Stage] Multiwfn NCI / IGMH & VMD (#8)...")
        nci_df = run_multiwfn_nci_igmh(results.get("dft_dimers", pd.DataFrame()), results.get("dft_trimers", pd.DataFrame()), cfg)
        results["nci_igmh"] = nci_df
        _save_tables({"nci_igmh": nci_df}, tables_dir)

    if _active("synth"):
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
    if args.list_analyses:
        print(json.dumps(export_catalog_payload(), indent=2))
        return
    cfg = load_config(args.config)
    analyses_list = [x.strip() for x in args.analyses.split(",") if x.strip()] if args.analyses else None
    execute_pipeline(
        cfg,
        pilot=args.pilot,
        stage=args.stage,
        profile=args.profile,
        analyses=analyses_list,
    )


if __name__ == "__main__":
    main()
