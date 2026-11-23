from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalysisSpec:
    id: str
    category: str
    title: str
    description: str
    requirements: tuple[str, ...]
    stage_group: str
    default_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "requirements": list(self.requirements),
        }


ANALYSIS_REGISTRY: tuple[AnalysisSpec, ...] = (
    # --- Core QC ---
    AnalysisSpec(
        id="equil",
        category="qc",
        title="Equilibration QC & Time-Window Convergence",
        description="Density, temperature, pressure, potential energy stability and dynamic window convergence.",
        requirements=("thermo.xvg in run analysis directory",),
        stage_group="equil",
        default_enabled=True,
    ),
    # --- Bulk Thermodynamics ---
    AnalysisSpec(
        id="thermo_fluctuations",
        category="thermo",
        title="NPT Fluctuations (Cp, κT, αP, Cv) & Solubility Parameters (CED / Hildebrand / Hansen)",
        description="Extracts Cp, isothermal compressibility κT, thermal expansion αP, Cv, and cohesive energy density / Hildebrand & Hansen parameters from NPT fluctuations.",
        requirements=(
            "Equilibrated NPT thermo.xvg with Temperature, Volume, and Enthalpy/Potential/Total-Energy columns",
            "Optional gas-phase single-molecule energy (analyses.options.gas_energies_kj_mol) for exact ΔHvap & CED",
        ),
        stage_group="thermo",
        default_enabled=True,
    ),
    # --- Kirkwood-Buff Suite ---
    AnalysisSpec(
        id="kbi",
        category="kbi",
        title="Core Kirkwood–Buff Integrals (GvdV + Krüger + N-Component Inversion)",
        description="Ganguly–van der Vegt tail-corrected RDFs, Krüger finite-size KBI extrapolation, and constant-(T,P) N-component matrix inversion (PMVs, Γ).",
        requirements=("00_input/topol.top and either rdf_*.xvg or production trajectory (prod.tpr + prod.xtc)",),
        stage_group="kbi",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="kbi_full",
        category="kbi",
        title="Full KBI Suite (Dawass 1/R, Sub-Volume Fluctuations, Bhatia–Thornton, S(q)/SAXS-SANS, Block SEM)",
        description="Adds Dawass 1/R thermodynamic-limit extrapolation with R² convergence QC, direct spherical sub-volume <ΔNiΔNj> KBI, Bhatia–Thornton S_NN/S_CC/S_NC, q-dependent S(q) & SAXS/SANS intensity, preferential solvation, κT^KB, and block-averaged uncertainties.",
        requirements=("00_input/topol.top and trajectory (or rdf_*.xvg for Dawass/Bhatia–Thornton/S(q))",),
        stage_group="kbi",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="kbi_composition_series",
        category="kbi",
        title="Composition-Series KBI Integration (ln γi, GE, Δμex, Setschenow KS)",
        description="Integrates thermodynamic factor Γ(x) across a composition sweep (with Redlich–Kister fit & Gibbs–Duhem consistency check) to obtain activity coefficients ln γi, excess Gibbs energy GE, and Setschenow salting-out/cosolvency constants.",
        requirements=(
            "Multi-composition dataset: at least 3 distinct mole-fraction compositions across runs (>=5 recommended)",
            "Completed KBI results for each composition point",
        ),
        stage_group="kbi",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="kbi_temperature_series",
        category="kbi",
        title="Temperature-Series Gibbs–Helmholtz (HE, TSE)",
        description="Computes excess enthalpy HE and excess entropy TSE via Gibbs–Helmholtz differentiation across multi-temperature runs.",
        requirements=("At least 2 distinct temperatures across runs (>=3 recommended)",),
        stage_group="kbi",
        default_enabled=True,
    ),
    # --- Transport & Electrochemistry ---
    AnalysisSpec(
        id="gk",
        category="transport",
        title="Green–Kubo Viscosity, Einstein MSD Diffusion & Stokes–Einstein",
        description="Rotationally invariant shear viscosity from stress ACF (with ballistic artifact guard), unwrapped COM self-diffusion, and species-specific Stokes–Einstein ratios.",
        requirements=("stress.xvg for Green–Kubo viscosity and/or prod.tpr + prod.xtc for MSD self-diffusion",),
        stage_group="gk",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="electrochemistry_onsager",
        category="transport",
        title="Electrochemistry & Onsager Transport (σ_EH, σ_NE, Haven Ratio, Lij, Transference t+)",
        description="Full collective Einstein–Helfand ionic conductivity, Nernst–Einstein limit, Haven ratio & ionicity (Walden product), full Onsager transport matrix Lij (self + distinct), and true vs ideal transference numbers.",
        requirements=(
            "At least one ionic species with non-zero net molecular charge (|q_mol| > 1e-3 e) in .itp topology",
            "Unwrapped COM trajectory (prod.tpr + prod.xtc)",
        ),
        stage_group="electrolyte",
        default_enabled=True,
    ),
    # --- Rotational, Glassy & Dielectric Dynamics ---
    AnalysisSpec(
        id="rotational_dynamics",
        category="dynamics",
        title="Rotational Reorientation Dynamics (C1(t), C2(t), τ1, τ2, KWW β, NMR/Debye Ratio)",
        description="Legendre P1 and P2 orientational autocorrelation functions, KWW stretched-exponential & bi-exponential correlation times τ1, τ2, and Debye τ1/τ2 jump-reorientation ratio.",
        requirements=("Trajectory (prod.tpr + prod.xtc) containing at least one polyatomic species (>=2 atoms)",),
        stage_group="dynamics",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="glassy_dynamics",
        category="dynamics",
        title="Glassy & Heterogeneous Dynamics (α2(t), Fs(q*,t), van Hove Gs(r,t), Cage Residence)",
        description="Non-Gaussian parameter α2(t), self-intermediate scattering function Fs(q*,t) with α-relaxation time τα, self van Hove distribution Gs(r,t), and first-shell cage residence time.",
        requirements=("Trajectory (prod.tpr + prod.xtc) with at least 10 equilibrated frames",),
        stage_group="dynamics",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="dielectric",
        category="dynamics",
        title="Static Dielectric Constant (ε0) & Kirkwood gK Factor",
        description="Static relative permittivity ε0 from total box dipole fluctuations (conducting PME boundary) and Kirkwood dipole correlation factor gK, with running convergence and ionic-rotational flagging.",
        requirements=(
            "Atomic partial charges in .itp topology and trajectory (prod.tpr + prod.xtc)",
            "Long production window (>=10 ns recommended for quantitative ε0 convergence)",
        ),
        stage_group="dynamics",
        default_enabled=True,
    ),
    # --- Spectroscopy ---
    AnalysisSpec(
        id="spectroscopy_vdos_ir",
        category="spectroscopy",
        title="Vibrational Density of States (VDOS) & IR / Far-IR / THz Spectrum",
        description="Mass-weighted VACF power spectrum (VDOS) + VACF diffusivity check, and dipole-derivative autocorrelation IR/Far-IR/THz spectrum with harmonic quantum correction.",
        requirements=(
            "Fine frame output spacing (dt <= 0.05 ps required; <= 0.005 ps recommended for full mid-IR up to 3300 cm^-1)",
            "Atomic velocities (.trr) for VDOS; atomic charges in .itp for IR spectrum",
        ),
        stage_group="spectroscopy",
        default_enabled=False,
    ),
    # --- Atomistic Structure ---
    AnalysisSpec(
        id="md_solvation",
        category="structure",
        title="Solvation Shell Coordination & N-Species Local Enrichment",
        description="First-shell cutoff detection, coordination numbers, molar & volume-weighted local enrichment across N solvents, and site-specific RDFs.",
        requirements=("Trajectory (prod.tpr + prod.xtc) and .itp topology",),
        stage_group="md",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="md_hbond",
        category="structure",
        title="Hydrogen-Bond Population, Residence ACF & Angle Sensitivity",
        description="Intermittent H-bond counts, donor→acceptor survival ACFs (bi-exponential fit), and geometric angle sensitivity.",
        requirements=("Trajectory (prod.tpr + prod.xtc) and .itp topology",),
        stage_group="md",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="md_sdf",
        category="structure",
        title="Molecular Local-Frame 3D Spatial Distribution Functions (OpenDX + VMD)",
        description="Local-frame aligned 3D spatial density grids (.dx) and 1D/2D radial-angular probability distributions.",
        requirements=("Trajectory (prod.tpr + prod.xtc) and polyatomic reference species",),
        stage_group="md",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="md_network",
        category="structure",
        title="Collective H-Bond Graph Percolation, Rings & Shell-vs-Bulk Degree",
        description="Graph-theoretic H-bond network percolation, cyclomatic ring counts, largest cluster fraction, and shell-vs-bulk connectivity shifts.",
        requirements=("Trajectory (prod.tpr + prod.xtc) and .itp topology",),
        stage_group="md",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="voronoi_free_volume",
        category="structure",
        title="Periodic Voronoi Tessellation & Free-Volume Cavity Distribution",
        description="3D periodic Voronoi cell volumes, asphericity, Voronoi coordination numbers, and grid-probed void/cavity radius distribution.",
        requirements=("Trajectory (prod.tpr + prod.xtc) with 3D periodic box",),
        stage_group="structure",
        default_enabled=False,
    ),
    # --- Conformation & Free Energy ---
    AnalysisSpec(
        id="conformation_sasa_fes",
        category="conformation",
        title="Solute Conformation: SASA (Polar/Apolar), Rg, Dihedral FES & PCA Clustering",
        description="Shrake–Rupley polar/apolar SASA, radius of gyration Rg, 1D/2D dihedral free-energy surfaces (-kBT ln P), and conformational PCA + K-Means basin clustering.",
        requirements=("Trajectory (prod.tpr + prod.xtc) and polyatomic solute/species (>=4 atoms for dihedrals)",),
        stage_group="conformation",
        default_enabled=False,
    ),
    AnalysisSpec(
        id="interface_slab",
        category="interface",
        title="Interface / Slab Analysis (Surface Tension γ, z-Density Tanh Fit, Orientation P2)",
        description="Surface tension γ from pressure tensor anisotropy, 1D z-density profiles with Gibbs dividing surface & interfacial width fit, and z-resolved orientational order P2(cos θ).",
        requirements=(
            "Slab / interfacial box geometry (Lz/max(Lx,Ly) >= 1.5, vacuum gap detected, or analyses.options.force_slab=true)",
            "stress.xvg / thermo.xvg with Pres-XX/YY/ZZ for surface tension; trajectory for z-profiles",
        ),
        stage_group="interface",
        default_enabled=False,
    ),
    AnalysisSpec(
        id="alchemical_free_energy",
        category="free_energy",
        title="Alchemical Free Energy (ΔGsolv via TI, BAR & MBAR + Overlap Matrix)",
        description="Parses multi-window GROMACS dhdl.xvg files and computes ΔG via Thermodynamic Integration (TI), Bennett Acceptance Ratio (BAR), and Multi-state BAR (MBAR) with overlap matrix QC.",
        requirements=("Multi-window λ files (dhdl*.xvg) inside run or study alchemical subdirectories",),
        stage_group="alchemical",
        default_enabled=False,
    ),
    # --- MD-to-QM Pipeline ---
    AnalysisSpec(
        id="extract",
        category="md_to_qm",
        title="PBC-Safe Cluster Extraction (Solute Strain, Stratified Dimers, Trimers)",
        description="BFS covalent bond unwrapping and minimum-image cluster extraction for downstream QM calculations.",
        requirements=("Trajectory (prod.tpr + prod.xtc) and .itp topology",),
        stage_group="extract",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="xtb",
        category="md_to_qm",
        title="GFN2-xTB Screening (Strain, Dimers, 3-Body Trimers)",
        description="Automated GFN2-xTB reference conformers, conformational strain, dimer interaction, and 3-body cooperativity.",
        requirements=("xtb executable installed and completed extract stage",),
        stage_group="xtb",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="orca",
        category="md_to_qm",
        title="ORCA DFT-D, Counterpoise BSSE & DLPNO-CCSD(T) LED",
        description="High-level ORCA r2SCAN-3c, wB97X-D3 CP-BSSE, and DLPNO-CCSD(T) Local Energy Decomposition with RAM watchdog & MPI fallback.",
        requirements=("orca executable installed and completed xtb stage",),
        stage_group="orca",
        default_enabled=True,
    ),
    AnalysisSpec(
        id="nci",
        category="md_to_qm",
        title="Multiwfn NCI / IGMH & VMD Visualization Scripts",
        description="Non-covalent interaction (NCI) and Independent Gradient Model based on Hirshfeld partition (IGMH) topological analysis.",
        requirements=("Multiwfn_noGUI executable installed and completed orca stage",),
        stage_group="nci",
        default_enabled=True,
    ),
    # --- Reporting ---
    AnalysisSpec(
        id="synth",
        category="reporting",
        title="Master Multi-Replicate Synthesis & Publication Figures",
        description="Aggregates all active modules into mean ± SD multi-replicate tables, optional COSMO-RS/experimental benchmark comparisons, and publication PNG/PDF plots.",
        requirements=("At least one completed analysis table in results/tables/",),
        stage_group="synth",
        default_enabled=True,
    ),
)

SPEC_BY_ID: dict[str, AnalysisSpec] = {s.id: s for s in ANALYSIS_REGISTRY}
ALL_ANALYSIS_IDS: tuple[str, ...] = tuple(s.id for s in ANALYSIS_REGISTRY)

ANALYSIS_PROFILES: dict[str, tuple[str, ...]] = {
    "standard": (
        "equil",
        "thermo_fluctuations",
        "kbi",
        "kbi_full",
        "kbi_composition_series",
        "kbi_temperature_series",
        "gk",
        "md_solvation",
        "md_hbond",
        "md_sdf",
        "md_network",
        "extract",
        "xtb",
        "orca",
        "nci",
        "synth",
    ),
    "quick_qc": (
        "equil",
        "thermo_fluctuations",
        "kbi",
        "kbi_full",
        "synth",
    ),
    "full_kbi_thermo": (
        "equil",
        "thermo_fluctuations",
        "kbi",
        "kbi_full",
        "kbi_composition_series",
        "kbi_temperature_series",
        "synth",
    ),
    "electrolyte_des": (
        "equil",
        "thermo_fluctuations",
        "kbi",
        "kbi_full",
        "kbi_composition_series",
        "gk",
        "electrochemistry_onsager",
        "rotational_dynamics",
        "glassy_dynamics",
        "dielectric",
        "md_solvation",
        "md_hbond",
        "md_network",
        "synth",
    ),
    "spectroscopy": (
        "equil",
        "spectroscopy_vdos_ir",
        "rotational_dynamics",
        "dielectric",
        "synth",
    ),
    "interface_slab": (
        "equil",
        "thermo_fluctuations",
        "interface_slab",
        "md_hbond",
        "synth",
    ),
    "conformation_free_energy": (
        "equil",
        "conformation_sasa_fes",
        "voronoi_free_volume",
        "alchemical_free_energy",
        "md_solvation",
        "synth",
    ),
    "md_only_everything": tuple(
        s.id for s in ANALYSIS_REGISTRY if s.category != "md_to_qm"
    ),
    "everything": ALL_ANALYSIS_IDS,
}


def resolve_selected_analyses(
    cfg_analyses: dict[str, Any] | None = None,
    *,
    stage: str = "all",
    profile_override: str | None = None,
    analyses_override: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Resolve ordered list of requested analysis module IDs from CLI overrides + YAML config."""
    cfg_a = dict(cfg_analyses or {})

    if analyses_override:
        selected = [a.strip() for a in analyses_override if a and a.strip()]
    elif stage != "all":
        stage_map = {
            "equil": ["equil"],
            "thermo": ["thermo_fluctuations"],
            "kbi": ["kbi", "kbi_full", "kbi_composition_series", "kbi_temperature_series"],
            "gk": ["gk"],
            "electrolyte": ["electrochemistry_onsager"],
            "dynamics": ["rotational_dynamics", "glassy_dynamics", "dielectric"],
            "spectroscopy": ["spectroscopy_vdos_ir"],
            "md": ["md_solvation", "md_hbond", "md_sdf", "md_network"],
            "structure": ["md_solvation", "md_hbond", "md_sdf", "md_network", "voronoi_free_volume"],
            "conformation": ["conformation_sasa_fes", "voronoi_free_volume"],
            "interface": ["interface_slab"],
            "alchemical": ["alchemical_free_energy"],
            "extended": [
                "thermo_fluctuations",
                "kbi_full",
                "kbi_composition_series",
                "kbi_temperature_series",
                "electrochemistry_onsager",
                "rotational_dynamics",
                "glassy_dynamics",
                "dielectric",
                "spectroscopy_vdos_ir",
                "voronoi_free_volume",
                "conformation_sasa_fes",
                "interface_slab",
                "alchemical_free_energy",
            ],
            "extract": ["extract"],
            "xtb": ["xtb"],
            "orca": ["orca"],
            "nci": ["nci"],
            "synth": ["synth"],
        }
        if stage in stage_map:
            selected = list(stage_map[stage])
        elif stage in SPEC_BY_ID:
            selected = [stage]
        else:
            raise ValueError(f"Unknown stage or analysis ID: {stage}")
    else:
        profile_name = profile_override or str(cfg_a.get("profile") or "standard")
        explicit_enabled = cfg_a.get("enabled")
        if explicit_enabled:
            selected = [str(x).strip() for x in explicit_enabled if str(x).strip()]
        else:
            if profile_name not in ANALYSIS_PROFILES:
                raise ValueError(
                    f"Unknown analysis profile '{profile_name}'. Available: {sorted(ANALYSIS_PROFILES)}"
                )
            selected = list(ANALYSIS_PROFILES[profile_name])
        disabled = {str(x).strip() for x in (cfg_a.get("disabled") or []) if str(x).strip()}
        selected = [s for s in selected if s not in disabled]

    unknown = [s for s in selected if s not in SPEC_BY_ID]
    if unknown:
        raise ValueError(f"Unknown analysis IDs: {unknown}. Valid IDs: {list(ALL_ANALYSIS_IDS)}")

    # Preserve canonical registry ordering while deduplicating
    sel_set = set(selected)
    return [s.id for s in ANALYSIS_REGISTRY if s.id in sel_set]


@dataclass
class GateDecision:
    module_id: str
    category: str
    enabled: bool
    status: str  # "RUN" | "SKIPPED" | "DISABLED"
    reason: str
    warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "category": self.category,
            "enabled": self.enabled,
            "status": self.status,
            "reason": self.reason,
            "warnings": "; ".join(self.warnings),
        }


def _find_dhdl_files(runs: list[Any], workflow_root: Path | None) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    roots: list[Path] = []
    for r in runs:
        if getattr(r, "run_dir", None):
            roots.append(Path(r.run_dir))
    if workflow_root and workflow_root.exists():
        roots.append(workflow_root)
    for root in roots:
        for pat in ("dhdl*.xvg", "*/dhdl*.xvg", "05_fep/**/dhdl*.xvg", "alchemical/**/dhdl*.xvg", "fep/**/dhdl*.xvg"):
            for p in sorted(root.glob(pat)):
                rp = p.resolve()
                if rp.is_file() and rp not in seen:
                    seen.add(rp)
                    found.append(rp)
    return found


def _inspect_runs_capabilities(runs: list[Any], cfg: Any) -> dict[str, Any]:
    """Inspect discovered runs cheaply to determine physics/data eligibility for every module."""
    has_thermo = any(getattr(r, "thermo_xvg", None) and Path(r.thermo_xvg).exists() for r in runs)
    has_stress = any(getattr(r, "stress_xvg", None) and Path(r.stress_xvg).exists() for r in runs)
    has_traj = any(
        getattr(r, "prod_tpr", None)
        and Path(r.prod_tpr).exists()
        and getattr(r, "prod_xtc", None)
        and Path(r.prod_xtc).exists()
        for r in runs
    )
    has_trr = any(
        getattr(r, "prod_tpr", None)
        and (Path(r.prod_tpr).parent / "prod.trr").exists()
        for r in runs
    )
    has_rdf = any(bool(getattr(r, "rdf_xvgs", None)) for r in runs)
    has_topol = any(bool(getattr(r, "topologies", None)) for r in runs)

    # Check charges, ionic species, polyatomic species
    has_atomic_charges = False
    has_ionic_species = False
    has_polyatomic = False
    has_torsion_candidate = False
    for r in runs:
        for topo in (getattr(r, "topologies", None) or {}).values():
            charges = getattr(topo, "atom_charges", None) or ()
            if any(abs(float(q)) > 1e-6 for q in charges):
                has_atomic_charges = True
            if abs(float(sum(charges))) > 1e-3:
                has_ionic_species = True
            n_atoms = len(getattr(topo, "atom_names", None) or ())
            if n_atoms >= 2:
                has_polyatomic = True
            if n_atoms >= 4:
                has_torsion_candidate = True

    # Distinct compositions & temperatures across runs
    comp_signatures: set[tuple[tuple[str, float], ...]] = set()
    temperatures: set[float] = set()
    for r in runs:
        counts = getattr(r, "mol_counts", None) or {}
        tot = sum(counts.values())
        if tot > 0:
            sig = tuple(sorted((k, round(v / tot, 4)) for k, v in counts.items()))
            comp_signatures.add(sig)
        meta = getattr(r, "metadata", None) or {}
        t_val = meta.get("temperatureK") or meta.get("temperature_k")
        if t_val is not None:
            try:
                temperatures.add(round(float(t_val), 2))
            except (TypeError, ValueError):
                pass
    if not temperatures and getattr(cfg, "temperature_k", None):
        temperatures.add(round(float(cfg.temperature_k), 2))

    # Inspect first available trajectory metadata cheaply (dt, n_frames, velocities, box aspect ratio)
    dt_ps: float | None = None
    n_frames: int = 0
    has_velocities = False
    is_slab_box = False
    for r in runs:
        tpr = getattr(r, "prod_tpr", None)
        trr = Path(tpr).parent / "prod.trr" if tpr else None
        xtc = getattr(r, "prod_xtc", None)
        traj_cand = trr if (trr and trr.exists()) else xtc
        if tpr and Path(tpr).exists() and traj_cand and Path(traj_cand).exists():
            try:
                import MDAnalysis as mda

                u = mda.Universe(str(tpr), str(traj_cand))
                n_frames = len(u.trajectory)
                ts0 = u.trajectory[0]
                has_velocities = bool(getattr(ts0, "has_velocities", False))
                if ts0.dimensions is not None and len(ts0.dimensions) >= 3:
                    lx, ly, lz = float(ts0.dimensions[0]), float(ts0.dimensions[1]), float(ts0.dimensions[2])
                    if max(lx, ly) > 0 and lz / max(lx, ly) >= 1.5:
                        is_slab_box = True
                if n_frames >= 2:
                    t0 = float(ts0.time)
                    t1 = float(u.trajectory[1].time)
                    if t1 > t0:
                        dt_ps = t1 - t0
                break
            except Exception:
                continue

    dhdl_files = _find_dhdl_files(runs, getattr(cfg, "workflow_root", None))
    opts = (getattr(cfg, "analyses", None) or {}).get("options") or {}
    prod_ns = max(0.0, float(getattr(cfg, "t_end_ns", 0.0)) - float(getattr(cfg, "t_eq_ns", 0.0)))

    return {
        "n_runs": len(runs),
        "has_thermo": has_thermo,
        "has_stress": has_stress,
        "has_traj": has_traj or has_trr,
        "has_trr": has_trr,
        "has_rdf": has_rdf,
        "has_topol": has_topol,
        "has_atomic_charges": has_atomic_charges,
        "has_ionic_species": has_ionic_species,
        "has_polyatomic": has_polyatomic,
        "has_torsion_candidate": has_torsion_candidate,
        "n_compositions": len(comp_signatures),
        "n_temperatures": len(temperatures),
        "dt_ps": dt_ps,
        "n_frames": n_frames,
        "has_velocities": has_velocities,
        "is_slab_box": is_slab_box or bool(opts.get("force_slab", False)),
        "n_dhdl_files": len(dhdl_files),
        "prod_ns": prod_ns,
        "vdos_max_dt_ps": float(opts.get("vdos_max_dt_ps", 0.05)),
    }


def evaluate_eligibility(
    runs: list[Any],
    cfg: Any,
    selected_ids: list[str],
    *,
    caps_override: dict[str, Any] | None = None,
) -> tuple[dict[str, GateDecision], pd.DataFrame]:
    """Run the Physics & Data Gatekeeper for every registered module."""
    caps = caps_override if caps_override is not None else _inspect_runs_capabilities(runs, cfg)
    sel_set = set(selected_ids)
    decisions: dict[str, GateDecision] = {}

    for spec in ANALYSIS_REGISTRY:
        mid = spec.id
        if mid not in sel_set:
            decisions[mid] = GateDecision(mid, spec.category, False, "DISABLED", "Not selected in active profile/checklist")
            continue

        warnings: list[str] = []
        status = "RUN"
        reason = "All physics and data prerequisites satisfied"

        if mid == "equil":
            if not caps["has_thermo"] and caps["n_runs"] == 0:
                status, reason = "SKIPPED", "No discovered runs with thermo.xvg"
        elif mid == "thermo_fluctuations":
            if not caps["has_thermo"]:
                status, reason = "SKIPPED", "Missing thermo.xvg (NPT energy/volume time series required for Cp, κT, αP)"
        elif mid in {"kbi", "kbi_full"}:
            if not (caps["has_topol"] and (caps["has_traj"] or caps["has_rdf"])):
                status, reason = "SKIPPED", "Requires 00_input/topol.top and either production trajectory or rdf_*.xvg"
        elif mid == "kbi_composition_series":
            if caps["n_compositions"] < 3:
                status = "SKIPPED"
                reason = (
                    f"Requires >=3 distinct mole-fraction compositions across runs to integrate Γ(x) -> ln γi, GE "
                    f"(found {caps['n_compositions']})"
                )
            elif caps["n_compositions"] < 5:
                warnings.append(
                    f"Only {caps['n_compositions']} compositions available; >=5 points recommended for accurate Redlich–Kister / Gibbs–Duhem integration"
                )
        elif mid == "kbi_temperature_series":
            if caps["n_temperatures"] < 2:
                status = "SKIPPED"
                reason = (
                    f"Requires >=2 distinct temperatures across runs for Gibbs–Helmholtz HE/TSE "
                    f"(found {caps['n_temperatures']})"
                )
            elif caps["n_temperatures"] < 3:
                warnings.append("Only 2 temperatures available; >=3 temperatures recommended for curvature/error checks")
        elif mid == "gk":
            if not (caps["has_stress"] or caps["has_traj"]):
                status, reason = "SKIPPED", "Requires stress.xvg (Green–Kubo viscosity) and/or trajectory (MSD diffusion)"
        elif mid == "electrochemistry_onsager":
            if not caps["has_traj"]:
                status, reason = "SKIPPED", "Requires unwrapped production trajectory (prod.tpr + prod.xtc)"
            elif not caps["has_ionic_species"]:
                status = "SKIPPED"
                reason = "No ionic species detected (all species have net molecular charge ≈ 0); Onsager ionic conductivity/Haven ratio requires charged ions"
        elif mid == "rotational_dynamics":
            if not caps["has_traj"]:
                status, reason = "SKIPPED", "Requires production trajectory (prod.tpr + prod.xtc)"
            elif not caps["has_polyatomic"]:
                status, reason = "SKIPPED", "Requires at least one polyatomic species (>=2 atoms) to define molecular orientation vectors"
        elif mid == "glassy_dynamics":
            if not caps["has_traj"]:
                status, reason = "SKIPPED", "Requires production trajectory (prod.tpr + prod.xtc)"
            elif caps["n_frames"] > 0 and caps["n_frames"] < 10:
                status, reason = "SKIPPED", f"Requires >=10 trajectory frames for α2(t)/Fs(q,t) (found {caps['n_frames']})"
        elif mid == "dielectric":
            if not caps["has_traj"]:
                status, reason = "SKIPPED", "Requires production trajectory (prod.tpr + prod.xtc)"
            elif not caps["has_atomic_charges"]:
                status, reason = "SKIPPED", "No atomic partial charges found in .itp topologies; dipole fluctuations cannot be computed"
            else:
                if caps["prod_ns"] > 0 and caps["prod_ns"] < 10.0:
                    warnings.append(f"Production window ({caps['prod_ns']:.1f} ns) < 10 ns; static ε0 may not be fully converged")
                if caps["has_ionic_species"]:
                    warnings.append("Ionic system detected: ε0 reported from molecular rotational dipole fluctuations (conducting ionic term separated)")
        elif mid == "spectroscopy_vdos_ir":
            if not caps["has_traj"]:
                status, reason = "SKIPPED", "Requires trajectory with fine time spacing"
            elif caps["dt_ps"] is not None and caps["dt_ps"] > caps["vdos_max_dt_ps"]:
                nyquist_cm1 = 1.0 / (2.0 * 2.99792458e-2 * caps["dt_ps"])
                status = "SKIPPED"
                reason = (
                    f"Frame spacing dt={caps['dt_ps']:.4f} ps exceeds limit {caps['vdos_max_dt_ps']} ps "
                    f"(Nyquist={nyquist_cm1:.0f} cm^-1 too low for vibrational/IR spectroscopy; save frames at <=5 fs)"
                )
            elif not caps["has_velocities"] and not caps["has_atomic_charges"]:
                status, reason = "SKIPPED", "Requires atomic velocities (.trr) for VDOS or partial charges for IR spectrum"
            else:
                if caps["dt_ps"] is not None and caps["dt_ps"] > 0.005:
                    nyquist_cm1 = 1.0 / (2.0 * 2.99792458e-2 * caps["dt_ps"])
                    warnings.append(f"dt={caps['dt_ps']:.4f} ps limits spectrum to Far-IR/THz (Nyquist ≈ {nyquist_cm1:.0f} cm^-1)")
        elif mid in {"md_solvation", "md_hbond", "md_sdf", "md_network", "voronoi_free_volume"}:
            if not caps["has_traj"]:
                status, reason = "SKIPPED", "Requires production trajectory (prod.tpr + prod.xtc)"
        elif mid == "conformation_sasa_fes":
            if not caps["has_traj"]:
                status, reason = "SKIPPED", "Requires production trajectory (prod.tpr + prod.xtc)"
            elif not caps["has_polyatomic"]:
                status, reason = "SKIPPED", "Requires polyatomic molecules for SASA/Rg/conformational analysis"
        elif mid == "interface_slab":
            if not caps["is_slab_box"]:
                status = "SKIPPED"
                reason = "Bulk isotropic box detected (Lz/max(Lx,Ly) < 1.5); slab geometry required for surface tension & z-profiles (or set analyses.options.force_slab: true)"
        elif mid == "alchemical_free_energy":
            if caps["n_dhdl_files"] < 2:
                status = "SKIPPED"
                reason = f"Requires >=2 multi-window λ dhdl*.xvg files for TI/BAR/MBAR (found {caps['n_dhdl_files']})"

        decisions[mid] = GateDecision(mid, spec.category, True, status, reason, warnings)

    df = pd.DataFrame([d.to_row() for d in decisions.values()])
    return decisions, df


def export_catalog_payload() -> dict[str, Any]:
    """JSON-serializable catalog of all analyses & profiles for CLI / MoDy Studio UI."""
    return {
        "profiles": {k: list(v) for k, v in ANALYSIS_PROFILES.items()},
        "modules": [s.to_dict() for s in ANALYSIS_REGISTRY],
    }
