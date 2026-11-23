from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    project_root: Path
    workflow_root: Path
    cosmors_csv: Path | None
    results_dir: Path
    conda_env: str
    orca_bin: Path
    orca_2mkl_bin: Path
    xtb_bin: Path
    multiwfn_bin: Path
    vmd_bin: Path
    orca_nprocs: int
    orca_maxcore_mb: int
    led_nprocs: int
    led_maxcore_mb: int
    led_ram_limit_gb: float
    xtb_workers: int
    temperature_k: float
    t_eq_ns: float
    t_end_ns: float
    kbi_rebin_nm: float
    kbi_rmax_nm: float
    kbi_plateau_range_nm: tuple[float, float]
    hbond_dist_nm: float
    hbond_angle_deg: float
    hbond_angle_sensitivity_deg: float
    rdf_stride: int
    hbond_stride: int
    network_stride: int
    sdf_stride: int
    sdf_grid_spacing_a: float
    sdf_half_width_a: float
    strain_snapshots_per_rep: int
    strain_ref_conformers: int
    dimers_per_pair_per_rep: int
    trimers_per_system_per_rep: int
    dft_dimers_per_pair: int
    cp_validation_dimers_per_pair: int
    dft_trimers_per_solute: int
    nci_clusters_total: int
    led_dimers_total: int
    run_glob: str = "*"
    openmpi_dir: Path | None = None
    solvents: tuple[str, ...] = ()
    primary_solvent: str | None = None
    solutes: tuple[str, ...] = ()
    pure_system_code: str = "PURE"
    species_properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    analyses: dict[str, Any] = field(default_factory=dict)

    def raw_dict(self) -> dict[str, Any]:
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}


def _resolve_optional_path(raw_val: Any, base_dir: Path) -> Path | None:
    if raw_val is None or str(raw_val).strip() == "":
        return None
    p = Path(str(raw_val))
    return p.resolve() if p.is_absolute() else (base_dir / p).resolve()


def _resolve_path(raw_val: Any, base_dir: Path) -> Path:
    p = Path(str(raw_val))
    return p.resolve() if p.is_absolute() else (base_dir / p).resolve()


def load_config(path: str | Path) -> Config:
    cfg_path = Path(path).resolve()
    base_dir = cfg_path.parent
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    b = raw["binaries"]
    c = raw["compute"]
    m = raw["md"]
    q = raw["qm_sampling"]
    sys_sec = raw.get("system") or {}
    bench_sec = raw.get("benchmark") or {}
    paths_sec = raw.get("paths") or {}
    analyses_sec = raw.get("analyses") or {}

    ompi_raw = b.get("openmpi_dir")
    cosmors_raw = raw.get("cosmors_csv", bench_sec.get("csv_path"))
    wf_raw = raw.get("workflow_root", ".")
    wf_root = _resolve_path(wf_raw, base_dir)

    solvents = tuple(str(s) for s in (sys_sec.get("solvents") or []))
    primary_solv = sys_sec.get("primary_solvent")
    if not primary_solv and solvents:
        primary_solv = solvents[0]

    return Config(
        project_root=_resolve_path(raw["project_root"], base_dir),
        workflow_root=wf_root,
        cosmors_csv=_resolve_optional_path(cosmors_raw, base_dir),
        results_dir=_resolve_path(raw.get("results_dir", "./results"), base_dir),
        conda_env=str(b["conda_env"]),
        orca_bin=Path(str(b["orca_bin"])),
        orca_2mkl_bin=Path(str(b["orca_2mkl_bin"])),
        xtb_bin=Path(str(b["xtb_bin"])),
        multiwfn_bin=Path(str(b["multiwfn_bin"])),
        vmd_bin=Path(str(b["vmd_bin"])),
        orca_nprocs=int(c["orca_nprocs"]),
        orca_maxcore_mb=int(c["orca_maxcore_mb"]),
        led_nprocs=int(c["led_nprocs"]),
        led_maxcore_mb=int(c["led_maxcore_mb"]),
        led_ram_limit_gb=float(c["led_ram_limit_gb"]),
        xtb_workers=int(c["xtb_workers"]),
        temperature_k=float(m["temperature_k"]),
        t_eq_ns=float(m["t_eq_ns"]),
        t_end_ns=float(m["t_end_ns"]),
        kbi_rebin_nm=float(m["kbi_rebin_nm"]),
        kbi_rmax_nm=float(m["kbi_rmax_nm"]),
        kbi_plateau_range_nm=(float(m["kbi_plateau_range_nm"][0]), float(m["kbi_plateau_range_nm"][1])),
        hbond_dist_nm=float(m["hbond_dist_nm"]),
        hbond_angle_deg=float(m["hbond_angle_deg"]),
        hbond_angle_sensitivity_deg=float(m["hbond_angle_sensitivity_deg"]),
        rdf_stride=int(m["rdf_stride"]),
        hbond_stride=int(m["hbond_stride"]),
        network_stride=int(m["network_stride"]),
        sdf_stride=int(m["sdf_stride"]),
        sdf_grid_spacing_a=float(m["sdf_grid_spacing_a"]),
        sdf_half_width_a=float(m["sdf_half_width_a"]),
        strain_snapshots_per_rep=int(q["strain_snapshots_per_rep"]),
        strain_ref_conformers=int(q["strain_ref_conformers"]),
        dimers_per_pair_per_rep=int(q["dimers_per_pair_per_rep"]),
        trimers_per_system_per_rep=int(q["trimers_per_system_per_rep"]),
        dft_dimers_per_pair=int(q["dft_dimers_per_pair"]),
        cp_validation_dimers_per_pair=int(q["cp_validation_dimers_per_pair"]),
        dft_trimers_per_solute=int(q["dft_trimers_per_solute"]),
        nci_clusters_total=int(q["nci_clusters_total"]),
        led_dimers_total=int(q["led_dimers_total"]),
        run_glob=str(raw.get("run_glob", "*")),
        openmpi_dir=Path(str(ompi_raw)) if ompi_raw else None,
        solvents=solvents,
        primary_solvent=str(primary_solv) if primary_solv else None,
        solutes=tuple(str(s) for s in (sys_sec.get("solutes") or [])),
        pure_system_code=str(sys_sec.get("pure_system_code", "PURE")),
        species_properties={str(k): dict(v or {}) for k, v in (sys_sec.get("species_properties") or {}).items()},
        benchmark=dict(bench_sec),
        paths={str(k): str(v) for k, v in paths_sec.items()},
        analyses=dict(analyses_sec),
    )
