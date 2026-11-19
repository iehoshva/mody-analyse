# MoDy Analyse Architecture

## Pipeline Stages (`mody_analyse.cli`)

| CLI `--stage` | Subpackage | Primary Functions |
| :--- | :--- | :--- |
| `equil` | `analyzers.equilibration` | Equilibration drift & 50 ns window convergence (`thermo.xvg` + late block RDF L2) |
| `kbi` | `analyzers.kbi_analyzer` | Ganguly–van der Vegt tail correction, Kruger finite-size KBI, ternary KB matrix inversion |
| `gk` | `analyzers.green_kubo` | 5-component traceless shear stress ACF, ballistically guarded GK integral, unwrapped COM MSD + Stokes–Einstein |
| `md` | `analyzers.md_structural` | #1 Solvation shell & enrichment, #2 H-bond populations & bi-exponential lifetimes, #3 2D/1D orientation & 3D OpenDX SDF, #4 H-bond network graphs & cycles |
| `extract` | `md_to_qm.cluster_extractor` | PBC-whole BFS molecular reconstruction, solute conformations, stratified H-bond/vdW dimers, and trimers |
| `xtb` | `qm_runners.qm_xtb` | RDKit bond-order mapping from `.itp`, GFN2-xTB conformers, dihedral-constrained strain, dimer interaction & 3-body non-additivity |
| `orca` | `qm_runners.qm_orca` | `r2SCAN-3c` DFT energies, `wB97X-D3` Counterpoise BSSE, DLPNO-CCSD(T)/def2-TZVP LED decomposition with MPI safety ladder |
| `nci` | `qm_runners.nci_igmh` | `orca_2mkl` Molden conversion, `Multiwfn_noGUI` IGMH/NCI scatter & integral summaries, VMD visualization scripts |
| `synth` | `reporting.synthesis`, `reporting.plotting` | Multi-replicate `mean ± SD` tables, optional COSMO-RS cross-reference join, publication PNG/PDF figures |

## Caching & Fault Tolerance
- `cli.py` reloads any pre-existing `results/tables/*.csv` before running a single stage, allowing stages to execute independently or resume after failures.
- Individual `xtb`, `ORCA`, and `Multiwfn` jobs detect up-to-date normal termination outputs (`****ORCA TERMINATED NORMALLY****` / matching input mtime) before re-running.
