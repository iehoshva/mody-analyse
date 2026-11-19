# MoDy Analyse v1.0.0

**Global, Modular MD–KBI–Green–Kubo and MD-to-QM Computational Chemistry Workflow Engine (Open-Source Core)**

`MoDy Analyse` is the open-source Python compute engine (**GPL-3.0-only**) built for arbitrary liquid mixtures, deep eutectic solvents (DES), ionic liquids, aqueous co-solvents, and pharmaceutical/natural-product solvation studies across the world.

It forms the open-source scientific core alongside two ecosystem tools:
- **[SolvForge MD](https://github.com/iehoshva/solvforge-md)** (browser/local planning UI for GROMACS preparation)
- **MoDy Studio** (`../mody-studio` — commercial workstation & HPC companion app: offline Web Studio UI, automated `Packmol`/GROMACS system builder, `SLURM`/`PBS` batch scheduler, curated solvent library, and interactive HTML/LaTeX publication reports)

---

## Key Capabilities (`v1.0.0` Global Engine)

- **Zero Chemistry Hardcoding**: Works automatically for **1-component pure liquids**, **binary DES/co-solvents**, or **$N$-component solvent mixtures** paired with **0 or more solutes**. Molar masses, effective hydrodynamic radii ($r_H$), molecular alignment triplets, polar H-bond sites, ring groups, and hydrophobic tail anchors are derived directly from `.itp` + RDKit topology (with optional overrides in YAML).
- **Flexible Directory Discovery**: Supports both **SolvForge Run Packages** (`metadata.json`, `00_input/`, `04_prod/`, `analysis/`) and standard flat GROMACS simulation directories (even without `metadata.json` or pre-extracted `rdf_*.xvg` curves).
- **Thermodynamics & Transport**:
  - Equilibration QC & dynamic time-window convergence
  - Ganguly–van der Vegt tail-corrected Kirkwood–Buff integrals (KBI) + Krüger finite-size extrapolation + $N$-component constant-$(T,P)$ KB inversion
  - Rotationally invariant Green–Kubo shear viscosity (with coarse-dump ballistic artifact guard) + unwrapped COM Einstein diffusion + species-specific Stokes–Einstein checks
- **Atomistic Structural Mechanics (#1–#4)**:
  - Solvation shell coordination numbers, molar & volume-weighted local enrichment across $N$ solvents, sensitivity definitions, and site-specific RDFs
  - Intermittent hydrogen-bond populations, donor→acceptor residence ACFs (bi-exponential fit), and angle sensitivity
  - Molecular local-frame 3D Spatial Distribution Functions (OpenDX `.dx` + VMD scripts) and 1D/2D radial-angular probability maps
  - Collective H-bond graph percolation, cyclomatic ring counts, and shell-vs-bulk degree shifts
- **PBC-Safe MD→QM Pipeline (#5–#9)**:
  - BFS covalent bond unwrapping & minimum-image clustering for solute strain snapshots, stratified H-bond/vdW dimers, and arbitrary hetero/homo-trimers
  - Automated `GFN2-xTB` conformer/strain/interaction/3-body screening, `ORCA` (`r2SCAN-3c`, `wB97X-D3` CP BSSE, `DLPNO-CCSD(T)/def2-TZVP` LED with RAM watchdog & MPI fallback), and `Multiwfn` NCI/IGMH topological analysis
- **Optional External Benchmark Integration & Dynamic Plots**:
  - Plugs cleanly into COSMO-RS or experimental CSV benchmarks via YAML mapping (or runs completely standalone without external benchmarks) and produces multi-replicate (`mean ± SD`) tables + publication PNG/PDF figures.

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate mody-analyse
pip install -e . --no-build-isolation

pytest -v
mody-analyse --help
```

Copy [`configs/config.template.yaml`](configs/config.template.yaml) (or a preset from [`configs/presets/`](configs/presets/)) to `config.yaml`, point `project_root` to your GROMACS runs, and execute all stages cleanly with resume/caching:

```bash
mody-analyse --config config.yaml --all --stage all
```

---

## Open-Source Core vs. Paid Companion (`MoDy Studio`)

- **`MoDy Analyse` (`GPL-3.0-only`, this repository)**: Complete CLI scientific workflow engine, unit & benchmark test suites, reproducible Conda environment, and integration schemas — free forever for researchers to `git clone`, audit, cite, and run.
- **`MoDy Studio` (`Commercial`, separate package)**: Offline local Web Studio UI (eliminating cloud hosting dependencies), SMILES→`Packmol`→GROMACS automated system builder, `SLURM`/`PBS` HPC job-array generators, curated DES preset packs, and self-contained interactive HTML/LaTeX report publishers communicating strictly via CLI & JSON/CSV boundaries.

See [`docs/open_core_strategy.md`](docs/open_core_strategy.md) and [`docs/v1_architecture.md`](docs/generalization_roadmap.md).
