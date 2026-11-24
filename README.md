# MoDy Analyse v2.0.1

**Global, Modular MD–KBI–Green–Kubo, Electrolyte/Spectroscopy/FEP, and MD-to-QM Computational Chemistry Workflow Engine (Open-Source Core)**

`MoDy Analyse` is the open-source Python compute engine (**GPL-3.0-only**) built for arbitrary liquid mixtures, deep eutectic solvents (DES), ionic liquids, aqueous co-solvents, and pharmaceutical/natural-product solvation studies across the world.

It forms the open-source scientific core alongside two ecosystem tools:
- **[SolvForge MD](https://github.com/iehoshva/solvforge-md)** (browser/local planning UI for GROMACS preparation)
- **MoDy Studio** (`../mody-studio` — commercial workstation & HPC companion app: offline Web Studio UI, automated `Packmol`/GROMACS system builder, SolvForge importer, `SLURM`/`PBS` batch scheduler, modular analysis checklist, and interactive HTML/LaTeX publication reports)

---

## Key Capabilities (`v2.0.1` Universal Modular Engine)

- **Automatic `.edr` $\to$ `.xvg` Self-Healing Sync (`--sync-edr` & `scripts/sync_xvg_from_edr.py`)**: Automatically creates or updates missing/incomplete/stale `analysis/thermo.xvg` and `analysis/stress.xvg` from `prod.edr` via `panedr` (pure Python) or native `gmx energy` before the Gatekeeper and analyzers run.

- **Modular Analysis Selector & Automated Physics Gatekeeper**: Choose from 9 named profiles (`standard`, `quick_qc`, `full_kbi_thermo`, `electrolyte_des`, `spectroscopy`, `interface_slab`, `conformation_free_energy`, `md_only_everything`, `everything`) or pass any custom `--analyses` checklist (`--list-analyses` exports the full JSON catalog). Every module validates physical and data prerequisites automatically (Nyquist $\Delta t$ guard, ionic-charge check, multi-composition check, slab check, multi-window `dhdl.xvg` check) and records an explicit `analysis_gatekeeper_report.csv`.
- **General Triclinic PBC & Cromer–Mann WAXS/SAXS**: Exact orthorhombic and triclinic minimum-image geometry (rhombic dodecahedron / truncated octahedron) plus International Tables 9-parameter Cromer–Mann $q$-dependent X-ray atomic form factors $f_a(q)$.
- **Full Kirkwood–Buff & Bulk NPT Thermodynamics**: Dawass $1/R$ extrapolation with $R^2$ QC, direct sub-volume particle fluctuation KBI (Cortes-Huerto/Schnell canonical finite-size correction), block-averaged SEM, Bhatia–Thornton ($S_{NN}, S_{CC}, S_{NC}$), $S(q)$/SAXS/WAXS, multi-composition Redlich–Kister integration ($\ln\gamma_i, G^E$, Setschenow $K_S$), Gibbs–Helmholtz ($H^E, TS^E$), and NPT fluctuations ($C_P, C_V, \kappa_T, \alpha_P$, CED, Hildebrand/Hansen).
- **Electrochemistry, Rotational/Glassy Dynamics, Dielectric & Spectroscopy**: Full Onsager transport matrix ($L_{ij}$), Einstein–Helfand vs Nernst–Einstein conductivity ($\sigma_{\text{EH}}, \sigma_{\text{NE}}$, Haven ratio, ionicity, true/ideal transference $t_+$), rotational $C_1/C_2$ (KWW + Debye ratio), non-Gaussian $\alpha_2(t)$ / $F_s(q^*, t)$ / van Hove $G_s(r, t^*)$, static dielectric $\varepsilon_0$ & Kirkwood $g_K$, and VDOS / IR-THz spectra.
- **Conformation, Voronoi, Interface/Slab & Alchemical Free Energy**: Shrake–Rupley polar/apolar SASA, $R_g$, rotatable dihedral 1D FES, PCA/K-Means clustering, 3D periodic Voronoi tessellation & free-volume cavities, slab surface tension $\gamma$ / tanh $z$-density / $P_2$ orientation, and zero-dependency TI, BAR & MBAR with overlap matrix QC.
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
