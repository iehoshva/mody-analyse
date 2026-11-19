# Open-Core & Digital Product Strategy (`SolvForge MD` + `MoDy Analyse`)

Because **[SolvForge MD](https://github.com/iehoshva/solvforge-md)** (`GPL-3.0-only`) is a browser/Cloudflare TypeScript app (`Next.js + Vite + IndexedDB`) and **`MoDy Analyse`** (`GPL-3.0-only`) is a Python/HPC compute engine (`MDAnalysis + RDKit + xTB + ORCA + Multiwfn`), keeping them as **two connected repositories** gives each side a clean runtime boundary.

## Recommended Community vs. Pro Split

### 1. Open-Source Community Core (GitHub + Zenodo DOI + Academic Paper)
- Target audience: computational chemists, graduate students, and labs comfortable with `conda`, `YAML`, CLI, and compiling/installing their own `GROMACS`/`ORCA`/`Multiwfn`.
- Distribution: Public GitHub (`git clone` / `pip install mody-analyse`) + reproducible `environment.yml`.
- Purpose: Maximizes scientific credibility, peer-reviewed citations, bug reports, and global adoption.

### 2. Digital Product / "Pro Studio & HPC Pack" *(Optional Commercial Layer)*
Rather than restricting scientific formulas, commercial value in computational research tools comes from **zero-friction packaging, UI convenience, automation, and curated datasets**:
1. **Offline Local Bundle (`SolvForge Desktop / Container Studio`)**:
   Since hosted websites like `solvforge.coffenova.com` require ongoing server upkeep, package the SolvForge web UI + Python backend in an offline local launcher (Electron/Tauri or Docker/Apptainer container) where users load trajectories and trigger `MoDy Analyse` with one click from localhost.
2. **Automated Upstream System Builder**:
   Turnkey `SMILES → Force-Field (GAFF/OPLS) → Packmol → GROMACS MDP` generation wizard feeding directly into `mody_analyse.builders`.
3. **HPC Scheduler & Watchdog Pack**:
   Pre-built `SLURM` / `PBS` job-array templates, ORCA scratch-cleanup monitors, and automated multi-node retry scripts for large screening campaigns.
4. **Curated Solvent / DES Topology Library & Video Course**:
   Pre-validated force-field parameters, benchmark cases, and step-by-step publication figure walkthroughs.

## Licensing & Redistribution Rules
- **Never bundle third-party proprietary or non-commercial binaries** (such as `ORCA`, `Multiwfn`, or `VMD`) inside downloadable or paid packages without explicit permission from their authors.
- Keep **SolvForge MD** and **MoDy Analyse** strictly compliant with `GPL-3.0-only` when distributing code linked to `MDAnalysis` / SolvForge modules; sell convenience wrappers, independent assets (parameter datasets, tutorials, support), or container build recipes transparently.
