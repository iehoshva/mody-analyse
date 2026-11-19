# Upstream MD System Builders (`mody_analyse.builders`)

This subpackage is reserved for upstream MD system-preparation automation that turns chemical definitions into runnable GROMACS simulation boxes ready for `MoDy Analyse`:

1. **SolvForge Run Package Adapter** *(planned)*:
   Imports exported archives / `project_manifest.json` + `metadata.json` directly from [SolvForge MD](https://github.com/iehoshva/solvforge-md) and validates directory layout (`00_input/`, `04_prod/`, `analysis/`).
2. **SMILES → OPLS-AA / GAFF → Packmol Builder** *(planned)*:
   Automates molecular geometry optimization, charge assignment, `.itp`/`.gro` assembly, `packmol.inp` generation, and baseline GROMACS `.mdp` protocols for CLI batch users or Pro desktop workflows.

Until these builders are implemented in `v0.2+`, users prepare runnable GROMACS packages with **SolvForge MD** (recommended) or standard GROMACS/Packmol scripts matching the contract in `integrations/solvforge_bridge/`.
