# External Tools Directory (`tools/`)

Do **not** commit or redistribute proprietary / third-party binaries here (`tools/*` is git-ignored except this file).

To enable the full pipeline, install the external software independently according to their own licenses and configure their executable paths in your local `config.yaml`:

1. **GROMACS** (`gmx`): MD trajectory & `.xvg` energy/RDF extraction.
2. **GFN2-xTB** (`xtb`): Installed automatically via Conda (`environment.yml`).
3. **ORCA 6.x** (`orca`, `orca_2mkl` + matching OpenMPI): Academic/licensed download from the ORCA forum.
4. **Multiwfn** (`Multiwfn_noGUI`): Download from the official Multiwfn release page; set `isilent= 1` in `settings.ini` for non-interactive batch mode.
5. **VMD** (`vmd`, optional): Renders SDF `.dx` grids and NCI/IGMH `.cub` isosurfaces using generated `.vmd` scripts.
