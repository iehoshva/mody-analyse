# Global Generalization Status (`v1.0.0`)

All core thesis hardcodes identified during `v0.1.0` extraction have been replaced in **`v1.0.0`**:

1. **Dynamic System & Layout Discovery (`core/config.py`, `core/discovery.py`)**:
   - Arbitrary `system.solvents`, `system.primary_solvent`, `system.solutes`, and custom `paths` hints in YAML.
   - Automatic solvent/solute classification from `[ molecules ]` counts and `topol.top` when `metadata.json` is omitted.
   - Supports inline `[ moleculetype ]` blocks inside `topol.top` as well as external `.itp` includes, plus automatic MDAnalysis COM RDF fallback if `analysis/rdf/*.xvg` is absent.
2. **Automatic Chemistry Inference (`core/chemistry.py`)**:
   - Infers molar weights from topology masses, Bondi van der Waals hydrodynamic radii (`rh_nm`), molar packing volumes, H-bond donors/acceptors, polar/ring/tail site atoms, and non-collinear 3-atom alignment frames automatically (with YAML overrides supported via `system.species_properties`).
3. **$N$-Solvent KBI, Transport, Structural & MD→QM Engines**:
   - Generalized `analyzers/kbi_analyzer.py`, `analyzers/green_kubo.py`, `analyzers/md_structural.py`, and `md_to_qm/cluster_extractor.py` cleanly across 1-solvent neat liquids, binary DES/co-solvents, and $N$-component mixtures.
4. **Dynamic Reporting & Optional External Benchmarks**:
   - `reporting/synthesis.py` and `reporting/plotting.py` run cleanly with or without COSMO-RS / experimental CSVs, auto-assign colorblind-safe Okabe-Ito palettes across arbitrary numbers of systems, and generate generic multi-solvent figures and tables.

## Planned Beyond `v1.0.0`
- Trajectory adapter plugins for **LAMMPS** (`.data`/`.lammpstrj`) and **AMBER** (`.prmtop`/`.nc`) alongside GROMACS.
- Native **OpenMM** trajectory analysis integration.
