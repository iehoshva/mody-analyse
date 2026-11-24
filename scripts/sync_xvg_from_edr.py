#!/usr/bin/env python3
"""Standalone helper: auto-create or update thermo.xvg and stress.xvg from GROMACS .edr files.

Works on either a single GROMACS run directory or an entire study/runs directory.
Uses `panedr` (pure Python, no GROMACS required) or falls back to `gmx energy`.

Examples:
    python scripts/sync_xvg_from_edr.py /path/to/study/runs
    python scripts/sync_xvg_from_edr.py /path/to/run_01 --extra "LJ-14,Coulomb-14"
    python scripts/sync_xvg_from_edr.py /path/to/study/runs --backend gmx --force
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running directly from repo checkout without editable install
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mody_analyse.core.edr_extractor import sync_directory_xvg_from_edr  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sync_xvg_from_edr.py",
        description="Auto-create or update missing/incomplete thermo.xvg & stress.xvg columns from GROMACS .edr files.",
    )
    parser.add_argument("target", type=Path, help="Single GROMACS run directory or parent study/runs directory")
    parser.add_argument(
        "--extra",
        default="",
        help="Comma-separated extra GROMACS energy terms to ensure in thermo.xvg (e.g. 'LJ-14,Coulomb-14')",
    )
    parser.add_argument(
        "--extra-stress",
        default="",
        help="Comma-separated extra terms to ensure in stress.xvg",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "panedr", "gmx"],
        default="auto",
        help="Extraction backend (default: auto -> panedr if installed, else gmx energy)",
    )
    parser.add_argument("--gmx-bin", default=None, help="Explicit path to gmx / gmx_mpi executable")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if XVG looks up-to-date")
    args = parser.parse_args(argv)

    if not args.target.exists():
        print(f"Error: target path does not exist: {args.target}", file=sys.stderr)
        return 1

    extra_th = [t.strip() for t in args.extra.split(",") if t.strip()]
    extra_st = [t.strip() for t in args.extra_stress.split(",") if t.strip()]
    df = sync_directory_xvg_from_edr(
        args.target,
        extra_thermo_terms=extra_th,
        extra_stress_terms=extra_st,
        backend=args.backend,
        gmx_bin=args.gmx_bin,
        force=args.force,
    )
    if df.empty:
        print(f"No .edr files found under {args.target}")
        return 0

    for _, row in df.iterrows():
        added = f" (+ {row['added_columns']})" if row["added_columns"] else ""
        print(f"[{row['status']:16s}] {row['run_id']} / {row['target']:6s} -> {row['xvg_path']}{added}")
        if row["status"] == "error":
            print(f"   ERROR: {row['message']}", file=sys.stderr)

    return 1 if (df["status"] == "error").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
