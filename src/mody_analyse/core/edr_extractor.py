from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd

# Default energy terms to synchronize into analysis/thermo.xvg and analysis/stress.xvg
DEFAULT_THERMO_TERMS: tuple[str, ...] = (
    "Temperature",
    "Pressure",
    "Density",
    "Volume",
    "Potential",
    "Kinetic En.",
    "Total Energy",
    "Enthalpy",
    "LJ (SR)",
    "Disper. corr.",
    "Coulomb (SR)",
    "Coul. recip.",
)

DEFAULT_STRESS_TERMS: tuple[str, ...] = (
    "Temperature",
    "Volume",
    "Pres-XX",
    "Pres-YY",
    "Pres-ZZ",
    "Pres-XY",
    "Pres-XZ",
    "Pres-YZ",
)

# Common GROMACS term aliases (case-insensitive matching)
_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "Total Energy": ("Total Energy", "Total-Energy"),
    "Kinetic En.": ("Kinetic En.", "Kinetic-En."),
    "LJ (SR)": ("LJ (SR)", "LJ-SR"),
    "Coulomb (SR)": ("Coulomb (SR)", "Coulomb-SR", "Coul-SR"),
    "Coul. recip.": ("Coul. recip.", "Coul-recip.", "Coul-recip"),
    "Disper. corr.": ("Disper. corr.", "Disp-corr"),
}


@dataclass
class EdrSyncResult:
    run_id: str
    target: str
    xvg_path: Path
    edr_path: Path | None
    status: str  # "up_to_date" | "created" | "updated" | "skipped_no_edr" | "skipped_no_terms" | "error"
    backend: str = ""
    columns: list[str] = field(default_factory=list)
    added_columns: list[str] = field(default_factory=list)
    n_rows: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target": self.target,
            "xvg_path": str(self.xvg_path),
            "edr_path": str(self.edr_path) if self.edr_path else "",
            "status": self.status,
            "backend": self.backend,
            "columns": ", ".join(self.columns),
            "added_columns": ", ".join(self.added_columns),
            "n_rows": self.n_rows,
            "message": self.message,
        }


def read_xvg_legends(xvg_path: Path) -> list[str]:
    """Quickly parse only `@ sN legend "..."` header lines from an XVG file without loading data rows."""
    if not xvg_path.is_file():
        return []
    legends: list[str] = []
    with xvg_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            if line.startswith("@"):
                m = re.match(r'@\s+s\d+\s+legend\s+"([^"]+)"', line.strip())
                if m:
                    legends.append(m.group(1))
                continue
            # First numeric data line reached -> stop scanning
            if line.strip():
                break
    return legends


def write_xvg_from_dataframe(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "GROMACS Energies (extracted by MoDy Analyse)",
    source_edr: Path | None = None,
) -> int:
    """Write a GROMACS-compatible XVG file atomically with `@ sN legend` headers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or df.shape[1] < 2:
        raise ValueError("DataFrame must have at least Time + 1 energy column")

    cols = list(df.columns)
    val_cols = cols[1:]
    lines = [
        f"# Auto-extracted from {source_edr.name if source_edr else 'EDR'} by mody_analyse.core.edr_extractor",
        f'@    title "{title}"',
        '@    xaxis  label "Time (ps)"',
        '@    yaxis  label "(kJ/mol, K, bar, nm^3)"',
        "@TYPE xy",
    ]
    for idx, name in enumerate(val_cols):
        lines.append(f'@ s{idx} legend "{name}"')

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    arr = df.to_numpy(dtype=float)
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        for row in arr:
            fh.write(" ".join(f"{v:14.6f}" for v in row) + "\n")
    tmp_path.replace(out_path)
    return int(arr.shape[0])


def _resolve_term_name(desired: str, available: list[str]) -> str | None:
    """Match a desired energy term against available EDR term names (exact or alias, case-insensitive)."""
    avail_lower = {a.lower(): a for a in available}
    candidates = _TERM_ALIASES.get(desired, (desired,))
    for cand in candidates:
        if cand in available:
            return cand
        if cand.lower() in avail_lower:
            return avail_lower[cand.lower()]
    return None


def _find_gmx_bin(gmx_bin: str | None = None) -> str | None:
    if gmx_bin:
        return shutil.which(gmx_bin) or (gmx_bin if Path(gmx_bin).is_file() else None)
    return shutil.which("gmx") or shutil.which("gmx_mpi")


def _probe_gmx_edr_terms(edr_path: Path, gmx_exe: str) -> dict[str, int]:
    """Query `gmx energy -f <edr>` for the exact term-name -> term-number mapping."""
    res = subprocess.run(
        [gmx_exe, "energy", "-f", str(edr_path), "-o", "/dev/null"],
        input="0\n",
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (res.stdout or "") + "\n" + (res.stderr or "")
    # GROMACS prints numbered columns like: "  1  Bond  2  Angle  3  Proper-Dih."
    terms: dict[str, int] = {}
    for num_str, name in re.findall(r"(?:^|\s)(\d+)\s+([A-Za-z][A-Za-z0-9_.\-()/ ]*?)(?=\s+\d+\s+[A-Za-z]|\s*$)", combined, flags=re.MULTILINE):
        clean_name = name.strip()
        if clean_name:
            terms[clean_name] = int(num_str)
    return terms


def _extract_via_panedr(
    edr_path: Path,
    desired_terms: list[str],
    out_xvg: Path,
    *,
    title: str,
) -> tuple[list[str], int]:
    """Extract requested terms directly in Python via `panedr` (no GROMACS binary required)."""
    import panedr

    df_all = panedr.edr_to_df(str(edr_path))
    avail = [str(c) for c in df_all.columns if str(c).lower() != "time"]
    matched: list[str] = []
    for term in desired_terms:
        hit = _resolve_term_name(term, avail)
        if hit and hit not in matched:
            matched.append(hit)
    if not matched:
        return [], 0
    time_col = next((str(c) for c in df_all.columns if str(c).lower() == "time"), df_all.columns[0])
    sub = df_all[[time_col, *matched]].copy()
    n_rows = write_xvg_from_dataframe(sub, out_xvg, title=title, source_edr=edr_path)
    return matched, n_rows


def _extract_via_gmx(
    edr_path: Path,
    desired_terms: list[str],
    out_xvg: Path,
    gmx_exe: str,
) -> tuple[list[str], int]:
    """Extract requested terms via native `gmx energy` using dynamically probed term names."""
    term_map = _probe_gmx_edr_terms(edr_path, gmx_exe)
    avail = list(term_map.keys())
    matched: list[str] = []
    for term in desired_terms:
        hit = _resolve_term_name(term, avail)
        if hit and hit not in matched:
            matched.append(hit)
    if not matched:
        return [], 0

    out_xvg.parent.mkdir(parents=True, exist_ok=True)
    tmp_xvg = out_xvg.with_suffix(".tmp.xvg")
    stdin_payload = "\n".join(str(term_map[m]) for m in matched) + "\n0\n"
    res = subprocess.run(
        [gmx_exe, "energy", "-f", str(edr_path), "-o", str(tmp_xvg)],
        input=stdin_payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or not tmp_xvg.is_file():
        raise RuntimeError(f"gmx energy failed (rc={res.returncode}): {res.stderr.strip()}")
    tmp_xvg.replace(out_xvg)
    legends = read_xvg_legends(out_xvg)
    # Count data lines cheaply
    n_rows = sum(1 for ln in out_xvg.open("r", encoding="utf-8", errors="ignore") if ln.strip() and not ln.startswith(("#", "@")))
    return legends or matched, n_rows


def _has_panedr() -> bool:
    try:
        import panedr  # noqa: F401
        return True
    except Exception:
        return False


def sync_single_xvg_from_edr(
    edr_path: Path | None,
    xvg_path: Path,
    desired_terms: list[str] | tuple[str, ...],
    *,
    run_id: str = "",
    target: str = "thermo",
    backend: str = "auto",  # "auto" | "panedr" | "gmx"
    gmx_bin: str | None = None,
    force: bool = False,
) -> EdrSyncResult:
    """Ensure `xvg_path` exists and contains all `desired_terms` available in `edr_path`.

    - If `xvg_path` already exists, is newer than `edr_path`, and contains all terms present in `edr_path`,
      it is left untouched (`status="up_to_date"`).
    - If missing, stale (`edr` newer than `xvg`), missing requested terms, or `force=True`,
      re-extracts cleanly from `edr_path` using `panedr` or `gmx energy`.
    """
    existing_legends = read_xvg_legends(xvg_path)
    if edr_path is None or not edr_path.is_file():
        if xvg_path.is_file():
            return EdrSyncResult(
                run_id=run_id,
                target=target,
                xvg_path=xvg_path,
                edr_path=edr_path,
                status="up_to_date",
                columns=existing_legends,
                message="EDR not found; keeping existing XVG",
            )
        return EdrSyncResult(
            run_id=run_id,
            target=target,
            xvg_path=xvg_path,
            edr_path=edr_path,
            status="skipped_no_edr",
            message="Neither XVG nor EDR file found",
        )

    # Choose backend
    chosen = backend.lower()
    gmx_exe = _find_gmx_bin(gmx_bin)
    if chosen == "auto":
        if _has_panedr():
            chosen = "panedr"
        elif gmx_exe:
            chosen = "gmx"
        else:
            return EdrSyncResult(
                run_id=run_id,
                target=target,
                xvg_path=xvg_path,
                edr_path=edr_path,
                status="error",
                message="Neither `panedr` (Python) nor `gmx` (GROMACS) is available to extract EDR",
            )
    elif chosen == "gmx" and not gmx_exe:
        return EdrSyncResult(
            run_id=run_id,
            target=target,
            xvg_path=xvg_path,
            edr_path=edr_path,
            status="error",
            message="Requested backend='gmx' but gmx executable was not found on PATH",
        )

    # Merge any existing columns with desired_terms so we never drop columns the user already had
    combined_desired: list[str] = []
    for t in [*existing_legends, *desired_terms]:
        if t not in combined_desired:
            combined_desired.append(t)

    # Check if XVG already satisfies all desired terms and is up-to-date vs EDR
    is_stale = xvg_path.is_file() and (edr_path.stat().st_mtime > xvg_path.stat().st_mtime + 1.0)
    if xvg_path.is_file() and not force and not is_stale:
        # Check which desired terms are missing from existing_legends
        missing_prelim = [t for t in desired_terms if _resolve_term_name(t, existing_legends) is None]
        if not missing_prelim:
            return EdrSyncResult(
                run_id=run_id,
                target=target,
                xvg_path=xvg_path,
                edr_path=edr_path,
                status="up_to_date",
                backend=chosen,
                columns=existing_legends,
                message="All requested columns already present and newer than EDR",
            )

    existed_before = xvg_path.is_file()
    try:
        if chosen == "panedr":
            cols_out, n_rows = _extract_via_panedr(
                edr_path,
                combined_desired,
                xvg_path,
                title=f"GROMACS {target} energies",
            )
        else:
            assert gmx_exe is not None
            cols_out, n_rows = _extract_via_gmx(edr_path, combined_desired, xvg_path, gmx_exe)
    except Exception as exc:
        return EdrSyncResult(
            run_id=run_id,
            target=target,
            xvg_path=xvg_path,
            edr_path=edr_path,
            status="error",
            backend=chosen,
            columns=existing_legends,
            message=str(exc),
        )

    if not cols_out:
        return EdrSyncResult(
            run_id=run_id,
            target=target,
            xvg_path=xvg_path,
            edr_path=edr_path,
            status="skipped_no_terms",
            backend=chosen,
            columns=existing_legends,
            message="None of the requested terms were present in the EDR file",
        )

    existing_lower = {c.lower() for c in existing_legends}
    added = [c for c in cols_out if c.lower() not in existing_lower]
    status = "created" if not existed_before else ("updated" if (added or is_stale or force) else "up_to_date")
    return EdrSyncResult(
        run_id=run_id,
        target=target,
        xvg_path=xvg_path,
        edr_path=edr_path,
        status=status,
        backend=chosen,
        columns=cols_out,
        added_columns=added,
        n_rows=n_rows,
        message=f"Extracted {len(cols_out)} columns ({n_rows} rows) via {chosen}",
    )


def find_edr_in_run_dir(run_dir: Path) -> Path | None:
    """Locate the production .edr file in a SolvForge or standard flat GROMACS run directory."""
    candidates = [
        run_dir / "04_prod" / "prod.edr",
        run_dir / "prod.edr",
        run_dir / "md.edr",
    ]
    for c in candidates:
        if c.is_file():
            return c
    edrs = sorted(run_dir.rglob("*.edr"), key=lambda p: p.stat().st_mtime, reverse=True)
    return edrs[0] if edrs else None


def sync_runs_xvg_from_edr(
    runs: list[Any],
    *,
    extra_thermo_terms: list[str] | tuple[str, ...] = (),
    extra_stress_terms: list[str] | tuple[str, ...] = (),
    backend: str = "auto",
    gmx_bin: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Synchronize analysis/thermo.xvg and analysis/stress.xvg from prod.edr for all discovered RunRecords."""
    thermo_terms = [*DEFAULT_THERMO_TERMS, *extra_thermo_terms]
    stress_terms = [*DEFAULT_STRESS_TERMS, *extra_stress_terms]
    rows: list[dict[str, Any]] = []

    for run in runs:
        run_dir = Path(getattr(run, "path", "."))
        edr = getattr(run, "edr_path", None)
        if edr is None or not Path(edr).is_file():
            edr = find_edr_in_run_dir(run_dir)
        else:
            edr = Path(edr)

        thermo_p = Path(getattr(run, "thermo_path", run_dir / "analysis" / "thermo.xvg"))
        stress_p_raw = getattr(run, "stress_path", None)
        stress_p = Path(stress_p_raw) if stress_p_raw else (thermo_p.parent / "stress.xvg")

        res_th = sync_single_xvg_from_edr(
            edr,
            thermo_p,
            thermo_terms,
            run_id=getattr(run, "run_id", run_dir.name),
            target="thermo",
            backend=backend,
            gmx_bin=gmx_bin,
            force=force,
        )
        rows.append(res_th.to_dict())

        res_st = sync_single_xvg_from_edr(
            edr,
            stress_p,
            stress_terms,
            run_id=getattr(run, "run_id", run_dir.name),
            target="stress",
            backend=backend,
            gmx_bin=gmx_bin,
            force=force,
        )
        rows.append(res_st.to_dict())

    return pd.DataFrame(rows)


def sync_directory_xvg_from_edr(
    root_dir: Path,
    *,
    extra_thermo_terms: list[str] | tuple[str, ...] = (),
    extra_stress_terms: list[str] | tuple[str, ...] = (),
    backend: str = "auto",
    gmx_bin: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Standalone folder sync: auto-discovers single run dir or multi-run study dir containing *.edr files."""
    root = Path(root_dir).resolve()
    run_dirs: list[Path] = []
    if find_edr_in_run_dir(root) and ((root / "04_prod" / "prod.edr").is_file() or (root / "prod.edr").is_file() or (root / "00_input").is_dir()):
        run_dirs = [root]
    else:
        seen: set[Path] = set()
        for edr in sorted(root.rglob("*.edr")):
            # If inside 04_prod/, run root is parent of 04_prod; else parent of edr
            rdir = edr.parent.parent if edr.parent.name == "04_prod" else edr.parent
            if rdir not in seen:
                seen.add(rdir)
                run_dirs.append(rdir)

    thermo_terms = [*DEFAULT_THERMO_TERMS, *extra_thermo_terms]
    stress_terms = [*DEFAULT_STRESS_TERMS, *extra_stress_terms]
    rows: list[dict[str, Any]] = []
    for rdir in run_dirs:
        edr = find_edr_in_run_dir(rdir)
        analysis_dir = rdir / "analysis" if (rdir / "04_prod").is_dir() or (rdir / "analysis").is_dir() else rdir
        for target, fname, terms in (
            ("thermo", "thermo.xvg", thermo_terms),
            ("stress", "stress.xvg", stress_terms),
        ):
            res = sync_single_xvg_from_edr(
                edr,
                analysis_dir / fname,
                terms,
                run_id=rdir.name,
                target=target,
                backend=backend,
                gmx_bin=gmx_bin,
                force=force,
            )
            rows.append(res.to_dict())
    return pd.DataFrame(rows)
