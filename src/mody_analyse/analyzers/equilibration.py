from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from mody_analyse.core.config import Config
from mody_analyse.core.discovery import RunRecord, read_xvg


def _block_l2_convergence(block_paths: list[Path], t_eq_ns: float) -> tuple[float, float]:
    """Compare early (< t_eq) and late (>= t_eq) block RDF curves dynamically against the final window."""
    by_pair: dict[str, list[tuple[float, float, np.ndarray]]] = {}
    for p in block_paths:
        m = re.match(r"^(rdf_[A-Za-z0-9]+_[A-Za-z0-9]+)_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)ns$", p.stem)
        if not m:
            continue
        pair, t0, t1 = m.group(1), float(m.group(2)), float(m.group(3))
        _, arr = read_xvg(p)
        if arr.size == 0:
            continue
        by_pair.setdefault(pair, []).append((t0, t1, arr[:, 1]))

    early_l2: list[float] = []
    late_l2: list[float] = []
    for entries in by_pair.values():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda x: (x[1], x[0]))
        _, _, ref = entries[-1]
        for t0, t1, g_vals in entries[:-1]:
            if len(g_vals) != len(ref):
                continue
            l2 = float(np.sqrt(np.mean((g_vals - ref) ** 2)))
            if t1 <= t_eq_ns + 1e-6 or t0 < t_eq_ns * 0.5:
                early_l2.append(l2)
            else:
                late_l2.append(l2)

    return (
        float(np.mean(early_l2)) if early_l2 else float("nan"),
        float(np.mean(late_l2)) if late_l2 else float("nan"),
    )


def analyze_equilibration(runs: list[RunRecord], cfg: Config) -> pd.DataFrame:
    records: list[dict] = []
    t_eq_ps = cfg.t_eq_ns * 1000.0
    t_end_ps = cfg.t_end_ns * 1000.0

    for run in runs:
        cols, arr = read_xvg(run.thermo_path)
        t_ps = arr[:, cols.index("Time")]
        temp = arr[:, cols.index("Temperature")]
        dens = arr[:, cols.index("Density")]
        pot = arr[:, cols.index("Potential")]
        vol = arr[:, cols.index("Volume")]

        early_mask = t_ps < t_eq_ps
        eq_mask = (t_ps >= t_eq_ps) & (t_ps <= t_end_ps)
        if not np.any(eq_mask):
            eq_mask = t_ps >= float(np.min(t_ps))
        span_ps = max(t_end_ps - t_eq_ps, 0.0)
        tail_len = min(200_000.0, 0.5 * span_ps) if span_ps > 0 else 0.0
        tail_start_ps = max(t_eq_ps, float(np.max(t_ps[eq_mask])) - tail_len)
        tail_mask = (t_ps >= tail_start_ps) & (t_ps <= t_end_ps)
        if not np.any(tail_mask):
            tail_mask = eq_mask

        dens_early = float(np.mean(dens[early_mask])) if np.any(early_mask) else float("nan")
        dens_eq = float(np.mean(dens[eq_mask]))
        dens_eq_sd = float(np.std(dens[eq_mask], ddof=1)) if np.sum(eq_mask) > 1 else 0.0
        dens_tail = float(np.mean(dens[tail_mask]))
        dens_drift_pct = abs(dens_tail - dens_eq) / max(dens_eq, 1e-9) * 100.0

        pot_early = float(np.mean(pot[early_mask])) if np.any(early_mask) else float("nan")
        pot_eq = float(np.mean(pot[eq_mask]))
        pot_tail = float(np.mean(pot[tail_mask]))

        rdf_early_l2, rdf_late_l2 = _block_l2_convergence(run.block_rdf_paths, cfg.t_eq_ns)

        records.append(
            {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "replica": run.replica,
                "t_start_ns": float(t_ps[0] / 1000.0),
                "t_end_ns": float(t_ps[-1] / 1000.0),
                "t_eq_ns": cfg.t_eq_ns,
                "n_thermo_eq": int(np.sum(eq_mask)),
                "temp_mean_eq_K": float(np.mean(temp[eq_mask])),
                "volume_mean_eq_nm3": float(np.mean(vol[eq_mask])),
                "density_mean_early_kg_m3": dens_early,
                "density_mean_eq_kg_m3": dens_eq,
                "density_sd_eq_kg_m3": dens_eq_sd,
                "density_mean_tail_kg_m3": dens_tail,
                "density_mean_300_500ns_kg_m3": dens_tail,
                "density_drift_pct": float(dens_drift_pct),
                "potential_early_kJ_mol": pot_early,
                "potential_eq_kJ_mol": pot_eq,
                "potential_tail_kJ_mol": pot_tail,
                "rdf_l2_early_vs_late": rdf_early_l2,
                "rdf_l2_0_100_vs_400_500": rdf_early_l2,
                "rdf_l2_late_mean": rdf_late_l2,
            }
        )

    return pd.DataFrame(records)


def analyze_equilibration_windows(runs: list[RunRecord], window_ns: float | None = None) -> pd.DataFrame:
    """Compute thermo averages in fixed time windows across the full trajectory."""
    rows: list[dict] = []
    for run in runs:
        cols, arr = read_xvg(run.thermo_path)
        t_ps = arr[:, cols.index("Time")]
        temp = arr[:, cols.index("Temperature")]
        dens = arr[:, cols.index("Density")]
        pot = arr[:, cols.index("Potential")]
        vol = arr[:, cols.index("Volume")]
        t_max_ns = max(float(t_ps[-1]) / 1000.0, 1e-6)
        w_ns = window_ns if window_ns is not None else max(1.0, round(t_max_ns / 10.0, 2))
        win_ps = w_ns * 1000.0
        n_win = max(1, int(round(float(t_ps[-1]) / win_ps)))
        for wi in range(n_win):
            w0 = wi * win_ps
            w1 = (wi + 1) * win_ps
            mask = (t_ps >= w0) & (t_ps <= w1 if wi == n_win - 1 else t_ps < w1)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "run_id": run.run_id,
                    "system_code": run.system_code,
                    "replica": run.replica,
                    "window_ns": f"{int(round(wi * w_ns))}-{int(round((wi + 1) * w_ns))}",
                    "t_mid_ns": (wi + 0.5) * w_ns,
                    "temp_mean_K": float(np.mean(temp[mask])),
                    "density_mean_kg_m3": float(np.mean(dens[mask])),
                    "density_sd_kg_m3": float(np.std(dens[mask], ddof=1)) if np.sum(mask) > 1 else 0.0,
                    "potential_mean_kJ_mol": float(np.mean(pot[mask])),
                    "volume_mean_nm3": float(np.mean(vol[mask])),
                }
            )
    return pd.DataFrame(rows)
