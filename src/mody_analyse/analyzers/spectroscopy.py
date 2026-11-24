from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from mody_analyse.core.config import Config
from mody_analyse.core.discovery import RunRecord, equilibrated_frame_slice

C_CM_PER_PS = 2.99792458e-2  # speed of light in cm / ps -> 1 THz = 33.35641 cm^-1
THZ_TO_CM1 = 1.0 / C_CM_PER_PS
HC_OVER_KB_CM_K = 1.438776877  # h c / k_B in cm * K


def _fft_acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Fast unbiased autocorrelation for 1D signal using zero-padded FFT."""
    n = len(x)
    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=n_fft)
    acf_raw = np.fft.irfft(fx * np.conj(fx), n=n_fft)[: max_lag + 1]
    counts = np.arange(n, n - max_lag - 1, -1, dtype=float)
    return acf_raw / np.maximum(counts, 1.0)


def compute_vdos_spectrum(
    velocities_nm_ps: np.ndarray,
    masses_amu: np.ndarray,
    dt_ps: float,
    *,
    max_lag_frames: int | None = None,
) -> dict[str, Any]:
    """Compute mass-weighted VACF and Vibrational Density of States (VDOS) power spectrum.

    velocities_nm_ps: array of shape (n_frames, n_atoms, 3) in nm/ps.
    masses_amu: array of shape (n_atoms,) in amu.
    Returns wavenumber (cm^-1), frequency (THz), normalized VDOS, VACF, peak table, and D_vacf (m^2/s).
    """
    v = np.asarray(velocities_nm_ps, dtype=float)
    m = np.asarray(masses_amu, dtype=float)
    n_frames, n_atoms, _ = v.shape
    if n_frames < 8 or dt_ps <= 0:
        return {
            "wavenumber_cm1": np.array([]),
            "freq_thz": np.array([]),
            "vdos": np.array([]),
            "vacf": np.array([]),
            "t_ps": np.array([]),
            "d_vacf_m2_s": float("nan"),
            "top_peaks_cm1": [],
        }

    max_lag = min(max_lag_frames or (n_frames // 2), n_frames - 2)
    # Mass-weighted sum of 1D autocorrelations across all (atom, xyz)
    vacf_num = np.zeros(max_lag + 1, dtype=float)
    vacf_unweighted = np.zeros(max_lag + 1, dtype=float)
    m_sum = max(float(np.sum(m)), 1e-12)
    for ai in range(n_atoms):
        w = float(m[ai])
        atom_acf = np.zeros(max_lag + 1, dtype=float)
        for d in range(3):
            atom_acf += _fft_acf_1d(v[:, ai, d], max_lag)
        vacf_num += w * atom_acf
        vacf_unweighted += atom_acf

    c0 = vacf_num[0]
    vacf_norm = vacf_num / c0 if abs(c0) > 1e-18 else np.zeros_like(vacf_num)
    t_ps = np.arange(max_lag + 1, dtype=float) * float(dt_ps)

    # Green-Kubo VACF diffusivity check: D = (1/3) int_0^inf <v(0).v(t)> dt
    mean_v_acf = vacf_unweighted / max(n_atoms, 1)  # nm^2 / ps^2
    d_vacf_nm2_ps = (1.0 / 3.0) * float(trapezoid(mean_v_acf, t_ps))
    d_vacf_m2_s = d_vacf_nm2_ps * 1e-6

    # Apply half-Hann apodization window to smooth finite-lag truncation and compute cosine transform
    window = 0.5 * (1.0 + np.cos(math.pi * np.arange(max_lag + 1) / max(max_lag, 1)))
    c_win = vacf_norm * window
    # Mirror for symmetric real FFT -> exact cosine Fourier transform
    n_pad = max(2048, 1 << (2 * max_lag).bit_length())
    sym = np.zeros(n_pad, dtype=float)
    sym[: max_lag + 1] = c_win
    sym[n_pad - max_lag :] = c_win[1:][::-1]
    spec = np.real(np.fft.rfft(sym)) * float(dt_ps)
    spec = np.maximum(spec, 0.0)
    freqs_thz = np.fft.rfftfreq(n_pad, d=float(dt_ps))
    wavenum = freqs_thz * THZ_TO_CM1

    area = float(trapezoid(spec, wavenum))
    vdos_norm = spec / area if area > 1e-18 else spec

    peaks, _ = find_peaks(vdos_norm, height=float(np.max(vdos_norm)) * 0.05 if len(vdos_norm) else 0.0, distance=5)
    top_idx = sorted(peaks, key=lambda idx: vdos_norm[idx], reverse=True)[:8]
    top_peaks = [float(wavenum[i]) for i in top_idx]

    return {
        "wavenumber_cm1": wavenum,
        "freq_thz": freqs_thz,
        "vdos": vdos_norm,
        "vacf": vacf_norm,
        "t_ps": t_ps,
        "d_vacf_m2_s": d_vacf_m2_s,
        "top_peaks_cm1": top_peaks,
    }


def compute_ir_spectrum(
    dipole_derivative_e_nm_ps: np.ndarray,
    dt_ps: float,
    temperature_k: float,
    *,
    quantum_correction: str = "harmonic",
    max_lag_frames: int | None = None,
) -> dict[str, Any]:
    """Compute IR / Far-IR / THz absorption spectrum from total dipole current J_M(t) = dM/dt (n_frames, 3).

    I_IR(nu) ~ Q(nu) * Re[ integral exp(-i omega t) <dM/dt(0) . dM/dt(t)> dt ]
    where Q_harmonic(nu) = x / (1 - exp(-x)), x = h c nu / (k_B T).
    """
    jm = np.asarray(dipole_derivative_e_nm_ps, dtype=float)
    n_frames = jm.shape[0]
    if n_frames < 8 or dt_ps <= 0:
        return {
            "wavenumber_cm1": np.array([]),
            "freq_thz": np.array([]),
            "intensity_arb": np.array([]),
            "top_peaks_cm1": [],
        }

    jm_centered = jm - np.mean(jm, axis=0, keepdims=True)
    max_lag = min(max_lag_frames or (n_frames // 2), n_frames - 2)
    acf = np.zeros(max_lag + 1, dtype=float)
    for d in range(3):
        acf += _fft_acf_1d(jm_centered[:, d], max_lag)

    window = 0.5 * (1.0 + np.cos(math.pi * np.arange(max_lag + 1) / max(max_lag, 1)))
    c_win = acf * window
    n_pad = max(2048, 1 << (2 * max_lag).bit_length())
    sym = np.zeros(n_pad, dtype=float)
    sym[: max_lag + 1] = c_win
    sym[n_pad - max_lag :] = c_win[1:][::-1]
    spec = np.maximum(np.real(np.fft.rfft(sym)) * float(dt_ps), 0.0)
    freqs_thz = np.fft.rfftfreq(n_pad, d=float(dt_ps))
    wavenum = freqs_thz * THZ_TO_CM1

    if quantum_correction == "harmonic" and temperature_k > 0:
        x = HC_OVER_KB_CM_K * wavenum / float(temperature_k)
        # Safe limit x/(1 - exp(-x)) -> 1 as x -> 0
        q_corr = np.where(x > 1e-6, x / np.maximum(1.0 - np.exp(-np.clip(x, 0.0, 500.0)), 1e-18), 1.0)
        spec = spec * q_corr

    peak_max = float(np.max(spec)) if len(spec) else 0.0
    spec_norm = spec / peak_max if peak_max > 1e-18 else spec
    peaks, _ = find_peaks(spec_norm, height=0.05, distance=5)
    top_idx = sorted(peaks, key=lambda idx: spec_norm[idx], reverse=True)[:8]
    return {
        "wavenumber_cm1": wavenum,
        "freq_thz": freqs_thz,
        "intensity_arb": spec_norm,
        "top_peaks_cm1": [float(wavenum[i]) for i in top_idx],
    }


def run_spectroscopy_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Run VDOS and IR/Far-IR/THz spectroscopy on runs with sufficiently fine time spacing."""
    opts = (cfg.analyses or {}).get("options") or {}
    max_dt_ps = float(opts.get("vdos_max_dt_ps", 0.05))
    summary_rows: list[dict[str, Any]] = []
    vdos_rows: list[dict[str, Any]] = []
    ir_rows: list[dict[str, Any]] = []

    for run in runs:
        trr = run.tpr_path.parent / "prod.trr"
        traj = trr if trr.exists() else run.xtc_path
        if not (run.tpr_path.exists() and traj.exists()):
            continue
        u = mda.Universe(str(run.tpr_path), str(traj))
        f_start, f_stop, f_step = equilibrated_frame_slice(u, cfg.t_eq_ns, cfg.t_end_ns, stride=1)
        frames = list(range(f_start, f_stop, f_step))
        if len(frames) < 16:
            continue
        # Cap frame count for memory efficiency (e.g. up to 4000 contiguous frames)
        frames = frames[:4000]
        u.trajectory[frames[0]]
        t0 = float(u.trajectory.ts.time)
        u.trajectory[frames[1]]
        t1 = float(u.trajectory.ts.time)
        dt_ps = t1 - t0
        if dt_ps <= 0 or dt_ps > max_dt_ps:
            continue

        has_vel = bool(u.trajectory.ts.has_velocities)
        temp_k = float(run.metadata.get("temperatureK") or run.metadata.get("temperature_k") or cfg.temperature_k)

        # Build atom charges & masses arrays
        all_atoms = u.atoms
        charges = np.zeros(len(all_atoms), dtype=float)
        masses = np.asarray(all_atoms.masses, dtype=float)
        # Populate charges from parsed .itp topologies if available
        for sp, topo in run.topologies.items():
            ag = u.select_atoms(f"resname {sp}")
            n_at = len(topo.atom_names)
            if n_at > 0 and len(ag) > 0 and len(topo.charges) == n_at:
                n_mol = len(ag) // n_at
                charges[ag.indices[: n_mol * n_at]] = np.tile(np.asarray(topo.charges, dtype=float), n_mol)

        has_charges = bool(np.any(np.abs(charges) > 1e-6))
        vels_list: list[np.ndarray] = []
        jm_list: list[np.ndarray] = []
        m_list: list[np.ndarray] = []
        for fi in frames:
            u.trajectory[fi]
            ts = u.trajectory.ts
            if has_vel:
                v_nm_ps = np.asarray(ts.velocities, dtype=float) / 10.0  # A/ps -> nm/ps
                vels_list.append(v_nm_ps)
                if has_charges:
                    jm_list.append(np.sum(v_nm_ps * charges[:, None], axis=0))
            elif has_charges:
                pos_nm = np.asarray(ts.positions, dtype=float) / 10.0
                m_list.append(np.sum(pos_nm * charges[:, None], axis=0))

        vdos_peaks: list[float] = []
        ir_peaks: list[float] = []
        d_vacf = float("nan")

        if vels_list:
            v_arr = np.stack(vels_list, axis=0)
            vd = compute_vdos_spectrum(v_arr, masses, dt_ps)
            vdos_peaks = vd["top_peaks_cm1"]
            d_vacf = vd["d_vacf_m2_s"]
            # Downsample spectrum to <= 500 rows per run for compact CSV
            step = max(1, len(vd["wavenumber_cm1"]) // 400)
            for wn, ft, intensity in zip(
                vd["wavenumber_cm1"][::step], vd["freq_thz"][::step], vd["vdos"][::step]
            ):
                vdos_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "wavenumber_cm1": float(wn),
                        "freq_thz": float(ft),
                        "vdos_norm": float(intensity),
                    }
                )

        if jm_list:
            ir = compute_ir_spectrum(np.stack(jm_list, axis=0), dt_ps, temp_k)
        elif len(m_list) >= 10:
            jm_fd = np.diff(np.stack(m_list, axis=0), axis=0) / dt_ps
            ir = compute_ir_spectrum(jm_fd, dt_ps, temp_k)
        else:
            ir = None

        if ir is not None and len(ir["wavenumber_cm1"]):
            ir_peaks = ir["top_peaks_cm1"]
            step = max(1, len(ir["wavenumber_cm1"]) // 400)
            for wn, ft, intensity in zip(
                ir["wavenumber_cm1"][::step], ir["freq_thz"][::step], ir["intensity_arb"][::step]
            ):
                ir_rows.append(
                    {
                        "run_id": run.run_id,
                        "system_code": run.system_code,
                        "wavenumber_cm1": float(wn),
                        "freq_thz": float(ft),
                        "ir_intensity_norm": float(intensity),
                    }
                )

        summary_rows.append(
            {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "replica": run.replica,
                "dt_ps": dt_ps,
                "nyquist_cm1": 0.5 * THZ_TO_CM1 / dt_ps,
                "n_frames_used": len(frames),
                "has_velocities": has_vel,
                "d_vacf_1e9_m2_s": d_vacf * 1e9 if np.isfinite(d_vacf) else float("nan"),
                "vdos_top_peaks_cm1": ", ".join(f"{p:.1f}" for p in vdos_peaks[:5]),
                "ir_top_peaks_cm1": ", ".join(f"{p:.1f}" for p in ir_peaks[:5]),
            }
        )

    return {
        "spectroscopy_summary": pd.DataFrame(summary_rows),
        "spectroscopy_vdos": pd.DataFrame(vdos_rows),
        "spectroscopy_ir": pd.DataFrame(ir_rows),
    }
