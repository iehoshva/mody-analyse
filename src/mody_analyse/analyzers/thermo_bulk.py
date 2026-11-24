from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from mody_analyse.core.config import Config
from mody_analyse.core.discovery import RunRecord, read_xvg_dataframe

AVOGADRO = 6.02214076e23
KB_J_K = 1.380649e-23
R_J_MOL_K = 8.314462618
NM3_TO_M3 = 1e-27
KJ_MOL_PER_NM3_TO_MPA = 1000.0 / (AVOGADRO * NM3_TO_M3 * 1e6)  # ~1.660539067 MPa


def compute_npt_fluctuations(
    temp_k: np.ndarray,
    vol_nm3: np.ndarray,
    enthalpy_kj_mol: np.ndarray,
    n_molecules: int,
    *,
    potential_kj_mol: np.ndarray | None = None,
    lj_kj_mol: np.ndarray | None = None,
    coul_kj_mol: np.ndarray | None = None,
    gas_energy_kj_mol: float | None = None,
    n_blocks: int = 5,
) -> dict[str, Any]:
    """Compute NPT bulk thermodynamic response functions (Cp, κT, αP, Cv) and solubility parameters.

    All GROMACS box energies are in kJ/(mol of simulation boxes); volumes are in nm^3.
    Outputs:
      - cp_j_mol_k, cv_j_mol_k: molar heat capacities (J mol^-1 K^-1)
      - kappa_t_gpa_inv: isothermal compressibility (GPa^-1)
      - alpha_p_1e3_k_inv: thermal expansion coefficient (10^-3 K^-1)
      - ced_mpa, hildebrand_delta_mpa_half, hansen_delta_d/p_mpa_half
    """
    T = np.asarray(temp_k, dtype=float)
    V = np.asarray(vol_nm3, dtype=float)
    H = np.asarray(enthalpy_kj_mol, dtype=float)
    n_mol = max(1, int(n_molecules))
    n = min(len(T), len(V), len(H))
    if n < 4:
        return {
            "temperature_mean_k": float(np.mean(T)) if len(T) else float("nan"),
            "volume_mean_nm3": float(np.mean(V)) if len(V) else float("nan"),
            "molar_volume_cm3_mol": float("nan"),
            "cp_j_mol_k": float("nan"),
            "cp_sem_j_mol_k": float("nan"),
            "cv_j_mol_k": float("nan"),
            "kappa_t_gpa_inv": float("nan"),
            "kappa_t_sem_gpa_inv": float("nan"),
            "alpha_p_1e3_k_inv": float("nan"),
            "alpha_p_sem_1e3_k_inv": float("nan"),
            "ced_mpa": float("nan"),
            "hildebrand_delta_mpa_half": float("nan"),
            "hansen_delta_d_mpa_half": float("nan"),
            "hansen_delta_p_mpa_half": float("nan"),
            "ced_mode": "insufficient_samples",
        }

    T, V, H = T[:n], V[:n], H[:n]

    def _calc_window(t_w: np.ndarray, v_w: np.ndarray, h_w: np.ndarray) -> tuple[float, float, float, float]:
        t_mean = float(np.mean(t_w))
        v_mean = float(np.mean(v_w))
        dh_j = (h_w - float(np.mean(h_w))) * 1000.0  # J/mol-box
        dv = v_w - v_mean  # nm^3
        cp = float(np.mean(dh_j * dh_j)) / (n_mol * R_J_MOL_K * (t_mean ** 2))
        kappa_pa = float(np.mean(dv * dv)) * NM3_TO_M3 / (KB_J_K * t_mean * max(v_mean, 1e-18))
        kappa_gpa = kappa_pa * 1e9
        alpha_k = float(np.mean(dh_j * dv)) / (R_J_MOL_K * (t_mean ** 2) * max(v_mean, 1e-18))
        vm_m3 = (v_mean * NM3_TO_M3 * AVOGADRO) / n_mol
        cv = cp - (t_mean * vm_m3 * (alpha_k ** 2) / kappa_pa) if kappa_pa > 1e-30 else float("nan")
        return cp, kappa_gpa, alpha_k * 1e3, cv

    cp_all, kappa_all, alpha_all, cv_all = _calc_window(T, V, H)

    # Block averaging for uncertainty
    nb = min(max(2, int(n_blocks)), n // 2)
    splits = np.array_split(np.arange(n), nb)
    blk_cp, blk_k, blk_a = [], [], []
    for idx in splits:
        if len(idx) >= 2:
            c_b, k_b, a_b, _ = _calc_window(T[idx], V[idx], H[idx])
            blk_cp.append(c_b)
            blk_k.append(k_b)
            blk_a.append(a_b)

    def _sem(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))

    v_mean_nm3 = float(np.mean(V))
    molar_vol_cm3 = (v_mean_nm3 * 1e-21 * AVOGADRO) / n_mol

    # Cohesive Energy Density & Hildebrand / Hansen solubility parameters
    ced_mpa = float("nan")
    delta_hil = float("nan")
    delta_d = float("nan")
    delta_p = float("nan")
    ced_mode = "unavailable"

    if gas_energy_kj_mol is not None and potential_kj_mol is not None and len(potential_kj_mol) >= n:
        e_liq = float(np.mean(np.asarray(potential_kj_mol, dtype=float)[:n]))
        e_coh = float(gas_energy_kj_mol) - e_liq
        if e_coh > 0 and v_mean_nm3 > 0:
            ced_mpa = e_coh * KJ_MOL_PER_NM3_TO_MPA / v_mean_nm3
            delta_hil = math.sqrt(ced_mpa)
            ced_mode = "gas_reference"
    elif lj_kj_mol is not None or coul_kj_mol is not None:
        e_lj = float(np.mean(np.asarray(lj_kj_mol, dtype=float)[:n])) if lj_kj_mol is not None else 0.0
        e_coul = float(np.mean(np.asarray(coul_kj_mol, dtype=float)[:n])) if coul_kj_mol is not None else 0.0
        e_inter_coh = -(e_lj + e_coul)
        if e_inter_coh > 0 and v_mean_nm3 > 0:
            ced_mpa = e_inter_coh * KJ_MOL_PER_NM3_TO_MPA / v_mean_nm3
            delta_hil = math.sqrt(ced_mpa)
            ced_mode = "intermolecular_nonbonded_approx"
        if -e_lj > 0 and v_mean_nm3 > 0:
            delta_d = math.sqrt((-e_lj) * KJ_MOL_PER_NM3_TO_MPA / v_mean_nm3)
        if -e_coul > 0 and v_mean_nm3 > 0:
            delta_p = math.sqrt((-e_coul) * KJ_MOL_PER_NM3_TO_MPA / v_mean_nm3)

    if lj_kj_mol is not None and not np.isfinite(delta_d):
        e_lj = float(np.mean(np.asarray(lj_kj_mol, dtype=float)[:n]))
        if -e_lj > 0 and v_mean_nm3 > 0:
            delta_d = math.sqrt((-e_lj) * KJ_MOL_PER_NM3_TO_MPA / v_mean_nm3)
    if coul_kj_mol is not None and not np.isfinite(delta_p):
        e_coul = float(np.mean(np.asarray(coul_kj_mol, dtype=float)[:n]))
        if -e_coul > 0 and v_mean_nm3 > 0:
            delta_p = math.sqrt((-e_coul) * KJ_MOL_PER_NM3_TO_MPA / v_mean_nm3)

    return {
        "temperature_mean_k": float(np.mean(T)),
        "volume_mean_nm3": v_mean_nm3,
        "molar_volume_cm3_mol": float(molar_vol_cm3),
        "cp_j_mol_k": float(cp_all),
        "cp_sem_j_mol_k": _sem(blk_cp),
        "cv_j_mol_k": float(cv_all),
        "kappa_t_gpa_inv": float(kappa_all),
        "kappa_t_sem_gpa_inv": _sem(blk_k),
        "alpha_p_1e3_k_inv": float(alpha_all),
        "alpha_p_sem_1e3_k_inv": _sem(blk_a),
        "ced_mpa": float(ced_mpa),
        "hildebrand_delta_mpa_half": float(delta_hil),
        "hansen_delta_d_mpa_half": float(delta_d),
        "hansen_delta_p_mpa_half": float(delta_p),
        "ced_mode": ced_mode,
    }


def _match_col(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for kw in keywords:
        for lc, orig in lower_map.items():
            if kw in lc:
                return orig
    return None


def run_thermo_bulk_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Run NPT bulk fluctuation thermodynamics & solubility parameter analysis across all runs."""
    opts = (cfg.analyses or {}).get("options") or {}
    gas_map = dict(opts.get("gas_energies_kj_mol") or {})
    n_blocks = int(opts.get("n_blocks", 5))
    t_eq_ps = float(cfg.t_eq_ns) * 1000.0

    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.thermo_path.exists():
            continue
        df = read_xvg_dataframe(run.thermo_path)
        if df.empty:
            continue
        time_col = df.columns[0]
        eq_df = df[df[time_col] >= t_eq_ps]
        if len(eq_df) < 10:
            eq_df = df.iloc[len(df) // 2 :] if len(df) >= 8 else df

        col_t = _match_col(eq_df, ("temperature", "temp"))
        col_v = _match_col(eq_df, ("volume", "vol"))
        col_h = _match_col(eq_df, ("enthalpy", "total energy", "total-energy", "potential"))
        col_pot = _match_col(eq_df, ("potential",))
        col_p = _match_col(eq_df, ("pressure", "pres"))
        col_lj = _match_col(eq_df, ("lj (sr)", "lj-sr", "lj_sr"))
        col_coul_sr = _match_col(eq_df, ("coulomb (sr)", "coul-sr", "coulomb-sr"))
        col_coul_rec = _match_col(eq_df, ("coul. recip", "coul-recip"))

        if col_t is None or col_v is None or col_h is None:
            continue

        t_arr = eq_df[col_t].to_numpy(dtype=float)
        v_arr = eq_df[col_v].to_numpy(dtype=float)
        h_arr = eq_df[col_h].to_numpy(dtype=float)
        # If only Total Energy or Potential was available and Pressure is present, add P*V term (1 bar * nm^3 = 0.0602214 kJ/mol)
        if "enthalpy" not in col_h.lower() and col_p is not None:
            p_bar = eq_df[col_p].to_numpy(dtype=float)
            h_arr = h_arr + p_bar * v_arr * 0.0602214076

        pot_arr = eq_df[col_pot].to_numpy(dtype=float) if col_pot else None
        lj_arr = eq_df[col_lj].to_numpy(dtype=float) if col_lj else None
        coul_arr = None
        if col_coul_sr or col_coul_rec:
            coul_arr = np.zeros(len(eq_df), dtype=float)
            if col_coul_sr:
                coul_arr += eq_df[col_coul_sr].to_numpy(dtype=float)
            if col_coul_rec:
                coul_arr += eq_df[col_coul_rec].to_numpy(dtype=float)

        # Compute total gas-phase reference energy for the box if species gas energies provided
        gas_box_kj: float | None = None
        if gas_map and run.molecule_counts:
            if all(sp in gas_map for sp in run.molecule_counts if run.molecule_counts[sp] > 0):
                gas_box_kj = sum(float(gas_map[sp]) * cnt for sp, cnt in run.molecule_counts.items() if cnt > 0)

        n_mol_total = max(1, sum(run.molecule_counts.values()))
        res = compute_npt_fluctuations(
            t_arr,
            v_arr,
            h_arr,
            n_mol_total,
            potential_kj_mol=pot_arr,
            lj_kj_mol=lj_arr,
            coul_kj_mol=coul_arr,
            gas_energy_kj_mol=gas_box_kj,
            n_blocks=n_blocks,
        )
        rows.append(
            {
                "run_id": run.run_id,
                "system_code": run.system_code,
                "replica": run.replica,
                "n_molecules": n_mol_total,
                "n_samples": len(eq_df),
                **res,
            }
        )

    return {"thermo_fluctuations": pd.DataFrame(rows)}
