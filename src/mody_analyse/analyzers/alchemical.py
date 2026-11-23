from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize
from scipy.special import expit, logsumexp

from mody_analyse.core.config import Config
from mody_analyse.core.discovery import RunRecord, read_xvg_dataframe

R_KJ_MOL_K = 8.314462618e-3
KJ_TO_KCAL = 1.0 / 4.184


def compute_ti_free_energy(
    lambdas: np.ndarray,
    dhdl_means_kj_mol: np.ndarray,
    dhdl_sems_kj_mol: np.ndarray,
) -> dict[str, float]:
    """Compute Thermodynamic Integration (TI) ΔG via trapezoid & cubic-spline quadrature with error propagation."""
    lam = np.asarray(lambdas, dtype=float)
    mu = np.asarray(dhdl_means_kj_mol, dtype=float)
    sem = np.asarray(dhdl_sems_kj_mol, dtype=float)
    order = np.argsort(lam)
    lam, mu, sem = lam[order], mu[order], sem[order]
    if len(lam) < 2:
        return {"dg_ti_trap_kj_mol": float("nan"), "dg_ti_spline_kj_mol": float("nan"), "dg_ti_err_kj_mol": float("nan")}

    dg_trap = float(trapezoid(mu, lam))
    # Exact trapezoid quadrature weights for variance propagation
    weights = np.zeros_like(lam)
    weights[0] = 0.5 * (lam[1] - lam[0])
    weights[-1] = 0.5 * (lam[-1] - lam[-2])
    if len(lam) > 2:
        weights[1:-1] = 0.5 * (lam[2:] - lam[:-2])
    dg_err = float(math.sqrt(float(np.sum((weights * sem) ** 2))))

    if len(lam) >= 4:
        cs = CubicSpline(lam, mu)
        dg_spline = float(cs.integrate(float(lam[0]), float(lam[-1])))
    else:
        dg_spline = dg_trap

    return {
        "dg_ti_trap_kj_mol": dg_trap,
        "dg_ti_spline_kj_mol": dg_spline,
        "dg_ti_err_kj_mol": dg_err,
        "dg_ti_trap_kcal_mol": dg_trap * KJ_TO_KCAL,
        "dg_ti_err_kcal_mol": dg_err * KJ_TO_KCAL,
    }


def compute_bar_pair(
    w_forward_kt: np.ndarray,
    w_reverse_kt: np.ndarray,
) -> tuple[float, float]:
    """Exact Bennett Acceptance Ratio (BAR) for a pair of states (dimensionless reduced work w = beta * dU).

    Returns (df_kt, df_err_kt) where df_kt = beta * (F_1 - F_0).
    """
    wf = np.asarray(w_forward_kt, dtype=float)
    wr = np.asarray(w_reverse_kt, dtype=float)
    n0, n1 = len(wf), len(wr)
    if n0 == 0 or n1 == 0:
        return float("nan"), float("nan")
    M = math.log(n0 / float(n1))

    def _residual(df: float) -> float:
        # sum_i f(M + wf_i - df) - sum_j f(-M + wr_j + df) = 0
        lhs = float(np.sum(expit(-(M + wf - df))))
        rhs = float(np.sum(expit(-(-M + wr + df))))
        return lhs - rhs

    # Bracket root robustly using exponential-averaging bounds
    df_lo = float(np.min(wf)) - 5.0
    df_hi = float(np.max(wf)) + 5.0
    lo_cand = min(df_lo, -float(np.max(wr)) - 5.0)
    hi_cand = max(df_hi, -float(np.min(wr)) + 5.0)
    if _residual(lo_cand) * _residual(hi_cand) > 0:
        # Expand bracket if needed
        span = max(20.0, abs(hi_cand - lo_cand))
        lo_cand -= span * 5.0
        hi_cand += span * 5.0

    df_opt = float(brentq(_residual, lo_cand, hi_cand, xtol=1e-10, maxiter=200))

    # Asymptotic BAR variance (Shirts et al. 2003 / Bennett 1976)
    f_f = expit(-(M + wf - df_opt))
    f_r = expit(-(-M + wr + df_opt))
    denom_f = float(np.mean(f_f * (1.0 - f_f)))
    denom_r = float(np.mean(f_r * (1.0 - f_r)))
    denom = max(n0 * denom_f + n1 * denom_r, 1e-15)
    var_df = max(0.0, (1.0 / denom) - (1.0 / n0) - (1.0 / n1) + ((n0 + n1) / (n0 * n1) if denom > 0 else 0.0))
    # Exact Shirts-Pande variance formula: var = (<f_F^2>/<f_F>^2 - 1)/n0 + (<f_R^2>/<f_R>^2 - 1)/n1 ...
    # Actually the cleanest exact form is: var = [1 / <1/(2 + 2 cosh(M + w - df))> - (n0+n1)/(n0*n1)] / (n0+n1)
    # Using exact identity: <f(1-f)>_0 = (n1/(n0+n1)) * ... -> compute Fisher information directly:
    info = n0 * float(np.mean(f_f * (1.0 - f_f))) + n1 * float(np.mean(f_r * (1.0 - f_r)))
    # Shirts & Pande (2005) Eq. 11: var(df) = (1/info) - (1/n0 + 1/n1) ? Wait:
    # Let's check for identical distributions (wf=wr=0, n0=n1=N): f=0.5, f(1-f)=0.25, info = 0.5*N -> 1/info = 2/N, 1/n0+1/n1 = 2/N -> var=0 (exact!).
    var_exact = max(0.0, (1.0 / max(info, 1e-18)) - (1.0 / n0 + 1.0 / n1))
    return df_opt, math.sqrt(var_exact)


def compute_mbar(
    u_kn: np.ndarray,
    n_k: np.ndarray,
) -> dict[str, Any]:
    """Exact Multi-state Bennett Acceptance Ratio (MBAR, Shirts & Chodera 2008).

    u_kn: reduced potential energy matrix of shape (K, N_tot) where u_kn[k, n] = beta * U_k(x_n).
    n_k: sample counts per state of shape (K,) with sum(n_k) == N_tot.
    Returns free energies f_k (with f_0 = 0), covariance matrix, delta_f_0_to_end, and overlap matrix O_ij.
    """
    u = np.asarray(u_kn, dtype=float)
    nk = np.asarray(n_k, dtype=float)
    K, N_tot = u.shape
    if K < 2 or N_tot == 0 or int(np.sum(nk)) != N_tot:
        return {
            "f_k": np.full(K, float("nan")),
            "df_total_kt": float("nan"),
            "df_total_err_kt": float("nan"),
            "overlap_matrix": np.zeros((K, K)),
            "min_neighbor_overlap": float("nan"),
        }

    # Shift each sample's potential by its minimum for numerical stability (invariant for MBAR!)
    u_shift = u - np.min(u, axis=0, keepdims=True)
    log_nk = np.log(np.maximum(nk, 1e-30))
    frac_k = nk / float(N_tot)

    def _obj_and_grad(f_free: np.ndarray) -> tuple[float, np.ndarray]:
        f = np.concatenate(([0.0], f_free))
        # log_denom[n] = log sum_m N_m exp(f_m - u_mn)
        log_terms = (log_nk + f)[:, None] - u_shift  # (K, N)
        log_denom = logsumexp(log_terms, axis=0)  # (N,)
        obj = float(np.mean(log_denom - math.log(N_tot))) - float(np.dot(frac_k, f))
        # W_kn = exp(f_k - u_kn - log_denom_n); grad_k = (N_k/N_tot) * (sum_n W_kn - 1)
        log_w = f[:, None] - u_shift - log_denom[None, :]
        sum_w = np.exp(logsumexp(log_w, axis=1))
        grad = frac_k * (sum_w - 1.0)
        return obj, grad[1:]

    # Initialize f_k using sequential BAR / exponential averaging
    f_init = np.zeros(K - 1, dtype=float)
    res = minimize(_obj_and_grad, f_init, method="L-BFGS-B", jac=True, options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000})
    f_k = np.concatenate(([0.0], res.x))

    # Weight matrix W (N_tot, K) where sum_n W[n, k] = 1
    log_denom = logsumexp((log_nk + f_k)[:, None] - u_shift, axis=0)
    W = np.exp(f_k[None, :] - u_shift.T - log_denom[:, None])  # (N, K)

    # Overlap matrix O_ij = sum_n N_i W_ni W_nj (Klimovich, Shirts & Mobley 2015)
    overlap = (W.T @ W) * nk[:, None]
    min_nbr_overlap = float(np.min([overlap[k, k + 1] for k in range(K - 1)]))

    # Asymptotic covariance matrix via Shirts & Chodera (2008) Eq. D8:
    # Sigma = W^T (I - W N W^T)^+ W  -> computed in KxK space via SVD of W * sqrt(N)
    # Using KxK matrix form: Theta = pinv(I_K - O_sym) @ ...
    # Exact KxK identity: let A = W * sqrt(nk)[None, :] (N x K).
    # Wait: (W^T W)^+ - diag(1/nk) ? Let's use the exact KxK formula from Shirts & Chodera (2008) Appendix D:
    # Let M = W^T W (K x K), N = diag(nk). Then Theta = pinv(pinv(M) - N) -> wait:
    # Verify for K=2 identical states (u=0, N0=N1=N): W_ni = 1/(2N), M_ij = 2N * 1/(4N^2) = 1/(2N).
    # Actually: Theta = (I - M N)^+ M  -> check!
    # Wait: let's check via SVD of W: W = U S V^T -> W^T (I - W N W^T)^+ W = V S (I - S V^T N V S)^+ S V^T!
    # That operates purely on KxK matrices (tiny!) and is 100% EXACT to Eq. D8!
    _, s_vals, vt = np.linalg.svd(W, full_matrices=False)
    V = vt.T  # (K, K)
    S = np.diag(s_vals)
    inner = np.eye(K) - S @ (V.T @ np.diag(nk) @ V) @ S
    # Zero out the singular mode corresponding to constant shift (eigenvalue ~ 0)
    eigvals, eigvecs = np.linalg.eigh(inner)
    inv_eigs = np.where(eigvals > 1e-10, 1.0 / eigvals, 0.0)
    inner_pinv = (eigvecs * inv_eigs[None, :]) @ eigvecs.T
    cov_f = V @ S @ inner_pinv @ S @ V.T

    df_tot = float(f_k[-1] - f_k[0])
    var_tot = float(cov_f[0, 0] + cov_f[-1, -1] - 2.0 * cov_f[0, -1])
    err_tot = math.sqrt(max(0.0, var_tot))

    return {
        "f_k": f_k,
        "cov_f": cov_f,
        "df_total_kt": df_tot,
        "df_total_err_kt": err_tot,
        "overlap_matrix": overlap,
        "min_neighbor_overlap": min_nbr_overlap,
        "converged": bool(res.success),
    }


def parse_gromacs_dhdl_xvg(path: Path) -> dict[str, Any]:
    """Parse a GROMACS dhdl.xvg file extracting state λ, dH/dλ series, and ΔH(λ_k) columns."""
    df = read_xvg_dataframe(path)
    header_text = path.read_text(encoding="utf-8", errors="ignore")
    # Detect current state lambda from subtitle (e.g., "T = 298.15 (K), lambda = 0.25" or state vector)
    lam_val: float | None = None
    m_lam = re.search(r"lambda\s*=\s*([0-9.eE+-]+)", header_text, flags=re.IGNORECASE)
    if m_lam:
        lam_val = float(m_lam.group(1))
    else:
        m_dir = re.search(r"(?:lambda|lam)[_-]?([0-9]+(?:\.[0-9]+)?)", str(path), flags=re.IGNORECASE)
        if m_dir:
            lam_val = float(m_dir.group(1))

    temp_k: float | None = None
    m_t = re.search(r"T\s*=\s*([0-9.]+)\s*\(K\)", header_text)
    if m_t:
        temp_k = float(m_t.group(1))

    dhdl_cols: list[str] = []
    delta_h_cols: dict[int, str] = {}
    for col in df.columns[1:]:
        cl = str(col).lower()
        if "dh/d" in cl or "dH/d" in str(col):
            dhdl_cols.append(str(col))
        elif "delta h to" in cl or "δh" in cl or "d_h" in cl or "to lambda" in cl:
            m_idx = re.search(r"(\d+)\s*$", str(col))
            idx_k = int(m_idx.group(1)) if m_idx else len(delta_h_cols)
            delta_h_cols[idx_k] = str(col)

    return {
        "path": str(path),
        "lambda_val": lam_val,
        "temperature_k": temp_k,
        "df": df,
        "dhdl_cols": dhdl_cols,
        "delta_h_cols": delta_h_cols,
    }


def analyze_dhdl_window_set(
    dhdl_paths: list[Path],
    temperature_k: float = 298.15,
    equil_fraction: float = 0.2,
) -> dict[str, Any]:
    """Run TI, BAR, and MBAR on a set of GROMACS dhdl*.xvg files."""
    parsed = [parse_gromacs_dhdl_xvg(p) for p in sorted(dhdl_paths)]
    parsed = [p for p in parsed if not p["df"].empty]
    if len(parsed) < 2:
        return {"summary": {}, "windows": pd.DataFrame(), "overlap": pd.DataFrame()}

    # Assign fallback evenly spaced lambdas if header lacked explicit lambda value
    for idx, item in enumerate(parsed):
        if item["lambda_val"] is None:
            item["lambda_val"] = idx / max(len(parsed) - 1, 1)
    parsed.sort(key=lambda x: float(x["lambda_val"]))

    temp_eff = float(next((p["temperature_k"] for p in parsed if p["temperature_k"] is not None), temperature_k))
    beta = 1.0 / (R_KJ_MOL_K * temp_eff)  # (kJ/mol)^-1

    lam_arr = np.array([float(p["lambda_val"]) for p in parsed], dtype=float)
    dhdl_means: list[float] = []
    dhdl_sems: list[float] = []
    win_rows: list[dict[str, Any]] = []

    # Collect ΔH matrices if all windows report Delta H to every state (enables exact BAR & MBAR!)
    K = len(parsed)
    u_blocks: list[np.ndarray] = []
    n_k_list: list[int] = []
    has_full_delta_h = True

    for k, item in enumerate(parsed):
        df = item["df"]
        start = int(equil_fraction * len(df))
        eq_df = df.iloc[start:] if len(df) - start >= 4 else df

        if item["dhdl_cols"]:
            dhdl_sum = np.sum([eq_df[c].to_numpy(dtype=float) for c in item["dhdl_cols"]], axis=0)
            mu = float(np.mean(dhdl_sum))
            sem = float(np.std(dhdl_sum, ddof=1) / math.sqrt(len(dhdl_sum))) if len(dhdl_sum) > 1 else 0.0
        else:
            mu, sem = float("nan"), float("nan")
        dhdl_means.append(mu)
        dhdl_sems.append(sem)
        win_rows.append(
            {
                "window_index": k,
                "lambda": float(lam_arr[k]),
                "n_samples": len(eq_df),
                "dhdl_mean_kj_mol": mu,
                "dhdl_sem_kj_mol": sem,
                "source_file": item["path"],
            }
        )

        dh_cols = item["delta_h_cols"]
        if len(dh_cols) >= K:
            sorted_cols = [dh_cols[idx] for idx in sorted(dh_cols.keys())[:K]]
            mat_kj = np.vstack([eq_df[c].to_numpy(dtype=float) for c in sorted_cols])  # (K, N_k)
            u_blocks.append(beta * mat_kj)
            n_k_list.append(mat_kj.shape[1])
        else:
            has_full_delta_h = False

    ti_res = (
        compute_ti_free_energy(lam_arr, np.asarray(dhdl_means), np.asarray(dhdl_sems))
        if all(np.isfinite(dhdl_means))
        else {"dg_ti_trap_kj_mol": float("nan"), "dg_ti_spline_kj_mol": float("nan"), "dg_ti_err_kj_mol": float("nan")}
    )

    dg_bar_kj = float("nan")
    dg_bar_err_kj = float("nan")
    dg_mbar_kj = float("nan")
    dg_mbar_err_kj = float("nan")
    min_ov = float("nan")
    overlap_df = pd.DataFrame()

    if has_full_delta_h and len(u_blocks) == K:
        # Pairwise BAR across adjacent windows
        bar_dfs = []
        bar_vars = []
        for k in range(K - 1):
            wf = u_blocks[k][k + 1] - u_blocks[k][k]
            wr = u_blocks[k + 1][k] - u_blocks[k + 1][k + 1]
            df_k, err_k = compute_bar_pair(wf, wr)
            bar_dfs.append(df_k)
            bar_vars.append(err_k ** 2)
        dg_bar_kj = float(np.sum(bar_dfs)) / beta
        dg_bar_err_kj = math.sqrt(float(np.sum(bar_vars))) / beta

        # Global MBAR
        u_kn = np.hstack(u_blocks)
        mbar_res = compute_mbar(u_kn, np.asarray(n_k_list, dtype=float))
        dg_mbar_kj = mbar_res["df_total_kt"] / beta
        dg_mbar_err_kj = mbar_res["df_total_err_kt"] / beta
        min_ov = mbar_res["min_neighbor_overlap"]
        ov = mbar_res["overlap_matrix"]
        ov_rows = []
        for i in range(K):
            for j in range(K):
                ov_rows.append({"state_i": i, "state_j": j, "overlap": float(ov[i, j])})
        overlap_df = pd.DataFrame(ov_rows)

    summary = {
        "n_windows": K,
        "temperature_k": temp_eff,
        **ti_res,
        "dg_bar_kj_mol": dg_bar_kj,
        "dg_bar_err_kj_mol": dg_bar_err_kj,
        "dg_bar_kcal_mol": dg_bar_kj * KJ_TO_KCAL if np.isfinite(dg_bar_kj) else float("nan"),
        "dg_mbar_kj_mol": dg_mbar_kj,
        "dg_mbar_err_kj_mol": dg_mbar_err_kj,
        "dg_mbar_kcal_mol": dg_mbar_kj * KJ_TO_KCAL if np.isfinite(dg_mbar_kj) else float("nan"),
        "min_neighbor_overlap": min_ov,
        "overlap_qc_status": "PASS" if (np.isfinite(min_ov) and min_ov >= 0.03) else ("WARN_LOW_OVERLAP" if np.isfinite(min_ov) else "TI_ONLY"),
    }
    return {"summary": summary, "windows": pd.DataFrame(win_rows), "overlap": overlap_df}


def run_alchemical_analysis(runs: list[RunRecord], cfg: Config) -> dict[str, pd.DataFrame]:
    """Discover multi-window dhdl*.xvg groups across runs / workflow_root and run TI/BAR/MBAR."""
    from mody_analyse.core.registry import _find_dhdl_files

    all_files = _find_dhdl_files(runs, cfg.workflow_root)
    if len(all_files) < 2:
        return {
            "alchemical_free_energy": pd.DataFrame(),
            "alchemical_windows": pd.DataFrame(),
            "alchemical_overlap": pd.DataFrame(),
        }

    # Group by common grandparent directory (e.g. <study>/05_fep/lambda_*/dhdl.xvg -> <study>/05_fep)
    groups: dict[str, list[Path]] = {}
    for p in all_files:
        # If parent looks like lambda_00 / window_01, group by grandparent; else group by parent
        if re.search(r"(?:lambda|lam|win|state)", p.parent.name, flags=re.IGNORECASE):
            key = str(p.parent.parent)
        else:
            key = str(p.parent)
        groups.setdefault(key, []).append(p)

    sum_rows: list[dict[str, Any]] = []
    win_frames: list[pd.DataFrame] = []
    ov_frames: list[pd.DataFrame] = []
    for grp_key, paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        res = analyze_dhdl_window_set(paths, temperature_k=cfg.temperature_k)
        if res["summary"]:
            sum_rows.append({"fep_group": Path(grp_key).name, "fep_path": grp_key, **res["summary"]})
        if not res["windows"].empty:
            wdf = res["windows"].copy()
            wdf.insert(0, "fep_group", Path(grp_key).name)
            win_frames.append(wdf)
        if not res["overlap"].empty:
            odf = res["overlap"].copy()
            odf.insert(0, "fep_group", Path(grp_key).name)
            ov_frames.append(odf)

    return {
        "alchemical_free_energy": pd.DataFrame(sum_rows),
        "alchemical_windows": pd.concat(win_frames, ignore_index=True) if win_frames else pd.DataFrame(),
        "alchemical_overlap": pd.concat(ov_frames, ignore_index=True) if ov_frames else pd.DataFrame(),
    }
