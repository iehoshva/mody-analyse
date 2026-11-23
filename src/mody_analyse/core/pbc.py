from __future__ import annotations

import math
import numpy as np

# Precompute 27 lattice shifts in {-1, 0, 1}^3 for exact triclinic minimum-image refinement
_SHIFTS_27 = np.array(
    [[ix, iy, iz] for ix in (-1, 0, 1) for iy in (-1, 0, 1) for iz in (-1, 0, 1)],
    dtype=float,
)


def is_orthorhombic(box: np.ndarray, tol_deg: float = 1e-3) -> bool:
    """Return True if box represents an orthorhombic cell ((3,), (6,) with 90° angles, or diagonal (3,3))."""
    arr = np.asarray(box, dtype=float)
    if arr.shape == (3,):
        return True
    if arr.shape == (6,):
        return bool(np.all(np.abs(arr[3:6] - 90.0) <= tol_deg))
    if arr.shape == (3, 3):
        off = arr - np.diag(np.diag(arr))
        return bool(np.max(np.abs(off)) <= 1e-9)
    raise ValueError(f"Unsupported box shape: {arr.shape}")


def box_matrix_from_dimensions(box: np.ndarray) -> np.ndarray:
    """Convert box lengths/angles or 3x3 matrix into row-vector lattice matrix H (shape 3x3, in nm).

    Accepts:
      - (3,): [lx, ly, lz] in nm
      - (6,): [a, b, c, alpha_deg, beta_deg, gamma_deg] in nm & degrees (GROMACS/MDAnalysis convention)
      - (3, 3): row vectors [a_vec; b_vec; c_vec] in nm
    """
    arr = np.asarray(box, dtype=float)
    if arr.shape == (3, 3):
        return arr.copy()
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (6,):
        a, b, c, alpha_d, beta_d, gamma_d = [float(x) for x in arr]
        if is_orthorhombic(arr):
            return np.diag([a, b, c])
        alpha = math.radians(alpha_d)
        beta = math.radians(beta_d)
        gamma = math.radians(gamma_d)
        cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)
        sin_g = math.sin(gamma)
        if abs(sin_g) < 1e-12:
            raise ValueError(f"Degenerate triclinic angle gamma={gamma_d}")
        bx = b * cos_g
        by = b * sin_g
        cx = c * cos_b
        cy = c * (cos_a - cos_b * cos_g) / sin_g
        cz_sq = max(0.0, c * c - cx * cx - cy * cy)
        cz = math.sqrt(cz_sq)
        return np.array(
            [
                [a, 0.0, 0.0],
                [bx, by, 0.0],
                [cx, cy, cz],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported box shape: {arr.shape}")


def dimensions_angstrom_to_box_nm(dimensions: np.ndarray) -> np.ndarray:
    """Convert MDAnalysis ts.dimensions [a_A, b_A, c_A, alpha, beta, gamma] to nm + degrees (6,)."""
    d = np.asarray(dimensions, dtype=float)
    if len(d) >= 6:
        return np.array([d[0] / 10.0, d[1] / 10.0, d[2] / 10.0, d[3], d[4], d[5]], dtype=float)
    return d[:3] / 10.0


def box_volume_nm3(box: np.ndarray) -> float:
    """Exact cell volume |det(H)| in nm^3 for orthorhombic or arbitrary triclinic cells."""
    H = box_matrix_from_dimensions(box)
    return float(abs(np.linalg.det(H)))


def box_inscribed_radius_nm(box: np.ndarray) -> float:
    """Maximum inscribed sphere radius R_in = 0.5 * min_k (|det H| / ||a_i x a_j||) in nm."""
    H = box_matrix_from_dimensions(box)
    vol = float(abs(np.linalg.det(H)))
    if vol <= 1e-18:
        return 0.0
    a, b, c = H[0], H[1], H[2]
    areas = [
        float(np.linalg.norm(np.cross(b, c))),
        float(np.linalg.norm(np.cross(c, a))),
        float(np.linalg.norm(np.cross(a, b))),
    ]
    heights = [vol / max(ar, 1e-18) for ar in areas]
    return 0.5 * min(heights)


def wrap_positions(pos_nm: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap coordinates (..., 3) into primary unit cell [0, 1)^3 in fractional coordinates."""
    pos = np.asarray(pos_nm, dtype=float)
    if is_orthorhombic(box):
        L = np.diag(box_matrix_from_dimensions(box))
        return np.mod(pos, L)
    H = box_matrix_from_dimensions(box)
    H_inv = np.linalg.inv(H)
    frac = np.mod(pos @ H_inv, 1.0)
    return frac @ H


def minimum_image_displacement(diff_nm: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Exact minimum-image displacement vector for orthorhombic or arbitrary triclinic cells.

    For skewed triclinic cells (e.g., rhombic dodecahedron / truncated octahedron), refines the
    rounded fractional vector across all 27 neighboring lattice images to guarantee exact shortest image.
    """
    diff = np.asarray(diff_nm, dtype=float)
    if is_orthorhombic(box):
        L = np.diag(box_matrix_from_dimensions(box))
        return diff - L * np.round(diff / L)

    H = box_matrix_from_dimensions(box)
    H_inv = np.linalg.inv(H)
    orig_shape = diff.shape
    flat = diff.reshape(-1, 3)
    frac = flat @ H_inv
    frac -= np.round(frac)
    d0 = frac @ H  # (M, 3)

    # Check all 27 lattice shifts: d_cand = d0 + shift @ H
    shift_vecs = _SHIFTS_27 @ H  # (27, 3)
    # Process in chunks to keep peak memory small on large pair matrices
    out = np.empty_like(d0)
    chunk = 4096
    for start in range(0, len(d0), chunk):
        sub = d0[start : start + chunk]  # (C, 3)
        cands = sub[:, None, :] + shift_vecs[None, :, :]  # (C, 27, 3)
        d2 = np.sum(cands * cands, axis=-1)  # (C, 27)
        best = np.argmin(d2, axis=1)
        out[start : start + chunk] = cands[np.arange(len(sub)), best]
    return out.reshape(orig_shape)
