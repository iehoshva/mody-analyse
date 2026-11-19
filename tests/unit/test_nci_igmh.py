from pathlib import Path
import numpy as np
import pytest

from mody_analyse.qm_runners.nci_igmh import (
    _summarize_scatter,
    parse_igmh_integral_au,
    write_multiwfn_igmh_script,
    write_multiwfn_nci_script,
    write_vmd_nci_script,
)


def test_write_multiwfn_and_vmd_scripts(tmp_path: Path):
    nci_in = tmp_path / "nci.mwfn.in"
    write_multiwfn_nci_script(nci_in, grid_quality=2)
    txt = nci_in.read_text()
    assert "20" in txt
    assert "1" in txt

    igmh_in = tmp_path / "igmh.mwfn.in"
    write_multiwfn_igmh_script(igmh_in, frag1=(1, 34), frag2=(35, 59), grid_quality=2)
    itxt = igmh_in.read_text()
    assert "20" in itxt
    assert "11" in itxt
    assert "1-34" in itxt and "35-59" in itxt

    vmd_path = tmp_path / "render_igmh.vmd"
    write_vmd_nci_script(vmd_path, mol_xyz=Path("complex.xyz"), func1_cub=Path("sl2r.cub"), func2_cub=Path("dg_inter.cub"))
    vtxt = vmd_path.read_text()
    assert "sl2r.cub" in vtxt and "dg_inter.cub" in vtxt
    assert "Isosurface" in vtxt


def test_summarize_scatter_handles_4col_igmh_5col_nci_and_integral_log(tmp_path: Path):
    # Multiwfn IGMH 20->11 writes 4 columns: dg_inter, dg_intra, dg, sign(lambda2)rho
    igmh_txt = tmp_path / "igmh_output.txt"
    np.savetxt(
        igmh_txt,
        np.array([
            [0.0, 0.0, 0.0, 0.0],             # outside pocket zero-padding (must be excluded)
            [0.025, 0.010, 0.035, -0.025],    # H-bond
            [0.008, 0.005, 0.013, -0.005],    # vdW
            [0.005, 0.010, 0.015, 0.018],     # steric
        ]),
    )
    res_igmh = _summarize_scatter(igmh_txt, kind="igmh")
    assert res_igmh["n_points_active"] == 3
    assert np.isclose(res_igmh["frac_attractive_hb"], 1.0 / 3.0)
    assert np.isclose(res_igmh["frac_vdw"], 1.0 / 3.0)
    assert np.isclose(res_igmh["frac_steric"], 1.0 / 3.0)
    assert np.isclose(res_igmh["min_sl2r"], -0.025)

    # Multiwfn NCI 20->1 writes 5 columns: x, y, z, sign(lambda2)rho, RDG
    nci_txt = tmp_path / "nci_output.txt"
    np.savetxt(
        nci_txt,
        np.array([
            [0.0, 0.0, 0.0, -0.020, 0.25],
            [1.0, 0.0, 0.0, -0.004, 0.40],
            [2.0, 0.0, 0.0, 0.000, 3.00],    # high RDG bulk point (excluded)
        ]),
    )
    res_nci = _summarize_scatter(nci_txt, kind="nci")
    assert res_nci["n_points_active"] == 2
    assert np.isclose(res_nci["frac_attractive_hb"], 0.5)
    assert np.isclose(res_nci["min_sl2r"], -0.020)

    log_p = tmp_path / "igmh.log"
    log_p.write_text(
        " Integral of delta-g over whole space: 1.947153 a.u.\n"
        " Integral of delta-g_inter over whole space: 0.146135 a.u.\n"
        " Integral of delta-g_intra over whole space: 1.801019 a.u.\n"
    )
    integ = parse_igmh_integral_au(log_p)
    assert np.isclose(integ["integral_dg_inter_au"], 0.146135)
    assert np.isclose(integ["integral_dg_intra_au"], 1.801019)
