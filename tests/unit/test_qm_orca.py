from pathlib import Path
import pytest

from mody_analyse.qm_runners.qm_orca import (
    _run_orca,
    choose_led_execution_settings,
    parse_orca_led_output,
    parse_orca_sp_energy_eh,
    write_orca_cp_dimer_inputs,
    write_orca_dft_input,
    write_orca_led_input,
)


def test_write_orca_dft_and_cp_inputs(tmp_path: Path):
    coords = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.96), ("O", 2.9, 0.0, 0.0), ("H", 2.9, 0.8, 0.0)]
    inp = tmp_path / "dimer.inp"
    write_orca_dft_input(inp, coords, method="r2SCAN-3c", nprocs=4, maxcore_mb=1200)
    txt = inp.read_text()
    assert "! r2SCAN-3c TightSCF SP" in txt
    assert "%pal\n  nprocs 4\nend" in txt
    assert "%maxcore 1200" in txt

    cp_dir = tmp_path / "cp"
    paths = write_orca_cp_dimer_inputs(
        cp_dir,
        coords_a=coords[:2],
        coords_b=coords[2:],
        method="wB97X-D3 def2-TZVP",
        nprocs=4,
        maxcore_mb=1200,
    )
    assert set(paths.keys()) == {"ab", "a_ghost_b", "b_ghost_a", "a_only", "b_only"}
    ghost_txt = paths["a_ghost_b"].read_text()
    assert "O :" in ghost_txt or "O:" in ghost_txt


def test_write_and_parse_orca_led_and_fallback_ladder(tmp_path: Path):
    coords_a = [("C", 0.0, 0.0, 0.0)]
    coords_b = [("O", 3.0, 0.0, 0.0)]
    led_inp = tmp_path / "led.inp"
    write_orca_led_input(led_inp, coords_a, coords_b, basis="def2-TZVP", nprocs=2, maxcore_mb=1000)
    txt = led_inp.read_text()
    assert "DLPNO-CCSD(T)" in txt
    assert "def2-TZVP/C" in txt
    assert "RIJCOSX" in txt and "RIJK" not in txt
    assert "LED" in txt
    assert "(1)" in txt and "(2)" in txt

    # Fallback priority: nprocs 2 def2-TZVP -> nprocs 1 def2-TZVP -> nprocs 1 def2-SVP
    s1 = choose_led_execution_settings(peak_rss_gb=6.4, ram_limit_gb=6.0, current_nprocs=2, current_basis="def2-TZVP")
    assert s1 == {"nprocs": 1, "basis": "def2-TZVP", "maxcore_mb": 1000, "emit_hpc_deck": False}
    s2 = choose_led_execution_settings(peak_rss_gb=6.4, ram_limit_gb=6.0, current_nprocs=1, current_basis="def2-TZVP")
    assert s2 == {"nprocs": 1, "basis": "def2-SVP", "maxcore_mb": 1000, "emit_hpc_deck": True}

    sample_sp = "\nFINAL SINGLE POINT ENERGY      -152.123456789012\n"
    assert abs(parse_orca_sp_energy_eh(sample_sp) - (-152.123456789012)) < 1e-12

    sample_led = """
    Electrostatic interaction (Ref.)      :   -0.015000000 Eh    -9.41 kcal/mol
    Exchange interaction (Ref.)           :   -0.004000000 Eh    -2.51 kcal/mol
    Electronic preparation (Ref.)         :    0.012000000 Eh     7.53 kcal/mol
    Dispersion (strong pairs)             :   -0.003000000 Eh    -1.88 kcal/mol
    Dispersion (weak pairs)               :   -0.000500000 Eh    -0.31 kcal/mol
    Total interaction energy              :   -0.010500000 Eh    -6.59 kcal/mol
    """
    led = parse_orca_led_output(sample_led)
    assert abs(led["e_elstat_kcal"] - (-9.41)) < 1e-6
    assert abs(led["e_disp_kcal"] - (-2.19)) < 1e-6


def test_run_orca_reexecutes_on_updated_input_and_aborts_on_rss(tmp_path: Path):
    import os
    import time

    fake_bin = tmp_path / "fake_orca"
    fake_bin.write_text("#!/usr/bin/env bash\necho 'FINAL SINGLE POINT ENERGY -100.0'\necho '****ORCA TERMINATED NORMALLY****'\n")
    fake_bin.chmod(0o755)

    inp = tmp_path / "job.inp"
    inp.write_text("! r2SCAN-3c TightSCF SP\n")
    out = tmp_path / "job.out"
    out.write_text("OLD\n****ORCA TERMINATED NORMALLY****\n")
    # Make inp strictly newer than out -> must NOT return stale cached output
    os.utime(out, (time.time() - 10, time.time() - 10))
    res_out, _ = _run_orca(inp, fake_bin)
    assert "FINAL SINGLE POINT ENERGY -100.0" in res_out.read_text()

    # RSS limit abort check with a tiny memory-allocating script
    mem_bin = tmp_path / "mem_orca"
    mem_bin.write_text("#!/usr/bin/env python3\nimport time\nbuf = bytearray(40 * 1024 * 1024)\ntime.sleep(5)\n")
    mem_bin.chmod(0o755)
    inp2 = tmp_path / "mem.inp"
    inp2.write_text("x\n")
    with pytest.raises(MemoryError):
        _run_orca(inp2, mem_bin, monitor_rss=True, abort_rss_gb=0.005)
