import numpy as np
import pandas as pd
import pytest

from mody_analyse.analyzers.green_kubo import (
    compute_stress_acf,
    integrate_green_kubo,
    stokes_einstein_viscosity_mpa_s,
)


def test_integrate_green_kubo_recovers_analytical_exponential_acf():
    # Analytical C(tau) = sigma^2 * exp(-tau / tau_c) -> integral = sigma^2 * tau_c
    dt_ps = 0.01
    tau = np.arange(0, 2000) * dt_ps
    sigma_bar = 200.0
    tau_c = 1.5  # ps
    acf = (sigma_bar**2) * np.exp(-tau / tau_c)
    res = integrate_green_kubo(tau, acf, volume_nm3=125.0, temp_K=323.15, dt_ps=dt_ps)
    # Prefactor: V[m^3] * 1e10[Pa^2/bar^2] * 1e-12[s/ps] / (kB*T) * 1e3[mPa.s/Pa.s]
    prefactor = 125.0 * 1e-27 * 1e10 * 1e-12 / (1.380649e-23 * 323.15) * 1e3
    expected_mpa_s = prefactor * (sigma_bar**2) * tau_c
    assert res["regime"] == "high_freq_full"
    assert np.isclose(res["eta_plateau_mpa_s"], expected_mpa_s, rtol=0.03)


def test_integrate_green_kubo_guards_2ps_sub_lag_quadrature_artifact():
    dt_ps = 2.0
    tau = np.arange(0, 50) * dt_ps
    acf = np.zeros_like(tau)
    acf[0] = 185000.0
    acf[1] = 3700.0
    acf[2] = 4500.0
    acf[3] = 1200.0
    res = integrate_green_kubo(tau, acf, volume_nm3=127.0, temp_K=323.15, dt_ps=dt_ps)
    assert res["regime"] == "coarse_2ps_tail_diagnostic"
    # Tail-only integral starts at tau=2.0 ps and must NOT include 0.5 * 2 ps * 185000 (~53 mPa.s)
    assert res["eta_tail_only_mpa_s"] < 5.0
    assert res["eta_naive_trapezoid_mpa_s"] > 45.0


def test_stokes_einstein_viscosity_positive_and_consistent():
    # D in 1e-5 cm^2/s = 1e-9 m^2/s, r_h in nm
    eta = stokes_einstein_viscosity_mpa_s(d_cm2_s=1e-6, temp_K=323.15, r_h_nm=0.45)
    assert 2.0 < eta < 10.0
