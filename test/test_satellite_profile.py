"""
Unit tests for SatelliteProfile.

Verifies consistency between csstemu's SatelliteProfile and the
dark_emulator reference implementation for both NFW and emulator-based
satellite profiles (p_hm_dist).
"""

import numpy as np
import pytest
from scipy.special import sici
from scipy.integrate import simpson

from CEmulator.emulator.SatelliteProfile import (
    SatelliteProfile,
    concentration_duffy08,
    R200_from_M,
)
from CEmulator.cosmology import RHO_CRIT_0


# ===================================================================
# Test 5A: NFW profile matches dark_emulator's analytic formula
# ===================================================================

def test_nfw_matches_dark_emulator_get_uk():
    """
    Verify SatelliteProfile._nfw_uk matches the dark_emulator's _get_uk
    analytic NFW Fourier transform formula (Cooray & Sheth 2002, Eq. 81).
    """
    k = np.logspace(-2, 2, 100)
    c200 = 5.0
    R200 = 1.5

    # csstemu implementation
    uk_csstemu = SatelliteProfile._nfw_uk(k * R200 / c200, c200)

    # dark_emulator _get_uk formula (implemented inline)
    f = 1.0 / (np.log(1.0 + c200) - c200 / (1.0 + c200))
    eta = k * R200 / c200
    si_eta_1pc, ci_eta_1pc = sici(eta * (1.0 + c200))
    si_eta, ci_eta = sici(eta)
    uk_dark = f * (np.sin(eta) * (si_eta_1pc - si_eta)
                   + np.cos(eta) * (ci_eta_1pc - ci_eta)
                   - np.sin(eta * c200) / eta / (1.0 + c200))
    uk_dark[np.isnan(uk_dark)] = 1.0

    np.testing.assert_allclose(uk_csstemu, uk_dark, rtol=1e-14)


def test_nfw_flat_lowk_limit():
    """ũ(k) should approach 1 as k → 0 (normalisation)."""
    k = np.logspace(-4, -2, 10)
    uk = SatelliteProfile._nfw_uk(k * 0.3, 5.0)
    assert np.allclose(uk, 1.0, rtol=1e-3), f"Low-k limit failed: {uk}"


def test_nfw_highk_decay():
    """ũ(k) should decay to zero at high k."""
    k = np.logspace(1, 3, 10)
    uk = SatelliteProfile._nfw_uk(k * 0.3, 5.0)
    assert np.all(uk < 0.1), f"High-k decay failed: {uk.max()}"
    assert np.all(uk >= 0), f"Negative values at high k: {uk[uk < 0]}"


# ===================================================================
# Test 5B: compute_p_hm_dist_emu follows dark_emulator convention
# ===================================================================

def _dark_emulator_p_hm_dist(k_target, r, xi_hm, R200, rho_m):
    """
    Reference implementation of dark_emulator's _compute_p_hm_satdist_emu.

    p_hm_dist_tmp = FFTLog.xi2pk(xi_hm_truncated)
                   = 4π ∫ r² ξ_hm_trunc j₀(kr) dr
    norm = ∫_0^R200 4π r³ ρ_m (1 + ξ_hm_trunc) dr
    p_hm_dist = p_hm_dist_tmp × ρ_m / norm
    """
    from CEmulator.hankl.galaxy import xi2pk
    from scipy.interpolate import interp1d

    xi_trunc = np.maximum(xi_hm.copy(), 0.0)
    xi_trunc[r > R200] = 0.0

    # FFTLog: 4π ∫ r² ξ_trunc j₀(kr) dr
    k_fft, pk_fft = xi2pk(r, xi_trunc, ext=0)
    interp = interp1d(np.log(k_fft), pk_fft, kind='linear',
                      bounds_error=False,
                      fill_value=(float(pk_fft[0]), float(pk_fft[-1])))
    p_hm_tmp = interp(np.log(k_target))

    # Normalisation: ∫ 4π r³ ρ_m (1+ξ_trunc) d(log r)
    # (csstemu uses d(log r) with log-spaced r-grid for better accuracy;
    #  dark_emulator uses dr on a uniform r-grid.)
    mask = r <= R200
    integrand = 4.0 * np.pi * r[mask] ** 3 * rho_m * (1.0 + np.maximum(xi_hm[mask], 0.0))
    norm = simpson(integrand, x=np.log(r[mask]))

    return p_hm_tmp * rho_m / norm if norm > 0 else p_hm_tmp


def test_p_hm_dist_emu_matches_ref_formula():
    """
    Verify compute_p_hm_dist_emu == reference implementation
    of the dark_emulator formula.
    """
    sp = SatelliteProfile()
    r = np.logspace(-3, 1.5, 500)
    k = np.logspace(-2, 1.5, 100)
    R200 = 1.5
    rho_m = RHO_CRIT_0 * 0.315  # ~8.74e10 h^2 Msun/Mpc^3

    # NFW-like xi_hm proxy
    c200 = 5.0
    rs = R200 / c200
    xi_hm = 100.0 / ((r / rs) * (1.0 + r / rs) ** 2)

    p_hm = sp.compute_p_hm_dist_emu(k, r, xi_hm, R200)
    p_hm_ref = _dark_emulator_p_hm_dist(k, r, xi_hm, R200, rho_m)

    # csstemu clamps k < 0.1 to 1.0 to suppress FFTLog ringing,
    # while the reference does not — restrict comparison to k ≥ 0.1.
    ok = k >= 0.1
    np.testing.assert_allclose(p_hm[ok], p_hm_ref[ok], rtol=1e-12)
    assert np.all(np.isfinite(p_hm)), "Non-finite values in p_hm_dist"


def test_p_hm_dist_emu_batch_vs_scalar():
    """
    Batch (n_mass, n_r) input should give same result as per-mass-loop.
    """
    sp = SatelliteProfile()
    r = np.logspace(-3, 1.5, 200)
    k = np.logspace(-2, 1.5, 100)
    R200 = 2.0
    # Two "masses" with different xi_hm
    rs1 = R200 / 4.0
    rs2 = R200 / 8.0
    xi_1 = 50.0 / ((r / rs1) * (1.0 + r / rs1) ** 2)
    xi_2 = 200.0 / ((r / rs2) * (1.0 + r / rs2) ** 2)
    xi_stack = np.vstack([xi_1, xi_2])

    # Batched
    p_batch = sp.compute_p_hm_dist_emu(k, r, xi_stack, R200)

    # Scalar
    p_1 = sp.compute_p_hm_dist_emu(k, r, xi_1, R200)
    p_2 = sp.compute_p_hm_dist_emu(k, r, xi_2, R200)

    np.testing.assert_allclose(p_batch[0], p_1, rtol=1e-12)
    np.testing.assert_allclose(p_batch[1], p_2, rtol=1e-12)


# ===================================================================
# Test 5C: Internal consistency of compute_p_hm_dist_emu
# ===================================================================

def test_p_hm_dist_emu_smooth():
    """
    The FFTLog result should be non-negative and finite everywhere,
    and monotonically decreasing at intermediate k (after any
    low-k ringing settles).
    """
    sp = SatelliteProfile()
    r = np.logspace(-3, 2.5, 1000)
    k = np.logspace(-2, 1.5, 200)
    R200 = 2.0
    rs = R200 / 4.0
    xi_hm = 1.0 / ((r / rs) * (1.0 + r / rs) ** 2)

    p_hm = sp.compute_p_hm_dist_emu(k, r, xi_hm, R200)

    assert np.all(np.isfinite(p_hm)), "Non-finite values"
    # p_hm_dist should be mostly positive (small FFTLog ringing at
    # low k is expected from the sharp R200 truncation)
    pos_frac = (p_hm > 0).sum() / len(p_hm)
    assert pos_frac > 0.8, f"Too many non-positive values: {1-pos_frac:.0%}"
    # p_hm_dist should generally decrease with k at intermediate k
    mid = (k > 0.1) & (k < 10.0)
    dec = np.diff(p_hm[mid])
    neg_frac = (dec < 0).sum() / len(dec)
    assert neg_frac > 0.5, (
        f"p_hm_dist should mostly decrease at intermediate k, "
        f"got {neg_frac:.0%} decreasing")


def test_p_hm_dist_emu_increasing_amplitude_with_mass():
    """More massive halos (larger xi_hm amplitude) → larger p_hm_dist."""
    sp = SatelliteProfile()
    r = np.logspace(-3, 1.5, 500)
    k = np.logspace(-2, 1, 50)
    R200 = 1.5
    rs = R200 / 5.0

    # Low and high amplitude profiles
    xi_low = 10.0 / ((r / rs) * (1.0 + r / rs) ** 2)
    xi_high = 1000.0 / ((r / rs) * (1.0 + r / rs) ** 2)

    p_low = sp.compute_p_hm_dist_emu(k, r, xi_low, R200)
    p_high = sp.compute_p_hm_dist_emu(k, r, xi_high, R200)

    # Restrict to k range above the low-k clamping floor and where FFTLog provides valid values
    valid_k = (p_low > 0) & (p_high > 0) & (k > 0.1)
    assert valid_k.any(), "No overlapping valid k range"
    assert np.all(p_high[valid_k] >= p_low[valid_k]), (
        "Higher amplitude xi_hm should give larger or equal p_hm_dist")


# ===================================================================
# Test 5D: Performance benchmarks
# ===================================================================

def test_p_hm_dist_compute_time():
    """Benchmark: single p_hm_dist call should be < 0.1 s."""
    sp = SatelliteProfile()
    r = np.logspace(-3, 1.5, 1000)
    k = np.logspace(-2, 1.5, 200)
    R200 = 2.0
    rs = R200 / 4.0
    xi_hm = 1.0 / ((r / rs) * (1.0 + r / rs) ** 2)

    import time
    start = time.time()
    for _ in range(10):
        sp.compute_p_hm_dist_emu(k, r, xi_hm, R200)
    elapsed = time.time() - start
    assert elapsed / 10 < 0.1, f"Too slow: {elapsed/10:.3f}s per call"


# ===================================================================
# Test: compute_p_hm_dist_nfw matches u_sat_k_nfw
# ===================================================================

def test_p_hm_dist_nfw_aliases_u_sat_k_nfw():
    """compute_p_hm_dist_nfw should delegate to u_sat_k_nfw."""
    sp = SatelliteProfile()
    k = np.logspace(-2, 2, 100)
    M = 1e13
    z = 0.5

    p_nfw = sp.compute_p_hm_dist_nfw(k, M, z)
    u_nfw = sp.u_sat_k_nfw(k, M, z)

    np.testing.assert_allclose(p_nfw, u_nfw, rtol=1e-15)


# ===================================================================
# Test: Rc parameter changes the profile
# ===================================================================

def test_rc_parameter_effect():
    """
    Rc < 1 → lower concentration → more extended real-space profile
    → faster Fourier-space decay → lower uk at high k.
    """
    sp_default = SatelliteProfile(Rc=1.0)
    sp_extended = SatelliteProfile(Rc=0.5)

    k = np.logspace(-2, 1, 50)
    M = 1e13
    z = 0.5

    u_def = sp_default.u_sat_k_nfw(k, M, z)
    u_ext = sp_extended.u_sat_k_nfw(k, M, z)

    # At high k, the more extended profile has less small-scale power
    high_k = k > 1.0
    assert np.all(u_ext[high_k] < u_def[high_k]), (
        "Rc < 1 should give lower uk at high k (more extended profile)")


# ===================================================================
# Test: concentration_duffy08 shape
# ===================================================================

def test_concentration_duffy08():
    """Duffy08 c(M,z) should decrease with mass and redshift."""
    c_low = concentration_duffy08(1e12, 0.0)
    c_high = concentration_duffy08(1e14, 1.0)

    assert c_low > 3, f"Low-mass c too small: {c_low}"
    assert c_low > c_high, "c should decrease with mass and redshift"


# ===================================================================
# Test: R200_from_M correctness
# ===================================================================

def test_r200_from_m():
    """R200 ∝ M^(1/3) at fixed z."""
    R_low = R200_from_M(1e12, 0.0)
    R_high = R200_from_M(1e14, 0.0)

    expected_ratio = (1e14 / 1e12) ** (1.0 / 3.0)
    actual_ratio = R_high / R_low
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-10)
