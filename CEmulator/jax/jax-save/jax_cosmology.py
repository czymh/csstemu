"""
JAX-compatible cosmology functions.

Replaces the stateful Cosmology class with pure JAX functions.
Only supports neutrino_mass_split='single' (Nncdm=1).
"""

import jax.numpy as jnp
import jax
import numpy as np
import os

# ── Physical constants ─────────────────────────────────────────────────

RHO_CRIT_0 = 2.77536627e11  # Critical density at z=0 in h^2 Msun/Mpc^3
TCMB = 2.7255
KB = 8.617333262145e-5  # Boltzmann in eV/K
GAMMA_NU = (4.0 / 11.0) ** (1.0 / 3.0)

# ── Load Fermi-Dirac interpolation table ───────────────────────────────

def _load_fermi_dirac_table():
    """Load the precomputed Fermi-Dirac interpolation table."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'fermi_dirac_F_interp.npz'
    )
    raw = np.load(data_path)
    return raw['y_grid'].astype(np.float64), raw['F_grid'].astype(np.float64)

_FD_Y_GRID, _FD_F_GRID = _load_fermi_dirac_table()


# ── Cosmology pure functions ───────────────────────────────────────────

@jax.jit
def _fermi_dirac_F(y):
    """Fermi-Dirac integral F(y) via linear interpolation.

    Args:
        y: scalar or (n,) array.
    Returns:
        Interpolated F(y).
    """
    return jnp.interp(y, _FD_Y_GRID, _FD_F_GRID)


@jax.jit
def _omega_nu_times_h2(z, mnu, h0):
    """Compute Omega_nu(z) * (H(z)/H0)^2, the neutrino contribution to E(z)^2.

    Args:
        z: (nz,) redshift array.
        mnu: scalar, sum of neutrino masses in eV.
        h0: scalar, H0/100.
    Returns:
        (nz,) array.
    """
    fac = (15.0 / jnp.pi ** 4) * (GAMMA_NU ** 4) * _omega_gamma(h0) * (1.0 + z) ** 4
    T_nu_0 = GAMMA_NU * TCMB

    def _single_nu_term(z_i):
        y = mnu / (1.0 + z_i) / KB / T_nu_0
        return _fermi_dirac_F(y)

    # Vectorized over z
    F_sum = jax.vmap(_single_nu_term)(z)
    return fac * F_sum


@jax.jit
def _omega_gamma(h0):
    """Photon radiation density Omega_gamma."""
    return 2.4721231034210734e-5 / (h0 * h0)


@jax.jit
def get_Ez(z, omegam, omegab, h0, w0, wa, mnu):
    """Normalized Hubble parameter E(z) = H(z)/H0.

    Args:
        z: (nz,) redshift array.
        omegam: Omega_m (CDM + baryons only, no neutrinos).
        omegab: Omega_b.
        h0: H0 / 100.
        w0: dark energy EoS.
        wa: dark energy EoS evolution.
        mnu: sum of neutrino masses in eV.
    Returns:
        (nz,) E(z) values.
    """
    omegak = 0.0  # flat universe

    # Derived densities
    omeganu = mnu / 93.14 / (h0 * h0)
    omegag = _omega_gamma(h0)
    Nur = 2.0328  # Neff for 1 massive neutrino
    f_nnu = 7.0 / 8.0 * (GAMMA_NU ** 4) * Nur
    omegaR = omegag * (1.0 + f_nnu)
    omegaL = 1.0 - omegam - omeganu - omegaR
    omegaM = omegam + omeganu

    a = 1.0 / (1.0 + z)
    de_exp = jnp.exp(3.0 * ((a - 1.0) * wa - (1.0 + w0 + wa) * jnp.log(a)))

    neutrino_term = _omega_nu_times_h2(z, mnu, h0)

    return jnp.sqrt(
        omegam * (1.0 + z) ** 3
        + neutrino_term
        + omegak * (1.0 + z) ** 2
        + omegaR * (1.0 + z) ** 4
        + omegaL * de_exp
    )


@jax.jit
def comoving_distance_precomputed(z, a_grid, chi_at_grid, chi_fac):
    """Comoving distance via interpolation on a precomputed grid.

    The grid (a_grid, chi_at_grid) should be computed once per cosmology
    using a numpy helper (see precompute_chi_grid below).

    Args:
        z: (nz,) redshift array.
        a_grid: (n_grid,) scale factor grid, log-spaced.
        chi_at_grid: (n_grid,) cumulative trapezoidal integral values.
        chi_fac: conversion factor = 1e-5 * c / (H0/100) / 100.
    Returns:
        (nz,) comoving distance in Mpc.
    """
    a = 1.0 / (1.0 + z)
    log_a = jnp.log10(a)
    log_a_grid = jnp.log10(a_grid)
    chi = jnp.interp(log_a, log_a_grid, chi_at_grid)
    return chi * chi_fac


def precompute_chi_grid(cosmo_dict, n_grid=2048):
    """Precompute the comoving distance integration grid (numpy, once per cosmology).

    Args:
        cosmo_dict: dict with keys: 'Omegam', 'Omegab', 'H0', 'w', 'wa', 'mnu'.
        n_grid: number of grid points.
    Returns:
        (a_grid, chi_at_grid, chi_fac) to pass to comoving_distance_precomputed.
    """
    z = np.atleast_1d(cosmo_dict.get('z_ref', 0.0))
    if z[0] > 0:
        a_min = 1.0 / (1.0 + np.max(z))
    else:
        a_min = 1e-5

    a_grid = np.logspace(np.log10(max(a_min, 1e-5)), 0.0, n_grid)
    z_grid = 1.0 / a_grid - 1.0

    # Use numpy for the integration (same formula as original cosmology.py)
    omegam = cosmo_dict['Omegam']
    omegab = cosmo_dict['Omegab']
    h0_val = cosmo_dict['H0'] / 100.0
    w0 = cosmo_dict['w']
    wa = cosmo_dict['wa']
    mnu = cosmo_dict['mnu']

    # Compute E(z) on the grid (numpy)
    omegak = 0.0
    omeganu_np = mnu / 93.14 / (h0_val * h0_val)
    omegag_np = 2.4721231034210734e-5 / (h0_val * h0_val)
    Nur_np = 2.0328
    f_nnu_np = 7.0 / 8.0 * ((4.0 / 11.0) ** (4.0 / 3.0)) * Nur_np
    omegaR_np = omegag_np * (1.0 + f_nnu_np)
    omegaL_np = 1.0 - omegam - omeganu_np - omegaR_np

    # Neutrino term
    gamma_nu = (4.0 / 11.0) ** (1.0 / 3.0)
    T_nu_0 = gamma_nu * TCMB
    fac_nu = (15.0 / np.pi ** 4) * (gamma_nu ** 4) * omegag_np

    if mnu > 0:
        y_vals = mnu / (1.0 + z_grid) / KB / T_nu_0
        F_grid_vals = np.interp(y_vals, _FD_Y_GRID, _FD_F_GRID)
    else:
        F_grid_vals = 0.0
    neutrino_term_np = fac_nu * (1.0 + z_grid) ** 4 * F_grid_vals

    a_vals = 1.0 / (1.0 + z_grid)
    de_exp_np = np.exp(3.0 * ((a_vals - 1.0) * wa - (1.0 + w0 + wa) * np.log(a_vals)))
    Ez_grid = np.sqrt(
        omegam * (1.0 + z_grid) ** 3
        + neutrino_term_np
        + omegak * (1.0 + z_grid) ** 2
        + omegaR_np * (1.0 + z_grid) ** 4
        + omegaL_np * de_exp_np
    )
    inv_Ez_a2 = 1.0 / (Ez_grid * a_vals ** 2)

    # Trapezoidal cumulative integral from a_grid[i] to a_grid[-1]=1
    da = np.diff(a_grid)
    avg = 0.5 * (inv_Ez_a2[:-1] + inv_Ez_a2[1:])
    cum = np.cumsum(avg[::-1] * da[::-1])[::-1]
    chi_at_grid = np.concatenate([cum, [0.0]])

    vel_light = 2.99792458e10  # cm/s
    chi_fac = 1e-5 * vel_light / (h0_val * 100.0) / 100.0

    return a_grid.astype(np.float64), chi_at_grid.astype(np.float64), chi_fac
