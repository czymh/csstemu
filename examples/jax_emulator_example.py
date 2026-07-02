#!/usr/bin/env python
"""
JAX Emulator Usage Example
===========================

This script demonstrates the core features of the JAX-based CSST emulator:

1. Linear matter power spectrum P_lin(k, z)
2. Nonlinear power spectrum P_nl(k, z) via HMCODE-2020
3. Automatic differentiation with jax.grad
4. Hubble parameter E(z)
5. Hankel transform P(k) -> xi(r)

Requirements
------------
- jax[cpu] >= 0.4.38
- CEmulator (numpy version) for comparison

Author: Zhao Chen
"""

import numpy as np
import jax.numpy as jnp
import jax

# ── Step 0: Load the JAX emulator (one-time, takes ~2 seconds for JIT warm-up)
from CEmulator.jax.jax_emulator import JAXEmulator

print("Loading JAX emulator...")
emu = JAXEmulator(verbose=False)
print("Done.\n")

# ── Step 1: Define a cosmology
#
# 8 parameters: [Omega_b, Omega_m, H0, n_s, A_s*1e9, w_0, w_a, Sum m_nu]
#
cosmo = jnp.array([
    0.049,   # Omega_b   — baryon density
    0.313,   # Omega_m   — total CDM + baryon density (NO neutrinos)
    67.66,   # H0        — Hubble constant [km/s/Mpc]
    0.9665,  # n_s       — scalar spectral index
    2.105,   # A_s × 10^9 — primordial power amplitude
    -1.0,    # w_0       — dark energy EoS (ΛCDM)
    0.0,     # w_a       — dark energy EoS evolution (ΛCDM)
    0.06,    # Sum m_nu  — neutrino mass sum [eV]
])

z = jnp.array([0.0, 0.25, 0.5, 1.0, 2.0])   # redshifts
k = jnp.logspace(-2, 0, 50)                   # wavenumbers [h/Mpc]

# ── Step 2: Predict linear power spectrum ──────────────────────────────

print("=" * 60)
print("2. Linear Power Spectrum P_lin(k, z)")
print("=" * 60)

# P_cb(k,z) — only CDM+baryon (cb) component, no neutrinos
p_cb = emu.get_pkcblin(cosmo, z, k)
print(f"   P_cb_lin shape: {p_cb.shape}  (n_z={len(z)}, n_k={len(k)})")
print(f"   P_cb_lin(z=0, k=0.1 h/Mpc) = {p_cb[0, 30]:.1f} [Mpc/h]^3")

# P_total(k,z) — total matter (cb + massive neutrinos)
p_total = emu.get_pklin(cosmo, z, k, Pcb=False)
print(f"   P_tot_lin(z=0, k=0.1 h/Mpc) = {p_total[0, 30]:.1f} [Mpc/h]^3")

# ── Step 3: Predict nonlinear power spectrum ───────────────────────────

print("\n" + "=" * 60)
print("3. Nonlinear Power Spectrum P_nl(k, z)")
print("=" * 60)

# HMCODE-2020 model prediction: P_hmcode2020 = P_lin × B_lin2hmcode
p_hmcode = emu.get_pkhmcode2020(cosmo, z, k, Pcb=True)
print(f"   P_hmcode2020 shape: {p_hmcode.shape}")
print(f"   P_hmcode2020/P_lin(z=0, k=0.1) = {p_hmcode[0,30] / p_cb[0,30]:.2f}")
print(f"   P_hmcode2020/P_lin(z=0, k=1.0) = {p_hmcode[0,-1] / p_cb[0,-1]:.2f}")

# get_pknl(nltype='hmcode2020'): simulation-level P_nl
#   P_nl = P_hmcode2020 × B_hmcode2020,  where B_hmcode2020 = P_nl_sim / P_hmcode2020
p_nl_hmcode = emu.get_pknl(cosmo, z, k, nltype='hmcode2020', Pcb=True)
print(f"\n   get_pknl(hmcode2020): P_nl/P_hmcode2020(z=0,k=0.1) = {p_nl_hmcode[0,30]/p_hmcode[0,30]:.2f}")

# get_pknl(nltype='linear'): full boost from linear
#   P_nl = P_lin × B_lin,  where B_lin = P_nl_sim / P_lin
p_nl_linear = emu.get_pknl(cosmo, z, k, nltype='linear', Pcb=True)
print(f"   get_pknl(linear):    P_nl/P_lin(z=0,k=0.1)      = {p_nl_linear[0,30]/p_cb[0,30]:.2f}")
print(f"   P_nl(linear) / P_nl(hmcode2020) at k=1.0            = {p_nl_linear[0,-1]/p_nl_hmcode[0,-1]:.4f}")
print(f"   (each nltype uses a different GP ratio model;")
print(f"    values differ slightly because GP training residuals differ per denominator)")

# ── Step 4: Automatic differentiation ──────────────────────────────────

print("\n" + "=" * 60)
print("4. Gradients via jax.grad")
print("=" * 60)

# Define scalar functions
def pk_sum(cosmo):
    return emu.get_pklin(cosmo, z, k, Pcb=False).sum()

def pknl_sum(cosmo):
    return emu.get_pknl(cosmo, z, k, nltype='hmcode2020', Pcb=True).sum()

grad_pk = jax.grad(pk_sum)(cosmo)
grad_nl = jax.grad(pknl_sum)(cosmo)

param_names = ['Omega_b', 'Omega_m', 'H0', 'n_s', 'A_s', 'w_0', 'w_a', 'm_nu']
print("   d(Sum P_lin) / d(cosmo):        d(Sum P_nl) / d(cosmo):")
for name, gl, gn in zip(param_names, grad_pk, grad_nl):
    print(f"      {name:12s} = {gl:+.4e}        {gn:+.4e}")

# ── Step 5: Hubble parameter E(z) ──────────────────────────────────────

print("\n" + "=" * 60)
print("5. Hubble Parameter E(z) = H(z)/H0")
print("=" * 60)

Ez = emu.get_Ez(cosmo, z)
for zi, ezi in zip(z, Ez):
    print(f"   E(z={zi:.2f}) = {ezi:.6f}")

# ── Step 6: Hankel Transform P(k) -> xi(r) ─────────────────────────────

print("\n" + "=" * 60)
print("6. FFTLog Hankel Transform P(k) -> xi(r)")
print("=" * 60)

from CEmulator.jax.hankl.jax_fftlog import pk2xi_jax

# Use a denser k-grid for the transform
k_fft = jnp.logspace(-3, 1, 1024)
pk_z0 = emu.get_pklin(cosmo, jnp.array([0.0]), k_fft)[0]

r, xi = pk2xi_jax(k_fft, pk_z0, lowring=False)
print(f"   r shape: {r.shape}, xi shape: {xi.shape}")
print(f"   xi(r=1 h^-1 Mpc) = {float(jnp.interp(1.0, r, xi)):.4f}")
print(f"   xi(r=10 h^-1 Mpc) = {float(jnp.interp(10.0, r, xi)):.4f}")

# ── Step 7: Comparison with numpy emulator (optional) ───────────────────

print("\n" + "=" * 60)
print("7. Cross-check with numpy emulator")
print("=" * 60)

try:
    from CEmulator.Emulator import Pkmm_CEmulator

    emu_np = Pkmm_CEmulator()
    emu_np.set_cosmos(
        Omegab=0.049, Omegac=0.264,   # Omega_b, Omega_c = Omega_m - Omega_b
        H0=67.66, As=2.105e-9, ns=0.9665,
        w=-1.0, wa=0.0, mnu=0.06,
    )

    # Convert to numpy for fair comparison
    z_np = np.asarray(z)
    k_np = np.asarray(k)

    pk_np = emu_np.get_pknl(z_np, k_np, lintype='Emulator', nltype='hmcode2020', Pcb=True)
    pk_jax = np.asarray(emu.get_pknl(cosmo, z, k, nltype='hmcode2020', Pcb=True))

    rel_diff = np.max(np.abs(pk_jax - pk_np) / pk_np)
    print(f"   JAX vs numpy get_pknl(hmcode2020): max relative diff = {rel_diff:.4e}")
    if rel_diff < 1e-5:
        print("   PASS: Agreement is excellent (< 1e-5)")
    else:
        print("   WARNING: Difference above tolerance")

except ImportError:
    print("   (numpy emulator not available for comparison)")

print("\n" + "=" * 60)
print("Example complete.")
