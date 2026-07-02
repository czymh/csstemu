#!/usr/bin/env python
"""
JAX Emulator Test Suite
========================

Validates the JAX-based CSST emulator against the numpy reference.

Tests:
  1. Linear power spectrum P_lin(k,z) — cb and total
  2. Nonlinear power spectrum P_nl(k,z) — HMCODE-2020
  3. Automatic differentiation (jax.grad)
  4. Multiple random cosmologies
  5. FFTLog Hankel transform P(k) -> xi(r)
  6. Hubble parameter E(z)

Usage:
    cd /path/to/csstemu-jax
    conda activate csstemu
    python test/test_jax_emulator.py
"""

import numpy as np
import jax.numpy as jnp
import jax
import sys

# ── Test infrastructure ─────────────────────────────────────────────────

_failures = []


def check(name, value, threshold=1e-3, verbose=True):
    """Check that `value` is below `threshold`. Record failures."""
    ok = value < threshold
    if verbose:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:45s} = {value:.4e}")
    if not ok:
        _failures.append((name, value, threshold))
    return ok


# ── Load emulators ──────────────────────────────────────────────────────

print("=" * 70)
print("Loading emulators...")
from CEmulator.jax.jax_emulator import JAXEmulator
from CEmulator.Emulator import Pkmm_CEmulator

emu_jax = JAXEmulator()
emu_np = Pkmm_CEmulator()

# ── Reference cosmology ─────────────────────────────────────────────────

Ob, Oc = 0.049, 0.264
H0v, ns_v, As_v = 67.66, 0.9665, 2.105e-9
w_v, wa_v, mnu_v = -1.0, 0.0, 0.06

emu_np.set_cosmos(
    Omegab=Ob, Omegac=Oc, H0=H0v, As=As_v,
    ns=ns_v, w=w_v, wa=wa_v, mnu=mnu_v,
)
cosmo_j = jnp.array([Ob, Ob + Oc, H0v, ns_v, As_v * 1e9, w_v, wa_v, mnu_v])

z_arr = jnp.array([0.0, 0.25, 0.5, 1.0, 2.0])
k_arr = jnp.logspace(-2, 0, 20)

# ═══════════════════════════════════════════════════════════════════════
# Test 1: Hubble Parameter E(z) — test FIRST to avoid JIT cache pollution
# ═══════════════════════════════════════════════════════════════════════

print("\n─── 1. Hubble Parameter E(z) ───")

Ez_jax = np.asarray(emu_jax.get_Ez(cosmo_j, z_arr))
Ez_np = np.asarray(emu_np.Cosmo.get_Ez(np.asarray(z_arr)))
rd = np.max(np.abs(Ez_jax - Ez_np) / Ez_np)
check("E(z) vs numpy", rd, threshold=1e-6)

# ═══════════════════════════════════════════════════════════════════════
# Test 2: Linear power spectrum
# ═══════════════════════════════════════════════════════════════════════

print("\n─── 2. Linear Power Spectrum ───")

# 1a. P_cb_lin (cb-only)
pk_np = emu_np.get_pklin(np.asarray(z_arr), np.asarray(k_arr), type='Emulator', Pcb=True)
pk_jax = np.asarray(emu_jax.get_pkcblin(cosmo_j, z_arr, k_arr))
rd = np.max(np.abs(pk_jax - pk_np) / pk_np)
check("P_cb_lin vs numpy", rd, threshold=1e-5)
assert pk_jax.shape == (len(z_arr), len(k_arr)), "Shape mismatch"

# 1b. P_lin total (cb + neutrino)
pk_np = emu_np.get_pklin(np.asarray(z_arr), np.asarray(k_arr), type='Emulator', Pcb=False)
pk_jax = np.asarray(emu_jax.get_pklin(cosmo_j, z_arr, k_arr, Pcb=False))
rd = np.max(np.abs(pk_jax - pk_np) / pk_np)
check("P_lin total vs numpy", rd, threshold=1e-5)

# ═══════════════════════════════════════════════════════════════════════
# Test 3: Nonlinear power spectrum
# ═══════════════════════════════════════════════════════════════════════

print("\n─── 3. Nonlinear Power Spectrum ───")

# 3a. HMCODE-2020 model prediction (denominator for get_pknl)
pk_np = emu_np.get_pkHMCODE2020(
    np.asarray(z_arr), np.asarray(k_arr), lintype='Emulator', Pcb=True,
)
pk_jax = np.asarray(emu_jax.get_pkhmcode2020(cosmo_j, z_arr, k_arr, Pcb=True))
rd = np.max(np.abs(pk_jax - pk_np) / pk_np)
check("P_hmcode2020 vs numpy", rd, threshold=1e-5)

# 3b. get_pknl(nltype='linear') — P_nl = P_lin × B_lin
pk_np = emu_np.get_pknl(
    np.asarray(z_arr), np.asarray(k_arr), lintype='Emulator',
    nltype='linear', Pcb=True,
)
pk_jax = np.asarray(emu_jax.get_pknl(cosmo_j, z_arr, k_arr, nltype='linear', Pcb=True))
rd = np.max(np.abs(pk_jax - pk_np) / pk_np)
check("get_pknl(nltype='linear') vs numpy", rd, threshold=1e-5)

# 3c. get_pknl(nltype='hmcode2020') — P_nl = P_hmcode2020 × B_hmcode2020
pk_np = emu_np.get_pknl(
    np.asarray(z_arr), np.asarray(k_arr), lintype='Emulator',
    nltype='hmcode2020', Pcb=True,
)
pk_jax = np.asarray(emu_jax.get_pknl(cosmo_j, z_arr, k_arr, nltype='hmcode2020', Pcb=True))
rd = np.max(np.abs(pk_jax - pk_np) / pk_np)
check("get_pknl(nltype='hmcode2020') vs numpy", rd, threshold=1e-5)

# ═══════════════════════════════════════════════════════════════════════
# Test 3: Automatic differentiation
# ═══════════════════════════════════════════════════════════════════════

print("\n─── 4. Automatic Differentiation ───")

# 3a. Gradient of P_lin
grad_lin = jax.grad(lambda c: emu_jax.get_pklin(c, z_arr, k_arr).sum())(cosmo_j)
ok = not (np.any(np.isnan(grad_lin)) or np.any(np.isinf(grad_lin)))
print(f"  [{'PASS' if ok else 'FAIL'}] grad(P_lin)           NaN={np.any(np.isnan(grad_lin))}, Inf={np.any(np.isinf(grad_lin))}")
if not ok:
    _failures.append(("grad P_lin has NaN/Inf", 0, 0))

# 3b. Gradient of P_nl
grad_nl = jax.grad(
    lambda c: emu_jax.get_pkhmcode2020(c, z_arr, k_arr, Pcb=True).sum()
)(cosmo_j)
ok = not (np.any(np.isnan(grad_nl)) or np.any(np.isinf(grad_nl)))
print(f"  [{'PASS' if ok else 'FAIL'}] grad(P_nl)            NaN={np.any(np.isnan(grad_nl))}, Inf={np.any(np.isinf(grad_nl))}")
if not ok:
    _failures.append(("grad P_nl has NaN/Inf", 0, 0))

# 3c. Gradient of get_pknl(nltype='hmcode2020')
grad_pknl = jax.grad(
    lambda c: emu_jax.get_pknl(c, z_arr, k_arr, nltype='hmcode2020', Pcb=True).sum()
)(cosmo_j)
ok = not (np.any(np.isnan(grad_pknl)) or np.any(np.isinf(grad_pknl)))
print(f"  [{'PASS' if ok else 'FAIL'}] grad(get_pknl hmcode)  NaN={np.any(np.isnan(grad_pknl))}, Inf={np.any(np.isinf(grad_pknl))}")
if not ok:
    _failures.append(("grad get_pknl hmcode has NaN/Inf", 0, 0))

# 3d. Gradient of get_pknl(nltype='linear')
grad_pknl_lin = jax.grad(
    lambda c: emu_jax.get_pknl(c, z_arr, k_arr, nltype='linear', Pcb=True).sum()
)(cosmo_j)
ok = not (np.any(np.isnan(grad_pknl_lin)) or np.any(np.isinf(grad_pknl_lin)))
print(f"  [{'PASS' if ok else 'FAIL'}] grad(get_pknl linear)  NaN={np.any(np.isnan(grad_pknl_lin))}, Inf={np.any(np.isinf(grad_pknl_lin))}")
if not ok:
    _failures.append(("grad get_pknl linear has NaN/Inf", 0, 0))

# 3e. Gradient sign sanity check
# d(Pk)/d(A_s) should be positive (more primordial power -> more Pk everywhere)
ok = float(grad_lin[4]) > 0
print(f"  [{'PASS' if ok else 'FAIL'}] d(P_lin)/dA_s > 0   A_s grad = {grad_lin[4]:.4e}")
if not ok:
    _failures.append(("d(P_lin)/dA_s not positive", float(grad_lin[4]), 0))

# n_s gradient can be either sign depending on k-range relative to pivot
# (n_s tilts the spectrum; sign of sum over k depends on which k dominate)
# Just verify it is finite and non-trivial
ok = abs(float(grad_lin[3])) > 1e-6
print(f"  [{'PASS' if ok else 'FAIL'}] |d(P_lin)/dn_s| > 1e-6  n_s grad = {grad_lin[3]:.4e}")
if not ok:
    _failures.append(("d(P_lin)/dn_s too small", float(grad_lin[3]), 1e-6))

# 3d. Gradient magnitude sanity check
# All gradients should be non-zero (emulator is non-trivial)
abs_grad = np.abs(np.asarray(grad_lin))
all_nonzero = np.all(abs_grad > 1e-6)
print(f"  [{'PASS' if all_nonzero else 'FAIL'}] All |grad| > 1e-6    min|grad|={np.min(abs_grad):.2e}")
if not all_nonzero:
    _failures.append(("Some gradients are zero", 0, 0))

# ═══════════════════════════════════════════════════════════════════════
# Test 4: Multiple random cosmologies
# ═══════════════════════════════════════════════════════════════════════

print("\n─── 5. Random Cosmologies ───")

np.random.seed(42)
max_rd = 0.0
for ic in range(5):
    Ob_r = np.random.uniform(0.04, 0.06)
    Om_r = np.random.uniform(0.24, 0.40)
    H0_r = np.random.uniform(60, 80)
    ns_r = np.random.uniform(0.92, 1.0)
    A_r = np.random.uniform(1.7, 2.5)
    w_r = np.random.uniform(-1.3, -0.7)
    wa_r = np.random.uniform(-0.5, 0.5)
    mnu_r = np.random.uniform(0, 0.3)

    emu_np.set_cosmos(
        Omegab=Ob_r, Omegac=Om_r - Ob_r, H0=H0_r,
        As=A_r * 1e-9, ns=ns_r, w=w_r, wa=wa_r, mnu=mnu_r,
    )
    cj = jnp.array([Ob_r, Om_r, H0_r, ns_r, A_r, w_r, wa_r, mnu_r])

    pk_np = emu_np.get_pklin(
        np.asarray(z_arr), np.asarray(k_arr), type='Emulator', Pcb=False,
    )
    pk_jax = np.asarray(emu_jax.get_pklin(cj, z_arr, k_arr, Pcb=False))
    rd = np.max(np.abs(pk_jax - pk_np) / pk_np)
    max_rd = max(max_rd, rd)

    if ic == 0:
        # Also test grad for this random cosmo
        g = jax.grad(lambda c: emu_jax.get_pklin(c, z_arr, k_arr).sum())(cj)
        assert not np.any(np.isnan(g)), f"NaN grad at cosmo {ic}"

check(f"5 random cosmologies", max_rd, threshold=1e-5)

# ═══════════════════════════════════════════════════════════════════════
# Test 5: FFTLog Hankel transform
# ═══════════════════════════════════════════════════════════════════════

print("\n─── 6. FFTLog Hankel Transform ───")

from CEmulator.jax.hankl.jax_fftlog import pk2xi_jax
from CEmulator.hankl import pk2xi as pk2xi_np

k_fft = np.logspace(-3, 1, 1024)
pk_fft = emu_np.get_pklin(np.array([0.0]), k_fft, type='Emulator')[0]

r_jax, xi_jax = pk2xi_jax(jnp.asarray(k_fft), jnp.asarray(pk_fft), lowring=False)
r_np, xi_np = pk2xi_np(k_fft, pk_fft, lowring=False)

rd = np.max(np.abs(np.asarray(xi_jax) - xi_np) / np.maximum(np.abs(xi_np), 1e-37))
check("pk2xi vs numpy", rd, threshold=1e-2)

# ═══════════════════════════════════════════════════════════════════════
# Test 7: JIT performance (smoke test for compilation)
# ═══════════════════════════════════════════════════════════════════════

print("\n─── 7. JIT Compilation ───")

import time

# Warm-up call (already done above, but let's time a fresh JIT)
# Actually the JIT is already compiled from previous tests.
# Just verify that multiple calls are fast.
t0 = time.perf_counter()
for _ in range(10):
    _ = emu_jax.get_pklin(cosmo_j, z_arr, k_arr, Pcb=False)
t1 = time.perf_counter()
avg_ms = (t1 - t0) / 10 * 1000
print(f"  [INFO] 10 predictions: {t1 - t0:.3f}s total,  {avg_ms:.1f} ms/call")
check("Prediction speed < 100ms", avg_ms, threshold=100.0)

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
if _failures:
    print(f"FAILED: {len(_failures)} test(s)")
    for name, value, threshold in _failures:
        print(f"  - {name}: {value:.4e} > {threshold:.1e}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
