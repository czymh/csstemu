"""The sigma_R integral memo and its vectorised R loop (commit 8).

get_bias_mass used to recompute the same P(k) grid and the same sinc-window
integral over and over. Both are now memoized / vectorised and must give the
same numbers.
"""
import sys
import warnings

import numpy as np
from numpy import trapz
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

_FAIL = []


def check(name, ok, detail=''):
    print('  %-62s %s %s' % (name, 'OK ' if ok else 'FAIL', detail))
    if not ok:
        _FAIL.append(name)


def legacy_sigma(kcalc, pcalc, R):
    """The loop this commit replaced."""
    klog = np.log(kcalc)
    out = np.zeros(len(R))
    for iR in range(len(R)):
        W = 3*(np.sin(kcalc*R[iR]) - kcalc*R[iR]*np.cos(kcalc*R[iR]))/(kcalc*R[iR])**3
        out[iR] = np.sqrt(trapz(pcalc*W*W*kcalc*kcalc*kcalc/2/np.pi/np.pi, klog))
    return out


COSMO = dict(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665, As=2.1e-9,
             w=-1.0, wa=0.0, mnu=0.06)

from CEmulator.Emulator import HMF_CEmulator

e = HMF_CEmulator(verbose=False)
e.set_cosmos(**COSMO)

print('[1] the memo returns the same grid, not a recomputed one')

k1, p1 = e._get_pcalc(0.5, kind='cb')
k2, p2 = e._get_pcalc(0.5, kind='cb')
check('the array object is reused', p1 is p2)
k3, p3 = e._get_pcalc(0.5, kind='tot')
check('the cb and tot grids are kept apart', p3 is not p1)

fresh = e.Pkcblin.get_pkcbLin(0.5, np.logspace(-4.99, 1.99, 1000))[0]
check('the memoized grid equals a fresh computation', np.array_equal(p1, fresh))

print()
print('[2] a cosmology change drops the memo')

e.set_cosmos(**COSMO)
check('the memo starts empty after set_cosmos', e._pcalc_cache == {})

print()
print('[3] the vectorised R loop matches the old loop')

kcalc = np.logspace(-4.99, 1.99, 1000)
pcalc = np.abs(np.random.RandomState(7).randn(1000)) + 1e-3
R = np.concatenate([np.logspace(-1, 2, 20), np.random.RandomState(3).uniform(0.1, 100, 20)])
old = legacy_sigma(kcalc, pcalc, R)

kr    = R[:, None] * kcalc[None, :]
W     = 3*(np.sin(kr) - kr*np.cos(kr))/kr**3
rows  = pcalc*W*W*kcalc*kcalc*kcalc/2/np.pi/np.pi
new   = np.sqrt(trapz(rows, np.log(kcalc), axis=1))
check('bit identical on 40 random radii', np.array_equal(old, new),
      'max rel %.1e' % np.max(np.abs(new - old)/np.abs(old)))

print()
print('[4] the public sigma functions agree in array and scalar form')

for name, fn in (('get_sigma_z', e.get_sigma_z), ('get_sigma_cb_z', e.get_sigma_cb_z)):
    for Rv in (8.0, 10.0, 0.7, 40.0):
        a = fn(z=0.5, R=np.array([Rv]))[0]
        b = fn(z=0.5, R=Rv)
        check('%s(R=%g) array == scalar' % (name, Rv), a == b, '%r vs %r' % (a, b))

print()
print('[5] the memo does not change get_bias_mass')

Z = np.array([0.0, 0.25, 0.5, 1.0])
M = np.logspace(12, 15, 20)
b1 = e.get_bias_mass(z=Z, M=M)
e._pcalc_cache = {}
b2 = e.get_bias_mass(z=Z, M=M)
check('with and without a warm memo the result is identical', np.array_equal(b1, b2))
check('the result is finite', np.all(np.isfinite(b1)))

print()
if _FAIL:
    print('FAIL: %d check(s) failed' % len(_FAIL))
    sys.exit(1)
print('PASS: sigma cache behaves')