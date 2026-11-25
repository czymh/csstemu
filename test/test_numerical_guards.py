"""Guards for the numerical danger points hardened in commit 4.

Each check here fails on the parent commit, either by raising (UnboundLocalError,
ValueError from an obscure place) or by silently returning inf/NaN.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

_FAIL = []


def check(name, ok, detail=''):
    print('  %-58s %s %s' % (name, 'OK ' if ok else 'FAIL', detail))
    if not ok:
        _FAIL.append(name)


print('[1] finite-difference conditioning mask')

from CEmulator.Emulator import HMF_CEmulator

e = HMF_CEmulator(verbose=False)
e.set_cosmos(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665, As=2.1e-9)

ok = e._fd_ok(np.array([1.0 - 1e-9, 0.5]), np.array([1.0, 1.0]))
check('a 1e-9 relative step is rejected, a 50% step is accepted',
      list(ok) == [False, True], str(ok))
check('the rejection is reported once', e._fd_warned is True)
e._fd_warned = False
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    e._fd_ok(np.array([1.0]), np.array([1.0 - 1e-9]))
check('the warning is emitted (once per instance)', len(w) == 1, '%d warnings' % len(w))

# the four public finite-difference APIs must stay finite even where the step
# collapses; the guard degrades to the tree/Castro23 value instead of inf/NaN
from CEmulator.Emulator import Xihm_CEmulator, Xihh_CEmulator, Pkhm_CEmulator

Z = np.array([0.0, 1.0])
M = np.logspace(12, 15, 5)
R = np.logspace(-1, 1.5, 6)
K = np.logspace(-2, 0.0, 5)
for cls, name, fn, args in (
        (HMF_CEmulator, 'get_bias_mass', 'get_bias_mass', dict(z=Z, M=M)),
        (Pkhm_CEmulator, 'get_pkhm_mass', 'get_pkhm_mass', dict(z=Z, k=K, M=M)),
        (Xihm_CEmulator, 'get_xihm_mass', 'get_xihm_mass', dict(z=Z, r=R, M=M)),
        (Xihh_CEmulator, 'get_xihh_mass', 'get_xihh_mass',
         dict(z=Z, r=R, M1=M[:3], M2=M[:3]))):
    obj = cls(verbose=False)
    obj.set_cosmos(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665, As=2.1e-9)
    out = getattr(obj, fn)(**args)
    check('%s returns finite values' % name, np.all(np.isfinite(out)))

print()
print('[2] FFTLog gamma poles, low ringing and input validation')

from CEmulator.hankl.fftlog import _gamma_term, _lowring_xy
from CEmulator.hankl import P2xi

g = _gamma_term(0.0, np.array([1.0, 3.0, 5.0], dtype=complex))
check('poles at x = mu+1+2k give 0, not NaN',
      list(g) == [0j, 0j, 0j], str(g))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    xy = _lowring_xy(0.0, 1.5, np.log(10.0), 16384)
check('N=16384 gives a finite low-ringing value', np.isfinite(xy), '%.10g' % xy)
check('N=16384 no longer warns (log gamma underflow)', len(w) == 0,
      '%d warnings' % len(w))

k = np.logspace(-3, 1, 8)
P = k ** -2.0
check('the well conditioned path still runs', np.all(np.isfinite(P2xi(k, P, 0)[1])))

for label, xx, ff, expect in (
        ('odd N', k[:7], P[:7], 'even'),
        ('non-finite f_x', k, np.r_[P[:-1], np.nan], 'non-finite'),
        ('non-positive x', np.r_[0.0, k[1:]], P, 'strictly positive')):
    try:
        P2xi(xx, ff, 0)
        check('%s is rejected' % label, False, 'no exception')
    except ValueError as exc:
        check('%s is rejected' % label, expect in str(exc), str(exc)[:60])

print()
print('[3] power-law extrapolation of a vanishing edge')

from CEmulator.hankl.preprocess import preprocess

x = np.logspace(0, 2, 8)
f = np.zeros_like(x)
xp, fp, nl, nr = preprocess(x, f, ext=3)
check('an all-zero edge stays zero instead of NaN', np.all(fp == 0.0),
      'nonfinite=%d' % np.sum(~np.isfinite(fp)))

f = np.r_[0.1, np.linspace(0.05, 1e-3, 7)]
xp, fp, nl, nr = preprocess(x, f, ext=3)
check('a finite tail stays finite', np.all(np.isfinite(fp)),
      'max|f|=%.3e' % np.max(np.abs(fp)))

try:
    preprocess(x, f, ext=7)
    check('an invalid ext is rejected', False, 'no exception')
except ValueError as exc:
    check('an invalid ext is rejected', 'ext' in str(exc).lower(), str(exc)[:60])
x9 = np.logspace(0, 2, 9)
check('a scalar ext is still unpacked as before',
      preprocess(x9, np.ones(9), ext=1)[0].size > x9.size)

print()
print('[4] neutrino Fermi-Dirac integral past the table')

from CEmulator.cosmology import Cosmology

c = Cosmology()
vals = []
for mnu in (0.0, 0.06, 0.3, 0.5, 2.0):
    c.set_cosmos(dict(Omegab=0.0486, Omegam=0.3086, H0=68.98, ns=0.9665,
                      A=2.1, w=-1.0, wa=0.0, mnu=mnu))
    vals.append(c.get_Ez(np.array([0.0, 3.0])))
check('Ez stays finite for mnu up to 2 eV (was inf above 0.336 eV)',
      all(np.all(np.isfinite(v)) for v in vals))
check('Ez(z=0) is still 1 for a massless neutrino', abs(vals[0][0] - 1.0) < 1e-4,
      '%.8f' % vals[0][0])

print()
print('[5] halofit bisection no longer leaves rneff unbound')

from CEmulator.Emulator import Pkmm_CEmulator

p = Pkmm_CEmulator(verbose=False)
p.set_cosmos(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665, As=2.1e-9)
out = p.get_pknl(z=np.array([0.0, 1.0]), k=np.logspace(-2, 0.5, 6), nltype='halofit')
check('get_pknl[halofit] returns finite values', np.all(np.isfinite(out)))

print()
print('[6] HMF cache hit does not recompute the GP prediction')

e2 = HMF_CEmulator(verbose=False)
e2.set_cosmos(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665, As=2.1e-9)
e2.get_Nhalo(z=Z, M=M)          # warm both (massdef, diff) cache keys
e2.get_dndlnM(z=Z, M=M)
calls = []
for attr in e2._HMF_ATTRS.values():
    obj = getattr(e2, attr)
    orig = obj.get_data
    obj.get_data = (lambda _o=orig: (calls.append(1), _o())[1])
e2.get_Nhalo(z=Z, M=M)
e2.get_dndlnM(z=Z, M=M)
check('no get_data() call on a warm cache', len(calls) == 0, '%d calls' % len(calls))

print()
print('[7] the FFTLog u_m term is memoized, and the memo is safe')

from CEmulator.hankl.fftlog import _UM_CACHE

k = np.logspace(-5, 5, 1024)
P = k ** -2.0
first = P2xi(k, P, 0)[1]
second = P2xi(k, P, 0)[1]
check('a repeated transform is bit identical', np.array_equal(first, second))
check('the cache stays small', len(_UM_CACHE) <= 8, '%d entries' % len(_UM_CACHE))
check('the cached array is not mutated by the caller',
      np.array_equal(P2xi(k, P, 0)[1], first))

print()
if _FAIL:
    print('FAIL: %d check(s) failed' % len(_FAIL))
    sys.exit(1)
print('PASS: numerical guards behave')