"""set_hod reuse and per-z lazy precomputation (commit 7).

The calculator used to rebuild all 12 redshift slabs on every set_hod. It now
keeps the calculator and builds each redshift's slab on first use.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

_FAIL = []


def check(name, ok, detail=''):
    print('  %-60s %s %s' % (name, 'OK ' if ok else 'FAIL', detail))
    if not ok:
        _FAIL.append(name)


COSMO = dict(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665, As=2.1e-9)
HOD = dict(logMmin=12.45, sigma_logM=0.5, logM0=12.5, logM1=14.10, alpha=1.05)
R = np.logspace(-1, 2.2, 24)


def build():
    from CEmulator.Emulator import GalaxyEmulator
    e = GalaxyEmulator(verbose=False)
    e.set_cosmos(**COSMO)
    e.set_hod(**HOD)
    return e


print('[1] set_hod reuses the calculator')

e = build()
calc = e._galaxy_calc
check('a calculator exists after the first set_hod', calc is not None)
e._galaxy_calc._ensure_for(np.array([0.5]))
n_before = len(e._galaxy_calc._z_slabs['xihm'])
e.set_hod(**HOD)
check('the calculator object survives a second set_hod', e._galaxy_calc is calc)
check('the computed slabs survive too',
      len(e._galaxy_calc._z_slabs['xihm']) == n_before,
      '%d slabs' % len(e._galaxy_calc._z_slabs['xihm']))

print()
print('[2] only the bracketing redshifts are built')

fresh = build()
check('nothing is precomputed at construction',
      all(len(v) == 0 for v in fresh._galaxy_calc._z_slabs.values()))
fresh.Xigm(R, 0.5)
zs = sorted(fresh._galaxy_calc._z_slabs['xihm'])
check('one z needs only its bracketing pair', len(zs) == 2, str(zs))
check('the requested z is one of them', 0.5 in zs)

print()
print('[3] lazy slabs reproduce the full 12-node grids')

lazy = build()
xi_lazy = lazy.Xigm(R, 0.5)

full = build()
full._galaxy_calc._ensure_for(full.zlists)   # force the old eager behaviour
xi_full = full.Xigm(R, 0.5)
check('the same statistic at a snapshot z is bit identical',
      np.array_equal(xi_lazy, xi_full))

# between nodes the old 12-node linear interpolator and the lazily built
# 2-node interpolator must agree
lazy2 = build()
lazy2._galaxy_calc._ensure_for(np.array([0.6]))
a = lazy2.Xigm(R, 0.6)
full2 = build()
full2._galaxy_calc._ensure_for(full2.zlists)
b = full2.Xigm(R, 0.6)
check('an interpolated z also agrees', np.allclose(a, b, rtol=0, atol=1e-12),
      'max abs dev %.3e' % np.max(np.abs(a - b)))

print()
print('[4] a cosmology change drops the slabs')

e._galaxy_calc.clear_cosmology_cache()
check('clear_cosmology_cache empties every slab kind',
      all(len(v) == 0 for v in e._galaxy_calc._z_slabs.values()))
out = e.Xigm(R, 0.5)
check('the next call rebuilds what it needs',
      len(e._galaxy_calc._z_slabs['xihm']) == 2 and np.all(np.isfinite(out)))

print()
if _FAIL:
    print('FAIL: %d check(s) failed' % len(_FAIL))
    sys.exit(1)
print('PASS: lazy precomputation behaves')