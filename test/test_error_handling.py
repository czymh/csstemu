"""Error handling and boundary fixes from commit 5.

Each check here either raised the wrong exception type, formatted the builtin
``type`` instead of the caller's argument, or printed one line per mass bin.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

_FAIL = []


def check(name, ok, detail=''):
    print('  %-62s %s %s' % (name, 'OK ' if ok else 'FAIL', detail))
    if not ok:
        _FAIL.append(name)


def expect(name, fn, exc, needle=''):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - the point is to inspect it
        ok = isinstance(e, exc) and needle in str(e)
        check(name, ok, '%s: %s' % (type(e).__name__, str(e)[:70]))
        return
    check(name, False, 'no exception')


print('[1] argument validation')

from CEmulator.Emulator import Pkmm_CEmulator, HMF_CEmulator

COSMO = dict(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665, As=2.1e-9)
pk = Pkmm_CEmulator(verbose=False)
pk.set_cosmos(**COSMO)
K = np.logspace(-2, 0.0, 5)

expect('descending z raises a clear ValueError',
       lambda: pk.get_pklin(z=np.array([0.5, 0.0]), k=K),
       ValueError, 'strictly increasing')
expect('an invalid nltype raises ValueError, not NameError',
       lambda: pk.get_pknl(z=0.0, k=K, nltype='bogus'),
       ValueError, 'bogus')
expect('get_pkhalofit reports the lintype value it got',
       lambda: pk.get_pkhalofit(z=0.0, k=K, lintype='bogus'),
       ValueError, 'bogus')
expect('get_pkHMCODE2020 reports the lintype value it got',
       lambda: pk.get_pkHMCODE2020(z=0.0, k=K, lintype='bogus'),
       ValueError, 'bogus')
expect('get_pknl reports the lintype value it got',
       lambda: pk.get_pknl(z=0.0, k=K, lintype='bogus'),
       ValueError, 'bogus')

h = HMF_CEmulator(verbose=False)
h.set_cosmos(**COSMO)
expect('get_mass_from_lgden rejects a descending lgden',
       lambda: h.get_mass_from_lgden(z=0.0, lgden=np.array([-3.0, -3.5])),
       ValueError, 'strictly increasing')

print()
print('[2] control flow that used to fail after a wasted CLASS run')

out = pk.get_pknl(z=0.0, k=K, lintype='CAMB', nltype='halofit')
check('lintype=CAMB + nltype=halofit returns a value', np.all(np.isfinite(out)),
      str(out))

print()
print('[3] extrapolation warnings are reported once, not per call')

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    pk.get_pklin(z=3.5, k=K)
    pk.get_pklin(z=3.7, k=K)
    pk.get_pknl(z=0.0, k=np.array([5.0, 20.0]))
    pk.get_pknl(z=0.0, k=np.array([5.0, 40.0]))
zs = [str(x.message) for x in w if 'outside the trained range' in str(x.message)]
ks = [str(x.message) for x in w if 'beyond the trained 10' in str(x.message)]
check('z out of range warns exactly once', len(zs) == 1, '%d warnings' % len(zs))
check('k out of range warns exactly once', len(ks) == 1, '%d warnings' % len(ks))

print()
print('[4] zero number density is reported once')

hv = HMF_CEmulator(verbose=True)
hv.set_cosmos(**COSMO)
hv.get_bias_mass(z=np.array([0.0]), M=np.array([1e17]))
check('the flag is set after the first report', hv._zero_den_warned is True)
hv.get_bias_mass(z=np.array([0.0, 1.0]), M=np.array([1e17]))
check('further reports are suppressed', hv._zero_den_warned is True)

print()
print('[5] a failing internal block raises instead of silently returning zeros')

from CEmulator.Emulator import GalaxyEmulator

ge = GalaxyEmulator(verbose=False)
ge.set_cosmos(**COSMO)
ge.set_hod(logMmin=12.45, sigma_logM=0.5, logM0=12.5, logM1=14.10, alpha=1.05)
g = ge._galaxy_calc
orig = g.xihm.BrhmRockstarM200m.get_Brhm


def boom(*a, **kw):
    raise RuntimeError('injected failure')


g.xihm.BrhmRockstarM200m.get_Brhm = boom
try:
    g._compute_xi_hm_on_grid(g._z_pre, g.fftbase.r, g.M_grid)
    check('a failing xi_hm block raises RuntimeError', False, 'no exception')
except RuntimeError as e:
    check('a failing xi_hm block raises RuntimeError',
          'silently corrupt' in str(e), str(e)[:70])
finally:
    g.xihm.BrhmRockstarM200m.get_Brhm = orig

print()
if _FAIL:
    print('FAIL: %d check(s) failed' % len(_FAIL))
    sys.exit(1)
print('PASS: error handling behaves')