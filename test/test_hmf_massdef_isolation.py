"""
HMF cross-massdef cache isolation.

`_2d_interp` caches on `(massdef, diff)` and the cached spline needs the `addnum`
offset that was derived alongside it. That offset used to live in the keyless
instance attribute `self.addnum`, which the cache-hit path did not refresh, so
evaluating massdef A after massdef B could apply B's offset to A's result.

Today the fault is inert: `addnum` happens to equal 1 for every massdef because
the `UnivariateSpline(k=1, ext=0)` at Emulator.py:1266 extrapolates log10 density
out to 1e17 while the training data stops near 1e15.2, so `10**(-large)`
underflows to exactly 0.0 and `np.min(ypred_)` is 0. That is an underflow
by-product, not an invariant -- enabling the commented-out `ypred_[ypred_<=0] =
1e-32` on the next line, narrowing the mass range, or raising `mhmax_ind` would
make `addnum` massdef-dependent and turn this into an absolute number-density
error that explodes relatively at the high-mass end.

Because that degeneracy makes a numerical before/after comparison uninformative,
the invariant is asserted three ways: production offsets must survive a cache hit
unchanged, the public API must be permutation-invariant across massdefs, and the
keyless attribute must be structurally gone.

Run: python test/test_hmf_massdef_isolation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CEmulator.Emulator import HMF_CEmulator

MASSDEFS = ['RockstarM200m', 'FoFM200c', 'FoFM200m', 'RockstarMvir', 'RockstarMvir_bound']

COSMO = dict(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665,
             As=2.1e-9, w=-1.0, wa=0.0, mnu=0.06)

Z = np.array([0.0, 0.5, 1.0])
M = np.logspace(12.0, 15.0, 12)

_failures = []


def check(name, ok, detail=''):
    print('  %-58s %s%s' % (name, 'OK' if ok else 'FAIL', ('  ' + detail) if detail else ''))
    if not ok:
        _failures.append(name)


def build():
    e = HMF_CEmulator(verbose=False)
    e.set_cosmos(**COSMO)
    return e


def gpattr(massdef):
    return {'RockstarM200m': 'HMFRockstarM200m', 'FoFM200c': 'HMFFoFM200c',
            'FoFM200m': 'HMFFoFM200m', 'RockstarMvir': 'HMFRockstarMvir',
            'RockstarMvir_bound': 'HMFRockstarMvirb'}[massdef]


def test_production_addnum_is_consistent():
    """Every (massdef, diff) pair must reproduce its own offset on a cache hit."""
    print('\n[1] production addnum consistency across all massdefs')
    e = build()
    ref = {}
    for massdef in MASSDEFS:
        for diff in (False, True):
            ypred = getattr(e, gpattr(massdef)).get_data()
            spline, addnum = e._2d_interp(ypred, diff=diff, massdef=massdef)
            ref[(massdef, diff)] = (spline(z=Z, M=M), addnum)
    # second pass: every key is now a cache hit
    for massdef in MASSDEFS:
        for diff in (False, True):
            ypred = getattr(e, gpattr(massdef)).get_data()
            spline, addnum = e._2d_interp(ypred, diff=diff, massdef=massdef)
            got = spline(z=Z, M=M)
            want, want_addnum = ref[(massdef, diff)]
            check('%-20s diff=%-5s cache hit' % (massdef, diff),
                  addnum == want_addnum and np.array_equal(got, want),
                  'addnum=%.10g' % addnum)


def test_permutation_sweep():
    """get_Nhalo/get_dndlnM must not depend on which massdef was called before."""
    print('\n[2] public API permutation sweep (build under A, pollute, re-hit A)')
    for target in MASSDEFS:
        clean = build()
        want_n = clean.get_Nhalo(z=Z, M=M, massdef=target)
        want_d = clean.get_dndlnM(z=Z, M=M, massdef=target)
        del clean

        dirty = build()
        dirty.get_Nhalo(z=Z, M=M, massdef=target)
        dirty.get_dndlnM(z=Z, M=M, massdef=target)
        for other in MASSDEFS:
            if other == target:
                continue
            for diff in (False, True):
                fn = dirty.get_dndlnM if diff else dirty.get_Nhalo
                fn(z=Z, M=M, massdef=other)
        got_n = dirty.get_Nhalo(z=Z, M=M, massdef=target)
        got_d = dirty.get_dndlnM(z=Z, M=M, massdef=target)
        rel = max(np.abs(got_n - want_n).max() / np.abs(want_n).max(),
                  np.abs(got_d - want_d).max() / np.abs(want_d).max())
        check('%-20s polluted by the other 4' % target, rel == 0.0, 'worst rel dev=%.3e' % rel)
        del dirty


def test_no_keyless_shared_state():
    """
    Structural check: the offset must travel with its spline instead of living in
    an instance attribute that every massdef writes to.

    Synthetic inputs cannot demonstrate a deviation numerically, because ypred_ is
    multiplied by the Castro23 cumulative HMF at Emulator.py:1268 and that tail
    underflows to exactly 0.0 for any cosmology and any input scale, pinning
    addnum to 1. That degeneracy is exactly why the sweeps above pass both before
    and after the fix, so the invariant has to be asserted structurally.
    """
    print('\n[3] structural: the offset travels with its spline')
    e = build()
    for massdef in MASSDEFS:
        ypred = getattr(e, gpattr(massdef)).get_data()
        e._2d_interp(ypred, massdef=massdef)
        e._2d_interp(ypred, diff=True, massdef=massdef)
    check('instance no longer carries a keyless addnum', not hasattr(e, 'addnum'))
    for key, val in sorted(e._interp_cache.items(), key=lambda kv: str(kv[0])):
        ok = (isinstance(val, tuple) and len(val) == 2 and callable(val[0])
              and np.isfinite(val[1]))
        check('cache[%s] is a (callable, finite offset) pair' % (key,), ok,
              'offset=%.10g' % val[1] if isinstance(val, tuple) and len(val) == 2 else '')
    del e


def test_mass_from_lgden_isolation():
    """The second write path: _get_mass_from_lgden_interp calls get_Nhalo internally."""
    print('\n[4] get_mass_from_lgden isolation across massdefs')
    lgden = np.array([-4.0, -3.5, -3.0])
    clean = {}
    for massdef in MASSDEFS:
        e = build()
        clean[massdef] = e.get_mass_from_lgden(z=Z, lgden=lgden, massdef=massdef)
        del e
    dirty = build()
    for massdef in MASSDEFS:
        dirty.get_mass_from_lgden(z=Z, lgden=lgden, massdef=massdef)
    worst = 0.0
    for massdef in MASSDEFS:
        got = dirty.get_mass_from_lgden(z=Z, lgden=lgden, massdef=massdef)
        rel = np.abs(got - clean[massdef]).max() / np.abs(clean[massdef]).max()
        worst = max(worst, rel)
        check('%-20s after all 5 built' % massdef, rel == 0.0, 'rel dev=%.3e' % rel)
    del dirty


if __name__ == '__main__':
    test_production_addnum_is_consistent()
    test_permutation_sweep()
    test_no_keyless_shared_state()
    test_mass_from_lgden_isolation()
    print('\n' + '=' * 70)
    if _failures:
        print('FAIL (%d): %s' % (len(_failures), ', '.join(_failures)))
        sys.exit(1)
    print('PASS: HMF caches are isolated per (massdef, diff)')
