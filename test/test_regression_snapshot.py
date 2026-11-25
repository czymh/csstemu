"""
Numerical regression snapshot harness.

Every public entry point is evaluated on fixed grids for three cosmologies and
stored as a baseline. After each optimization commit the suite is re-run and
must reproduce the baseline bit-for-bit (or within --tol), proving that private
refactoring did not change observable behaviour.

Usage
-----
    python test/test_regression_snapshot.py --rebuild      # write baseline
    python test/test_regression_snapshot.py                # strict compare
    python test/test_regression_snapshot.py --allow commit1 # whitelisted deltas

Probes are grouped by the emulator instance they need so that only one heavy
instance is alive at a time (GalaxyEmulator alone reaches ~900 MB RSS).
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CEmulator.Emulator import (Pkmm_CEmulator, GalaxyEmulator, Tkmm_CEmulator,
                                Xihm_CEmulator)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_NPZ = os.path.join(HERE, 'regression_baseline.npz')
BASE_JSON = os.path.join(HERE, 'regression_baseline.json')

# ---------------------------------------------------------------- grids
# check_z()/checkdata() require strictly increasing inputs; the sort inside
# check_z is commented out (utils.py:33) and its warning is verbose-gated, so a
# descending grid only fails later with a cryptic RectBivariateSpline error.
Z_ALL = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
Z_LOW = np.array([0.0, 0.25, 0.5, 0.8])          # Pkhm/Xihm/Xihh are trained for z <= 0.8
K_NL = np.logspace(np.log10(0.0063), np.log10(9.9), 40)
K_LIN = np.logspace(-4, 1.5, 40)
R_GRID = np.logspace(-1, 2.2, 30)
M_GRID = np.logspace(12.0, 15.0, 20)
LGDEN = np.array([-4.0, -3.5, -3.0])
R_PROJ = np.logspace(-0.5, 1.3, 12)
RP_PROJ = np.logspace(-1, 1.5, 12)
SIGMA_R = np.array([8.0, 12.0])
MASSDEFS = ['RockstarM200m', 'FoFM200c', 'FoFM200m', 'RockstarMvir', 'RockstarMvir_bound']

HOD_PARAMS = dict(logMmin=12.45, sigma_logM=0.5, logM0=12.5, logM1=14.10, alpha=1.05)

# All three stay inside the training box so the CAMB fallback is never engaged
# and the baseline stays fast and deterministic.
COSMOS = {
    'c0': dict(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665,
               As=2.1e-9, w=-1.0, wa=0.0, mnu=0.06),
    'c1': dict(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665,
               As=2.1e-9, w=-1.0, wa=0.0, mnu=0.0),
    'c2': dict(Omegab=0.0450, Omegac=0.3100, H0=63.00, ns=0.9800,
               As=1.9e-9, w=-0.9, wa=0.2, mnu=0.15),
}

# A probe whose deviation is legitimate only once the named commit has landed.
WHITELIST = {
    'commit1': {
        'Pgg': 'GalaxyStats.py:1139 cache_matches typo fixed -> crash becomes a value',
        'co_': 'Emulator.py:1782 Brhm cache invalidation fixed -> sequential path '
               'converges onto the fresh-instance reference',
        'get_pkhm_mass_threshold': 'descending-lgden caller fixed -> crash becomes a value',
    },
    'commit4': {
        'get_bias_mass': 'finite-difference denominator guard -> ill-conditioned '
                         'high-mass bins degrade to tree/Castro23 instead of inf/NaN',
    },
    'commit6': {
        "nltype='linear'": 'linear branch removed; probe dropped from the suite',
    },
}


def _rss_mb():
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float('nan')


def _flat(a):
    a = np.asarray(a, dtype=np.float64)
    return np.atleast_1d(a).ravel()


class _Collector:
    """
    Runs each probe under its own try/except so that one crash cannot hide the
    remaining probes of the same group. Values and error records are kept apart.
    """

    def __init__(self):
        self.vals = {}
        self.errs = {}

    def add(self, name, fn, *args, **kwargs):
        try:
            self.vals[name] = _flat(fn(*args, **kwargs))
        except Exception as exc:
            self.errs[name] = '%s: %s' % (type(exc).__name__, exc)
        return self

    def done(self):
        return self.vals, self.errs


# ---------------------------------------------------------------- probes
def probes_pkmm(e):
    """Pkmm_CEmulator: linear/nonlinear P(k) and sigma."""
    c = _Collector()
    for Pcb in (False, True):
        tag = 'cb' if Pcb else 'tot'
        for ty in ('Emulator', 'CLASS', 'CAMB'):
            c.add('get_pklin[%s,%s]' % (tag, ty), e.get_pklin,
                  z=Z_ALL, k=K_LIN, Pcb=Pcb, type=ty)
    for nl in ('hmcode2020', 'halofit'):
        c.add("get_pknl[nltype='%s']" % nl, e.get_pknl, z=Z_ALL, k=K_NL, Pcb=True, nltype=nl)
        c.add("get_pknl_extrp[nltype='%s']" % nl, e.get_pknl_extrp,
              z=Z_ALL, k=K_NL, Pcb=True, nltype=nl)
    c.add('get_pkhalofit', e.get_pkhalofit, z=Z_ALL, k=K_NL, Pcb=True)
    c.add('get_pkHMCODE2020', e.get_pkHMCODE2020, z=Z_ALL, k=K_NL, Pcb=True)
    c.add('get_sigma8', e.get_sigma8)
    c.add('get_sigma8_cb', e.get_sigma8_cb)
    c.add('get_sigma_z', lambda: np.array(
        [e.get_sigma_z(z=float(z), R=SIGMA_R) for z in Z_ALL]))
    c.add('get_sigma_cb_z', lambda: np.array(
        [e.get_sigma_cb_z(z=float(z), R=SIGMA_R) for z in Z_ALL]))
    return c.done()


def probes_hmf(e):
    """HMF_CEmulator: mass function, bias, mass-from-density."""
    c = _Collector()
    for md in MASSDEFS:
        c.add('get_Nhalo[%s]' % md, e.get_Nhalo, z=Z_ALL, M=M_GRID, massdef=md)
        c.add('get_dndlnM[%s]' % md, e.get_dndlnM, z=Z_ALL, M=M_GRID, massdef=md)
        c.add('get_dndlnM_Castro23[%s]' % md, e.get_dndlnM_Castro23,
              z=Z_ALL, M=M_GRID, Pcb=True, revisted=True, massdef=md)
        c.add('get_bias_Castro23[%s]' % md, e.get_bias_Castro23,
              z=Z_ALL, M=M_GRID, Pcb=True, revisted=True, massdef=md)
    c.add('get_dndlnM_Tinker08', e.get_dndlnM_Tinker08, z=Z_ALL, M=M_GRID, Pcb=True)
    c.add('get_bias_mass', e.get_bias_mass, z=Z_ALL, M=M_GRID, massdef='RockstarM200m')
    c.add('get_bias_mass_threshold', e.get_bias_mass_threshold,
          z=Z_ALL, M=M_GRID, Pcb=True, massdef='RockstarM200m')
    # ascending lgden is the documented contract; descending is exercised
    # separately because get_pkhm_mass_threshold feeds a descending one in.
    c.add('get_mass_from_lgden[asc]', e.get_mass_from_lgden,
          z=Z_ALL, lgden=LGDEN, massdef='RockstarM200m')
    c.add('get_mass_from_lgden[desc]', e.get_mass_from_lgden,
          z=Z_ALL, lgden=LGDEN[::-1], massdef='RockstarM200m')
    c.add('get_bias_lgnbar_threshold', e.get_bias_lgnbar_threshold,
          z=Z_LOW, lgden=LGDEN, massdef='RockstarM200m')
    return c.done()


def probes_ximm(e):
    c = _Collector()
    c.add('get_ximmlinear', e.get_ximmlinear, z=Z_ALL, r=R_GRID, Pcb=True)
    c.add('get_ximmnl', e.get_ximmnl, z=Z_ALL, r=R_GRID, Pcb=True)
    c.add('get_ximmhalofit', e.get_ximmhalofit, z=Z_ALL, r=R_GRID, Pcb=True)
    c.add('get_ximmHMCODE2020', e.get_ximmHMCODE2020, z=Z_ALL, r=R_GRID, Pcb=True)
    return c.done()


def probes_pkhm(e):
    c = _Collector()
    c.add('get_pkhm_lgnbar_threshold', e.get_pkhm_lgnbar_threshold,
          z=Z_LOW, k=K_NL, lgden=LGDEN, massdef='RockstarM200m', Pcb=True)
    c.add('get_pkhm_mass_threshold', e.get_pkhm_mass_threshold,
          z=Z_LOW, k=K_NL, M=M_GRID, massdef='RockstarM200m', Pcb=True)
    # the exact call from examples/pkhm.ipynb cell 5
    c.add('get_pkhm_mass_threshold[notebook]', e.get_pkhm_mass_threshold,
          z=np.array([0.5]), k=np.logspace(-3, np.log10(8), 200),
          M=np.array([2e12, 1e13, 5e13]), massdef='RockstarM200m', Pcb=True)
    c.add('get_pkhm_mass', e.get_pkhm_mass,
          z=Z_LOW, k=K_NL, M=M_GRID, massdef='RockstarM200m', Pcb=True)
    return c.done()


def probes_xihm(e):
    c = _Collector()
    c.add('get_xihm_lgnbar_threshold', e.get_xihm_lgnbar_threshold,
          z=Z_LOW, r=R_GRID, lgden=LGDEN, massdef='RockstarM200m', Pcb=True)
    c.add('get_xihm_mass', e.get_xihm_mass,
          z=Z_LOW, r=R_GRID, M=M_GRID, massdef='RockstarM200m', Pcb=True)
    return c.done()


def probes_xihh(e):
    x = e.Xihh
    c = _Collector()
    c.add('get_xihh_lgnbar_threshold', x.get_xihh_lgnbar_threshold,
          z=Z_LOW, r=R_GRID, lgden1=LGDEN, lgden2=LGDEN, massdef='RockstarM200m')
    c.add('get_xihh_mass', x.get_xihh_mass,
          z=Z_LOW, r=R_GRID, M1=M_GRID, M2=M_GRID, massdef='RockstarM200m')
    return c.done()


def probes_galaxy(e):
    """GalaxyEmulator statistics; HOD is set once by the caller."""
    z0 = 0.5
    c = _Collector()
    c.add('ngal', e.ngal, Z_LOW)
    c.add('f_sat', e.f_sat, Z_LOW)
    c.add('M_eff', e.M_eff, Z_LOW)
    c.add('bias_gal', e.bias_gal, Z_LOW)
    c.add('Xigm', e.Xigm, R_GRID, z0)
    c.add('Xigg', e.Xigg, R_GRID, z0)
    c.add('Xigm_fourier', e.Xigm_fourier, R_GRID, z0)
    c.add('wgm', e.wgm, RP_PROJ, z0)
    c.add('Pgm', e.Pgm, K_NL, z0)
    c.add('Pgg', e.Pgg, K_NL, z0)
    # An array z reaches the cache-hit branch that a scalar z short-circuits.
    c.add('Pgg[array-z]', e.Pgg, K_NL, np.array([z0]))
    c.add('wp', e.wp, RP_PROJ, z0, pimax=40.0)
    c.add('wp_cc_2h', e.wp_cc_2h, RP_PROJ, z0, pimax=40.0)
    c.add('wp_cs_1h', e.wp_cs_1h, RP_PROJ, z0, pimax=40.0)
    c.add('wp_cs_2h', e.wp_cs_2h, RP_PROJ, z0, pimax=40.0)
    c.add('wp_ss_1h', e.wp_ss_1h, RP_PROJ, z0, pimax=40.0)
    c.add('wp_ss_2h', e.wp_ss_2h, RP_PROJ, z0, pimax=40.0)
    c.add('DeltaSigma', e.DeltaSigma, R_PROJ, z0)
    c.add('DeltaSigma_components', e.DeltaSigma_components, R_PROJ, z0)
    return c.done()


def probes_galaxy_pgg(e):
    """
    Pgg with an array z on a cold cache — the only ordering that reaches the
    `self.cache_matches` typo at GalaxyStats.py:1139. The first _get_weights
    populates _uk_grid_cache and the second one inside the same call hits it.
    A scalar z short-circuits earlier (len() on a numpy scalar raises TypeError
    inside the shadowing static _cache_matches), and a warm _pgg_fourier_cache
    returns at GalaxyStats.py:1301 before _get_weights is ever called.
    Must therefore run before probes_galaxy/probes_galaxy_cache.
    """
    c = _Collector()
    c.add('Pgg[array-z,cold]', e.Pgg, K_NL, np.array([0.5]))
    return c.done()


def probes_galaxy_cache(e):
    """
    Repeated identical calls on scalar z/r. GalaxyStats keys its cache on those,
    so this is the invariant that removing the shadowing static _cache_matches
    (GalaxyStats.py:922-930) must preserve: the repeat must equal the warm-up.
    """
    z0 = 0.5
    c = _Collector()
    for name, fn, grid, kw in (('Xigm', e.Xigm, R_GRID, {}),
                               ('Xigg', e.Xigg, R_GRID, {}),
                               ('Pgm', e.Pgm, K_NL, {}),
                               ('Pgg', e.Pgg, K_NL, {}),
                               ('wgm', e.wgm, RP_PROJ, {}),
                               ('wp', e.wp, RP_PROJ, dict(pimax=40.0)),
                               ('DeltaSigma', e.DeltaSigma, R_PROJ, {})):
        fn(grid, z0, **kw)                              # warm the cache
        c.add('%s[repeat]' % name, fn, grid, z0, **kw)  # must be identical
    return c.done()


def probes_tk(e):
    c = _Collector()
    c.add('get_Tk[linear]', e.get_Tk, z=Z_ALL, k=K_LIN, Pcb=False, Tk_type='linear')
    return c.done()


# ------------------------------------------------------- call-order probes
def probes_callorder(cosmo_a, cosmo_b):
    """
    set_cosmos(A) -> evaluate -> set_cosmos(B) -> evaluate, against a freshly
    constructed instance at B. Equal results mean no cache survived the
    cosmology change. This is the target of commit 1.
    """
    out = {}
    for tag, factory, runner in (
        ('xihm', Xihm_CEmulator, lambda e: _flat(
            e.get_xihm_lgnbar_threshold(z=Z_LOW[:1], r=R_GRID[:8],
                                        lgden=LGDEN[:1], massdef='RockstarM200m'))),
        ('pkhm', Xihm_CEmulator, lambda e: _flat(
            e.get_pkhm_lgnbar_threshold(z=Z_LOW[:1], k=K_NL[:8],
                                        lgden=LGDEN[:1], massdef='RockstarM200m'))),
        ('galaxy_Xigm', GalaxyEmulator, lambda e: _flat(e.Xigm(R_GRID[:8], 0.5))),
        ('galaxy_DeltaSigma', GalaxyEmulator, lambda e: _flat(e.DeltaSigma(R_PROJ[:8], 0.5))),
    ):
        seq = factory(verbose=False)
        seq.set_cosmos(**cosmo_a)
        is_gal = tag.startswith('galaxy')
        if is_gal:
            seq.set_hod('Zheng05', **HOD_PARAMS)
        runner(seq)                                   # populate caches under A
        seq.set_cosmos(**cosmo_b)
        if is_gal:
            seq.set_hod('Zheng05', **HOD_PARAMS)
        out['co_%s_seq' % tag] = runner(seq)
        del seq
        gc.collect()

        fresh = factory(verbose=False)
        fresh.set_cosmos(**cosmo_b)
        if is_gal:
            fresh.set_hod('Zheng05', **HOD_PARAMS)
        out['co_%s_fresh' % tag] = runner(fresh)
        del fresh
        gc.collect()
    return out


# ---------------------------------------------------------------- driver
GROUPS = [
    ('pkmm', Pkmm_CEmulator, [('pkmm', probes_pkmm)]),
    ('galaxy', GalaxyEmulator, [('hmf', probes_hmf), ('ximm', probes_ximm),
                                ('pkhm', probes_pkhm), ('xihm', probes_xihm),
                                ('xihh', probes_xihh), ('galpgg', probes_galaxy_pgg),
                                ('galaxy', probes_galaxy),
                                ('galcache', probes_galaxy_cache)]),
    ('tk', Tkmm_CEmulator, [('tk', probes_tk)]),
]


def collect(cosmo_name, cosmo, verbose=False):
    vals, errs, timing, rss = {}, {}, {}, {}
    for gname, factory, runners in GROUPS:
        t0 = time.time()
        emu = factory(verbose=verbose)
        emu.set_cosmos(**cosmo)
        if gname == 'galaxy':
            emu.set_hod('Zheng05', **HOD_PARAMS)
        timing['%s|%s|construct+set' % (cosmo_name, gname)] = time.time() - t0
        rss['%s|%s' % (cosmo_name, gname)] = _rss_mb()
        for rname, fn in runners:
            t1 = time.time()
            try:
                got_vals, got_errs = fn(emu)
            except Exception as exc:
                errs['%s.%s.%s' % (cosmo_name, gname, rname)] = \
                    '%s: %s' % (type(exc).__name__, exc)
                if verbose:
                    traceback.print_exc()
            else:
                for kk, vv in got_vals.items():
                    vals['%s|%s|%s|%s' % (cosmo_name, gname, rname, kk)] = vv
                for kk, vv in got_errs.items():
                    errs['%s|%s|%s|%s' % (cosmo_name, gname, rname, kk)] = vv
            timing['%s|%s|%s|%s' % (cosmo_name, gname, rname, '')] = time.time() - t1
        del emu
        gc.collect()
    return vals, errs, timing, rss


def collect_all(verbose=False, with_callorder=True):
    vals, errs, timing, rss = {}, {}, {}, {}
    for cname, cosmo in COSMOS.items():
        print('[%s] collecting ...' % cname, flush=True)
        v, e, t, m = collect(cname, cosmo, verbose=verbose)
        vals.update(v)
        errs.update(e)
        timing.update(t)
        rss.update(m)
        print('[%s] %d arrays, %d probe errors, peak RSS %.0f MB'
              % (cname, len(v), len(e), _rss_mb()), flush=True)
    if with_callorder:
        print('[callorder] collecting ...', flush=True)
        t0 = time.time()
        co = probes_callorder(COSMOS['c0'], COSMOS['c2'])
        for kk, vv in co.items():
            vals['callorder|callorder|callorder|%s' % kk] = _flat(vv)
        timing['callorder|callorder|total|'] = time.time() - t0
        rss['callorder'] = _rss_mb()
    return vals, errs, timing, rss


def rebuild(verbose=False):
    vals, errs, timing, rss = collect_all(verbose=verbose)
    np.savez(BASE_NPZ, **vals)
    meta = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
                numpy=np.__version__, errors=errs, timing=timing, rss_mb=rss,
                cosmos={k: {kk: float(vv) for kk, vv in v.items()} for k, v in COSMOS.items()},
                shapes={k: list(v.shape) for k, v in vals.items()})
    with open(BASE_JSON, 'w') as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print('\nwrote %s (%d arrays) and %s' % (BASE_NPZ, len(vals), BASE_JSON))
    if errs:
        print('\nprobes that raised (recorded as expected-to-raise):')
        for kk in sorted(errs):
            print('  %-66s %s' % (kk, errs[kk]))


def _allowed(name, allows):
    for grp in allows:
        for pat in WHITELIST.get(grp, {}):
            if pat in name:
                return grp, pat
    return None, None


def compare(tol, allows, verbose=False):
    if not (os.path.exists(BASE_NPZ) and os.path.exists(BASE_JSON)):
        print('no baseline found; run with --rebuild first')
        return 2
    with open(BASE_JSON) as fh:
        meta = json.load(fh)
    base = np.load(BASE_NPZ)
    bvals = {k: base[k] for k in base.files}
    berrs = meta['errors']

    vals, errs, timing, rss = collect_all(verbose=verbose)

    bad, whitelisted, fixed, nowbroken = [], [], [], []
    for name in sorted(set(bvals) | set(vals)):
        hit, pat = _allowed(name, allows)
        inb, inc = name in bvals, name in vals
        if inb and not inc:
            (whitelisted if hit else bad).append(
                (name, 'probe disappeared', hit or ''))
            continue
        if inc and not inb:
            (whitelisted if hit else bad).append((name, 'new probe appeared', hit or ''))
            continue
        a, b = bvals[name], vals[name]
        if a.shape != b.shape:
            (whitelisted if hit else bad).append(
                (name, 'shape %s -> %s' % (a.shape, b.shape), hit or ''))
            continue
        same_special = all(np.array_equal(mask(a), mask(b))
                           for mask in (np.isnan, np.isposinf, np.isneginf))
        fin = np.isfinite(a) & np.isfinite(b)
        denom = np.maximum(np.abs(a[fin]), 1e-300)
        rel = np.abs(b[fin] - a[fin]) / denom if fin.any() else np.array([0.0])
        absd = np.abs(b[fin] - a[fin]) if fin.any() else np.array([0.0])
        ok = same_special and (rel.max() <= tol or absd.max() <= tol)
        if not ok:
            msg = 'max rel %.3e / max abs %.3e / nonfinite-pattern-equal %s' % (
                rel.max(), absd.max(), same_special)
            (whitelisted if hit else bad).append((name, msg, hit or ''))

    for name in sorted(set(berrs) | set(errs)):
        hit, pat = _allowed(name, allows)
        if name in berrs and name not in errs:
            (whitelisted if hit else nowbroken).append(
                (name, 'was raising "%s", now returns a value' % berrs[name], hit or ''))
        elif name in errs and name not in berrs:
            bad.append((name, 'newly raises: %s' % errs[name], ''))
        elif name in errs and errs[name].split(':')[0] != berrs[name].split(':')[0]:
            bad.append((name, 'exception type changed: %s -> %s'
                        % (berrs[name], errs[name]), ''))

    # Call-order self-consistency: seq must equal fresh once commit 1 lands.
    co_bad = []
    for tag in ('xihm', 'pkhm', 'galaxy_Xigm', 'galaxy_DeltaSigma'):
        ks, kf = ('callorder|callorder|callorder|co_%s_seq' % tag,
                  'callorder|callorder|callorder|co_%s_fresh' % tag)
        if ks in vals and kf in vals:
            a, b = vals[ks], vals[kf]
            finite = np.all(np.isfinite(a)) and np.all(np.isfinite(b))
            rel = (np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
                   if finite and a.shape == b.shape else np.inf)
            if not (rel <= tol):
                co_bad.append((tag, rel))
        else:
            co_bad.append((tag, np.inf))

    print('\n' + '=' * 74)
    print('arrays compared : %d' % len(set(bvals) & set(vals)))
    print('tolerance       : %.1e' % tol)
    print('allow groups    : %s' % (','.join(allows) or '(none)'))
    print('peak RSS        : %.0f MB' % _rss_mb())
    if whitelisted:
        print('\nwhitelisted deviations (%d) -- expected for the in-flight commit:' % len(whitelisted))
        for n, m, g in whitelisted:
            print('  [%s] %-52s %s' % (g, n, m))
    if nowbroken:
        print('\nprobes that started working (%d):' % len(nowbroken))
        for n, m, g in nowbroken:
            print('  %-58s %s' % (n, m))
    if co_bad:
        print('\ncall-order inconsistency (commit 1 target) (%d):' % len(co_bad))
        for tag, rel in co_bad:
            print('  %-20s max|seq/fresh - 1| = %.4e' % (tag, rel))
    if bad:
        print('\nREGRESSIONS (%d):' % len(bad))
        for n, m, _ in bad:
            print('  %-58s %s' % (n, m))
    print('=' * 74)

    slow = sorted(timing.items(), key=lambda kv: -kv[1])[:12]
    if slow:
        print('\nslowest probes:')
        for k, v in slow:
            print('  %9.1f ms  %s' % (v * 1e3, k))
    print('\npeak RSS per group: %s' % ', '.join(
        '%s=%.0fMB' % (k, v) for k, v in sorted(rss.items())))

    if bad or co_bad:
        print('\nFAIL: %d unwhitelisted regression(s), %d call-order failure(s)'
              % (len(bad), len(co_bad)))
        return 1
    print('\nPASS: no unwhitelisted regressions')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rebuild', action='store_true', help='(re)write the baseline')
    ap.add_argument('--tol', type=float, default=1e-12, help='relative/absolute tolerance')
    ap.add_argument('--allow', nargs='*', default=[], choices=sorted(WHITELIST),
                    help='whitelist groups whose deviations are expected')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    if a.rebuild:
        rebuild(verbose=a.verbose)
        return 0
    return compare(a.tol, a.allow, verbose=a.verbose)


if __name__ == '__main__':
    sys.exit(main())
