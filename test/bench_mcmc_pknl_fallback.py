#!/usr/bin/env python3
"""Benchmark time and memory of repeated ``set_cosmos`` + ``get_pknl``.

Simulates an MCMC chain that draws a new cosmology, calls
``CEmulator.Emulator.Pkmm_CEmulator.set_cosmos()`` and then predicts the
nonlinear power spectrum with ``get_pknl()``. Four cases are run on the same
kind of parameter draws:

  1. in-bounds cosmology, ``fallback_to_camb=False``  -> GP emulator (strict)
  2. in-bounds cosmology, ``fallback_to_camb=True``   -> GP emulator (default)
  3. out-of-bounds cosmology, ``fallback_to_camb=True``  -> CAMB fallback
  4. out-of-bounds cosmology, ``fallback_to_camb=False`` -> ValueError

Case 3 is by far the slowest: every ``set_cosmos`` runs CAMB on all 12
training redshifts (``non_linear='mead2020'``), about 6-10 s/step on the
shared node, so 200 steps take roughly 20-35 minutes. Case 4 aborts at the
parameter-range check and never predicts P(k), which is the point: without
fallback an out-of-domain chain cannot run.

Memory is read from ``/proc/self/status`` (VmRSS). Note that the ~236 MB
baseline is the pre-trained GP data loaded once at construction, not
per-step growth.

Single-threaded BLAS is forced by default: the GP predictions are small
matrices and multithreaded MKL only adds contention on a shared node.

Usage
-----
    python3 test/bench_mcmc_pknl_fallback.py                 # 200 steps/case
    python3 test/bench_mcmc_pknl_fallback.py --niter 20      # quick check
    python3 test/bench_mcmc_pknl_fallback.py --csv bench.csv # dump per-step data
"""
import os

# Must be set before numpy is imported. GP predictions are tiny matrices, so
# multithreaded MKL/OpenMP only adds spin-wait contention on shared nodes.
for _var in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import csv
import gc
import time
import warnings

import numpy as np

from CEmulator.Emulator import Pkmm_CEmulator


def rss_mb():
    """Current resident set size in MB (Linux)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_params(rng, out_of_bounds):
    """Draw one cosmology; only Omegam is pushed outside the design range."""
    Omegab = rng.uniform(0.040, 0.060)
    if out_of_bounds:
        Omegam = rng.uniform(0.41, 0.49)   # limit is [0.24, 0.40]
    else:
        Omegam = rng.uniform(0.25, 0.39)
    return dict(
        Omegab=Omegab,
        Omegac=Omegam - Omegab,
        H0=rng.uniform(60.0, 80.0),
        As=rng.uniform(1.7, 2.5) * 1e-9,
        ns=rng.uniform(0.92, 1.00),
        w=rng.uniform(-1.3, -0.7),
        wa=rng.uniform(-0.5, 0.5),
        mnu=rng.uniform(0.0, 0.3),
    )


def stats_ms(values):
    """(mean, median, p90) in milliseconds."""
    if not values:
        return float('nan'), float('nan'), float('nan')
    a = np.asarray(values, dtype=float) * 1000.0
    return a.mean(), np.median(a), np.percentile(a, 90)


def run_case(emu, name, niter, out_of_bounds, fallback, kvec, seed,
             csv_writer=None):
    """Run ``niter`` set_cosmos+get_pknl steps and return a result dict."""
    rng = np.random.default_rng(seed)
    gc.collect()
    rss_start = rss_mb()
    rss_peak = rss_start
    t_set, t_pk = [], []
    n_fail = 0
    t_case = time.perf_counter()

    print('\n--- %s ---' % name, flush=True)
    print('    niter=%d, fallback_to_camb=%s, no pknl will be computed on failure'
          % (niter, fallback), flush=True)

    for i in range(niter):
        params = make_params(rng, out_of_bounds)

        t0 = time.perf_counter()
        try:
            emu.set_cosmos(checkbound=True,
                           fallback_to_camb=fallback,
                           fallback_verbose=False, **params)
        except ValueError:
            n_fail += 1
            continue
        t1 = time.perf_counter()

        emu.get_pknl(z=0.0, k=kvec, Pcb=False)
        t2 = time.perf_counter()

        t_set.append(t1 - t0)
        t_pk.append(t2 - t1)

        rss = rss_mb()
        if rss > rss_peak:
            rss_peak = rss

        if csv_writer is not None:
            csv_writer.writerow([name, i, t1 - t0, t2 - t1, rss])

        if (i + 1) % max(1, niter // 10) == 0:
            print('    %3d/%d  elapsed=%6.1fs  mean set_cosmos=%7.1f ms  '
                  'mean pknl=%6.2f ms  rss=%.1f MB'
                  % (i + 1, niter, time.perf_counter() - t_case,
                     np.mean(t_set) * 1000 if t_set else float('nan'),
                     np.mean(t_pk) * 1000 if t_pk else float('nan'),
                     rss), flush=True)

    if n_fail and not t_pk:
        print('    all %d steps raised ValueError at the parameter-range '
              'check (no P(k) produced)' % n_fail, flush=True)

    gc.collect()
    rss_end = rss_mb()
    wall = time.perf_counter() - t_case

    set_mean, set_med, set_p90 = stats_ms(t_set)
    pk_mean, pk_med, pk_p90 = stats_ms(t_pk)
    window = min(20, max(1, len(t_pk) // 5))
    pk_first = np.mean(np.asarray(t_pk[:window]) * 1000) if t_pk else float('nan')
    pk_last = np.mean(np.asarray(t_pk[-window:]) * 1000) if t_pk else float('nan')

    return dict(
        name=name, niter=niter, n_ok=len(t_pk), n_fail=n_fail,
        set_mean=set_mean, set_med=set_med, set_p90=set_p90,
        pk_mean=pk_mean, pk_med=pk_med, pk_p90=pk_p90,
        pk_first=pk_first, pk_last=pk_last,
        rss_start=rss_start, rss_peak=rss_peak, rss_end=rss_end,
        wall=wall,
    )


def print_table(results):
    print('\n' + '=' * 132)
    print('%-38s %5s %5s %11s %11s %11s %9s %11s %9s %9s'
          % ('case', 'ok', 'fail', 'set mean', 'set med', 'set p90',
             'pknl mean', 'pknl med', 'first20', 'last20'))
    print('-' * 132)
    for r in results:
        print('%-38s %5d %5d %9.1f ms %9.1f ms %9.1f ms %7.2f ms %9.2f ms '
              '%7.2f ms %7.2f ms'
              % (r['name'], r['n_ok'], r['n_fail'],
                 r['set_mean'], r['set_med'], r['set_p90'],
                 r['pk_mean'], r['pk_med'], r['pk_first'], r['pk_last']))
    print('=' * 132)
    print('\n%-38s %11s %11s %11s %11s %11s'
          % ('case', 'total wall', 'RSS start', 'RSS peak', 'RSS end',
             'RSS growth'))
    print('-' * 132)
    for r in results:
        print('%-38s %9.1f s %9.1f MB %9.1f MB %9.1f MB %9.1f MB'
              % (r['name'], r['wall'], r['rss_start'], r['rss_peak'],
                 r['rss_end'], r['rss_end'] - r['rss_start']))
    print('=' * 132)
    print('Baseline RSS is the pre-trained GP data (~236 MB), loaded once at '
          'construction; "growth" is the per-case change after gc.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--niter', type=int, default=200,
                        help='steps per case (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for cosmology draws (default: 42)')
    parser.add_argument('--csv', type=str, default=None,
                        help='optional CSV file for per-step timings')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    kvec = np.logspace(-3.0, 1.0, 256)   # h/Mpc

    print('Loading Pkmm_CEmulator ...', flush=True)
    emu = Pkmm_CEmulator(verbose=False)
    print('Loaded. Baseline RSS = %.1f MB' % rss_mb(), flush=True)
    print('BLAS threads: MKL=%s OMP=%s OPENBLAS=%s'
          % (os.environ.get('MKL_NUM_THREADS'),
             os.environ.get('OMP_NUM_THREADS'),
             os.environ.get('OPENBLAS_NUM_THREADS')), flush=True)
    print('NOTE: out-of-bounds + fallback=True runs CAMB on 12 redshifts at '
          'every step (~6-10 s/step); 200 steps ~= 20-35 min.', flush=True)

    csv_file = open(args.csv, 'w', newline='') if args.csv else None
    csv_writer = csv.writer(csv_file) if csv_file is not None else None
    if csv_writer is not None:
        csv_writer.writerow(['case', 'step', 'set_cosmos_s', 'pknl_s', 'rss_mb'])

    results = []
    try:
        # In-domain: pure GP emulator, both fallback settings
        results.append(run_case(
            emu, 'in-bounds, fallback=False',
            args.niter, out_of_bounds=False, fallback=False,
            kvec=kvec, seed=args.seed, csv_writer=csv_writer))
        results.append(run_case(
            emu, 'in-bounds, fallback=True',
            args.niter, out_of_bounds=False, fallback=True,
            kvec=kvec, seed=args.seed + 1, csv_writer=csv_writer))
        # Out-of-domain: fallback=True uses CAMB; fallback=False raises
        results.append(run_case(
            emu, 'out-of-bounds, fallback=True (CAMB)',
            args.niter, out_of_bounds=True, fallback=True,
            kvec=kvec, seed=args.seed + 2, csv_writer=csv_writer))
        results.append(run_case(
            emu, 'out-of-bounds, fallback=False',
            args.niter, out_of_bounds=True, fallback=False,
            kvec=kvec, seed=args.seed + 2, csv_writer=csv_writer))
    finally:
        if csv_file is not None:
            csv_file.close()

    print_table(results)


if __name__ == '__main__':
    main()
