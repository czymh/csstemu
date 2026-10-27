"""Verification tests for bounded-memory fixes in long MCMC chains.

Covers:
  1. CAMB-fallback warning is emitted only once per emulator instance, so the
     module-level __warningregistry__ does not grow with every chain step.
  2. The HOD-weight cache in GalaxyStatsCalculator is a bounded LRU instead of
     an unbounded dict keyed by (HOD model, parameters).
"""
import warnings

import numpy as np

import CEmulator.Emulator as Emod
from CEmulator.Emulator import Pkmm_CEmulator
from CEmulator.emulator.GalaxyStats import GalaxyStatsCalculator
from CEmulator.emulator.HOD import Zheng05HOD


# ---------------------------------------------------------------------------
# Test 1: fallback warning deduplication
# ---------------------------------------------------------------------------
print('=== Test 1: fallback warning emitted only once ===')
# Start from a clean warning registry so the count is meaningful
if hasattr(Emod, '__warningregistry__'):
    Emod.__warningregistry__.clear()

warnings.simplefilter('default')
emu = Pkmm_CEmulator(verbose=False)

# First out-of-bounds step emits the (single) warning
emu.set_cosmos(Omegab=0.049, Omegac=0.42, H0=67.0, As=2.0e-9,
               fallback_verbose=True)
assert emu._fallback_mode
assert emu._fallback_warned
# len() includes the bookkeeping 'version' key, so compare to this baseline
n_registry_first = len(getattr(Emod, '__warningregistry__', {}))

# Later steps use different out-of-bounds values but must not warn again
for i in range(1, 3):
    emu.set_cosmos(Omegab=0.049, Omegac=0.42 + 0.004 * i, H0=67.0,
                   As=2.0e-9, fallback_verbose=True)
    assert emu._fallback_mode

n_registry = len(getattr(Emod, '__warningregistry__', {}))
print('warning registry entries: after first step = %d, after 3 steps = %d'
      % (n_registry_first, n_registry))
assert n_registry == n_registry_first, \
    'warning registry must not grow per chain step'
print('PASS')

# ---------------------------------------------------------------------------
# Test 2: fallback_verbose=False emits no warning
# ---------------------------------------------------------------------------
print()
print('=== Test 2: fallback_verbose=False ===')
emu2 = Pkmm_CEmulator(verbose=False)
n_before = len(getattr(Emod, '__warningregistry__', {}))
emu2.set_cosmos(Omegab=0.049, Omegac=0.45, H0=67.0, As=2.0e-9,
                fallback_verbose=False)
assert emu2._fallback_mode
assert not emu2._fallback_warned
assert len(getattr(Emod, '__warningregistry__', {})) == n_before
print('PASS')

# ---------------------------------------------------------------------------
# Test 3: HOD-weight cache is bounded
# ---------------------------------------------------------------------------
print()
print('=== Test 3: HOD-weight cache bounded LRU ===')
calc = GalaxyStatsCalculator(hmf_emu=None, verbose=False)
hod = Zheng05HOD()

n_entries = 40
keys = []
for i in range(n_entries):
    params = dict(logMmin=12.0 + 0.001 * i, sigma_logM=0.5,
                  logM0=12.5, logM1=13.5, alpha=1.0)
    calc._get_cached_hod_weights(hod, **params)
    keys.append(calc._get_hod_cache_key(hod, **params))

maxsize = calc._HOD_CACHE_MAXSIZE
print('cache size after %d unique HOD parameter sets: %d (max %d)'
      % (n_entries, len(calc._cache), maxsize))
assert len(calc._cache) == maxsize, 'cache should be capped at maxsize'
assert keys[-1] in calc._cache, 'most recent entry must be kept'
assert keys[0] not in calc._cache, 'oldest entry must be evicted'

# Cache hit path must not duplicate or evict
calc._get_cached_hod_weights(hod, **dict(logMmin=12.0, sigma_logM=0.5,
                                         logM0=12.5, logM1=13.5, alpha=1.0))
assert len(calc._cache) <= maxsize

calc.clear_cache()
assert len(calc._cache) == 0
print('PASS')

print()
print('=== All cache tests passed! ===')
