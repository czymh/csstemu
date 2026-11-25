"""Verification tests for CAMB fallback feature."""
import warnings
warnings.filterwarnings('ignore')
from CEmulator.Emulator import Pkmm_CEmulator
from CEmulator import CEmulatorFallbackError
import numpy as np

# Test 1: Normal parameters - no fallback
print('=== Test 1: Normal parameters - no fallback ===')
emu = Pkmm_CEmulator(verbose=False)
emu.set_cosmos(Omegab=0.049, Omegac=0.25, H0=67.66, As=2.0e-9)
assert emu._fallback_mode == False, 'fallback_mode should be False'
assert emu._camb_results_cache is None, 'camb cache should be None'
print('PASS')

# Test 2: Out-of-bounds parameters - fallback triggered
print()
print('=== Test 2: Out-of-bounds Omegac=0.45 - fallback ===')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    emu.set_cosmos(Omegab=0.049, Omegac=0.45, H0=67.66, As=2.0e-9)
    if w:
        print('Warning: {}'.format(w[0].message))
assert emu._fallback_mode == True, 'fallback_mode should be True'
assert emu._camb_results_cache is not None, 'camb cache should not be None'
print('PASS')

# Test 3: get_pklin in fallback mode
print()
print('=== Test 3: get_pklin in fallback ===')
k = np.logspace(-3, 0, 50)
pk = emu.get_pklin(z=0, k=k)
assert pk.shape == (1, 50), 'shape mismatch'
assert np.all(np.isfinite(pk)), 'values not finite'
print('PASS')

# Test 4: get_pknl in fallback mode
print()
print('=== Test 4: get_pknl in fallback ===')
k = np.logspace(-2, 0, 30)
pk = emu.get_pknl(z=0, k=k, Pcb=False)
assert pk.shape == (1, 30), 'shape mismatch'
assert np.all(np.isfinite(pk)), 'values not finite'
print('PASS')

# Test 5: get_pkHMCODE2020 with Pcb=False (was ValueError before)
print()
print('=== Test 5: get_pkHMCODE2020 Pcb=False ===')
k = np.logspace(-2, 0, 30)
pk = emu.get_pkHMCODE2020(z=0, k=k, Pcb=False)
assert pk.shape == (1, 30), 'shape mismatch'
assert np.all(np.isfinite(pk)), 'values not finite'
print('PASS')

# Test 6: get_sigma8 in fallback
print()
print('=== Test 6: get_sigma8 in fallback ===')
s8 = emu.get_sigma8()
assert 0.5 < s8 < 1.2, 'sigma8 value out of expected range'
print('sigma8: {:.6f}'.format(s8))
print('PASS')

# Test 7: fallback_to_camb=False raises
print()
print('=== Test 7: fallback_to_camb=False ===')
try:
    emu.set_cosmos(Omegab=0.049, Omegac=0.45, H0=67.66, As=2.0e-9,
                   fallback_to_camb=False)
    print('ERROR: Should have raised ValueError')
    assert False, 'should have raised'
except ValueError as e:
    print('ValueError raised as expected')
print('PASS')

# Test 8: R bug fix - different R gives different sigma
print()
print('=== Test 8: R bug fix ===')
emu2 = Pkmm_CEmulator(verbose=False)
emu2.set_cosmos(Omegab=0.049, Omegac=0.25, H0=67.66, As=2.0e-9)
s8_camb = emu2.get_sigma_z(0, 8.0, type='CAMB')
s10_camb = emu2.get_sigma_z(0, 10.0, type='CAMB')
print('sigma(R=8): {:.6f}, sigma(R=10): {:.6f}'.format(s8_camb, s10_camb))
assert not np.isclose(s8_camb, s10_camb), 'R=8 and R=10 should give different sigma'
print('PASS')

# Test 9: Re-set_cosmos with valid params clears fallback
print()
print('=== Test 9: Cache invalidation on re-set_cosmos ===')
emu.set_cosmos(Omegab=0.049, Omegac=0.25, H0=67.66, As=2.0e-9)
assert emu._fallback_mode == False, 'fallback_mode should be cleared'
print('PASS')

# Test 10: sigma8 conversion with fallback
print()
print('=== Test 10: sigma8=0.8 with out-of-bounds via sigma8type=CAMB ===')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    emu.set_cosmos(Omegab=0.049, Omegac=0.45, H0=67.66, sigma8=0.8,
                   sigma8type='CAMB')
assert emu._fallback_mode == True, 'should be in fallback'
print('PASS')

print()
print('=== Test 11: default sigma8type uses CAMB when out of bounds ===')
_types = []
_raw_s8 = emu.get_sigma8
emu.get_sigma8 = lambda *a, **kw: (
    _types.append(kw.get('type', a[0] if a else 'Emulator')), _raw_s8(*a, **kw))[1]
emu.set_cosmos(Omegab=0.049, Omegac=0.45, H0=67.66, sigma8=0.81)
emu.get_sigma8 = _raw_s8
assert emu._fallback_mode == True, 'should be in fallback'
assert len(_types) > 0, 'the sigma8 iteration never ran'
assert set(_types) == {'CAMB'}, \
    'out-of-bounds iteration must not extrapolate the emulator, used %s' % _types
assert abs(emu.get_sigma8() - 0.81) < 1e-6, 'sigma8 did not converge'
print('sigma8 types used inside the iteration: {}'.format(_types))
print('PASS')

print()
print('=== Test 12: sigma8 target whose As lands outside [1.7, 2.5] ===')
for _s8 in (0.70, 0.95):
    emu.set_cosmos(Omegab=0.0486, Omegac=0.2604, H0=68.98, ns=0.9665,
                   sigma8=_s8, w=-1.0, wa=0.0, mnu=0.06)
    _A = emu.cosmologies[0][4]
    _out = not (1.7 <= _A <= 2.5)
    assert emu._fallback_mode == _out, 'fallback flag disagrees with the A bound'
    assert abs(emu.get_sigma8() - _s8) < 1e-4, 'sigma8=%.3f did not converge' % _s8
    print('sigma8=%.2f -> As*1e9=%.6f fallback=%s err=%.2e'
          % (_s8, _A, emu._fallback_mode, abs(emu.get_sigma8() - _s8)))
print('PASS')

print()
print('=== Test 13: CAMB failure becomes CEmulatorFallbackError ===')
emu3 = Pkmm_CEmulator(verbose=False)
try:
    emu3.set_cosmos(Omegab=0.049, Omegac=0.25, H0=67.66, As=1.0e-7)
    assert False, 'an absurd As must not silently succeed'
except CEmulatorFallbackError as e:
    assert isinstance(e, RuntimeError), 'must be catchable as RuntimeError'
    assert e.__cause__ is not None, 'the original CAMB error must be chained'
    assert 'Omegab=' in str(e), 'the message must carry the parameter set'
    print('CEmulatorFallbackError, cause={}'.format(type(e.__cause__).__name__))
print('PASS')

print()
print('=== Test 14: one CAMB run per redshift, not per sigma call ===')
from CEmulator.Emulator import HMF_CEmulator
emu4 = HMF_CEmulator(verbose=False)
_ncamb = [0]
_raw_camb = emu4.get_camb_results
emu4.get_camb_results = lambda *a, **kw: (_ncamb.__setitem__(0, _ncamb[0]+1), _raw_camb(*a, **kw))[1]
emu4.set_cosmos(Omegab=0.049, Omegac=0.45, H0=67.66, As=2.0e-9)
_nset = _ncamb[0]
_zs = np.array([0.0, 0.5, 1.0])
_dn = emu4.get_dndlnM_Castro23(z=_zs, M=np.logspace(12, 15, 8), massdef='RockstarM200m')
emu4.get_camb_results = _raw_camb
assert np.all(np.isfinite(_dn)), 'dndlnM not finite'
# each redshift needs 3 sigma calls (2 in _get_dlns_dlnR + 1 for sigma itself);
# the per-z memo must collapse them onto a single CAMB run
assert _ncamb[0] - _nset == len(_zs), \
    'expected %d CAMB runs, got %d' % (len(_zs), _ncamb[0] - _nset)
print('CAMB runs for %d redshifts: %d' % (len(_zs), _ncamb[0] - _nset))
print('PASS')

print()
print('=== All tests passed! ===')
