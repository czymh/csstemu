"""Verification tests for CAMB fallback feature."""
import warnings
warnings.filterwarnings('ignore')
from CEmulator.Emulator import Pkmm_CEmulator
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
print('=== All tests passed! ===')
