# Changelog

All notable changes to CEmulator will be documented in this file.

---

## [Unreleased]

### Added

- **CAMB fallback mechanism** (`fallback_to_camb=True` by default in `set_cosmos()`). When any cosmological parameter exceeds the GP emulator training range, the emulator automatically falls back to CAMB for computation instead of raising `ValueError`. Affected functions: `get_pklin`, `get_pknl`, `get_pkHMCODE2020`, `get_pkhalofit`, `get_sigma_z`, `get_sigma_cb_z`, `get_sigma8`, `get_sigma8_cb`.
- **`fallback_verbose` parameter** (default `True`). Controls whether a `UserWarning` is emitted when falling back to CAMB.
- **CAMB results caching** (`_camb_results_cache`). When fallback mode is triggered, CAMB is computed once in `set_cosmos()` and cached for reuse across all `get_*` calls, avoiding repeated expensive computations.
- **CLASS lazy-initialization cache** (`_cosmo_class_cache`). The CLASS cosmology object is now created on first use (via `_get_cosmo_class_cache()`) rather than per-`get_*` call, then reused. The cache is cleared in `_sync_cosmologies()` alongside other caches.
- **`_get_cosmo_class_cache()` helper method** on `CBaseEmulator`. Provides lazy-init access to the cached CLASS object with `kmax=100`.

### Fixed

- **`get_sigma_z()` and `get_sigma_cb_z()` CAMB path**: hardcoded `R=8.0` replaced with the actual `R` argument, so `type='CAMB'` now correctly returns `sigma(R)` for any smoothing scale, not just `R=8.0`.
- **`get_pkHMCODE2020()` with `Pcb=False`**: when in CAMB fallback mode, `Pcb=False` now works correctly (previously raised `ValueError` in the Emulator path).

### Changed

- **`set_cosmos()` behavior change**: with `fallback_to_camb=True` (default), out-of-bounds parameters no longer raise `ValueError` — they trigger CAMB fallback instead. Set `fallback_to_camb=False` to restore the previous strict behavior.
- **CLASS paths in `get_*` functions**: `if cosmo_class is None: cosmo_class = self.get_cosmo_class(...)` replaced with `cosmo_class = self._get_cosmo_class_cache()`, avoiding repeated CLASS object creation.
- **`set_cosmos()` now resets `_fallback_mode = False`** at the start of each call, ensuring clean state on re-invocation.
- **sigma8→As iteration** in `set_cosmos()`: if fallback mode is active and `sigma8type='Emulator'`, automatically switches to `sigma8type='CAMB'` to avoid GP emulation failure.

### Internal

- Added instance variables `_fallback_mode`, `_camb_results_cache`, `_cosmo_class_cache` to `CBaseEmulator.__init__`.
- Cache clearing logic added to `CBaseEmulator._sync_cosmologies()`, which propagates to all subclasses (`Pkmm_CEmulator`, `HMF_CEmulator`, etc.).
- Verification tests added at `test/test_camb_fallback.py` covering 10 scenarios: normal/fallback modes, cache invalidation, `R` bug fix, `Pcb=False`, `fallback_to_camb=False`, and sigma8 conversion.
