# Changelog

All notable changes to CEmulator will be documented in this file.

---

## [2.2.0] - 2026-11-25

### Added

- Weak-lensing `Cell.ipynb` example for source planes and source distributions, plus an examples index.
- Snapshot-verdict tests and a direct xi_hh cosmology call-order guard, keeping the existing numerical baseline.

- **CAMB fallback mechanism** (`fallback_to_camb=True` by default in `set_cosmos()`). When any cosmological parameter exceeds the GP emulator training range, the emulator automatically falls back to CAMB for computation instead of raising `ValueError`. Affected functions: `get_pklin`, `get_pknl`, `get_pkHMCODE2020`, `get_pkhalofit`, `get_sigma_z`, `get_sigma_cb_z`, `get_sigma8`, `get_sigma8_cb`.
- **`fallback_verbose` parameter** (default `True`). Controls whether a `UserWarning` is emitted when falling back to CAMB.
- **CAMB results caching** (`_camb_results_cache`). When fallback mode is triggered, CAMB is computed once in `set_cosmos()` and cached for reuse across all `get_*` calls, avoiding repeated expensive computations.
- **CLASS lazy-initialization cache** (`_cosmo_class_cache`). The CLASS cosmology object is now created on first use (via `_get_cosmo_class_cache()`) rather than per-`get_*` call, then reused. The cache is cleared in `_sync_cosmologies()` alongside other caches.
- **`_get_cosmo_class_cache()` helper method** on `CBaseEmulator`. Provides lazy-init access to the cached CLASS object with `kmax=100`.
- **`Pkmm_CEmulator.get_pknl_extrp()`** function. Provides a stable extrapolation of the nonlinear power spectrum beyond the GP training range (k ∈ [1e-5, 1e2] h/Mpc). Computes the ratio `P_nl(k₀) / P_lin(k₀)` on a well-sampled interior k-grid where the GP emulator is accurate, then extrapolates this ratio to the target k values and multiplies by `P_lin(k)`. Supports `Pcb=True/False`, `nltype='halofit'/'hmcode2020'`, and automatically inherits CAMB fallback behavior.
- **`CEmulatorFallbackError`** (`CEmulator/utils.py`, exported from `CEmulator`). Raised instead of a bare `CAMBError` when the CAMB fallback fails for an out-of-range parameter set, carrying the offending parameters and chaining the original exception; also raised when the sigma8 -> As iteration cannot converge. Subclasses `RuntimeError`, so a sampler can catch it and assign zero likelihood.
- **`set_hod()` reuses the galaxy statistics calculator.** `GalaxyStatsCalculator` holds no HOD state (every `compute_*` receives the model and parameters), so changing the HOD now drops its caches instead of rebuilding the calculator and all 12 redshift grids.
- **Per-redshift lazy precomputation** in `GalaxyStatsCalculator`. The xi_hm / P_hm / xi_hh / P_hh grids are built on demand for the two snapshots bracketing the requested z and reused thereafter.
- **`_get_pcalc()`** on `CBaseEmulator`: memoizes the 1000-point linear P(k) grid used by the sigma_R integral per `(kind, z)`, dropped by `_sync_cosmologies()`.
- **Finite-difference conditioning guard** (`HMF_CEmulator._fd_ok`), applied to `get_bias_mass`, `get_pkhm_mass`, `get_xihm_mass`, `get_xihh_mass` and the three `GalaxyStats` node grids. Bins whose n(>M) barely changes over the 1% step fall back to the tree/Castro23 value instead of dividing by noise, reported once per instance.
- **Extrapolation warnings**: z outside [0, 3] in `get_pklin`, k beyond 10 h/Mpc in `get_pknl`, and lgden outside [-8, -1] in `get_mass_from_lgden` each warn once per instance.
- **`FFTLog` input validation**: rejects non-finite `f_x`, non-positive `x`, odd N and mismatched shapes with an explicit message.
- New verification tests: `test/test_numerical_guards.py`, `test/test_error_handling.py`, `test/test_lazy_precompute.py`, `test/test_sigma_cache.py`.

### Fixed

- Comoving distances now use an 8192-point shared integral, resolving the existing 1e-6 accuracy failures; the cubic interpolator is constructed once per cache build. This intentionally corrects distances and derived weak-lensing angular spectra.
- Finite LOS projections now extend xi itself by zero outside the supplied radial support, instead of restoring a negative value from a zero-filled shifted logarithm. Negative correlations inside the support retain their sign.
- The HOD test entry point now follows all test classes, so lensing and finite-projection checks are executed; their fixtures follow the current lazy-grid interfaces.

- Snapshot comparisons now reject changed NaN/infinity patterns and fail on inconsistent or missing call-order pairs.
- Public examples no longer depend on private simulation files or custom plotting styles; obsolete experiment notebooks and stored outputs were removed.

- **`get_sigma_z()` and `get_sigma_cb_z()` CAMB path**: hardcoded `R=8.0` replaced with the actual `R` argument, so `type='CAMB'` now correctly returns `sigma(R)` for any smoothing scale, not just `R=8.0`.
- **`get_pkHMCODE2020()` with `Pcb=False`**: when in CAMB fallback mode, `Pcb=False` now works correctly (previously raised `ValueError` in the Emulator path).
- **Unbounded warning-registry growth in fallback mode**: `set_cosmos(fallback_verbose=True)` now emits the out-of-range warning only once per emulator instance (`CBaseEmulator._warn_fallback()`, flag `_fallback_warned`). Previously every unique parameter combination added an entry to the module-level `__warningregistry__`, which grew linearly during long MCMC chains.
- **Unbounded HOD-weight cache in `GalaxyStatsCalculator`**: `_cache` is now a bounded LRU (`OrderedDict`, `_HOD_CACHE_MAXSIZE = 16`) instead of a plain dict. With HOD parameters changing at every MCMC step, the old dict grew without limit.
- **Stale caches after `set_cosmos()`**: `Xihm_CEmulator._sync_cosmologies` cleared `_Brhm_interp` on the wrong object, so the xi_hm interpolator never invalidated (4% error after a cosmology change); `GalaxyEmulator` never re-synced an already-built `_xihh_emu` (11.7% error); and the `Omegam` copy frozen into `SatelliteProfile` was not refreshed.
- **`Pgg` crash for array z**: the `cache_matches` typo in `compute_Pgg` raised `AttributeError` for any array-z input; the shadowing `@staticmethod` that hid it is gone.
- **`get_pkhm_mass_threshold` crashed for `len(M) > 1`**: it passed a descending lgden to a consumer that requires ascending; `examples/pkhm.ipynb` cell 5 works again.
- **HMF `addnum` was keyless shared state** across mass definitions; it is now returned alongside each cached spline and bound to it.
- **sigma8 -> As iteration silently used an extrapolating emulator** for out-of-range cosmologies (its fallback branch was disabled by `_sync_cosmologies`), exited without notice on a non-finite sigma8, and extrapolated past the A bounds as As drifted. It is now bounded at 15 steps and re-tests the bounds every iteration.
- **CAMB fallback was unprotected**: a pathological parameter set killed the process with a bare `CAMBError`.
- **`get_pknl(lintype='CAMB', nltype='halofit')`** assigned `cosmo_class` where it needed `camb_results`: one wasted multi-second CLASS run, then `AttributeError` on the neutrino term.
- **An unknown `nltype`** left `Bkpred` unbound (`NameError`); the three `lintype` dispatches formatted the builtin `type` instead of the argument.
- **`GalaxyStats` swallowed every exception** in three precompute loops and left an all-zero block, producing a blank galaxy statistic with no signal; they now raise with the z and mass indices.
- **FFTLog gamma poles**: only `x = mu+1` was special-cased, so `x = mu+1+2k` returned NaN. `_lowring_xy` lost the phase when `gamma()` overflowed or underflowed (measured at N=4096 and 16384) and returned a wrong low-ringing value; it falls back to `loggamma()` there.
- **Power-law extrapolation** divided 0/0 for a zero edge and the two `except:` clauses caught everything including `KeyboardInterrupt`; the range loops could double the array without bound.
- **`Ez` was infinite for m_nu > 0.336 eV** because the neutrino Fermi-Dirac table returned `inf` past its last row; it is now continued linearly.
- **`compute_Rsigma_neff_C` raised `UnboundLocalError`** on the "still linear" exit and its bisection was unbounded.
- **An HMF cache hit re-fit the spline and recomputed 10 GP predictions**; the hit path went from 0.76 ms to 0.075 ms.

### Changed

- **`set_cosmos()` behavior change**: with `fallback_to_camb=True` (default), out-of-bounds parameters no longer raise `ValueError` — they trigger CAMB fallback instead. Set `fallback_to_camb=False` to restore the previous strict behavior.
- **CLASS paths in `get_*` functions**: `if cosmo_class is None: cosmo_class = self.get_cosmo_class(...)` replaced with `cosmo_class = self._get_cosmo_class_cache()`, avoiding repeated CLASS object creation.
- **`set_cosmos()` now resets `_fallback_mode = False`** at the start of each call, ensuring clean state on re-invocation.
- **sigma8→As iteration** in `set_cosmos()`: if fallback mode is active and `sigma8type='Emulator'`, automatically switches to `sigma8type='CAMB'` to avoid GP emulation failure.
- **`nltype='linear'` removed.** `get_pknl` accepts only `'halofit'` and `'hmcode2020'` and raises `ValueError` otherwise. `Bkcb_gp` and `CEmulator/data/Bk_lin.npz` are deleted, which also stops loading 6.1 MB and building 20 extra GPRs on construction. The k-range check now uses the equivalent `Bkcb_hmcode2020.klist`.
- **A descending z is rejected** by `check_z` with a message naming the ordering, instead of surfacing later as scipy's "x must be strictly increasing when grid is True". `get_mass_from_lgden` does the same for a descending lgden.
- **`get_sigma_z` / `get_sigma_cb_z`** evaluate the sinc-window integral over the (nR, nk) mesh in one `trapz` instead of one call per radius (bit identical).
- **`_u_m_term`** (FFTLog) is memoized per transform grid; a 1024-point `P2xi` went from 0.88 ms to 0.16 ms.
- **Zero-number-density warnings** go through a verbose-gated helper that reports once, and the `SatelliteProfile` k-range warning uses a constant message instead of a per-mass f-string.

### Internal

- Added instance variables `_fallback_mode`, `_camb_results_cache`, `_cosmo_class_cache` to `CBaseEmulator.__init__`.
- Cache clearing logic added to `CBaseEmulator._sync_cosmologies()`, which propagates to all subclasses (`Pkmm_CEmulator`, `HMF_CEmulator`, etc.).
- Verification tests added at `test/test_camb_fallback.py` covering 10 scenarios: normal/fallback modes, cache invalidation, `R` bug fix, `Pcb=False`, `fallback_to_camb=False`, and sigma8 conversion.
- Verification tests added at `test/test_cache_bounded.py` covering fallback-warning deduplication and the bounded HOD-weight LRU cache.
- Benchmark script `test/bench_mcmc_pknl_fallback.py`: measures per-step time and RSS of 200 `set_cosmos` + `get_pknl` iterations for in-bounds emulator chains and out-of-bounds CAMB-fallback chains (`--niter`, `--seed`, `--csv`).
- Instance state added: `_camb_z_cache` (per-z single-redshift CAMB results), `_pcalc_cache`, `_fd_warned`, `_zero_den_warned`, `_lgden_warned`, `_extrap_warned`, `_z_slabs` (lazy grids), and the module-level `_UM_CACHE`.
- `_2d_interp` builds its `RectBivariateSpline` once per (massdef, diff) instead of per call, and the five-way mass-definition dispatch collapsed into `_HMF_ATTRS` / `_get_hmf_gp`.
- `GalaxyStats` gained `_node_pair` / `_ensure_z` / `_ensure_for` / `_stacked` and `_compute_phh_slab`; `_precompute_xi_hm_pkhm_grids` / `_precompute_xihh_phh_grids` are gone.
- `test/test_camb_fallback.py` grew four scenarios (11-14).

