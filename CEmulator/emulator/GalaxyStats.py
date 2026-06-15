"""
Galaxy Statistics Calculator for CSST Emulator.

This module provides efficient calculation of galaxy clustering statistics
from Halo Occupation Distribution (HOD) models combined with halo statistics
from the CSST emulator. It supports galaxy-matter correlation, galaxy-galaxy
correlation, and related observables.

The calculations use efficient integration methods with caching for HOD
weights to optimize performance when computing statistics for multiple
redshifts or scales.

Classes
-------
GalaxyStatsCalculator
    Main class for computing galaxy statistics from HOD and halo emulators

Functions
---------
compute_projected_correlation
    Compute projected correlation function w_p(r_p) from xi(r)

References
----------
.. [1] Zheng, Z. et al. 2005, ApJ, 633, 791
.. [2] Zehavi, I. et al. 2005, ApJ, 630, 1

Examples
--------
>>> from CEmulator.emulator.GalaxyStats import GalaxyStatsCalculator
>>> from CEmulator.emulator.HOD import Zheng05HOD
>>> calc = GalaxyStatsCalculator(hmf_emu, xihm_emu, xihh_emu)
>>> hod = Zheng05HOD()
>>> hod_params = {'logMmin': 12.0, 'sigma_logM': 0.5,
...               'logM0': 12.5, 'logM1': 13.5, 'alpha': 1.0}
>>> ngal = calc.compute_ngal(z=0.5, hod_model=hod, **hod_params)
>>> Xigm = calc.compute_Xigm(r, z, hod_model=hod, **hod_params)
"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, interp1d
from ..hankl.galaxy import fftbase, xi2P, pk2xi, pk2wp, pk2dwp
from .SatelliteProfile import SatelliteProfile, concentration_duffy08
from ..cosmology import RHO_CRIT_0

class GalaxyStatsCalculator:
    """
    Calculator for galaxy clustering statistics from HOD models.

    This class integrates HOD models with halo statistics (HMF, Xihm, Xihh)
    to compute galaxy clustering observables. It provides caching mechanisms
    to optimize repeated calculations with the same HOD parameters.

    Parameters
    ----------
    hmf_emu : object
        Halo mass function emulator (e.g., HMFRockstarM200m_gp or HMF_CEmulator)
        Must provide: get_dndlnM(z, M), get_Nhalo(z, M)
    xihm_emu : object, optional
        Halo-matter correlation emulator (e.g., BrhmRockstarM200m_gp or Xihm_CEmulator)
        Must provide: get_xihm_mass(z, r, M, massdef)
    xihh_emu : object, optional
        Halo-halo correlation emulator (e.g., Xihh_gp or Xihh_CEmulator)
        Must provide: get_xihh_mass(z, r, M1, M2, massdef) or Xihh attribute with get_xihh()
    mass_def : str, default='RockstarM200m'
        Halo mass definition to use
    verbose : bool, default=False
        Whether to print diagnostic information

    Attributes
    ----------
    hmf : object
        Halo mass function emulator instance
    xihm : object
        Halo-matter correlation emulator instance
    xihh : object
        Halo-halo correlation emulator instance
    mass_def : str
        Current mass definition
    M_grid : ndarray
        Fixed mass grid for integrations (log-spaced from 1e11 to 1e15 Msun/h)
    _cache : dict
        Internal cache for HOD weights and derived quantities

    Notes
    -----
    The calculator uses a fixed logarithmic mass grid for integrations.
    HOD weights are cached based on the HOD model class and parameters,
    so repeated calculations with the same parameters are fast.

    The mass-based APIs internally handle finite differencing of cumulative
    emulator predictions; dndlnM=0 bins naturally contribute 0 to integrals.

    Examples
    --------
    >>> calc = GalaxyStatsCalculator(hmf_emu, xihm_emu, xihh_emu)
    >>> hod = Zheng05HOD()
    >>> params = {'logMmin': 12.0, 'sigma_logM': 0.5,
    ...           'logM0': 12.5, 'logM1': 13.5, 'alpha': 1.0}
    >>> ngal = calc.compute_ngal(z=0.5, hod_model=hod, **params)
    >>> r = np.logspace(-1, 1, 50)
    >>> Xigm = calc.compute_Xigm(r, z=0.5, hod_model=hod, **params)
    """

    # Fixed lgden nodes for xi_hh GP+tree precomputation
    _LGDEN_NODES = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5])

    def __init__(self, hmf_emu, xihm_emu=None, xihh_emu=None,
                 mass_def='RockstarM200m', Omegam=0.315,
                 profile_type="xihm", concentration_model=concentration_duffy08,
                 zlists=None, verbose=False):
        """Initialize the galaxy statistics calculator."""
        self.hmf = hmf_emu
        self.xihm = xihm_emu
        self.xihh = xihh_emu
        self.mass_def = mass_def
        self.Omegam = Omegam
        self.verbose = verbose

        # Fixed mass grid for integrations (user specification)
        self.M_grid = np.logspace(12.5, 15.5, 16)
        self.lnM_grid = np.log(self.M_grid)

        # Fourier-space infrastructure
        # r range: [0.001, 316] Mpc/h covers xi_hh -> 0 at large scales
        self.fftbase = fftbase(fft_num=9, fft_logrmin=-5.0, fft_logrmax=3.0)
        self.satellite_profile = SatelliteProfile(
            profile_type=profile_type,
            concentration_model=concentration_model,
            Omegam=self.Omegam
        )

        # Cache for HOD weights
        self._cache = {}
        self._current_hod_key = None

        self.rswitch = 40.0 # xi_tree + GP blending scale for xi
        
        # Cache for pre-computed xi_hm on M_arr (per z, r)
        self._xihm_mass_cache = None  # tuple: (z, r_checksum, xi_hm_on_M_arr)

        # Cache for P_hh matrix on mass nodes (per redshift)
        self._phh_matrix_cache = None  # tuple: (z, P_hh_grid)

        # Cache for P_hm on M_arr (per redshift)
        self._pkhm_mass_cache = None  # tuple: (z, k_checksum, P_hm_on_M_arr)

        # Cache for mass grid, ngal and uk_grid (per redshift)
        self._mass_grid_cache = None  # tuple: (z, M_arr, lnM, dndlnM)
        self._ngal_cache = None  # tuple: (z, hod_key, ngal)
        self._uk_grid_cache = None  # tuple: (z, uk_grid)

        # Cache for Fourier-space Pgg and Pgm (per (z, hod_key))
        self._pgg_fourier_cache = None  # tuple: (z, hod_key, Pgg)
        self._pgm_fourier_cache = None  # tuple: (z, hod_key, Pgm)

        # Cache for blended xi_hh grid on mass nodes, keyed by (z, r)
        self._xihh_blend_cache = None  # tuple: (z, r_checksum, grid)

        # Cache for tree-level xi_mm (matter correlation) per (z, r)
        self._xi_mm_cache = None  # tuple: (z, r_checksum, xi_mm)

        # Cache for bias on _LGDEN_NODES (per redshift)
        self._bias_mass_node_cache = None  # tuple: (z, bias_grid)
        self._mass_nodes_cache = None  # tuple: (z, mass_nodes)

        # Precomputed full grids (zlists x M_arr x fftbase.r/k) — Xihm
        self._xi_hm_full = None  # (n_zPre, n_mass, n_rPre)
        self._pkhm_full = None   # (n_zPre, n_mass, n_kPre)
        # Precomputed full grids (zlists x mass_nodes x fftbase.r/k) — Xihh
        self._xi_hh_full = None  # (n_zPre, G, G, n_rPre)  blended xi_hh on mass nodes
        self._phh_full = None    # (n_zPre, G, G, n_kPre)  P_hh on mass nodes
        # Interpolators built from precomputed grids
        self._xi_hm_interpolator = None
        self._pkhm_interpolator = None
        self._xi_hh_interpolator = None
        self._phh_interpolator = None
        self._z_pre = None       # z grid for interpolation
        if zlists is not None:
            self._z_pre = np.sort(np.asarray(zlists, dtype=float))
            if self.xihm is not None:
                self._precompute_xi_hm_pkhm_grids()
            if self.xihh is not None:
                self._precompute_xihh_phh_grids()

    def clear_cosmology_cache(self):
        """
        Clear cosmology-dependent caches (Xihm/Xihh grids, P_hh matrix).

        Called by the parent emulator when cosmology changes, to prevent
        stale grid values from being returned.
        """
        self._xihm_mass_cache = None
        self._phh_matrix_cache = None
        self._pkhm_mass_cache = None
        self._mass_grid_cache = None
        self._ngal_cache = None
        self._uk_grid_cache = None
        self._pgg_fourier_cache = None
        self._pgm_fourier_cache = None
        self._xihh_blend_cache = None
        self._xi_mm_cache = None
        self._bias_mass_node_cache = None
        self._mass_nodes_cache = None
        # Clear interpolators; will be rebuilt by precompute methods
        self._xi_hm_interpolator = None
        self._pkhm_interpolator = None
        self._xi_hh_interpolator = None
        self._phh_interpolator = None
        # Recompute full grids with new cosmology
        if self._z_pre is not None:
            if self.xihm is not None:
                self._precompute_xi_hm_pkhm_grids()
            if self.xihh is not None:
                self._precompute_xihh_phh_grids()

    def _cache_matches(self, cached, target):
        """Check if cached value matches target (both atleast_1d)."""
        cached = np.atleast_1d(cached)
        target = np.atleast_1d(target)
        if len(cached) != len(target):
            return False
        return np.allclose(cached, target)

    def _get_xi_mm(self, z, r):
        """
        Tree-level matter correlation xi_mm(r) = P2xi[P_lin](r).

        Cached per (z, r) to avoid redundant FFTLog transforms.
        Uses the separable decomposition: xi_tree(l1,l2) = b(l1)*b(l2)*xi_mm,
        so this single transform serves all G×G mass-node pairs.
        """
        z_key = np.atleast_1d(z)
        r_key = np.atleast_1d(r)
        if self._xi_mm_cache is not None:
            cached_z, cached_r, cached_val = self._xi_mm_cache
            if self._cache_matches(cached_z, z_key) and self._cache_matches(cached_r, r_key):
                return cached_val
        xi_mm_interp = self.hmf.get_ximmlinear(z=z, r=r, Pcb=True) # nz, nr
        xi_mm_interp = np.squeeze(xi_mm_interp)
        self._xi_mm_cache = (z_key.copy(), r_key.copy(), xi_mm_interp)
        return xi_mm_interp

    def _get_mass_nodes(self, z):
        """Convert _LGDEN_NODES to mass nodes for each z, cached.

        Returns (n_z, G) array, each row sorted ascending.
        """
        z_key = np.atleast_1d(z)
        if self._mass_nodes_cache is not None:
            cached_z, cached_val = self._mass_nodes_cache
            if len(cached_z) == len(z_key) and np.allclose(cached_z, z_key):
                return cached_val
        mass_nodes = np.asarray(
            self.hmf.get_mass_from_lgden(z=z_key, lgden=self._LGDEN_NODES,
                                         massdef=self.mass_def)
        )  # (n_z, G), M decreases with lgden
        mass_nodes = np.sort(mass_nodes, axis=1)  # ascending per row
        self._mass_nodes_cache = (z_key.copy(), mass_nodes)
        return mass_nodes  # (n_z, G)

    def _get_cached_bias_nodes(self, z):
        """Mass-bin bias b(z, M) on mass nodes from _LGDEN_NODES, cached.

        Returns (n_z, G).
        """
        z_key = np.atleast_1d(z)
        if self._bias_mass_node_cache is not None:
            cached_z, cached_val = self._bias_mass_node_cache
            if len(cached_z) == len(z_key) and np.allclose(cached_z, z_key):
                return cached_val
        mass_nodes = self._get_mass_nodes(z_key)  # (n_z, G)
        n_z, G = mass_nodes.shape
        biases = np.zeros((n_z, G))
        for iz in range(n_z):
            for ig in range(G):
                biases[iz, ig] = np.float64(self.xihh.get_bias_mass(
                    z=z_key[iz], M=mass_nodes[iz, ig], massdef=self.mass_def)[0, 0])
        self._bias_mass_node_cache = (z_key.copy(), biases)
        return biases  # (n_z, G)

    def _get_hod_cache_key(self, hod_model, **hod_params):
        """
        Generate a unique cache key for HOD model and parameters.

        Parameters
        ----------
        hod_model : HODModel
            HOD model instance
        **hod_params : dict
            HOD parameters

        Returns
        -------
        tuple
            Hashable cache key
        """
        return (hod_model.__class__.__name__,
                frozenset(hod_params.items()))

    def _get_cached_hod_weights(self, hod_model, **hod_params):
        """
        Get cached HOD weights or compute and cache them.

        Parameters
        ----------
        hod_model : HODModel
            HOD model instance
        **hod_params : dict
            HOD parameters

        Returns
        -------
        tuple
            (Ncen, Nsat, Ntot) arrays on the mass grid
        """
        cache_key = self._get_hod_cache_key(hod_model, **hod_params)

        if cache_key not in self._cache:
            if self.verbose:
                print(f"Computing HOD weights for {cache_key[0]}")

            Ncen = hod_model.Ncen(self.M_grid, **hod_params)
            Nsat = hod_model.Nsat(self.M_grid, **hod_params)
            Ntot = Ncen + Nsat

            self._cache[cache_key] = {
                'Ncen': Ncen,
                'Nsat': Nsat,
                'Ntot': Ntot,
                'M_grid': self.M_grid.copy()
            }

        self._current_hod_key = cache_key
        c = self._cache[cache_key]
        return c['Ncen'], c['Nsat'], c['Ntot']

    def clear_cache(self):
        """Clear the HOD weights cache."""
        self._cache.clear()
        self._current_hod_key = None
        self._pgg_fourier_cache = None
        self._pgm_fourier_cache = None
        if self.verbose:
            print("HOD cache cleared")

    # ------------------------------------------------------------------
    # Basic galaxy properties
    # ------------------------------------------------------------------

    def compute_ngal(self, z, hod_model, **hod_params):
        r"""
        Compute the galaxy number density n_gal.

        The galaxy number density is computed by integrating the HOD-weighted
        halo mass function:

        .. math::
            n_{\rm gal} = \int dM \frac{dn}{dM} \langle N | M \rangle

        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        hod_model : HODModel
            HOD model instance
        **hod_params : dict
            HOD model parameters

        Returns
        -------
        float or ndarray
            Galaxy number density in (Mpc/h)^(-3)

        Notes
        -----
        Uses Simpson's integration on a logarithmic mass grid for better
        accuracy than simple trapezoidal rule.
        """
        Ncen, Nsat, Ntot = self._get_cached_hod_weights(hod_model, **hod_params)

        z = np.atleast_1d(z)
        ngal = np.zeros(len(z))

        for i, zi in enumerate(z):
            # Get differential mass function on the fixed grid
            dndlnM = np.atleast_1d(np.squeeze(
                self.hmf.get_dndlnM(z=zi, M=self.M_grid)))
            ngal[i] = simpson(dndlnM * Ntot, x=self.lnM_grid)

        return ngal[0] if len(ngal) == 1 else ngal

    def compute_f_sat(self, z, hod_model, **hod_params):
        r"""
        Compute the satellite fraction f_sat = N_sat / N_total.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        hod_model : HODModel
            HOD model instance
        **hod_params : dict
            HOD model parameters

        Returns
        -------
        float or ndarray
            Satellite fraction (0 to 1)
        """
        Ncen, Nsat, Ntot = self._get_cached_hod_weights(hod_model, **hod_params)

        z = np.atleast_1d(z)
        f_sat = np.zeros(len(z))

        for i, zi in enumerate(z):
            dndlnM = np.atleast_1d(np.squeeze(
                self.hmf.get_dndlnM(z=zi, M=self.M_grid)))
            nsat = simpson(dndlnM * Nsat, x=self.lnM_grid)
            ntot = simpson(dndlnM * Ntot, x=self.lnM_grid)
            f_sat[i] = nsat / ntot if ntot > 0 else 0.0

        return f_sat[0] if len(f_sat) == 1 else f_sat

    def compute_M_eff(self, z, hod_model, **hod_params):
        r"""
        Compute the effective halo mass for galaxy hosting.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        hod_model : HODModel
            HOD model instance
        **hod_params : dict
            HOD model parameters

        Returns
        -------
        float or ndarray
            Effective mass in Msun/h
        """
        Ncen, Nsat, Ntot = self._get_cached_hod_weights(hod_model, **hod_params)

        z = np.atleast_1d(z)
        M_eff = np.zeros(len(z))

        for i, zi in enumerate(z):
            dndlnM = np.atleast_1d(np.squeeze(
                self.hmf.get_dndlnM(z=zi, M=self.M_grid)))
            numer = simpson(dndlnM * Ntot * self.M_grid, x=self.lnM_grid)
            denom = simpson(dndlnM * Ntot, x=self.lnM_grid)
            M_eff[i] = numer / denom if denom > 0 else 0.0

        return M_eff[0] if len(M_eff) == 1 else M_eff

    def compute_bias(self, z, hod_model, **hod_params):
        r"""
        Compute the large-scale galaxy bias.

        The galaxy bias is computed as the weighted average of the halo bias:

        .. math::
            b_g = \frac{1}{n_{\rm gal}} \int dM \frac{dn}{dM}
            \langle N | M \rangle b_h(M)

        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        hod_model : HODModel
            HOD model instance
        **hod_params : dict
            HOD model parameters

        Returns
        -------
        float or ndarray
            Galaxy bias b_g
        """
        Ncen, Nsat, Ntot = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self.compute_ngal(z, hod_model, **hod_params)
        ngal_arr = np.atleast_1d(ngal)

        z = np.atleast_1d(z)
        bias_g = np.zeros(len(z))

        for i, zi in enumerate(z):
            dndlnM = np.atleast_1d(np.squeeze(
                self.hmf.get_dndlnM(z=zi, M=self.M_grid)))

            # Use the HMF's bias model
            bias_h = np.atleast_1d(np.squeeze(self.hmf.get_bias_mass(z=zi, M=self.M_grid, massdef=self.mass_def)))
            integrand = dndlnM * Ntot * bias_h
            bias_g[i] = simpson(integrand, x=self.lnM_grid) / ngal_arr[i]

        return bias_g[0] if len(bias_g) == 1 else bias_g

    # ------------------------------------------------------------------
    # Galaxy-matter correlation: mass-bin based approach
    # ------------------------------------------------------------------

    def _compute_xi_hm_on_grid(self, z, r, M_arr):
        """Internal: compute xi_hm(r|M) on a fixed grid via finite differencing.

        No per-call caching. Returns (n_z, n_mass, n_r).
        """
        z_key = np.atleast_1d(z)
        r_key = np.atleast_1d(r)
        mass_nodes = self._get_mass_nodes(z)  # (n_z, G)
        biases_interp = self._get_cached_bias_nodes(z)  # (n_z, G)
        n_z, G = mass_nodes.shape
        n_r = len(r_key)
        xi_mm_all = self._get_xi_mm(z_key, r_key)
        result = np.zeros((n_z, len(M_arr), n_r))
        for iz in range(n_z):
            biases_iz = interp1d(np.log(mass_nodes[iz]), biases_interp[iz], kind='linear',
                                fill_value='extrapolate')(np.log(M_arr))
            xi_mm = xi_mm_all[iz]  # (n_r,)
            xi_tree = biases_iz[:,None] * xi_mm[None,:]
            
            Mp = mass_nodes[iz] * 1.02
            Mm = mass_nodes[iz] * 0.98
            den_p = self.hmf.get_Nhalo(z=z_key[iz], M=Mp, massdef=self.mass_def)[0,::-1]
            den_m = self.hmf.get_Nhalo(z=z_key[iz], M=Mm, massdef=self.mass_def)[0,::-1]
            valid = (den_p > 0) & (den_m > 0)
            lgden_p = np.log10((den_p[valid]))
            lgden_m = np.log10((den_m[valid]))
            den_p = 10.0 ** lgden_p
            den_m = 10.0 ** lgden_m
            try:
                xi_p = np.squeeze(self.xihm.BrhmRockstarM200m.get_Brhm(z=z_key[iz], r=r_key, lgden=lgden_p))[valid,:]
                xi_m = np.squeeze(self.xihm.BrhmRockstarM200m.get_Brhm(z=z_key[iz], r=r_key, lgden=lgden_m))[valid,:]
                xi_p = xi_p * xi_mm[None, :]
                xi_m = xi_m * xi_mm[None, :]
                dden = den_p - den_m
                xi_gp = (xi_p * den_p[:, None] - xi_m * den_m[:, None]) / dden[:, None]
                xi_gp = xi_gp[::-1,:]  # reverse back to match mass_nodes order
                # (G, n_r)
                interp = interp1d(np.log(mass_nodes[iz]), xi_gp, kind='linear', axis=0,
                                  bounds_error=False, fill_value='extrapolate')
                xi_gp = interp(np.log(M_arr))
                w = np.exp(-(r_key / self.rswitch) ** 4)[None,:]
                result[iz] = xi_gp * w + xi_tree * (1.0 - w)
                
            except Exception as e:
                print(f"Error computing xi_hm for z={z_key[iz]}: {e}")
            
        return result

    def _compute_pkhm_on_grid(self, z, k, M_arr):
        """Internal: compute P_hm(k|M) on a fixed grid via finite differencing.

        No per-call caching. Returns (n_z, n_mass, n_k).
        """
        z_key = np.atleast_1d(z)
        k_key = np.atleast_1d(k)
        n_z = len(z_key)
        n_k = len(k_key)
        result = np.zeros((n_z, len(M_arr), n_k))
        for iz in range(n_z):
            try:
                if self._xi_hm_full is not None:
                    # FFTLog path: more accurate on small scales
                    r = self.fftbase.r
                    G = len(M_arr)
                    for i in range(G):
                        xi_1d = self._xi_hm_full[iz, i, :]  # nr
                        k_out, P_hm_gp = xi2P(r, xi_1d, l=0, ext=0)
                        interp_pk = interp1d(np.log(k_out), np.real(P_hm_gp),
                                             kind='linear', bounds_error=False,
                                             fill_value='extrapolate')
                        result[iz, i, :] = interp_pk(np.log(k_key))
                else:
                    # GP finite-differencing fallback
                    mass_nodes = self._get_mass_nodes(z_key)
                    Mp = mass_nodes[iz] * 1.01
                    Mm = mass_nodes[iz] * 0.99
                    den_p = np.atleast_1d(np.squeeze(
                        self.hmf.get_Nhalo(z=z_key[iz], M=Mp, massdef=self.mass_def)))[::-1]
                    den_m = np.atleast_1d(np.squeeze(
                        self.hmf.get_Nhalo(z=z_key[iz], M=Mm, massdef=self.mass_def)))[::-1]
                    valid = (den_p > 0) & (den_m > 0)
                    lgden_p = np.log10((den_p))[valid]
                    lgden_m = np.log10((den_m))[valid]
                    den_p = 10.0 ** lgden_p
                    den_m = 10.0 ** lgden_m
                    pk_p = np.squeeze(self.xihm.get_pkhm_lgnbar_threshold(
                        z=z_key[iz], k=k_key, lgden=lgden_p, massdef=self.mass_def))[valid,:]
                    pk_m = np.squeeze(self.xihm.get_pkhm_lgnbar_threshold(
                        z=z_key[iz], k=k_key, lgden=lgden_m, massdef=self.mass_def))[valid,:]
                    dden = den_p - den_m
                    pk_nodes = (pk_p * den_p[:, None] - pk_m * den_m[:, None]) / dden[:, None]
                    # G, n_k
                    pk_nodes = pk_nodes[::-1]  # reverse back to match mass_nodes order
                    interp = interp1d(np.log(mass_nodes[iz]), pk_nodes, kind='linear', axis=0,
                                    bounds_error=False, fill_value='extrapolate')
                    result[iz] = interp(np.log(M_arr))
            except Exception as e:
                print(f"Error computing P_hm for z={z_key[iz]}: {e}")
            
        return result

    def _compute_xihh_on_grid(self, z, r):
        """
        Internal: blended xi_hh on mass nodes via GP+tree. No caching.

        Returns (n_z, G, G, n_r).
        """
        ### directly compute xihh on M_grid
        M_arr = self.M_grid
        z_key = np.atleast_1d(z)
        r_key = np.atleast_1d(r)
        mass_nodes = self._get_mass_nodes(z_key)
        biases_interp = self._get_cached_bias_nodes(z_key)  # (n_z, G)
        n_z, G = mass_nodes.shape
        n_r = len(r_key)
        result = np.zeros((n_z, len(M_arr), len(M_arr), n_r))
        for iz in range(n_z):
            biases_iz = interp1d(np.log(mass_nodes[iz]), biases_interp[iz], kind='linear',
                                fill_value='extrapolate')(np.log(M_arr))
            xi_mm = self._get_xi_mm(z_key[iz], r_key)  # (n_r,)
            xi_tree = (biases_iz[:, None, None] * biases_iz[None, :, None]
                       * xi_mm[None, None, :])
            # xi_gp = self.xihh.get_xihh_mass(z=z_key[iz], r=r, M1=mass_nodes[iz], M2=mass_nodes[iz], massdef=self.mass_def)[:,:,0]
            # interp = Fast2DInterpolator(F=xi_gp, x=np.log10(mass_nodes[iz]), y=np.log10(mass_nodes[iz]),
            #                             extrapolate='linear')
            # GP: 2D finite differencing
            Mp = mass_nodes[iz] * 1.02
            Mm = mass_nodes[iz] * 0.98
            den_p = self.hmf.get_Nhalo(z=z_key[iz], M=Mp, massdef=self.mass_def)[0,::-1]
            den_m = self.hmf.get_Nhalo(z=z_key[iz], M=Mm, massdef=self.mass_def)[0,::-1] # inverse order to ensure increasing order
            valid = (den_p > 0) & (den_m > 0) 
            lgden_p = np.log10((den_p[valid]))
            lgden_m = np.log10((den_m[valid]))
            den_p = 10.0 ** lgden_p
            den_m = 10.0 ** lgden_m
            xi_gp = np.zeros((G, G, n_r))
            if hasattr(self.xihh, 'Xihh') and hasattr(self.xihh.Xihh, 'get_xihh'):
                xipp = self.xihh.Xihh.get_xihh(
                    z=z_key[iz], r=r_key, lgden1=lgden_p[valid], lgden2=lgden_p[valid])[:, :, 0]
                ximm = self.xihh.Xihh.get_xihh(
                    z=z_key[iz], r=r_key, lgden1=lgden_m[valid], lgden2=lgden_m[valid])[:, :, 0]
                xipm = self.xihh.Xihh.get_xihh(
                    z=z_key[iz], r=r_key, lgden1=lgden_p[valid], lgden2=lgden_m[valid])[:, :, 0]
                ximp = self.xihh.Xihh.get_xihh(
                    z=z_key[iz], r=r_key, lgden1=lgden_m[valid], lgden2=lgden_p[valid])[:, :, 0]
                dpp = den_p[valid, None] * den_p[None, valid]
                dmm = den_m[valid, None] * den_m[None, valid]
                dpm = den_p[valid, None] * den_m[None, valid]
                dmp = den_m[valid, None] * den_p[None, valid]
                numer = (xipp * dpp[:, :, None] + ximm * dmm[:, :, None]
                      - xipm * dpm[:, :, None] - ximp * dmp[:, :, None])
                denom = dpp + dmm - dpm - dmp
                xi_gp = numer / denom[:, :, None]
                xi_gp = xi_gp[::-1, ::-1, :]  # reverse to match mass_nodes order
            else:
                xi_gp = np.asarray(self.xihh.get_xihh_mass(
                    z=z_key[iz], r=r_key, M1=M_arr, M2=M_arr,
                    massdef=self.mass_def), dtype=float
                )[:, :, 0, :]
           
            ### we should be careful when change the interpolation 
            interp = RegularGridInterpolator((np.log(mass_nodes[iz][valid]), np.log(mass_nodes[iz][valid]), np.log(r_key)),
                                             xi_gp, bounds_error=False, fill_value=None, method='linear')
            L1, L2, L3 = np.meshgrid(np.log(M_arr), np.log(M_arr), np.log(r_key), indexing='ij')
            points = np.stack([L1.flatten(), L2.flatten(), L3.flatten()], axis=-1)
            xi_gp = interp(points).reshape(len(M_arr), len(M_arr), len(r_key))
            w = np.exp(-(r_key / self.rswitch) ** 4)[None,None,:]
            result[iz] = xi_gp * w + xi_tree * (1.0 - w)
        return result  # (n_z, len(M), len(M), n_r)

    def _compute_phh_on_grid(self, z):
        """Transform precomputed xi_hh_full to P_hh via FFTLog.

        Uses self._xi_hh_full on self.fftbase.r.
        Returns (n_zPre, G, G, n_kPre).
        """
        z_key = np.atleast_1d(z)
        M_arr = self.M_grid
        r = self.fftbase.r
        k_target = self.fftbase.k
        n_k = len(k_target)
        n_z = len(z_key)
        G   = len(M_arr)
        phh = np.zeros((n_z, G, G, n_k))
        for iz in range(n_z):
            for i in range(G):
                for j in range(i, G):
                    try:
                        xi_1d = self._xi_hh_full[iz, i, j, :]
                        k_out, P_hh_gp = xi2P(r, xi_1d, l=0, ext=0, lowring=True)
                        interp_pk = interp1d(np.log(k_out), np.real(P_hh_gp),
                                             kind='linear', bounds_error=False,
                                             fill_value='extrapolate')
                        phh[iz, i, j, :] = interp_pk(np.log(k_target))
                    except Exception:
                        k_lin = k_target[k_target <= 50]
                        Plin_lin = np.squeeze(self.xihh.get_pklin(
                            z=self._z_pre[iz], k=k_lin, Pcb=True))
                        Plin_interp = interp1d(
                            np.log10(k_lin), np.log10(Plin_lin),
                            kind='linear', bounds_error=False,
                            fill_value="extrapolate")
                        phh[iz, i, j, :] = np.zeros_like(k_target) 
                    if j > i:
                        phh[iz, j, i, :] = phh[iz, i, j, :]
        return phh
    
    ########## precomputation  ##########
    def _precompute_xi_hm_pkhm_grids(self):
        """Precompute xi_hm and P_hm on all zlists x M_arr x fftbase.r/k."""
        if self.verbose:
            print(f"Precomputing xi_hm and P_hm on {len(self._z_pre)} redshifts ...")
        self._xi_hm_full = self._compute_xi_hm_on_grid(self._z_pre, self.fftbase.r, self.M_grid)
        self._xihm_mass_cache = (self._z_pre, self.fftbase.r, self.M_grid, self._xi_hm_full)
        self._pkhm_full = self._compute_pkhm_on_grid(self._z_pre, self.fftbase.k, self.M_grid)
        self._pkhm_mass_cache = (self._z_pre, self.fftbase.k, self.M_grid, self._pkhm_full)
        if self.verbose:
            print("  Done.")

    def _precompute_xihh_phh_grids(self):
        """Precompute xi_hh and P_hh on all zlists x fftbase.r/k x mass_nodes."""
        if self.verbose:
            print(f"Precomputing xi_hh and P_hh on {len(self._z_pre)} redshifts ...")
        self._xi_hh_full = self._compute_xihh_on_grid(self._z_pre, self.fftbase.r)
        self._xihh_blend_cache = (self._z_pre, self.fftbase.r, self.M_grid, self.M_grid, self._xi_hh_full)
        self._phh_full = self._compute_phh_on_grid(self._z_pre)
        self._phh_matrix_cache = (self._z_pre, self.fftbase.k, self.M_grid, self.M_grid, self._phh_full)
        if self.verbose:
            print("  Done.")
    
    ####### Get r M z dependent quantities with caching and interpolation ########
    def _get_xihh_mass_blend_grid(self, z, r, M1, M2):
        """
        Blended mass-bin xi_hh(r) for all GxG mass-node pairs, multi-z.

        Three-tier: per-call cache → precomputed grid (z+r interp) → on-demand.

        Returns (n_z, G, G, n_r).
        """
        z_key  = np.atleast_1d(z)
        if len(z_key) > 1:
            raise ValueError("Multi-z input not supported for xi_hh retrieval.")
        r_key  = np.atleast_1d(r)
        M1_key = np.atleast_1d(M1)
        M2_key = np.atleast_1d(M2)
        # Layer 1: per-call cache
        if self._xihh_blend_cache is not None:
            cached_z, cached_r, cached_M1, cached_M2, cached_val = self._xihh_blend_cache
            if self._cache_matches(cached_z, z_key) and self._cache_matches(cached_r, r_key) and\
               self._cache_matches(cached_M1, M1_key) and self._cache_matches(cached_M2, M2_key):
                return cached_val

        if self._xi_hh_interpolator is None:
            self._xi_hh_interpolator = RegularGridInterpolator(
                ((self._z_pre), np.log10(self.M_grid), np.log10(self.M_grid), np.log(self.fftbase.r)),
                self._xi_hh_full, bounds_error=False, fill_value=None)
            uz = (z_key)
            logr = np.log(r_key)
            mass_idx1 = np.log10(M1_key)
            mass_idx2 = np.log10(M2_key)
            uz_g, mi_g, mj_g, logr_g = np.meshgrid(
                uz, mass_idx1, mass_idx2, logr, indexing='ij')
            points = np.column_stack(
                [uz_g.ravel(), mi_g.ravel(), mj_g.ravel(), logr_g.ravel()])
            result = self._xi_hh_interpolator(points).reshape(
                len(z_key), len(M1_key), len(M2_key), len(r_key))
            if len(z_key) == 1:
                result = result[0]
        else:
            uz = (z_key)
            logr = np.log(r_key)
            mass_idx1 = np.log10(M1_key)
            mass_idx2 = np.log10(M2_key)
            uz_g, mi_g, mj_g, logr_g = np.meshgrid(
                uz, mass_idx1, mass_idx2, logr, indexing='ij')
            points = np.column_stack(
                [uz_g.ravel(), mi_g.ravel(), mj_g.ravel(), logr_g.ravel()])
            result = self._xi_hh_interpolator(points).reshape(
                len(z_key), len(M1_key), len(M2_key), len(r_key))
            if len(z_key) == 1: 
                result = result[0]

        return result

    def _get_pkhh_for_masses(self, z, k, M1, M2):
        """Cached P_hh matrix: from precomputed grid if available, else on-demand."""
        z_key  = np.atleast_1d(z)
        if len(z_key) > 1:
            raise ValueError("Multi-z input not supported for P_hh retrieval.")
        k_key  = np.atleast_1d(k)
        M1_key = np.atleast_1d(M1)
        M2_key = np.atleast_1d(M2)
        if self._phh_matrix_cache is not None:
            cached_z, cached_k, cached_M1, cached_M2, cached_data = self._phh_matrix_cache
            if self._cache_matches(cached_z, z_key) and self._cache_matches(cached_k, k_key) and \
                self._cache_matches(cached_M1, M1_key) and self._cache_matches(cached_M2, M2_key):
                return cached_data

        if self._phh_interpolator is None:
            self._phh_interpolator = RegularGridInterpolator(
                ((self._z_pre), np.log10(self.M_grid), np.log10(self.M_grid), np.log(self.fftbase.k)),
                self._phh_full, bounds_error=False, fill_value=None)
            uz = z_key
            mass_idx1 = np.log10(M1_key)
            mass_idx2 = np.log10(M2_key)
            uz_g, mi_g, mj_g, k_g = np.meshgrid(
                uz, mass_idx1, mass_idx2, np.log(k_key), indexing='ij')
            points = np.column_stack(
                [uz_g.ravel(), mi_g.ravel(), mj_g.ravel(), k_g.ravel()])
            result = self._phh_interpolator(points).reshape(
                len(z_key), len(M1_key), len(M2_key), len(k_key))
            if len(z_key) == 1:
                result = result[0]
        else:
            uz = z_key
            mass_idx1 = np.log10(M1)
            mass_idx2 = np.log10(M2)
            uz_g, mi_g, mj_g, k_g = np.meshgrid(
                uz, mass_idx1, mass_idx2, np.log(k_key), indexing='ij')
            points = np.column_stack(
                [uz_g.ravel(), mi_g.ravel(), mj_g.ravel(), k_g.ravel()])
            result = self._phh_interpolator(points).reshape(
                len(z_key), len(M1_key), len(M2_key), len(k_key))
            if len(z_key) == 1: 
                result = result[0]
        return result # (n_z, G, G, n_k) or (G, G, n_k) if single z

    def _get_xi_hm_for_masses(self, z, r, M_arr):
        """
        Mass-bin xi_hm(r|M): from precomputed grid if available, else on-demand.

        Returns (n_z, n_mass, n_r).
        """
        z_key = np.atleast_1d(z)
        r_key = np.atleast_1d(r)
        M_key = np.atleast_1d(M_arr)
        if self._xihm_mass_cache is not None:
            cached_z, cached_r, cached_M, cached_data = self._xihm_mass_cache
            if self._cache_matches(cached_z, z_key) and self._cache_matches(cached_r, r_key) and \
                self._cache_matches(cached_M, M_key):
                return cached_data

        if self._xi_hm_interpolator is None:
            # Build interpolators
            self._xi_hm_interpolator = RegularGridInterpolator(
                ((self._z_pre), np.log10(self.M_grid), np.log(self.fftbase.r)),
                self._xi_hm_full, bounds_error=False, fill_value=None)
            uz   = (z_key)
            logM = np.log10(M_key)
            logr = np.log(r_key)
            uz_g, logM_g, logr_g = np.meshgrid(uz, logM, logr, indexing='ij')
            points = np.column_stack([uz_g.ravel(), logM_g.ravel(), logr_g.ravel()])
            result = self._xi_hm_interpolator(points).reshape(
                len(z_key), len(M_key), len(r_key))
        else:
            uz   = (z_key)
            logM = np.log10(M_key)
            logr = np.log(r_key)
            uz_g, logM_g, logr_g = np.meshgrid(uz, logM, logr, indexing='ij')
            points = np.column_stack([uz_g.ravel(), logM_g.ravel(), logr_g.ravel()])
            result = self._xi_hm_interpolator(points).reshape(
                len(z_key), len(M_key), len(r_key))
        return result  # (n_z, n_mass, n_r)
     
    def _get_pkhm_for_masses(self, z, k, M_arr):
        """
        Mass-bin P_hm(k|M): from precomputed grid if available, else on-demand.

        Returns (n_z, n_mass, n_k).
        """
        z_key = np.atleast_1d(z)
        k_key = np.atleast_1d(k)
        M_key = np.atleast_1d(M_arr)
        if self._pkhm_mass_cache is not None:
            cached_z, cached_k, cached_M, cached_data = self._pkhm_mass_cache
            if self._cache_matches(cached_z, z_key) and self._cache_matches(cached_k, k_key) and \
                self._cache_matches(cached_M, M_key):
                return cached_data

        if self._pkhm_interpolator is None:
            self._pkhm_interpolator = RegularGridInterpolator(
                ((self._z_pre), np.log10(self.M_grid), np.log(self.fftbase.k)),
                self._pkhm_full, bounds_error=False, fill_value=None)
            uz   = (z_key)
            logM = np.log10(M_key)
            logk = np.log(k_key)
            uz_g, logM_g, logk_g = np.meshgrid(uz, logM, logk, indexing='ij')
            points = np.column_stack([uz_g.ravel(), logM_g.ravel(), logk_g.ravel()])
            result = self._pkhm_interpolator(points).reshape(
                len(z_key), len(M_key), len(k_key))
            if len(z_key) == 1:
                result = result[0]
        else:
            uz   = (z_key)
            logM = np.log10(M_key)
            logk = np.log(k_key)
            uz_g, logM_g, logk_g = np.meshgrid(uz, logM, logk, indexing='ij')
            points = np.column_stack([uz_g.ravel(), logM_g.ravel(), logk_g.ravel()])
            result = self._pkhm_interpolator(points).reshape(
                len(z_key), len(M_key), len(k_key))
            if len(z_key) == 1:
                result = result[0]

        return result  # (n_z, n_mass, n_k)

    @staticmethod
    def _cache_matches(cached, new):
        """Check if cached array matches new array for cache invalidation."""
        if cached is None:
            return False
        try:
            return len(cached) == len(new) and np.allclose(cached, new)
        except (TypeError, ValueError):
            return False

    def compute_Xigm(self, r, z, hod_model, **hod_params):
        r"""
        Compute the galaxy-matter correlation function xi_{gm}(r).

        The galaxy-matter correlation is computed by weighting the halo-matter
        correlation by the HOD and integrating over halo mass:

        .. math::
            \xi_{gm}(r) = \frac{1}{n_{\rm gal}} \int dM \frac{dn}{dM}
            \langle N | M \rangle \xi_{hm}(r | M)

        Parameters
        ----------
        r : array_like
            Separation in Mpc/h
        z : float or array_like
            Redshift(s)
        hod_model : HODModel
            HOD model instance
        **hod_params : dict
            HOD model parameters

        Returns
        -------
        ndarray
            Galaxy-matter correlation with shape (len(z), len(r))

        Raises
        ------
        ValueError
            If xihm emulator was not provided
        """
        if self.xihm is None:
            raise ValueError("Xihm emulator is required for compute_Xigm")

        Ncen, Nsat, Ntot = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self.compute_ngal(z, hod_model, **hod_params)
        ngal_arr = np.atleast_1d(ngal)

        r = np.atleast_1d(r)
        z = np.atleast_1d(z)

        Xigm = np.zeros((len(z), len(r)))

        # Get xi_hm for all z at once from precomputed grid
        xihm_all = self._get_xi_hm_for_masses(z, r, self.M_grid)  # (n_z, n_mass, n_r)

        for i, zi in enumerate(z):
            M_arr, lnM, dndlnM_arr = self._get_mass_for_fourier(zi)
            if len(M_arr) == 0:
                if self.verbose:
                    print(f"Warning: no mass bins for Xigm at z={zi}")
                continue
            integrand = dndlnM_arr * Ntot
            Xigm[i, :] = simpson(integrand[:, np.newaxis] * xihm_all[i], x=lnM, axis=0)
            Xigm[i] /= ngal_arr[i]
        return Xigm

    # ------------------------------------------------------------------
    # Galaxy-galaxy correlation
    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    # Fourier-space HOD: cc/cs/ss decomposition
    # ------------------------------------------------------------------
    @staticmethod
    def _off_centering_kernel(k, f_off=0.0, R_off=0.0):
        r"""
        Off-centering kernel.

        .. math::

            H_{\rm off}(k) = 1 - f_{\rm off} + f_{\rm off}
            \exp\left[-\frac{1}{2}(k R_{\rm off})^2\right]

        Parameters
        ----------
        k : ndarray (n_k,)
            Wavenumbers in h/Mpc.
        f_off : float
            Fraction of off-centred centrals.
        R_off : float
            Off-centering scale in Mpc/h.

        Returns
        -------
        ndarray (n_k,)
        """
        return 1.0 - f_off + f_off * np.exp(-0.5 * (k * R_off) ** 2)

    def _get_satellite_uk_grid(self, k, M_arr, z):
        """
        Satellite profile u_s(k|M) on the full mass grid.

        Dispatches based on the profile type of *satellite_profile*:

        - ``'nfw'``: analytic NFW profile.
        - ``'xihm'``: numerical profile from xi_hm via FFTLog.

        Parameters
        ----------
        k : ndarray (n_k,)
        M_arr : ndarray (n_mass,)
        z : float

        Returns
        -------
        uk_grid : ndarray (n_z, n_k, n_mass)
        """
        z   = np.atleast_1d(z)
        n_z = len(z)
        n_k = len(k)
        n_mass = len(M_arr)
        uk_grid = np.zeros((n_z, n_k, n_mass))

        if self.satellite_profile.profile_type == 'nfw':
            for iz in range(n_z):
                for im in range(n_mass):
                    uk_grid[iz, :, im] = self.satellite_profile.u_sat_k_nfw(k, M_arr[im], z[iz])
        elif self.satellite_profile.profile_type == 'xihm':
            r_fft = self.fftbase.r
            xi_hm_grid = self._get_xi_hm_for_masses(z, r_fft, M_arr) # len(M_arr), len(z), len(r_fft)
            for iz in range(n_z):
                for im in range(n_mass):
                    R200 = self.satellite_profile.R200(M_arr[im], z[iz])
                    uk_grid[iz, :, im] = self.satellite_profile.u_sat_k_xihm(k, r_fft, xi_hm_grid[iz,im], R200, 
                                                                     self.satellite_profile.apodization_scale)
        else:
            raise ValueError(
                f"Unknown profile_type: "
                f"{self.satellite_profile.profile_type}")
        return uk_grid

    def _single_mass_integral_1h(self, dndlnM, lnM, integrand_kM):
        """
        ∫ dlnM dn/dlnM × integrand(k, M) per k-bin.

        Parameters
        ----------
        dndlnM : (n_mass,)
        lnM : (n_mass,)
        integrand_kM : (n_k, n_mass)

        Returns (n_k,).
        """
        return simpson(dndlnM[np.newaxis, :] * integrand_kM, x=lnM, axis=-1)

    def _double_mass_integral(self, dndlnM, lnM, W1, W2, P_hh):
        """
        ∬ dlnM₁ dlnM₂ dn/dlnM₁ dn/dlnM₂ W₁ W₂ P_hh.

        Vectorised: transposes P_hh to (n_k, n_mass, n_mass) and
        uses two vectorised Simpson passes.

        Parameters
        ----------
        dndlnM : (n_mass,)
        lnM : (n_mass,)
        W1, W2 : (n_k, n_mass)
        P_hh : (n_mass, n_mass, n_k)

        Returns (n_k,).
        """
        # Transpose to (n_k, n_mass, n_mass) for broadcasting
        P_hh_T = np.transpose(P_hh, (2, 0, 1))  # (n_k, n_mass, n_mass)
        gw1 = dndlnM * W1  # (n_k, n_mass)
        gw2 = dndlnM * W2  # (n_k, n_mass)

        # First integral over first mass index (i)
        # dw1[:, np.newaxis, :] * P_hh_T → (n_k, n_mass, n_mass)
        inner = simpson(gw1[:, np.newaxis, :] * P_hh_T, x=lnM, axis=-1)  # (n_k, n_mass)

        # Second integral over second mass index (j)
        return simpson(gw2 * inner, x=lnM, axis=-1)  # (n_k,)

    def _setup_mass_for_fourier(self, z):
        """
        Mass grid quantities for Fourier calculation at a single z.

        Returns (M_arr, lnM, dndlnM).
        """
        M_arr = self.M_grid
        lnM = self.lnM_grid
        dndlnM = np.atleast_1d(np.squeeze(
            self.hmf.get_dndlnM(z=z, M=M_arr)))
        return M_arr, lnM, dndlnM

    def _get_mass_for_fourier(self, z):
        """Cached wrapper for *setup_mass_for_fourier*."""
        if self._mass_grid_cache is not None:
            cached_z, *cached_data = self._mass_grid_cache
            if np.isclose(cached_z, z):
                return cached_data
        data = self._setup_mass_for_fourier(z)
        self._mass_grid_cache = (z, *data)
        return data

    def _get_cached_uk_grid(self, k, M_arr, z):
        """Cached wrapper for *get_satellite_uk_grid*, keyed by z."""
        z_key = np.atleast_1d(z)
        if len(z_key) > 1:
            raise ValueError("Multiple z not supported for _get_cached_uk_grid")
        M_key = np.atleast_1d(M_arr)
        k_key = np.atleast_1d(k)
        if self._uk_grid_cache is not None:
            cached_z, cached_k, cached_M, cached_uk = self._uk_grid_cache
            if self._cache_matches(cached_z, z) and self._cache_matches(cached_k, k_key) and\
                self.cache_matches(cached_M, M_key):
                return cached_uk
        uk = self._get_satellite_uk_grid(k_key, M_key, z_key)
        self._uk_grid_cache = (z_key.copy(), k_key.copy(), M_key.copy() ,uk)
        return uk

    def _get_cached_ngal(self, z, hod_model, **hod_params):
        """Cached wrapper for *compute_ngal*."""
        hod_key = self._get_hod_cache_key(hod_model, **hod_params)
        if self._ngal_cache is not None:
            cached_z, cached_key, cached_ngal = self._ngal_cache
            if np.isclose(cached_z, z) and cached_key == hod_key:
                return cached_ngal
        ngal = self.compute_ngal(z, hod_model, **hod_params)
        self._ngal_cache = (z, hod_key, ngal)
        return ngal

    # ------------------------------------------------------------------
    # 1-halo and 2-halo decomposition
    # ------------------------------------------------------------------

    def _get_weights(self, k, Ncen, Nsat, M_arr, z, f_off, R_off):
        """
        Compute common weight functions W_c and W_s.

        Parameters
        ----------
        k : ndarray (n_k,)
        Ncen, Nsat : ndarray (n_mass,)
        M_arr : ndarray (n_mass,)
        z : float
        f_off, R_off : float

        Returns
        -------
        W_c : (n_k, n_mass)  central weight = N_cen × H_off
        W_s : (n_k, n_mass)  satellite weight = N_sat × u_s(k|M)
        H_off : (n_k,)
        uk_grid : (n_k, n_mass)
        """
        uk_grid = self._get_cached_uk_grid(k, M_arr, z)[0]
        R200  = self.satellite_profile.R200(M_arr, z)
        H_off = self._off_centering_kernel(k[:, np.newaxis], f_off, R_off*R200[np.newaxis, :])
        W_c = Ncen[np.newaxis, :] * H_off
        W_s = Nsat[np.newaxis, :] * uk_grid
        return W_c, W_s, H_off, uk_grid

    def compute_Pcc_2h(self, k, z, hod_model, **hod_params):
        r"""
        P_cc^{2h}(k) — central-central 2-halo term.

        .. math::

            P_{cc}^{2h}(k) = \frac{1}{\bar{n}_g^2} \iint dM_1 dM_2
            \frac{dn}{dM_1}\frac{dn}{dM_2}
            W_c(k;M_1) W_c(k;M_2) P_{hh}(k;M_1,M_2)

        where :math:`W_c(k;M) = \langle N_c(M)\rangle H_{\rm off}(k;M)`.
        """
        f_off = hod_params.get('f_off', 0.0)
        R_off = hod_params.get('R_off', 0.0)
        Ncen, Nsat, _ = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self._get_cached_ngal(z, hod_model, **hod_params)
        M_arr, lnM, dndlnM = self._get_mass_for_fourier(z)
        P_hh = self._get_pkhh_for_masses(z, k, M_arr, M_arr) # (n_mass, n_mass, n_k) for single z
        W_c, _, _, _ = self._get_weights(k, Ncen, Nsat, M_arr, z, f_off, R_off)
        Pcc = self._double_mass_integral(dndlnM, lnM, W_c, W_c, P_hh)
        return Pcc / ngal ** 2

    def compute_Pcs_1h(self, k, z, hod_model, **hod_params):
        r"""
        P_cs^{1h}(k) — central-satellite 1-halo term.

        .. math::

            P_{cs}^{1h}(k) = \frac{2}{\bar{n}_g^2} \int dM \frac{dn}{dM}
            \langle N_c\rangle\langle N_s\rangle
            \tilde{u}_s(k|M) H_{\rm off}(k;M)
        """
        f_off = hod_params.get('f_off', 0.0)
        R_off = hod_params.get('R_off', 0.0)
        Ncen, Nsat, _ = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self._get_cached_ngal(z, hod_model, **hod_params)
        M_arr, lnM, dndlnM = self._get_mass_for_fourier(z)

        _, _, H_off, uk_grid = self._get_weights(
            k, Ncen, Nsat, M_arr, z, f_off, R_off)
        integrand = 1.0 * Nsat[np.newaxis, :] * uk_grid * H_off * Ncen[np.newaxis, :]
        Pcs = self._single_mass_integral_1h(dndlnM, lnM, integrand)
        return Pcs / ngal ** 2

    def compute_Pcs_2h(self, k, z, hod_model, **hod_params):
        r"""
        P_cs^{2h}(k) — central-satellite 2-halo term.

        .. math::

            P_{cs}^{2h}(k) = \frac{2}{\bar{n}_g^2} \iint dM_1 dM_2
            \frac{dn}{dM_1}\frac{dn}{dM_2}
            W_c(k;M_1) W_s(k;M_2) P_{hh}(k;M_1,M_2)
        """
        f_off = hod_params.get('f_off', 0.0)
        R_off = hod_params.get('R_off', 0.0)
        Ncen, Nsat, _ = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self._get_cached_ngal(z, hod_model, **hod_params)
        M_arr, lnM, dndlnM = self._get_mass_for_fourier(z)
        P_hh = self._get_pkhh_for_masses(z, k, M_arr, M_arr) # (n_mass, n_mass, n_k) for single z
        W_c, W_s, _, _ = self._get_weights(k, Ncen, Nsat, M_arr, z, f_off, R_off)
        Pcs = self._double_mass_integral(dndlnM, lnM, W_c, W_s, P_hh)
        return Pcs / ngal ** 2

    def compute_Pss_1h(self, k, z, hod_model, **hod_params):
        r"""
        P_ss^{1h}(k) — satellite-satellite 1-halo term.

        .. math::

            P_{ss}^{1h}(k) = \frac{1}{\bar{n}_g^2} \int dM \frac{dn}{dM}
            \langle N_c\rangle\langle N_s\rangle^2
            \tilde{u}_s(k|M)^2
        """
        Ncen, Nsat, _ = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self._get_cached_ngal(z, hod_model, **hod_params)
        M_arr, lnM, dndlnM = self._get_mass_for_fourier(z)

        # Satellite profile grid (no off-centering needed for ss 1-halo)
        uk_grid = self._get_cached_uk_grid(k, M_arr, z)[0] # limit n_z == 1

        integrand = (Nsat[np.newaxis, :]*Nsat[np.newaxis, :] * uk_grid ** 2) # / Ncen[np.newaxis, :]
        Pss = self._single_mass_integral_1h(dndlnM, lnM, integrand)
        return Pss / ngal ** 2

    def compute_Pss_2h(self, k, z, hod_model, **hod_params):
        r"""
        P_ss^{2h}(k) — satellite-satellite 2-halo term.

        .. math::

            P_{ss}^{2h}(k) = \frac{1}{\bar{n}_g^2} \iint dM_1 dM_2
            \frac{dn}{dM_1}\frac{dn}{dM_2}
            W_s(k;M_1) W_s(k;M_2) P_{hh}(k;M_1,M_2)
        """
        f_off = hod_params.get('f_off', 0.0)
        R_off = hod_params.get('R_off', 0.0)
        Ncen, Nsat, _ = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self._get_cached_ngal(z, hod_model, **hod_params)
        M_arr, lnM, dndlnM = self._get_mass_for_fourier(z)
        P_hh = self._get_pkhh_for_masses(z, k, M_arr, M_arr) # (n_mass, n_mass, n_k) for single z
        _, W_s, _, _ = self._get_weights(k, Ncen, Nsat, M_arr, z, f_off, R_off)
        Pss = self._double_mass_integral(dndlnM, lnM, W_s, W_s, P_hh)
        return Pss / ngal ** 2

    def _compute_Pgg(self, k, z, hod_model, **hod_params):
        """
        P_gg(k) = P_cc^{2h} + P_cs^{1h} + P_cs^{2h} + P_ss^{1h} + P_ss^{2h}.

        Results are cached per (z, hod_key) so that Xigg, wp and Pgg calls
        at the same redshift and HOD parameters reuse the previous result.
        """
        hod_key = self._get_hod_cache_key(hod_model, **hod_params)
        if self._pgg_fourier_cache is not None:
            cached_z, cached_k, cached_key, cached_Pgg = self._pgg_fourier_cache
            if np.isclose(cached_z, z) and cached_key == hod_key and len(cached_k)==len(k) and np.allclose(cached_k, k):
                return cached_Pgg

        Pcc_2h = self.compute_Pcc_2h(k, z, hod_model, **hod_params)
        Pcs_1h = self.compute_Pcs_1h(k, z, hod_model, **hod_params)
        Pcs_2h = self.compute_Pcs_2h(k, z, hod_model, **hod_params)
        Pss_1h = self.compute_Pss_1h(k, z, hod_model, **hod_params)
        Pss_2h = self.compute_Pss_2h(k, z, hod_model, **hod_params)

        if self.verbose:
            print(f"  P_cc_2h range: [{Pcc_2h.min():.4e}, {Pcc_2h.max():.4e}]")
            print(f"  P_cs_1h range: [{Pcs_1h.min():.4e}, {Pcs_1h.max():.4e}]")
            print(f"  P_cs_2h range: [{Pcs_2h.min():.4e}, {Pcs_2h.max():.4e}]")
            print(f"  P_ss_1h range: [{Pss_1h.min():.4e}, {Pss_1h.max():.4e}]")
            print(f"  P_ss_2h range: [{Pss_2h.min():.4e}, {Pss_2h.max():.4e}]")

        result = Pcc_2h + 2 * Pcs_1h + 2 * Pcs_2h + Pss_1h + Pss_2h
        self._pgg_fourier_cache = (z, k, hod_key, result.copy())
        return result

    def compute_Xigg_fourier(self, r, z, hod_model, **hod_params):
        """
        Compute ξ_gg(r) in Fourier space (cc/cs/ss decomposition).

        P_gg(k) computed on the internal k-grid, then FFT⁻¹ → ξ_gg(r).

        Parameters
        ----------
        r : array_like
            Target separations in Mpc/h.
        z : float
            Redshift.
        hod_model : HODModel
        **hod_params : dict
            May include 'f_off' and 'R_off'.

        Returns
        -------
        xi_gg : ndarray
        """
        k = self.fftbase.k
        Pgg = self._compute_Pgg(k, z, hod_model, **hod_params)
        # Combined P_gg → zero at high k (profile suppression);
        # ext=(3,1): power-law on left, zero on right.
        r_out, xi_out = pk2xi(k, Pgg, lowring=True, ext=(3, 1),
                               range=(k[0]*0.3, k[-1]*5.0))

        interp = interp1d(np.log(r_out), np.real(xi_out), kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(r)))

    def compute_Xigg_cc_2h(self, r, z, hod_model, **hod_params):
        r"""xi_gg contributed by P_cc^{2h} only.  2-halo → zero at high k."""
        k = self.fftbase.k
        Pk = self.compute_Pcc_2h(k, z, hod_model, **hod_params)
        r_out, xi_out = pk2xi(k, Pk, lowring=True, ext=(3, 1),
                            range=(k[0]*0.3, k[-1]*5.0))
    # r_out, xi_out = pk2xi(k, Pk, lowring=False, ext=0)
        interp = interp1d(np.log(r_out), np.real(xi_out), kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(r)))

    def compute_Xigg_cs_1h(self, r, z, hod_model, **hod_params):
        r"""xi_gg contributed by P_cs^{1h} only."""
        k = self.fftbase.k
        Pk = self.compute_Pcs_1h(k, z, hod_model, **hod_params)
        r_out, xi_out = pk2xi(k, Pk, lowring=True, ext=(3, 1),
                               range=(k[0]*0.3, k[-1]*5.0))
        # r_out, xi_out = pk2xi(k, Pk, lowring=False, ext=0)
        interp = interp1d(np.log(r_out), np.real(xi_out), kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(r)))

    def compute_Xigg_cs_2h(self, r, z, hod_model, **hod_params):
        r"""xi_gg contributed by P_cs^{2h} only.  2-halo → zero at high k."""
        k = self.fftbase.k
        Pk = self.compute_Pcs_2h(k, z, hod_model, **hod_params)
        r_out, xi_out = pk2xi(k, Pk, lowring=True, ext=(3, 1),
                               range=(k[0]*0.3, k[-1]*5.0))
        # r_out, xi_out = pk2xi(k, Pk, lowring=False, ext=0)
        interp = interp1d(np.log(r_out), np.real(xi_out), kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(r)))

    def compute_Xigg_ss_1h(self, r, z, hod_model, **hod_params):
        r"""xi_gg contributed by P_ss^{1h} only."""
        k = self.fftbase.k
        Pk = self.compute_Pss_1h(k, z, hod_model, **hod_params)
        r_out, xi_out = pk2xi(k, Pk, lowring=True, ext=(3, 1),
                               range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), np.real(xi_out), kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(r)))

    def compute_Xigg_ss_2h(self, r, z, hod_model, **hod_params):
        r"""xi_gg contributed by P_ss^{2h} only.  2-halo → zero at high k."""
        k = self.fftbase.k
        Pk = self.compute_Pss_2h(k, z, hod_model, **hod_params)
        r_out, xi_out = pk2xi(k, Pk, lowring=True, ext=(3, 1),
                               range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), np.real(xi_out), kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(r)))

    def compute_Pgg(self, k, z, hod_model, **hod_params):
        """
        P_gg(k) = P_cc(k) + P_cs(k) + P_ss(k).

        Public wrapper around the internal _compute_Pgg.

        Parameters
        ----------
        k : ndarray
            Wavenumbers in h/Mpc (should match internal grid).
        z : float
            Redshift.
        hod_model : HODModel
        **hod_params : dict
            May include 'f_off' and 'R_off'.

        Returns
        -------
        Pgg : ndarray
            Galaxy-galaxy power spectrum P_gg(k).
        """
        return self._compute_Pgg(k, z, hod_model, **hod_params)

    def compute_Pgm(self, k, z, hod_model, **hod_params):
        r"""
        P_gm(k) — galaxy-matter cross power spectrum (Eq. G8).

        .. math::

            P_{gm}(k) = \frac{1}{\bar{n}_g} \int dM \frac{dn}{dM}
            \bigl[ \langle N_c \rangle H_{\rm off}(k;M)
            + \langle N_s \rangle \tilde{u}_s(k|M) \bigr]
            P_{hm}(k|M)

        Parameters
        ----------
        k : ndarray (n_k,)
            Wavenumbers in h/Mpc.
        z : float
            Redshift.
        hod_model : HODModel
        **hod_params : dict
            May include 'f_off' and 'R_off'.

        Returns
        -------
        Pgm : ndarray (n_k,)
            Galaxy-matter cross power spectrum.
        """
        hod_key = self._get_hod_cache_key(hod_model, **hod_params)
        if self._pgm_fourier_cache is not None:
            cached_z, cached_k, cached_key, cached_Pgm = self._pgm_fourier_cache
            if np.isclose(cached_z, z) and (cached_key == hod_key) and len(cached_k)==len(k) and np.allclose(cached_k, k):
                return cached_Pgm

        if self.xihm is None:
            raise ValueError("Xihm emulator (with get_pkhm_mass) "
                             "is required for compute_Pgm")

        f_off = hod_params.get('f_off', 0.0)
        R_off = hod_params.get('R_off', 0.0)
        Ncen, Nsat, _ = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self._get_cached_ngal(z, hod_model, **hod_params)
        M_arr, lnM_arr, dndlnM = self._get_mass_for_fourier(z)
        R200 = self.satellite_profile.R200(M_arr, z)
        lnM = np.log(M_arr)

        n_mass = len(M_arr)
        n_k = len(k)

        if n_mass == 0:
            if self.verbose:
                print(f"Warning: no mass bins for Pgm at z={z}")
            return np.zeros(n_k)

        # Satellite profile on mass grid
        uk_grid = self._get_cached_uk_grid(k, M_arr, z)[0]  # (nz, n_mass, n_k) nz=1

        # Off-centering kernel
        H_off = self._off_centering_kernel(k[:, np.newaxis], f_off, R_off*R200[np.newaxis, :])

        # Weights
        Wc = Ncen[np.newaxis, :] * H_off 
        Ws = Nsat[np.newaxis, :] * uk_grid

        # P_hm(k|M) directly on M_arr (mass bin, differential)
        P_hm = self._get_pkhm_for_masses(z, k, M_arr)  # (n_mass, n_k) nz=1

        if self.verbose:
            print(f"  Pgm: n_mass={n_mass}, ngal={ngal:.4e}, "
                  f"M_range=[{M_arr[0]:.2e}, {M_arr[-1]:.2e}]")
            print(f"  P_hm range: [{P_hm.min():.4e}, {P_hm.max():.4e}]")
            print(f"  Wc range: [{Wc.min():.4e}, {Wc.max():.4e}]")
            print(f"  Ws range: [{Ws.min():.4e}, {Ws.max():.4e}]")

        # Integrate over mass (vectorised)
        weight = Wc + Ws  # (n_k, n_mass)
        integrand = dndlnM[None, :] * weight * P_hm.T  # (n_k, n_mass)
        Pgm = simpson(integrand, x=lnM, axis=-1)

        if self.verbose:
            print(f"  Pgm range: [{Pgm.min():.4e}, {Pgm.max():.4e}]")

        result = Pgm / ngal
        self._pgm_fourier_cache = (z, k, hod_key, result.copy())
        return result

    def compute_Pgm_components(self, k, z, hod_model, **hod_params):
        r"""
        Decompose P_gm(k) into central, off-centred central, and satellite terms.

        .. math::

            P_{\rm cen}(k) = \frac{1}{\bar{n}_g} \int dM \frac{dn}{dM}
            \langle N_c\rangle (1-f_{\rm off}) P_{\rm hm}(k|M)

            P_{\rm cen,off}(k) = \frac{1}{\bar{n}_g} \int dM \frac{dn}{dM}
            \langle N_c\rangle f_{\rm off} e^{-\frac12 (k R_{\rm off} R_{200})^2}
            P_{\rm hm}(k|M)

            P_{\rm sat}(k) = \frac{1}{\bar{n}_g} \int dM \frac{dn}{dM}
            \langle N_s\rangle \tilde{u}_s(k|M) P_{\rm hm}(k|M)

        Parameters
        ----------
        k : ndarray (n_k,)
        z : float
        hod_model : HODModel or None
        **hod_params : dict

        Returns
        -------
        P_cen : ndarray (n_k,)
        P_cen_off : ndarray (n_k,)
        P_sat : ndarray (n_k,)
            The three components; P_gm = P_cen + P_cen_off + P_sat.
        """
        f_off = hod_params.get('f_off', 0.0)
        R_off = hod_params.get('R_off', 0.0)
        Ncen, Nsat, _ = self._get_cached_hod_weights(hod_model, **hod_params)
        ngal = self._get_cached_ngal(z, hod_model, **hod_params)
        M_arr, lnM_arr, dndlnM = self._get_mass_for_fourier(z)
        R200 = self.satellite_profile.R200(M_arr, z)
        lnM = np.log(M_arr)
        n_mass = len(M_arr)
        n_k = len(k)

        if n_mass == 0:
            return np.zeros(n_k), np.zeros(n_k), np.zeros(n_k)

        uk_grid = self._get_cached_uk_grid(k, M_arr, z)[0] # (nz, n_mass, n_k) nz==1
        P_hm = self._get_pkhm_for_masses(z, k, M_arr) # (n_mass, n_k)
        k2d = k[:, np.newaxis] * R_off * R200[np.newaxis, :]  # (n_k, n_mass)

        # Centered central: (1 - f_off) * N_cen
        W_cen = (1.0 - f_off) * Ncen[np.newaxis, :]
        # Off-centered central: f_off * N_cen * exp(-½k²R²)
        W_cen_off = f_off * Ncen[np.newaxis, :] * np.exp(-0.5 * k2d ** 2)
        # Satellite: N_sat * ũ_s(k|M)
        W_sat = Nsat[np.newaxis, :] * uk_grid

        def _integrate(W):
            integrand = dndlnM[np.newaxis, :] * W * P_hm.T
            return simpson(integrand, x=lnM, axis=-1) / ngal

        P_cen = _integrate(W_cen)
        P_cen_off = _integrate(W_cen_off)
        P_sat = _integrate(W_sat)
        return P_cen, P_cen_off, P_sat

    def compute_DeltaSigma_components(self, R, z, hod_model, **hod_params):
        r"""
        Decompose ΔΣ(R) into central, off-centred central, and satellite terms.

        Each component is the J₂ Hankel transform of the corresponding
        P_gm component.

        Parameters
        ----------
        R : array_like
        z : float
        hod_model : HODModel or None
        **hod_params : dict

        Returns
        -------
        ds_cen : ndarray
        ds_cen_off : ndarray
        ds_sat : ndarray
        """
        k = self.fftbase.k
        P_cen, P_cen_off, P_sat = self.compute_Pgm_components(
            k, z, hod_model, **hod_params)

        rho_crit_0 = RHO_CRIT_0
        rho_m0 = self.Omegam * rho_crit_0
        factor = rho_m0 / 1e12

        def _transform(Pk):
            r_out, ds_kernel = pk2dwp(k, Pk, lowring=True, ext=3,
                                       range=(k[0]*0.3, k[-1]*5.0))
            ds = factor * ds_kernel
            interp = interp1d(np.log(r_out), ds, kind='linear',
                              bounds_error=False, fill_value=0.0)
            return interp(np.log(np.atleast_1d(R)))

        return _transform(P_cen), _transform(P_cen_off), _transform(P_sat)

    def compute_Xigm_fourier(self, r, z, hod_model, **hod_params):
        r"""
        Compute ξ_{gm}(r) via P_gm(k) + FFTLog (Eq. G8 → FFT).

        Uses the k-space galaxy-matter power spectrum (with proper
        off-centering and satellite profile convolution) and transforms
        to real-space via FFTLog. This is more accurate than the direct
        real-space integration used in `compute_Xigm` because it correctly
        includes the :math:`H_{\rm off}` and :math:`\tilde{u}_s` weights.

        Parameters
        ----------
        r : array_like
            Target separations in Mpc/h.
        z : float
            Redshift.
        hod_model : HODModel
        **hod_params : dict

        Returns
        -------
        xi_gm : ndarray
            Galaxy-matter correlation function ξ_{gm}(r).
        """
        k = self.fftbase.k
        Pgm = self.compute_Pgm(k, z, hod_model, **hod_params)
        r_out, xi_out = pk2xi(k, Pgm, lowring=True, ext=3,
                               range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), np.real(xi_out), kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(r)))

    def compute_wgm(self, rp, z, hod_model, **hod_params):
        r"""
        Projected galaxy-matter correlation w_{gp}(r_p) via FFTLog.

        .. math::

            w_{gm}(r_p) = \int_0^\infty \frac{k\,dk}{2\pi}
            P_{gm}(k) J_0(k r_p)

        This is the galaxy-matter analogue of w_p, computed from P_gm(k)
        via the J_0 Hankel transform (pk2wp).

        Parameters
        ----------
        rp : array_like
            Projected separations in Mpc/h.
        z : float
            Redshift.
        hod_model : HODModel
        **hod_params : dict

        Returns
        -------
        wgp : ndarray
            Projected galaxy-matter correlation.
        """
        k = self.fftbase.k
        Pgm = self.compute_Pgm(k, z, hod_model, **hod_params)
        r_out, wgp_out = pk2wp(k, Pgm, lowring=True, ext=3,
                                range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), wgp_out, kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(rp)))

    def compute_wp(self, rp, z, hod_model, pimax=100.0, **hod_params):
        r"""
        Projected galaxy correlation function w_p(r_p).

        Two computation paths:

        * :math:`pimax=None` → FFTLog J_0 Hankel transform of P_gg(k)
          (equivalent to integrating ξ_gg along LOS to infinity).
        * :math:`pimax>0` → real-space projection
          :math:`w_p(r_p) = 2\int_0^{\pi_{\rm max}} \xi_{gg}(\sqrt{r_p^2+\pi^2})\, d\pi`.

        Parameters
        ----------
        rp : array_like
            Projected separations in Mpc/h.
        z : float
            Redshift.
        hod_model : HODModel
        pimax : float or None, default=100.0
            Maximum LOS integration distance in Mpc/h.
            ``None`` uses the FFTLog method (infinite projection).
        **hod_params : dict

        Returns
        -------
        wp : ndarray
            Projected correlation function w_p(r_p).
        """
        if pimax is not None:
            r_grid = self.fftbase.r
            xi = self.compute_Xigg_fourier(r_grid, z, hod_model, **hod_params)
            return compute_projected_correlation(
                rp, r_grid, xi, pimax=float(pimax))

        # FFTLog infinite projection (default when pimax=None)
        # Combined P_gg → zero at high k → zero-pad right side
        k = self.fftbase.k
        Pgg = self._compute_Pgg(k, z, hod_model, **hod_params)
        r_out, wp_out = pk2wp(k, Pgg, lowring=True, ext=(3, 1),
                                range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), wp_out, kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(rp)))

    # ------------------------------------------------------------------
    # Decomposed w_p terms (1-halo / 2-halo decomposition)
    # ------------------------------------------------------------------

    def compute_wp_cc_2h(self, rp, z, hod_model, pimax=100.0, **hod_params):
        r"""w_p contributed by P_cc^{2h} only."""
        if pimax is not None:
            r_grid = self.fftbase.r
            xi = self.compute_Xigg_cc_2h(r_grid, z, hod_model, **hod_params)
            return compute_projected_correlation(
                rp, r_grid, xi, pimax=float(pimax))
        k = self.fftbase.k
        Pk = self.compute_Pcc_2h(k, z, hod_model, **hod_params)
        # 2-halo terms → zero at high k: zero-pad right side, power-law left
        r_out, wp_out = pk2wp(k, Pk, lowring=True, ext=(3, 1),
                                range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), wp_out, kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(rp)))

    def compute_wp_cs_1h(self, rp, z, hod_model, pimax=100.0, **hod_params):
        r"""w_p contributed by P_cs^{1h} only."""
        if pimax is not None:
            r_grid = self.fftbase.r
            xi = self.compute_Xigg_cs_1h(r_grid, z, hod_model, **hod_params)
            return compute_projected_correlation(
                rp, r_grid, xi, pimax=float(pimax))
        k = self.fftbase.k
        Pk = self.compute_Pcs_1h(k, z, hod_model, **hod_params)
        r_out, wp_out = pk2wp(k, Pk, lowring=True, ext=(3, 1),
                                range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), wp_out, kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(rp)))

    def compute_wp_cs_2h(self, rp, z, hod_model, pimax=100.0, **hod_params):
        r"""w_p contributed by P_cs^{2h} only."""
        if pimax is not None:
            r_grid = self.fftbase.r
            xi = self.compute_Xigg_cs_2h(r_grid, z, hod_model, **hod_params)
            return compute_projected_correlation(
                rp, r_grid, xi, pimax=float(pimax))
        k = self.fftbase.k
        Pk = self.compute_Pcs_2h(k, z, hod_model, **hod_params)
        r_out, wp_out = pk2wp(k, Pk, lowring=True, ext=(3, 1),
                                range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), wp_out, kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(rp)))

    def compute_wp_ss_1h(self, rp, z, hod_model, pimax=100.0, **hod_params):
        r"""w_p contributed by P_ss^{1h} only."""
        if pimax is not None:
            r_grid = self.fftbase.r
            xi = self.compute_Xigg_ss_1h(r_grid, z, hod_model, **hod_params)
            return compute_projected_correlation(
                rp, r_grid, xi, pimax=float(pimax))
        k = self.fftbase.k
        Pk = self.compute_Pss_1h(k, z, hod_model, **hod_params)
        r_out, wp_out = pk2wp(k, Pk, lowring=True, ext=(3, 1),
                                range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), wp_out, kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(rp)))

    def compute_wp_ss_2h(self, rp, z, hod_model, pimax=100.0, **hod_params):
        r"""w_p contributed by P_ss^{2h} only."""
        if pimax is not None:
            r_grid = self.fftbase.r
            xi = self.compute_Xigg_ss_2h(r_grid, z, hod_model, **hod_params)
            return compute_projected_correlation(
                rp, r_grid, xi, pimax=float(pimax))
        k = self.fftbase.k
        Pk = self.compute_Pss_2h(k, z, hod_model, **hod_params)
        r_out, wp_out = pk2wp(k, Pk, lowring=True, ext=(3, 1),
                                range=(k[0]*0.3, k[-1]*5.0))
        interp = interp1d(np.log(r_out), wp_out, kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(rp)))

    def compute_DeltaSigma(self, R, z, hod_model, **hod_params):
        r"""
        Excess surface density ΔΣ(R) for galaxy-galaxy lensing.

        Uses the J_2 Hankel transform (pk2dwp) of P_gm(k):

        .. math::

            \Delta\Sigma(R) = \bar{\rho}_m \int_0^\infty \frac{k\,dk}{2\pi}
            P_{gm}(k) J_2(kR)

        Parameters
        ----------
        R : array_like
            Projected separations in Mpc/h.
        z : float
            Redshift.
        hod_model : HODModel
        **hod_params : dict

        Returns
        -------
        ds : ndarray
            Excess surface density ΔΣ(R) in :math:`h M_\odot / \mathrm{pc}^2`.
        """
        k = self.fftbase.k
        Pgm = self.compute_Pgm(k, z, hod_model, **hod_params)
        r_out, ds_kernel = pk2dwp(k, Pgm, lowring=True, ext=3,
                                   range=(k[0]*0.3, k[-1]*5.0))

        # ΔΣ(R) = ρ̄_m × kernel [h Msun / Mpc²] → convert to h Msun / pc²
        # 1 Mpc² = 10¹² pc², so divide by 10¹²
        rho_crit_0 = RHO_CRIT_0 # h^2 Msun/Mpc^3 
        rho_m0 = self.Omegam * rho_crit_0
        ds = rho_m0 * ds_kernel / 1e12

        interp = interp1d(np.log(r_out), ds, kind='linear',
                          bounds_error=False, fill_value=0.0)
        return interp(np.log(np.atleast_1d(R)))

    # end of GalaxyStatsCalculator



def compute_projected_correlation(rp, r, xi, pimax=100.0, n_pi=1000):
    r"""
    Compute the projected correlation function w_p(r_p).

    The projected correlation function integrates along the line of sight:

    .. math::
        w_p(r_p) = 2 \int_0^{\pi_{\rm max}} d\pi
        \xi(r = \sqrt{r_p^2 + \pi^2})

    Parameters
    ----------
    rp : array_like
        Projected separation in Mpc/h
    r : array_like
        3D separation grid in Mpc/h (must be sorted)
    xi : array_like
        3D correlation function on grid r, shape (len(r),) or (nz, len(r))
    pimax : float, default=100.0
        Maximum integration limit along line of sight in Mpc/h
    n_pi : int, default=50
        Number of integration points along the LOS.
        Higher values improve accuracy at the cost of speed.

    Returns
    -------
    ndarray
        Projected correlation w_p(r_p) with same shape as xi's first dimensions
    """
    
    rp = np.atleast_1d(rp)
    # Check if xi is 1D or 2D
    xi = np.atleast_1d(xi)
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    nz, nr = xi.shape
    logr   = np.log10(r)
    wp = np.zeros((nz, len(rp)))
    for iz in range(nz):
        valid = np.isfinite(xi[iz])
        # Shift xi to be positive for log interpolation
        addnum = - np.min(xi[iz][valid])+ 10.0
        logxi  = np.log10(xi[iz][valid] + addnum)
        xi_interp = interp1d(logr[valid], logxi, kind='linear',
                             bounds_error=False, fill_value=0)
        for ir, rpi in enumerate(rp):
            pi_arr = np.linspace(1e-5, pimax, n_pi)
            r_3d   = np.sqrt(rpi ** 2 + pi_arr ** 2)
            logr_3d = np.log10(r_3d)
            xi_arr  = 10.0 ** xi_interp(logr_3d) - addnum
            wp[iz, ir] = 2.0 * simpson(xi_arr, x=pi_arr)
    return wp[0] if squeeze_output else wp
