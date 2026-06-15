"""
Satellite profile models for Fourier-space HOD galaxy clustering.

Provides the normalised satellite number density profile :math:`\\tilde{u}_s(k|M)`
in Fourier space, used in the HOD power spectrum calculation:

.. math::

    P_{\\rm gg}(k) = \\frac{1}{\\bar{n}_g^2} \\int dM \\frac{dn}{dM}
    \\left[ 2\\langle N_c\\rangle\\langle N_s\\rangle \\tilde{u}_s H_{\\rm off}
    + \\langle N_c\\rangle\\langle N_s\\rangle^2 \\tilde{u}_s^2 \\right] \\\\
    + \\frac{1}{\\bar{n}_g^2} \\left[\\int dM \\frac{dn}{dM} W(k;M)\\right]^2
    P_{\\rm hh}(k)

where :math:`W(k;M) = \\langle N_c\\rangle H_{\\rm off}(k;M) + \\langle N_s\\rangle \\tilde{u}_s(k;M)`.

Two profile types:
- ``'nfw'``: Analytic NFW profile (Duffy+2008 c-M relation)
- ``'xihm'``: Numerical profile from the halo-matter correlation emulator
"""

import numpy as np
from scipy.special import sici
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import warnings
        
# Critical density at z=0 in h^2 Msun / Mpc^3
from ..cosmology import RHO_CRIT_0


def R200_from_M(M, z, Omegam=0.315):
    """
    R_200m from halo mass.

    Uses *M = 200 rho_m(z) (4pi/3) R200^3* with
    *rho_m(z) = Omegam * rho_crit_0 * (1+z)^3*.

    Parameters
    ----------
    M : float or ndarray
        Halo mass in Msun/h.
    z : float
        Redshift.
    Omegam : float
        Omega_matter at z=0.

    Returns
    -------
    R200 : float or ndarray
        R_200m in Mpc/h.
    """
    rho_m_z = Omegam * RHO_CRIT_0 * 200.0 ## comoving rho (no (1+z)^3 factor for R200m)
    return (3.0 * M / (4 * np.pi * rho_m_z)) ** (1.0 / 3.0)


def concentration_duffy08(M, z):
    r"""
    Duffy+2008 c(M,z) relation for M200m.

    .. math::

        c = 5.71 \\, (M / 2\\times 10^{12})^{-0.084} \\, (1+z)^{-0.47}

    Parameters
    ----------
    M : float or ndarray
        Halo mass in Msun/h.
    z : float
        Redshift.

    Returns
    -------
    c : float or ndarray
        NFW concentration.
    """
    return 5.71 * (M / 2.0e12) ** (-0.084) * (1.0 + z) ** (-0.47)


class SatelliteProfile:
    """
    Normalised satellite profile :math:`\\tilde{u}_s(k|M)` in Fourier space.

    The profile satisfies :math:`\\int 4\\pi r^2 u_s(r|M) dr = 1`.

    Parameters
    ----------
    profile_type : {'nfw', 'xihm'}
        Profile model.
    concentration_model : callable or None
        Function ``c(M, z) -> concentration``.  Default is
        `concentration_duffy08`.
    Omegam : float
        Omega_matter at z=0 (used for R200).
    """

    def __init__(self, profile_type='nfw', concentration_model=None,
                 Omegam=0.315, Rc=1.0, apodization_scale=None):
        self.profile_type = profile_type
        self.Omegam = Omegam
        self.Rc = Rc
        self.apodization_scale = apodization_scale
        self.concentration = concentration_model or concentration_duffy08

    def R200(self, M, z):
        """R_200m in Mpc/h for halo mass *M* at redshift *z*."""
        return R200_from_M(M, z, Omegam=self.Omegam)

    # ------------------------------------------------------------------
    # NFW profile
    # ------------------------------------------------------------------

    @staticmethod
    def _nfw_uk(krs, c):
        r"""
        Analytic Fourier transform of the truncated NFW profile.

        From Cooray & Sheth (2002), Eq. 81:

        .. math::

            \\tilde{u}(k) = \\frac{1}{\\ln(1+c) - c/(1+c)}
            \\Bigl[ \\cos(k r_s)\\, \\bigl[\\operatorname{Ci}((1+c)k r_s)
            - \\operatorname{Ci}(k r_s)\\bigr] \\\\
            + \\sin(k r_s)\\, \\bigl[\\operatorname{Si}((1+c)k r_s)
            - \\operatorname{Si}(k r_s)\\bigr]
            - \\frac{\\sin(c k r_s)}{(1+c)k r_s} \\Bigr]

        Parameters
        ----------
        krs : ndarray
            k * r_s where r_s = R200 / c.
        c : float
            NFW concentration.

        Returns
        -------
        uk : ndarray
            :math:`\\tilde{u}(k)`.
        """
        x = np.asarray(krs, dtype=float)
        cx = c * x
        c1x = (1.0 + c) * x

        Si_c1x, Ci_c1x = sici(c1x)
        Si_x, Ci_x = sici(x)

        uk = (np.cos(x) * (Ci_c1x - Ci_x)
              + np.sin(x) * (Si_c1x - Si_x)
              - np.sin(cx) / c1x)

        norm = np.log(1.0 + c) - c / (1.0 + c)
        uk = uk / norm

        # k -> 0 limit: ũ(0) = 1 by normalisation
        uk[np.isnan(uk)] = 1.0

        return uk

    def u_sat_k_nfw(self, k, M, z):
        """
        NFW satellite profile in Fourier space.

        Parameters
        ----------
        k : ndarray
            Wavenumbers in h/Mpc.
        M : float
            Halo mass in Msun/h.
        z : float
            Redshift.

        Returns
        -------
        uk : ndarray
            :math:`\\tilde{u}_s(k|M)`, same shape as *k*.
        """
        c = self.Rc * self.concentration(M, z)
        rs = self.R200(M, z) / c
        return self._nfw_uk(k * rs, c)

    def u_sat_r_nfw(self, r, M, z):
        """
        Real-space NFW satellite profile u_s(r|M).

        Parameters
        ----------
        r : ndarray
            Radii in Mpc/h.
        M : float
            Halo mass in Msun/h.
        z : float
            Redshift.

        Returns
        -------
        us_r : ndarray
            u_s(r|M), same shape as *r*, zero for r > R200.
        """
        c = self.Rc * self.concentration(M, z)
        R200 = self.R200(M, z)
        rs = R200 / c
        x = r / rs

        norm = 4.0 * np.pi * rs ** 3 * (np.log(1.0 + c) - c / (1.0 + c))
        us_r = 1.0 / (norm * x * (1.0 + x) ** 2)
        us_r[r > R200] = 0.0
        return us_r

    # ------------------------------------------------------------------
    # Xihm-based profile
    # ------------------------------------------------------------------
    @staticmethod
    def _hankel_transform(r, f, k_target, ext=0):
        """
        FFTLog spherical Bessel j₀ Hankel transform.

        Computes:

        .. math::

            F(k) = 4\\pi \\int_0^\\infty r^2 f(r)\\, j_0(kr)\\, dr

        Uses FFTLog with no input-side extrapolation (ext=0) and
        nearest-boundary fallback via ``interp1d`` for any target
        wavenumbers outside the native FFTLog grid.  This prevents
        low-k values from being incorrectly set to zero when the
        user's k grid extends below the FFTLog's lowest k (typically
        set by 1/r_max).

        Parameters
        ----------
        r : ndarray (n_r,)
            Uniformly log-spaced radial grid in Mpc/h.
        f : ndarray (n_r,)
            Input function.
        k_target : ndarray (n_k,)
            Target wavenumbers in h/Mpc.
        ext : int, optional
            FFTLog extrapolation mode (default: 0).

        Returns
        -------
        F_k : ndarray (n_k,)
            Transformed function interpolated on the target k grid.
        k_fft : ndarray
            FFTLog native k-grid (for diagnostic / range-checking).
        """
        from ..hankl.galaxy import xi2pk
        from scipy.interpolate import interp1d

        k_fft, F_fft = xi2pk(r, f, ext=ext)
        # Nearest-boundary fill: for k below grid use F_fft[0]
        # (close to ũ→1); for k above grid use F_fft[-1] (close to 0).
        interp = interp1d(np.log(k_fft), F_fft, kind='linear',
                          bounds_error=False,
                          fill_value=(float(F_fft[0]), float(F_fft[-1])))
        return interp(np.log(k_target)), k_fft

    # ------------------------------------------------------------------
    # p_hm_dist — dark_emulator convention
    # ------------------------------------------------------------------

    def compute_p_hm_dist_emu(self, k_target, r, xi_hm, R200,
                              apodization_scale=None):
        r"""
        Compute p_hm_dist following the dark_emulator convention.

        Uses truncated :math:`\xi_{\rm hm}` (zero for :math:`r > R_{200}`,
        negative values clipped to zero) and the FFTLog transform:

        .. math::

            p_{\rm hm}^{\rm dist}(k) =
            \frac{\rho_m\; \mathcal{F}[\xi_{\rm hm}^{\rm trunc}](k)}
            {\int_0^{R_{200}} 4\pi r^3 \rho_m [1 + \xi_{\rm hm}^{\rm trunc}(r)]\, dr}

        where :math:`\mathcal{F}[\xi](k) = 4\pi \int r^2 \xi(r) j_0(kr)\, dr`.

        Parameters
        ----------
        k_target : ndarray (n_k,)
            Target wavenumbers in h/Mpc.
        r : ndarray (n_r,)
            Logarithmic radial grid in Mpc/h.
        xi_hm : ndarray (n_r,) or (n_mass, n_r)
            Halo-matter cross-correlation on *r* grid.
        R200 : float
            Truncation radius in Mpc/h.
        apodization_scale : float or None
            If set, applies Gaussian apodization
            :math:`\exp(-\tfrac12 (R_{200}/{\rm scale})^2 k^2)`.

        Returns
        -------
        p_hm_dist : ndarray (n_k,) or (n_mass, n_k)
        """
        rho_m = RHO_CRIT_0 * self.Omegam
        R200 = np.atleast_1d(R200)
        squeeze = (xi_hm.ndim == 1)
        if squeeze:
            xi_hm = xi_hm[np.newaxis, :]  # (1, n_r)
        # Batch dimension from xi_hm unless R200 provides it
        n_mass = max(R200.shape[0], xi_hm.shape[0])
        if R200.shape[0] < n_mass:
            R200 = np.broadcast_to(R200, (n_mass,))

        k_target = np.asarray(k_target, dtype=float)
        r = np.asarray(r, dtype=float)
        xi_hm = np.asarray(xi_hm, dtype=float)
        # FFTLog transform via shared helper
        p_hm_tmp = np.zeros((n_mass, len(k_target)))
        for im in range(n_mass):
            # Truncate: clip negative, zero beyond R200
            # xi_trunc = np.maximum(xi_hm[im], 0.0)
            xi_trunc  = np.copy(xi_hm[im])
            R200_im = R200[im] if R200.ndim > 0 else R200
            mask = r > R200_im
            xi_trunc[mask] = 0.0

            pk_tmp, k_fft = self._hankel_transform(
                r, xi_trunc, k_target, ext=0)

            # Warn if k_target outside FFTLog range
            if k_target.min() < k_fft.min() or k_target.max() > k_fft.max():
                warnings.warn(
                    f"compute_p_hm_dist_emu: target k range "
                    f"[{k_target.min():.3g}, {k_target.max():.3g}] "
                    f"exceeds FFTLog range "
                    f"[{k_fft.min():.3g}, {k_fft.max():.3g}]. "
                    "Extrapolating with fill_value=0.")
            p_hm_tmp[im] = pk_tmp
            # Apply apodization
            if apodization_scale is not None:
                sigma = R200_im / apodization_scale
                p_hm_tmp[im] *= np.exp(-0.5 * sigma ** 2 * k_target ** 2)

            # Normalisation: ∫ 4π r³ ρ_m (1+ξ_hm) dlogr from 0 to R200
            mask_r = r <= R200_im
            integrand = (4.0 * np.pi * r[mask_r] ** 3 * rho_m
                        * (1.0 + (xi_trunc[mask_r])))
            norm = simpson(integrand, x=np.log(r[mask_r]))  # float
            p_hm_tmp[im] *= (rho_m / norm)
             # Enforce p_hm_dist(k→0) = 1 by normalisation avoid ringing artifacts at low k due to small numerical errors in the integral.
            p_hm_tmp[im][k_target < 1/5/R200_im] = 1.0
        if squeeze:
            return p_hm_tmp[0]
        return p_hm_tmp

    def u_sat_k_xihm(self, k_target, r, xi_hm, R200, apodization_scale=None):
        return self.compute_p_hm_dist_emu(k_target, r, xi_hm, R200, apodization_scale)

    def compute_p_hm_dist_nfw(self, k, M, z):
        """Alias for NFW profile — delegates to `u_sat_k_nfw`.

        In the dark_emulator the NFW version of *p_hm_dist* is simply
        the analytic Fourier-space NFW profile, identical to
        `u_sat_k_nfw`.
        """
        return self.u_sat_k_nfw(k, M, z)
