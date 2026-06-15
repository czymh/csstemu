r"""
Galaxy clustering transforms using the FFTLog Hankel transform.

Wraps CEmulator.hankl.FFTLog for the standard transforms
needed in galaxy clustering:

- P(k) -> xi(r) via spherical Bessel j_0
- xi(r) -> P(k) via spherical Bessel j_0
- P(k) -> w_p(r_p) via cylindrical Bessel J_0
- P(k) -> DeltaSigma(R) via cylindrical Bessel J_2
"""

import numpy as np
from .fftlog import FFTLog
from .cosmology import P2xi, xi2P


def pk2xi(k, pk, lowring=False, ext=0, range=None):
    r"""
    Power spectrum to 3D correlation function via spherical Bessel :math:`j_0`.

    .. math::

        \xi(r) = \int_0^\infty \frac{k^2 dk}{2\pi^2} P(k) \, j_0(kr)

    Parameters
    ----------
    k : array_like
        Uniformly log-spaced wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`.
    pk : array_like
        Power spectrum :math:`P(k)` in :math:`(h^{-1}\mathrm{Mpc})^3`.
    lowring : bool, optional
        Use low-ringing :math:`kr` value (default ``False``).
    ext : int or tuple, optional
        Extrapolation mode (passed to :func:`FFTLog`).
    range : tuple, optional
        Minimum extrapolation range.

    Returns
    -------
    r : ndarray
        Uniformly log-spaced separations in :math:`h^{-1}\mathrm{Mpc}`.
    xi : ndarray
        Correlation function :math:`\xi(r)`.

    Examples
    --------
    >>> k = np.logspace(-3, 1, 100)
    >>> pk = 1.0 / (1.0 + (k / 0.1)**2)
    >>> r, xi = pk2xi(k, pk)
    """
    r, xi = P2xi(k, pk, l=0, n=0, lowring=lowring, ext=ext, range=range)
    return r, np.real(xi)


def xi2pk(r, xi, lowring=False, ext=0, range=None):
    r"""
    3D correlation function to power spectrum via spherical Bessel :math:`j_0`.

    .. math::
        P(k) = 4\pi \int_0^\infty r^2 dr \, \xi(r) \, j_0(kr)

    Parameters
    ----------
    r : array_like
        Uniformly log-spaced separations in :math:`h^{-1}\mathrm{Mpc}`.
    xi : array_like
        Correlation function :math:`\xi(r)`.
    lowring : bool, optional
        Use low-ringing :math:`kr` value.
    ext : int or tuple, optional
        Extrapolation mode.
    range : tuple, optional
        Minimum extrapolation range.

    Returns
    -------
    k : ndarray
        Uniformly log-spaced wavenumbers.
    pk : ndarray
        Power spectrum :math:`P(k)`.
    """
    k, pk = xi2P(r, xi, l=0, n=0, lowring=lowring, ext=ext, range=range)
    return k, np.real(pk)


def pk2wp(k, pk, lowring=False, ext=0, range=None):
    r"""
    Power spectrum to projected correlation function :math:`w_p(r_p)`.

    .. math::
        w_p(r_p) = \int_0^\infty \frac{k\,dk}{2\pi} P(k) \, J_0(k r_p)

    Uses the FFTLog Hankel transform with :math:`\mu = 0` (cylindrical Bessel
    :math:`J_0`).

    Parameters
    ----------
    k : array_like
        Uniformly log-spaced wavenumbers.
    pk : array_like
        Power spectrum :math:`P(k)`.
    lowring : bool, optional
        Use low-ringing :math:`kr` value.
    ext : int or tuple, optional
        Extrapolation mode.
    range : tuple, optional
        Minimum extrapolation range.

    Returns
    -------
    rp : ndarray
        Projected separations in :math:`h^{-1}\mathrm{Mpc}`.
    wp : ndarray
        Projected correlation function :math:`w_p(r_p)` in
        :math:`h^{-1}\mathrm{Mpc}`.
    """
    r, f = FFTLog(k, k * pk / (2.0 * np.pi), q=0, mu=0,
                  lowring=lowring, ext=ext, range=range)
    # f(r) = r * w_p(r)  →  w_p(r) = f(r) / r
    return r, f / r


def pk2dwp(k, pk, lowring=False, ext=0, range=None):
    r"""
    Power spectrum to excess surface density kernel via cylindrical Bessel
    :math:`J_2`.

    .. math::
        \Delta\tilde{\Sigma}(R) =
            \int_0^\infty \frac{k\,dk}{2\pi} P(k) \, J_2(kR)

    .. note::
        This is *without* the mean matter density factor :math:`\bar{\rho}_m`.
        Multiply the result by :math:`\bar{\rho}_m` to obtain the physical
        :math:`\Delta\Sigma(R)` in :math:`h\,M_\odot\,\mathrm{pc}^{-2}`.

    Uses the FFTLog Hankel transform with :math:`\mu = 2` (cylindrical Bessel
    :math:`J_2`).

    Parameters
    ----------
    k : array_like
        Uniformly log-spaced wavenumbers.
    pk : array_like
        Power spectrum :math:`P(k)`.
    lowring : bool, optional
        Use low-ringing :math:`kr` value.
    ext : int or tuple, optional
        Extrapolation mode.
    range : tuple, optional
        Minimum extrapolation range.

    Returns
    -------
    R : ndarray
        Projected separations in :math:`h^{-1}\mathrm{Mpc}`.
    ds : ndarray
        Unscaled excess surface density kernel (multiply by :math:`\bar{\rho}_m`
        for physical units).
    """
    r, f = FFTLog(k, k * pk / (2.0 * np.pi), q=0, mu=2,
                  lowring=lowring, ext=ext, range=range)
    # f(R) = R * ΔΣ̃(R)  →  ΔΣ̃(R) = f(R) / R
    return r, f / r


class fftbase:
    r"""
    Convenience class maintaining paired :math:`k` and :math:`r` grids
    and providing the standard power-spectrum transforms.

    Parameters
    ----------
    fft_num : int, default ``1``
        Sampling factor; the grid size is ``1024 * fft_num``.
    fft_logrmin : float, default ``-3.0``
        Minimum :math:`\log_{10}(r / [h^{-1}\mathrm{Mpc}])`.
    fft_logrmax : float, default ``3.0``
        Maximum :math:`\log_{10}(r / [h^{-1}\mathrm{Mpc}])`.
    kr : float, default ``1``
        Relates :math:`k` and :math:`r` via :math:`k = k_r / r_{\text{reversed}}`.

    Attributes
    ----------
    r : ndarray
        Uniformly log-spaced separations.
    k : ndarray
        Uniformly log-spaced wavenumbers, :math:`k_i = k_r / r_{N-i-1}`.
    """

    def __init__(self, fft_num=1, fft_logrmin=-5.0, fft_logrmax=3.0, kr=1):
        # , fft_logkmin=-4.0, fft_logkmax=2.0
        self.fft_num = fft_num
        self.fft_logrmin = fft_logrmin
        self.fft_logrmax = fft_logrmax

        n = int(2 ** fft_num)
        self.r = np.logspace(fft_logrmin, fft_logrmax, n)
        self.k = kr / self.r[::-1]

    def pk2xi(self, pk, lowring=False, ext=0, range=None):
        """Power spectrum to 3D correlation function."""
        return pk2xi(self.k, pk, lowring=lowring, ext=ext, range=range)

    def xi2pk(self, xi, lowring=False, ext=0, range=None):
        """3D correlation function to power spectrum."""
        return xi2pk(self.r, xi, lowring=lowring, ext=ext, range=range)

    def pk2wp(self, pk, lowring=False, ext=0, range=None):
        """Power spectrum to projected correlation :math:`w_p(r_p)`."""
        return pk2wp(self.k, pk, lowring=lowring, ext=ext, range=range)

    def pk2dwp(self, pk, lowring=False, ext=0, range=None):
        """Power spectrum to excess surface density kernel."""
        return pk2dwp(self.k, pk, lowring=lowring, ext=ext, range=range)
