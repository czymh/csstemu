import numpy as np
from .fftlog import FFTLog


def P2xi(k, P, l, n=0, lowring=False, ext=0, range=None, return_ext=False):
    r"""
    Hankel Transform Power Spectrum Multipole to Correlation Function Multipole.

    .. math:: \xi_{l}^{(n)}(r) = i^{l} \int_{0}^{\infty} k^{2} dk / (2 \pi^{2}) (kr)^{-n} P_{l}^{(n)}(k) j_{l}(ks)

    Parameters
    ----------
    k : array
        Array of uniformly logarithmically spaced wavenumbers.
    P : array
        Array of respective Power Spectrum values.
    l : int
        Degree of Power Spectrum multipole.
    n : int
        Order of expansion (Default is 0, plane-parallel).
    lowring : bool
        If True then use low-ringing value of kr (Default is False).
    ext : int or tuple or list
        Controls the extrapolation mode. When ext is an integer then the same extrapolation method will be used
        for both ends of the input array. Alternatively, when ext is an tuple (ext_left, ext_right) or a list
        [ext_left, ext_right] then different methods can be used for the two ends of the the input array.

        * if ext=0 then no extrapolation is performed (Default).
        * if ext=1 then zero padding is performed.
        * if ext=2 then constant padding is performed.
        * if ext=3 then Power-Law extrapolation is performed.
    range : tuple or list
        The minimum extrapolation range in the form of a tuple (k_min, k_max) or list [k_min, k_max]. When range=None
        (Default) then the extended range is chosen automatically such that its array-size is the next power of two.
    return_ext : bool
        When False (Default) the result is cropped to fit the original k range.

    Returns
    -------
    r, xi : array, array
        Array of uniformly logarithmically spaced r values and respective array of xi_{l}^{(n)}(r) values.
    """
    r, f = FFTLog(
        k,
        P * k ** 1.5,
        q=-n,
        mu=l + 0.5,
        lowring=lowring,
        ext=ext,
        range=range,
        return_ext=return_ext,
    )
    return r, f * (2.0 * np.pi) ** (-1.5) * r ** (-1.5) * (1j) ** l


def xi2P(r, xi, l, n=0, lowring=False, ext=0, range=None, return_ext=False):
    r"""
    Hankel Transform Correlation Function Multipole to Power Spectrum Multipole.

    .. math:: P_{l}^{(n)}(k) = 4 \pi (-i)^{l} \int_{0}^{\infty} r^{2} dr (kr)^{n} \xi_{l}^{(n)}(r) j_{l}(kr)

    Parameters
    ----------
    r : array
        Array of uniformly logarithmically spaced separations.
    xi : array
        Array of respective two point correlation function values.
    l : int
        Degree of Power Spectrum multipole.
    n : int
        Order of expansion (Default is 0, plane-parallel).
    lowring : bool
        If True then use low-ringing value of kr (Default is False).
    ext : int or tuple or list
        Controls the extrapolation mode. When ext is an integer then the same extrapolation method will be used
        for both ends of the input array. Alternatively, when ext is an tuple (ext_left, ext_right) or a list
        [ext_left, ext_right] then different methods can be used for the two ends of the the input array.

        * if ext=0 then no extrapolation is performed (Default).
        * if ext=1 then zero padding is performed.
        * if ext=2 then constant padding is performed.
        * if ext=3 then Power-Law extrapolation is performed.
    range : tuple or list
        The minimum extrapolation range in the form of a tuple (k_min, k_max) or list [k_min, k_max]. When range=None (Default)
        then the extended range is chosen automatically such that its array-size is the next power of two.
    return_ext : bool
        When False (Default) the result is cropped to fit the original k range.
        

    Returns
    -------
    k, P : array, array
        Array of uniformly logarithmically spaced k values and array of respective P_{l}^{(n)}(k) values.
    """
    k, F = FFTLog(
        r,
        xi * r ** 1.5,
        q=n,
        mu=l + 0.5,
        lowring=lowring,
        ext=ext,
        range=range,
        return_ext=return_ext,
    )
    return k, F * (2.0 * np.pi) ** 1.5 * k ** (-1.5) * (-1j) ** l
