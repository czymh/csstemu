import numpy as np
from scipy.special import gamma, loggamma
from .preprocess import preprocess


def _gamma_term(mu, x, cutoff=200.0):
    r"""
	Compute the following term:
	
		\Gamma[(\mu + 1 + x) / 2] / \Gamma[(\mu + 1 - x) / 2]

	(see eq.16 in https://jila.colorado.edu/~ajsh/FFTLog/)

	Parameters
    ----------
	mu : float
        Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
	x : float
        Argument of function U_{m}(x) of eq.16.
	cutoff : float
        Cuttoff value to switch to Gamma function using Stirling's approximation.
	
    Returns
    -------
    g_m : array
	    Gamma fraction term of eq. 16
	"""

    imag_x = np.imag(x)

    # gamma has poles at x = mu + 1 + 2k for integer k >= 0, where the ratio is 0
    n_pole   = (np.real(x) - (mu + 1.0)) / 2.0
    is_pole  = (imag_x == 0.0) & (n_pole >= 0) & (n_pole == np.floor(n_pole))
    asym_m   = np.absolute(imag_x) > cutoff
    good_m   = (~asym_m) & (~is_pole)

    g_m = np.zeros(x.size, dtype=complex)

    asym_x = x[asym_m]
    asym_plus = (mu + 1 + asym_x) / 2.0
    asym_minus = (mu + 1 - asym_x) / 2.0

    # high-order expansion
    g_m[asym_m] = np.exp(
        (asym_plus - 0.5) * np.log(asym_plus)
        - (asym_minus - 0.5) * np.log(asym_minus)
        - asym_x
        + 1.0 / 12.0 * (1.0 / asym_plus - 1.0 / asym_minus)
        + 1.0 / 360.0 * (1.0 / asym_minus ** 3.0 - 1.0 / asym_plus ** 3.0)
        + 1.0 / 1260.0 * (1.0 / asym_plus ** 5.0 - 1.0 / asym_minus ** 5.0)
    )

    alpha_plus = (mu + 1.0 + x[good_m]) / 2.0
    alpha_minus = (mu + 1.0 - x[good_m]) / 2.0
    with np.errstate(over='ignore', invalid='ignore'):
        good = gamma(alpha_plus) / gamma(alpha_minus)
    if not np.all(np.isfinite(good)):
        bad = x[good_m][~np.isfinite(good)]
        raise ValueError(
            'The gamma ratio overflows in float64 for x=%s (mu=%g); '
            'the FFTLog power-law bias q is too far from zero.' % (bad, mu))
    g_m[good_m] = good

    return g_m


def _lowring_xy(mu, q, L, N, xy=1.0):
    r"""
	Compute xy so that 

		(x y)^{- i \pi/dlnr} U_{\mu}(q + i \pi/dlnr)

	is real thus reducing lowringing of the Hankel Transform.

	Parameters
    ----------
	mu : float
        Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
	q : float
        Exponent of power law bias; q may be any real number, positive or negative.
	L : float
        Range of uniformly logarithmically spaced points.
	N : int
        Number of uniformly logarithmically spaced points.
	xy : float
        Input value of xy (Default is 1).

	Returns
    -------
	xy : float
        Low-ringing value of xy nearest to input xy.
	"""

    delta_L = L / float(N)

    x = q + 1j * np.pi / delta_L

    x_plus = (mu + 1 + x) / 2.0
    x_minus = (mu + 1 - x) / 2.0

    gp = gamma(x_plus)
    gm = gamma(x_minus)
    if np.isfinite(gp) and np.isfinite(gm) and gp != 0 and gm != 0:
        phip = np.imag(np.log(gp))
        phim = np.imag(np.log(gm))
    else:
        # gamma() overflows or underflows for a large imaginary part and log()
        # then loses the phase; loggamma() carries it continuously. Only fall
        # back there so the well conditioned path stays bit identical.
        phip = np.imag(loggamma(x_plus))
        phim = np.imag(loggamma(x_minus))

    if not (np.isfinite(phip) and np.isfinite(phim)):
        raise ValueError('_lowring_xy got non-finite phase (mu=%g, q=%g, N=%d).'
                         % (mu, q, N))
    if xy <= 0:
        raise ValueError('_lowring_xy needs xy > 0, got %g.' % xy)

    arg = np.log(2.0 / xy) / delta_L + (phip - phim) / np.pi
    iarg = np.rint(arg)
    if arg != iarg:
        xy = xy * np.exp((arg - iarg) * delta_L)

    return xy


def _u_m_term(m, mu, q, xy, L):
    r"""
	Compute u_{m}(\mu, q) term defined as

		u_{m}(\mu, q) = (x_{0}y_{0})^{-2\pi i m / L} U_{\mu}(q + 2\pi i m/ L)

	(see eq.18 in https://jila.colorado.edu/~ajsh/FFTLog/)

	Parameters
    ----------
	m : float)
        Index of u_{m} term.
	mu : float)
        Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
	q : float)
        Exponent of power law bias; q may be any real number, positive or negative.
	xy : float)
        Value of xy.
	L : float
        Range of uniformly logarithmically spaced points.
	
    Returns
    -------
	u_{m}(\mu, q) : (float)
        u_{m} term of eq. 18
	"""

    omega = 1j * 2 * np.pi * m / float(L)

    x = q + omega

    U_mu = 2 ** x * _gamma_term(mu, x)

    u_m = (xy) ** (-omega) * U_mu

    u_m[m.size - 1] = np.real(u_m[m.size - 1])

    return u_m


def FFTLog(x, f_x, q, mu, xy=1.0, lowring=False, ext=0, range=None, return_ext=False):
    r"""Hankel Transform based on the FFTLog algorithm of [1] and [2].

	Defined as:

	    .. math:: f(y)= \int_0^\infty F(x) (xy)^{q} J_\mu(xy) y dx


    Parameters
    ----------
    x : array
        Array of uniformly logarithmically spaced x values.
    f_x : array
        Array of respective F(x) values.
    q : float
        Exponent of power law bias; q may be any real number, positive (for forward transform) or negative (for inverse transform).
    mu : float
        Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
    xy : float
        Input value of xy (Default is 1).
    lowring : bool
        If True, then use low-ringing value of xy closest to input value of xy (Default is False).
    ext : int or tuple or list
        Controls the extrapolation mode. When ext is an integer then the same extrapolation method will
        be used for both ends of the input array. Alternatively, when ext is an tuple (ext_left, ext_right) or a list [ext_left,
        ext_right] then different methods can be used for the two ends of the the input array.

        * if ext=0 then no extrapolation is performed (Default).
        * if ext=1 then zero padding is performed.
        * if ext=2 then constant padding is performed.
        * if ext=3 then Power-Law extrapolation is performed.
    range : tuple or list
        The minimum extrapolation range in the form of a tuple (x_min, x_max) or list [x_min, x_max]. When range=None (Default) then the extended range is chosen automatically such that its array-size is the next power of two.
    return_ext : bool
        When False (Default) the result is cropped to fit the original x range.

    Returns
    -------
    y, f(y) : array, array
        Array of uniformly logarithmically spaced y values and array of respecive f(y) values.


    References
    ----------
        [1] J. D. Talman. Numerical Fourier and Bessel Transforms in Logarithmic Variables. Journal of Computational Physics, 29:35-48, October 1978.
		
        [2] A. J. S. Hamilton. Uncorrelated modes of the non-linear power spectrum. MNRAS, 312:257-284, February 2000.
	"""

    if mu + 1.0 + q == 0.0:
        raise ValueError("The FFTLog Hankel Transform is singular when mu + 1 + q = 0.")

    x = np.asarray(x)
    f_x = np.asarray(f_x)
    if x.ndim != 1 or x.size != f_x.size:
        raise ValueError('x and f_x must be 1-D arrays of the same size, got %s and %s.'
                         % (x.shape, f_x.shape))
    if x.size < 2:
        raise ValueError('FFTLog needs at least 2 samples, got %d.' % x.size)
    if not np.all(np.isfinite(f_x)):
        raise ValueError('f_x contains non-finite values; the transform would be meaningless.')
    if np.any(x <= 0):
        raise ValueError('x must be strictly positive for a logarithmic grid.')

    x, f_x, N_left, N_right = preprocess(x, f_x, ext=ext, range=range)

    N = f_x.size
    if N % 2 != 0:
        raise ValueError('FFTLog needs an even number of samples, got %d; the Nyquist '
                         'term is paired with an N//2 grid.' % N)
    L = np.log(np.max(x)) - np.log(np.min(x))
    delta_L = L / float(N - 1)
    log_x0 = np.log(x[N // 2])
    x0 = np.exp(log_x0)

    c_m = np.fft.rfft(f_x)
    m = np.fft.rfftfreq(N, d=1.0) * float(N)
    if lowring:
        xy = _lowring_xy(mu, q, L, N, xy)
    y0 = xy / x0
    log_y0 = np.log(y0)

    m_y = np.arange(-N // 2, N // 2)
    m_shift = np.fft.fftshift(m_y)

    s = delta_L * (-m_y) + log_y0
    id = m_shift
    y = 10 ** (s[id] / np.log(10))

    u_m = _u_m_term(m, mu, q, xy, L)
    b = c_m * u_m

    A_m = np.fft.irfft(b)
    f_y = A_m[id]

    f_y = f_y[::-1]
    y = y[::-1]

    if q != 0:
        f_y = f_y * (y) ** (-float(q))

    if return_ext:
        return y, f_y
    else:
        if N_right == 0:
            return y[N_left:], f_y[N_left:]
        else:
            return y[N_left:-N_right], f_y[N_left:-N_right]
