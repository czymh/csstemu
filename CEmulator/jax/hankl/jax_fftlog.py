"""
JAX-compatible FFTLog Hankel transform.

Replaces CEmulator.hankl.fftlog.FFTLog with pure JAX functions.

Note: JAX does not support complex gamma.  We bridge to scipy via
jax.pure_callback for the gamma-term computation.  This makes the
Hankel transform non-differentiable, which is acceptable for most
cosmology uses (the FFTLog is a numerical tool, not a physical model).
"""

from functools import partial
import jax.numpy as jnp
import jax
import scipy.special


# ── Gamma ratio via scipy bridge ───────────────────────────────────────

def _gamma_term_jax(mu, x):
    """Gamma((mu+1+x)/2) / Gamma((mu+1-x)/2)."""
    def _compute(mu_scalar, x_arr):
        alpha_plus = (mu_scalar + 1.0 + x_arr) / 2.0
        alpha_minus = (mu_scalar + 1.0 - x_arr) / 2.0
        return scipy.special.gamma(alpha_plus) / scipy.special.gamma(alpha_minus)

    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(_compute, result_shape, mu, x)


# ── FFTLog core ────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('lowring',))
def fftlog_core(x, f_x, q, mu, xy=1.0, lowring=False):
    """JAX FFTLog Hankel transform core.

    Args:
        x: (N,) uniformly log-spaced x values (ascending).
        f_x: (N,) function values F(x).
        q: power-law bias exponent.
        mu: Bessel function order (index of J_mu).
        xy: input xy value (default 1).
        lowring: whether to use low-ringing xy.
    Returns:
        y: (N,) uniformly log-spaced y values.
        f_y: (N,) transformed values.
    """
    N = f_x.shape[0]
    x_min = jnp.min(x)
    x_max = jnp.max(x)
    delta_L = (jnp.log(x_max) - jnp.log(x_min)) / (N - 1)
    L = jnp.log(x_max) - jnp.log(x_min)

    log_x0 = jnp.log(x[N // 2])
    x0 = jnp.exp(log_x0)

    # FFT
    c_m = jnp.fft.rfft(f_x)
    m = jnp.fft.rfftfreq(N, d=1.0) * N

    if lowring:
        xy = _lowring_xy_jax(mu, q, L, N, xy)

    y0 = xy / x0
    log_y0 = jnp.log(y0)

    # Frequency mapping
    m_y = jnp.arange(-N // 2, N // 2)
    m_shift = jnp.fft.fftshift(m_y)
    s = delta_L * (-m_y) + log_y0
    y = 10 ** (s[m_shift] / jnp.log(10.0))

    u_m = _u_m_term_jax(m, mu, q, xy, L)
    b = c_m * u_m

    A_m = jnp.fft.irfft(b)
    f_y = A_m[m_shift]
    f_y = f_y[::-1]
    y = y[::-1]

    f_y = f_y * y ** (-q)
    return y, f_y


def _u_m_term_jax(m, mu, q, xy, L):
    """Compute u_m(mu, q) term for FFTLog."""
    omega = 1j * 2.0 * jnp.pi * m / L
    x = q + omega
    U_mu = 2.0 ** x * _gamma_term_jax(mu, x)
    u_m = xy ** (-omega) * U_mu
    u_m = u_m.at[-1].set(jnp.real(u_m[-1]))
    return u_m


def _lowring_xy_jax(mu, q, L, N, xy):
    """Compute low-ringing xy value."""
    delta_L = L / N
    x = q + 1j * jnp.pi / delta_L

    def _imag_log_gamma(z):
        return jnp.imag(jnp.log(scipy.special.gamma(z)))

    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    phip = jax.pure_callback(
        lambda z: jnp.imag(jnp.log(scipy.special.gamma(z))),
        result_shape,
        (mu + 1.0 + x) / 2.0
    )
    phim = jax.pure_callback(
        lambda z: jnp.imag(jnp.log(scipy.special.gamma(z))),
        result_shape,
        (mu + 1.0 - x) / 2.0
    )

    arg = jnp.log(2.0 / xy) / delta_L + (phip - phim) / jnp.pi
    iarg = jnp.rint(arg)
    return jnp.where(arg != iarg, xy * jnp.exp((arg - iarg) * delta_L), xy)


# ── Cosmology-specific transforms ──────────────────────────────────────

@partial(jax.jit, static_argnames=('lowring',))
def pk2xi_jax(k, pk, lowring=False):
    """Power spectrum to 3D correlation function via j_0."""
    r, f = fftlog_core(k, pk * k ** 1.5, q=0, mu=0.5, lowring=lowring)
    xi = f * (2.0 * jnp.pi) ** (-1.5) * r ** (-1.5)
    return r, jnp.real(xi)


@partial(jax.jit, static_argnames=('lowring',))
def xi2pk_jax(r, xi, lowring=False):
    """3D correlation function to power spectrum via j_0."""
    k, F = fftlog_core(r, xi * r ** 1.5, q=0, mu=0.5, lowring=lowring)
    pk = F * (2.0 * jnp.pi) ** 1.5 * k ** (-1.5)
    return k, jnp.real(pk)


@partial(jax.jit, static_argnames=('lowring',))
def pk2wp_jax(k, pk, lowring=False):
    """Power spectrum to projected correlation function w_p(r_p) via J_0."""
    r, f = fftlog_core(k, k * pk / (2.0 * jnp.pi), q=0, mu=0, lowring=lowring)
    return r, f / r


@partial(jax.jit, static_argnames=('lowring',))
def pk2dwp_jax(k, pk, lowring=False):
    """Power spectrum to excess surface density via J_2."""
    r, f = fftlog_core(k, k * pk / (2.0 * jnp.pi), q=0, mu=2, lowring=lowring)
    return r, f / r
