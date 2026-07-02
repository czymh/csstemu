"""
JAX-compatible replacements for scipy utilities used throughout csstemu.

All functions are pure and compatible with jax.jit, jax.grad, and jax.vmap.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ── Pairwise distances (replacing scipy.spatial.distance) ──────────────

def pdist_sqeuclidean(X):
    """JAX equivalent of pdist(X, metric='sqeuclidean').

    Args:
        X: (n, d) array.
    Returns:
        (n*(n-1)/2,) upper-triangular pairwise squared Euclidean distances.
    """
    XX = jnp.sum(X * X, axis=1)
    D = XX[:, None] + XX[None, :] - 2 * (X @ X.T)
    return D[jnp.triu_indices_from(D, k=1)]


def cdist_sqeuclidean(X, Y):
    """JAX equivalent of cdist(X, Y, metric='sqeuclidean').

    Args:
        X: (m, d) array.
        Y: (n, d) array.
    Returns:
        (m, n) pairwise squared Euclidean distances.
    """
    XX = jnp.sum(X * X, axis=1)
    YY = jnp.sum(Y * Y, axis=1)
    return XX[:, None] + YY[None, :] - 2 * (X @ Y.T)


def pdist_euclidean(X):
    """JAX equivalent of pdist(X, metric='euclidean')."""
    return jnp.sqrt(pdist_sqeuclidean(X))


def cdist_euclidean(X, Y):
    """JAX equivalent of cdist(X, Y, metric='euclidean')."""
    return jnp.sqrt(cdist_sqeuclidean(X, Y))


# ── Interpolation (replacing scipy.interpolate) ────────────────────────

def interp1d_linear(x, xp, fp):
    """JAX 1D linear interpolation, vectorized over the last axis of fp.

    Args:
        x: scalar query point.
        xp: (n,) sorted grid points.
        fp: (n,) or (n, nk) values at grid points.
    Returns:
        Interpolated value(s), same shape as fp minus the xp axis.
    """
    return jnp.interp(x, xp, fp)


def _cubic_hermite_weights(t):
    """Cubic Hermite basis weights for t in [0,1].

    Returns weights w0, w1, w2, w3 for points p[-1], p[0], p[1], p[2]
    where the query point lies between p[0] and p[1].
    """
    t2 = t * t
    t3 = t2 * t
    # Catmull-Rom spline weights
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2
    return w0, w1, w2, w3


def _interp1d_cubic_scalar(x, xp, fp):
    """Cubic Catmull-Rom interpolation at a scalar x.

    Args:
        x: scalar.
        xp: (n,) sorted grid (ascending), n >= 4.
        fp: (n,) values at grid points.
    Returns:
        Interpolated scalar value.
    """
    i = jnp.searchsorted(xp, x, side='right')
    i = jnp.clip(i, 2, len(xp) - 2)  # ensure we have 2 points on each side
    t = (x - xp[i - 1]) / (xp[i] - xp[i - 1])
    w0, w1, w2, w3 = _cubic_hermite_weights(t)
    return w0 * fp[i - 2] + w1 * fp[i - 1] + w2 * fp[i] + w3 * fp[i + 1]


def interp1d_cubic(x, xp, fp):
    """JAX 1D cubic Catmull-Rom interpolation.

    Args:
        x: (nq,) query points.
        xp: (n,) sorted grid points (ascending).
        fp: (n,) or (n, m) values at grid points.
    Returns:
        (nq,) or (nq, m) interpolated values.
    """
    return jnp.vectorize(_interp1d_cubic_scalar, signature='(),(n),(n)->()')(x, xp, fp)


def interp2d_bilinear(x_grid, y_grid, data, x_query, y_query):
    """2D bilinear interpolation on a rectilinear grid.

    This replaces scipy.interpolate.RectBivariateSpline with kx=1, ky=1.

    Args:
        x_grid: (nx,) sorted grid points in x (ascending).
        y_grid: (ny,) sorted grid points in y (ascending).
        data: (nx, ny) values at grid points, indexed as data[ix, iy].
        x_query: (nqx,) query x values.
        y_query: (nqy,) query y values.
    Returns:
        (nqx, nqy) interpolated values.
    """
    # Step 1: interpolate in y (linear) for each fixed x grid point
    # data has shape (nx, ny), interp along axis 1
    # Result: (nx, nqy)
    val_y = jnp.array([jnp.interp(y_query, y_grid, data[ix, :]) for ix in range(len(x_grid))])

    # Step 2: interpolate in x (linear) for each query y point
    # val_y: (nx, nqy), interp along axis 0
    result = jnp.array([jnp.interp(x_query, x_grid, val_y[:, iy]) for iy in range(len(y_query))])
    return result.T  # (nqx, nqy)


def interp2d_cubic_linear(x_grid, y_grid, data, x_query, y_query):
    """2D interpolation: cubic in x, linear in y.

    Replaces scipy.interpolate.RectBivariateSpline with kx=3, ky=1.

    Args:
        x_grid: (nx,) sorted grid points in x (ascending), nx >= 4.
        y_grid: (ny,) sorted grid points in y (ascending).
        data: (nx, ny) values at grid points.
        x_query: (nqx,) query x values.
        y_query: (nqy,) query y values.
    Returns:
        (nqx, nqy) interpolated values.
    """
    # Step 1: linear in y for each x grid point -> (nx, nqy)
    val_y = jnp.array([jnp.interp(y_query, y_grid, data[ix, :]) for ix in range(len(x_grid))])

    # Step 2: cubic in x for each y query point
    # val_y: (nx, nqy), interpolate each column with cubic
    results = []
    for iy in range(len(y_query)):
        col = val_y[:, iy]
        interp_col = interp1d_cubic(x_query, x_grid, col)
        results.append(interp_col)
    return jnp.array(results).T  # (nqx, nqy)


# ── Normalization (replacing MyStandardScaler and NormCosmo) ───────────

def norm_cosmo(cosmologies, param_names, param_limits):
    """Normalize cosmological parameters to [0, 1] range.

    Args:
        cosmologies: (n_cosmo, n_params) array.
        param_names: list of parameter name strings.
        param_limits: dict mapping param name -> [min, max].
    Returns:
        (n_cosmo, n_params) normalized array.
    """
    ncosmo = jnp.zeros_like(cosmologies)
    for i, param in enumerate(param_names):
        lo, hi = param_limits[param]
        ncosmo = ncosmo.at[:, i].set((cosmologies[:, i] - lo) / (hi - lo))
    return ncosmo


def standard_scale_transform(X, mean, scale):
    """Apply standard scaling: (X - mean) / scale.

    Args:
        X: (n, m) array.
        mean: (m,) array.
        scale: (m,) array.
    Returns:
        (n, m) scaled array.
    """
    return (X - mean) / scale


def standard_scale_inverse(Y, mean, scale):
    """Reverse standard scaling: Y * scale + mean."""
    return Y * scale + mean


# ── Fermi-Dirac interpolation helper ───────────────────────────────────

def fermi_dirac_interp(y, y_grid, F_grid):
    """JAX interpolation on the precomputed Fermi-Dirac table.

    Args:
        y: scalar or (n,) query values.
        y_grid: precomputed y grid from .npz.
        F_grid: precomputed F(y) values.
    Returns:
        Interpolated F(y) value(s).
    """
    return jnp.interp(y, y_grid, F_grid)


# ── Utility: load .npz and convert selected arrays to JAX ──────────────

def load_npz_jax(filepath, *keys):
    """Load a .npz file and convert specified arrays to jnp arrays.

    Args:
        filepath: path to .npz file.
        *keys: array names to convert. If empty, convert all.
    Returns:
        dict mapping key -> jnp.ndarray.
    """
    raw = np.load(filepath, allow_pickle=True)
    if keys:
        return {k: jnp.asarray(raw[k]) if not isinstance(raw[k], np.ndarray) or raw[k].dtype != object
                else raw[k]
                for k in keys}
    result = {}
    for k in raw.keys():
        val = raw[k]
        if isinstance(val, np.ndarray) and val.dtype != object:
            result[k] = jnp.asarray(val)
        else:
            result[k] = val
    return result
