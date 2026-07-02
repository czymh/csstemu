"""
JAX-compatible Gaussian Process prediction for pre-trained emulator models.

The original code instantiates one GaussianProcessRegressor per PCA component
(stored in np.zeros(nvec, dtype=object)). This module replaces that with
batched pure functions using vmap.

Key insight: only predict() is used, never return_std or return_cov.
Prediction reduces to:  y = K(X_test, X_train) @ alpha_ + denormalization.
"""

import jax.numpy as jnp
import jax
import numpy as np
from .jax_utils import cdist_sqeuclidean, cdist_euclidean


# ── JAX kernel functions ───────────────────────────────────────────────

def rbf_kernel(X1, X2, length_scale, constant):
    """RBF (squared exponential) kernel.

    K(x1, x2) = constant * exp(-0.5 * |x1/l - x2/l|^2)

    Args:
        X1: (m, d) array.
        X2: (n, d) array.
        length_scale: scalar or (d,) array.
        constant: scalar.
    Returns:
        (m, n) kernel matrix.
    """
    dists = cdist_sqeuclidean(X1 / length_scale, X2 / length_scale)
    return constant * jnp.exp(-0.5 * dists)


def matern_kernel(X1, X2, length_scale, nu, constant):
    """Matern kernel with nu in {0.5, 1.5, 2.5}.

    Args:
        X1: (m, d) array.
        X2: (n, d) array.
        length_scale: scalar or (d,) array.
        nu: smoothness parameter (0.5, 1.5, or 2.5).
        constant: scalar.
    Returns:
        (m, n) kernel matrix.
    """
    dists = cdist_euclidean(X1 / length_scale, X2 / length_scale)

    if nu == 0.5:
        K = jnp.exp(-dists)
    elif nu == 1.5:
        K = dists * jnp.sqrt(3.0)
        K = (1.0 + K) * jnp.exp(-K)
    elif nu == 2.5:
        K = dists * jnp.sqrt(5.0)
        K = (1.0 + K + K ** 2 / 3.0) * jnp.exp(-K)
    else:
        raise ValueError(f"Matern nu={nu} not supported. Only 0.5, 1.5, 2.5 are.")

    return constant * K


# ── Batched GP prediction ──────────────────────────────────────────────

def predict_one_rbf(X_test, X_train, length_scale, constant, alpha_vec,
                    y_mean, y_std):
    """Single-component GP prediction with RBF kernel.

    Args:
        X_test: (n_test, d) query points.
        X_train: (n_train, d) training points.
        length_scale: scalar or (d,) RBF length scale.
        constant: scalar constant multiplier.
        alpha_vec: (n_train,) precomputed alpha = K^{-1} @ y.
        y_mean: scalar, training y mean.
        y_std: scalar, training y std.
    Returns:
        (n_test,) predicted values.
    """
    Kstar = rbf_kernel(X_test, X_train, length_scale, constant)  # (n_test, n_train)
    raw = Kstar @ alpha_vec  # (n_test,)
    return raw * y_std + y_mean


def predict_one_matern(X_test, X_train, length_scale, nu, constant, alpha_vec,
                       y_mean, y_std):
    """Single-component GP prediction with Matern kernel."""
    Kstar = matern_kernel(X_test, X_train, length_scale, nu, constant)
    raw = Kstar @ alpha_vec
    return raw * y_std + y_mean


# vmap over components for batch prediction
predict_batch_rbf = jax.vmap(predict_one_rbf,
    in_axes=(None, None, 0, 0, 0, 0, 0))
predict_batch_matern = jax.vmap(predict_one_matern,
    in_axes=(None, None, 0, 0, 0, 0, 0, 0))


# ── GP parameter builders (numpy side, run once at load time) ──────────

def build_rbf_gp_params(gpr_list, X_train_scaled):
    """Extract JAX-compatible parameters from trained numpy RBF GPR objects.

    Args:
        gpr_list: list of trained GaussianProcessRegressor objects.
        X_train_scaled: (n_train, n_params) scaled training cosmologies.
    Returns:
        dict with keys: X_train, alpha_batch, y_mean_batch, y_std_batch
    """
    nvec = len(gpr_list)
    n_train = len(X_train_scaled)
    alpha_batch = np.zeros((nvec, n_train))
    y_mean = np.zeros(nvec)
    y_std = np.zeros(nvec)
    for i, gpr in enumerate(gpr_list):
        alpha_batch[i] = gpr.alpha_
        y_mean[i] = gpr._y_train_mean
        y_std[i] = gpr._y_train_std
    return {
        'X_train': jnp.asarray(X_train_scaled),
        'alpha_batch': jnp.asarray(alpha_batch),
        'y_mean_batch': jnp.asarray(y_mean),
        'y_std_batch': jnp.asarray(y_std),
    }


def build_rbf_kernel_params(gprinfo_list):
    """Extract RBF kernel parameters from gprinfo dicts.

    Args:
        gprinfo_list: list of dicts with keys 'k1__constant_value' and 'k2__length_scale'.
    Returns:
        (constants, length_scales) as jnp arrays.
    """
    nvec = len(gprinfo_list)
    constants = np.array([gprinfo_list[i]['k1__constant_value'] for i in range(nvec)])
    length_scales = np.array([gprinfo_list[i]['k2__length_scale'] for i in range(nvec)])
    return jnp.asarray(constants), jnp.asarray(length_scales)


def build_matern_kernel_params(gprinfo_list):
    """Extract Matern kernel parameters from gprinfo dicts.

    Returns:
        (constants, length_scales, nus) as jnp arrays.
    """
    nvec = len(gprinfo_list)
    constants = np.array([gprinfo_list[i]['k1__constant_value'] for i in range(nvec)])
    length_scales = np.array([gprinfo_list[i]['k2__length_scale'] for i in range(nvec)])
    nus = np.array([gprinfo_list[i]['k2__nu'] for i in range(nvec)])
    return jnp.asarray(constants), jnp.asarray(length_scales), jnp.asarray(nus)


# ── JIT-compiled prediction pipelines ──────────────────────────────────

@jax.jit
def gp_predict_rbf(gp_params, kernel_constants, kernel_length_scales,
                   X_test):
    """Predict all PCA coefficients for a cosmology using RBF-kernel GPs.

    Args:
        gp_params: dict from build_rbf_gp_params.
        kernel_constants: (nvec,) constant multipliers.
        kernel_length_scales: (nvec,) RBF length scales.
        X_test: (n_test, d) normalized cosmology(ies), usually (1, d).
    Returns:
        (nvec,) PCA coefficients.
    """
    return predict_batch_rbf(
        X_test,
        gp_params['X_train'],
        kernel_length_scales,
        kernel_constants,
        gp_params['alpha_batch'],
        gp_params['y_mean_batch'],
        gp_params['y_std_batch'],
    ).squeeze()  # remove extra dim
