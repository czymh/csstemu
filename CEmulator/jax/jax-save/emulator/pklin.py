"""
JAX implementation of the linear matter power spectrum emulator (PkLin).

Replaces PkcbLin_gp and Pknn_cbLin_gp from CEmulator.emulator.PkLin.
"""

import jax.numpy as jnp
import jax
import numpy as np
from functools import partial

from CEmulator.GaussianProcess.GaussianProcess import (
    GaussianProcessRegressor, Constant, RBF
)
from CEmulator.utils import (
    data_path, zlists, cosmoNormLarge, MyStandardScaler
)
from ..jax_gp import build_rbf_gp_params, build_rbf_kernel_params, gp_predict_rbf
from ..jax_utils import interp2d_bilinear


# ── Data structures ────────────────────────────────────────────────────

class PkLinModel:
    """Pre-loaded JAX PkLin emulator model (immutable)."""

    def __init__(self, *, gp_params, kernel_constants, kernel_length_scales,
                 pca_mean, pca_components,
                 param_mean, param_scale, coeff_mean, coeff_scale,
                 z_grid, k_grid, nz):
        self.gp_params = gp_params
        self.kernel_constants = kernel_constants
        self.kernel_length_scales = kernel_length_scales
        self.pca_mean = pca_mean
        self.pca_components = pca_components
        self.param_mean = param_mean
        self.param_scale = param_scale
        self.coeff_mean = coeff_mean
        self.coeff_scale = coeff_scale
        self.z_grid = z_grid
        self.k_grid = k_grid
        self.nz = nz


# ── Model loading (numpy, run once at init) ────────────────────────────

def _build_gp_for_pklin(nvec, X_train, pkcoeff, gprinfo, norm_before_gp):
    """Build numpy GPR objects for all components, extract JAX params."""
    if norm_before_gp:
        coeff_ss = MyStandardScaler()
        param_ss = MyStandardScaler()
        pkcoeff_scaled = coeff_ss.fit_transform(pkcoeff)
        X_train_scaled = param_ss.fit_transform(X_train)
    else:
        coeff_ss = None
        param_ss = None
        pkcoeff_scaled = pkcoeff
        X_train_scaled = X_train

    gpr_list = []
    for ivec in range(nvec):
        k1 = Constant(gprinfo[ivec]['k1__constant_value'])
        k2 = RBF(gprinfo[ivec]['k2__length_scale'])
        kernel = k1 * k2
        gpr = GaussianProcessRegressor(
            X_train_scaled, pkcoeff_scaled[:, ivec],
            kernel=kernel, alpha=1e-10, normalize_y=True
        )
        gpr_list.append(gpr)

    return gpr_list, X_train_scaled, coeff_ss, param_ss


def load_pkcblin_model(verbose=False):
    """Load the pre-trained PkcbLin GP emulator and return a JAX model.

    This is the linear cb-power spectrum (Nncdm=1, 'single' neutrino split).
    """
    nvec = 20
    if verbose:
        print('Loading PkcbLin JAX emulator...')

    indexs = np.arange(513)
    X_train = cosmoNormLarge[indexs, :]
    allsavedata = np.load(data_path + 'lgpkLin.npz', allow_pickle=True)
    pca_data = allsavedata['pca_data']
    pca_mean = pca_data[0, :]
    pca_components = pca_data[1:, :]
    gprinfo = allsavedata['gprinfo']
    pkcoeff = allsavedata['Bcoeff']
    klist = np.load(data_path + 'karr_kmax100.npy')

    norm_before_gp = True
    gpr_list, X_train_scaled, coeff_ss, param_ss = _build_gp_for_pklin(
        nvec, X_train, pkcoeff, gprinfo, norm_before_gp
    )

    gp_params = build_rbf_gp_params(gpr_list, X_train_scaled)
    constants, length_scales = build_rbf_kernel_params(gprinfo)

    model = PkLinModel(
        gp_params=gp_params,
        kernel_constants=constants,
        kernel_length_scales=length_scales,
        pca_mean=jnp.asarray(pca_mean),
        pca_components=jnp.asarray(pca_components),
        param_mean=jnp.asarray(param_ss.mean_),
        param_scale=jnp.asarray(param_ss.scale_),
        coeff_mean=jnp.asarray(coeff_ss.mean_),
        coeff_scale=jnp.asarray(coeff_ss.scale_),
        z_grid=jnp.array(zlists[::-1]),  # ascending: 0, 0.1, ...
        k_grid=jnp.asarray(klist),
        nz=len(zlists),
    )
    return model


def load_pknn_cblin_model(verbose=False):
    """Load the pre-trained Pknn_cbLin GP emulator (Nncdm=1).

    Uses 512 training points (removes cosmology c0001 which has no massive neutrino).
    """
    nvec = 10
    if verbose:
        print('Loading Pknn_cbLin JAX emulator...')

    indexs = np.arange(513)
    indexs = np.delete(indexs, 1)  # remove c0001
    X_train = cosmoNormLarge[indexs, :]
    allsavedata = np.load(data_path + 'lgpkLin_nn_cb.npz', allow_pickle=True)
    pca_data = allsavedata['pca_data']
    pca_mean = pca_data[0, :]
    pca_components = pca_data[1:, :]
    gprinfo = allsavedata['gprinfo']
    pkcoeff = allsavedata['Bcoeff']
    klist = np.load(data_path + 'karr_kmax100.npy')

    norm_before_gp = True
    gpr_list, X_train_scaled, coeff_ss, param_ss = _build_gp_for_pklin(
        nvec, X_train, pkcoeff, gprinfo, norm_before_gp
    )

    gp_params = build_rbf_gp_params(gpr_list, X_train_scaled)
    constants, length_scales = build_rbf_kernel_params(gprinfo)

    model = PkLinModel(
        gp_params=gp_params,
        kernel_constants=constants,
        kernel_length_scales=length_scales,
        pca_mean=jnp.asarray(pca_mean),
        pca_components=jnp.asarray(pca_components),
        param_mean=jnp.asarray(param_ss.mean_),
        param_scale=jnp.asarray(param_ss.scale_),
        coeff_mean=jnp.asarray(coeff_ss.mean_),
        coeff_scale=jnp.asarray(coeff_ss.scale_),
        z_grid=jnp.array(zlists[::-1]),
        k_grid=jnp.asarray(klist),
        nz=len(zlists),
    )
    return model


# ── JAX prediction ─────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('model',))
def predict_pk(model, ncosmo, z, k):
    """Predict linear cb-power spectrum at (z, k).

    Args:
        model: PkLinModel.
        ncosmo: (1, 8) JAX array, cosmology ALREADY normalized to [0,1] range
                via NormCosmo (i.e., (raw - lo)/(hi - lo) for each param).
        z: (nz_out,) redshift array.
        k: (nk_out,) wavenumber array [h/Mpc].
    Returns:
        (nz_out, nk_out) P_cb_lin(k, z) in [Mpc/h]^3.
    """
    # Standard-scale the [0,1]-normalized cosmology
    norm_cosmo = (ncosmo - model.param_mean) / model.param_scale

    # GP prediction for PCA coefficients
    coeffs = gp_predict_rbf(
        model.gp_params,
        model.kernel_constants,
        model.kernel_length_scales,
        norm_cosmo,
    )  # (nvec,)
    coeffs = coeffs.reshape(1, -1)  # (1, nvec)

    # Inverse standard scaling
    coeffs = coeffs * model.coeff_scale + model.coeff_mean  # (1, nvec)

    # Inverse PCA
    log_pk_flat = coeffs @ model.pca_components + model.pca_mean  # (1, n_points)
    nk = len(model.k_grid)
    log_pk_grid = log_pk_flat.reshape(model.nz, nk)  # (nz, nk), z DESCENDING (3.0 ... 0.0)
    log_pk_grid = log_pk_grid[::-1, :]               # reverse to z ASCENDING for interpolation

    # 2D interpolation: bilinear in z and log10(k)
    log_pk_interp = interp2d_bilinear(
        model.z_grid,
        jnp.log10(model.k_grid),
        log_pk_grid,
        z,
        jnp.log10(k),
    )  # (nz_out, nk_out)

    return 10 ** log_pk_interp
