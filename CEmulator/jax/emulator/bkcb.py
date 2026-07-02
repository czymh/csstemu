"""
JAX implementation of the nonlinear boost factor emulators (Bkcb).

Replaces Bkcb_gp, Bkcb_halofit_gp, Bkcb_hmcode2020_gp, Bkcb_lin2hmcode_gp
from CEmulator.emulator.Bkcb.

These predict B(k, z) = P_nl(k, z) / P_lin(k, z).
"""

import jax.numpy as jnp
import jax
import numpy as np
from functools import partial

from CEmulator.GaussianProcess.GaussianProcess import (
    GaussianProcessRegressor, Constant, RBF
)
from CEmulator.utils import data_path, zlists, cosmoNormLarge, MyStandardScaler
from ..jax_gp import build_rbf_gp_params, build_rbf_kernel_params, gp_predict_rbf
from ..jax_utils import interp2d_bilinear


class BkModel:
    """Pre-loaded JAX Bk emulator model (immutable).

    Handles all Bkcb variants: Bk_lin, Bk_halofit, Bk_hmcode2020, lgBk_lin2hmcode2020.
    """

    def __init__(self, *, gp_params, kernel_constants, kernel_length_scales,
                 pca_mean, pca_components,
                 param_mean, param_scale, coeff_mean, coeff_scale,
                 z_grid, k_grid, nz, is_log_space):
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
        self.is_log_space = is_log_space


def _load_bk_model(nvec, emunamestr, n_sample):
    """Common loader for all Bk variants.

    Args:
        nvec: number of PCA components.
        emunamestr: name of the .npz file (without path).
        n_sample: number of training samples (129 or 513).
    Returns:
        BkModel instance.
    """
    is_log_space = emunamestr[:2] == 'lg'
    kmax100 = is_log_space  # only lgBk_lin2hmcode2020 uses kmax100

    X_train = cosmoNormLarge[:n_sample, :]

    # Load klist
    if kmax100:
        klist = np.load(data_path + 'karr_kmax100.npy')
    else:
        klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
        kcut = 10.01
        ind = klist <= kcut
        klist = klist[ind]

    allsavedata = np.load(data_path + f'{emunamestr}.npz', allow_pickle=True)
    pca_data = allsavedata['pca_data']
    pca_mean = pca_data[0, :]
    pca_components = pca_data[1:, :]
    gprinfo = allsavedata['gprinfo']
    pkcoeff = allsavedata['Bcoeff']

    # Standard scaling
    coeff_ss = MyStandardScaler()
    param_ss = MyStandardScaler()
    pkcoeff_scaled = coeff_ss.fit_transform(pkcoeff)
    X_train_scaled = param_ss.fit_transform(X_train)

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

    gp_params = build_rbf_gp_params(gpr_list, X_train_scaled)
    constants, length_scales = build_rbf_kernel_params(gprinfo)

    model = BkModel(
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
        is_log_space=is_log_space,
    )
    return model


def load_bklin_model():
    """Load Bkcb_gp (Bk_lin, P_nl/P_lin ratio from simulations)."""
    return _load_bk_model(nvec=20, emunamestr='Bk_lin', n_sample=129)

def load_bkhalofit_model():
    """Load Bkcb_halofit_gp (Bk_halofit)."""
    return _load_bk_model(nvec=20, emunamestr='Bk_halofit', n_sample=129)

def load_bkhmcode2020_model():
    """Load Bkcb_hmcode2020_gp (Bk_hmcode2020)."""
    return _load_bk_model(nvec=20, emunamestr='Bk_hmcode2020', n_sample=129)

def load_bklin2hmcode_model():
    """Load Bkcb_lin2hmcode_gp (lgBk_lin2hmcode2020)."""
    return _load_bk_model(nvec=20, emunamestr='lgBk_lin2hmcode2020', n_sample=513)


# ── JAX prediction ─────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('model',))
def predict_bk(model, ncosmo, z, k):
    """Predict the boost factor B(k, z) = P_nl(k, z) / P_lin(k, z).

    Args:
        model: BkModel.
        ncosmo: (1, 8) JAX array, cosmology ALREADY [0,1]-normalized
                via NormCosmo (i.e., (raw - lo)/(hi - lo)).
        z: (nz_out,) redshift array.
        k: (nk_out,) wavenumber array [h/Mpc].
    Returns:
        (nz_out, nk_out) B(k, z) values.
    """
    # Standard-scale the [0,1]-normalized cosmology
    norm_cosmo = (ncosmo - model.param_mean) / model.param_scale

    # GP prediction for PCA coefficients
    coeffs = gp_predict_rbf(
        model.gp_params,
        model.kernel_constants,
        model.kernel_length_scales,
        norm_cosmo,
    )
    coeffs = coeffs.reshape(1, -1)

    # Inverse standard scaling
    coeffs = coeffs * model.coeff_scale + model.coeff_mean

    # Inverse PCA
    if model.is_log_space:
        values_flat = 10 ** (coeffs @ model.pca_components + model.pca_mean)
    else:
        values_flat = coeffs @ model.pca_components + model.pca_mean

    nk = len(model.k_grid)
    values_grid = values_flat.reshape(model.nz, nk)  # (nz, nk), z DESCENDING
    values_grid = values_grid[::-1, :]                # reverse to z ASCENDING

    # 2D interpolation: bilinear in z and k
    result = interp2d_bilinear(
        model.z_grid,
        model.k_grid,
        values_grid,
        z,
        k,
    )
    return result
