"""
Top-level JAX emulator for cosmological power spectra.

Provides a class-based API that loads all pre-trained GP models and
exposes JIT-compiled prediction functions. Users can apply jax.grad
to obtain derivatives of any predicted statistic with respect to
the 8 cosmological parameters.

Only supports neutrino_mass_split='single' (Nncdm=1).

Usage:
    emu = JAXEmulator()
    pk_lin = emu.get_pklin(cosmo_8d, z, k)
    grad_pk = jax.grad(lambda c: emu.get_pklin(c, z, k).sum())(cosmo_8d)
"""

import jax.numpy as jnp
import jax
import numpy as np
from functools import partial

from .emulator.pklin import load_pkcblin_model, load_pknn_cblin_model, predict_pk
from .emulator.bkcb import (
    load_bklin_model, load_bkhalofit_model,
    load_bkhmcode2020_model, load_bklin2hmcode_model,
    predict_bk,
)
from .jax_cosmology import get_Ez, precompute_chi_grid, comoving_distance_precomputed
from .jax_utils import norm_cosmo as _norm_cosmo_jax


# Parameter names and limits (same as original utils.py)
PARAM_NAMES = ['Omegab', 'Omegam', 'H0', 'ns', 'A', 'w', 'wa', 'mnu']
PARAM_LIMITS = {
    'Omegab': [0.04, 0.06],
    'Omegam': [0.24, 0.40],
    'H0': [60, 80],
    'ns': [0.92, 1.00],
    'A': [1.7, 2.5],
    'w': [-1.3, -0.7],
    'wa': [-0.5, 0.5],
    'mnu': [0, 0.3],
}

# Default emulator redshifts
EMULATOR_ZLISTS = [3.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.8, 0.5, 0.25, 0.1, 0.0]


class JAXEmulator:
    """JAX-based CSST cosmological emulator.

    Loads all pre-trained GP models at construction time (numpy I/O),
    then runs pure JAX computations for prediction and differentiation.
    """

    param_names = PARAM_NAMES
    param_limits = PARAM_LIMITS
    zlists = EMULATOR_ZLISTS

    def __init__(self, verbose=False):
        self.verbose = verbose

        # Load sub-models (numpy I/O)
        self._model_pkcblin = load_pkcblin_model(verbose=verbose)
        self._model_pknn_cblin = load_pknn_cblin_model(verbose=verbose)
        # Boost-factor (Bk) emulators — each predicts P_nl_sim / P_base
        self._model_bk_lin = load_bklin_model()              # Bk = P_nl / P_lin
        self._model_bk_hmcode2020 = load_bkhmcode2020_model() # Bk = P_nl / P_hmcode2020
        self._model_bk_halofit = load_bkhalofit_model()       # Bk = P_nl / P_halofit
        # HMCODE-2020 helper: predicts P_hmcode2020 / P_lin (used as denominator)
        self._model_bk_lin2hmcode = load_bklin2hmcode_model()

    # ── Normalization helper ───────────────────────────────────────

    def _normalize(self, cosmo_8d):
        """Convert raw cosmology (8,) to [0,1]-normalized (1, 8).

        Used internally before passing to predict_pk / predict_bk.
        """
        cosmo_2d = jnp.atleast_2d(cosmo_8d)
        return _norm_cosmo_jax(cosmo_2d, self.param_names, self.param_limits)

    # ── Cosmology helpers ──────────────────────────────────────────

    def get_Ez(self, cosmo_8d, z):
        """Hubble parameter E(z) = H(z)/H0."""
        c = jnp.atleast_1d(cosmo_8d.squeeze())
        return get_Ez(z, c[1], c[0], c[2] / 100.0, c[5], c[6], c[7])

    def comoving_distance(self, cosmo_8d, z):
        """Comoving distance in Mpc (numpy precomputation + JAX interp)."""
        c = np.atleast_1d(np.asarray(cosmo_8d).squeeze())
        cosmo_dict = {
            'Omegab': c[0], 'Omegam': c[1], 'H0': c[2],
            'ns': c[3], 'A': c[4], 'w': c[5], 'wa': c[6], 'mnu': c[7],
            'z_ref': np.max(np.atleast_1d(np.asarray(z))),
        }
        a_grid, chi_grid, chi_fac = precompute_chi_grid(cosmo_dict)
        return comoving_distance_precomputed(
            z, jnp.asarray(a_grid), jnp.asarray(chi_grid), chi_fac)

    # ── Power spectra ─────────────────────────────────────────────

    def get_pkcblin(self, cosmo_8d, z, k):
        """Linear cb-power spectrum P_cb_lin(k, z)."""
        ncosmo = self._normalize(cosmo_8d)
        return predict_pk(self._model_pkcblin, ncosmo, z, k)

    def get_pknn_cblin(self, cosmo_8d, z, k):
        """Non-neutrino/neutrino power ratio (multiply by P_cb_lin to get P_nn)."""
        ncosmo = self._normalize(cosmo_8d)
        return predict_pk(self._model_pknn_cblin, ncosmo, z, k)

    @partial(jax.jit, static_argnames=('self', 'Pcb'))
    def get_pklin(self, cosmo_8d, z, k, Pcb=False):
        """Total linear matter power spectrum.

        P_tot = f_cb^2 * P_cb + f_nu^2 * P_nn + 2*f_cb*f_nu*sqrt(P_cb*P_nn)
        """
        ncosmo = self._normalize(cosmo_8d)
        pkcblin = predict_pk(self._model_pkcblin, ncosmo, z, k)

        if Pcb:
            return pkcblin

        pknnlin = predict_pk(self._model_pknn_cblin, ncosmo, z, k) * pkcblin

        c = cosmo_8d.squeeze()
        omegam = c[1]
        h0 = c[2] / 100.0
        mnu = c[7]
        omeganu = mnu / 93.14 / (h0 * h0)
        omegaM = omegam + omeganu
        fcb2M = omegam / omegaM
        fnu2M = omeganu / omegaM
        pk_total = (
            fcb2M ** 2 * pkcblin
            + fnu2M ** 2 * pknnlin
            + 2.0 * fcb2M * fnu2M * jnp.sqrt(pkcblin * pknnlin)
        )
        return jnp.where(omeganu < 1e-12, pkcblin, pk_total)

    @partial(jax.jit, static_argnames=('self', 'Pcb'))
    def get_pkhmcode2020(self, cosmo_8d, z, k, Pcb=False):
        """HMCODE-2020 model nonlinear power spectrum.

        P_hmcode2020(k,z) = P_lin(k,z) × B_lin2hmcode(k,z)

        This is the HMCODE-2020 prediction itself, used as the *denominator*
        in get_pknl(nltype='hmcode2020').  Equivalent to numpy
        CBaseEmulator.get_pkHMCODE2020(lintype='Emulator').
        """
        pklin = self.get_pklin(cosmo_8d, z, k, Pcb=Pcb)
        ncosmo = self._normalize(cosmo_8d)
        bk = predict_bk(self._model_bk_lin2hmcode, ncosmo, z, k)
        return pklin * bk

    def get_pknl(self, cosmo_8d, z, k, nltype='hmcode2020', Pcb=False):
        """Simulation-level nonlinear matter power spectrum.

        The emulator predicts the *ratio*  Bk(k,z) = P_nl_sim / P_base
        where P_base is a nonlinear model chosen by `nltype`.

        The final result is  P_nl = P_base(k,z) × Bk(k,z).

        Args:
            nltype: 'linear', 'hmcode2020', or 'halofit'.
                Controls which P_base is used as denominator.
                - 'linear':     P_nl = P_lin  ×  B_lin  (full boost from linear)
                - 'hmcode2020':  P_nl = P_hmcode2020 × B_hmcode2020
                - 'halofit':     P_nl = P_halofit × B_halofit
                  (NOT YET SUPPORTED — raises NotImplementedError)
            Pcb: if True, return cb-only spectrum (excludes neutrino component).
        """
        ncosmo = self._normalize(cosmo_8d)

        if nltype == 'linear':
            # Bk_lin = P_nl_sim / P_lin
            bk = predict_bk(self._model_bk_lin, ncosmo, z, k)
            p_base = self.get_pklin(cosmo_8d, z, k, Pcb=Pcb)
            return p_base * bk

        elif nltype == 'hmcode2020':
            # Bk_hmcode2020 = P_nl_sim / P_hmcode2020
            bk = predict_bk(self._model_bk_hmcode2020, ncosmo, z, k)
            # P_base = P_hmcode2020 = P_lin_cb × B_lin2hmcode
            p_base = self.get_pkhmcode2020(cosmo_8d, z, k, Pcb=Pcb)
            return p_base * bk

        elif nltype == 'halofit':
            raise NotImplementedError(
                "nltype='halofit' is not yet supported in the JAX emulator. "
                "The halofit P(k) requires an iterative root-finding solver "
                "(Takahashi formula) that has not been ported to JAX. "
                "Use nltype='linear' or nltype='hmcode2020' instead."
            )

        else:
            raise ValueError(f"Unknown nltype: {nltype}")
