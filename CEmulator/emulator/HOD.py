"""
Halo Occupation Distribution (HOD) Models for CSST Emulator.

This module provides various HOD models for calculating galaxy occupation
statistics in dark matter halos. The base class defines the interface,
and specific implementations provide different parameterizations.

Available Models
----------------
- Zheng05HOD : Zheng et al. 2005 (ApJ, 633, 791) 5-parameter HOD model

Examples
--------
>>> from CEmulator.emulator.HOD import Zheng05HOD
>>> hod = Zheng05HOD()
>>> M = np.logspace(11, 15, 100)
>>> Ncen = hod.Ncen(M, logMmin=12.0, sigma_logM=0.5)
>>> Nsat = hod.Nsat(M, logMmin=12.0, logM0=12.5, logM1=13.5, alpha=1.0)
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from scipy.special import erf, erfc


class HODModel(ABC):
    """
    Abstract base class for Halo Occupation Distribution models.

    This class defines the interface that all HOD models must implement.
    Subclasses should provide specific parameterizations for central and
    satellite galaxy occupation functions.

    Attributes
    ----------
    name : str
        Name of the HOD model
    param_names : list
        List of parameter names required by this model

    Notes
    -----
    The HOD describes the probability distribution P(N|M) that a halo of
    mass M contains N galaxies of a given type.
    """

    name = "BaseHOD"
    param_names = []

    @abstractmethod
    def Ncen(self, M, **kwargs):
        """
        Mean number of central galaxies as a function of halo mass.

        Parameters
        ----------
        M : array_like
            Halo mass in Msun/h
        **kwargs : dict
            Model-specific parameters

        Returns
        -------
        array_like
            Mean central occupation number <N_cen|M>, values between 0 and 1

        Notes
        -----
        Central galaxies typically follow a softened step function:
        low-mass halos have <N_cen> ≈ 0, high-mass halos have <N_cen> ≈ 1.
        """
        pass

    @abstractmethod
    def Nsat(self, M, **kwargs):
        """
        Mean number of satellite galaxies as a function of halo mass.

        Parameters
        ----------
        M : array_like
            Halo mass in Msun/h
        **kwargs : dict
            Model-specific parameters

        Returns
        -------
        array_like
            Mean satellite occupation number <N_sat|M>

        Notes
        -----
        Satellite galaxies typically follow a power-law at high masses,
        with a cutoff at low masses where <N_sat> ≈ 0.
        """
        pass

    def Ntotal(self, M, **kwargs):
        """
        Total mean occupation number <N(M)> = N_cen + N_sat.

        Parameters
        ----------
        M : array_like
            Halo mass in Msun/h
        **kwargs : dict
            Model-specific parameters passed to Ncen and Nsat

        Returns
        -------
        array_like
            Total mean occupation number <N|M>

        Examples
        --------
        >>> hod = Zheng05HOD()
        >>> M = np.array([1e12, 1e13, 1e14])
        >>> Ntot = hod.Ntotal(M, logMmin=12.0, sigma_logM=0.5,
        ...                   logM0=12.5, logM1=13.5, alpha=1.0)
        """
        return self.Ncen(M, **kwargs) + self.Nsat(M, **kwargs)

    def check_params(self, **kwargs):
        """
        Check if all required parameters are provided.

        Parameters
        ----------
        **kwargs : dict
            Parameters to check

        Raises
        ------
        ValueError
            If required parameters are missing
        """
        missing = [p for p in self.param_names if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")


class Zheng05HOD(HODModel):
    r"""
    Zheng et al. (2005) 5-parameter HOD model.

    This is the standard "vanilla" HOD model widely used in the literature.
    It parameterizes the central occupation as a softened step function
    and the satellite occupation as a power law with a cutoff.

    Parameters
    ----------
    logMmin : float
        Logarithm of minimum halo mass for hosting a central galaxy
    sigma_logM : float
        Width of the transition region in log mass
    logM0 : float
        Logarithm of cutoff mass below which satellites are suppressed
    logM1 : float
        Logarithm of characteristic mass scale for satellites
    alpha : float
        Power-law slope of the satellite occupation at high mass

    Model Definition
    ----------------
    Central galaxies:

    .. math::
        \langle N_{\rm cen} | M \rangle = \frac{1}{2}
        \left[ 1 + {\rm erf}\left(\frac{\log M - \log M_{\rm min}}
        {\sigma_{\log M}}\right) \right]

    Satellite galaxies:

    .. math::
        \langle N_{\rm sat} | M \rangle = \langle N_{\rm cen} | M \rangle
        \times \left(\frac{M - M_0}{M_1}\right)^\alpha \Theta(M - M_0)

    where :math:`\Theta` is the Heaviside step function.

    References
    ----------
    .. [1] Zheng, Z. et al. 2005, ApJ, 633, 791
           "Theoretical Models of the Halo Occupation Distribution:
           Separating Central and Satellite Galaxies"

    Examples
    --------
    >>> hod = Zheng05HOD()
    >>> M = np.logspace(11, 15, 100)
    >>> params = {
    ...     'logMmin': 12.0,
    ...     'sigma_logM': 0.5,
    ...     'logM0': 12.5,
    ...     'logM1': 13.5,
    ...     'alpha': 1.0
    ... }
    >>> Ncen = hod.Ncen(M, **params)
    >>> Nsat = hod.Nsat(M, **params)
    """

    name = "Zheng05"
    param_names = ['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha']

    def Ncen(self, M, logMmin, sigma_logM, **kwargs):
        r"""
        Mean central occupation for Zheng+2005 model.

        Parameters
        ----------
        M : array_like
            Halo mass in Msun/h
        logMmin : float
            Logarithm of minimum halo mass for central galaxies
        sigma_logM : float
            Width of the transition region

        Returns
        -------
        array_like
            Mean central occupation <N_cen|M>

        Notes
        -----
        The central occupation transitions from 0 to 1 over a mass range
        characterized by sigma_logM. Halos with M << M_min have <N_cen> ≈ 0,
        while halos with M >> M_min have <N_cen> ≈ 1.
        """
        M = np.atleast_1d(M)
        return 0.5 * (1.0 + erf((np.log10(M) - logMmin) / sigma_logM))

    def Nsat(self, M, logMmin, logM0, logM1, alpha, **kwargs):
        r"""
        Mean satellite occupation for Zheng+2005 model.

        Parameters
        ----------
        M : array_like
            Halo mass in Msun/h
        logMmin : float
            Logarithm of minimum halo mass (passed to Ncen)
        logM0 : float
            Logarithm of cutoff mass for satellites
        logM1 : float
            Logarithm of characteristic mass scale
        alpha : float
            Power-law slope at high mass

        Returns
        -------
        array_like
            Mean satellite occupation <N_sat|M>

        Notes
        -----
        The satellite occupation is proportional to the central occupation
        (ensuring no satellites without a central) times a power law that
        activates above the cutoff mass M0.
        """
        M = np.atleast_1d(M)
        Nc = self.Ncen(M, logMmin, **{k: v for k, v in kwargs.items()
                                       if k != 'logMmin'})

        M0 = 10.0 ** logM0
        M1 = 10.0 ** logM1

        # Suppress satellites below cutoff mass
        Msat = np.where(M > M0, ((M - M0) / M1) ** alpha, 0.0)

        return Nc * Msat


class Zheng07HOD(HODModel):
    r"""
    Zheng+2007 HOD model with assembly bias and central incompleteness.

    This extends the Zheng05 model with three modifications:
    1. The satellite cutoff mass :math:`M_0` is tied to the central mass
       threshold via :math:`M_0 = \kappa \times 10^{\log M_{\rm cut}}`,
       reducing the free satellite parameter count by one.
    2. A "central incompleteness" parameter :math:`f_{\rm ic}` allows
       the central occupation to saturate below unity, representing a
       fraction of halos that never host a central.
    3. Assembly bias shifts the central mass scale and satellite mass
       scale based on halo concentration.

    Parameters
    ----------
    logMcut : float
        Logarithm of characteristic minimum mass for centrals
    sigma : float
        Width of the central occupation transition
    logM1 : float
        Logarithm of characteristic mass scale for satellites
    alpha : float
        Power-law slope of satellite occupation
    kappa : float
        Ratio :math:`M_0/M_{\rm cut}` for satellite cutoff
    fic : float
        Central incompleteness factor (0 < fic <= 1); central
        occupation saturates at this value at high mass
    A_cen : float, optional
        Assembly bias amplitude for centrals (default: 0)
    A_sat : float, optional
        Assembly bias amplitude for satellites (default: 0)

    Model Definition
    ----------------
    Central galaxies:

    .. math::
        \langle N_{\rm cen} | M \rangle = \frac{f_{\rm ic}}{2}
        {\rm erfc}\left(\frac{\log M_{\rm cut} - \log M}
        {\sqrt{2} \, \sigma}\right)
        = \frac{f_{\rm ic}}{2}
        \left[ 1 + {\rm erf}\left(\frac{\log M - \log M_{\rm cut}}
        {\sqrt{2} \, \sigma}\right) \right]

    Satellite galaxies:

    .. math::
        \langle N_{\rm sat} | M \rangle = \langle N_{\rm cen} | M \rangle
        \times \left(\frac{M - M_0}{M_1}\right)^\alpha \Theta(M - M_0)

    where :math:`M_0 = \kappa \, 10^{\log M_{\rm cut}}`,
    :math:`M_1 = 10^{\log M_1}`, and :math:`\Theta` is the Heaviside
    step function.

    Assembly bias (when concentration :math:`c` is provided):

    .. math::
        \log M_{\rm cut}^{\rm eff} = \log M_{\rm cut} + A_{\rm cen} (c - 1)
        \log M_1^{\rm eff} = \log M_1 + A_{\rm sat} (c - 1)

    References
    ----------
    .. [1] Zheng, Z. et al. 2007, ApJ, 667, 760
    .. [2] Zhai, Z. et al. 2023, arXiv:2306.06314

    Examples
    --------
    >>> hod = Zheng07HOD()
    >>> M = np.logspace(11, 15, 100)
    >>> params = {'logMcut': 12.5, 'sigma': 0.3, 'logM1': 13.5,
    ...           'alpha': 1.0, 'kappa': 1.0, 'fic': 1.0}
    >>> Ncen = hod.Ncen(M, **params)
    >>> Nsat = hod.Nsat(M, **params)
    >>> # With assembly bias
    >>> c = np.random.uniform(0.8, 1.2, len(M))
    >>> Ncen_ab = hod.Ncen(M, c=c, A_cen=0.1, **params)
    """

    name = "Zheng07"
    param_names = ['logMcut', 'sigma', 'logM1', 'alpha',
                   'kappa', 'fic', 'A_cen', 'A_sat']

    def __init__(self):
        """Initialize with default assembly bias parameters set to zero."""
        pass

    def Ncen(self, M, logMcut, sigma, fic=1.0, A_cen=0.0, c=None, **kwargs):
        r"""
        Mean central occupation for the Zheng+2007 model.

        Parameters
        ----------
        M : array_like
            Halo mass in Msun/h
        logMcut : float
            Logarithm of characteristic minimum mass for centrals
        sigma : float
            Width of the transition region
        fic : float, default=1.0
            Central incompleteness factor
        A_cen : float, default=0.0
            Assembly bias amplitude for centrals
        c : array_like, optional
            Halo concentration (normalized to mean at given mass)

        Returns
        -------
        array_like
            Mean central occupation <N_cen|M>
        """
        M = np.atleast_1d(M)

        logMcut_eff = logMcut
        if A_cen != 0.0 and c is not None:
            c = np.atleast_1d(c)
            logMcut_eff = logMcut + A_cen * (c - 1.0)

        return (0.5 * fic *
                erfc((logMcut_eff - np.log10(M)) / (np.sqrt(2.0) * sigma)))

    def Nsat(self, M, logMcut, sigma, logM1, alpha, kappa, fic=1.0,
             A_cen=0.0, A_sat=0.0, c=None, **kwargs):
        r"""
        Mean satellite occupation for the Zheng+2007 model.

        Parameters
        ----------
        M : array_like
            Halo mass in Msun/h
        logMcut : float
            Logarithm of characteristic minimum mass for centrals
        sigma : float
            Width of central transition
        logM1 : float
            Logarithm of characteristic satellite mass scale
        alpha : float
            Power-law slope at high mass
        kappa : float
            Ratio M0/Mcut defining the satellite cutoff
        fic : float, default=1.0
            Central incompleteness factor
        A_cen : float, default=0.0
            Assembly bias amplitude for centrals
        A_sat : float, default=0.0
            Assembly bias amplitude for satellites
        c : array_like, optional
            Halo concentration

        Returns
        -------
        array_like
            Mean satellite occupation <N_sat|M>, same shape as M
        """
        M = np.atleast_1d(M)

        # Central occupation
        Nc = self.Ncen(M, logMcut, sigma, fic=fic, A_cen=A_cen, c=c)

        # Effective satellite mass scale with assembly bias
        logM1_eff = logM1
        if A_sat != 0.0 and c is not None:
            c = np.atleast_1d(c)
            logM1_eff = logM1 + A_sat * (c - 1.0)

        Mcut = 10.0 ** logMcut
        M0   = kappa * Mcut
        M1   = 10.0 ** logM1_eff

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Msat = np.where(M > M0, ((M - M0) / M1) ** alpha, 0.0)

        return Nc * Msat


def get_hod_model(model_name='Zheng05'):
    """
    Factory function to get HOD model by name.

    Parameters
    ----------
    model_name : str
        Name of the HOD model. Options: 'Zheng05', 'Zheng07'
        - 'Zheng05': Zheng et al. 2005 5-parameter model (logMmin, sigma_logM, logM0, logM1, alpha)
        - 'Zheng07': Zheng et al. 2007 model (logMcut, sigma, logM1, alpha, kappa, fic, A_cen, A_sat)

    Returns
    -------
    HODModel
        Instance of the requested HOD model

    Raises
    ------
    ValueError
        If the requested model is not available

    Examples
    --------
    >>> hod = get_hod_model('Zheng05')
    >>> print(hod.name)
    Zheng05
    """
    models = {
        'Zheng05': Zheng05HOD,
        'Zheng07': Zheng07HOD,
    }

    if model_name not in models:
        raise ValueError(f"Unknown HOD model '{model_name}'. "
                         f"Available models: {list(models.keys())}")

    return models[model_name]()
