import numpy as np
from scipy.interpolate import interp1d
from .utils import data_path

# Load pre-computed Fermi-Dirac integral interpolation table
_FERMI_DATA = np.load(data_path + 'fermi_dirac_F_interp.npz')
_Y_GRID = _FERMI_DATA['y_grid']
_F_GRID = _FERMI_DATA['F_grid']
_F_PRIME_GRID = _FERMI_DATA['F_prime_grid']
_F_INTERPOLATOR = interp1d(_Y_GRID, _F_GRID, kind='cubic', bounds_error=False,
                            fill_value=(0.0, float('inf')))
_F_PRIME_INTERPOLATOR = interp1d(_Y_GRID, _F_PRIME_GRID, kind='cubic',
                                  bounds_error=False, fill_value=(0.0, float('inf')))

# Critical density at z=0 in h^2 Msun / Mpc^3
RHO_CRIT_0 = 2.77536627e11

class Cosmology:

    def __init__(self, verbose=False, neutrino_mass_split='single'):
        '''
        Initialize the CSST cosmology class.
        
        Args:
            verbose : bool, whether to output the running information
            neutrino_mass_split : string, 'single' or 'degenerate', the neutrino mass split type.
        
        '''
        ## physical constants
        self.sigma_B   = 5.670373e-5; 
        self.vel_light = 2.99792458e10; # cm/s
        self.G_const   = 6.672e-8; 
        self.h0_units  = 3.2407789e-18; # h/sec
        self.kB        = 8.617333262145e-5   ## boltzman in eV/K
        self.rho_crit  = RHO_CRIT_0
        self.verbose   = verbose
        self.neutrino_mass_split = neutrino_mass_split
    
              
    def set_cosmos(self, cosmologies):
        '''
        set the cosmologies for the cosmology class
        '''
        Neff = 3.046
        self.mnu     = cosmologies['mnu'] # eV
        # number of ultra-relativistic neutrinos 
        # Now only support 3.046 or 2.0328 (0 or 1 massive neutrino)
        if   self.mnu == 0.0:
            self.Nur     = 3.046
            self.Nncdm   = 0
        else:
            if self.neutrino_mass_split == 'degenerate':
                self.Nur     = 0.0064
                self.Nncdm   = 3
                self.mnus    = np.array([self.mnu/3]*3)
            elif self.neutrino_mass_split =='single':
                self.Nur     = 2.0328
                self.Nncdm   = 1
                self.mnus    = np.array([self.mnu])
            else:
                raise ValueError('The neutrino_mass_split = %s is not supported yet.'%self.neutrino_mass_split)
        self.h0       = cosmologies['H0'] / 100
        self.Omeganu  = self.mnu/93.14/self.h0/self.h0
        # total matter without massive neutrinos
        # Only baryons and cold dark matter
        self.Omegam   = cosmologies['Omegam']  
        self.Omegab   = cosmologies['Omegab']
        self.Omegac   = cosmologies['Omegam'] - cosmologies['Omegab']
        self.w0       = cosmologies['w']
        self.wa       = cosmologies['wa']
        self.ns       = cosmologies['ns']
        self.As       = cosmologies['A'] * 1e-9
        self.Omegag   = 2.4721231034210734e-5/self.h0/self.h0 # photon radiation
        self.Gamma_nu = (4/11)**(1/3)
        self.f_nnu    = 7/8*(self.Gamma_nu**4)*self.Nur # neutrino radiation
        self.OmegaR   = self.Omegag*(1+self.f_nnu) # Total radiation
        self.OmegaL   = 1 - self.Omegam - self.Omeganu - self.OmegaR
        self.OmegaM   = self.Omegam + self.Omeganu
        self.TCMB     = 2.7255
        ##### Fix the curvature to be zero
        self.Omegak   = 0.0
        ## fast comoving_distance
        self._chi_interp = None
        self._a_min_cached = None
        self._neutrino_flag = None
        self._chi_fac = 1e-5 * self.vel_light / self.h0 / 100
        
       
    def _F_y(self, y, Fid=0):
        '''
        Get the Fermi-Dirac integral F(y) or F'(y) using pre-computed interpolation.

        Parameters
        ----------
        y : float
            The argument y = m_nu / ((1+z) * kB * T_nu)
        Fid : int
            0 for F(y), 1 for F'(y) = dF/dy

        Returns
        -------
        float
            The value of F(y) or F'(y)
        '''
        if Fid == 0:
            return float(_F_INTERPOLATOR(float(y)))
        elif Fid == 1:
            return float(_F_PRIME_INTERPOLATOR(float(y)))
        else:
            raise ValueError("Fid must be 0 or 1.")
        
    def _Omeganu_TimesHubbleSquare(self, z):
        z = np.atleast_1d(z)
        fac = 15/np.pi**4 * (self.Gamma_nu**4) * self.Omegag * (1+z)**4
        T_nu = self.Gamma_nu * self.TCMB
        if self.mnu != 0.0:
            y_vals = self.mnus[None,:] / (1+z[:,None]) / self.kB / T_nu
            # F_sumNu = np.asarray([self._F_y(y, Fid=0) for y in y_vals])
            F_sumNu = np.sum(_F_INTERPOLATOR(y_vals), axis=1)   # one time interpolation for all y values
        else:
            F_sumNu = 0
        return fac * F_sumNu

    def get_Ez(self, z, neutrino_matter_like=False):
        '''
        Get the normalized Hubble parameter H(z) at redshift z.

        Args:
            z : float or array-like, redshift
        Returns:
            array-like : array of shape (len(z)), normalized Hubble parameter H(z)/H0
        '''
        z = np.atleast_1d(z)
        de_exp = np.exp(3*((1/(1+z)-1)*self.wa - (1+self.w0+self.wa)*np.log(1/(1+z))))
        if neutrino_matter_like:
            return np.sqrt(self.OmegaM*(1+z)**3 +
                           self.Omegak*(1+z)**2 +
                           self.OmegaR*(1+z)**4 +
                           self.OmegaL*de_exp)
        else:
            return np.sqrt(self.Omegam*(1+z)**3 +
                        self._Omeganu_TimesHubbleSquare(z) +
                        self.Omegak*(1+z)**2 +
                        self.OmegaR*(1+z)**4 +
                        self.OmegaL*de_exp)
    
    def get_Omegam(self, z):
        '''
        Get the total matter density without massive neutrinos at redshift z.
        
        Args:
            z : float or array-like, redshift
        Returns:
            array-like : 2D array of shape (len(z)), total matter density without massive neutrinos
        '''
        z = np.atleast_1d(z)
        Ez = self.get_Ez(z)  # vectorized call to get_Ez for all z values
        return self.Omegam * (1+z)**3 / Ez**2
    
    def get_OmegaM(self, z):
        '''
        Get the total matter density at redshift z.
        
        Args:
            z : float or array-like, redshift
        Returns:
            array-like : 2D array of shape (len(z)), total matter density
        '''
        z = np.atleast_1d(z)
        Ez = self.get_Ez(z)  # vectorized call to get_Ez for all z values
        return self.OmegaM * (1+z)**3 / Ez**2 
    
    def get_OmegaL(self, z):
        '''
        Get the dark energy density at redshift z.
        
        Args:
            z : float or array-like, redshift
        Returns:
            array-like : 2D array of shape (len(z)), dark energy density
        '''
        z = np.atleast_1d(z)
        Ez = self.get_Ez(z)  # vectorized call to get_Ez for all z values
        de_exp = np.exp(3*((1/(1+z)-1)*self.wa-(1 + self.w0 + self.wa)*np.log(1/(1+z))))
        return self.OmegaL * de_exp / Ez**2 
    
    def comoving_distance(self, z, neutrino_matter_like=False):
        '''
        Get the comoving distance at redshift z.

        Uses a single trapezoidal integration over a shared fine grid
        so that the cost scales with one get_Ez call, not N individual quad calls.

        Args:
            z : float or array-like, redshift
        Returns:
            array-like : array of shape (len(z)), comoving distance in Mpc
        '''
        z = np.atleast_1d(z)
        # Build a fine, sorted grid of a from a_min to 1
        aarr = 1 / (1 + z)
        a_min = np.min(aarr)

        # Handle edge case: z=0 (a=1)
        if a_min >= 1.0:
            return np.zeros_like(z, dtype=float)

        if (self._chi_interp is None) or (a_min < self._a_min_cached) or (self._neutrino_flag != neutrino_matter_like):
            a_low = min(a_min, 1e-5)
            # ~2048 points gives sub-percent trapezoidal error for smooth integrands
            n_grid = 2048
            a_grid = np.logspace(np.log10(a_low), 0, n_grid)
            # Evaluate 1/(E(a) * a^2) on the shared grid — one vectorized call
            z_grid = 1 / a_grid - 1
            inv_Ez_a2 = 1 / (self.get_Ez(z_grid, neutrino_matter_like) * a_grid**2)
            # Trapezoidal cumulative integral from each a_grid point to a_grid[-1]=1
            da = np.diff(a_grid)
            avg = 0.5 * (inv_Ez_a2[:-1] + inv_Ez_a2[1:])
            # chi(a_i) = sum_{j=i}^{N-2} avg[j] * da[j]
            cum = np.cumsum(avg[::-1] * da[::-1])[::-1]
            # Prepend chi(a_N) = 0
            chi_at_grid = np.concatenate([cum, [0.0]])
            # Interpolate to the requested a values
            self._chi_interp = lambda a: interp1d(np.log10(a_grid), (chi_at_grid), kind='cubic', assume_sorted=True)(np.log10(a))
            self._a_min_cached = a_low
            self._neutrino_flag = neutrino_matter_like
            
        out = self._chi_interp(aarr) * self._chi_fac 
        return out
    
