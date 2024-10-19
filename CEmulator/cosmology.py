import numpy as np
from scipy.integrate import quad, simps
from scipy.special import expit

class Cosmology:
    ## physical constants
    sigma_B   = 5.670373e-5; 
    vel_light = 2.99792458e10; # cm/s
    G_const   = 6.672e-8; 
    h0_units  = 3.2407789e-18; # h/sec
    kB        = 8.617333262145e-5   ## boltzman in eV/K

    def __init__(self, verbose=False):
        '''
        Initialize the CSST cosmology class.
        
        Args:
            verbose : bool, whether to output the running information
        
        '''
        
        self.rho_crit = 2.77536627e11 # h^2 Msun/Mpc^3
        
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
            self.Nur     = 2.0328
            self.Nncdm   = 1
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
        
    def _F_y(self, y, Fid=0):
        #### 0, 1 means F(y) and F'(y)
        if Fid == 0:
            F_y_int = lambda x,y: x*x*np.sqrt(x*x + y*y)*expit(-x)
        elif Fid == 1:
            F_y_int = lambda x,y: x*x*y/np.sqrt(x*x + y*y)*expit(-x)
        else:
            raise ValueError("Fid must be 0 or 1.")
        upper = np.inf
        if isinstance(y, float):
            return quad(F_y_int, 0, upper, args=(y))[0]
        elif isinstance(y, np.ndarray) or isinstance(y, list):
            return np.array([quad(F_y_int, 0, upper, args=(iy))[0] for iy in list(y)])
    
    def _Omeganu_TimesHubbleSquare(self, z):
        fac = 15/np.pi/np.pi/np.pi/np.pi * (self.Gamma_nu**4) * self.Omegag *(1+z)*(1+z)*(1+z)*(1+z)
        T_nu = self.Gamma_nu*self.TCMB
        if self.mnu != 0.0:
            F_sumNu = self._F_y(self.mnu/(1+z)/self.kB/T_nu, Fid=0)
        else:
            F_sumNu = 0
        return fac*F_sumNu
    
    def get_Ez(self, z):
        '''
        Get the normalized Hubble parameter H(z) at redshift z.
        
        Args:
            z : float or array-like, redshift
        Returns:
            array-like : 2D array of shape (len(z)), normalized Hubble parameter H(z)/H0
        '''
        z = np.atleast_1d(z)
        out = np.zeros((len(z)))
        for iz in range(len(z)):
            out[iz] = np.sqrt(self.Omegam*(1+z[iz])**3 + \
                              self._Omeganu_TimesHubbleSquare(z[iz]) + \
                              self.Omegak*(1+z[iz])**2 + \
                              self.OmegaR*(1+z[iz])**4 + \
                              self.OmegaL*np.exp(3*((1/(1+z[iz])-1)*self.wa-(1 + self.w0 + self.wa)*np.log(1/(1+z[iz]))))
                             ) 
        return out
    
    def get_Omegam(self, z):
        '''
        Get the total matter density without massive neutrinos at redshift z.
        
        Args:
            z : float or array-like, redshift
        Returns:
            array-like : 2D array of shape (len(z)), total matter density without massive neutrinos
        '''
        z = np.atleast_1d(z)
        out = np.zeros((len(z)))
        for iz in range(len(z)):
            out[iz] = self.Omegam * (1+z[iz])**3 / self.get_Ez(z[iz]).reshape(-1)**2
        return out
    
    def get_OmegaM(self, z):
        '''
        Get the total matter density at redshift z.
        
        Args:
            z : float or array-like, redshift
        Returns:
            array-like : 2D array of shape (len(z)), total matter density
        '''
        z = np.atleast_1d(z)
        out = np.zeros((len(z)))
        for iz in range(len(z)):
            out[iz] = self.OmegaM * (1+z[iz])**3 / self.get_Ez(z[iz]).reshape(-1)**2
        return out
    
    def get_OmegaL(self, z):
        '''
        Get the dark energy density at redshift z.
        
        Args:
            z : float or array-like, redshift
        Returns:
            array-like : 2D array of shape (len(z)), dark energy density
        '''
        z = np.atleast_1d(z)
        out = np.zeros((len(z)))
        for iz in range(len(z)):
            out[iz] = self.OmegaL * np.exp(3*((1/(1+z[iz])-1)*self.wa-(1 + self.w0 + self.wa)*np.log(1/(1+z[iz])))) / self.get_Ez(z[iz]).reshape(-1)**2
        return out
    
    def comoving_distance(self, z):
        '''
        Get the comoving distance at redshift z.
        
        Args:
            z : float or array-like, redshift
        Returns:
            array-like : 2D array of shape (len(z)), comoving distance
        '''
        z = np.atleast_1d(z)
        aarr = 1/(1+z)
        out = np.zeros((len(z)))
        for ia in range(len(aarr)):
            out[ia] = quad(lambda a: 1e-5*self.vel_light/self.get_Ez(1/a-1)/a/a/self.h0/100, aarr[ia], 1.0)[0]
        return out
