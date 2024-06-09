import numpy as np

class Cosmology:
    def __init__(self, verbose=False):
        self.rho_crit = 2.77536627e11 # h^2 Msun/Mpc^3
    def set_cosmos(self, cosmologies):
        '''
        set the cosmologies for the cosmology class
        '''
        Neff = 3.046
        self.mnu     = cosmologies['mnu'] # eV
        # number of ultra-relativistic neutrinos
        self.Nur     = 2.0328*np.ones_like(self.mnu)
        self.Nur[self.mnu == 0] = Neff
        self.Nncdm   = np.ones_like(self.mnu)
        self.Nncdm[self.mnu == 0] = 0
        ### only support one massive neutrino
        self.h0      = cosmologies['H0'] / 100
        self.Omeganu = self.mnu/93.14/self.h0/self.h0
        # total matter without massive neutrinos
        # Only baryons and cold dark matter
        self.Omegam  = cosmologies['Omegam']  
        self.Omegab  = cosmologies['Omegab']
        self.Omegac  = cosmologies['Omegam'] - cosmologies['Omegab']
        self.w0      = cosmologies['w']
        self.wa      = cosmologies['wa']
        self.ns      = cosmologies['ns']
        self.As      = cosmologies['A'] * 1e-9
        self.Omegag  = 2.4735e-5 / self.h0**2 # photon radiation
        Gamma_nu = (4/11)**(1/3)
        f_nnu = 7/8*(Gamma_nu**4)*self.Nur # neutrino radiation
        self.OmegaR  = self.Omegag*(1+f_nnu) # Total radiation
        self.OmegaL  = 1 - self.Omegam - self.Omeganu - self.OmegaR
        self.Ncosmo  = len(self.Omegam)
        
    def get_Ez(self, z):
        '''
        Get the normalized Hubble parameter H(z) at redshift z.
        z : float or array-like, redshift
        '''
        z = np.atleast_1d(z)
        out = np.zeros((self.Ncosmo, len(z)))
        for iz in range(len(z)):
            out[:,iz] = np.sqrt((self.Omegam+self.Omeganu)*(1+z[iz])**3 + 
                                self.OmegaR*(1+z[iz])**4 + 
                                self.OmegaL*np.exp(3*((1/(1+z[iz])-1)*self.wa-(1 + self.w0 + self.wa)*np.log(1/(1+z[iz]))))
                                )
        return out
    
    def get_Omegam(self, z):
        '''
        Get the total matter density without massive neutrinos at redshift z.
        z : float or array-like, redshift
        '''
        z = np.atleast_1d(z)
        out = np.zeros((self.Ncosmo, len(z)))
        for iz in range(len(z)):
            out[:,iz] = self.Omegam * (1+z[iz])**3 / self.get_Ez(z[iz]).reshape(-1)**2
        return out
    
    def get_OmegaM(self, z):
        '''
        Get the total matter density at redshift z.
        z : float or array-like, redshift
        '''
        z = np.atleast_1d(z)
        out = np.zeros((self.Ncosmo, len(z)))
        for iz in range(len(z)):
            out[:,iz] = (self.Omegam + self.Omeganu) * (1+z[iz])**3 / self.get_Ez(z[iz]).reshape(-1)**2
        return out
    
    def get_OmegaL(self, z):
        '''
        Get the dark energy density at redshift z.
        z : float or array-like, redshift
        '''
        z = np.atleast_1d(z)
        out = np.zeros((self.Ncosmo, len(z)))
        for iz in range(len(z)):
            out[:,iz] = self.OmegaL * np.exp(3*((1/(1+z[iz])-1)*self.wa-(1 + self.w0 + self.wa)*np.log(1/(1+z[iz])))) / self.get_Ez(z[iz]).reshape(-1)**2
        return out
    
    
