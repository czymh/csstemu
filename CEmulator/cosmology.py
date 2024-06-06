import numpy as np

class Cosmology:
    def __init__(self, verbose=False):
        self.rho_crit = 2.77536627e11 # h^2 Msun/Mpc^3
    def set_cosmos(self, cosmologies):
        '''
        set the cosmologies for the cosmology class
        '''
        self.mnu     = cosmologies['mnu'] # eV
        # number of ultra-relativistic neutrinos
        self.Nur     = 2*np.ones_like(self.mnu)
        self.Nur[self.mnu == 0] = 3 
        self.Nncdm   = 1 - self.Nur
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
        self.Omegag  = 2.4735e-5 * self.h0**2 # photon radiation
        Gamma_nu = (4/11)**(1/3)
        f_nnu = 7/8*(Gamma_nu**4)*self.Nur # neutrino radiation
        self.OmegaR  = self.Omegag*(1+f_nnu)
        self.OmegaL  = 1 - self.Omegam - self.Omeganu - self.OmegaR
        
        
