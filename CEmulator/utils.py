import numpy as np
import os
import inspect

param_names  = ['Omegab', 'Omegam', 'H0', \
                'ns', 'A', 'w', 'wa', 'mnu']
param_limits = {}
param_limits['Omegab'] = [0.04, 0.06]
param_limits['Omegam'] = [0.24, 0.40]
param_limits['H0']     = [60,     80]
param_limits['ns']     = [0.92, 1.00]
param_limits['A']      = [1.7,   2.5]  # 1e-9
param_limits['w']      = [-1.3, -0.7]
param_limits['wa']     = [-0.5,  0.5]
param_limits['mnu']    = [0,     0.3]

## emulator redshifts [0, 3]
zlists = [3.00, 2.50, 2.00, 1.75, \
          1.50, 1.25, 1.00, 0.80, \
          0.50, 0.25, 0.10, 0.00]

def check_z(zlists=zlists,z=None):
    if z is None:
        z = zlists
        print('No redshift input, using default redshifts [0,3]-12.')
    zinput = np.atleast_1d(np.copy(z))
    if np.any(zinput < zlists[-1]) or np.any(zinput > zlists[0]):
        raise ValueError('Redshift z out of range [0, 3].')
    if not np.all(np.diff(zinput) > 0.0):
        zinput.sort()
        print('Predicting redshifts (sorted):', zinput)
    return zinput
    
def check_k(klists, k=None):
    if k is None:
        raise ValueError('Please provide the wavenumber k [h/Mpc].')
    kinput = np.atleast_1d(k)
    if np.any(kinput < klists[0]) or np.any(kinput > klists[-1]):
        raise ValueError('Wavenumber k out of range.')
    if not np.all(np.diff(kinput) > 0.0):
        raise ValueError('k must be strictly increasing!')
    return kinput

def NormCosmo(cosmologies, param_names, param_limits):
    '''
    Normalizes the cosmological parameters to the range [0, 1]
    cosmologies : array of shape (n_cosmologies, n_params)
    param_names : list of parameter names
    param_limits: dictionary with parameter names as keys and [min, max] as values
    '''
    ncosmo = np.zeros_like(cosmologies)
    for i, param in enumerate(param_names):
        ncosmo[:,i] = (cosmologies[:,i] - param_limits[param][0]) / (param_limits[param][1] - param_limits[param][0])
    return ncosmo

class MyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        '''
        X must be (nparam, nvec) array
        '''
        self.mean_  = np.mean(X, axis=0)
        self.scale_ = np.std (X, axis=0)

    def transform(self, X):
        return np.array([(X[:,ivec]-self.mean_[ivec])/self.scale_[ivec] for ivec in range(X.shape[1])]).T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        return np.array([Y[:,ivec] * self.scale_[ivec] + self.mean_[ivec] for ivec in range(Y.shape[1])]).T

def useCLASS(mypara, strzlists, non_linear=None, kmax=10.0):
    from classy import Class
    param_names  = ['Omegab', 'Omegam', 'H0', \
                   'ns', 'A', 'w', 'wa', 'mnu']
    para = {}
    for i_n in range(len(param_names)):
        para[param_names[i_n]] = mypara[i_n]
    h0 = para['H0']/100
    params = {
        'output': 'mPk, mTk', 
        'P_k_max_1/Mpc': kmax/h0,
        'A_s':       para['A']*1e-9,
        'n_s':       para['ns'], 
        'h':         h0,
        'Omega_Lambda': 0.0, 
        'wa_fld':    para['wa'],
        'w0_fld':    para['w'],
        'Omega_b':   para['Omegab'],
        'Omega_cdm': para['Omegam'] - para['Omegab'],
        'Omega_Lambda': 0.0, 
        'z_pk': strzlists,
        'T_cmb': 2.7255,
        'gauge': 'synchronous'
    }
    if para['mnu'] == 0:
        params['N_ur'] = 3.046
    else:
        params['N_ur'] = 3.046 - 1.0132
        params['N_ncdm'] = 1
        params['m_ncdm'] = "{:f}".format(para['mnu'])
    if non_linear is not None:
        params['non linear'] = non_linear
    cosmo_class = Class()
    cosmo_class.set(params)
    cosmo_class.compute()
    return cosmo_class

## load the trainning cosmologies
data_path = data_path = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), 'data/')
cosmoall  = np.load(data_path + 'cosmologies_8d_train_n129_Sobol.npy')
cosmoNorm = NormCosmo(cosmoall, param_names, param_limits)


##### for the halofit computation

def WinInt(Pklin, r):
    '''
    rewrite the wint function in the halofit computation
    from https://github.com/lcasarini/PKequal/blob/master/CAMB/halofit_ppf.f90
    '''
    nint = 1000
    anorm = 1 / (2 * np.pi ** 2)
    t  = (np.arange(1, nint+1) - 0.5)/nint
    y  = -1.0 + 1.0 / t
    rk = y
    d2 = Pklin(rk) * (rk*rk*rk * anorm)
    x  = y * r
    x2 = x * x
    w1 = np.exp(-x2)
    w2 = 2 * x2 * w1
    w3 = 4 * x2 * (1 - x2) * w1
    fac = d2 / y / t / t
    sum1 = np.sum(w1 * fac)/nint
    sum2 = np.sum(w2 * fac)/nint
    sum3 = np.sum(w3 * fac)/nint
    sig = np.sqrt(sum1)
    d1 = -sum2 / sum1
    d2 = -sum2 * sum2 / sum1 / sum1 - sum3 / sum1
    return sig, d1, d2

def compute_Rsigma_neff_C(Pklin):
    '''
    compute the R_sigma, neff, and Curv for the halofit computation
    
    Parameters
    ----------
    Pklin : callable function
        linear power spectrum
    
    '''
    # , R_sigma, h=1e-4
    ###### This method can not converge with the CLASS Halofit
    # lnR_0 = np.log(R_sigma)
    # lnR_p = lnR_0*(1.+h*R_sigma)
    # lnR_m = lnR_0*(1.-h*R_sigma)
    # lnsigma_0 = np.log(sigmaRSquareGaussian(Pklin, np.exp(lnR_0)))
    # lnsigma_p = np.log(sigmaRSquareGaussian(Pklin, np.exp(lnR_p)))
    # lnsigma_m = np.log(sigmaRSquareGaussian(Pklin, np.exp(lnR_m)))
    # neff = -3. - (lnsigma_p-lnsigma_0)/(lnR_p-lnR_0)
    # Curv = - (lnsigma_p - 2.*lnsigma_0 + lnsigma_m)/(lnR_p-lnR_0)**2
    ######################################
    abstol = 1e-4
    xlogr1 = -2.0
    xlogr2 = 3.5
    while True:
        rmid = (xlogr2 + xlogr1) / 2.0
        rmid = 10 ** rmid
        sig, d1, d2 = WinInt(Pklin, rmid)
        diff = sig - 1.0
        if abs(diff) <= abstol:
            rknl = 1. / rmid
            rneff = -3 - d1
            rncur = -d2
            break
        elif diff >  abstol:
            xlogr1 = np.log10(rmid)
        elif diff < -abstol:
            xlogr2 = np.log10(rmid)
        if xlogr2 < -1.9999:
            # is still linear, exit
            break
        elif xlogr2 > 3.4999:
            # Totally crazy non-linear
            raise ValueError('Error in halofit')
    neff = rneff
    Curv = rncur
    return rmid, neff, Curv

def PkHaloFit(k, pklin, R_sigma, Omegamz, OmegaLz, fnu, an, bn, cn, gamman, alphan, betan, mun, nun):
    y = k * R_sigma
    f = y/4. + y**2/8.
    if np.abs(1-Omegamz)>0.01: # /*then omega evolution */
        f1a  = Omegamz**(-0.0732) 
        f2a  = Omegamz**(-0.1423) 
        f3a  = Omegamz**( 0.0725)
        f1b  = Omegamz**(-0.0307) 
        f2b  = Omegamz**(-0.0585) 
        f3b  = Omegamz**( 0.0743)  
        frac = OmegaLz/(1-Omegamz)
        f1   = frac*f1b + (1-frac)*f1a 
        f2   = frac*f2b + (1-frac)*f2a 
        f3   = frac*f3b + (1-frac)*f3a 
    else:
        f1, f2, f3 = 1., 1., 1.
    delatfac = 1+fnu*47.48*k*k/(1+1.5*k*k)
    Delta_L = (k*k*k*pklin/2/np.pi**2) * delatfac
    ### Two-halo term
    Delta_Q = Delta_L * ((1+Delta_L)**betan)/(1+alphan*Delta_L) * np.exp(-f)
    ### One-halo term
    Delta_H = an*y**(3*f1) / (1+bn*y**f2 + (cn*y*f3)**(3-gamman))
    Delta_H = Delta_H / (1. + mun/y + nun/y**2) * (1 + fnu**0.977)
    return (Delta_Q + Delta_H) * (2*np.pi**2) / k**3
