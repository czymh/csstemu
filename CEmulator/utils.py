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
