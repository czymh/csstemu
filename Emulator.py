from turtle import st
import numpy as np
import os
import inspect
from GaussianProcess import GaussianProcessRegressor, czConstant, czRBF

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

class CEmulator:
    '''
    The CSST Emulator class for various statistics.
    '''
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
    
    def __init__(self, statistic='Pkmm'):
        data_path = data_path = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), 'data/')
        cosmoall = np.load(data_path + 'cosmologies_8d_train_n129_Sobol.npy')
        self.X_train = NormCosmo(cosmoall[:65,:], self.param_names, self.param_limits)
        if statistic == 'Pkmm':
            nvec = 10
            ### load the PCA transformation matrix
            _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgBk.npy'%nvec)
            self.__PCA_mean = _tmp[0,:]
            self.__PCA_components = _tmp[1:,:]
            del _tmp
            ### Load the Gaussian Process Regression model
            self.__GPR = np.zeros(nvec, dtype=object)
            gprinfo    = np.load(data_path + 'lgBk_gpr_kernel_nvec%d_n65_nb_Nmesh3072.npy'%nvec, allow_pickle=True)
            Bkcoeff    = np.load(data_path + 'lgBk_coeff_nvec%d_n65_nb_Nmesh3072.npy'%nvec)
            for ivec in range(nvec):
                k1    = czConstant(gprinfo[ivec]['k1__constant_value'])
                k2    = czRBF(gprinfo[ivec]['k2__length_scale'])
                kivec = k1 * k2
                alpha = 1e-10
                ynorm = True
                self.__GPR[ivec] = GaussianProcessRegressor(self.X_train, Bkcoeff[:,ivec], 
                                                            kernel=kivec, alpha=alpha, normalize_y=ynorm)
        else:
            raise ValueError('Statistic %s not supported yet.'%statistic)

    def predict(self, cosmologies, statistic='Pkmm', **kwargs):
        if statistic == 'Pkmm':
            cosmologies = np.atleast_2d(cosmologies)
            nvec = 10
            ncosmo = NormCosmo(cosmologies, self.param_names, self.param_limits)
            ## Gaussian Process Regression
            Bkpred = np.zeros(nvec)
            for ivec in range(nvec):
                Bkpred[ivec] = self.__GPR[ivec].predict(ncosmo)
            ## PCA inverse transform
            Bkpred = 10**np.dot(Bkpred, self.__PCA_components) + self.__PCA_mean
            return Bkpred
        else:
            raise ValueError('Statistic %s not supported yet.'%statistic)
        
        
        

    
    
