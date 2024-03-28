from turtle import st
import numpy as np
import os
import inspect
from GaussianProcess import GaussianProcessRegressor, czConstant, czRBF
from scipy.interpolate import RectBivariateSpline
            
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
        self.statistic = statistic
        data_path = data_path = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), 'data/')
        cosmoall = np.load(data_path + 'cosmologies_8d_train_n129_Sobol.npy')
        self.X_train = NormCosmo(cosmoall[:65,:], self.param_names, self.param_limits)
        if self.statistic == 'Pkmm':
            self.nvec = 10
            ### load the PCA transformation matrix
            _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgBk.npy'%self.nvec)
            self.__PCA_mean = _tmp[0,:]
            self.__PCA_components = _tmp[1:,:]
            ### load karr
            self.klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
            kcut = 10
            ind = self.klist<=kcut
            self.klist = self.klist[ind]
            ### Load the Gaussian Process Regression model
            self.__GPR = np.zeros(self.nvec, dtype=object)
            gprinfo    = np.load(data_path + 'lgBk_gpr_kernel_nvec%d_n65_nb_Nmesh3072.npy'%self.nvec, allow_pickle=True)
            Bkcoeff    = np.load(data_path + 'lgBk_coeff_nvec%d_n65_nb_Nmesh3072.npy'%self.nvec)
            for ivec in range(self.nvec):
                k1    = czConstant(gprinfo[ivec]['k1__constant_value'])
                k2    = czRBF(gprinfo[ivec]['k2__length_scale'])
                kivec = k1 * k2
                alpha = 1e-10
                ynorm = True
                self.__GPR[ivec] = GaussianProcessRegressor(self.X_train, Bkcoeff[:,ivec], 
                                                            kernel=kivec, alpha=alpha, normalize_y=ynorm)
        else:
            raise ValueError('Statistic %s not supported yet.'%statistic)

    def predict(self, **kwargs):
        if self.statistic == 'Pkmm':
            ## check the input redshift
            if 'z' not in kwargs.keys():
                kwargs['z'] = self.zlists
                print('No redshift input, using default redshifts [0,3]-12.')
            if 'k' not in kwargs.keys():
                raise ValueError('Please provide the wavenumber k [h/Mpc].')
            self.zinput = np.atleast_1d(kwargs['z'])
            self.kinput = np.atleast_1d(kwargs['k'])
            if np.any(self.zinput < 0) or np.any(self.zinput > 3):
                raise ValueError('Redshift z out of range [0, 3].')
            numcos = self.cosmologies.shape[0]
            print('Predicting %d cosmologies...'%numcos)
            ncosmo = NormCosmo(self.cosmologies, self.param_names, self.param_limits)
            ## Gaussian Process Regression
            Bkpred = np.zeros((numcos, self.nvec))
            for ivec in range(self.nvec):
                Bkpred[:,ivec] = self.__GPR[ivec].predict(ncosmo)
            ## PCA inverse transform
            Bkpred = 10**np.dot(Bkpred, self.__PCA_components) + self.__PCA_mean
            Bkpred = Bkpred.reshape(numcos, len(self.zlists), len(self.klist))
            
            self.zinput.sort()
            print('Predicting redshifts (sorted):', self.zinput)
            if not np.all(np.diff(self.kinput) > 0.0):
                raise ValueError('k must be strictly increasing!')
            ### z space use cubic spline while k space use linear interpolation
            for ic in range(numcos):
                spline = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred[ic], 
                                             kx=3, ky=1)
                Bkpred[ic] = spline(self.zinput, self.kinput)
            return Bkpred
        else:
            raise ValueError('Statistic %s not supported yet.'%self.statistic)
        
    def set_cosmos(self, Omegab=0.04897468, Omegam=0.30969282, 
                   H0=67.66, As=2.105e-9, ns=0.9665, w=-1.0, wa=0.0, 
                   mnu=0.06, z=0.0):
        '''
        Set the cosmological parameters. You can input the float or array-like.
        '''
        cosmos = {}
        cosmos['Omegab'] = np.atleast_1d(Omegab)
        cosmos['Omegam'] = np.atleast_1d(Omegam)
        cosmos['H0']     = np.atleast_1d(H0)
        cosmos['ns']     = np.atleast_1d(ns)
        cosmos['A']      = np.atleast_1d(As*1e9)
        cosmos['w']      = np.atleast_1d(w)
        cosmos['wa']     = np.atleast_1d(wa)
        cosmos['mnu']    = np.atleast_1d(mnu)
        n_params = len(self.param_names)
        ## check the parameter range
        lenlists = np.zeros((n_params), dtype=int)
        for ind, ikey in enumerate(self.param_names):
            if np.any(cosmos[ikey] < self.param_limits[ikey][0]) or \
               np.any(cosmos[ikey] > self.param_limits[ikey][1]):
                raise ValueError('Parameter %s out of range.'%ikey)
            lenlists[ind] = len(cosmos[ikey])
        numcosmos = np.unique(lenlists)
        if numcosmos.size > 1:
            raise ValueError('Inconsistent parameter array length.')
        numcosmos = numcosmos[0]
        ## into the cosmologies array
        self.cosmologies = np.zeros((numcosmos, n_params))
        for ind, ikey in enumerate(self.param_names):
            self.cosmologies[:,ind] = cosmos[ikey]
        
        
        
        
        

    
    
