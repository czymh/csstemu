import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, RBF 
from ..utils import data_path, zlists, param_limits, param_names, check_k, check_z, NormCosmo, MyStandardScaler

class PkcbLin_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 513
        if self.verbose:
            print('Loading the PkcbLin emulator...')
            print('Using %d training samples.'%n_sample)
        cosmoallLarge  = np.load(data_path + 'cosmologies_8d_train_n513_Sobol.npy')
        cosmoNormLarge = NormCosmo(cosmoallLarge, param_names, param_limits)
        self.X_train = cosmoNormLarge[:n_sample,:]
        self.nvec = 20
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgpkLin_n%d.npy'%(self.nvec, n_sample))
        self.__PCA_mean       = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        self.klist = np.load(data_path + 'karr_kmax100.npy')
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + 'lgpkLin_gpr_kernel_nvec%d_n%d_kmax100.npy'%(self.nvec,n_sample), allow_pickle=True)
        pkcoeff    = np.load(data_path + 'lgpkLin_coeff_nvec%d_n%d_kmax100.npy'%(self.nvec,n_sample))
        self.NormBeforeGP = True
        if self.NormBeforeGP:
            self.pkcoeffSS = MyStandardScaler()
            self.paramSS   = MyStandardScaler()
            pkcoeff        = self.pkcoeffSS.fit_transform(pkcoeff)
            self.X_train   = self.paramSS.fit_transform(self.X_train)
        for ivec in range(self.nvec):
            k1    = Constant(gprinfo[ivec]['k1__constant_value'])
            k2    = RBF(gprinfo[ivec]['k2__length_scale'])
            kivec = k1 * k2
            alpha = 1e-10
            ynorm = True
            self.__GPR[ivec] = GaussianProcessRegressor(self.X_train, pkcoeff[:,ivec], 
                                                        kernel=kivec, alpha=alpha, normalize_y=ynorm)
        ### End of __init__
    
    def get_pkLin(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        z = check_z(self.zlists, z)
        k = check_k(self.klist, k)
        numcos = self.ncosmo.shape[0]
        if self.verbose:
            print('Predicting PkcbLin of %d cosmologies...'%numcos)
        if self.NormBeforeGP:
            self.ncosmo = self.paramSS.transform(self.ncosmo)
        ## Gaussian Process Regression
        pkpred = np.zeros((numcos, self.nvec))
        for ivec in range(self.nvec):
            pkpred[:,ivec] = self.__GPR[ivec].predict(self.ncosmo)
        if self.NormBeforeGP:
            pkpred = self.pkcoeffSS.inverse_transform(pkpred)
        ## PCA inverse transform
        pkpred = 10**((pkpred @ self.__PCA_components) + self.__PCA_mean)
        pkpred = pkpred.reshape(numcos, len(self.zlists), len(self.klist))
        pkpred = pkpred[:,::-1,:]
        pkout  = np.zeros((pkpred.shape[0], len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        for ic in range(numcos):
            spline    = RectBivariateSpline(self.zlists[::-1], self.klist, pkpred[ic], 
                                            kx=3, ky=1)
            pkout[ic] = spline(z, k)
        return pkout
    
    