import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, RBF 
from ..utils import data_path, zlists, check_k, check_z, MyStandardScaler, cosmoNormLarge

class PkcbLin_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 513
        if self.verbose:
            print('Loading the PkcbLin emulator...')
            print('Using %d training samples.'%n_sample)
        indexs = np.arange(513)
        self.X_train = cosmoNormLarge[indexs,:]
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
    
    def get_pkcbLin(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        z = check_z(self.zlists, z)
        k = check_k(self.klist, k)
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        pkpred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            pkpred[ivec] = self.__GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            pkpred = self.pkcoeffSS.inverse_transform(pkpred.reshape(1,-1))[0]
        ## PCA inverse transform
        pkpred = 10**((pkpred @ self.__PCA_components) + self.__PCA_mean)
        pkpred = pkpred.reshape(len(self.zlists), len(self.klist))
        pkpred = pkpred[::-1,:]
        pkout  = np.zeros((pkpred.shape[0], len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.klist, pkpred, 
                                            kx=3, ky=1)
        pkout  = spline(z, k)
        return pkout
    
class Pknn_cbLin_gp:
    zlists = zlists  
    def __init__(self, verbose=False): 
        self.verbose = verbose
        n_sample = 512
        if self.verbose:
            print('Loading the PknnLin emulator...')
            print('Using %d training samples [remove c0001 (no massive neutrino)].'%n_sample)
        # remove c0001
        indexs = np.arange(513)
        indexs = np.delete(indexs, 1)
        self.X_train = cosmoNormLarge[indexs,:]
        self.nvec = 10
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgpkLin_nn_cb_n%d.npy'%(self.nvec, n_sample))
        self.__PCA_mean       = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        self.klist = np.load(data_path + 'karr_kmax100.npy')
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + 'lgpkLin_nn_cb_gpr_kernel_nvec%d_n%d_kmax100.npy'%(self.nvec,n_sample), allow_pickle=True)
        pkcoeff    = np.load(data_path + 'lgpkLin_nn_cb_coeff_nvec%d_n%d_kmax100.npy'%(self.nvec,n_sample))
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
    
    def get_pknn_cbLin(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        z = check_z(self.zlists, z)
        k = check_k(self.klist, k)
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        pkpred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            pkpred[ivec] = self.__GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            pkpred = self.pkcoeffSS.inverse_transform(pkpred.reshape(1,-1))[0]
        ## PCA inverse transform
        pkpred = 10**((pkpred @ self.__PCA_components) + self.__PCA_mean)
        pkpred = pkpred.reshape(len(self.zlists), len(self.klist))
        pkpred = pkpred[::-1,:]
        pkout  = np.zeros((pkpred.shape[0], len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.klist, pkpred, 
                                            kx=3, ky=1)
        pkout  = spline(z, k)
        return pkout


