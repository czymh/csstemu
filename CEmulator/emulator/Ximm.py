import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, Matern
from ..utils import cosmoNorm, data_path, zlists, check_r, check_z, MyStandardScaler

class Ximm_cb_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 129
        if self.verbose:
            print('Loading the ximm_cb emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNorm[:n_sample,:]
        self.nvec = 10
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgBximm_cb_n%d_HybridNBin127_rmax50.npy'%(self.nvec, n_sample))
        self.__PCA_mean = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        rbins = np.concatenate([np.logspace(-2, 1, 30+1)[:-1],
                        np.arange(10, 500, 5)])
        self.rmiddle = (rbins[1:] + rbins[:-1])/2  # Mpc/h
        rcut = 50
        rind = self.rmiddle <= rcut
        self.rmiddle = self.rmiddle[rind]
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + 'lgBximm_cb_gpr_kernel_nvec%d_n%d_HybridNBin127_rmax50.npy'%(self.nvec,n_sample), allow_pickle=True)
        Bkcoeff    = np.load(data_path + 'lgBximm_cb_coeff_nvec%d_n%d_HybridNBin127_rmax50.npy'%(self.nvec,n_sample))
        self.NormBeforeGP = True
        if self.NormBeforeGP:
            self.coeffSS = MyStandardScaler()
            self.paramSS = MyStandardScaler()
            Bkcoeff = self.coeffSS.fit_transform(Bkcoeff)
            self.X_train = self.paramSS.fit_transform(self.X_train)
        for ivec in range(self.nvec):
            k1    = Constant(gprinfo[ivec]['k1__constant_value'])
            k2    = Matern  (gprinfo[ivec]['k2__length_scale'])
            kivec = k1 * k2
            alpha = 1e-10
            ynorm = True
            self.__GPR[ivec] = GaussianProcessRegressor(self.X_train, Bkcoeff[:,ivec], 
                                                        kernel=kivec, alpha=alpha, normalize_y=ynorm)
        ### End of __init__
    
    def get_Br(self, z=None, r=None):
        '''
        Get the matter [cb] correlation function ratio B(r) at redshift z and wavenumber r.
        Br is defined as the ratio bewteen simulation and halofit.
        z : float or array-like, redshift
        r : float or array-like, wavenumber [Mpc/h]
        '''
        z = check_z(self.zlists, z)
        r = np.atleast_1d(r)
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        Brpred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            Brpred[ivec] = self.__GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            Brpred = self.coeffSS.inverse_transform(Brpred.reshape(1,-1))[0]
        ## PCA inverse transform
        Brpred = 10**((Brpred @ self.__PCA_components) + self.__PCA_mean)
        Brpred = Brpred.reshape(len(self.zlists), len(self.rmiddle))
        Brpred = Brpred[::-1,:]
        Brout  = np.zeros((len(z), len(r)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.rmiddle, Brpred, 
                                     kx=3, ky=1)
        Brout  = spline(z, r)
        return Brout 
    
