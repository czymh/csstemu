import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, RBF 
from ..utils import cosmoNorm, data_path, zlists, check_k, check_z, MyStandardScaler

class Bkmm_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 129
        if self.verbose:
            print('Loading the Bkmm emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNorm[:n_sample,:]
        self.nvec = 15
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgBk_n%d.npy'%(self.nvec, n_sample))
        self.__PCA_mean = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        self.klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
        kcut = 10
        ind = self.klist<=kcut
        self.klist = self.klist[ind]
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + 'lgBk_gpr_kernel_nvec%d_n%d_nb_Nmesh3072.npy'%(self.nvec,n_sample), allow_pickle=True)
        Bkcoeff    = np.load(data_path + 'lgBk_coeff_nvec%d_n%d_nb_Nmesh3072.npy'%(self.nvec,n_sample))
        self.NormBeforeGP = True
        if self.NormBeforeGP:
            self.coeffSS = MyStandardScaler()
            self.paramSS = MyStandardScaler()
            Bkcoeff      = self.coeffSS.fit_transform(Bkcoeff)
            self.X_train = self.paramSS.fit_transform(self.X_train)
        for ivec in range(self.nvec):
            k1    = Constant(gprinfo[ivec]['k1__constant_value'])
            k2    = RBF(gprinfo[ivec]['k2__length_scale'])
            kivec = k1 * k2
            alpha = 1e-10
            ynorm = True
            self.__GPR[ivec] = GaussianProcessRegressor(self.X_train, Bkcoeff[:,ivec], 
                                                        kernel=kivec, alpha=alpha, normalize_y=ynorm)
        ### End of __init__
    
    def get_Bk(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        Bk is defined as the ratio between nonlinear and linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        z = check_z(self.zlists, z)
        k = check_k(self.klist, k)
        numcos = self.ncosmo.shape[0]
        if self.verbose:
            print('Predicting Bk of %d cosmologies...'%numcos)
        if self.NormBeforeGP:
            self.ncosmo = self.paramSS.transform(self.ncosmo)
        ## Gaussian Process Regression
        Bkpred = np.zeros((numcos, self.nvec))
        for ivec in range(self.nvec):
            Bkpred[:,ivec] = self.__GPR[ivec].predict(self.ncosmo)
        if self.NormBeforeGP:
            Bkpred = self.coeffSS.inverse_transform(Bkpred)
        ## PCA inverse transform
        Bkpred = 10**((Bkpred @ self.__PCA_components) + self.__PCA_mean)
        Bkpred = Bkpred.reshape(numcos, len(self.zlists), len(self.klist))
        Bkpred = Bkpred[:,::-1,:]
        Bkout  = np.zeros((Bkpred.shape[0], len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        for ic in range(numcos):
            spline    = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred[ic], 
                                            kx=3, ky=1)
            Bkout[ic] = spline(z, k)
        return Bkout

class Bkmm_halofit_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 129
        if self.verbose:
            print('Loading the Bkmm_halofit emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNorm[:n_sample,:]
        self.nvec = 20
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_Bk_halofit_n%d.npy'%(self.nvec, n_sample))
        self.__PCA_mean = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        self.klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
        kcut = 10
        ind = self.klist<=kcut
        self.klist = self.klist[ind]
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + 'Bk_halofit_gpr_kernel_nvec%d_n%d_nb_Nmesh3072.npy'%(self.nvec,n_sample), allow_pickle=True)
        Bkcoeff    = np.load(data_path + 'Bk_halofit_coeff_nvec%d_n%d_nb_Nmesh3072.npy'%(self.nvec,n_sample))
        self.NormBeforeGP = True
        if self.NormBeforeGP:
            self.Bkcoeff_mean = np.mean(Bkcoeff, axis=0)
            self.Bkcoeff_std  = np.std (Bkcoeff, axis=0)
        
        if self.NormBeforeGP:
            for ivec in range(self.nvec):
                k1    = Constant(gprinfo[ivec]['k1__constant_value'])
                k2    = RBF(gprinfo[ivec]['k2__length_scale'])
                kivec = k1 * k2
                alpha = 1e-10
                ynorm = True
                self.__GPR[ivec] = GaussianProcessRegressor(self.X_train, (Bkcoeff[:,ivec]-self.Bkcoeff_mean[ivec])/self.Bkcoeff_std[ivec], 
                                                            kernel=kivec, alpha=alpha, normalize_y=ynorm)
        else:
            for ivec in range(self.nvec):
                k1    = Constant(gprinfo[ivec]['k1__constant_value'])
                k2    = RBF(gprinfo[ivec]['k2__length_scale'])
                kivec = k1 * k2
                alpha = 1e-10
                ynorm = True
                self.__GPR[ivec] = GaussianProcessRegressor(self.X_train, Bkcoeff[:,ivec], 
                                                            kernel=kivec, alpha=alpha, normalize_y=ynorm)
        ### End of __init__
    
    def get_Bk(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        Bk is defined as the ratio between nonlinear and linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        z = check_z(self.zlists, z)
        k = check_k(self.klist, k)
        numcos = self.ncosmo.shape[0]
        if self.verbose:
            print('Predicting Bk_halofit of %d cosmologies...'%numcos)
        ## Gaussian Process Regression
        Bkpred = np.zeros((numcos, self.nvec))
        for ivec in range(self.nvec):
            Bkpred[:,ivec] = self.__GPR[ivec].predict(self.ncosmo)
            if self.NormBeforeGP:
                Bkpred[:,ivec] = Bkpred[:,ivec] * self.Bkcoeff_std[ivec] + self.Bkcoeff_mean[ivec]
        ## PCA inverse transform
        Bkpred = ((Bkpred @ self.__PCA_components) + self.__PCA_mean)
        Bkpred = Bkpred.reshape(numcos, len(self.zlists), len(self.klist))
        Bkpred = Bkpred[:,::-1,:]
        Bkout  = np.zeros((Bkpred.shape[0], len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        for ic in range(numcos):
            spline    = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred[ic], 
                                            kx=3, ky=1)
            Bkout[ic] = spline(z, k)
        return Bkout 
    
