import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, RBF 
from ..utils import cosmoNorm, cosmoNormLarge, data_path, zlists, checkdata, check_z, MyStandardScaler

class Bkcb_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 129
        if self.verbose:
            print('Loading the Bkcb emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNorm[:n_sample,:]
        self.nvec = 15
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgBk_n%d.npy'%(self.nvec, n_sample))
        self.__PCA_mean = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        self.klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
        kcut = 10.01
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
        k = checkdata(self.klist, k, dname='wavenumber')
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        Bkpred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            Bkpred[ivec] = self.__GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            Bkpred = self.coeffSS.inverse_transform(Bkpred.reshape(1,-1))[0]
        ## PCA inverse transform
        Bkpred = 10**((Bkpred @ self.__PCA_components) + self.__PCA_mean)
        Bkpred = Bkpred.reshape(len(self.zlists), len(self.klist))
        Bkpred = Bkpred[::-1,:]
        Bkout  = np.zeros((len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred, 
                                        kx=3, ky=1)
        Bkout  = spline(z, k)
        return Bkout

class Bkcb_halofit_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 129
        if self.verbose:
            print('Loading the Bkcb_halofit emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNorm[:n_sample,:]
        self.nvec = 20
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_Bk_halofit_n%d.npy'%(self.nvec, n_sample))
        self.__PCA_mean = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        self.klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
        kcut = 10.01
        ind = self.klist<=kcut
        self.klist = self.klist[ind]
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + 'Bk_halofit_gpr_kernel_nvec%d_n%d_nb_Nmesh3072.npy'%(self.nvec,n_sample), allow_pickle=True)
        Bkcoeff    = np.load(data_path + 'Bk_halofit_coeff_nvec%d_n%d_nb_Nmesh3072.npy'%(self.nvec,n_sample))
        self.NormBeforeGP = True
        if self.NormBeforeGP:
            self.coeffSS = MyStandardScaler()
            self.paramSS = MyStandardScaler()
            Bkcoeff = self.coeffSS.fit_transform(Bkcoeff)
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
        k = checkdata(self.klist, k, dname='wavenumber')
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        Bkpred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            Bkpred[ivec] = self.__GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            Bkpred = self.coeffSS.inverse_transform(Bkpred.reshape(1,-1))[0]
        ## PCA inverse transform
        Bkpred = ((Bkpred @ self.__PCA_components) + self.__PCA_mean)
        Bkpred = Bkpred.reshape(len(self.zlists), len(self.klist))
        Bkpred = Bkpred[::-1,:]
        Bkout  = np.zeros((len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred, 
                                     kx=3, ky=1)
        Bkout  = spline(z, k)
        return Bkout    

class Bkcb_hmcode2020_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 129
        if self.verbose:
            print('Loading the Bkcb_hmcode2020 emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNorm[:n_sample,:]
        self.nvec = 20
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_Bk_hmcode2020_n%d.npy'%(self.nvec, n_sample))
        self.__PCA_mean = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        self.klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
        kcut = 10.01
        ind = self.klist<=kcut
        self.klist = self.klist[ind]
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + 'Bk_hmcode2020_gpr_kernel_nvec%d_n%d_nb_Nmesh3072.npy'%(self.nvec,n_sample), allow_pickle=True)
        Bkcoeff    = np.load(data_path + 'Bk_hmcode2020_coeff_nvec%d_n%d_nb_Nmesh3072.npy'%(self.nvec,n_sample))
        self.NormBeforeGP = True
        if self.NormBeforeGP:
            self.coeffSS = MyStandardScaler()
            self.paramSS = MyStandardScaler()
            Bkcoeff = self.coeffSS.fit_transform(Bkcoeff)
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
        k = checkdata(self.klist, k, dname='wavenumber')
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        Bkpred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            Bkpred[ivec] = self.__GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            Bkpred = self.coeffSS.inverse_transform(Bkpred.reshape(1,-1))[0]
        ## PCA inverse transform
        Bkpred = ((Bkpred @ self.__PCA_components) + self.__PCA_mean)
        Bkpred = Bkpred.reshape(len(self.zlists), len(self.klist))
        Bkpred = Bkpred[::-1,:]
        Bkout  = np.zeros((len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred, 
                                     kx=3, ky=1)
        Bkout  = spline(z, k)
        return Bkout 


### Transfer function from Pklin to Pkhmcode
class Bkcb_lin2hmcode_gp:
    zlists = zlists 
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 513
        if self.verbose:
            print('Loading the Bkcb_lin2hmcode emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNormLarge[:n_sample,:]
        self.nvec = 20
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgpkhmcode20_cb_n%d.npy'%(self.nvec, n_sample))
        self.__PCA_mean       = _tmp[0,:]
        self.__PCA_components = _tmp[1:,:]
        ### load karr
        self.klist = np.load(data_path + 'karr_kmax100.npy')
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + 'lgpkhmcode20_cb_gpr_kernel_nvec%d_n%d_kmax100.npy'%(self.nvec,n_sample), allow_pickle=True)
        Bkcoeff    = np.load(data_path + 'lgpkhmcode20_cb_coeff_nvec%d_n%d_kmax100.npy'%(self.nvec,n_sample))
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
        Get the matter power spectrum B(k) at redshift z and wavenumber k.
        Bk is defined as the ratio between HMCODE and linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        z = check_z(self.zlists, z)
        k = checkdata(self.klist,  k, dname='wavenumber')
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        Bkpred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            Bkpred[ivec] = self.__GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            Bkpred = self.coeffSS.inverse_transform(Bkpred.reshape(1,-1))[0]
        ## PCA inverse transform
        Bkpred = 10**((Bkpred @ self.__PCA_components) + self.__PCA_mean)
        Bkpred = Bkpred.reshape(len(self.zlists), len(self.klist))
        Bkpred = Bkpred[::-1,:]
        Bkout  = np.zeros((len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred, 
                                     kx=3, ky=1)
        Bkout  = spline(z, k)
        return Bkout
