import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, RBF 
from ..utils import cosmoNormLarge, data_path, zlists, check_k, check_z, MyStandardScaler

class Tkbase_gp:
    zlists = zlists
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.emunamestr = ''
        self.n_sample   = 513
        self.nvec       = 5
        self._load_data()
   
    def _load_data(self):
        if self.verbose:
            print('Loading the %s emulator...'%self.emunamestr)
            print('Using %d training samples.'%self.n_sample)
        self.X_train = cosmoNormLarge[:self.n_sample,:]
        ### load karr
        self.kstr  = '_kmax100'
        self.klist = np.load(data_path + 'karr%s.npy'%self.kstr)
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_%s_n%d.npy'%(self.nvec, self.emunamestr, self.n_sample))
        self._PCA_mean       = _tmp[0,:]
        self._PCA_components = _tmp[1:,:]
        ### Load the Gaussian Process Regression model
        self._GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + '%s_gpr_kernel_nvec%d_n%d%s.npy'%(self.emunamestr, self.nvec, self.n_sample, self.kstr), allow_pickle=True)
        pkcoeff    = np.load(data_path + '%s_coeff_nvec%d_n%d%s.npy'%(self.emunamestr, self.nvec, self.n_sample, self.kstr))
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
            self._GPR[ivec] = GaussianProcessRegressor(self.X_train, pkcoeff[:,ivec], 
                                                       kernel=kivec, alpha=alpha, normalize_y=ynorm)
        ### End of __init__    
        
    def get_Tkbase(self, z=None, k=None):
        '''
        Get the base matter power spectrum transformation T(k) at redshift z and wavenumber k.
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
            pkpred[ivec] = self._GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            pkpred = self.pkcoeffSS.inverse_transform(pkpred.reshape(1,-1))[0]
        ## PCA inverse transform
        pkpred = ((pkpred @ self._PCA_components) + self._PCA_mean)
        pkpred = pkpred.reshape(len(self.zlists), len(self.klist))
        pkpred = pkpred[::-1,:]
        pkout  = np.zeros((pkpred.shape[0], len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.klist, pkpred, 
                                     kx=3, ky=1)
        pkout  = spline(z, k)
        return pkout


class Tkcblin_gp(Tkbase_gp):
    def __init__(self, verbose=False):
        self.verbose = verbose
        ##### specify settings
        self.n_sample   = 513
        self.emunamestr = 'Tkcb_lin_N3_N1'
        self.nvec       = 5
        super()._load_data()
    
    def get_Tkcblin(self, z=None, k=None):
        '''
        Get the [cb] [linear] matter power spectrum transformation T(k) from Nncdm=1 to Nncdm=3 [degenerate] at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        return self.get_Tkbase(z=z, k=k)
    
class Tkmmlin_gp(Tkbase_gp):
    def __init__(self, verbose=False):
        self.verbose    = verbose
        ##### specify settings
        self.n_sample   = 513
        self.emunamestr = 'Tkmm_lin_N3_N1'
        self.nvec       = 5
        super()._load_data()
    
    def get_Tkmmlin(self, z=None, k=None):
        '''
        Get the [mm] [linear] matter power spectrum transformation T(k) from Nncdm=1 to Nncdm=3 [degenerate] at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        return self.get_Tkbase(z=z, k=k)
    
class Tkcbhalofit_gp(Tkbase_gp):
    def __init__(self, verbose=False):
        self.verbose    = verbose
        ##### specify settings
        self.n_sample   = 513
        self.emunamestr = 'Tkcb_halofit_N3_N1'
        self.nvec       = 5
        super()._load_data()
    
    def get_Tkcbhalofit(self, z=None, k=None):
        '''
        Get the [cb] [halofit] matter power spectrum transformation T(k) from Nncdm=1 to Nncdm=3 [degenerate] at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        return self.get_Tkbase(z=z, k=k)

class Tkmmhalofit_gp(Tkbase_gp):
    def __init__(self, verbose=False):
        self.verbose    = verbose
        ##### specify settings
        self.n_sample   = 513
        self.emunamestr = 'Tkmm_halofit_N3_N1'
        self.nvec       = 5
        super()._load_data()
    
    def get_Tkmmhalofit(self, z=None, k=None):
        '''
        Get the [mm] [halofit] matter power spectrum transformation T(k) from Nncdm=1 to Nncdm=3 [degenerate] at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        return self.get_Tkbase(z=z, k=k)
    
class Tkcbhmcode2020_gp(Tkbase_gp):
    def __init__(self, verbose=False):
        self.verbose    = verbose
        ##### specify settings
        self.n_sample   = 513
        self.emunamestr = 'Tkcb_hmcode2020_N3_N1'
        self.nvec       = 5
        super()._load_data()
    
    def get_Tkcbhmcode2020(self, z=None, k=None):
        '''
        Get the [cb] [hmcode2020] matter power spectrum transformation T(k) from Nncdm=1 to Nncdm=3 [degenerate] at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        return self.get_Tkbase(z=z, k=k)
    
class Tkmmhmcode2020_gp(Tkbase_gp):
    def __init__(self, verbose=False):
        self.verbose    = verbose
        ##### specify settings
        self.n_sample   = 513
        self.emunamestr = 'Tkmm_hmcode2020_N3_N1'
        self.nvec       = 5
        super()._load_data()
    
    def get_Tkmmhmcode2020(self, z=None, k=None):
        '''
        Get the [mm] [hmcode2020] matter power spectrum transformation T(k) from Nncdm=1 to Nncdm=3 [degenerate] at redshift z and wavenumber k.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        return self.get_Tkbase(z=z, k=k)


