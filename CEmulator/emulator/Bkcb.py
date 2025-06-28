import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, RBF 
from ..utils import cosmoNormLarge, data_path, zlists, checkdata, check_z, MyStandardScaler


class Bkcbbase_gp:
    zlists = zlists
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.emunamestr    = ''
        self.n_sample      = 129
        self.NormBeforeGP  = True
        self.kmax100       = False
        self._load_data()
   
    def _load_data(self):
        if self.verbose:
            print('Loading the %s emulator...'%self.emunamestr)
            print('Using %d training samples.'%self.n_sample)
        self.X_train = cosmoNormLarge[:self.n_sample,:]
        ### load karr
        if self.kmax100:
            self.klist = np.load(data_path + 'karr_kmax100.npy')
        else:
            self.klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
            kcut = 10.01
            ind = self.klist<=kcut
            self.klist = self.klist[ind] 
        ### load the PCA transformation matrix
        allsavedata = np.load(data_path + "%s.npz"%self.emunamestr, allow_pickle=True)
        self._PCA_mean       = allsavedata['pca_data'][0,:]
        self._PCA_components = allsavedata['pca_data'][1:,:]
        ### Load the Gaussian Process Regression model
        self._GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = allsavedata['gprinfo']
        pkcoeff    = allsavedata['Bcoeff']
        if self.NormBeforeGP:
            self.coeffSS   = MyStandardScaler()
            self.paramSS   = MyStandardScaler()
            pkcoeff        = self.coeffSS.fit_transform(pkcoeff)
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
        
    def get_data(self, z=None, k=None):
        z = check_z(self.zlists, z)
        k = checkdata(self.klist, k, dname='wavenumber')
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        Bkpred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            Bkpred[ivec] = self._GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            Bkpred = self.coeffSS.inverse_transform(Bkpred.reshape(1,-1))[0]
        ## PCA inverse transform
        if self.emunamestr[:2] == "lg":
            Bkpred = 10**((Bkpred @ self._PCA_components) + self._PCA_mean)
        else:   
            Bkpred = ((Bkpred @ self._PCA_components) + self._PCA_mean)
        Bkpred = Bkpred.reshape(len(self.zlists), len(self.klist))
        Bkpred = Bkpred[::-1,:]
        Bkout  = np.zeros((len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        spline = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred, 
                                        kx=3, ky=1)
        Bkout  = spline(z, k)
        return Bkout 

class Bkcb_gp(Bkcbbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'Bk_lin'
        self.n_sample      = 129
        self.nvec          = 20
        self.NormBeforeGP  = True
        self.kmax100       = False
        self._load_data()
        
    def get_Bk(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        Bk is defined as the ratio between nonlinear and linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        return super().get_data(z=z, k=k) 

class Bkcb_halofit_gp(Bkcbbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'Bk_halofit'
        self.n_sample      = 129
        self.nvec          = 20
        self.NormBeforeGP  = True
        self.kmax100       = False
        self._load_data()

    def get_Bk(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        Bk is defined as the ratio between nonlinear and linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        return super().get_data(z=z, k=k) 

class Bkcb_hmcode2020_gp(Bkcbbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'Bk_hmcode2020'
        self.n_sample      = 129
        self.nvec          = 20
        self.NormBeforeGP  = True
        self.kmax100       = False
        self._load_data()

    def get_Bk(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        Bk is defined as the ratio between HMcode-2020 and linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        ''' 
        return super().get_data(z=z, k=k) 
    
### Transfer function from Pklin to Pkhmcode
class Bkcb_lin2hmcode_gp(Bkcbbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'lgBk_lin2hmcode2020'
        self.n_sample      = 513
        self.nvec          = 20
        self.NormBeforeGP  = True
        self.kmax100       = True
        self._load_data()

    def get_Bk(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) at redshift z and wavenumber k.
        Bk is defined as the ratio between HMCODE and linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        ''' 
        return super().get_data(z=z, k=k)  