import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, RBF, Matern
from ..utils import cosmoNorm, data_path, zlists, check_z, MyStandardScaler


class HMFbase_gp:
    zlists = zlists
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.emunamestr    = ''
        self.n_sample      = 129
        self.NormBeforePCA = True
        self.NormBeforeGP  = True
        self._load_data()
   
    def _load_data(self):
        if self.verbose:
            print('Loading the %s emulator...'%self.emunamestr[3:])
            print('Using %d training samples.'%self.n_sample)
        self.X_train = cosmoNorm[:self.n_sample,:]
        ### load karr
        Nbin         = 60
        self.mstr    = '_Nbin%d'%Nbin
        self.m_edges = np.logspace(10, 16, Nbin+1) 
        self.mlow    = self.m_edges[:-1]
        self.mhmin_ind = 10 * np.ones(len(self.zlists), dtype=int)  # M_h > 1e11 Msun/h
        ### load the PCA transformation matrix
        allsavedata = np.load(data_path + "%s.npz"%self.emunamestr, allow_pickle=True)
        self._PCA_mean       = allsavedata['pca_data'][0,:]
        self._PCA_components = allsavedata['pca_data'][1:,:]
        ### Load the Gaussian Process Regression model
        self._GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = allsavedata['gprinfo']
        pkcoeff    = allsavedata['Bcoeff']
        if self.NormBeforePCA:
            self.pcaSS_mean  = allsavedata['pcaSS_data'][0,:]
            self.pcaSS_scale = allsavedata['pcaSS_data'][1,:] 
        if self.NormBeforeGP:
            self.pkcoeffSS = MyStandardScaler()
            self.paramSS   = MyStandardScaler()
            pkcoeff        = self.pkcoeffSS.fit_transform(pkcoeff)
            self.X_train   = self.paramSS.fit_transform(self.X_train)
        for ivec in range(self.nvec):
            k1    = Constant(gprinfo[ivec]['k1__constant_value'])
            k2    = Matern(gprinfo[ivec]['k2__length_scale'], nu=gprinfo[ivec]['k2__nu'])
            kivec = k1 * k2
            alpha = 1e-10
            ynorm = True
            self._GPR[ivec] = GaussianProcessRegressor(self.X_train, pkcoeff[:,ivec], 
                                                       kernel=kivec, alpha=alpha, normalize_y=ynorm)
        ### End of __init__    
        
    def get_data(self):
        '''
        Get the base Number of haloes at redshift z and halo mass M.
        '''
        if self.NormBeforeGP:
            Normcosmo = self.paramSS.transform(self.ncosmo)
        else:
            Normcosmo = np.copy(self.ncosmo)
        ## Gaussian Process Regression
        ypred = np.zeros((self.nvec))
        for ivec in range(self.nvec):
            ypred[ivec] = self._GPR[ivec].predict(Normcosmo)
        if self.NormBeforeGP:
            ypred = self.pkcoeffSS.inverse_transform(ypred.reshape(1,-1))[0]
        ## PCA inverse transform
        ypred  = ((ypred @ self._PCA_components) + self._PCA_mean)
        if self.NormBeforePCA:
            ypred = ypred * self.pcaSS_scale + self.pcaSS_mean
        return ypred
    
class HMFRockstarM200m_gp(HMFbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'cumhmf_rockstar_M200m'
        self.n_sample      = 129
        self.nvec          = 10
        self.NormBeforePCA = True
        self.NormBeforeGP  = True
        self.mhmax_ind = np.array([36,38,40,41,43,45,46,48,50,52,53,53]) # mean M_h max
        self._load_data()
        
    def get_data(self):
        return super().get_data()
    
class HMFFoFM200c_gp(HMFbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'cumhmf_fof_M200c'
        self.n_sample      = 129
        self.nvec          = 10
        self.NormBeforePCA = True
        self.NormBeforeGP  = True
        # self.mhmax_ind = np.array([39,41,43,44,45,48,49,52,53,54,54,54]) # c0000 M_h max
        self.mhmax_ind = np.array([36,38,40,41,43,44,46,48,49,51,51,51]) # mean M_h max 
        self._load_data()
        
    def get_data(self):
        return super().get_data()

class HMFRockstarMvir_gp(HMFbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'cumhmf_rockstar_Mvir'
        self.n_sample      = 129
        self.nvec          = 10
        self.NormBeforePCA = True
        self.NormBeforeGP  = True
        self.mhmax_ind = np.array([36,38,40,41,42,44,45,47,49,50,52,52]) # mean M_h max
        self._load_data()
        
    def get_data(self):
        return super().get_data() 

