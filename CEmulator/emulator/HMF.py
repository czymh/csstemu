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
        self.nvec          = 5
        self.NormBeforePCA = True
        self.NormBeforeGP  = True
        self._load_data()
   
    def _load_data(self):
        if self.verbose:
            print('Loading the %s emulator...'%self.emunamestr[3:])
            print('Using %d training samples.'%self.n_sample)
        self.X_train = cosmoNorm[:self.n_sample,:]
        ### load karr
        Nbin = 60
        self.mstr  = '_Nbin%d'%Nbin
        self.m_edges = np.logspace(10, 16, Nbin+1) 
        self.mlow    = self.m_edges[:-1]
        self.mhmin_ind = 10 * np.ones(len(self.zlists), dtype=int)  # M_h > 1e11 Msun/h
        self.mhmax_ind = np.array([39, 41, 44, 45, 45, 49, 49, 50, 53, 55, 56, 57]) # c0000 M_h max (Not include)
        ### load the PCA transformation matrix
        _tmp = np.load(data_path + 'pca_mean_components_nvec%d_%s_n%d%s.npy'%(self.nvec, self.emunamestr, self.n_sample, self.mstr))
        self._PCA_mean       = _tmp[0,:]
        self._PCA_components = _tmp[1:,:]
        ### Load the Gaussian Process Regression model
        self._GPR = np.zeros(self.nvec, dtype=object)
        gprinfo    = np.load(data_path + '%s_gpr_kernel_nvec%d_n%d%s.npy'%(self.emunamestr, self.nvec, self.n_sample, self.mstr), allow_pickle=True)
        pkcoeff    = np.load(data_path + '%s_coeff_nvec%d_n%d%s.npy'%(self.emunamestr, self.nvec, self.n_sample, self.mstr))
        if self.NormBeforePCA:
            pcaSS_data = np.load(data_path + 'norm_before_pca_nvec%d_%s_n%d%s.npy'%(self.nvec, self.emunamestr, self.n_sample, self.mstr), allow_pickle=True).item()
            self.pcaSS_mean  = pcaSS_data['mean']
            self.pcaSS_scale = pcaSS_data['scale']
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
            ypred = 10**np.array([ypred[ivec] * self.pcaSS_scale[ivec] + self.pcaSS_mean[ivec] for ivec in range(len(self.pcaSS_scale))]).T
        return ypred
    
class HMFRockstarM200m_gp(HMFbase_gp):
    def __init__(self, verbose=False):
        self.verbose    = verbose
        self.emunamestr = 'cumhmf_m200m_rockstar'
        self.n_sample   = 129
        self.nvec       = 6
        self.NormBeforePCA = True
        self.NormBeforeGP  = True
        self._load_data()
        
    def get_data(self):
        return super().get_data()
    


