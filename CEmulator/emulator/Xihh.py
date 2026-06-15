import numpy as np
from scipy.interpolate import RectBivariateSpline, RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, Matern
from ..utils import cosmoNorm, data_path, zlists, check_z, MyStandardScaler
from scipy.interpolate import RegularGridInterpolator

class Xihhbase_gp:
    zlists    = zlists
    lgdenlist = np.linspace(-5, -2.5, 6)
    nden      = len(lgdenlist)
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.emunamestr    = ''
        self.n_sample      = 129
        self.nvec          = 5
        self.NormBeforePCA = True
        self.NormBeforeGP  = False
        ### load karr
        rbins = np.concatenate([np.logspace(-2, 1, 30+1)[:-1],
                                np.arange(10, 500, 5)])
        rmid  = (rbins[1:] + rbins[:-1]) / 2
        rind  = (rmid <= 50 ) & (rmid >= 0.1 )
        self.rmid = rmid[rind]
        self._load_data()
   
    def _load_data(self):
        if self.verbose:
            print('Loading the %s emulator...'%self.emunamestr[3:])
            print('Using %d training samples.'%self.n_sample)
        self.X_train = cosmoNorm[:self.n_sample,:] 
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
    
class Xihh_gp(Xihhbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'xihhNumBin_RockstarM200m'
        self.n_sample      = 129
        self.nvec          = 5
        self.NormBeforePCA = True
        self.NormBeforeGP  = False
        ### load rs
        rbins = np.concatenate([np.logspace(-2, 1, 30+1)[:-1],
                                np.arange(10, 500, 5)])
        rmid  = (rbins[1:] + rbins[:-1]) / 2
        rind  = (rmid <= 50 ) & (rmid >= 0.1 )
        self.rmid = rmid[rind]
        self._xihh_data_cache = None
        self._xihh_interp     = None
        self._load_data()
    
    def _get_xihh_data_matrix(self):
        if self._xihh_data_cache is not None:
            return self._xihh_data_cache

        ypred = self.get_data()          # 原始数据，形状 (nz, nauto * nr) ?
        nauto = (self.nden + 1) * self.nden // 2
        ypred = ypred.reshape(len(self.zlists), nauto, len(self.rmid))

        data_matrix = np.zeros((len(self.zlists), self.nden, self.nden, len(self.rmid)))
        ind = 0
        for id1 in range(self.nden):
            for id2 in range(id1, self.nden):
                data_matrix[:, id1, id2, :] = ypred[:, ind, :]
                data_matrix[:, id2, id1, :] = ypred[:, ind, :]
                ind += 1

        self._xihh_data_cache = data_matrix
        return data_matrix

    def _get_xihh_interpolator(self):
        if self._xihh_interp is not None:
            return self._xihh_interp

        data = self._get_xihh_data_matrix()          # (nz, nden, nden, nr)
        z_arr = np.array(self.zlists)

        # mesh must be increasing（RegularGridInterpolator requirement）
        if z_arr[0] > z_arr[-1]:
            z_asc = z_arr[::-1]
            data  = data[::-1, :, :, :]
        else:
            z_asc = z_arr

        lgden_asc = np.array(self.lgdenlist)
        rmid_asc  = np.array(self.rmid)

        # 构建四维插值器，方法可选 'linear' 或 'cubic'
        self._xihh_interp = RegularGridInterpolator(
            (z_asc, lgden_asc, lgden_asc, np.log10(rmid_asc)),
            data,
            method='linear',        
            bounds_error=False,
            fill_value=None
        )
        return self._xihh_interp
     
    def get_xihh(self, r, z, lgden1, lgden2):
        '''
        Get the halo-halo correlation function Xihh at radius r [Mpc/h], redshift z,
        '''
        interp = self._get_xihh_interpolator()          # 缓存的四维插值器
        z  = np.atleast_1d(z)
        lgden1 = np.atleast_1d(lgden1)
        lgden2 = np.atleast_1d(lgden2)
        r = np.atleast_1d(r)
        
        nz = len(z)
        nr = len(r)
        n1 = len(lgden1)
        n2 = len(lgden2)

        # 广播生成全组合网格坐标，形状 (nz, n1, n2, nr, 4)
        z_b  = z[:, np.newaxis, np.newaxis, np.newaxis]       # (nz, 1, 1, 1)
        l1_b = lgden1[np.newaxis, :, np.newaxis, np.newaxis]  # (1, n1, 1, 1)
        l2_b = lgden2[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, n2, 1)
        r_b  = np.log10(r)[np.newaxis, np.newaxis, np.newaxis, :]       # (1, 1, 1, nr)

        points = np.empty((nz, n1, n2, nr, 4))
        points[..., 0] = z_b
        points[..., 1] = l1_b
        points[..., 2] = l2_b
        points[..., 3] = r_b
        points = points.reshape(-1, 4)

        # 一次性四维插值
        out_flat = interp(points)                       # (nz * n1 * n2 * nr,)
        out_grid = out_flat.reshape(nz, n1, n2, nr)

        # 调整维度顺序为 (n1, n2, nz, nr)
        out = out_grid.transpose(1, 2, 0, 3)
        # r is increasing along axis -1, once out < -0.1, set that and all smaller r to -1
        trigger = (out < -0.1) * (r < 10.0)[None,None,None,:] # Mpc/h  # boolean array
        # Use reverse cumulative OR: flip, cumsum > 0, flip back
        mask = np.flip(np.cumsum(np.flip(trigger, axis=-1), axis=-1) > 0, axis=-1)
        out[mask] = -1
        return out   # 形状 (n1, n2, nz, nr)，即使 n1=n2=1 也保留这两个维度
