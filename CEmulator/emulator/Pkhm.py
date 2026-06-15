import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d, RegularGridInterpolator
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, Matern
from ..utils import cosmoNorm, data_path, zlists, check_z, MyStandardScaler, checkdata

class Pkhmbase_gp:
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
        self._load_data()
   
    def _load_data(self):
        if self.verbose:
            print('Loading the %s emulator...'%self.emunamestr)
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
    
class bhmRockstarM200m_gp(Pkhmbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'bhmNumBin_RockstarM200m'
        self.n_sample      = 129
        self.nvec          = 10
        self.NormBeforePCA = True
        self.NormBeforeGP  = False
        self._load_data()
        
    def get_bias_ratio(self, z, lgden):
        z     = np.atleast_1d(z)
        lgden = np.atleast_1d(lgden)
        ypred = self.get_data()
        ypred = ypred.reshape(len(self.zlists), self.nden)
        spline = RectBivariateSpline(self.zlists[::-1], self.lgdenlist, ypred[::-1,:], kx=1, ky=1)
        return spline(z, lgden) # shape (len(z), len(lgden))
    
class BkhmRockstarM200m_gp(Pkhmbase_gp):
    def __init__(self, verbose=False):
        self.verbose       = verbose
        self.emunamestr    = 'BkhmNumBin_RockstarM200m'
        self.n_sample      = 129
        self.nvec          = 10
        self.NormBeforePCA = True
        self.NormBeforeGP  = False
        self.klist         = np.load(data_path + 'karr_nb_Nmesh3072.npy')[:1688]
        self.klist         = self.klist[(self.klist > 0.08)]
        self._Bkhm_interp  = None
        self._load_data()
    
    def _get_Bkhm_interpolator(self):
        """
        Build and cache a RegularGridInterpolator over (z, lgden, k).
        """
        if self._Bkhm_interp is not None:
            return self._Bkhm_interp

        ypred = self.get_data()
        ypred = ypred.reshape(len(self.zlists), self.nden, len(self.klist))
        # interpolation requires ascending order
        z_arr     = np.array(self.zlists[::-1])
        lgden_arr = np.array(self.lgdenlist)
        k_arr     = np.array(self.klist)

        self._Bkhm_interp = RegularGridInterpolator(
            (z_arr, lgden_arr, np.log10(k_arr)),   # coordinate order must match data axes
            ypred[::-1,:,:],                      # reverse z-axis to match ascending order
            method='linear',              # cubic in z and lgden, linear in k (cubic in k causes issues)
            bounds_error=False,
            fill_value=None              # extrapolate outside grid
        )
        return self._Bkhm_interp
     
    def get_Bkhm(self, k, z, lgden):
        """
        Get the halo-mass cross-power spectrum Bkhm.

        Parameters
        ----------
        k : array_like
            Wavenumbers [h/Mpc].
        z : array_like
            Redshifts, scalar or array.
        lgden : array_like
            Log10 halo number density threshold(s). Scalar or array.

        Returns
        -------
        out : ndarray, shape (len(lgden), len(z), len(k))
            Bkhm evaluated on the grid (lgden, z, k).
        """
        k     = checkdata(self.klist, k, dname='wavenumber')
        z     = np.atleast_1d(z)
        lgden = np.atleast_1d(lgden).astype(float)

        interp = self._get_Bkhm_interpolator()

        nlg = len(lgden)
        nz  = len(z)
        nk  = len(k)

        # Build full query grid with axes order (lgden, z, k)
        lg_grid, z_grid, lgk_grid = np.meshgrid(lgden, z, np.log10(k), indexing='ij')  # each (nlg, nz, nk)

        # Pack points in the exact coordinate order of the interpolator: (z, lgden, k)
        points = np.empty((nlg, nz, nk, 3))
        points[..., 0] = z_grid
        points[..., 1] = lg_grid
        points[..., 2] = lgk_grid
        points = points.reshape(-1, 3)

        out_flat = interp(points)
        out = out_flat.reshape(nlg, nz, nk)
        return out # shape (len(lgden), len(z), len(r))

