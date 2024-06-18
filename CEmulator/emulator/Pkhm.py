import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, Matern
from ..utils import cosmoNorm, data_path, zlists, check_k, check_z

class PkhmMassBin_gp:
    zstart = 7
    zlists_reduce = zlists[zstart:]
    massedges = np.array([13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.4, 15.0])
    nmassbin  = len(massedges) - 1
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 65
        if self.verbose:
            print('Loading the PkhmNL emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNorm[:n_sample,:]
        self.nvec = 5
        ### load the PCA transformation matrix
        pcafname = data_path + 'pca_mean_components_nvec%d_lgpkhm_n%d_kmax_1.0.npy'%(self.nvec, n_sample)
        self.__PCA_Data = np.load(pcafname, allow_pickle=True)
        ### load karr
        self.klist = np.load(data_path + 'karr_nb_Nmesh1536_nmerge8.npy')
        kcut = 1.0
        ind = self.klist<=kcut
        self.klist = self.klist[ind]
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros((self.nmassbin, self.nvec), dtype=object)
        gprinfo    = np.load(data_path + 'lgpkhm_gpr_kernel_nvec%d_n%d_kmax_1.0.npy'%(self.nvec,n_sample), allow_pickle=True)
        Bkcoeff    = np.load(data_path + 'lgpkhm_coeff_nvec%d_n%d_kmax_1.0.npy'%(self.nvec,n_sample), allow_pickle=True)
        for im in range(self.nmassbin):
            for ivec in range(self.nvec):
                k1    = Constant(gprinfo[im][ivec]['k1__constant_value'])
                k2    = Matern(gprinfo[im][ivec]['k2__length_scale'])
                kivec = k1 * k2
                alpha = 1e-10
                ynorm = True
                self.__GPR[im, ivec] = GaussianProcessRegressor(self.X_train, Bkcoeff[im][:,ivec], 
                                                                kernel=kivec, alpha=alpha, 
                                                                normalize_y=ynorm)
        ### End of __init__
    
    def get_pkhmMassBin(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        Bk is defined as the ratio between nonlinear and linear power spectrum.
        z : float or array-like, redshift
        r : float or array-like, wavenumber [Mpc/h]
        '''
        z = check_z(self.zlists_reduce, z)
        k = check_k(self.klist,  k)
        
        ## Gaussian Process Regression
        Bkpred = np.zeros((self.nmassbin, self.nvec))
        Bkout0 = np.zeros((self.nmassbin, len(self.zlists_reduce), len(self.klist)))
        for im in range(self.nmassbin):
            for ivec in range(self.nvec):
                Bkpred[im,ivec] = self.__GPR[im,ivec].predict(self.ncosmo)[0]
            ## PCA inverse transform
            Bkout0[im,:,:] = ((Bkpred[im,:] @ self.__PCA_Data[im][1:]) 
                           + self.__PCA_Data[im][0]).reshape(len(self.zlists_reduce), len(self.klist))
        Bkout0 = 10**Bkout0[:,::-1,:]  ## reverse the redshift axis
        Bkout = np.zeros((self.nmassbin, len(z), len(k)))
        ### z space use cubic spline while k space use linear interpolation
        for im in range(self.nmassbin):
            spline    = RectBivariateSpline(self.zlists_reduce[::-1], np.log10(self.klist), (Bkout0[im]), 
                                            kx=3, ky=3)
            Bkout[im] = (spline(z, np.log10(k)))
        return Bkout
    
    
