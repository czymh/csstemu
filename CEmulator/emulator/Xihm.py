import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..GaussianProcess.GaussianProcess import GaussianProcessRegressor, Constant, Matern
from ..utils import cosmoNorm, data_path, zlists, check_z

class XihmMassBin_gp:
    zstart = 7
    zlists = zlists[zstart:]
    massedges = np.array([13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.4, 15.0])
    nmassbin  = len(massedges) - 1
    def __init__(self, verbose=False):
        self.verbose = verbose
        n_sample = 65
        if self.verbose:
            print('Loading the XihmNL emulator...')
            print('Using %d training samples.'%n_sample)
        self.X_train = cosmoNorm[:n_sample,:]
        self.nvec = 5
        ### load the PCA transformation matrix
        pcafname = data_path + 'pca_mean_components_nvec%d_xihm_n%d_r_0.01_50.npy'%(self.nvec, n_sample)
        self.__PCA_Data = np.load(pcafname, allow_pickle=True)
        ### load karr
        rbins = np.concatenate([np.logspace(-2, 1, 30+1)[:-1],
                                np.arange(10, 500, 5)])
        rmid  = (rbins[1:] + rbins[:-1]) / 2
        rind  = (rmid <= 50 ) & (rmid >= 0.01 )
        self.rmid = rmid[rind]
        ### Load the Gaussian Process Regression model
        self.__GPR = np.zeros((self.nmassbin, self.nvec), dtype=object)
        gprinfo    = np.load(data_path + 'xihm_gpr_kernel_nvec%d_n%d_r_0.01_50.npy'%(self.nvec,n_sample), allow_pickle=True)
        Bkcoeff    = np.load(data_path + 'xihm_coeff_nvec%d_n%d_r_0.01_50.npy'%(self.nvec,n_sample), allow_pickle=True)
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
    
    def get_xihmMassBin(self, z=None, r=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        Bk is defined as the ratio between nonlinear and linear power spectrum.
        z : float or array-like, redshift
        r : float or array-like, wavenumber [Mpc/h]
        '''
        z = check_z(self.zlists, z)
        r = np.atleast_1d(r)
        ## Gaussian Process Regression
        Bkpred = np.zeros((self.nmassbin, self.nvec))
        Bkout0 = np.zeros((self.nmassbin, len(self.zlists), len(self.rmid)))
        for im in range(self.nmassbin):
            for ivec in range(self.nvec):
                Bkpred[im,ivec] = self.__GPR[im,ivec].predict(self.ncosmo)[0]
            ## PCA inverse transform
            Bkout0[im,:,:] = ((Bkpred[im,:] @ self.__PCA_Data[im][1:]) 
                           + self.__PCA_Data[im][0]).reshape(len(self.zlists), len(self.rmid))
        Bkout0 = Bkout0[:,::-1,:]  ## reverse the redshift axis
        Bkout = np.zeros((self.nmassbin, len(z), len(r)))
        ### z space use cubic spline while k space use linear interpolation
        for im in range(self.nmassbin):
            spline = RectBivariateSpline(self.zlists[::-1], np.log10(self.rmid), (Bkout0[im]), 
                                         kx=3, ky=3)
            Bkout[im] = (spline(z, np.log10(r)))
        return Bkout
    
    
