import numpy as np
import os
import inspect
from GaussianProcess import GaussianProcessRegressor, czConstant, czRBF
from scipy.interpolate import RectBivariateSpline
            
def NormCosmo(cosmologies, param_names, param_limits):
    '''
    Normalizes the cosmological parameters to the range [0, 1]
    cosmologies : array of shape (n_cosmologies, n_params)
    param_names : list of parameter names
    param_limits: dictionary with parameter names as keys and [min, max] as values
    '''
    ncosmo = np.zeros_like(cosmologies)
    for i, param in enumerate(param_names):
        ncosmo[:,i] = (cosmologies[:,i] - param_limits[param][0]) / (param_limits[param][1] - param_limits[param][0])
    return ncosmo

def useCLASS(mypara, strzlists, non_linear=None):
    from classy import Class
    param_names  = ['Omegab', 'Omegam', 'H0', \
                   'ns', 'A', 'w', 'wa', 'mnu']
    para = {}
    for i_n in range(len(param_names)):
        para[param_names[i_n]] = mypara[i_n]
    params = {
        'output': 'mPk', 
        'P_k_max_1/Mpc': 20,
        'A_s':       para['A']*1e-9,
        'n_s':       para['ns'], 
        'h':         para['H0']/100,
        'Omega_Lambda': 0.0, 
        'wa_fld':    para['wa'],
        'w0_fld':    para['w'],
        'Omega_b':   para['Omegab'],
        'Omega_cdm': para['Omegam'] - para['Omegab'],
        'Omega_Lambda': 0.0, 
        'z_pk': strzlists,
        'T_cmb': 2.7255,
        'gauge': 'synchronous'
    }
    if para['mnu'] == 0:
        params['N_ur'] = 3.046
    else:
        params['N_ur'] = 3.046 - 1.0132
        params['N_ncdm'] = 1
        params['m_ncdm'] = "{:f}".format(para['mnu'])
    if non_linear is not None:
        params['non linear'] = non_linear
    cosmo_class = Class()
    cosmo_class.set(params)
    cosmo_class.compute()
    return cosmo_class

class CEmulator:
    '''
    The CSST Emulator class for various statistics.
    '''
    param_names  = ['Omegab', 'Omegam', 'H0', \
                   'ns', 'A', 'w', 'wa', 'mnu']
    param_limits = {}
    param_limits['Omegab'] = [0.04, 0.06]
    param_limits['Omegam'] = [0.24, 0.40]
    param_limits['H0']     = [60,     80]
    param_limits['ns']     = [0.92, 1.00]
    param_limits['A']      = [1.7,   2.5]  # 1e-9
    param_limits['w']      = [-1.3, -0.7]
    param_limits['wa']     = [-0.5,  0.5]
    param_limits['mnu']    = [0,     0.3]
    ## emulator redshifts [0, 3]
    zlists = [3.00, 2.50, 2.00, 1.75, \
              1.50, 1.25, 1.00, 0.80, \
              0.50, 0.25, 0.10, 0.00]
    
    def __init__(self, statistic='Pkmm'):
        self.statistic = statistic
        data_path = data_path = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), 'data/')
        cosmoall = np.load(data_path + 'cosmologies_8d_train_n129_Sobol.npy')
        self.X_train = NormCosmo(cosmoall[:65,:], self.param_names, self.param_limits)
        if self.statistic == 'Pkmm':
            print('Loading the Pkmm emulator...')
            self.Pkmmload = True
            self.nvec = 10
            ### load the PCA transformation matrix
            _tmp = np.load(data_path + 'pca_mean_components_nvec%d_lgBk.npy'%self.nvec)
            self.__PCA_mean = _tmp[0,:]
            self.__PCA_components = _tmp[1:,:]
            ### load karr
            self.klist = np.load(data_path + 'karr_nb_Nmesh3072.npy')
            kcut = 10
            ind = self.klist<=kcut
            self.klist = self.klist[ind]
            ### Load the Gaussian Process Regression model
            self.__GPR = np.zeros(self.nvec, dtype=object)
            gprinfo    = np.load(data_path + 'lgBk_gpr_kernel_nvec%d_n65_nb_Nmesh3072.npy'%self.nvec, allow_pickle=True)
            Bkcoeff    = np.load(data_path + 'lgBk_coeff_nvec%d_n65_nb_Nmesh3072.npy'%self.nvec)
            for ivec in range(self.nvec):
                k1    = czConstant(gprinfo[ivec]['k1__constant_value'])
                k2    = czRBF(gprinfo[ivec]['k2__length_scale'])
                kivec = k1 * k2
                alpha = 1e-10
                ynorm = True
                self.__GPR[ivec] = GaussianProcessRegressor(self.X_train, Bkcoeff[:,ivec], 
                                                            kernel=kivec, alpha=alpha, normalize_y=ynorm)
        else:
            raise ValueError('Statistic %s not supported yet.'%statistic)
        #### some flags
        self.zcheckflag      = False
        self.kcheckflag      = False
        self.cosmo_class_arr = None
        
    def set_cosmos(self, Omegab=0.04897468, Omegam=0.30969282, 
                   H0=67.66, As=2.105e-9, ns=0.9665, w=-1.0, wa=0.0, 
                   mnu=0.06, z=0.0):
        '''
        Set the cosmological parameters. You can input the float or array-like.
        '''
        cosmos = {}
        cosmos['Omegab'] = np.atleast_1d(Omegab)
        cosmos['Omegam'] = np.atleast_1d(Omegam)
        cosmos['H0']     = np.atleast_1d(H0)
        cosmos['ns']     = np.atleast_1d(ns)
        cosmos['A']      = np.atleast_1d(As*1e9)
        cosmos['w']      = np.atleast_1d(w)
        cosmos['wa']     = np.atleast_1d(wa)
        cosmos['mnu']    = np.atleast_1d(mnu)
        n_params = len(self.param_names)
        ## check the parameter range
        lenlists = np.zeros((n_params), dtype=int)
        for ind, ikey in enumerate(self.param_names):
            if np.any(cosmos[ikey] < self.param_limits[ikey][0]) or \
               np.any(cosmos[ikey] > self.param_limits[ikey][1]):
                raise ValueError('Parameter %s out of range.'%ikey)
            lenlists[ind] = len(cosmos[ikey])
        numcosmos = np.unique(lenlists)
        if numcosmos.size > 1:
            raise ValueError('Inconsistent parameter array length.')
        numcosmos = numcosmos[0]
        ## into the cosmologies array
        self.cosmologies = np.zeros((numcosmos, n_params))
        for ind, ikey in enumerate(self.param_names):
            self.cosmologies[:,ind] = cosmos[ikey]
            
    def check_z(self, z=None):
        if self.zcheckflag == False:
            if z is None:
                z = self.zlists
                print('No redshift input, using default redshifts [0,3]-12.')
            self.zinput = np.atleast_1d(z)
            if np.any(self.zinput < 0) or np.any(self.zinput > 3):
                raise ValueError('Redshift z out of range [0, 3].')
            self.zinput.sort()
            print('Predicting redshifts (sorted):', self.zinput)
            self.zcheckflag = True
        
    def check_k(self, k=None):
        if self.kcheckflag == False:
            if k is None:
                raise ValueError('Please provide the wavenumber k [h/Mpc].')
            self.kinput = np.atleast_1d(k)
            if np.any(self.kinput < self.klist[0]) or np.any(self.kinput > self.klist[-1]):
                raise ValueError('Wavenumber k out of range.')
            if not np.all(np.diff(self.kinput) > 0.0):
                raise ValueError('k must be strictly increasing!')
            self.kcheckflag = True
        
    def get_Bk(self, z=None, k=None):
        '''
        Get the matter power spectrum B(k) [Mpc/h]^3 at redshift z and wavenumber k.
        Bk is defined as the ratio between nonlinear and linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        '''
        if self.Pkmmload == False:
            raise ValueError('The Pkmm emulator is not loaded.')
        ## check the input redshift and wavenumber
        self.check_z(z)
        self.check_k(k)

        numcos = self.cosmologies.shape[0]
        print('Predicting Bk of %d cosmologies...'%numcos)
        ncosmo = NormCosmo(self.cosmologies, self.param_names, self.param_limits)
        ## Gaussian Process Regression
        Bkpred = np.zeros((numcos, self.nvec))
        for ivec in range(self.nvec):
            Bkpred[:,ivec] = self.__GPR[ivec].predict(ncosmo)
        ## PCA inverse transform
        Bkpred = 10**((Bkpred @ self.__PCA_components) + self.__PCA_mean)
        Bkpred = Bkpred.reshape(numcos, len(self.zlists), len(self.klist))
        Bkpred = Bkpred[:,::-1,:]
        Bkout  = np.zeros((Bkpred.shape[0], len(self.zinput), len(self.kinput)))
        ### z space use cubic spline while k space use linear interpolation
        for ic in range(numcos):
            spline    = RectBivariateSpline(self.zlists[::-1], self.klist, Bkpred[ic], 
                                            kx=3, ky=1)
            Bkout[ic] = spline(self.zinput, self.kinput)
        return Bkout
    
    def get_cosmos_class(self, z=None, non_linear=None):
        '''
        Get the CLASS cosmology object.
        z : float or array-like, redshift
        non_linear: 'halofit' or 'HMcode'
        '''
        self.check_z(z)
        
        numcos = self.cosmologies.shape[0]
        str_zlists = "{:.4f}".format(self.zinput[0])
        if len(self.zinput) > 1:
            for i_z in range(len(self.zinput) - 1):
                str_zlists += ", {:.4f}".format(self.zinput[i_z+1])
        self.cosmo_class_arr = np.zeros((numcos,), dtype=object)
        for ic in range(numcos):
            self.cosmo_class_arr[ic] = useCLASS(self.cosmologies[ic], str_zlists, non_linear=non_linear)
        return self.cosmo_class_arr
    
    def get_pklin(self, z=None, k=None, type='CLASS', Pcb=True):
        '''
        Get the linear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        type : string, 'CLASS' or 'symbolic_pofk'
        '''
        ## check the input redshift and wavenumber
        self.check_z(z)
        self.check_k(k)
        
        if   type == 'CLASS':
            if self.cosmo_class_arr is None:
                self.get_cosmos_class(z)
            numcos = self.cosmologies.shape[0]
            if Pcb and self.cosmo_class_arr[0].Omega_nu != 0:
                print('Only cb-component linear power spectrum is calculated.')
                pkfunc = self.cosmo_class_arr[0].pk_cb_lin
            else:
                pkfunc = self.cosmo_class_arr[0].pk_lin
            pklin = np.zeros((numcos, len(self.zinput), len(self.kinput)))
            for ic in range(numcos):
                h0 = self.cosmo_class_arr[ic].h()
                for iz in range(len(self.zinput)):
                    pklin[ic, iz] = np.array([pkfunc(ik*h0, self.zinput[iz])*h0*h0*h0 
                                                for ik in list(self.kinput)])
        else:
            raise ValueError('Type %s not supported yet.'%type)
        return pklin
    
    def get_pknl(self, z=None, k=None, Pcb=True, lintype='CLASS'):
        '''
        Get the nonlinear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        type : string, 'CLASS' or 'symbolic_pofk'
        '''
        ## get the linear power spectrum
        if   lintype == 'CLASS':
            pklin = self.get_pklin(z, k, type=lintype, Pcb=Pcb)
        else:
            raise ValueError('Type %s not supported yet.'%lintype)
        ## get the nonlinear transfer
        Bkpred = self.get_Bk(z, k)
        pknl = pklin * Bkpred
        return pknl
        
        
        
        

    
    
