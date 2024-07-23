import numpy as np
import warnings
from scipy.integrate import trapz
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import savgol_filter
from .cosmology import Cosmology
from .emulator.Bkmm import Bkmm_gp, Bkmm_halofit_gp
from .emulator.PkcbLin import PkcbLin_gp
from .emulator.Xihm import XihmMassBin_gp
from .emulator.Pkhm import PkhmMassBin_gp
from .hankl import P2xi
from .utils import *

class CEmulator:
    '''
    The CSST Emulator class for various statistics.
    '''
    param_names  = param_names
    param_limits = param_limits
    zlists       = zlists    
    def __init__(self, verbose=False):
        '''
        Initialize the CSST Emulator class.
        
        Args:
            verbose : bool, whether to output the running information
        
        '''
        self.verbose = verbose
        self.Cosmo        = Cosmology(verbose=verbose)
        self.Bkmm         = Bkmm_gp(verbose=verbose) 
        self.Bkmm_halofit = Bkmm_halofit_gp(verbose=verbose) 
        self.Pkmmlin      = PkcbLin_gp(verbose=verbose) 
        self.XihmMassBin  = XihmMassBin_gp(verbose=verbose) 
        self.PkhmMassBin  = PkhmMassBin_gp(verbose=verbose)  
        
        
    def set_cosmos(self, Omegab=0.049, Omegac=0.26, 
                   H0=67.66, As=None, sigma8=None, 
                   ns=0.9665, w=-1.0, wa=0.0, 
                   mnu=0.06):
        '''
        Set the cosmological parameters.
        
        Args:
            Omegab : float, baryon density
            Omegac : float, CDM density
            H0     : float, Hubble constant
            As     : float, amplitude of the primordial power spectrum
            ns     : float, spectral index
            w      : float, dark energy equation of state
            wa     : float, dark energy equation of state evolution
            mnu    : float, sum of neutrino masses with unit eV
            sigma8 : float, amplitude of the total matter power spectrum. If both As and sigma8 are provided, the As will be used. You can set As=None to activate sigma8.
        '''
        cosmos = {}
        cosmos['Omegab'] = (Omegab)
        cosmos['Omegac'] = (Omegac)
        cosmos['H0']     = (H0)
        cosmos['ns']     = (ns)
        cosmos['w']      = (w)
        cosmos['wa']     = (wa)
        cosmos['mnu']    = (mnu)
        cosmos['Omegam'] = Omegab + Omegac # Only CDM + baryon
        cosmos.pop('Omegac')
        n_params = len(self.param_names)
        ## check the parameter range / except As 
        for ind, ikey in enumerate(self.param_names):
            if ikey != 'A':
                if   np.any(cosmos[ikey] > self.param_limits[ikey][1]):
                    raise ValueError(r'Parameter out of range %s = %.4f > %f.'%(ikey, cosmos[ikey], self.param_limits[ikey][1]))
                elif np.any(cosmos[ikey] < self.param_limits[ikey][0]):
                    raise ValueError(r'Parameter out of range %s = %.4f < %f.'%(ikey, cosmos[ikey], self.param_limits[ikey][0])) 

        if As is not None:
            cosmos['A']      = (As*1e9)
            if sigma8 is not None:
                warnings.warn('Both As and sigma8 are provided, the As will be used and ignore the sigma8 value.', UserWarning)
        elif sigma8 is not None:
            ### convert sigma8 to As
            sigma8_g = 0.0
            As       = 2.105
            abserr   = 1e-4
            while np.abs(sigma8_g - sigma8) > abserr:
                cosmos['A'] = As
                cosmologies_g = np.zeros((1, n_params))
                for ind, ikey in enumerate(self.param_names):
                    cosmologies_g[0,ind] = cosmos[ikey]
                # ncosmo_g = NormCosmo(cosmologies_g, self.param_names, self.param_limits)
                # self.Pkmmlin.ncosmo = ncosmo_g
                # sigma8_g = self.get_sigma8(type='Emulator')
                
                ## only use the CLASS to calculate the sigma8
                ## this is slow so we will replace it with the Emulator in the future.
                self.cosmologies = cosmologies_g
                sigma8_g = self.get_sigma8(type='CLASS') 
                # print('Guess sigma8:', sigma8_g)
                As = As * sigma8*sigma8/sigma8_g/sigma8_g
            if self.verbose:
                print('The As is set to %.6e (sigma8=%.6f) to match the input sigma8=%.6f.'%(cosmos['A']*1e-9, sigma8_g, sigma8)) 
        else:
            raise ValueError('Both As and sigma8 are None, please provide one of them at least.')
        # Omeganu = cosmos['mnu']/93.14/cosmos['H0']/cosmos['H0'] * 1e4
        ## check the parameter range for As
        for ikey in ['A']:
            if   np.any(cosmos[ikey] > self.param_limits[ikey][1]):
                raise ValueError(r'Parameter out of range %ss1e9 = %.4f > %f.'%(ikey, cosmos[ikey], self.param_limits[ikey][1]))
            elif np.any(cosmos[ikey] < self.param_limits[ikey][0]):
                raise ValueError(r'Parameter out of range %ss1e9 = %.4f < %f.'%(ikey, cosmos[ikey], self.param_limits[ikey][0])) 
        ### set the cosmology class
        self.Cosmo.set_cosmos(cosmos)
        ## into the cosmologies array only One cosmology each tim
        self.cosmologies = np.zeros((1, n_params))
        for ind, ikey in enumerate(self.param_names):
            self.cosmologies[0,ind] = cosmos[ikey]
        ### normalize the cosmologies with the shape (1, n_params)
        self.ncosmo = NormCosmo(self.cosmologies, self.param_names, self.param_limits)
        
        ### pass the cosmologies (normalized) to other class
        self.Bkmm.ncosmo         = self.ncosmo 
        self.Bkmm_halofit.ncosmo = self.ncosmo
        self.Pkmmlin.ncosmo      = self.ncosmo  
        self.XihmMassBin.ncosmo  = self.ncosmo
        self.PkhmMassBin.ncosmo  = self.ncosmo
        
        #### from init move to here 
        #### refresh the cosmology class for each cosmology set 
        # cosmo_class = None ## Not store this in the class
                    
    def get_cosmos_class(self, z=None, non_linear=None, kmax=10):
        '''
        Get the CLASS cosmology object.
        
        Args:
            z         : float or array-like, redshift
            non_linear: string, 'halofit' or 'HMcode'
        '''
        # check_z(z)
        z = np.atleast_1d(z)
        str_zlists = "{:.4f}".format(z[0])
        if len(z) > 1:
            for i_z in range(len(z) - 1):
                str_zlists += ", {:.4f}".format(z[i_z+1])
        cosmo_class = useCLASS(self.cosmologies[0], str_zlists, non_linear=non_linear, kmax=kmax)
        return cosmo_class
    
    def get_pklin(self, z=None, k=None, Pcb=True, type='CLASS', cosmo_class=None):
        '''
        Get the linear power spectrum.
        
        Args:
            z : float or array-like, redshift
            k : float or array-like, wavenumber with unit of [h/Mpc]
            Pcb  : bool, whether to output the total power spectrum (if False) or the cb power spectrum (if True [default])
            type : string, 'CLASS' or 'Emulator'
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
        Return:
            array-like : linear power spectrum with shape (len(z), len(k))
        '''
        z = check_z(self.zlists,     z)
        # k = check_k(self.Bkmm.klist, k)
        if   type == 'CLASS':
            if cosmo_class is None:
                kmax = np.max(k)
                cosmo_class = self.get_cosmos_class(z, kmax=kmax)
            pklin = np.zeros((len(z), len(k)))
            if Pcb and cosmo_class.Omega_nu != 0:
                pkfunc = cosmo_class.pk_cb_lin
            else:
                pkfunc = cosmo_class.pk_lin
            h0 = cosmo_class.h()
            for iz in range(len(z)):
                pklin[iz] = np.array([pkfunc(ik*h0, z[iz])*h0*h0*h0 
                                      for ik in list(k)])
        elif type == 'Emulator':
            if Pcb:
                pklin = self.Pkmmlin.get_pkLin(z, k)
            else:
                raise ValueError('For total power spectrum, use type=CLASS NOW.')     
        else:
            raise ValueError('Type %s not supported yet.'%type)
        return pklin
    
    def get_sigma_z(self, z=None, R=None, type='CLASS', cosmo_class=None):
        '''
        Get the sigma(z, R) of tot matter changing with the redshift.
        
        Args:
            z : float, redshift
            R : float, smoothing scale [Mpc/h]
            type : string, 'CLASS' or 'Emulator'
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
        Return:
            float : sigma8 value with shape (len(z))
        '''
        if not isinstance(z, (int, float)):
            raise ValueError('Only support one redshift now.')
        if not isinstance(R, (int, float)):
            raise ValueError('Only support one smoothing scale now.')

        if type == 'CLASS':
            # h0 = self.Cosmo.h0
            if cosmo_class is None:
                cosmo_class = self.get_cosmos_class(z)
            h0 = cosmo_class.h()
            sigma = cosmo_class.sigma(R/h0, z)
        elif type == 'Emulator':
            raise ValueError('For sigma8, we need add the neutrino power emulation.')
        else:
            raise ValueError('Type %s not supported yet.'%type)
        return sigma
    
    def get_sigma_cb_z(self, z=None, R=None, type='CLASS', cosmo_class=None):
        '''
        Get the sigma_cb(z, R) of tot matter changing with the redshift.
        
        Args:
            z : float, redshift
            R : float, smoothing scale [Mpc/h]
            type : string, 'CLASS' or 'Emulator'
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
        Return:
            float : sigma8 value with shape (len(z))
        '''
        if not isinstance(z, (int, float)):
            raise ValueError('Only support one redshift now.')
        if not isinstance(R, (int, float)):
            raise ValueError('Only support one smoothing scale now.')

        if type == 'CLASS':
            # h0 = self.Cosmo.h0
            if cosmo_class is None:
                cosmo_class = self.get_cosmos_class(z)
            h0 = cosmo_class.h()
            sigma_cb = cosmo_class.sigma_cb(R/h0, z)
        elif type == 'Emulator':
            kcalc    = np.logspace(-5, 2, 10000)
            W_R      = 3*(np.sin(kcalc*R) - kcalc*R*np.cos(kcalc*R))/(kcalc*R)**3
            pcalc    = self.Pkmmlin.get_pkLin(z, kcalc)[0]
            sigma_cb = np.sqrt(trapz(pcalc*W_R*W_R*kcalc*kcalc*kcalc/2/np.pi/np.pi, np.log(kcalc)))
        else:
            raise ValueError('Type %s not supported yet.'%type)
        return sigma_cb
    
    def get_sigma8(self, type='CLASS', cosmo_class=None):
        '''
        Get the sigma8 of tot matter.
        
        Args:
            type : string, 'CLASS' or 'Emulator'
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
        
        '''
        return self.get_sigma_z(0, 8.0, type=type, cosmo_class=cosmo_class)
    
    def get_sigma8_cb(self, type='CLASS', cosmo_class=None):
        '''
        Get the sigma8 of cb matter.
        
        Args:
            type : string, 'CLASS' or 'Emulator'
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
        
        '''
        return self.get_sigma_cb_z(0, 8.0, type=type, cosmo_class=cosmo_class)

    def _get_Pcurv(self, k=None):
        As = self.Cosmo.As
        ns = self.Cosmo.ns
        h0 = self.Cosmo.h0
        kp = 0.05 ## Mpc^{-1}
        Pcurv = lambda k: 2*np.pi*np.pi*As * (h0*k/kp)**(ns-1)/k/k/k
        return Pcurv(k)
    
    def get_pkhalofit(self, z=None, k=None, Pcb=True, lintype='CLASS', cosmo_class=None):
        '''
        Get the halofit power spectrum.
       
        .. note:: 
            This version can not converge with CLASS in the high redshift (z>2.5)
            for the c0001 and c0091 (w0 and wa near the lower limit).
        
       
        Args:
            z           : float or array-like, redshift 
            k           : float or array-like, wavenumber [h/Mpc]
            Pcb         : bool, whether to output the total power spectrum (if False) or the cb power spectrum (if True [default])
            lintype     : string, 'CLASS' or 'Emulator'
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.    
        Return: 
            array-like: halofit power spectrum with shape (len(z), len(k))
         
        '''
        z = check_z(self.zlists,     z)
        if  lintype == 'Emulator':
            if Pcb:
                fnu = 0.0
            else:
                fnu = self.Cosmo.Omeganu/(self.Cosmo.Omegam + self.Cosmo.Omeganu)
                raise ValueError('For total power spectrum, use type=CLASS NOW.')
            OmegaLzall   = self.Cosmo.get_OmegaL(z)
            OmegaMzall   = self.Cosmo.get_OmegaM(z)
            pkhalofit    = np.zeros((len(z), len(k)))
            kinterp      = np.logspace(-5, 1, 1024)[1:]
            pklinintp    = self.get_pklin(z, kinterp, Pcb=Pcb, type=lintype)
            for iz in range(len(z)):
                OmegaLz = OmegaLzall[iz]
                OmegaMz = OmegaMzall[iz]
                w_eff   = self.Cosmo.w0 + self.Cosmo.wa*(z[iz]/(1+z[iz]))
                Pklin = lambda k: 10**interp1d(np.log10(kinterp), 
                                               np.log10(pklinintp[iz]),
                                               kind='slinear', 
                                               fill_value="extrapolate")(np.log10(k))
                R_sigma, neff, Curv = compute_Rsigma_neff_C(Pklin)
                an = 10.**( 1.5222 + 2.8553*neff + 2.3706*neff*neff
                + 0.9903*neff*neff*neff + 0.2250*neff*neff*neff*neff \
                - 0.6038*Curv + 0.1749*OmegaLz*(1.+w_eff) ) 
                bn = 10.**(-0.5642 + 0.5864*neff + 0.5716*neff*neff \
                - 1.5474*Curv + 0.2279*OmegaLz*(1.+w_eff) ) 
                cn = 10.**( 0.3698 + 2.0404*neff + 0.8161*neff*neff \
                + 0.5869*Curv)
                gamman = 0.1971 - 0.0843*neff + 0.8460*Curv  
                alphan = np.abs( 6.0835 + 1.3373*neff - 0.1959*neff*neff - 5.5274*Curv)
                betan  = 2.0379 - 0.7354*neff + 0.3157*neff*neff \
                    + 1.2490*neff*neff*neff + 0.3980*neff*neff*neff*neff \
                    - 0.1682*Curv \
                    + fnu*(1.081 + 0.395*neff*neff)
                mun    = 0.
                nun    = 10.**(5.2105 + 3.6902*neff)
                pkhalofit[iz] = PkHaloFit(k, Pklin(k), R_sigma, OmegaMz, OmegaLz, fnu, \
                                          an, bn, cn, gamman, alphan, betan, mun, nun)
        else:
            if cosmo_class is None:
                cosmo_class = self.get_cosmos_class(z, non_linear='halofit')
            if (Pcb) and (cosmo_class.Omega_nu != 0):
                pkfunc = cosmo_class.pk_cb
            else:
                pkfunc = cosmo_class.pk
            pkhalofit = np.zeros((len(z), len(k)))
            h0 = cosmo_class.h()
            for iz in range(len(z)):
                pkhalofit[iz] = np.array([pkfunc(ik*h0, z[iz])*h0*h0*h0 
                                          for ik in list(k)])
        return pkhalofit
        
    def get_pknl(self, z=None, k=None, Pcb=True, lintype='CLASS', nltype='linear', cosmo_class=None):
        '''
        Get the nonlinear power spectrum.
        
        Args:
            z           : float or array-like, redshift. 
            k           : float or array-like, wavenumber [h/Mpc]. 
            lintype     : string, 'CLASS' or 'Emulator'. 
            nltype      : string, 'linear' or 'halofit'.  'linear' means ratio of nonlinear to linear power spectrum. 'halofit' means ratio of nonlinear to halofit power spectrum.
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
        
        .. note::
           For now, nltype = 'halofit' can give a better result than nltype = 'linear', especially for the high redshift.
        
        
        Return:
            array-like : nonlinear power spectrum with shape (len(z), len(k))
        '''
        z = check_z(self.zlists,     z)
        k = check_k(self.Bkmm.klist, k)
        ## get the nonlinear transfer for Pcb
        if   nltype == 'linear':
                Bkpred = self.Bkmm.get_Bk(z, k)
        elif nltype == 'halofit':
                Bkpred = self.Bkmm_halofit.get_Bk(z, k)
                
        if Pcb:
            if   nltype == 'linear':
                ## get the linear power spectrum for cb
                pkcblin = self.get_pklin(z, k, Pcb=True, type=lintype, cosmo_class=cosmo_class)
            elif nltype == 'halofit':
                ## get the halofit power spectrum for cb
                pkcblin = self.get_pkhalofit(z, k, Pcb=True, lintype=lintype, cosmo_class=cosmo_class)
            pknl = pkcblin * Bkpred
        else:
            if lintype == 'CLASS':
                if nltype == 'linear':
                    if cosmo_class is None:
                        cosmo_class = self.get_cosmos_class(z, non_linear='none')
                    pkcblin  = self.get_pklin(z, k, Pcb=True, type=lintype, cosmo_class=cosmo_class)
                elif nltype == 'halofit':
                    if (cosmo_class is None) or (cosmo_class.pars['non linear'] != 'halofit'):
                        cosmo_class = self.get_cosmos_class(z, non_linear='halofit')
                    pkcblin  = self.get_pkhalofit(z, k, Pcb=True, lintype=lintype, cosmo_class=cosmo_class)
                pknl   = np.zeros_like(pkcblin)
                Pcurv  = self._get_Pcurv(k)
                pkcbnl = pkcblin * Bkpred
                if cosmo_class.Omega_nu != 0:
                    for iz in range(len(z)):
                        tkall = cosmo_class.get_transfer(z=z[iz])
                        # Tktotsquare_interp = interp1d(tkall['k (h/Mpc)'], tkall['d_tot']*tkall['d_tot'], kind='cubic')
                        Tknusquare_interp  = interp1d(tkall['k (h/Mpc)'], tkall['d_ncdm[0]']*tkall['d_ncdm[0]'], kind='cubic')
                        fnu2M = self.Cosmo.Omeganu / self.Cosmo.OmegaM
                        fcb2M = self.Cosmo.Omegam  / self.Cosmo.OmegaM
                        ## use the linear neutrino power spectrum to calculate the nonlinear total power spectrum
                        pknunulin = Pcurv * Tknusquare_interp(k)
                        pknl[iz] = fcb2M*fcb2M*pkcbnl[iz] + fnu2M*fnu2M*pknunulin + 2*fnu2M*fcb2M*np.sqrt(pkcbnl[iz]*pknunulin)
                else:
                    pknl = pkcblin * Bkpred
            else:
                raise ValueError('For total power spectrum, only support type=CLASS NOW.')                           
        return pknl
    
    def _get_bkhmMassBin(self, z=None, k=None):
        '''
        Get the ratio between halo-matter power spectrum and cb Lin Pk.
        Note this is only for the transfer to the correlation function.
        '''
        kinterp = np.load(data_path + 'karr_nb_Nmesh1536_nmerge8.npy')
        kcut = 1.0
        ind = kinterp<=kcut
        kinterp = kinterp[ind]
        pinterp = self.PkhmMassBin.get_pkhmMassBin(z, kinterp)
        linterp = self.get_pklin(z, kinterp, Pcb=True, type='Emulator')
        bkout = np.zeros((pinterp.shape[0], pinterp.shape[1], len(k)))
        for i1 in range(pinterp.shape[0]):     # massbin 
            for i2 in range(pinterp.shape[1]): # redshift
                ### introduce a smooth process
                datainterp = pinterp[i1,i2]/linterp[i2]
                datainterp = savgol_filter(datainterp, window_length=9, polyorder=3)
                funcinterp = interp1d(np.log10(kinterp),
                                        datainterp,
                                        kind='slinear', fill_value="extrapolate")
                bkout[i1,i2] = funcinterp(np.log10(k))     
        return bkout

    def _get_xi_tree(self, z=None, r=None):
        '''
        Get the tree-level matter correlation function.
        z : float or array-like, redshift
        r : float or array-like, wavenumber [Mpc/h]
        '''
        z = check_z(self.zlists, z)
        ks = np.logspace(-5, 2, 1024)
        pkcblin = self.get_pklin(z, ks, type='Emulator', Pcb=True)
        bkhm    = self._get_bkhmMassBin(z, ks)
        ### number density
        xi_trees = np.zeros((bkhm.shape[0], len(z), len(r)))
        for im in range(bkhm.shape[0]):   # massbin
            for iz in range(bkhm.shape[1]):
                r0, xi0 = P2xi(ks, bkhm[im,iz]*pkcblin[iz], 0)
                xi_trees[im,iz] = ius(r0,xi0.real)(r)
        return xi_trees
       
    def get_xihmMassBin(self, z=None, r=None):
        '''
        Get the halo-matter cross correlation function.
        This function only supports the **fixed** mass bin Now.
        Mass bin is `[13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.4, 15.0]`.
        
        Args:
            z : float or array-like, redshift
            r : float or array-like, wavenumber [Mpc/h] 
        Return:
            array-like : halo-matter cross correlation function with shape (len(z), len(r))
        '''
        xi_dire = self.XihmMassBin.get_xihmMassBin(z, r)
        xi_tree = self._get_xi_tree(z, r)
        rswitch = 40.0 # Mpc/h
        xi_comb = np.zeros_like(xi_dire)
        for im in range(xi_dire.shape[0]): # massbin
            xi_comb[im] = xi_dire[im] * np.exp(-(r/rswitch)**4) \
                        + xi_tree[im] * (1 - np.exp(-(r/rswitch)**4))
        return xi_comb
