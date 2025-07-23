import numpy as np
import warnings
import time
from scipy.special import gamma
from scipy.interpolate import interp1d, RectBivariateSpline, RegularGridInterpolator, UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import savgol_filter
from .cosmology import Cosmology
from .emulator.Bkcb import Bkcb_gp, Bkcb_halofit_gp, Bkcb_lin2hmcode_gp, Bkcb_hmcode2020_gp
from .emulator.TkNuNncdm import Tkcblin_gp, Tkmmlin_gp, Tkcbhalofit_gp, Tkmmhalofit_gp, Tkcbhmcode2020_gp, Tkmmhmcode2020_gp
from .emulator.PkLin import PkcbLin_gp, Pknn_cbLin_gp
from .emulator.HMF import HMFRockstarM200m_gp, HMFFoFM200c_gp, HMFRockstarMvir_gp
from .emulator.Ximm import Ximm_cb_gp
from .emulator.Xihm import XihmMassBin_gp
from .emulator.Pkhm import PkhmMassBin_gp
from .hankl import P2xi
from .utils import *


####### base class for the whole emulator
class CBaseEmulator:
    '''
    The CSST Emulator class for various statistics.
    '''
    param_names  = param_names
    param_limits = param_limits
    zlists       = zlists    
    def __init__(self, verbose=False, neutrino_mass_split='single'):
        '''
        Initialize the CSST Emulator class.
        
        Args:
            verbose : bool, whether to output the running information
            neutrino_mass_split : string, 'single' or 'degenerate', the neutrino mass split type.
            
        .. note:: 
        
            The :py:class:`neutrino_mass_split = 'single'` means the neutrino mass is treated as a single massive component.
            Our training data is based on this treatment.
            The :py:class:`neutrino_mass_split = 'degenerate'` means the neutrino mass is treated as three degenerate components.
            This is achieved by using transformation of linear or HMCODE-2020 power spectrum from :py:class:`Nncdm=1` to :py:class:`Nncdm=3`.
            The transform spectrum can also be directly obtained from the :py:class:`Tkmm_CEmulator` class.
        
        '''
        self.verbose             = verbose
        self.neutrino_mass_split = neutrino_mass_split
        self.Cosmo               = Cosmology(verbose=verbose, neutrino_mass_split=neutrino_mass_split)
        self.Pkcblin             = PkcbLin_gp(verbose=verbose)
        self.Pknn_cblin          = Pknn_cbLin_gp(verbose=verbose)
        
        if self.neutrino_mass_split == 'single':
            if self.verbose: 
                print('The neutrino mass is treated as a single massive component.')
        elif self.neutrino_mass_split == 'degenerate':
            self.Tkcblin         = Tkcblin_gp(verbose=verbose)
            self.Tkmmlin         = Tkmmlin_gp(verbose=verbose)
        else:
            raise ValueError('The neutrino_mass_split = %s is not supported yet.'%self.neutrino_mass_split)
   
    def set_cosmos(self, Omegab=0.049, Omegac=0.26, 
                   H0=67.66, As=None, sigma8=None, 
                   ns=0.9665, w=-1.0, wa=0.0, 
                   mnu=0.06, sigma8type='Emulator',
                   checkbound=True):
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
            sigma8type: str, 'Emulator', 'CLASS' or 'CAMB', the method to calculate the sigma8.
            checkbound: bool, whether to check the parameter range.
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
        if checkbound:
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
            As1e9    = 2.105
            abserr   = 1e-4
            while np.abs(sigma8_g - sigma8) > abserr:
                cosmos['A'] = As1e9
                #use 'Emulator', 'CLASS' or 'CAMB' to get sigma8
                self.cosmologies = np.array([[cosmos['Omegab'], cosmos['Omegam'], cosmos['H0'], cosmos['ns'], \
                                              cosmos['A'], cosmos['w'], cosmos['wa'], cosmos['mnu']]]) 
                ### set the cosmology class
                self.Cosmo.set_cosmos(cosmos)
                self._sync_cosmologies()
                
                sigma8_g = self.get_sigma8(type=sigma8type) 
                # print('Guess sigma8:', sigma8_g)
                As1e9 = As1e9 * sigma8*sigma8/sigma8_g/sigma8_g
            if self.verbose:
                print('The As is set to %.6e (sigma8=%.6f) to match the input sigma8=%.6f.'%(cosmos['A']*1e-9, sigma8_g, sigma8)) 
        else:
            raise ValueError('Both As and sigma8 are None, please provide one of them at least.')
        # Omeganu = cosmos['mnu']/93.14/cosmos['H0']/cosmos['H0'] * 1e4
        ## check the parameter range for As
        if checkbound:
            for ikey in ['A']:
                if   np.any(cosmos[ikey] > self.param_limits[ikey][1]):
                    raise ValueError(r'Parameter out of range %ss1e9 = %.4f > %f.'%(ikey, cosmos[ikey], self.param_limits[ikey][1]))
                elif np.any(cosmos[ikey] < self.param_limits[ikey][0]):
                    raise ValueError(r'Parameter out of range %ss1e9 = %.4f < %f.'%(ikey, cosmos[ikey], self.param_limits[ikey][0])) 
        ### set the cosmology class
        self.Cosmo.set_cosmos(cosmos)
        ## into the cosmologies array only One cosmology each time
        self.cosmologies = np.array([[cosmos['Omegab'], cosmos['Omegam'], cosmos['H0'], cosmos['ns'], \
                                      cosmos['A'], cosmos['w'], cosmos['wa'], cosmos['mnu']]])
        ### sync the cosmologies for all objects
        self._sync_cosmologies()

    def _sync_cosmologies(self):
        '''
        Sync the cosmologies for all objects with the cosmology class.
        '''
        ### normalize the cosmologies with the shape (1, n_params)
        self.ncosmo = NormCosmo(self.cosmologies, self.param_names, self.param_limits)
        ### pass the cosmologies (normalized) to other class
        self.Pkcblin.ncosmo         = self.ncosmo
        self.Pknn_cblin.ncosmo      = self.ncosmo
        
        if self.neutrino_mass_split == 'single':
            pass
        elif self.neutrino_mass_split == 'degenerate':
            self.Tkcblin.ncosmo     = self.ncosmo
            self.Tkmmlin.ncosmo     = self.ncosmo
        else:
            raise ValueError('The neutrino_mass_split = %s is not supported yet.'%self.neutrino_mass_split) 
                                
    def get_cosmo_class(self, z=None, non_linear=None, kmax=10, neutrino_mass_split=None):
        '''
        Get the CLASS cosmology object.
        
        Args:
            z         : float or array-like, redshift.
            non_linear: string, None, 'halofit', 'HMcode' or other camb arguments.
            kmax      : maximum wave number for CLASS calculation.
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        # check_z(z)
        # z = np.atleast_1d(z)
        z = check_z(self.zlists, z)
        str_zlists = "{:.4f}".format(z[0])
        if len(z) > 1:
            for i_z in range(len(z) - 1):
                str_zlists += ", {:.4f}".format(z[i_z+1])
        cosmo_class = useCLASS(self.cosmologies[0], str_zlists, non_linear=non_linear, kmax=kmax, neutrino_mass_split=neutrino_mass_split)
        return cosmo_class
    
    def get_camb_results(self, z=None, non_linear=None, kmax=10, neutrino_mass_split=None):
        '''
        Get the CAMB results object.
        
        Args:
            z         : float or array-like, redshift
            non_linear: string, None, 'takahashi' or other camb arguments.
            kmax      : maximum wave number for CAMB calculation. 
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        # z = np.atleast_1d(z)
        z = check_z(self.zlists, z)
        ## reverse redshift for CAMB
        camb_results = useCAMB(self.cosmologies[0], zlists=z[::-1], non_linear=non_linear, kmax=kmax, neutrino_mass_split=neutrino_mass_split)
        return camb_results
        
    def get_pklin(self, z=None, k=None, Pcb=False, type='Emulator', cosmo_class=None, camb_results=None, neutrino_mass_split=None):
        '''
        Get the linear power spectrum.
        
        Args:
            z : float or array-like, redshift
            k : float or array-like, wavenumber with unit of [h/Mpc]
            Pcb  : bool, whether to output the total power spectrum (if False [default]) or the cb power spectrum (if True)
            type : string, 'Emulator', 'CLASS' or 'CAMB', liner Pk calcultion method.
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
            camb_results: CAMB results, if type is CAMB, then you can provide the CAMB object directly to avoid the repeated calculation for CAMB.
        Return:
            array-like : linear power spectrum with shape (len(z), len(k)) 
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        z = check_z(self.zlists,     z)
        # k = checkdata(self.Bkcb.klist, k, dname='wavenumber')
        if   type == 'CLASS':
            if cosmo_class is None:
                kmax = np.max(k)
                cosmo_class = self.get_cosmo_class(z, kmax=kmax)
            pklin = np.zeros((len(z), len(k)))
            if Pcb and (not np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10)):
                pkfunc = cosmo_class.pk_cb_lin
            else:
                pkfunc = cosmo_class.pk_lin
            h0 = cosmo_class.h()
            for iz in range(len(z)):
                pklin[iz] = np.array([pkfunc(ik*h0, z[iz])*h0*h0*h0 
                                      for ik in list(k)])
        elif type == 'CAMB':
            if camb_results is None:
                kmax = np.max(k)
                camb_results = self.get_camb_results(z, kmax=kmax)
            if Pcb and (not np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10)): 
                pkfunc = camb_results.get_matter_power_interpolator(nonlinear=False, 
                                                                    var1='delta_nonu', var2='delta_nonu',
                                                                    hubble_units=True, k_hunit=True)
            else:
                pkfunc = camb_results.get_matter_power_interpolator(nonlinear=False, 
                                                                    var1='delta_tot', var2='delta_tot',
                                                                    hubble_units=True, k_hunit=True)
            pklin = pkfunc.P(z,k)
        elif type == 'Emulator':
            if Pcb:
                pklin = self.Pkcblin.get_pkcbLin(z=z, k=k)
            else:
                pkcblin = self.Pkcblin.get_pkcbLin(z, k)
                pknnlin = self.Pknn_cblin.get_pknn_cbLin(z, k) * pkcblin
                fcb2M   = self.Cosmo.Omegam / self.Cosmo.OmegaM 
                fnu2M   = self.Cosmo.Omeganu / self.Cosmo.OmegaM
                pklin   = fcb2M*fcb2M*pkcblin + fnu2M*fnu2M*pknnlin \
                        + 2*fcb2M*fnu2M*np.sqrt(pkcblin*pknnlin)
            if neutrino_mass_split == 'single':
                pklin = pklin
            elif neutrino_mass_split == 'degenerate':
                if Pcb:
                    pklin = self.Tkcblin.get_Tkcblin(z=z, k=k) * pklin
                else:
                    pklin = self.Tkmmlin.get_Tkmmlin(z=z, k=k) * pklin
            else:
                raise ValueError('The neutrino_mass_split = %s is not supported yet.'%self.neutrino_mass_split)
        else:
            raise ValueError('Type %s not supported yet.'%type)
        return pklin
    
    def get_sigma_z(self, z=None, R=None, type='Emulator', cosmo_class=None, camb_results=None):
        '''
        Get the sigma(z, R) of tot matter changing with the redshift.
        
        Args:
            z : float, redshift
            R : float, smoothing scale [Mpc/h]
            type : string, 'Emulator', 'CLASS' or 'CAMB' sigma
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
            camb_results: CAMB results, if type is CAMB, then you can provide the CAMB object directly to avoid the repeated calculation for CAMB.
        Return:
            float : sigma8 value with shape (len(z))
        '''
        if not isinstance(z, (int, float)):
            raise ValueError('Only support one redshift now.')
        # h0 = self.Cosmo.h0 ## if match the sigma8 there is no Cosmo object
        h0 = self.cosmologies[0][2]/100
        if type == 'CLASS':
            if cosmo_class is None:
                cosmo_class = self.get_cosmo_class(z)
            if np.isscalar(R):
                sigma = cosmo_class.sigma(R/h0, z)
            else:
                sigma = np.zeros(len(R))
                for iR in range(len(R)):
                    sigma[iR] = cosmo_class.sigma(R[iR]/h0, z)
        elif type == 'CAMB':
            if camb_results is None:
                camb_results = self.get_camb_results(z)
                zind = 0
            else:
                cambzout = np.array(camb_results.transfer_redshifts)
                if z not in cambzout:
                    camb_results = self.get_camb_results(z)
                    zind = 0
                else:
                    zind = np.where(cambzout==z)[0][0]
            sigma = camb_results.get_sigmaR(R=8.0, z_indices=zind, hubble_units=True,
                                            var1='delta_tot', var2='delta_tot')
        elif type == 'Emulator':
            kcalc    = np.logspace(-4.99, 1.99, 1000)
            pcalc    = self.get_pklin(z, kcalc, Pcb=False, type='Emulator')[0]
            if np.isscalar(R):
                W_R      = 3*(np.sin(kcalc*R) - kcalc*R*np.cos(kcalc*R))/(kcalc*R)**3
                sigma    = np.sqrt(np.trapz(pcalc*W_R*W_R*kcalc*kcalc*kcalc/2/np.pi/np.pi, np.log(kcalc))) 
            else:
                sigma = np.zeros(len(R))
                for iR in range(len(R)):
                    W_R      = 3*(np.sin(kcalc*R[iR]) - kcalc*R[iR]*np.cos(kcalc*R[iR]))/(kcalc*R[iR])**3
                    sigma[iR] = np.sqrt(np.trapz(pcalc*W_R*W_R*kcalc*kcalc*kcalc/2/np.pi/np.pi, np.log(kcalc)))
        else:
            raise ValueError('Type %s not supported yet.'%type)
        return sigma
    
    def get_sigma_cb_z(self, z=None, R=None, type='Emulator', cosmo_class=None, camb_results=None):
        '''
        Get the sigma_cb(z, R) of tot matter changing with the redshift.
        
        Args:
            z : float, redshift
            R : float, smoothing scale [Mpc/h]
            type : string, 'Emulator', 'CLASS' or 'CAMB' sigma_cb for cdm + baryon components
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
            camb_results: CAMB results, if type is CAMB, then you can provide the CAMB object directly to avoid the repeated calculation for CAMB.
        Return:
            float : sigma8 value with shape (len(z))
        '''
        if not isinstance(z, (int, float)):
            raise ValueError('Only support one redshift now.')
        # h0 = self.Cosmo.h0
        h0 = self.cosmologies[0][2]/100
        if type == 'CLASS':
            if cosmo_class is None:
                cosmo_class = self.get_cosmo_class(z)
            if np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10):
                if np.isscalar(R):
                    sigma_cb = cosmo_class.sigma(R/h0, z)
                else:
                    sigma_cb = np.zeros(len(R))
                    for iR in range(len(R)):
                        sigma_cb[iR] = cosmo_class.sigma(R[iR]/h0, z)
            else:
                if np.isscalar(R):
                    sigma_cb = cosmo_class.sigma_cb(R/h0, z)
                else:
                    sigma_cb = np.zeros(len(R))
                    for iR in range(len(R)):
                        sigma_cb[iR] = cosmo_class.sigma_cb(R[iR]/h0, z) 
        elif type == 'CAMB':
            if camb_results is None:
                camb_results = self.get_camb_results(z)
                zind = 0
            else:
                cambzout = np.array(camb_results.transfer_redshifts)
                if z not in cambzout:
                    camb_results = self.get_camb_results(z)
                    zind = 0
                else:
                    zind = np.where(cambzout==z)[0][0]
            sigma_cb = camb_results.get_sigmaR(R=8.0, z_indices=zind, hubble_units=True,
                                               var1='delta_nonu', var2='delta_nonu')
        elif type == 'Emulator':
            kcalc    = np.logspace(-4.99, 1.99, 1000)
            pcalc    = self.Pkcblin.get_pkcbLin(z, kcalc)[0]
            if np.isscalar(R):
                W_R      = 3*(np.sin(kcalc*R) - kcalc*R*np.cos(kcalc*R))/(kcalc*R)**3
                sigma_cb = np.sqrt(np.trapz(pcalc*W_R*W_R*kcalc*kcalc*kcalc/2/np.pi/np.pi, np.log(kcalc))) 
            else:
                sigma_cb = np.zeros(len(R))
                for iR in range(len(R)):
                    W_R          = 3*(np.sin(kcalc*R[iR]) - kcalc*R[iR]*np.cos(kcalc*R[iR]))/(kcalc*R[iR])**3
                    sigma_cb[iR] = np.sqrt(np.trapz(pcalc*W_R*W_R*kcalc*kcalc*kcalc/2/np.pi/np.pi, np.log(kcalc)))
        else:
            raise ValueError('Type %s not supported yet.'%type)
        return sigma_cb
    
    def get_sigma8(self, type='Emulator', cosmo_class=None, camb_results=None):
        '''
        Get the sigma8 of tot matter.
        
        Args:
            type : string, 'Emulator', 'CLASS' or 'CAMB'
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
            camb_results: CAMB results, if type is CAMB, then you can provide the CAMB object directly to avoid the repeated calculation for CAMB.
        '''
        return self.get_sigma_z(0, 8.0, type=type, cosmo_class=cosmo_class, camb_results=camb_results)
    
    def get_sigma8_cb(self, type='Emulator', cosmo_class=None, camb_results=None):
        '''
        Get the sigma8 of cb matter.
        
        Args:
            type : string, 'Emulator', 'CLASS' or 'CAMB'
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
            camb_results: CAMB results, if type is CAMB, then you can provide the CAMB object directly to avoid the repeated calculation for CAMB. 
        '''
        return self.get_sigma_cb_z(0, 8.0, type=type, cosmo_class=cosmo_class, camb_results=camb_results)

####### class for transfrom from Nncdm=1 to Nncdm=3
class Tkmm_CEmulator(CBaseEmulator):
    '''
    The matter power spectrum transformation from Nncdm=1 to Nncdm=3 emulator class.
    '''
    def __init__(self, verbose=False, neutrino_mass_split='degenerate'):
        super().__init__(verbose, neutrino_mass_split)
        self.Tkcbhalofit    = Tkcbhalofit_gp(verbose=verbose)
        self.Tkmmhalofit    = Tkmmhalofit_gp(verbose=verbose)
        self.Tkcbhmcode2020 = Tkcbhmcode2020_gp(verbose=verbose)
        self.Tkmmhmcode2020 = Tkmmhmcode2020_gp(verbose=verbose)
    
    def _sync_cosmologies(self):
        '''
        Sync the cosmologies for all objects with the cosmology class.
        '''
        super()._sync_cosmologies()
        self.Tkcbhalofit.ncosmo    = self.ncosmo
        self.Tkmmhalofit.ncosmo    = self.ncosmo
        self.Tkcbhmcode2020.ncosmo = self.ncosmo
        self.Tkmmhmcode2020.ncosmo = self.ncosmo
    
    def get_Tk(self, z=None, k=None, Pcb=False, Tk_type='linear'):
        z = check_z(self.zlists,     z) 
        if Tk_type == 'linear':
            if Pcb:
                return self.Tkcblin.get_Tkcblin(z=z, k=k)
            else:
                return self.Tkmmlin.get_Tkmmlin(z=z, k=k)
        elif Tk_type == 'halofit':
            if Pcb:
                return self.Tkcbhalofit.get_Tkcbhalofit(z=z, k=k)
            else:
                return self.Tkmmhalofit.get_Tkmmhalofit(z=z, k=k)
        elif Tk_type == 'hmcode2020':
            if Pcb:
                return self.Tkcbhmcode2020.get_Tkcbhmcode2020(z=z, k=k)
            else:
                return self.Tkmmhmcode2020.get_Tkmmhmcode2020(z=z, k=k)
        
####### class for the matter power spectrum emulator
class Pkmm_CEmulator(CBaseEmulator):
    '''
    The matter power spectrum emulator class.
    '''
    def __init__(self, verbose=False, neutrino_mass_split='single'):
        '''
        Initialize the matter power spectrum emulator class.
        
        Args:
            verbose : bool, whether to output the running information
        '''
        super().__init__(verbose=verbose, neutrino_mass_split=neutrino_mass_split)
        self.Bkcb            = Bkcb_gp(verbose=verbose) 
        self.Bkcb_halofit    = Bkcb_halofit_gp(verbose=verbose) 
        self.Bkcb_lin2hmcode = Bkcb_lin2hmcode_gp(verbose=verbose)
        self.Bkcb_hmcode2020 = Bkcb_hmcode2020_gp(verbose=verbose)
        if self.neutrino_mass_split == 'single':
            pass
        elif self.neutrino_mass_split == 'degenerate':
            self.Tkcbhmcode2020 = Tkcbhmcode2020_gp(verbose=verbose)
            self.Tkmmhmcode2020 = Tkmmhmcode2020_gp(verbose=verbose)
        else:
            raise ValueError('The neutrino_mass_split = %s is not supported yet.'%self.neutrino_mass_split)
    
    def _sync_cosmologies(self):
        '''
        Sync the cosmologies for all objects with the cosmology class.
        ''' 
        super()._sync_cosmologies()
        ##### pass the cosmologies (normalized) to other class
        self.Bkcb.ncosmo            = self.ncosmo
        self.Bkcb_halofit.ncosmo    = self.ncosmo
        self.Bkcb_lin2hmcode.ncosmo = self.ncosmo
        self.Bkcb_hmcode2020.ncosmo = self.ncosmo
        if self.neutrino_mass_split == 'single':
            pass
        elif self.neutrino_mass_split == 'degenerate':
            self.Tkcbhmcode2020.ncosmo = self.ncosmo
            self.Tkmmhmcode2020.ncosmo = self.ncosmo
        else:
            raise ValueError('The neutrino_mass_split = %s is not supported yet.'%self.neutrino_mass_split)
        
    def _get_Pcurv(self, k=None):
        As = self.Cosmo.As
        ns = self.Cosmo.ns
        h0 = self.Cosmo.h0
        kp = 0.05 ## Mpc^{-1}
        Pcurv = lambda k: 2*np.pi*np.pi*As * (h0*k/kp)**(ns-1)/k/k/k
        return Pcurv(k)
    
    def get_pkhalofit(self, z=None, k=None, Pcb=False, lintype='Emulator', cosmo_class=None, camb_results=None, neutrino_mass_split=None):
        '''
        Get the halofit power spectrum. [only Nncdm=1]
       
        .. note:: 
            This version can not converge with CLASS in the high redshift (z>2.5)
            for the c0001 and c0091 (w0 and wa near the lower limit).
       
        Args:
            z           : float or array-like, redshift 
            k           : float or array-like, wavenumber [h/Mpc]
            Pcb         : bool, whether to output the total power spectrum (if False [default]) or the cb power spectrum (if True)
            lintype     : string, 'Emulator', 'CLASS' or 'CAMB' halofit results
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.    
            camb_results: CAMB results, if type is CAMB, then you can provide the CAMB object directly to avoid the repeated calculation for CAMB. 
        Return: 
            array-like: halofit power spectrum with shape (len(z), len(k))
         
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        z = check_z(self.zlists,     z)
        if  lintype == 'Emulator':
            if Pcb:
                fnu = 0.0
            else:
                fnu = self.Cosmo.Omeganu/self.Cosmo.OmegaM
            OmegaLzall   = self.Cosmo.get_OmegaL(z)
            OmegaMzall   = self.Cosmo.get_OmegaM(z)
            pkhalofit    = np.zeros((len(z), len(k)))
            ## TODO: adjust the maximum k for the interpolation may match the result of CLASS
            ## Notice: adjust the maximum k for the interpolation will affect the result significantly
            kinterp      = np.logspace(-4.99, 1, 1024)
            pklinintp    = self.get_pklin(z=z, k=kinterp, Pcb=Pcb, type=lintype, neutrino_mass_split=neutrino_mass_split)
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
                h0 = self.Cosmo.h0
                pkhalofit[iz] = PkHaloFit(k, Pklin(k), R_sigma, OmegaMz, OmegaLz, fnu, \
                                          an, bn, cn, gamman, alphan, betan, mun, nun, h0)
        elif lintype == 'CLASS':
            if cosmo_class is None:
                cosmo_class = self.get_cosmo_class(z, non_linear='halofit', neutrino_mass_split=neutrino_mass_split)
            if (Pcb) and (not np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10)):
                pkfunc = cosmo_class.pk_cb
            else:
                pkfunc = cosmo_class.pk
            pkhalofit = np.zeros((len(z), len(k)))
            h0 = cosmo_class.h()
            for iz in range(len(z)):
                pkhalofit[iz] = np.array([pkfunc(ik*h0, z[iz])*h0*h0*h0 
                                          for ik in list(k)])
        elif lintype == 'CAMB':
            if camb_results is None:
                camb_results = self.get_camb_results(z, non_linear='takahashi', neutrino_mass_split=neutrino_mass_split)
            if Pcb and (not np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10)): 
                pkfunc = camb_results.get_matter_power_interpolator(nonlinear=True, 
                                                                    var1='delta_nonu', var2='delta_nonu',
                                                                    hubble_units=True, k_hunit=True)
            else:
                pkfunc = camb_results.get_matter_power_interpolator(nonlinear=True, 
                                                                    var1='delta_tot', var2='delta_tot',
                                                                    hubble_units=True, k_hunit=True)
            pkhalofit = pkfunc.P(z,k)
        else:
            raise ValueError('Type %s not supported yet.'%type) 
        return pkhalofit
    
    def get_pkHMCODE2020(self, z=None, k=None, Pcb=False, lintype='Emulator', cosmo_class=None, camb_results=None, neutrino_mass_split=None):
        '''
        Get the linear power spectrum from HMCODE2020.
        
        Args:
            z           : float or array-like, redshift
            k           : float or array-like, wavenumber [h/Mpc]
            lintype     : string, 'Emulator', 'CLASS' or 'CAMB' halofit results
            Pcb         : bool, whether to output the total power spectrum (if False [default]) or the cb power spectrum (if True)
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
            camb_results: CAMB results, if type is CAMB, then you can provide the CAMB object directly to avoid the repeated calculation for CAMB.
        Return:
            array-like : linear power spectrum with shape (len(z), len(k))
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        z = check_z(self.zlists,     z)
        if  lintype == 'Emulator':
            if Pcb or np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10):
                Bkcbhmcode = self.Bkcb_lin2hmcode.get_Bk(z, k)
                ## get the linear power spectrum for cb
                ## we only emulate the lin2hmcode for the cb power spectrum with Nncdm=1
                pkcblin = self.get_pklin(z, k, Pcb=True, type=lintype, cosmo_class=cosmo_class, camb_results=camb_results, neutrino_mass_split='single')
                pkhmcode = pkcblin * Bkcbhmcode
                if neutrino_mass_split == "single":
                    pkhmcode = pkhmcode
                elif neutrino_mass_split == 'degenerate':
                    pkhmcode = self.Tkcbhmcode2020.get_Tkcbhmcode2020(z=z, k=k) * pkhmcode
                else:
                    raise ValueError('The neutrino_mass_split = %s is not supported yet.'%neutrino_mass_split)                    
            else:
                raise ValueError('Only support the cb power spectrum [Pcb=True] now.')
            
        elif lintype == 'CLASS':
            raise ValueError('CLASS does not support the HMCODE2020 yet.')
        elif lintype == 'CAMB':
            if camb_results is None:
                camb_results = self.get_camb_results(z, non_linear='mead2020', neutrino_mass_split=neutrino_mass_split)
            if Pcb and (not np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10)): 
                pkfunc = camb_results.get_matter_power_interpolator(nonlinear=True, 
                                                                    var1='delta_nonu', var2='delta_nonu',
                                                                    hubble_units=True, k_hunit=True)
            else:
                pkfunc = camb_results.get_matter_power_interpolator(nonlinear=True, 
                                                                    var1='delta_tot', var2='delta_tot',
                                                                    hubble_units=True, k_hunit=True)
            pkhmcode = pkfunc.P(z,k)
        else:
            raise ValueError('Type %s not supported yet.'%type) 
        return pkhmcode 
      
    def get_pknl(self, z=None, k=None, Pcb=False, lintype='Emulator', nltype='hmcode2020', cosmo_class=None, camb_results=None, neutrino_mass_split=None):
        '''
        Get the nonlinear power spectrum.
        
        .. note::
           For now, nltype = 'hmcode2020', 'halofit' can give a better result than nltype = 'linear'.
           For nltype = 'halofit' or 'hmcode2020', we only use the 'Emulator' method to generate the halofit Pk for consistency between trainning and output data.
           The lintype only determine which method to generate the neutrino linear Pk.
           For nltype = 'linear', lintype determine which method to generate the linear cb and mm Pk.
           Because the agreements of linear Pk between CAMB, CLASS and Emulator is better than 0.5%.
        
        Args:
            z           : float or array-like, redshift. 
            k           : float or array-like, wavenumber [h/Mpc]. 
            Pcb         : bool, whether to output the total power spectrum (if False [default]) or the cb power spectrum (if True).
            lintype     : string, use 'Emulator', 'CLASS' or 'CAMB' method to generate linear Pk. Default is 'Emulator'.
            nltype      : string, 'linear', 'halofit' or 'hmcode2020'.  means ratio of nonlinear to **nltype** power spectrum. Default is 'hmcode2020'.
            cosmo_class : CLASS object, if type is 'CLASS', then you can provide the CLASS object directly to avoid the repeated calculation for CLASS.
            camb_results: CAMB results, if type is CAMB, then you can provide the CAMB object directly to avoid the repeated calculation for CAMB. 
        Return:
            array-like : nonlinear power spectrum with shape (len(z), len(k))
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        z = check_z(self.zlists,     z)
        k = checkdata(self.Bkcb.klist, k, dname='wavenumber')
        ## get the nonlinear transfer for Pcb
        if   nltype == 'linear':
            Bkpred = self.Bkcb.get_Bk(z, k)
        elif nltype == 'halofit':
            Bkpred = self.Bkcb_halofit.get_Bk(z, k)
        elif nltype == 'hmcode2020':
            Bkpred = self.Bkcb_hmcode2020.get_Bk(z, k)
        if Pcb or np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10):
            if   nltype == 'linear':
                ## get the linear power spectrum for cb
                pkcblin = self.get_pklin(z, k, Pcb=True, type=lintype, cosmo_class=cosmo_class, camb_results=camb_results, neutrino_mass_split='single')
            elif nltype == 'halofit':
                ## for consistency, we use the cb halofit power spectrum only from my Emulator for the nonlinear emulation
                ## get the halofit power spectrum for cb
                pkcblin = self.get_pkhalofit(z, k, Pcb=True, lintype='Emulator', cosmo_class=cosmo_class, camb_results=camb_results, neutrino_mass_split='single')
            elif nltype == 'hmcode2020':
                pkcblin = self.get_pkHMCODE2020(z, k, Pcb=True, lintype=lintype, cosmo_class=cosmo_class, camb_results=camb_results, neutrino_mass_split='single')
            pknl = pkcblin * Bkpred
            ####### for the neutrino mass split
            if neutrino_mass_split == 'single':
                pknl = pknl
            elif neutrino_mass_split == 'degenerate':
                pknl = self.Tkcbhmcode2020.get_Tkcbhmcode2020(z=z, k=k) * pknl
            else:
                raise ValueError('The neutrino_mass_split = %s is not supported yet.'%neutrino_mass_split)
            
        else:
            fnu2M = self.Cosmo.Omeganu / self.Cosmo.OmegaM
            fcb2M = self.Cosmo.Omegam  / self.Cosmo.OmegaM    
            if lintype == 'CLASS':
                if nltype == 'linear':
                    if (cosmo_class is None):
                        cosmo_class = self.get_cosmo_class(z, non_linear=None)
                    pkcblin  = self.get_pklin(z, k, Pcb=True, type=lintype, cosmo_class=cosmo_class, neutrino_mass_split='single')
                elif nltype == 'halofit':
                    if (cosmo_class is None):
                        cosmo_class = self.get_cosmo_class(z, non_linear=None) # not use class halofit
                    ## for consistency, we use the cb halofit power spectrum only from my Emulator for the nonlinear emulation
                    pkcblin  = self.get_pkhalofit(z, k, Pcb=True, lintype='Emulator', cosmo_class=cosmo_class, neutrino_mass_split='single')
                elif nltype == 'hmcode2020':
                    if (cosmo_class is None):
                        cosmo_class = self.get_cosmo_class(z, non_linear='mead2020')
                    pkcblin = self.get_pkHMCODE2020(z, k, Pcb=True, lintype=lintype, cosmo_class=cosmo_class, neutrino_mass_split='single')
                pknl   = np.zeros_like(pkcblin)
                Pcurv  = self._get_Pcurv(k)
                pkcbnl = pkcblin * Bkpred
                if not np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10):
                    for iz in range(len(z)):
                        tkall = cosmo_class.get_transfer(z=z[iz])
                        # Tktotsquare_interp = interp1d(tkall['k (h/Mpc)'], tkall['d_tot']*tkall['d_tot'], kind='cubic')
                        Tknusquare_interp  = interp1d(np.log10(tkall['k (h/Mpc)']),                    \
                                                      np.log10(tkall['d_ncdm[0]']*tkall['d_ncdm[0]']), \
                                                      kind='linear',                                   \
                                                      fill_value="extrapolate")
                        ## use the linear neutrino power spectrum to calculate the nonlinear total power spectrum
                        pknunulin = Pcurv * (10**Tknusquare_interp(np.log10(k)))
                        pknl[iz] = fcb2M*fcb2M*pkcbnl[iz] + fnu2M*fnu2M*pknunulin + 2*fnu2M*fcb2M*np.sqrt(pkcbnl[iz]*pknunulin)
                else:
                    raise ValueError('This should not happen!')
            elif lintype == 'CAMB':
                if nltype == 'linear':
                    if (camb_results is None):
                        camb_results = self.get_camb_results(z, non_linear=None)
                    pkcblin  = self.get_pklin(z, k, Pcb=True, type=lintype, camb_results=camb_results, neutrino_mass_split='single')
                elif nltype == 'halofit':
                    if (camb_results is None):
                        cosmo_class = self.get_cosmo_class(z, non_linear=None) # not use class halofit
                    ## for consistency, we use the cb halofit power spectrum only from my Emulator for the nonlinear emulation
                    pkcblin  = self.get_pkhalofit(z, k, Pcb=True, lintype='Emulator', camb_results=camb_results, neutrino_mass_split='single')
                elif nltype == 'hmcode2020':
                    if (camb_results is None):
                        camb_results = self.get_camb_results(z, non_linear='mead2020')
                    pkcblin = self.get_pkHMCODE2020(z, k, Pcb=True, lintype=lintype, camb_results=camb_results, neutrino_mass_split='single')
                pkcbnl = pkcblin * Bkpred
                if not np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10):
                    pkfunc    = camb_results.get_matter_power_interpolator(nonlinear=False, 
                                                                           var1='delta_nu', var2='delta_nu',
                                                                           hubble_units=True, k_hunit=True)
                    pknunulin = pkfunc.P(z, k)
                    pknl      = fcb2M*fcb2M*pkcbnl + fnu2M*fnu2M*pknunulin + 2*fnu2M*fcb2M*np.sqrt(pkcbnl*pknunulin) 
            elif lintype == 'Emulator':
                if nltype == 'linear':
                    pkcblin = self.get_pklin(z, k, Pcb=True, type=lintype, neutrino_mass_split='single')
                elif nltype == 'halofit':
                    pkcblin = self.get_pkhalofit(z, k, Pcb=True, lintype=lintype, neutrino_mass_split='single')
                elif nltype == 'hmcode2020':
                    pkcblin = self.get_pkHMCODE2020(z, k, Pcb=True, lintype=lintype, neutrino_mass_split='single')
                pkcbnl = pkcblin * Bkpred
                if not np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10):
                    pknnlin = self.Pknn_cblin.get_pknn_cbLin(z, k) * pkcblin
                    pknl    = fcb2M*fcb2M*pkcbnl  \
                            + fnu2M*fnu2M*pknnlin \
                            + 2*fcb2M*fnu2M*np.sqrt(pkcbnl*pknnlin)
                else:
                    raise ValueError('This should not happen!')
            else:
                raise ValueError('Type %s not supported yet.'%type) 
            ######## for the neutrino mass split
            if neutrino_mass_split == 'single':
                pknl = pknl
            elif neutrino_mass_split == 'degenerate':
                pknl = self.Tkmmhmcode2020.get_Tkmmhmcode2020(z=z, k=k) * pknl
            else:
                raise ValueError('The neutrino_mass_split = %s is not supported yet.'%neutrino_mass_split)                        
        return pknl

####### class for the matter correlation function emulator
class Ximm_CEmulator(CBaseEmulator):
    '''
    The matter correlation function emulator class.
    '''
    def __init__(self, verbose=False, neutrino_mass_split='single'):
        '''
        Initialize the matter correlation function emulator class.
        
        Args:
            verbose : bool, whether to output the running information
        '''
        super().__init__(verbose=verbose, neutrino_mass_split=neutrino_mass_split)
        self.Ximm_cb = Ximm_cb_gp(verbose=verbose)
        self.pkemu   = Pkmm_CEmulator(verbose=verbose)
    
    def _sync_cosmologies(self):
        super()._sync_cosmologies()
        self.Ximm_cb.ncosmo    = self.ncosmo
        self.pkemu.ncosmo      = self.ncosmo
        self.pkemu.cosmologies = self.cosmologies
        self.pkemu.Cosmo       = self.Cosmo
        self.pkemu._sync_cosmologies()
    
    def get_ximmhalofit(self, z=None, r=None, Pcb=False, neutrino_mass_split=None):
        '''
        Get the matter [cb] correlation function by combining the halofit power spectrum and FFTLog.
        
        Args:
            z : float or array-like, redshift
            r : float or array-like, wavenumber [Mpc/h]
            Pcb : bool, whether to output the total power spectrum (if False [default]) or the cb power spectrum (if True)
        Return:
            array-like : matter-matter correlation function with shape (len(z), len(r))
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        z = check_z(self.zlists, z)
        if Pcb or np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10):
            k0 = np.logspace(-4.99, 1.99, 512)
            p0 = self.pkemu.get_pkhalofit(z, k0, Pcb=True, lintype='Emulator', neutrino_mass_split=neutrino_mass_split)
            kfft = np.logspace(-5, 2, 1024)
            pfft = 10**interp1d(np.log10(k0), np.log10(p0), 
                                kind='slinear', fill_value='extrapolate')(np.log10(kfft))
            ximmhalofit = np.zeros((len(z), len(r)))
            for iz in range(len(z)):
                r0, xi0 = P2xi(kfft, pfft[iz], 0)
                ximmhalofit[iz] = ius(r0, xi0.real)(r)
        else:
            raise ValueError('This is coming soon ~~ :)')
        return ximmhalofit
    
    def _get_ximmnl_from_pknl(self, z=None, r=None, Pcb=False, neutrino_mass_split=None):
        '''
        Get the matter-matter correlation function by combining the pknl and FFTLog.
        
        Args:
            z : float or array-like, redshift
            r : float or array-like, wavenumber [Mpc/h]
            Pcb : bool, whether to output the total power spectrum (if False [default]) or the cb power spectrum (if True)
        Return:
            array-like : matter-matter correlation function with shape (len(z), len(r))
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        z = check_z(self.zlists, z)
        r = np.atleast_1d(r)
        k0 = np.logspace(-2.2, 1, 512)
        p0 = self.pkemu.get_pknl(z=z, k=k0, Pcb=Pcb, lintype='Emulator', nltype='hmcode2020', neutrino_mass_split=neutrino_mass_split)
        p1 = self.pkemu.get_pklin(z=z, k=k0, Pcb=Pcb, type='Emulator', neutrino_mass_split=neutrino_mass_split)
        kfft = np.logspace(-5, 2, 1024)
        plin = self.pkemu.get_pklin(z=z, k=kfft, Pcb=Pcb, type='Emulator', neutrino_mass_split=neutrino_mass_split)
        pfft = interp1d(np.log10(k0), p0/p1, 
                        kind='slinear', fill_value='extrapolate')(np.log10(kfft)) * plin
        ximmnl = np.zeros((len(z), len(r)))
        for iz in range(len(z)):
            r0, xi0 = P2xi(kfft, pfft[iz], 0)
            ximmnl[iz] = ius(r0, xi0.real)(r)
        return ximmnl
        
    def get_ximmnl(self, z=None, r=None, Pcb=False, neutrino_mass_split=None):
        '''
        Get the matter-matter correlation function.
        
        Args:
            z : float or array-like, redshift
            r : float or array-like, wavenumber [Mpc/h]
            Pcb : bool, whether to output the total power spectrum (if False [default]) or the cb power spectrum (if True)
        Return:
            array-like : matter-matter correlation function with shape (len(z), len(r))
        '''
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        if neutrino_mass_split == 'degenerate':
            raise ValueError('The neutrino_mass_split = %s is not supported yet.'%neutrino_mass_split)
        z = check_z(self.zlists, z)
        r = np.atleast_1d(r)
        if Pcb or np.isclose(self.Cosmo.Omeganu, 0, atol=1e-10):
            ximmhalofit = self.get_ximmhalofit(z=z, r=r, Pcb=Pcb, neutrino_mass_split='single')
            ximm_dire = ximmhalofit * self.Ximm_cb.get_Br(z=z, r=r)
            ximm_tree = self._get_ximmnl_from_pknl(z=z, r=r, Pcb=Pcb, neutrino_mass_split='single')
            rswitch = 40.0
            ximm_comb = np.zeros_like(ximm_dire)
            for iz in range(ximm_dire.shape[0]):
                ximm_comb[iz] = ximm_dire[iz] * np.exp(-(r/rswitch)**4) \
                              + ximm_tree[iz] * (1 - np.exp(-(r/rswitch)**4))
        else:
            raise ValueError('Please set Pcb=True. The emulator for ximm[Pcb=False] is coming soon ~~ :)')
        return ximm_comb  

####### class for the halo-matter correlation function with specified Mass Bin emulator
class XihmMassBin_CEmulator(CBaseEmulator):
    '''
    The halo-matter correlation function [for specified mass bin] emulator class.
    '''
    def __init__(self, verbose=False, neutrino_mass_split='single'):
        '''
        Initialize the halo-matter correlation function [for specified mass bin] emulator class.
        
        Args:
            verbose : bool, whether to output the running information
        '''
        super().__init__(verbose=verbose, neutrino_mass_split=neutrino_mass_split)
        self.XihmMassBin = XihmMassBin_gp(verbose=verbose)
        self.PkhmMassBin = PkhmMassBin_gp(verbose=verbose)
        
    def _sync_cosmologies(self):
        super()._sync_cosmologies()
        self.XihmMassBin.ncosmo    = self.ncosmo
        self.PkhmMassBin.ncosmo    = self.ncosmo
    
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
        linterp = self.get_pklin(z, kinterp, Pcb=True, type='Emulator', neutrino_mass_split='single')
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
        ks = np.logspace(-4.99, 1.99, 1024)
        pkcblin = self.get_pklin(z, ks, type='Emulator', Pcb=True, neutrino_mass_split='single')
        bkhm    = self._get_bkhmMassBin(z, ks)
        ### number density
        xi_trees = np.zeros((bkhm.shape[0], len(z), len(r)))
        for im in range(bkhm.shape[0]):   # massbin
            for iz in range(bkhm.shape[1]):
                r0, xi0 = P2xi(ks, bkhm[im,iz]*pkcblin[iz], 0)
                xi_trees[im,iz] = ius(r0,xi0.real)(r)
        return xi_trees
       
    def get_xihmMassBin(self, z=None, r=None, neutrino_mass_split=None):
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
        if neutrino_mass_split is None:
            neutrino_mass_split = self.neutrino_mass_split
        if neutrino_mass_split == 'degenerate':
            raise ValueError('The neutrino_mass_split = %s is not supported yet.'%neutrino_mass_split)
        xi_dire = self.XihmMassBin.get_xihmMassBin(z, r)
        xi_tree = self._get_xi_tree(z, r)
        rswitch = 40.0 # Mpc/h
        xi_comb = np.zeros_like(xi_dire)
        for im in range(xi_dire.shape[0]): # massbin
            xi_comb[im] = xi_dire[im] * np.exp(-(r/rswitch)**4) \
                        + xi_tree[im] * (1 - np.exp(-(r/rswitch)**4))
        return xi_comb

####### class for the halo mass function emulator
class HMF_CEmulator(CBaseEmulator):
    '''
    The halo mass function emulator class.
    '''
    def __init__(self, verbose=False, neutrino_mass_split='single'):
        '''
        Initialize the halo mass function emulator class.
        
        Args:
            verbose : bool, whether to output the running information
        '''
        super().__init__(verbose=verbose, neutrino_mass_split=neutrino_mass_split)
        self.HMFRockstarM200m = HMFRockstarM200m_gp(verbose=verbose)
        self.HMFFoFM200c      = HMFFoFM200c_gp(verbose=verbose)
        self.HMFRockstarMvir  = HMFRockstarMvir_gp(verbose=verbose)
    
    def _sync_cosmologies(self):
        super()._sync_cosmologies()
        self.HMFRockstarM200m.ncosmo = self.ncosmo
        self.HMFFoFM200c.ncosmo      = self.ncosmo
        self.HMFRockstarMvir.ncosmo  = self.ncosmo
    
    def _2d_interp(self, ypred, diff=False, massdef='RockstarM200m'):
        '''
        2D interpolation for the cumlative halo mass function.
        '''
        if massdef == 'RockstarM200m':
            Delta = 200
            rho_type = 'matter'
            mhmin_ind = self.HMFRockstarM200m.mhmin_ind
            mhmax_ind = self.HMFRockstarM200m.mhmax_ind
            m_edges   = self.HMFRockstarM200m.m_edges   
        elif massdef == 'FoFM200c':
            Delta = 200
            rho_type = 'critical'
            mhmin_ind = self.HMFFoFM200c.mhmin_ind
            mhmax_ind = self.HMFFoFM200c.mhmax_ind
            m_edges   = self.HMFFoFM200c.m_edges
        elif massdef == 'RockstarMvir':
            Delta = "vir"
            rho_type = 'matter'
            mhmin_ind = self.HMFRockstarMvir.mhmin_ind
            mhmax_ind = self.HMFRockstarMvir.mhmax_ind
            m_edges   = self.HMFRockstarMvir.m_edges           
        else:
            raise ValueError('Mass definition %s is not supported.'%massdef)
        mcen      = 10**((np.log10(m_edges[1:]) + np.log10(m_edges[:-1]))/2)
        dlnM      = np.log(m_edges[1]) - np.log(m_edges[0])
        m_edges_l = np.logspace(10, 17, 70+1)
        mcen_l    = 10**((np.log10(m_edges_l[1:]) + np.log10(m_edges_l[:-1]))/2)
        t08base   = self.get_dndlnM_Castro23(z=self.zlists, M=mcen_l, Pcb=True, revisted=True, massdef=massdef)
        t08base   = np.cumsum(t08base[:,::-1] * dlnM * 1e9, axis=1)[:,::-1]  
        ypred_ = np.zeros((len(self.zlists), len(mcen_l)))
        for iz in range(len(self.zlists)):
            icol_sta = 0 if iz == 0 else np.sum(mhmax_ind[:iz]-mhmin_ind[:iz])
            icol_end = icol_sta + mhmax_ind[iz]-mhmin_ind[iz]
            xplt     = np.log10(m_edges[mhmin_ind[iz]:mhmax_ind[iz]])
            yplt     = ypred[icol_sta:icol_end]# * t08base[iz, mhmin_ind[iz]:mhmax_ind[iz]]
            yind     = yplt > 0
            yfunc    = UnivariateSpline(xplt[yind], np.log10(yplt[yind]), k=1, s=0, ext=0)
            #yfunc    = interp1d(xplt[yind], np.log10(yplt[yind]), kind='cubic', fill_value='extrapolate')
            ypred_[iz,:] = 10**yfunc(np.log10(m_edges_l[:-1]))
            
        # ypred_[ypred_<=0] = 1e-32
        ypred_  = ypred_[::-1,:] * t08base[::-1,:] # invert the redshift
        ### z space use cubic spline while M space use cubic interpolation
        if diff:
            dlgM_ = 0.1
            dlnM_ = dlgM_ * np.log(10)
            addnum = 1 - np.min(ypred_)
            ypred_new = np.zeros_like(ypred_)
            for iz in range(len(self.zlists)):
                Nfunc = UnivariateSpline(np.log10(m_edges_l[:-1]), np.log10(ypred_[iz,:]+self.addnum), k=3, s=0, ext=0)
                ypred_new[iz,:] = (10**Nfunc(np.log10(mcen_l)-dlgM_/2)-addnum - 10**Nfunc(np.log10(mcen_l)+dlgM_/2)-addnum) / dlnM_
            self.addnum = 1 - np.min(ypred_new)
            spline = lambda z,M: 10**RectBivariateSpline(self.zlists[::-1], np.log10(mcen_l), \
                                                         np.log10(ypred_new+self.addnum), kx=1, ky=1)(z, np.log10(M))
        else:
            self.addnum = 1 - np.min(ypred_)
            spline = lambda z,M: 10**RectBivariateSpline(self.zlists[::-1], np.log10(m_edges_l[:-1]), \
                                                         np.log10(ypred_+self.addnum), kx=1, ky=1)(z, np.log10(M))
        return spline
    
    def _get_massdef_Delta(self, z, Pcb=True, Delta=200, rho_type='matter'):
        if Pcb:
            Omegam_z = self.Cosmo.get_Omegam
        else:
            Omegam_z = self.Cosmo.get_OmegaM
        if str(Delta) == "vir":
            x = Omegam_z(z) - 1
            Delta = 18*np.pi**2 + 82*x - 39*x**2
            Delta = Delta / Omegam_z(z)
            return Delta
        if rho_type == "critical":
            Delta = Delta / Omegam_z(z)
        elif rho_type == "matter":
            Delta = Delta
        return Delta

    def _get_lagrangian_Radius(self, M=None, Pcb=True):
        '''
        Get the Lagrangian radius of the haloes with mass M.
        Args:
            M  : float or array-like, halo mass [Msun/h]
            Pcb: bool, whether to use the total power spectrum (if False) or the cb power spectrum (if True [default])
        
        Return:
            array-like : Lagrangian radius [Mpc/h]
        '''
        M = np.atleast_1d(M)
        if Pcb:
            rho_m = self.Cosmo.rho_crit * self.Cosmo.Omegam ## h^{2}M_{sun}Mpc^{-3}
        else:
            rho_m = self.Cosmo.rho_crit * self.Cosmo.OmegaM ## h^{2}M_{sun}Mpc^{-3}
        return (3*M/4/np.pi/rho_m)**(1/3)   # Mpc/h

    def _get_delta_c(self, z=None, Pcb=True):
        '''
        Get the overdensity parameter delta_c.
        Args:
            z  : float or array-like, redshift
            Pcb: bool, whether to use the total power spectrum (if False) or the cb power spectrum (if True [default])
        Return:
            array-like : overdensity parameter delta_c
        '''
        z = np.atleast_1d(z)
        if Pcb:
            Omega_m_zall = self.Cosmo.get_Omegam(z=z)
        else:
            Omega_m_zall = self.Cosmo.get_OmegaM(z=z)
        return (3/20 * (12*np.pi)**(2/3) * (1 + 0.0123*np.log10(Omega_m_zall)))

    def _get_dlns_dlnR(self, z=None, M=None, dlnR=0.01, Pcb=True):
        '''
        Get the dln\sigma/dlnR.
        Args:
            z   : float, redshift
            M   : float or array-like, halo mass [Msun/h]
            dlnR: float, the step of the Lagrangian radius [Mpc/h]
            Pcb : bool, whether to use the total power spectrum (if False) or the cb power spectrum (if True [default]) 
        '''
        lnR_c  = np.log(self._get_lagrangian_Radius(M=M, Pcb=Pcb))
        R_m    = np.exp(lnR_c - dlnR/2)
        R_p    = np.exp(lnR_c + dlnR/2)
        if Pcb:
            sigma_m = self.get_sigma_cb_z(R=R_m, z=z, type='Emulator')
            sigma_p = self.get_sigma_cb_z(R=R_p, z=z, type='Emulator')
        else:
            sigma_m = self.get_sigma_z(R=R_m, z=z, type='Emulator')
            sigma_p = self.get_sigma_z(R=R_p, z=z, type='Emulator') 
        return np.log(sigma_p/sigma_m)/dlnR
    
    def get_dndlnM_Tinker08(self, z=None, M=None, Pcb=True, Delta=200, rho_type='matter'):
        '''
        Get the halo mass function with Tinker08 model.
        
        Args:
            z  : float or array-like, redshift
            M  : float or array-like, halo mass [Msun/h]
            Pcb: bool, whether to use the total power spectrum (if False) or the cb power spectrum (if True [default]) 
        '''
        
        z  = np.atleast_1d(z)
        delta = np.array(
            [200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.])
        A0arr = np.array(
            [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        a0arr = np.array(
            [1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        b0arr = np.array(
            [2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        c0arr = np.array(
            [1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44]) 
        ldelta = np.log10(delta)
        A0func = interp1d(ldelta, A0arr, fill_value="extrapolate") 
        a0func = interp1d(ldelta, a0arr, fill_value="extrapolate")
        b0func = interp1d(ldelta, b0arr, fill_value="extrapolate")
        c0func = interp1d(ldelta, c0arr, fill_value="extrapolate")
        Delta  = self._get_massdef_Delta(z, Pcb=Pcb, Delta=Delta, rho_type=rho_type)
        if Pcb:
            Omegacb   = self.Cosmo.Omegam # better to use the cb value for nuCDM universe
            sigmafunc = self.get_sigma_cb_z
        else:
            Omegacb = self.Cosmo.OmegaM
            sigmafunc = self.get_sigma_z
        A0      = A0func(np.log10(Delta))
        a0      = a0func(np.log10(Delta))
        b0      = b0func(np.log10(Delta))
        c0      = c0func(np.log10(Delta))
        alpha   = 10**( -(0.75/np.log10(Delta/75))**1.2 )
        A       = A0*(1+z)**(-0.14)
        a       = a0*(1+z)**(-0.06)
        b       = b0*(1+z)**(-alpha)
        c       = c0*np.ones_like(z)
        dlnM    = 0.01
        rho_cb  = Omegacb*self.Cosmo.rho_crit ## h^{2}M_{sun}Mpc^{-3}
        M2R     = lambda M: (3*M/4/np.pi/rho_cb)**(1/3)  # Mpc/h
        out = np.zeros((len(z), len(M)))
        for zind, iz in enumerate(z):
            sigma_p   = sigmafunc(z=iz, R=M2R(M*np.exp( dlnM/2)), type='Emulator')
            sigma_m   = sigmafunc(z=iz, R=M2R(M*np.exp(-dlnM/2)), type='Emulator')
            sigma_c   = sigmafunc(z=iz, R=M2R(M)                , type='Emulator')
            dlns      = np.log(sigma_p**(-1)) - np.log(sigma_m**(-1))
            fsigma    = A[zind]*((sigma_c/b[zind])**(-a[zind]) + 1) * np.exp(-c[zind]/sigma_c/sigma_c)
            out[zind] = fsigma * rho_cb/M/dlnM*dlns
        return out
    
    def get_dndlnM_Castro23(self, z=None, M=None, Pcb=True, revisted=True, massdef='RockstarMvir'):
        '''
        Get the halo mass function with Castro23 model.
        
        Args:
            z      : float or array-like, redshift
            M      : float or array-like, halo mass [Msun/h]
            Pcb    : bool, whether to use the total power spectrum (if False[just for test]) or the cb power spectrum (if True [default]) 
            revisted: bool, whether to use the revised Castro23 model (if True[default]) or the original Castro23 model (if False)
            massdef : string, the mass definition used in the emulator
            
        .. note:: 
            For the original Castro23 model, the mass definition only supports RockstarM200m.
            For the revised Castro23 model, the mass definition supports RockstarM200m, FoFM200c, and RockstarMvir.
            
        '''
        z  = np.atleast_1d(z)
        if revisted:
            if massdef == 'RockstarM200m':
                a1,a2,az,p1,p2,q1,q2,qz = np.array([0.77283085,0.3686431,-0.03214047,-0.46080809,-0.53419608,0.37336379,-0.32322067,-0.19204407])
            elif massdef == 'FoFM200c':
                a1,a2,az,p1,p2,q1,q2,qz = np.array([0.76595201,0.38539018,-0.14234157,-0.52358138,-0.73425449,0.32078283,-0.40464143,0.1526956])
            elif massdef == 'RockstarMvir':
                a1,a2,az,p1,p2,q1,q2,qz = np.array([0.76447491,0.41714731,-0.06177372,-0.46860976,-0.65112204,0.38826178,-0.36955881,-0.0068741])
            else:
                raise ValueError('The massdef = %s is not supported yet.'%massdef)
        else:
            if massdef == 'RockstarMvir':
                a1,a2,az,p1,p2,q1,q2,qz = np.array([0.7962, 0.1449, -0.0658, -0.5612, -0.4743, 0.3688, -0.2804, 0.0251])
            else:
                raise ValueError('The massdef = %s is not supported yet.'%massdef)
        if Pcb:
            Omegamfunc = self.Cosmo.get_Omegam
            rho_m      = self.Cosmo.rho_crit * self.Cosmo.Omegam ## h^{2}M_{sun}Mpc^{-3}
        else:
            Omegamfunc = self.Cosmo.get_OmegaM
            rho_m      = self.Cosmo.rho_crit * self.Cosmo.OmegaM ## h^{2}M_{sun}Mpc^{-3}
        dndlnM = np.zeros((len(z), len(M)))
        for iz in range(len(z)):
            dlnsdlnR = self._get_dlns_dlnR(z=z[iz], M=M, Pcb=Pcb)
            sigma    = self.get_sigma_cb_z(R=self._get_lagrangian_Radius(M=M, Pcb=Pcb), z=z[iz])
            Omegamz  = Omegamfunc(z=z[iz])
            nu       = self._get_delta_c(z=z[iz], Pcb=Pcb)[0]/sigma
            aR = a1 + a2*(dlnsdlnR+0.6125)*(dlnsdlnR+0.6125)
            qR = q1 + q2*(dlnsdlnR+0.5)
            a  = aR * Omegamz**az
            q  = qR * Omegamz**qz
            p  = p1 + p2*(dlnsdlnR+0.5)
            A  = 1/((2**(-0.5-p+q/2)/np.sqrt(np.pi))*(2**p*gamma(0.5*q)+gamma(-p+0.5*q)))
            multiplicity = A*np.sqrt(2*a*nu*nu/np.pi)*np.exp(-0.5*a*nu*nu)*(1+1/((a*nu*nu)**p))*(np.sqrt(a)*nu)**(q-1)
            dndlnM[iz] = multiplicity * rho_m / M * (-1/3 * dlnsdlnR)
        return dndlnM
 
    def get_Nhalo(self, z=None, M=None, V=1, massdef='RockstarM200m'):
        '''
        Get the number of haloes whose mass is not smaller than M. [ $N(\geq M)$ ]
        
        Args:
            z : float or array-like, redshift
            M : float or array-like, halo mass [Msun/h]
            V : float, volume [Mpc/h]^3
        Return:
            array-like : halo mass function with shape (len(z), len(m))
        '''
        z = check_z(self.zlists, z)
        M = np.atleast_1d(M)
        if massdef == 'RockstarM200m':
            ypred_ = self.HMFRockstarM200m.get_data()
        elif massdef == 'FoFM200c':
            ypred_ = self.HMFFoFM200c.get_data()
        elif massdef == 'RockstarMvir':
            ypred_ = self.HMFRockstarMvir.get_data()
        else:
            raise ValueError('The massdef = %s is not supported yet.'%massdef)
        cumhmf = self._2d_interp(ypred_, massdef=massdef)(z=z, M=M) - self.addnum
        return cumhmf * V * 1e-9

    def get_dndlnM(self, z=None, M=None, massdef='RockstarM200m'):
        '''
        Get the number density of haloes in the mass bin [ $dn/d\ln M$ ] with unit of [Mpc/h]^-3.
        [Affected by the binning effect, not recommended to use.]
        
        Args:
            z : float or array-like, redshift
            M : float or array-like, halo mass [Msun/h]
        Return:
            array-like : halo mass function with shape (len(z), len(M))
        '''
        z = check_z(self.zlists, z)
        M = np.atleast_1d(M)
        if massdef == 'RockstarM200m':
            ypred_ = self.HMFRockstarM200m.get_data()
        elif massdef == 'FoFM200c':
            ypred_ = self.HMFFoFM200c.get_data()
        elif massdef == 'RockstarMvir':
            ypred_ = self.HMFRockstarMvir.get_data()
        else:
            raise ValueError('The massdef = %s is not supported yet.'%massdef) 
        dndlnM = self._2d_interp(ypred_, diff=True, massdef=massdef)(z=z, M=M) - self.addnum
        return dndlnM * 1e-9


####### class for the weak lensing statistics emulator
class WeakLensingBaseEmulator(CBaseEmulator):
    '''
    weak lensing part
    '''

    def _get_distance_interp(self, use_ccl=False):
        t0 = time.time()
        zmax = 3.1
        if use_ccl:
            import pyccl
            Ob = self.Cosmo.Omegab; Oc  = self.Cosmo.Omegac
            h0 = self.Cosmo.h0;     As  = self.Cosmo.As
            ns = self.Cosmo.ns;     mnu = self.Cosmo.mnu
            w0 = self.Cosmo.w0;     wa  = self.Cosmo.wa
            cosmo_ccl = pyccl.cosmology.Cosmology(Omega_c=Oc, Omega_b=Ob, h=h0, A_s=As, n_s=ns, m_nu=mnu, w0=w0, wa=wa,
                                                  transfer_function='boltzmann_camb', mass_split='single')
            func = lambda z: cosmo_ccl.comoving_radial_distance(1/(1+z))
            self.zinterp   = np.linspace(0, zmax, 1000)
        else:
            func = self.Cosmo.comoving_distance
            self.zinterp   = np.linspace(0, zmax, 1000)
        self.chiinterp = func(self.zinterp)
        self.chi2z   = interp1d(self.chiinterp, self.zinterp, 
                                kind='linear') # comoving distance [Mpc] to redshift 
        self.z2chi   = interp1d(self.zinterp, self.chiinterp, 
                                kind='linear') # redshift to comoving distance [Mpc]
        t1 = time.time()
        if self.verbose:
            print('Time for perparing the comoving distance to z function: %.2f seconds'%(t1-t0))
        return self.chi2z, self.z2chi
    
    def get_lensing_kernel(self, chi=None, dndz=None, Pcb=False, use_ccl=False):
        '''
        Get the weak lensing kernel.
        
        Args:
            chi : float or array-like, comoving distance [Mpc]
            dndz: tuple, redshift distribution function, tuple (zarr, narr)
            Pcb : bool, whether to output the total matter (if False [default]) or the cb component (if True )
            Pcb=True only for my specific case. For general case, you should set Pcb=False.
        Return:
            array-like : weak lensing kernel with shape (len(z), len(chi))
        '''
        vc  = self.Cosmo.vel_light * 1e-5 # cm/s -> km/s
        chi = np.atleast_1d(chi)
        zarr, narr = np.copy(dndz)
        narr = narr/narr.max()
        dndzfunc = interp1d(zarr, narr, kind='linear', 
                            fill_value=0, bounds_error=False)
        znew  = np.linspace(np.min(zarr), np.max(zarr), 512+1)
        dzarr = znew[1:] - znew[:-1]
        znew  = (znew[1:] + znew[:-1]) / 2
        nnew  = dndzfunc(znew)
        H0 = self.Cosmo.h0 * 100
        if Pcb:
            Omegam = self.Cosmo.Omegam
        else:
            Omegam = self.Cosmo.OmegaM
        kernel = np.zeros((len(chi)))
        if not hasattr(self, 'z2chi') or not hasattr(self, 'chi2z'):
            self.chi2z, self.z2chi = self._get_distance_interp(use_ccl=use_ccl)
        elif self.z2chi is None or self.chi2z is None:
            self.chi2z, self.z2chi = self._get_distance_interp(use_ccl=use_ccl) 
        z_chi = self.chi2z(chi)
        for ii, ichi in enumerate(chi):
            zi = z_chi[ii]
            ai = 1/(1+zi)
            zind  = znew>=zi
            chiz  = self.z2chi(znew[zind])
            # norm = np.trapz(nnew, znew)
            norm  = np.sum(nnew*dzarr)
            if np.isclose(norm, 0, atol=1e-5):
                kernel[ii] = 0
            else:
                warr  = 3*H0*H0*Omegam/2/ai/vc/vc * (ichi*(chiz - ichi)/chiz) * dzarr[zind]*nnew[zind] # 1/Mpc/Mpc
                kernel[ii] = np.sum(warr)/norm
        return kernel

    def get_kappa_kernel(self, chi=None, z_s=1100, Pcb=False, use_ccl=False):
        '''
        Get the lensing kernel for a specific source redshift.
        
        Args:
            chi : float or array-like, comoving distance [Mpc]
            z_s : float, source redshift
            Pcb : bool, whether to output the total matter (if False [default]) or the cb component (if True )
        Return:
            array-like : lensing kernel with shape (len(z))
        '''
        vc   = self.Cosmo.vel_light * 1e-5 # cm/s -> km/s
        chi  = np.atleast_1d(chi)
        if not hasattr(self, 'z2chi') or not hasattr(self, 'chi2z'): 
            self.chi2z, self.z2chi = self._get_distance_interp(use_ccl=use_ccl)
        elif self.z2chi is None or self.chi2z is None:
            self.chi2z, self.z2chi = self._get_distance_interp(use_ccl=use_ccl) 
        aarr = 1/(1+self.chi2z(chi)) 
        chis = self.Cosmo.comoving_distance(z_s)
        H0   = self.Cosmo.h0 * 100
        if Pcb:
            Omegam = self.Cosmo.Omegam
        else:
            Omegam = self.Cosmo.OmegaM
        return 3*H0*H0*Omegam/2/aarr/vc/vc * (chi*(chis - chi)/chis) # 1/Mpc/Mpc

####### class for the convergence power spectrum emulator
class Cell_CEmulator(Pkmm_CEmulator, WeakLensingBaseEmulator):
    '''
    The convergence power spectrum emulator class.
    '''
    def get_Limber_Cells(self, ells=None, dndz=None, z_s=None, Pcb=False, non_linear='Emulator', return_shot_noise=None, verbose=False, use_ccl=False):
        '''
        Get the weak lensing power spectrum.
        
        Args:
            ells : float or array-like, multipole
            dndz : tuple, redshift distribution function, tuple (zarr, narr)
            z_s  : float, source redshift for single source plane
            Pcb  : bool, whether to output the total matter (if False [default]) or the cb component (if True )
            non_linear : string, 'Emulator', 'halofit' or 'linear' power spectrum
            return_shot_noise : float, return the shot noise if not None. The vaule should be the shot noise level V_{sim}/N_{sim} [Mpc^3/h^3].
            verbose : bool, whether to output the time for each step
        Return:
            array-like : weak lensing power spectrum with shape (len(ells))
        '''
        ells = np.atleast_1d(ells)
        if dndz is not None:
            if z_s is not None:
                raise ValueError('You should provide the dndz or z_s, not both.')
            zarr, narr = dndz
            narr = narr/narr.max()
            zmax = np.max(zarr)
        elif z_s is not None:
            zmax = z_s
        elif z_s is None:
            raise ValueError('You should provide the dndz or z_s.')
            
        nell = len(ells)
        cl_kappa = np.zeros(nell)
        zall  = np.linspace(0, zmax, 100+1, endpoint=True) 
        if not hasattr(self, 'z2chi') or not hasattr(self, 'chi2z'):
            self.chi2z, self.z2chi = self._get_distance_interp(use_ccl=use_ccl)
        elif self.z2chi is None or self.chi2z is None:
            self.chi2z, self.z2chi = self._get_distance_interp(use_ccl=use_ccl) 
        chis  = self.z2chi(zall) 
        dchis = chis[1:] - chis[:-1]
        chis  = (chis[1:] + chis[:-1]) / 2
        zall  = (zall[1:] + zall[:-1]) / 2
        t0 = time.time()
        if dndz is not None:
            Wallchi = self.get_lensing_kernel(chi=chis, dndz=dndz, Pcb=Pcb, use_ccl=use_ccl)
        elif z_s is not None:
            Wallchi = self.get_kappa_kernel(chi=chis, z_s=z_s, Pcb=Pcb, use_ccl=use_ccl)
        t1 = time.time()
        if verbose:
            print('Time for kernel:', t1-t0)
        h0 = self.Cosmo.h0
        if return_shot_noise is None:
            if non_linear == 'Emulator':
                karr  = np.logspace(-2.2, 1.0, 1000)
                bkarr = self.get_pknl (z=zall, k=karr, Pcb=Pcb, lintype='Emulator') \
                      / self.get_pklin(z=zall, k=karr, Pcb=Pcb,    type='Emulator')
                bk2Dfunc = RegularGridInterpolator(method='linear', bounds_error=False, fill_value=None, points=(zall, np.log10(karr)), values=np.log10(bkarr)) 
                karr2    = np.logspace(-4.99, 1, 1000)
                plarr    = self.get_pklin(z=zall, k=karr2, Pcb=Pcb, type='Emulator')
                pl2Dfunc = RegularGridInterpolator(method='linear', bounds_error=False, fill_value=None, points=(zall, np.log10(karr2)), values=np.log10(plarr))
                Pk2Dfunc = lambda z, k: (10**pl2Dfunc(list(zip(z, np.log10(k/h0)))))*(10**bk2Dfunc(list(zip(z, np.log10(k/h0)))))/h0/h0/h0

            elif non_linear == 'halofit':
                karr  = np.logspace(-4.99, 1, 1000)
                pkarr = self.get_pkhalofit(z=zall, k=karr, Pcb=Pcb, lintype='Emulator')
                pk2dfunc = RegularGridInterpolator(method='linear', bounds_error=False, fill_value=None, points=(zall, np.log10(karr)), values=np.log10(pkarr))
                Pk2Dfunc = lambda z, k: (10**pk2dfunc(list(zip(z, np.log10(k/h0)))))/h0/h0/h0
                
            elif non_linear == 'linear':
                karr  = np.logspace(-4.99, 1, 1000)
                pkarr = self.get_pklin(z=zall, k=karr, Pcb=Pcb, type='Emulator')
                pk2dfunc = RegularGridInterpolator(method='linear', bounds_error=False, fill_value=None, points=(zall, np.log10(karr)), values=np.log10(pkarr))
                Pk2Dfunc = lambda z, k: (10**pk2dfunc(list(zip(z, np.log10(k/h0)))))/h0/h0/h0
                
            else:
                raise ValueError('non_linear %s not supported yet.'%non_linear)
        else:
            Pk2Dfunc = lambda z, k: return_shot_noise /h0/h0/h0 * np.ones_like(k)
        t2 = time.time()
        if verbose:
            print('Time for preparing 2D interpolation:', t2-t1)
        
        for i, l in enumerate(ells):
            k = (l + 0.5)/chis
            # grid interpolation
            pkgrid = Pk2Dfunc(z=zall, k=k) 
            cl_kappa[i] = np.dot(dchis, (pkgrid*Wallchi*Wallchi/chis/chis))
        t3 = time.time()
        if verbose:
            print('Time for ells(n=%d) loop:'%nell, t3-t2)
            print('Total time:', t3-t0)
        # clear the interpolation
        self.chi2z = None
        self.z2chi = None
        return cl_kappa
   
   
 
