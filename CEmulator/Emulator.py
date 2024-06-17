import numpy as np
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
        self.verbose = verbose
        self.Cosmo        = Cosmology(verbose=verbose)
        self.Bkmm         = Bkmm_gp(verbose=verbose) 
        self.Bkmm_halofit = Bkmm_halofit_gp(verbose=verbose) 
        self.Pkmmlin      = PkcbLin_gp(verbose=verbose) 
        self.XihmMassBin  = XihmMassBin_gp(verbose=verbose) 
        self.PkhmMassBin  = PkhmMassBin_gp(verbose=verbose)  
        
        
    def set_cosmos(self, Omegab=0.04897468, Omegam=0.30969282, 
                   H0=67.66, As=2.105e-9, ns=0.9665, w=-1.0, wa=0.0, 
                   mnu=0.06):
        '''
        Set the cosmological parameters. You can input the float or array-like.
        Omegab : float or array-like, baryon density
        Omegam : float or array-like, Only baryon and CDM density
        H0     : float or array-like, Hubble constant
        As     : float or array-like, amplitude of the primordial power spectrum
        ns     : float or array-like, spectral index
        w      : float or array-like, dark energy equation of state
        wa     : float or array-like, dark energy equation of state evolution
        mnu    : float or array-like, sum of neutrino masses with unit eV
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
        # Omeganu = cosmos['mnu']/93.14/cosmos['H0']/cosmos['H0'] * 1e4
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
        ### set the cosmology class
        self.Cosmo.set_cosmos(cosmos)
        ## into the cosmologies array
        self.cosmologies = np.zeros((numcosmos, n_params))
        for ind, ikey in enumerate(self.param_names):
            self.cosmologies[:,ind] = cosmos[ikey]
        ### normalize the cosmologies
        self.ncosmo = NormCosmo(self.cosmologies, self.param_names, self.param_limits)
        
        ### pass the cosmologies (normalized) to other class
        self.Bkmm.ncosmo         = self.ncosmo 
        self.Bkmm_halofit.ncosmo = self.ncosmo
        self.Pkmmlin.ncosmo      = self.ncosmo  
        self.XihmMassBin.ncosmo  = self.ncosmo
        self.PkhmMassBin.ncosmo  = self.ncosmo
        
        self.numcos = self.cosmologies.shape[0]
        #### from init move to here 
        #### refresh the cosmology class for each cosmology set 
        self.cosmo_class_arr = None
                    
    def get_cosmos_class(self, z=None, non_linear=None, kmax=10):
        '''
        Get the CLASS cosmology object.
        z : float or array-like, redshift
        non_linear: 'halofit' or 'HMcode'
        '''
        # check_z(z)
        str_zlists = "{:.4f}".format(z[0])
        if len(z) > 1:
            for i_z in range(len(z) - 1):
                str_zlists += ", {:.4f}".format(z[i_z+1])
        self.cosmo_class_arr = np.zeros((self.numcos,), dtype=object)
        for ic in range(self.numcos):
            self.cosmo_class_arr[ic] = useCLASS(self.cosmologies[ic], str_zlists, non_linear=non_linear, kmax=kmax)
        return True
    
    def get_pklin(self, z=None, k=None, type='CLASS', Pcb=True, kmax=10):
        '''
        Get the linear power spectrum.
        
        Parameters
        ----------
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        type : string, 'CLASS' or 'Emulator'
        Pcb  : bool, whether to output the total power spectrum (if False) or 
               the cb power spectrum (if True [default])
        kmax : float, maximum wavenumber [h/Mpc] for CLASS
        '''
        z = check_z(self.zlists,     z)
        # k = check_k(self.Bkmm.klist, k)
        if   type == 'CLASS':
            if self.cosmo_class_arr is None:
                self.get_cosmos_class(z, kmax=kmax)
            pklin = np.zeros((self.numcos, len(z), len(k)))
            for ic in range(self.numcos):
                if Pcb and self.cosmo_class_arr[ic].Omega_nu != 0:
                    pkfunc = self.cosmo_class_arr[ic].pk_cb_lin
                else:
                    pkfunc = self.cosmo_class_arr[ic].pk_lin
                h0 = self.cosmo_class_arr[ic].h()
                for iz in range(len(z)):
                    pklin[ic, iz] = np.array([pkfunc(ik*h0, z[iz])*h0*h0*h0 
                                                for ik in list(k)])
        elif type == 'Emulator':
            if Pcb:
                pklin = self.Pkmmlin.get_pkLin(z, k)
            else:
                raise ValueError('For total power spectrum, use type=CLASS NOW.')     
        else:
            raise ValueError('Type %s not supported yet.'%type)
        return pklin
        
    def get_pkhalofit(self, z=None, k=None, Pcb=True, lintype='CLASS'):
        '''
        Get the halofit power spectrum.
        ******
        Notice This version can not converge with CLASS in the high redshift (z>2.5)
        for the c0001 and c0091 (w0 and wa near the lower limit).
        ******
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        Pcb : bool, whether to output the total power spectrum (if False) or 
              the cb power spectrum (if True [default])
        lintype : string, 'CLASS' or 'Emulator'
        '''
        z = check_z(self.zlists,     z)
        if Pcb:
            fnuall = np.zeros((self.numcos))
        else:
            fnuall = self.Cosmo.Omeganu/(self.Cosmo.Omegam + self.Cosmo.Omeganu)
        OmegaLzall = self.Cosmo.get_OmegaL(z)
        OmegaMzall = self.Cosmo.get_OmegaM(z)
        pkhalofit  = np.zeros((self.numcos, len(z), len(k)))
        kinterp    = np.logspace(-5, 1, 1024)
        pklinintp  = self.get_pklin(z, kinterp, type=lintype, Pcb=Pcb, kmax=kinterp.max())
        for ic in range(self.numcos):
            fnu = fnuall[ic]
            for iz in range(len(z)):
                OmegaLz = OmegaLzall[ic,iz]
                OmegaMz = OmegaMzall[ic,iz]
                w_eff   = self.Cosmo.w0[ic] + self.Cosmo.wa[ic]*(z[iz]/(1+z[iz]))
                Pklin = lambda k: 10**interp1d(np.log10(kinterp), 
                                               np.log10(pklinintp[ic,iz]),
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
                pkhalofit[ic,iz] = PkHaloFit(k, Pklin(k), R_sigma, OmegaMz, OmegaLz, fnu, \
                                             an, bn, cn, gamman, alphan, betan, mun, nun)
        return pkhalofit
        
    def get_pknl(self, z=None, k=None, Pcb=True, lintype='CLASS', nltype='linear'):
        '''
        Get the nonlinear power spectrum.
        z : float or array-like, redshift
        k : float or array-like, wavenumber [h/Mpc]
        lintype : string, 'CLASS' or 'Emulator'
        nltype  : string, 'linear' or 'halofit'
                  'linear' means ratio of nonlinear to linear power spectrum
                  'halofit' means ratio of nonlinear to halofit power spectrum
        '''
        z = check_z(self.zlists,     z)
        k = check_k(self.Bkmm.klist, k)
        if   nltype == 'linear':
            ## get the nonlinear transfer for Pcb
            Bkpred = self.Bkmm.get_Bk(z, k)
            ## get the linear power spectrum for cb
            pkcblin = self.get_pklin(z, k, type=lintype, Pcb=Pcb)
        elif nltype == 'halofit':
            ## get the nonlinear transfer for Pcb
            Bkpred = self.Bkmm_halofit.get_Bk(z, k)
            ## get the halofit power spectrum for cb
            if   lintype == 'Emulator':
                pkcblin = self.get_pkhalofit(z, k, Pcb=Pcb, lintype=lintype)
            elif lintype == 'CLASS':
                if self.cosmo_class_arr is None:
                    self.get_cosmos_class(z, non_linear='halofit')
                pkcblin = np.zeros((self.numcos, len(z), len(k)))
                for ic in range(self.numcos):
                    h0 = self.cosmo_class_arr[ic].h()
                    for iz in range(len(z)):
                        pkcblin[ic,iz] = np.array([self.cosmo_class_arr[ic].pk_cb(ik*h0, z[iz])*h0*h0*h0 
                                                   for ik in list(k)])
        if Pcb:
            pknl = pkcblin * Bkpred
        else:
            if lintype == 'CLASS':
                if self.cosmo_class_arr is None:
                    self.get_cosmos_class(z)
                pknl   = np.zeros_like(pkcblin)
                h0     = self.cosmo_class_arr[0].h()
                # pklin  = self.get_pklin(z, k, type=lintype, Pcb=Pcb)
                pkcbnl = pkcblin * Bkpred
                for ic in range(self.numcos):
                    if self.cosmo_class_arr[ic].Omega_nu != 0:
                        for iz in range(len(z)):
                            tkall = self.cosmo_class_arr[ic].get_transfer(z=z[iz])
                            # Tktotsquare_interp = interp1d(tkall['k (h/Mpc)'], tkall['d_tot']*tkall['d_tot'], kind='linear')
                            Tknusquare_interp  = interp1d(tkall['k (h/Mpc)'], tkall['d_ncdm[0]']*tkall['d_ncdm[0]'], kind='linear')
                            fb2cb = self.cosmo_class_arr[ic].Omega_b()/(self.cosmo_class_arr[ic].Omega0_cdm()+self.cosmo_class_arr[ic].Omega_b())
                            fc2cb = self.cosmo_class_arr[ic].Omega0_cdm()/(self.cosmo_class_arr[ic].Omega0_cdm()+self.cosmo_class_arr[ic].Omega_b())
                            Tkcb  = fc2cb*tkall['d_cdm']+fb2cb*tkall['d_b']
                            Tkcb_interp = interp1d(tkall['k (h/Mpc)'], Tkcb, kind='linear')
                            fnu   = self.cosmo_class_arr[ic].Omega_nu/self.cosmo_class_arr[ic].Omega_m()
                            # Tkcb_nu_interp = interp1d(tkall['k (h/Mpc)'], tkall['d_ncdm[0]']*Tkcb, kind='linear')
                            ## use the linear neutrino power spectrum to calculate the nonlinear total power spectrum
                            pknunulin = pkcblin[ic,iz]/Tkcb_interp(k)/Tkcb_interp(k) * Tknusquare_interp(k)
                            pknl[ic,iz] = (1-fnu)**2*pkcbnl[ic,iz] + fnu*fnu*pknunulin + 2*fnu*(1-fnu)*np.sqrt(pkcbnl[ic,iz]*pknunulin)
                    else:
                        pknl[ic] = pkcblin[ic] * Bkpred[ic] 
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
        linterp = self.get_pklin(z, kinterp, type='Emulator', Pcb=True)
        bkout = np.zeros((pinterp.shape[0], pinterp.shape[1], pinterp.shape[2], len(k)))
        for i1 in range(pinterp.shape[0]):         # cosmology
            for i2 in range(pinterp.shape[1]):     # massbin 
                for i3 in range(pinterp.shape[2]): # redshift
                    ### introduce a smooth process
                    datainterp = pinterp[i1,i2,i3]/linterp[i1,i3]
                    datainterp = savgol_filter(datainterp, window_length=9, polyorder=3)
                    funcinterp = interp1d(np.log10(kinterp),
                                          datainterp,
                                          kind='slinear', fill_value="extrapolate")
                    bkout[i1,i2,i3] = funcinterp(np.log10(k))     
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
        ### cosmology and number density
        xi_trees = np.zeros((bkhm.shape[0], bkhm.shape[1], len(z), len(r)))
        for ic in range(bkhm.shape[0]):       # cosmology
            for im in range(bkhm.shape[1]):   # massbin
                for iz in range(bkhm.shape[2]):
                    r0, xi0 = P2xi(ks, bkhm[ic,im,iz]*pkcblin[ic,iz], 0)
                    xi_trees[ic,im,iz] = ius(r0,xi0.real)(r)
        return xi_trees
       
    def get_xihmMassBin(self, z=None, r=None):
        '''
        Get the halo-matter cross correlation function.
        z : float or array-like, redshift
        r : float or array-like, wavenumber [Mpc/h]
        '''
        xi_dire = self.XihmMassBin.get_xihmMassBin(z, r)
        xi_tree = self._get_xi_tree(z, r)
        rswitch = 40.0 # Mpc/h
        xi_comb = np.zeros_like(xi_dire)
        for ic in range(xi_dire.shape[0]):
            for im in range(xi_dire.shape[1]):
                xi_comb[ic, im] = xi_dire[ic, im] * np.exp(-(r/rswitch)**4) \
                                + xi_tree[ic, im] * (1 - np.exp(-(r/rswitch)**4))
        return xi_comb
