from scipy.optimize import minimize
from .cosmology import Cosmology
import numpy as np
from copy import deepcopy
#####################################################################################################################
######################## For the spectral equivalence method ########################################################
#####################################################################################################################

######################## A stable Newton's method with finite difference derivative #################################
def finite_diff_newton(f, x0, tol=1e-10, max_iter=100, h=1e-5):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, fx
        # adaptive step size
        h = 0.5 * h * (tol / abs(fx))**0.5
        h_adaptive = max(h, 1e-8)
        df = (f(x + h_adaptive) - f(x - h_adaptive)) / (2 * h_adaptive)
        if abs(df) < 1e-32:
            ### when derivative is very small, use a stable method to find the root
            x_new = x*(1+fx)**10
        else:
            x_new = x - fx / df
            if abs(x_new - x) < tol:
                return x_new, fx
        if abs(x_new) > 10:
            return np.NaN, np.NaN # if x_new is too large, return NaN
        x = x_new
    return np.NaN, np.NaN # if max_iter is reached

def GetErr_csstemu_w(wz, z, ichi, csstemu, zstar=1100):
    '''
    wz: [w0]
    z: redshift
    ichi: comoving distance from z to zstar
    csstemu: the CEmulator object
    zstar: redshift of the last scattering surface, varying it has little effect
    '''
    cosmos ={}
    cosmos['Omegab'] = csstemu.Cosmo.Omegab
    cosmos['Omegam'] = csstemu.Cosmo.Omegac + csstemu.Cosmo.Omegab
    cosmos['H0']     = csstemu.Cosmo.h0 * 100
    cosmos['A']      = csstemu.Cosmo.As * 1e9
    cosmos['ns']     = csstemu.Cosmo.ns
    cosmos['mnu']    = csstemu.Cosmo.mnu
    cosmos['w']      = wz
    cosmos['wa']     = 0
    csstemu_i  = Cosmology()
    csstemu_i.set_cosmos(cosmos)
    dlss_eq    = csstemu_i.comoving_distance(zstar) - csstemu_i.comoving_distance(z)
    return np.abs(1 - dlss_eq/ichi)

def GetErr_csstemu_w0wa(wz, z, ichi, csstemu, zstar=1100):
    '''
    wz: [w0, wa]
    z: redshift
    ichi: comoving distance from z to zstar
    csstemu: the CEmulator object
    zstar: redshift of the last scattering surface, varying it has little effect
    '''
    cosmos ={}
    cosmos['Omegab'] = csstemu.Cosmo.Omegab
    cosmos['Omegam'] = csstemu.Cosmo.Omegac + csstemu.Cosmo.Omegab
    cosmos['H0']     = csstemu.Cosmo.h0 * 100
    cosmos['A']      = csstemu.Cosmo.As * 1e9
    cosmos['ns']     = csstemu.Cosmo.ns
    cosmos['mnu']    = csstemu.Cosmo.mnu
    cosmos['w']      = wz[0]
    cosmos['wa']     = wz[1]
    csstemu_i  = Cosmology()
    csstemu_i.set_cosmos(cosmos)
    dlss_eq    = csstemu_i.comoving_distance(zstar) - csstemu_i.comoving_distance(z)
    return np.abs(1 - dlss_eq/ichi)

def SpectralEquivalence_DE(csstemu, zlists, wCDM=False, return_err=False):
    '''
    Spectral equivalence method to find the auxiliary wCDM or w0waCDM model for a given CPL model
    
    csstemu: the CEmulator object
    zlists: redshift list
    wCDM: if True, find the equivalent wCDM model, otherwise find the equivalent w0waCDM model
    return_err: if True, also return the error of the distance to lss
    '''
    chi_z_to_lss = csstemu.Cosmo.comoving_distance(1100) - csstemu.Cosmo.comoving_distance(zlists)
    if wCDM:
        wconst_arr = np.ones((len(chi_z_to_lss))) * csstemu.Cosmo.w0
        err_chi    = 100 * np.ones((len(chi_z_to_lss)))
        max_iter   = 100
        for cind, ichi in enumerate(chi_z_to_lss):
            func = lambda w: GetErr_csstemu_w(w, zlists[cind], ichi, csstemu)
            wconst_arr[cind], err_chi[cind] = finite_diff_newton(func, wconst_arr[cind], max_iter=max_iter, tol=1e-7)
        if return_err:        
            return wconst_arr, err_chi
        else:
            return wconst_arr
    else:
        ### modify the starting point to reduce the distance between the starting point and the target value
        w0i = -1.3 if csstemu.Cosmo.w0 < -1.3 else -0.7 if csstemu.Cosmo.w0 > -0.7 else csstemu.Cosmo.w0
        wai = -0.5 if csstemu.Cosmo.wa < -0.5 else 0.5 if csstemu.Cosmo.wa > 0.5 else csstemu.Cosmo.wa
        w0_arr       = np.ones((len(chi_z_to_lss))) * w0i
        wa_arr       = np.ones((len(chi_z_to_lss))) * wai
        err_chi      = 100 * np.ones((len(chi_z_to_lss)))
        bounds       = [(-1.3, -0.7), (-0.5, 0.5)]
        for cind, ichi in enumerate(chi_z_to_lss):
            func = lambda wz: GetErr_csstemu_w0wa(wz, zlists[cind], ichi, csstemu)
            min_result = minimize(func, [w0_arr[cind], wa_arr[cind]],
                                        method='L-BFGS-B', bounds=bounds)
            w0_arr[cind], wa_arr[cind] = min_result.x
            err_chi[cind] = min_result.fun
        if return_err:
            return w0_arr, wa_arr, err_chi
        else:
            return w0_arr, wa_arr

def SpectralEquivalence_As(csstemu, zlists, w0_arr, wa_arr=None, sigma8_type='CLASS'):
    '''
    Get the equivalent As for the auxiliary wCDM or w0waCDM model for a given CPL model
    csstemu: the CEmulator object
    zlists: redshift list
    w0_arr: equivalent w0 array
    wa_arr: equivalent wa array, if None, assume wa=0
    sigma8_type: 'CLASS' or 'CAMB', which code to use to calculate sigma8(z)
    '''
    csstemu_r = deepcopy(csstemu)
    Omegab = csstemu_r.Cosmo.Omegab
    Omegac = csstemu_r.Cosmo.Omegac
    H0     = csstemu_r.Cosmo.h0 * 100
    ns     = csstemu_r.Cosmo.ns
    mnu    = csstemu_r.Cosmo.mnu
    Aseq_list = np.zeros_like(zlists)
    assert len(w0_arr) == len(zlists), 'Length of w0_arr must be the same as zlists.'
    if wa_arr is None:
        wa_arr = np.zeros_like(w0_arr)    
    else:
        assert len(wa_arr) == len(zlists), 'Length of wa_arr must be the same as zlists.'
    if sigma8_type == 'CLASS':
        cosmo_class = csstemu_r.get_cosmo_class(z=zlists)
        sigma8_target = np.array([csstemu_r.get_sigma_cb_z(z=z, R=8, type='CLASS', cosmo_class=cosmo_class) for z in zlists])
        As = csstemu_r.Cosmo.As
    elif sigma8_type == 'CAMB':
        camb_results = csstemu_r.get_camb_results(z=zlists)
        sigma8_target = np.array([csstemu_r.get_sigma_cb_z(z=z, R=8, type='CAMB', camb_results=camb_results) for z in zlists])
        As = csstemu_r.Cosmo.As 
    for iz in range(len(zlists)):
        csstemu_r.set_cosmos(Omegab=Omegab, Omegac=Omegac, H0=H0, As=As, ns=ns, 
                             w=w0_arr[iz], wa=wa_arr[iz], mnu=mnu) #, checkbound=False
        Asfactor = (sigma8_target[iz]/csstemu_r.get_sigma_cb_z(z=zlists[iz], R=8))**2 
        Aseq_list[iz] = As * Asfactor
    return Aseq_list
            
        
