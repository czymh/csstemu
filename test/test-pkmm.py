%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import sys
sys.path.append(path + './csstemu/CEmulator')
from CEmulator.Emulator import CEmulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from scipy.signal import savgol_filter
import scienceplots
from matplotlib.colors import Normalize

def SmoothBk(Bk, window_length=10, polyorder=3):
    fnew = np.ones_like(Bk)
    for ic in range(Bk.shape[0]):
        for iz in range(Bk.shape[1]):
            fnew[ic,iz,:] = savgol_filter(Bk[ic,iz,:], 
                                          window_length=window_length, 
                                          polyorder=polyorder)
    return fnew

zlists = np.array([3.00, 2.50, 2.00, 1.75, 
                   1.50, 1.25, 1.00, 0.80, 
                   0.50, 0.25, 0.10, 0.00])
kcut = 10
k_list = np.load('data/karr_nb_Nmesh3072.npy')
ind = k_list<=kcut

k_list = k_list[ind]
cosmoall       = np.load('./csstemu/CEmulator/data/cosmologies_8d_train_n129_Sobol.npy')
para_validate  = cosmoall[65:97,:]
Bk_validate    = np.load('data/bk_simu_n44from65_nb_Nmesh3072.npy')[:32]
pklin_validate = np.load('data/pk_lin_n44from65_nb_Nmesh3072.npy')[:32]
Bk_validate    = Bk_validate[:,::-1,:]
pklin_validate = pklin_validate[:,::-1,:]
Bk_validate    = SmoothBk(Bk_validate)
pknln_validate = pklin_validate * Bk_validate
param_names  = ['Omegab', 'Omegam', 'H0', 'ns', 'A', 'w', 'wa', 'mnu']
pkemu = CEmulator(statistic='Pkmm')
icind = np.arange(65, 97)
pkemu.set_cosmos(Omegab=cosmoall[icind,0], Omegam=cosmoall[icind,1], H0=cosmoall[icind,2],
                 ns=cosmoall[icind,3], As=cosmoall[icind,4]*1e-9, w=cosmoall[icind,5], 
                 wa=cosmoall[icind,6], mnu=cosmoall[icind,7])
Bk_predited = pkemu.get_Bk(k=k_list)
print(Bk_predited.shape)
# print(Bk_predited.min())
std_predicted = np.std(Bk_predited/Bk_validate[:,:,ind], axis=0)
print(std_predicted.shape)

norm = Normalize(vmin=zlists.min(), vmax=zlists.max())
cmap = plt.cm.coolwarm
sm = plt.cm.ScalarMappable(cmap=cmap, 
                           norm=norm)
sm.set_array([])
with plt.style.context(['science', 'no-latex']):
    ax0 = plt.subplot(111)
    for indz, iz in enumerate(zlists[::-1]):
        plt.plot(k_list, std_predicted[indz], 
                #  label="z={:.2f}".format(iz),
                 color=cmap(norm(iz)),
                 )
    plt.xscale('log')
    # plt.legend(ncol=3, bbox_to_anchor=(-0.15, 1.08), loc="lower left")
    # plt.title(r'Leave-One-Out Error')
    plt.grid('on')
    plt.xlabel(r'$k [h/Mpc]$')
    plt.ylabel(r'$\sigma(P_\mathrm{pred}/P_\mathrm{true})$')
    plt.ylim([-0.001, 0.06])
    plt.colorbar(sm, ax=ax0,
                 pad=0.02, label=r'$z$')
    
#### compare with HMcode   
pkemu = CEmulator(statistic='Pkmm')
icind = np.arange(65, 66)
pkemu.set_cosmos(Omegab=cosmoall[icind,0], Omegam=cosmoall[icind,1], H0=cosmoall[icind,2],
                 ns=cosmoall[icind,3], As=cosmoall[icind,4]*1e-9, w=cosmoall[icind,5], 
                 wa=cosmoall[icind,6], mnu=cosmoall[icind,7])
pklin = pkemu.get_pklin(k=k_list)
pknl  = pkemu.get_pknl(k=k_list)

cosmo_class = pkemu.get_cosmos_class('0.0', non_linear='HMcode')[0]
h0 = cosmo_class.h()
pkhm = [cosmo_class.pk_cb(ik*h0, 0.0)*h0*h0*h0 for ik in k_list]
pksimu = pknln_validate[0,0,ind]

with plt.style.context(['seaborn']):
    gridp = plt.GridSpec(2, 1, hspace=0.05)
    ax0 = plt.subplot(gridp[:1,0])
    plt.title('c0065')
    plt.loglog(k_list, pklin[0,0,:], 'b', label='Linear')
    plt.loglog(k_list, pknl[0,0,:], 'g', label='NL-CEmulator')
    plt.loglog(k_list, pkhm, 'c--', label='NL-HMcode')
    plt.loglog(k_list, pksimu, 'k:', label='simu')
    plt.legend()
    plt.ylabel(r'$P(k)$')
    ax0.set_xticklabels([])
    plt.ylim([0.1, 2e5])
    ax1 = plt.subplot(gridp[-1,0])
    plt.semilogx(k_list, pknl[0,0,:]/pksimu, 'g', label='NL-CEmulator') 
    plt.semilogx(k_list, pkhm/pksimu, 'c--', label='NL-HMcode')
    plt.xlabel(r'$k\ [{\rm h\ Mpc^{-1}}]$')
    plt.ylabel(r'$P_{\rm X}/P_{\rm simu}$')
    plt.ylim([0.97, 1.07])
    plt.legend()
    # plt.savefig('./pic/c0065_pk.png', dpi=300, bbox_inches='tight')
