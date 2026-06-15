#!/usr/bin/env python
"""
Example: Using the HOD module with CSST Emulator.

This script demonstrates how to use the GalaxyEmulator to compute
galaxy clustering statistics using Halo Occupation Distribution models.
"""

import numpy as np
import matplotlib.pyplot as plt
from CEmulator.Emulator import GalaxyEmulator


def example_basic_usage():
    """Example 1: Basic HOD usage with GalaxyEmulator."""
    print("=" * 60)
    print("Example 1: Basic HOD Usage")
    print("=" * 60)

    # Initialize the GalaxyEmulator
    emu = GalaxyEmulator(verbose=True)

    # Set cosmological parameters
    emu.set_cosmos(
        Omegab=0.049,
        Omegac=0.251,
        H0=67.66,
        ns=0.9665,
        As=2.0e-9,
        w=-1.0,
        wa=0.0,
        mnu=0.06
    )

    # Set HOD parameters (Zheng+2005 model)
    print("\nSetting HOD parameters...")
    emu.set_hod(
        model_name='Zheng05',
        logMmin=12.0,      # Minimum halo mass for centrals
        sigma_logM=0.5,    # Transition width
        logM0=12.5,        # Satellite cutoff mass
        logM1=13.5,        # Satellite characteristic mass
        alpha=1.0          # Satellite power-law slope
    )

    # Compute galaxy properties at z=0.5
    z = 0.5
    print(f"\nGalaxy properties at z={z}:")
    print(f"  n_gal    = {emu.ngal(z):.4e} h^3/Mpc^3")
    print(f"  f_sat    = {emu.f_sat(z):.3f}")
    print(f"  M_eff    = {emu.M_eff(z):.3e} Msun/h")
    print(f"  bias_gal = {emu.bias_gal(z):.3f}")


def example_correlation_functions():
    """Example 2: Computing galaxy correlation functions."""
    print("\n" + "=" * 60)
    print("Example 2: Galaxy Correlation Functions")
    print("=" * 60)

    # Initialize emulator
    emu = GalaxyEmulator(verbose=False)
    emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)

    # Set HOD parameters
    emu.set_hod('Zheng05',
                logMmin=12.0, sigma_logM=0.5,
                logM0=12.5, logM1=13.5, alpha=1.0)

    # Compute galaxy-matter correlation
    r = np.logspace(-1, 1.5, 50)  # 0.1 to ~30 Mpc/h
    z = 0.5

    print(f"\nComputing galaxy-matter correlation at z={z}...")
    Xigm = emu.Xigm(r, z)

    print(f"Computing galaxy-galaxy correlation at z={z}...")
    Xigg = emu.Xigg(r, z)

    # Print some values
    print(f"\nxi_gm(r=1 Mpc/h) = {Xigm[20]:.4f}")
    print(f"xi_gg(r=1 Mpc/h) = {Xigg[20]:.4f}")

    return r, Xigm, Xigg


def example_hod_variations():
    """Example 3: Comparing different HOD parameters."""
    print("\n" + "=" * 60)
    print("Example 3: HOD Parameter Variations")
    print("=" * 60)

    emu = GalaxyEmulator(verbose=False)
    emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)

    z = 0.5

    # Compare different logMmin values
    logMmin_values = [11.5, 12.0, 12.5, 13.0]

    print(f"\nVarying logMmin at z={z}:")
    print(f"{'logMmin':>10} {'n_gal':>12} {'f_sat':>8} {'bias':>8}")
    print("-" * 45)

    for logMmin in logMmin_values:
        emu.set_hod('Zheng05',
                    logMmin=logMmin,
                    sigma_logM=0.5,
                    logM0=logMmin + 0.5,
                    logM1=logMmin + 1.5,
                    alpha=1.0)

        ngal = emu.ngal(z)
        f_sat = emu.f_sat(z)
        bias = emu.bias_gal(z)

        print(f"{logMmin:10.1f} {ngal:12.4e} {f_sat:8.3f} {bias:8.3f}")


def example_redshift_evolution():
    """Example 4: Galaxy properties as a function of redshift."""
    print("\n" + "=" * 60)
    print("Example 4: Redshift Evolution")
    print("=" * 60)

    emu = GalaxyEmulator(verbose=False)
    emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)

    emu.set_hod('Zheng05',
                logMmin=12.0, sigma_logM=0.5,
                logM0=12.5, logM1=13.5, alpha=1.0)

    z_arr = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    print("\nRedshift evolution of galaxy properties:")
    print(f"{'z':>5} {'n_gal':>12} {'f_sat':>8} {'M_eff':>12} {'bias':>8}")
    print("-" * 55)

    ngal_arr = emu.ngal(z_arr)
    f_sat_arr = emu.f_sat(z_arr)
    M_eff_arr = emu.M_eff(z_arr)
    bias_arr = emu.bias_gal(z_arr)

    for i, z in enumerate(z_arr):
        print(f"{z:5.1f} {ngal_arr[i]:12.4e} {f_sat_arr[i]:8.3f} "
              f"{M_eff_arr[i]:12.3e} {bias_arr[i]:8.3f}")


def example_hod_summary():
    """Example 5: Getting HOD summary information."""
    print("\n" + "=" * 60)
    print("Example 5: HOD Summary")
    print("=" * 60)

    emu = GalaxyEmulator(verbose=False)
    emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)

    emu.set_hod('Zheng05',
                logMmin=12.0, sigma_logM=0.5,
                logM0=12.5, logM1=13.5, alpha=1.0)

    # Get summary at z=0.5
    summary = emu.get_hod_summary(z=0.5)

    print("\nHOD Summary at z=0.5:")
    print(f"  Model: {summary['hod_model']}")
    print(f"  Parameters:")
    for param, value in summary['hod_params'].items():
        print(f"    {param:12s} = {value}")
    print(f"\n  Derived quantities:")
    print(f"    n_gal    = {summary['ngal']:.4e} h^3/Mpc^3")
    print(f"    f_sat    = {summary['f_sat']:.3f}")
    print(f"    M_eff    = {summary['M_eff']:.3e} Msun/h")
    print(f"    bias_gal = {summary['bias']:.3f}")


def example_projected_correlation():
    """Example 6: Computing projected correlation function."""
    print("\n" + "=" * 60)
    print("Example 6: Projected Correlation Function")
    print("=" * 60)

    emu = GalaxyEmulator(verbose=False)
    emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)

    emu.set_hod('Zheng05',
                logMmin=12.0, sigma_logM=0.5,
                logM0=12.5, logM1=13.5, alpha=1.0)

    z = 0.5
    rp = np.logspace(-1, 1.5, 30)  # Projected separation

    print(f"\nComputing projected correlation w_p(r_p) at z={z}...")
    print(f"Integration limit: pi_max = 100 Mpc/h")

    w_p = emu.wp(rp, z)

    print(f"\nSample values:")
    print(f"  w_p(r_p=0.1 Mpc/h) = {w_p[0]:.3f}")
    print(f"  w_p(r_p=1.0 Mpc/h) = {w_p[10]:.3f}")
    print(f"  w_p(r_p=10 Mpc/h)  = {w_p[25]:.3f}")

    return rp, w_p


def example_galaxy_lensing():
    """Example 7: Galaxy-galaxy lensing (w_gm and ΔΣ)."""
    print("\n" + "=" * 60)
    print("Example 7: Galaxy-Galaxy Lensing")
    print("=" * 60)

    emu = GalaxyEmulator(verbose=False)
    emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)

    emu.set_hod('Zheng05',
                logMmin=12.0, sigma_logM=0.5,
                logM0=12.5, logM1=13.5, alpha=1.0)

    # Radial grid: 0.3 to 30 Mpc/h
    R = np.logspace(-0.5, 1.5, 15)

    # Single redshift
    z = 0.5
    print(f"\n--- Single redshift z={z} ---")
    print("Computing galaxy-matter projected correlation w_gm(r_p)...")
    wgm = emu.wgm(R, z)
    print("Computing excess surface density ΔΣ(R)...")
    DS = emu.DeltaSigma(R, z)

    print(f"\n{'R [Mpc/h]':>12} {'w_gm':>12} {'ΔΣ [h Msun/pc²]':>18}")
    print("-" * 44)
    for i in range(len(R)):
        print(f"{R[i]:12.3f} {wgm[0, i] if wgm.ndim > 1 else wgm[i]:12.3f} "
              f"{DS[0, i] if DS.ndim > 1 else DS[i]:18.3f}")

    # Multi-redshift
    z_arr = np.array([0.2, 0.5, 1.0])
    print(f"\n--- Multi-redshift ΔΣ at R = 1 Mpc/h ---")
    DS_multi = emu.DeltaSigma(R, z_arr)
    for i, zi in enumerate(z_arr):
        r_idx = np.argmin(np.abs(R - 1.0))
        print(f"  z={zi:.1f}: ΔΣ(R=1 Mpc/h) = {DS_multi[i, r_idx]:.3f} h Msun/pc²")

    return R, wgm, DS


def example_lensing_components():
    """Example 8: Decompose ΔΣ into central / off-centered / satellite terms."""
    print("\n" + "=" * 60)
    print("Example 8: Lensing Component Decomposition")
    print("=" * 60)

    emu = GalaxyEmulator(verbose=False)
    emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)
    # f_off and R_off are set through set_hod so that total and components are consistent
    emu.set_hod('Zheng05',
                logMmin=12.0, sigma_logM=0.5,
                logM0=12.5, logM1=13.5, alpha=1.0,
                f_off=0.15, R_off=0.25)

    z = 0.5
    R = np.logspace(-0.5, 1.5, 15)

    # Total ΔΣ
    DS = emu.DeltaSigma(R, z)
    # Decomposed components (f_off/R_off must match set_hod values)
    DS_cen, DS_off, DS_sat = emu.DeltaSigma_components(R, z)

    print(f"\n{'R [Mpc/h]':>12} {'ΔΣ_total':>12} {'ΔΣ_cen':>12} "
          f"{'ΔΣ_off':>12} {'ΔΣ_sat':>12}")
    print("-" * 62)
    for i in range(len(R)):
        ds = DS[0, i] if DS.ndim > 1 else DS[i]
        dc = DS_cen[0, i] if DS_cen.ndim > 1 else DS_cen[i]
        do = DS_off[0, i] if DS_off.ndim > 1 else DS_off[i]
        dsat = DS_sat[0, i] if DS_sat.ndim > 1 else DS_sat[i]
        print(f"{R[i]:12.3f} {ds:12.3f} {dc:12.3f} {do:12.3f} {dsat:12.3f}")

    # Verify sum
    DS_sum = (DS_cen + DS_off + DS_sat)
    np.testing.assert_allclose(DS, DS_sum, rtol=1e-10)
    print("\n✓ Component sum matches total ΔΣ")

    return R, DS, (DS_cen, DS_off, DS_sat)


if __name__ == '__main__':
    # Run all examples
    example_basic_usage()
    example_correlation_functions()
    example_hod_variations()
    example_redshift_evolution()
    example_hod_summary()
    example_projected_correlation()
    example_galaxy_lensing()
    example_lensing_components()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
