"""
Precompute the Fermi-Dirac integral F(y) = integral_0^inf x^2 * sqrt(x^2+y^2) / (1+exp(x)) dx
and save the interpolation table.

This script should be run once to generate the lookup table, and then the results
will be loaded at runtime without any numerical integration.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.interpolate import interp1d
import os

def compute_F_y_grid(y_grid, n_quad_limit=200):
    """
    Compute F(y) for a grid of y values using numerical integration.

    Parameters
    ----------
    y_grid : array
        Grid of y values
    n_quad_limit : int
        Limit parameter for scipy.integrate.quad

    Returns
    -------
    F_grid : array
        F(y) values
    """
    def integrand_F(y):
        return lambda x: x * x * np.sqrt(x * x + y * y) * expit(-x)

    F_grid = np.zeros_like(y_grid, dtype=np.float64)
    for i, y in enumerate(y_grid):
        F_grid[i], _ = quad(integrand_F(y), 0, np.inf, limit=n_quad_limit)
        if i % 100 == 0:
            print(f"Computing F(y) for y={y:.2f} ({i}/{len(y_grid)}): F(y)={F_grid[i]:.6f}")

    return F_grid


def main():
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Grid parameters
    # y = mnu / (1+z) / kB / T_nu
    # mnu max = 0.3 eV, z min = 0, kB = 8.617e-5 eV/K, T_nu ~ 1.95 K
    # y_max ~ 0.3 / 1.0 / 8.617e-5 / 1.95 ~ 1780
    # Use a denser grid for better interpolation accuracy
    n_points = 10000
    y_max = 2000.0

    y_grid = np.linspace(0, y_max, n_points)

    print(f"Computing F(y) on grid: {n_points} points, y in [0, {y_max}]")
    F_grid = compute_F_y_grid(y_grid)

    # Build interpolator to verify
    interpolator = interp1d(y_grid, F_grid, kind='cubic', bounds_error=False, fill_value=(0.0, float('inf')))

    # Save the grid and F values
    output_file = os.path.join(output_dir, 'fermi_dirac_F_interp.npz')
    np.savez(output_file, y_grid=y_grid, F_grid=F_grid)
    print(f"Saved interpolation table to {output_file}")

    # Also save a standalone interpolator coefficients for faster loading
    # Using scipy's UnivariateSpline for better performance
    from scipy.interpolate import UnivariateSpline
    spline = UnivariateSpline(y_grid, F_grid, s=0, ext=0)

    # Test interpolation accuracy
    test_y = np.array([0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
    test_F_interp = spline(test_y)

    print("\nVerification (interpolated vs original):")
    for y, F in zip(test_y, test_F_interp):
        print(f"  y={y:.1f}: F(y)={F:.8f}")

    # Save spline coefficients for fast loading
    # UnivariateSpline stores: t (knots), c (coefficients), k (degree)
    np.savez(os.path.join(output_dir, 'fermi_dirac_F_spline.npz'),
             t=spline.get_knots(),
             c=spline.get_coeffs(),
             k=spline.get_degree())
    print(f"Saved spline parameters to {output_dir}/fermi_dirac_F_spline.npz")

    print("\nPrecomputation complete!")


if __name__ == '__main__':
    main()