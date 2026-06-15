"""
Unit tests to verify consistency between the original and optimized cosmology calculations.
"""
import numpy as np
import unittest
from scipy.integrate import quad
from scipy.special import expit
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFermiDiracConsistency(unittest.TestCase):
    """Test the consistency of Fermi-Dirac integral interpolation."""

    @classmethod
    def setUpClass(cls):
        """Compute reference values using the original quad integration."""
        cls.y_test_values = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        cls.tol = 1e-5  # Relaxed tolerance due to cubic interpolation

        # Compute reference F(y) using original method
        cls.F_reference = {}
        cls.F_prime_reference = {}

        for y in cls.y_test_values:
            # F(y) = integral_0^inf x^2 * sqrt(x^2+y^2) / (1+exp(x)) dx
            cls.F_reference[y], _ = quad(
                lambda x: x * x * np.sqrt(x * x + y * y) * expit(-x),
                0, np.inf, limit=200
            )

            # F'(y) = integral_0^inf x^2 * y / sqrt(x^2+y^2) / (1+exp(x)) dx
            cls.F_prime_reference[y], _ = quad(
                lambda x: x * x * y / np.sqrt(x * x + y * y) * expit(-x),
                0, np.inf, limit=200
            )

    def test_F_y_consistency(self):
        """Test F(y) interpolation consistency."""
        from CEmulator.cosmology import _F_INTERPOLATOR

        for y in self.y_test_values:
            with self.subTest(y=y):
                F_interp = float(_F_INTERPOLATOR(y))
                F_ref = self.F_reference[y]
                rel_error = abs(F_interp - F_ref) / abs(F_ref)
                self.assertLess(rel_error, self.tol,
                    f"F(y) at y={y}: interp={F_interp:.10f}, ref={F_ref:.10f}, "
                    f"rel_error={rel_error:.2e}")

    def test_F_prime_consistency(self):
        """Test F'(y) interpolation consistency."""
        from CEmulator.cosmology import _F_PRIME_INTERPOLATOR

        for y in self.y_test_values:
            with self.subTest(y=y):
                F_prime_interp = float(_F_PRIME_INTERPOLATOR(y))
                F_prime_ref = self.F_prime_reference[y]
                rel_error = abs(F_prime_interp - F_prime_ref) / (abs(F_prime_ref) + 1e-10)
                self.assertLess(rel_error, self.tol,
                    f"F'(y) at y={y}: interp={F_prime_interp:.10f}, ref={F_prime_ref:.10f}, "
                    f"rel_error={rel_error:.2e}")


class TestCosmologyConsistency(unittest.TestCase):
    """Test the consistency of cosmology calculations."""

    def setUp(self):
        """Set up test cosmologies."""
        # Test cases: (Omegab, Omegam, H0, ns, A, w, wa, mnu)
        self.test_params = [
            (0.049, 0.30, 67.66, 0.9665, 2.0, -1.0, 0.0, 0.0),      # No neutrino
            (0.049, 0.30, 67.66, 0.9665, 2.0, -1.0, 0.0, 0.06),     # Small neutrino mass
            (0.049, 0.30, 67.66, 0.9665, 2.0, -1.0, 0.0, 0.3),      # Max neutrino mass
            (0.045, 0.25, 70.0, 0.95, 2.2, -1.2, 0.3, 0.1),         # w0-wa CDM
            (0.055, 0.38, 65.0, 0.98, 1.8, -0.8, -0.3, 0.15),       # Different params
        ]
        self.z_test_values = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
        self.tol_Ez = 1e-8
        self.tol_comoving = 1e-6

    def _create_cosmo_reference(self, params, neutrino_mass_split='single'):
        """Create a cosmology object using original quad integration for reference."""
        from CEmulator.cosmology import Cosmology

        cosmo = Cosmology(verbose=False, neutrino_mass_split=neutrino_mass_split)
        Omegab, Omegam, H0, ns, A, w, wa, mnu = params
        cosmologies = {
            'Omegab': Omegab,
            'Omegam': Omegam,
            'H0': H0,
            'ns': ns,
            'A': A,
            'w': w,
            'wa': wa,
            'mnu': mnu,
        }
        cosmo.set_cosmos(cosmologies)
        return cosmo

    def _F_y_reference(self, y):
        """Original F(y) calculation using quad."""
        F_y_int = lambda x, y: x * x * np.sqrt(x * x + y * y) * expit(-x)
        return quad(F_y_int, 0, np.inf, args=(y))[0]

    def _compute_Omeganu_reference(self, cosmo, z):
        """Reference calculation for neutrino density contribution."""
        fac = 15/np.pi/np.pi/np.pi/np.pi * (cosmo.Gamma_nu**4) * cosmo.Omegag * (1+z)**4
        T_nu = cosmo.Gamma_nu * cosmo.TCMB
        if cosmo.mnu != 0.0:
            F_sumNu = self._F_y_reference(cosmo.mnu / (1+z) / cosmo.kB / T_nu)
        else:
            F_sumNu = 0
        return fac * F_sumNu

    def _get_Ez_reference(self, cosmo, z):
        """Reference calculation for E(z)."""
        return np.sqrt(
            cosmo.Omegam * (1+z)**3 +
            self._compute_Omeganu_reference(cosmo, z) +
            cosmo.Omegak * (1+z)**2 +
            cosmo.OmegaR * (1+z)**4 +
            cosmo.OmegaL * np.exp(3*((1/(1+z)-1)*cosmo.wa - (1 + cosmo.w0 + cosmo.wa)*np.log(1/(1+z))))
        )

    def _comoving_distance_reference(self, cosmo, z):
        """Reference calculation for comoving distance using quad."""
        def chi_int(a):
            return 1 / self._get_Ez_reference(cosmo, 1/a - 1) / a / a

        z = np.atleast_1d(z)
        aarr = 1 / (1 + z)
        fac = 1e-5 * cosmo.vel_light / cosmo.h0 / 100
        out = np.array([quad(chi_int, ia, 1)[0] for ia in aarr]) * fac
        return out

    def test_get_Ez_consistency(self):
        """Test E(z) consistency for various cosmologies and redshifts."""
        for params in self.test_params:
            with self.subTest(params=params):
                cosmo = self._create_cosmo_reference(params)

                for z in self.z_test_values:
                    Ez_new = float(cosmo.get_Ez(z))
                    Ez_ref = float(self._get_Ez_reference(cosmo, z))
                    rel_error = abs(Ez_new - Ez_ref) / (Ez_ref + 1e-10)
                    self.assertLess(rel_error, self.tol_Ez,
                        f"get_Ez at z={z}: new={Ez_new:.10f}, ref={Ez_ref:.10f}, "
                        f"rel_error={rel_error:.2e}")

    def test_comoving_distance_consistency(self):
        """Test comoving distance consistency for various cosmologies and redshifts."""
        for params in self.test_params:
            with self.subTest(params=params):
                cosmo = self._create_cosmo_reference(params)

                for z in self.z_test_values:
                    chi_new = float(cosmo.comoving_distance(z))
                    chi_ref = float(self._comoving_distance_reference(cosmo, z))
                    rel_error = abs(chi_new - chi_ref) / (abs(chi_ref) + 1e-10)
                    self.assertLess(rel_error, self.tol_comoving,
                        f"comoving_distance at z={z}: new={chi_new:.6f}, ref={chi_ref:.6f}, "
                        f"rel_error={rel_error:.2e}")


class TestPerformanceImprovement(unittest.TestCase):
    """Test that the optimization provides significant speedup."""

    def test_F_y_performance(self):
        """Test that F(y) interpolation is much faster than quad integration."""
        import time
        from CEmulator.cosmology import _F_INTERPOLATOR

        n_iterations = 1000
        y_test = 50.0

        # Time interpolation method
        start = time.time()
        for _ in range(n_iterations):
            _ = float(_F_INTERPOLATOR(y_test))
        interp_time = time.time() - start

        # Time original quad method
        start = time.time()
        for _ in range(n_iterations):
            F_y_int = lambda x, y: x * x * np.sqrt(x * x + y * y) * expit(-x)
            _ = quad(F_y_int, 0, np.inf, args=(y_test))[0]
        quad_time = time.time() - start

        speedup = quad_time / interp_time
        print(f"\nF(y) performance: interp={interp_time*1000:.2f}ms, quad={quad_time*1000:.2f}ms, speedup={speedup:.1f}x")

        # Interpolation should be at least 10x faster
        self.assertGreater(speedup, 10,
            f"Expected at least 10x speedup, got {speedup:.1f}x")


class TestEmulatorIntegration(unittest.TestCase):
    """Test that the cosmology changes work correctly with the Emulator."""

    def test_emulator_creation(self):
        """Test that Emulator can be created with new Cosmology."""
        from CEmulator.Emulator import CBaseEmulator

        emulator = CBaseEmulator(verbose=False)
        self.assertIsNotNone(emulator)
        self.assertIsNotNone(emulator.Cosmo)

    def test_emulator_set_cosmos(self):
        """Test that Emulator can set cosmologies and compute distances."""
        from CEmulator.Emulator import CBaseEmulator

        emulator = CBaseEmulator(verbose=False)
        emulator.set_cosmos(
            Omegab=0.049, Omegac=0.25,
            H0=67.66, ns=0.9665, As=2.0e-9,
            w=-1.0, wa=0.0, mnu=0.06
        )

        # Test comoving distance
        z = np.array([0.1, 0.5, 1.0, 2.0])
        chi = emulator.Cosmo.comoving_distance(z)

        self.assertEqual(len(chi), len(z))
        self.assertTrue(np.all(chi > 0))  # Comoving distance should be positive
        self.assertTrue(np.all(np.diff(chi) > 0))  # Should increase with z


if __name__ == '__main__':
    unittest.main(verbosity=2)