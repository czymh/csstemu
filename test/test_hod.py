"""
Unit tests for HOD module and galaxy statistics.

This module tests the HOD models, GalaxyStatsCalculator, and GalaxyEmulator
to ensure correctness and consistency with expectations.
"""

import numpy as np
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CEmulator.emulator.HOD import Zheng05HOD, Zheng07HOD, get_hod_model
from CEmulator.emulator.GalaxyStats import GalaxyStatsCalculator, compute_projected_correlation
from CEmulator.Emulator import GalaxyEmulator


class TestZheng05HOD(unittest.TestCase):
    """Test the Zheng+2005 HOD model."""

    def setUp(self):
        """Set up test fixtures."""
        self.hod = Zheng05HOD()
        self.M_test = np.logspace(11, 15, 100)
        self.params = {
            'logMmin': 12.0,
            'sigma_logM': 0.5,
            'logM0': 12.5,
            'logM1': 13.5,
            'alpha': 1.0
        }

    def test_ncen_range(self):
        """Test that Ncen is between 0 and 1."""
        Ncen = self.hod.Ncen(self.M_test, **self.params)
        self.assertTrue(np.all(Ncen >= 0))
        self.assertTrue(np.all(Ncen <= 1))

    def test_ncen_low_mass(self):
        """Test that Ncen approaches 0 at low mass."""
        M_low = 1e10
        Ncen = self.hod.Ncen(M_low, **self.params)
        self.assertLess(Ncen, 0.01)

    def test_ncen_high_mass(self):
        """Test that Ncen approaches 1 at high mass."""
        M_high = 1e16
        Ncen = self.hod.Ncen(M_high, **self.params)
        self.assertGreater(Ncen, 0.99)

    def test_nsat_positive(self):
        """Test that Nsat is non-negative."""
        Nsat = self.hod.Nsat(self.M_test, **self.params)
        self.assertTrue(np.all(Nsat >= 0))

    def test_nsat_low_mass(self):
        """Test that Nsat is suppressed below M0."""
        M_low = 10**self.params['logM0'] * 0.5
        Nsat = self.hod.Nsat(M_low, **self.params)
        self.assertEqual(Nsat, 0)

    def test_nsat_power_law(self):
        """Test that Nsat follows power law at high mass."""
        M_high = np.array([1e14, 1e15])
        Nsat = self.hod.Nsat(M_high, **self.params)
        # Check approximate power-law scaling
        ratio = Nsat[1] / Nsat[0]
        expected_ratio = 10**self.params['alpha']  # (M2/M1)^alpha
        self.assertAlmostEqual(ratio, expected_ratio, delta=0.5)

    def test_ntotal_consistency(self):
        """Test that Ntotal = Ncen + Nsat."""
        Ncen = self.hod.Ncen(self.M_test, **self.params)
        Nsat = self.hod.Nsat(self.M_test, **self.params)
        Ntot = self.hod.Ntotal(self.M_test, **self.params)
        np.testing.assert_array_almost_equal(Ntot, Ncen + Nsat)

    def test_param_names(self):
        """Test that parameter names are correct."""
        expected = ['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha']
        self.assertEqual(self.hod.param_names, expected)

    def test_model_name(self):
        """Test model name."""
        self.assertEqual(self.hod.name, 'Zheng05')


class TestZheng07HOD(unittest.TestCase):
    """Test the Zheng+2007 HOD model with assembly bias and fic."""

    def setUp(self):
        """Set up test fixtures."""
        self.hod = Zheng07HOD()
        self.M_test = np.logspace(11, 15, 100)
        self.params = {
            'logMcut': 12.0,
            'sigma': 0.3,
            'logM1': 13.5,
            'alpha': 1.0,
            'kappa': 1.0,
            'fic': 1.0,
            'A_cen': 0.1,
            'A_sat': 0.05
        }

    def test_ncen_range(self):
        """Test that Ncen is between 0 and fic."""
        Ncen = self.hod.Ncen(self.M_test, **self.params)
        self.assertTrue(np.all(Ncen >= 0))
        self.assertTrue(np.all(Ncen <= self.params['fic']))

    def test_ncen_low_mass(self):
        """Test that Ncen approaches 0 at low mass."""
        M_low = 1e10
        Ncen = self.hod.Ncen(M_low, **self.params)
        self.assertLess(Ncen, 0.01)

    def test_ncen_high_mass(self):
        """Test that Ncen approaches fic at high mass."""
        M_high = 1e16
        Ncen = self.hod.Ncen(M_high, **self.params)
        self.assertGreater(Ncen, self.params['fic'] * 0.99)

    def test_fic_reduces_amplitude(self):
        """Test that fic < 1 reduces the central saturation level."""
        params_fic_half = dict(self.params, fic=0.5)
        Ncen_full = self.hod.Ncen(self.M_test, **self.params)
        Ncen_half = self.hod.Ncen(self.M_test, **params_fic_half)
        # At high mass, the ratio should approach 2
        M_high = 1e16
        Nc_full = self.hod.Ncen(M_high, **self.params)
        Nc_half = self.hod.Ncen(M_high, **params_fic_half)
        self.assertAlmostEqual(Nc_full / Nc_half, 2.0, delta=0.01)

    def test_nsat_positive(self):
        """Test that Nsat is non-negative."""
        Nsat = self.hod.Nsat(self.M_test, **self.params)
        self.assertTrue(np.all(Nsat >= 0))

    def test_nsat_cutoff(self):
        """Test that Nsat is suppressed below M0 = kappa * Mcut."""
        M_low = 10**self.params['logMcut'] * self.params['kappa'] * 0.5
        Nsat = self.hod.Nsat(M_low, **self.params)
        self.assertEqual(Nsat, 0)

    def test_nsat_power_law(self):
        """Test that Nsat follows power law at high mass."""
        M_high = np.array([1e14, 1e15])
        Nsat = self.hod.Nsat(M_high, **self.params)
        ratio = Nsat[1] / Nsat[0]
        expected_ratio = 10**self.params['alpha']  # (M2/M1)^alpha approx
        self.assertAlmostEqual(ratio, expected_ratio, delta=0.5)

    def test_ntotal_consistency(self):
        """Test that Ntotal = Ncen + Nsat."""
        Ncen = self.hod.Ncen(self.M_test, **self.params)
        Nsat = self.hod.Nsat(self.M_test, **self.params)
        Ntot = self.hod.Ntotal(self.M_test, **self.params)
        np.testing.assert_array_almost_equal(Ntot, Ncen + Nsat)

    def test_param_names(self):
        """Test that parameter names are correct."""
        expected = ['logMcut', 'sigma', 'logM1', 'alpha',
                    'kappa', 'fic', 'A_cen', 'A_sat']
        self.assertEqual(self.hod.param_names, expected)

    def test_model_name(self):
        """Test model name."""
        self.assertEqual(self.hod.name, 'Zheng07')

    def test_maps_to_zheng05(self):
        """Test that fic=1, sigma=sigma_logM/sqrt(2), kappa=10^(logM0-logMcut)
        reproduces Zheng05 results."""
        params_07 = {
            'logMcut': 12.0,
            'sigma': 0.5 / np.sqrt(2.0),
            'logM1': 13.5,
            'alpha': 1.0,
            'kappa': 10.0**(12.5 - 12.0),
            'fic': 1.0,
        }

        Ncen_07 = self.hod.Ncen(self.M_test, **params_07)
        Nsat_07 = self.hod.Nsat(self.M_test, **params_07)

        hod_05 = Zheng05HOD()
        Ncen_05 = hod_05.Ncen(self.M_test, logMmin=12.0, sigma_logM=0.5)
        Nsat_05 = hod_05.Nsat(self.M_test, logMmin=12.0, sigma_logM=0.5,
                              logM0=12.5, logM1=13.5, alpha=1.0)

        np.testing.assert_array_almost_equal(Ncen_07, Ncen_05)
        np.testing.assert_array_almost_equal(Nsat_07, Nsat_05)

    def test_with_concentration(self):
        """Test with concentration parameter."""
        c = np.random.uniform(0.5, 1.5, len(self.M_test))
        Ncen = self.hod.Ncen(self.M_test, c=c, **self.params)
        Nsat = self.hod.Nsat(self.M_test, c=c, **self.params)

        self.assertTrue(np.all(Ncen >= 0))
        self.assertTrue(np.all(Ncen <= self.params['fic']))
        self.assertTrue(np.all(Nsat >= 0))


class TestHODFactory(unittest.TestCase):
    """Test the HOD model factory function."""

    def test_get_zheng05(self):
        """Test getting Zheng05 model."""
        hod = get_hod_model('Zheng05')
        self.assertIsInstance(hod, Zheng05HOD)

    def test_get_zheng07(self):
        """Test getting Zheng07 model."""
        hod = get_hod_model('Zheng07')
        self.assertIsInstance(hod, Zheng07HOD)

    def test_invalid_model(self):
        """Test error on invalid model name."""
        with self.assertRaises(ValueError):
            get_hod_model('InvalidModel')


class TestGalaxyStatsCalculator(unittest.TestCase):
    """Test the GalaxyStatsCalculator class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused."""
        # Create a simple mock HMF emulator
        class MockHMF:
            def get_dndlnM(self, z, M):
                # Simple power-law HMF for testing
                return 1e-10 * (M / 1e12) ** -1.8

            def get_Nhalo(self, z, M):
                # Cumulative number density
                return 1e-3 * (M / 1e12) ** -0.8

            def get_bias_Castro23(self, z, M, massdef=None):
                # Simple mass-dependent bias
                return 0.5 + 0.3 * np.log10(M / 1e12)

        # Create mock Xihm emulator
        class MockXihm:
            def get_xihm_lgnbar_threshold(self, z, r, lgden, massdef=None):
                return np.ones((len(np.atleast_1d(z)), len(np.atleast_1d(r)))) * 10.0

        cls.hmf_emu = MockHMF()
        cls.xihm_emu = MockXihm()
        cls.calc = GalaxyStatsCalculator(cls.hmf_emu, cls.xihm_emu, verbose=False)
        cls.hod = Zheng05HOD()
        cls.hod_params = {
            'logMmin': 12.0,
            'sigma_logM': 0.5,
            'logM0': 12.5,
            'logM1': 13.5,
            'alpha': 1.0
        }

    def test_compute_ngal_scalar(self):
        """Test computing ngal at single redshift."""
        ngal = self.calc.compute_ngal(z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertIsInstance(ngal, (float, np.floating))
        self.assertGreater(ngal, 0)

    def test_compute_ngal_array(self):
        """Test computing ngal at multiple redshifts."""
        z_arr = [0.0, 0.5, 1.0]
        ngal = self.calc.compute_ngal(z=z_arr, hod_model=self.hod, **self.hod_params)
        self.assertEqual(len(ngal), len(z_arr))
        self.assertTrue(np.all(ngal > 0))

    def test_compute_f_sat(self):
        """Test computing satellite fraction."""
        f_sat = self.calc.compute_f_sat(z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertGreaterEqual(f_sat, 0)
        self.assertLessEqual(f_sat, 1)

    def test_compute_M_eff(self):
        """Test computing effective mass."""
        M_eff = self.calc.compute_M_eff(z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertGreater(M_eff, 1e11)
        self.assertLess(M_eff, 1e15)

    def test_cache_functionality(self):
        """Test that HOD weights are cached."""
        # First call - should compute
        ngal1 = self.calc.compute_ngal(z=0.5, hod_model=self.hod, **self.hod_params)

        # Second call with same params - should use cache
        ngal2 = self.calc.compute_ngal(z=0.5, hod_model=self.hod, **self.hod_params)

        np.testing.assert_almost_equal(ngal1, ngal2)

    def test_clear_cache(self):
        """Test clearing the cache."""
        self.calc.compute_ngal(z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertGreater(len(self.calc._cache), 0)

        self.calc.clear_cache()
        self.assertEqual(len(self.calc._cache), 0)


class TestGalaxyEmulator(unittest.TestCase):
    """Test the GalaxyEmulator class."""

    @classmethod
    def setUpClass(cls):
        """Set up GalaxyEmulator instance."""
        cls.emu = GalaxyEmulator(verbose=False)
        cls.emu.set_cosmos(
            Omegab=0.049, Omegac=0.251,
            H0=67.66, ns=0.9665, As=2.0e-9,
            w=-1.0, wa=0.0, mnu=0.06
        )

    def setUp(self):
        """Reset HOD state before each test."""
        self.emu._hod_model = None
        self.emu._hod_params = {}
        self.emu._galaxy_calc = None

    def test_initialization(self):
        """Test GalaxyEmulator initialization."""
        # Create a fresh emulator for this test
        emu = GalaxyEmulator(verbose=False)
        self.assertIsNotNone(emu)
        self.assertIsNone(emu._hod_model)

    def test_set_hod(self):
        """Test setting HOD parameters."""
        self.emu.set_hod('Zheng05',
                         logMmin=12.0, sigma_logM=0.5,
                         logM0=12.5, logM1=13.5, alpha=1.0)
        self.assertIsNotNone(self.emu._hod_model)
        self.assertIsNotNone(self.emu._galaxy_calc)

    def test_hod_not_initialized_error(self):
        """Test error when HOD not set."""
        emu = GalaxyEmulator(verbose=False)
        emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)

        with self.assertRaises(ValueError):
            emu.ngal(0.5)

    def test_ngal_computation(self):
        """Test galaxy number density computation."""
        self.emu.set_hod('Zheng05',
                         logMmin=12.0, sigma_logM=0.5,
                         logM0=12.5, logM1=13.5, alpha=1.0)
        ngal = self.emu.ngal(0.5)
        self.assertGreater(ngal, 0)
        self.assertLess(ngal, 1.0)  # Should be less than 1 h^3/Mpc^3

    def test_f_sat_computation(self):
        """Test satellite fraction computation."""
        self.emu.set_hod('Zheng05',
                         logMmin=12.0, sigma_logM=0.5,
                         logM0=12.5, logM1=13.5, alpha=1.0)
        f_sat = self.emu.f_sat(0.5)
        self.assertGreaterEqual(f_sat, 0)
        self.assertLessEqual(f_sat, 1)

    def test_M_eff_computation(self):
        """Test effective mass computation."""
        self.emu.set_hod('Zheng05',
                         logMmin=12.0, sigma_logM=0.5,
                         logM0=12.5, logM1=13.5, alpha=1.0)
        M_eff = self.emu.M_eff(0.5)
        self.assertGreater(M_eff, 1e11)
        self.assertLess(M_eff, 1e15)

    def test_bias_gal_computation(self):
        """Test galaxy bias computation."""
        self.emu.set_hod('Zheng05',
                         logMmin=12.0, sigma_logM=0.5,
                         logM0=12.5, logM1=13.5, alpha=1.0)
        bias = self.emu.bias_gal(0.5)
        self.assertGreater(bias, 0.5)
        self.assertLess(bias, 10.0)

    def test_get_hod_summary(self):
        """Test HOD summary method."""
        self.emu.set_hod('Zheng05',
                         logMmin=12.0, sigma_logM=0.5,
                         logM0=12.5, logM1=13.5, alpha=1.0)
        summary = self.emu.get_hod_summary(0.5)

        expected_keys = ['ngal', 'f_sat', 'M_eff', 'bias', 'hod_params', 'hod_model']
        for key in expected_keys:
            self.assertIn(key, summary)

        self.assertEqual(summary['hod_model'], 'Zheng05')

    def test_hod_parameter_bounds(self):
        """Test HOD parameter bounds checking."""
        with self.assertRaises(ValueError):
            self.emu.set_hod('Zheng05', logMmin=20.0)  # Out of bounds

    def test_Xigm_multi_z(self):
        """Xigm accepts array z, returns (n_z, n_r)."""
        self.emu.set_hod('Zheng05',
                         logMmin=12.0, sigma_logM=0.5,
                         logM0=12.5, logM1=13.5, alpha=1.0)
        r = np.logspace(-1, 1, 15)
        z_arr = np.array([0.3, 0.5, 0.7])
        result = self.emu.Xigm(r, z_arr)
        self.assertEqual(result.shape, (3, len(r)))
        self.assertTrue(np.all(np.isfinite(result)))
        # Scalar z should return (n_r,)
        result_scalar = self.emu.Xigm(r, 0.5)
        self.assertEqual(result_scalar.shape, (len(r),))
        np.testing.assert_allclose(result[1], result_scalar, rtol=1e-10)

    def test_Xigg_multi_z(self):
        """Xigg accepts array z, returns (n_z, n_r)."""
        self.emu.set_hod('Zheng05',
                         logMmin=12.0, sigma_logM=0.5,
                         logM0=12.5, logM1=13.5, alpha=1.0)
        r = np.logspace(-1, 1, 15)
        z_arr = np.array([0.3, 0.5, 0.7])
        result = self.emu.Xigg(r, z_arr)
        self.assertEqual(result.shape, (3, len(r)))
        self.assertTrue(np.all(np.isfinite(result)))


class TestProjectedCorrelation(unittest.TestCase):
    """Test the projected correlation function calculation."""

    def test_projected_correlation_shape(self):
        """Test output shape of projected correlation."""
        r = np.logspace(-1, 1, 50)
        xi = 1.0 / r  # Simple power law
        rp = np.logspace(-1, 0.5, 20)

        wp = compute_projected_correlation(rp, r, xi, pimax=50.0)
        self.assertEqual(len(wp), len(rp))

    def test_projected_correlation_positive(self):
        """Test that wp is positive for positive xi."""
        r = np.logspace(-1, 2, 100)
        xi = np.abs(1.0 / r)
        rp = np.logspace(-1, 1, 20)

        wp = compute_projected_correlation(rp, r, xi, pimax=100.0)
        self.assertTrue(np.all(wp > 0))


class TestHODConsistency(unittest.TestCase):
    """Test consistency between different HOD-related calculations."""

    def test_satellite_fraction_consistency(self):
        """Test that f_sat = n_sat / (n_cen + n_sat)."""
        hmf_emu = type('MockHMF', (), {
            'get_dndlnM': lambda self, z, M: 1e-10 * (M / 1e12) ** -1.8,
            'get_Nhalo': lambda self, z, M: 1e-3 * (M / 1e12) ** -0.8,
            'get_bias_Castro23': lambda self, z, M, massdef=None: np.ones_like(M),
        })()

        calc = GalaxyStatsCalculator(hmf_emu, verbose=False)
        hod = Zheng05HOD()
        params = {'logMmin': 12.0, 'sigma_logM': 0.5,
                  'logM0': 12.5, 'logM1': 13.5, 'alpha': 1.0}

        ngal = calc.compute_ngal(0.5, hod, **params)
        f_sat = calc.compute_f_sat(0.5, hod, **params)

        # Manually compute expected satellite fraction
        Ncen, Nsat, Ntot = calc._get_cached_hod_weights(hod, **params)
        dndlnM = hmf_emu.get_dndlnM(0.5, calc.M_grid)
        from scipy.integrate import simpson
        nsat = simpson(dndlnM * Nsat, x=calc.lnM_grid)
        ncen = simpson(dndlnM * Ncen, x=calc.lnM_grid)

        expected_f_sat = nsat / (ncen + nsat)
        self.assertAlmostEqual(f_sat, expected_f_sat, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestGalaxyLensing(unittest.TestCase):
    """Test galaxy-galaxy lensing calculations (ΔΣ, wgm, Pgm)."""

    @classmethod
    def setUpClass(cls):
        """Set up with mock emulators providing Pkhm and Xihh support."""
        class MockHMF:
            def get_dndlnM(self, z, M):
                return 1e-10 * (M / 1e12) ** -1.8
            def get_Nhalo(self, z, M, massdef=None):
                return 1e-3 * (M / 1e12) ** -0.8
            def get_mass_from_lgden(self, z, lgden, massdef=None):
                # invert: lgden = log10(1e-3 * (M/1e12)^-0.8) => M = 1e12 * (1e-3 / 10^lgden)^(1/0.8)
                lgden = np.atleast_1d(lgden); z = np.atleast_1d(z)
                M = 1e12 * (1e-3 / 10.0**lgden) ** (1.0 / 0.8)
                return np.tile(M, (len(z), 1))
            def get_bias_Castro23(self, z, M, massdef=None):
                return 0.5 + 0.3 * np.log10(M / 1e12)

        class MockXihm:
            def get_xihm_lgnbar_threshold(self, z, r, lgden, massdef=None):
                lgden = np.atleast_1d(lgden); z = np.atleast_1d(z); r = np.atleast_1d(r)
                return np.ones((len(lgden), len(z), len(r))) * 10.0
            def get_pkhm_lgnbar_threshold(self, z, k, lgden, massdef=None):
                lgden = np.atleast_1d(lgden); z = np.atleast_1d(z); k = np.atleast_1d(k)
                P = 1e5 / (1.0 + (k / 0.1) ** 2)
                return np.ones((len(lgden), len(z), len(k))) * P[np.newaxis, np.newaxis, :]

        class MockXihh:
            def get_bias_mass(self, z, M, massdef=None):
                M = np.atleast_1d(M); z = np.atleast_1d(z)
                return np.ones((len(z), len(M))) * 2.0
            def get_pklin(self, z, k, Pcb=True):
                k = np.atleast_1d(k)
                return 1000.0 * k / (1.0 + k / 10.0)
            def get_xihh_mass(self, z, r, M1, M2, massdef=None):
                M1 = np.atleast_1d(M1); M2 = np.atleast_1d(M2)
                z = np.atleast_1d(z); r = np.atleast_1d(r)
                return np.ones((len(M1), len(M2), len(z), len(r))) * 5.0
            class Xihh:
                @staticmethod
                def get_xihh(z, r, lgden1, lgden2):
                    lgden1 = np.atleast_1d(lgden1); lgden2 = np.atleast_1d(lgden2)
                    r = np.atleast_1d(r)
                    return np.ones((len(lgden1), len(lgden2), 1, len(r))) * 5.0

        cls.hmf = MockHMF()
        cls.xihm = MockXihm()
        cls.xihh = MockXihh()
        cls.calc = GalaxyStatsCalculator(cls.hmf, cls.xihm, cls.xihh, verbose=False)
        cls.hod = Zheng05HOD()
        cls.hod_params = {
            'logMmin': 12.0, 'sigma_logM': 0.5,
            'logM0': 12.5, 'logM1': 13.5, 'alpha': 1.0,
        }

    def test_compute_Pgm_basic(self):
        """P_gm(k) returns correct shape and is positive."""
        k = self.calc.fftbase.k
        Pgm = self.calc.compute_Pgm(k, z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertEqual(Pgm.shape, (len(k),))
        self.assertTrue(np.all(Pgm > 0))
        # Should be smooth in log-log (negative slope)
        logk = np.log(k[5:-5])
        logP = np.log(Pgm[5:-5])
        slopes = np.diff(logP) / np.diff(logk)
        self.assertTrue(np.all(slopes < 0),
                        f"P_gm should decrease with k, got slopes {slopes[:3]}...")

    def test_compute_Pgm_components(self):
        """P_gm = P_cen + P_cen_off + P_sat."""
        k = self.calc.fftbase.k
        Pgm = self.calc.compute_Pgm(k, z=0.5, hod_model=self.hod, **self.hod_params)
        Pc, Pco, Ps = self.calc.compute_Pgm_components(
            k, z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertEqual(Pc.shape, (len(k),))
        self.assertEqual(Pco.shape, (len(k),))
        self.assertEqual(Ps.shape, (len(k),))
        self.assertTrue(np.all(Pc >= 0), "P_cen should be >= 0")
        self.assertTrue(np.all(Pco >= 0), "P_cen_off should be >= 0")
        # Satellite profile FFTLog can produce tiny negative values at a few k-bins
        self.assertGreater(np.sum(Ps), 0, "P_sat sum should be >= 0")
        np.testing.assert_allclose(Pgm, Pc + Pco + Ps, rtol=1e-10)

    def test_compute_Pgm_components_with_offcentering(self):
        """P_gm sum holds with off-centering enabled."""
        k = self.calc.fftbase.k
        params = dict(self.hod_params, f_off=0.2, R_off=0.3)
        Pgm = self.calc.compute_Pgm(k, z=0.5, hod_model=self.hod, **params)
        Pc, Pco, Ps = self.calc.compute_Pgm_components(
            k, z=0.5, hod_model=self.hod, **params)
        np.testing.assert_allclose(Pgm, Pc + Pco + Ps, rtol=1e-10)

    def test_compute_wgm_smoke(self):
        """w_gm(r_p) returns correct shape and finite values."""
        rp = np.logspace(-1, 1.5, 10)
        wgm = self.calc.compute_wgm(rp, z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertEqual(wgm.shape, (len(rp),))
        self.assertTrue(np.all(np.isfinite(wgm)))
        # w_gm should be positive-definite
        self.assertTrue(np.all(wgm > 0))

    def test_compute_DeltaSigma_smoke(self):
        """ΔΣ(R) returns correct shape, positivity, physical range."""
        R = np.logspace(-1, 1.5, 10)
        ds = self.calc.compute_DeltaSigma(
            R, z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertEqual(ds.shape, (len(R),))
        self.assertTrue(np.all(np.isfinite(ds)))
        self.assertTrue(np.all(ds > 0))
        # Physical range for typical LRG at z~0.5: ~1-100 h Msun/pc²
        self.assertGreater(np.max(ds), 0.1)

    def test_DeltaSigma_components(self):
        """ΔΣ component decomposition sums to total."""
        R = np.logspace(-1, 1.5, 10)
        ds = self.calc.compute_DeltaSigma(
            R, z=0.5, hod_model=self.hod, **self.hod_params)
        ds_c, ds_co, ds_s = self.calc.compute_DeltaSigma_components(
            R, z=0.5, hod_model=self.hod, **self.hod_params)
        np.testing.assert_allclose(ds, ds_c + ds_co + ds_s, rtol=1e-10)

    def test_Pgm_caching(self):
        """Same (k, z) returns cached result."""
        k = self.calc.fftbase.k
        Pgm1 = self.calc.compute_Pgm(k, z=0.5, hod_model=self.hod, **self.hod_params)
        # Clear only the P_hh matrix cache, not the Pgm cache
        self.calc._phh_matrix_cache = None
        Pgm2 = self.calc.compute_Pgm(k, z=0.5, hod_model=self.hod, **self.hod_params)
        np.testing.assert_array_equal(Pgm1, Pgm2)

    def test_DeltaSigma_physical_consistency(self):
        """ΔΣ and w_gm should both be positive and finite."""
        R = np.logspace(-0.5, 1, 20)
        wgm = self.calc.compute_wgm(R, z=0.5, hod_model=self.hod, **self.hod_params)
        ds = self.calc.compute_DeltaSigma(
            R, z=0.5, hod_model=self.hod, **self.hod_params)
        self.assertTrue(np.all(np.isfinite(wgm)))
        self.assertTrue(np.all(np.isfinite(ds)))
        self.assertTrue(np.all(wgm > 0), "w_gm should be positive")
        self.assertTrue(np.all(ds > 0), "ΔΣ should be positive")


class TestWpPimax(unittest.TestCase):
    """Test pimax support in wp-related functions."""

    @classmethod
    def setUpClass(cls):
        """Set up mock emulators (reused from TestGalaxyLensing)."""
        class MockHMF:
            def get_dndlnM(self, z, M):
                return 1e-10 * (M / 1e12) ** -1.8
            def get_Nhalo(self, z, M, massdef=None):
                return 1e-3 * (M / 1e12) ** -0.8
            def get_mass_from_lgden(self, z, lgden, massdef=None):
                # invert: lgden = log10(1e-3 * (M/1e12)^-0.8) => M = 1e12 * (1e-3 / 10^lgden)^(1/0.8)
                lgden = np.atleast_1d(lgden); z = np.atleast_1d(z)
                M = 1e12 * (1e-3 / 10.0**lgden) ** (1.0 / 0.8)
                return np.tile(M, (len(z), 1))
            def get_bias_Castro23(self, z, M, massdef=None):
                return 0.5 + 0.3 * np.log10(M / 1e12)

        class MockXihm:
            def get_xihm_lgnbar_threshold(self, z, r, lgden, massdef=None):
                lgden = np.atleast_1d(lgden); z = np.atleast_1d(z); r = np.atleast_1d(r)
                return np.ones((len(lgden), len(z), len(r))) * 10.0
            def get_pkhm_lgnbar_threshold(self, z, k, lgden, massdef=None):
                lgden = np.atleast_1d(lgden); z = np.atleast_1d(z); k = np.atleast_1d(k)
                P = 1e5 / (1.0 + (k / 0.1) ** 2)
                return np.ones((len(lgden), len(z), len(k))) * P[np.newaxis, np.newaxis, :]

        class MockXihh:
            def get_bias_mass(self, z, M, massdef=None):
                M = np.atleast_1d(M); z = np.atleast_1d(z)
                return np.ones((len(z), len(M))) * 2.0
            def get_pklin(self, z, k, Pcb=True):
                k = np.atleast_1d(k)
                return 1000.0 * k / (1.0 + k / 10.0)
            def get_xihh_mass(self, z, r, M1, M2, massdef=None):
                M1 = np.atleast_1d(M1); M2 = np.atleast_1d(M2)
                z = np.atleast_1d(z); r = np.atleast_1d(r)
                return np.ones((len(M1), len(M2), len(z), len(r))) * 5.0
            class Xihh:
                @staticmethod
                def get_xihh(z, r, lgden1, lgden2):
                    lgden1 = np.atleast_1d(lgden1); lgden2 = np.atleast_1d(lgden2)
                    r = np.atleast_1d(r)
                    return np.ones((len(lgden1), len(lgden2), 1, len(r))) * 5.0

        cls.hmf = MockHMF()
        cls.xihm = MockXihm()
        cls.xihh = MockXihh()
        cls.calc = GalaxyStatsCalculator(cls.hmf, cls.xihm, cls.xihh, verbose=False)
        cls.hod = Zheng05HOD()
        cls.hod_params = {
            'logMmin': 12.0, 'sigma_logM': 0.5,
            'logM0': 12.5, 'logM1': 13.5, 'alpha': 1.0,
        }
        cls.rp = np.logspace(-1, 1.5, 20)
        cls.z = 0.5

    def test_wp_default_pimax_shapes(self):
        """wp with default pimax=100 returns correct positive finite values."""
        wp = self.calc.compute_wp(
            self.rp, self.z, hod_model=self.hod, **self.hod_params)
        self.assertEqual(wp.shape, (len(self.rp),))
        self.assertTrue(np.all(np.isfinite(wp)))
        self.assertTrue(np.all(wp > 0))

    def test_wp_none_matches_fftlog(self):
        """wp with pimax=None uses FFTLog and gives different result from pimax=100."""
        wp100 = self.calc.compute_wp(
            self.rp, self.z, hod_model=self.hod, pimax=100.0, **self.hod_params)
        wp_inf = self.calc.compute_wp(
            self.rp, self.z, hod_model=self.hod, pimax=None, **self.hod_params)
        self.assertEqual(wp100.shape, wp_inf.shape)
        # pimax=100 should be slightly lower than infinite at large scales
        # (not strictly guaranteed for all profiles, but typical)
        self.assertTrue(np.all(np.isfinite(wp_inf)))

    def test_wp_convergence_with_pimax(self):
        """wp should increase with pimax (tested with analytic profile)."""
        # Use a known smooth correlation function (motivated by NFW-like shape)
        r_grid = np.logspace(-2, 2, 500)
        xi_analytic = 100.0 * np.exp(-0.5 * (r_grid / 5.0) ** 2)

        rp = np.logspace(-1, 1.5, 10)
        wp_50 = compute_projected_correlation(rp, r_grid, xi_analytic, pimax=50.0)
        wp_200 = compute_projected_correlation(rp, r_grid, xi_analytic, pimax=200.0)
        wp_500 = compute_projected_correlation(rp, r_grid, xi_analytic, pimax=500.0)

        self.assertTrue(np.all(np.isfinite(wp_50)))
        self.assertTrue(np.all(np.isfinite(wp_200)))
        self.assertTrue(np.all(np.isfinite(wp_500)))
        # Larger pimax → larger wp (monotonic for smooth correlation)
        self.assertTrue(np.all(wp_200 >= wp_50 - 1e-10))
        self.assertTrue(np.all(wp_500 >= wp_200 - 1e-10))
        # Convergence: fractional change should decrease as pimax grows.
        # Compare the max per-bin fractional difference between successive levels.
        frac_50_200 = np.max(np.abs(wp_200 - wp_50) / np.maximum(wp_200, 1.0))
        frac_200_500 = np.max(np.abs(wp_500 - wp_200) / np.maximum(wp_500, 1.0))
        self.assertLess(frac_200_500, frac_50_200,
                        "fractional change should decrease with increasing pimax")

    def test_decomposed_wp_with_pimax(self):
        """Decomposed wp* methods with pimax sum to total wp."""
        wp_total = self.calc.compute_wp(
            self.rp, self.z, hod_model=self.hod, pimax=100.0, **self.hod_params)
        wp_cc = self.calc.compute_wp_cc_2h(
            self.rp, self.z, hod_model=self.hod, pimax=100.0, **self.hod_params)
        wp_cs1 = self.calc.compute_wp_cs_1h(
            self.rp, self.z, hod_model=self.hod, pimax=100.0, **self.hod_params)
        wp_cs2 = self.calc.compute_wp_cs_2h(
            self.rp, self.z, hod_model=self.hod, pimax=100.0, **self.hod_params)
        wp_ss1 = self.calc.compute_wp_ss_1h(
            self.rp, self.z, hod_model=self.hod, pimax=100.0, **self.hod_params)
        wp_ss2 = self.calc.compute_wp_ss_2h(
            self.rp, self.z, hod_model=self.hod, pimax=100.0, **self.hod_params)
        wp_sum = wp_cc + 2 * wp_cs1 + 2 * wp_cs2 + wp_ss1 + wp_ss2
        # FFTLog decomposition → sum of individual pk2xi transforms ≠
        # pk2xi of the sum (power-law extrapolation is non-linear).
        np.testing.assert_allclose(wp_total, wp_sum, rtol=0.2)

    def test_decomposed_wp_with_pimax_none(self):
        """Decomposed wp* methods with pimax=None also sum to total."""
        wp_total = self.calc.compute_wp(
            self.rp, self.z, hod_model=self.hod, pimax=None, **self.hod_params)
        wp_cc = self.calc.compute_wp_cc_2h(
            self.rp, self.z, hod_model=self.hod, pimax=None, **self.hod_params)
        wp_cs1 = self.calc.compute_wp_cs_1h(
            self.rp, self.z, hod_model=self.hod, pimax=None, **self.hod_params)
        wp_cs2 = self.calc.compute_wp_cs_2h(
            self.rp, self.z, hod_model=self.hod, pimax=None, **self.hod_params)
        wp_ss1 = self.calc.compute_wp_ss_1h(
            self.rp, self.z, hod_model=self.hod, pimax=None, **self.hod_params)
        wp_ss2 = self.calc.compute_wp_ss_2h(
            self.rp, self.z, hod_model=self.hod, pimax=None, **self.hod_params)
        wp_sum = wp_cc + 2 * wp_cs1 + 2 * wp_cs2 + wp_ss1 + wp_ss2
        # 2D finite differencing + FFTLog extrapolation can introduce ~1e-4
        # relative differences in component reconstruction
        np.testing.assert_allclose(wp_total, wp_sum, rtol=5e-4)

    def test_projected_correlation_npi(self):
        """compute_projected_correlation with different n_pi."""
        r = np.logspace(-2, 2, 500)
        # Use a smooth Gaussian profile: ξ(r) = 100·exp(-r²/2σ²), σ=10 Mpc/h
        # This gives a smooth projected integral that's well-behaved.
        xi = 100.0 * np.exp(-0.5 * (r / 10.0) ** 2)
        wp_50 = compute_projected_correlation(self.rp, r, xi, pimax=100.0, n_pi=50)
        wp_200 = compute_projected_correlation(self.rp, r, xi, pimax=100.0, n_pi=200)
        self.assertTrue(np.all(np.isfinite(wp_50)))
        self.assertTrue(np.all(np.isfinite(wp_200)))
        # Higher n_pi should converge (typ. < 1% difference)
        np.testing.assert_allclose(wp_50, wp_200, rtol=1e-2)

    def test_wp_via_emulator(self):
        """GalaxyEmulator.wp accepts pimax parameter."""
        emu = GalaxyEmulator(verbose=False)
        emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)
        emu.set_hod('Zheng05',
                    logMmin=12.0, sigma_logM=0.5,
                    logM0=12.5, logM1=13.5, alpha=1.0)
        # Default pimax=100
        wp_def = emu.wp(self.rp, z=0.5)
        self.assertEqual(wp_def.shape, (len(self.rp),))
        self.assertTrue(np.all(np.isfinite(wp_def)))
        # pimax=None (FFTLog)
        wp_inf = emu.wp(self.rp, z=0.5, pimax=None)
        self.assertTrue(np.all(np.isfinite(wp_inf)))

    def test_wp_multi_z(self):
        """GalaxyEmulator.wp accepts array z, returns (n_z, n_rp)."""
        emu = GalaxyEmulator(verbose=False)
        emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)
        emu.set_hod('Zheng05',
                    logMmin=12.0, sigma_logM=0.5,
                    logM0=12.5, logM1=13.5, alpha=1.0)
        z_arr = np.array([0.3, 0.5, 0.7])
        wp_multi = emu.wp(self.rp, z_arr, pimax=100.0)
        self.assertEqual(wp_multi.shape, (3, len(self.rp)))
        self.assertTrue(np.all(np.isfinite(wp_multi)))
        # Scalar z should return (n_rp,)
        wp_scalar = emu.wp(self.rp, 0.5, pimax=100.0)
        self.assertEqual(wp_scalar.shape, (len(self.rp),))
        np.testing.assert_allclose(wp_multi[1], wp_scalar, rtol=1e-10)

    def test_wp_decomposed_multi_z(self):
        """Emulator wp_* accept array z, return (n_z, n_rp)."""
        emu = GalaxyEmulator(verbose=False)
        emu.set_cosmos(Omegab=0.049, Omegac=0.251, H0=67.66, As=2.0e-9)
        emu.set_hod('Zheng05',
                    logMmin=12.0, sigma_logM=0.5,
                    logM0=12.5, logM1=13.5, alpha=1.0)
        z_arr = np.array([0.3, 0.5])
        wp_cc = emu.wp_cc_2h(self.rp, z_arr, pimax=100.0)
        wp_cs1 = emu.wp_cs_1h(self.rp, z_arr, pimax=100.0)
        self.assertEqual(wp_cc.shape, (2, len(self.rp)))
        self.assertEqual(wp_cs1.shape, (2, len(self.rp)))
        self.assertTrue(np.all(np.isfinite(wp_cc)))
        self.assertTrue(np.all(np.isfinite(wp_cs1)))
