import unittest

import numpy as np

from mapext.emission import continuum as mapext_cont

phys_const = {
    "c": 299792458.0,
    "k": 1.3806488e-23,
    "h": 6.62606957e-34,
}


class TestSynchrotronEmissionModel(unittest.TestCase):

    def setUp(self):
        # Create an instance of the synchrotron_1comp model
        self.model = mapext_cont.synchrotron_1comp()

    def test_evaluate(self):
        # Test evaluate with scalar inputs
        nu = 2.0  # GHz
        area = 1e-4  # steradians
        synch_S1 = 0.01
        synch_alp = -0.7

        result = self.model.evaluate(nu, area, synch_S1, synch_alp)
        expected_result = synch_S1 * (nu**synch_alp)
        self.assertAlmostEqual(result, expected_result)

    def test_fit_deriv_with_numerical_approximation(self):
        # Test that fit_deriv is numerically consistent with evaluate

        nu = 2.0  # GHz
        area = 1e-4  # steradians
        synch_S1 = 0.01
        synch_alp = -0.7
        delta = 1e-5  # Perturbation to numerically approximate the derivative

        # Numerical approximation of the derivative w.r.t. synch_S1
        evaluate_at_synch_S1_plus_delta = self.model.evaluate(
            nu, area, synch_S1 + delta, synch_alp
        )
        evaluate_at_synch_S1 = self.model.evaluate(nu, area, synch_S1, synch_alp)
        numerical_deriv_S1 = (
            evaluate_at_synch_S1_plus_delta - evaluate_at_synch_S1
        ) / delta

        # Numerical approximation of the derivative w.r.t. synch_alp
        evaluate_at_synch_alp_plus_delta = self.model.evaluate(
            nu, area, synch_S1, synch_alp + delta
        )
        evaluate_at_synch_alp = self.model.evaluate(nu, area, synch_S1, synch_alp)
        numerical_deriv_alp = (
            evaluate_at_synch_alp_plus_delta - evaluate_at_synch_alp
        ) / delta

        # Compute the derivatives using fit_deriv
        deriv = self.model.fit_deriv(nu, area, synch_S1, synch_alp)

        # Compare numerical derivatives with fit_deriv results
        self.assertAlmostEqual(
            numerical_deriv_S1, deriv[0], places=5
        )  # Allow a small tolerance
        self.assertAlmostEqual(
            numerical_deriv_alp, deriv[1], places=5
        )  # Allow a small tolerance


class TestFreeFree7500KEmissionModel(unittest.TestCase):

    def setUp(self):
        # Create an instance of the freeFree_7000k model
        self.model = mapext_cont.freeFree_7500k()

    def test_evaluate(self):
        # Test evaluate with scalar inputs
        nu = 1.0  # GHz (Frequency)
        area = 1e-4  # steradians (Beam area)
        ff_em = 1000  # Free-free emission measure (arbitrary units)

        # Expected result will be calculated inside the evaluate method.
        result = self.model.evaluate(nu, area, ff_em)

        # Use a known result for comparison (if available), or validate the model works as expected.
        # The expected value here will depend on the actual calculations from the model.
        expected_result = (
            2.0
            * phys_const["k"]
            * area
            * np.power(np.multiply(nu, 1e9), 2)
            / phys_const["c"] ** 2
            * 8.235e-2
            * 0.366
            * np.power(nu, 0.1)
            * np.power(7500, -0.15)
            * (np.log(np.divide(4.995e-2, nu)) + 1.5 * np.log(7500))
            * np.power(7500, -0.35)
            * np.power(nu, -2.1)
            * (1.0 + 0.08)
            * ff_em
            * 1e26
        )
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_fit_deriv_with_numerical_approximation(self):
        # Test that fit_deriv is numerically consistent with evaluate

        nu = 1.0  # GHz
        area = 1e-4  # steradians
        ff_em = 100  # Free-free emission measure
        delta = 1e-5  # Perturbation to numerically approximate the derivative

        # Numerical approximation of the derivative w.r.t. ff_em
        evaluate_at_ff_em_plus_delta = self.model.evaluate(nu, area, ff_em + delta)
        evaluate_at_ff_em = self.model.evaluate(nu, area, ff_em)
        numerical_deriv_ff_em = (
            evaluate_at_ff_em_plus_delta - evaluate_at_ff_em
        ) / delta

        # Compute the derivative using fit_deriv
        deriv = self.model.fit_deriv(nu, area, ff_em)

        # Compare numerical derivative with fit_deriv result
        self.assertAlmostEqual(
            numerical_deriv_ff_em, deriv[0], places=2
        )  # Allow a small tolerance


class TestFreeFreeEmissionModel(unittest.TestCase):

    def setUp(self):
        # Create an instance of the freeFree model
        self.model = mapext_cont.freeFree()

    def test_evaluate(self):
        # Test evaluate with scalar inputs
        nu = 1.0  # GHz
        area = 1e-4  # steradians
        ff_em = 100  # Free-free emission measure
        ff_Te = 7000  # Electron temperature in Kelvin

        # Evaluate the model
        result = self.model.evaluate(nu, area, ff_em, ff_Te)

        # Expected result will be calculated inside the evaluate method.
        # The expected value here will depend on the exact calculation needed.
        # For example, use the formula for S computed in the evaluate method.
        expected_result = (
            2.0
            * phys_const["k"]
            * area
            * np.power(np.multiply(nu, 1e9), 2)
            / phys_const["c"] ** 2
            * ff_Te
            * (
                1
                - np.exp(
                    -1
                    * (
                        5.468e-2
                        * (ff_Te**-1.5)
                        * (nu**-2)
                        * ff_em
                        * np.log(
                            np.exp(
                                5.90
                                - (
                                    np.sqrt(3)
                                    / np.pi
                                    * np.log(nu * ((ff_Te / 1e4) ** 1.5))
                                )
                            )
                            + 2.71828
                        )
                    )
                )
            )
            * 1e26
        )

        self.assertAlmostEqual(result, expected_result, places=5)

    def test_fit_deriv_with_numerical_approximation(self):
        # Test that fit_deriv is numerically consistent with evaluate

        nu = 1.0  # GHz
        area = 1e-4  # steradians
        ff_em = 100  # Free-free emission measure
        ff_Te = 7000  # Electron temperature
        delta = 1e-5  # Perturbation to numerically approximate the derivative

        # Numerical approximation of the derivative w.r.t. ff_em
        evaluate_at_ff_em_plus_delta = self.model.evaluate(
            nu, area, ff_em + delta, ff_Te
        )
        evaluate_at_ff_em = self.model.evaluate(nu, area, ff_em, ff_Te)
        numerical_deriv_ff_em = (
            evaluate_at_ff_em_plus_delta - evaluate_at_ff_em
        ) / delta

        # Numerical approximation of the derivative w.r.t. ff_Te
        evaluate_at_ff_Te_plus_delta = self.model.evaluate(
            nu, area, ff_em, ff_Te + delta
        )
        evaluate_at_ff_Te = self.model.evaluate(nu, area, ff_em, ff_Te)
        numerical_deriv_ff_Te = (
            evaluate_at_ff_Te_plus_delta - evaluate_at_ff_Te
        ) / delta

        # Compute the derivative using fit_deriv
        deriv = self.model.fit_deriv(nu, area, ff_em, ff_Te)

        # Compare numerical derivatives with fit_deriv results
        self.assertAlmostEqual(
            numerical_deriv_ff_em, deriv[0], places=2
        )  # Allow a small tolerance
        self.assertAlmostEqual(
            numerical_deriv_ff_Te, deriv[1], places=2
        )  # Allow a small tolerance


class TestAmeLognormalEmissionModel(unittest.TestCase):

    def setUp(self):
        # Create an instance of the ame_lognormal model
        self.model = mapext_cont.ame_lognormal()

    def test_evaluate(self):
        # Test evaluate with scalar inputs
        nu = 30.0  # GHz (example frequency near the peak)
        area = 1e-4  # steradians (example beam area)
        ame_ampl = 0.1  # AME peak flux density
        ame_peak = 27.0  # AME peak frequency
        ame_width = 0.5  # AME lognormal width

        # Evaluate the model
        result = self.model.evaluate(nu, area, ame_ampl, ame_peak, ame_width)

        # Calculate the expected result based on the formula in the model
        nlog = np.log(nu)
        nmaxlog = np.log(ame_peak)
        expected_result = ame_ampl * np.exp(-0.5 * ((nlog - nmaxlog) / ame_width) ** 2)

        self.assertAlmostEqual(result, expected_result, places=5)

    def test_fit_deriv_with_numerical_approximation(self):
        # Test that fit_deriv is numerically consistent with evaluate

        nu = 30.0  # GHz (example frequency near the peak)
        area = 1e-4  # steradians (example beam area)
        ame_ampl = 0.1  # AME peak flux density
        ame_peak = 27.0  # AME peak frequency
        ame_width = 0.5  # AME lognormal width
        delta = 1e-5  # Perturbation to numerically approximate the derivative

        # Numerical approximation of the derivative w.r.t. ame_ampl
        evaluate_at_ame_ampl_plus_delta = self.model.evaluate(
            nu, area, ame_ampl + delta, ame_peak, ame_width
        )
        evaluate_at_ame_ampl = self.model.evaluate(
            nu, area, ame_ampl, ame_peak, ame_width
        )
        numerical_deriv_ame_ampl = (
            evaluate_at_ame_ampl_plus_delta - evaluate_at_ame_ampl
        ) / delta

        # Numerical approximation of the derivative w.r.t. ame_peak
        evaluate_at_ame_peak_plus_delta = self.model.evaluate(
            nu, area, ame_ampl, ame_peak + delta, ame_width
        )
        evaluate_at_ame_peak = self.model.evaluate(
            nu, area, ame_ampl, ame_peak, ame_width
        )
        numerical_deriv_ame_peak = (
            evaluate_at_ame_peak_plus_delta - evaluate_at_ame_peak
        ) / delta

        # Numerical approximation of the derivative w.r.t. ame_width
        evaluate_at_ame_width_plus_delta = self.model.evaluate(
            nu, area, ame_ampl, ame_peak, ame_width + delta
        )
        evaluate_at_ame_width = self.model.evaluate(
            nu, area, ame_ampl, ame_peak, ame_width
        )
        numerical_deriv_ame_width = (
            evaluate_at_ame_width_plus_delta - evaluate_at_ame_width
        ) / delta

        # Compute the derivative using fit_deriv
        deriv = self.model.fit_deriv(nu, area, ame_ampl, ame_peak, ame_width)

        # Compare numerical derivatives with fit_deriv results
        self.assertAlmostEqual(
            numerical_deriv_ame_ampl, deriv[0], places=5
        )  # Allow a small tolerance
        self.assertAlmostEqual(
            numerical_deriv_ame_peak, deriv[1], places=5
        )  # Allow a small tolerance
        self.assertAlmostEqual(
            numerical_deriv_ame_width, deriv[2], places=5
        )  # Allow a small tolerance


class TestThermalDustEmissionModel(unittest.TestCase):

    def setUp(self):
        # Create an instance of the thermalDust model
        self.model = mapext_cont.thermalDust()

    def test_evaluate(self):
        # Test evaluate with scalar inputs
        nu = 150.0  # GHz (example frequency)
        area = 1e-4  # steradians (example beam area)
        tdust_Td = 20.0  # Thermal dust temperature in Kelvin
        tdust_tau = -4.0  # Log10 of dust opacity
        tdust_beta = 1.5  # Thermal dust spectral index

        # Evaluate the model
        result = self.model.evaluate(nu, area, tdust_Td, tdust_tau, tdust_beta)

        # Compute the expected result based on the formula
        nu9 = np.multiply(nu, 1e9)
        planck = np.exp(phys_const["h"] * nu9 / phys_const["k"] / tdust_Td) - 1.0
        modify = 10**tdust_tau * (nu9 / 1.2e12) ** tdust_beta
        expected_result = (
            2
            * phys_const["h"]
            * nu9**3
            / phys_const["c"] ** 2
            / planck
            * modify
            * area
            * 1e26
        )

        self.assertAlmostEqual(result, expected_result, places=5)

    def test_fit_deriv_with_numerical_approximation(self):
        # Test that fit_deriv is numerically consistent with evaluate

        nu = 150.0  # GHz (example frequency)
        area = 1e-4  # steradians (example beam area)
        tdust_Td = 20.0  # Thermal dust temperature in Kelvin
        tdust_tau = -4.0  # Log10 of dust opacity
        tdust_beta = 1.5  # Thermal dust spectral index
        delta = 1e-8  # Perturbation to numerically approximate the derivative

        # Numerical approximation of the derivative w.r.t. tdust_Td
        evaluate_at_tdust_Td_plus_delta = self.model.evaluate(
            nu, area, tdust_Td + delta, tdust_tau, tdust_beta
        )
        evaluate_at_tdust_Td = self.model.evaluate(
            nu, area, tdust_Td, tdust_tau, tdust_beta
        )
        numerical_deriv_tdust_Td = (
            evaluate_at_tdust_Td_plus_delta - evaluate_at_tdust_Td
        ) / delta

        # Numerical approximation of the derivative w.r.t. tdust_tau
        evaluate_at_tdust_tau_plus_delta = self.model.evaluate(
            nu, area, tdust_Td, tdust_tau + delta, tdust_beta
        )
        evaluate_at_tdust_tau = self.model.evaluate(
            nu, area, tdust_Td, tdust_tau, tdust_beta
        )
        numerical_deriv_tdust_tau = (
            evaluate_at_tdust_tau_plus_delta - evaluate_at_tdust_tau
        ) / delta

        # Numerical approximation of the derivative w.r.t. tdust_beta
        evaluate_at_tdust_beta_plus_delta = self.model.evaluate(
            nu, area, tdust_Td, tdust_tau, tdust_beta + delta
        )
        evaluate_at_tdust_beta = self.model.evaluate(
            nu, area, tdust_Td, tdust_tau, tdust_beta
        )
        numerical_deriv_tdust_beta = (
            evaluate_at_tdust_beta_plus_delta - evaluate_at_tdust_beta
        ) / delta

        # Compute the derivative using fit_deriv
        deriv = self.model.fit_deriv(nu, area, tdust_Td, tdust_tau, tdust_beta)

        # Compare numerical derivatives with fit_deriv results
        self.assertAlmostEqual(numerical_deriv_tdust_Td, deriv[0], places=2)
        self.assertAlmostEqual(numerical_deriv_tdust_tau, deriv[1], places=2)
        self.assertAlmostEqual(numerical_deriv_tdust_beta, deriv[2], places=0)
