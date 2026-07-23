"""Comprobaciones numéricas del modelo SKR documentado en la tesis."""

import math
import unittest

from protocols.skr_bb84 import (
    qber,
    skr_bb84_asymptotic,
    skr_bb84_decoy,
)


class SkrBb84AsymptoticTests(unittest.TestCase):
    def test_reference_points_from_declared_equations(self):
        expected = {
            10: 6.4664382800499395e-3,
            50: 1.0188456107580379e-3,
            100: 9.996046924682088e-5,
            150: 8.253730896477398e-6,
        }
        for distance, rate in expected.items():
            with self.subTest(distance=distance):
                self.assertTrue(
                    math.isclose(
                        skr_bb84_asymptotic(distance),
                        rate,
                        rel_tol=1e-12,
                        abs_tol=0.0,
                    )
                )

    def test_sifting_factor_is_applied(self):
        default_rate = skr_bb84_asymptotic(50)
        unsifted_rate = skr_bb84_asymptotic(50, q=1.0)
        self.assertTrue(
            math.isclose(default_rate * 2, unsifted_rate, rel_tol=1e-12)
        )

    def test_reference_cut_is_crossed_between_190_and_191_km(self):
        self.assertGreater(skr_bb84_asymptotic(190), 1e-8)
        self.assertLess(skr_bb84_asymptotic(191), 1e-8)

    def test_signal_qber_at_50_km(self):
        self.assertTrue(math.isclose(qber(50), 0.015097238303001298,
                                     rel_tol=1e-12))

    def test_historical_alias_uses_same_model(self):
        self.assertEqual(skr_bb84_decoy(45), skr_bb84_asymptotic(45))

    def test_finite_decoy_intensity_is_not_silently_ignored(self):
        with self.assertRaisesRegex(TypeError, "no forma parte"):
            skr_bb84_decoy(45, mu_decoy=0.1)

    def test_invalid_parameters_fail_explicitly(self):
        with self.assertRaises(ValueError):
            skr_bb84_asymptotic(-1)
        with self.assertRaises(ValueError):
            skr_bb84_asymptotic(10, q=1.1)


if __name__ == "__main__":
    unittest.main()
