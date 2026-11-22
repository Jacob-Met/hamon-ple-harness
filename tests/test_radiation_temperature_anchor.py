"""Temperature identifiability in a closed diffuse-gray enclosure."""
import unittest
from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


def enclosure(factors, temperatures, emissivities=None, adiabatic=()):
    return dispatch_wave7_verifier("exchangers_radiation", {
        "subkind": "radiation_network",
        "F": factors,
        "areas": [1.0] * len(factors),
        "emissivities": emissivities or [0.75] * len(factors),
        "T_K": temperatures,
        "adiabatic_indices": list(adiabatic),
    })


class TemperatureAnchorTests(unittest.TestCase):
    def test_partitioned_spherical_cavity_has_a_free_temperature(self):
        # On a sphere, dF_ij = dA_j/(4*pi*R^2) for each interior point.
        for n in (2, 3, 5, 7, 9):
            with self.subTest(n=n):
                factors = [[1 / n] * n for _ in range(n)]
                result = enclosure(factors, [None] * n)
                self.assertFalse(result["verified"])
                self.assertEqual(result["error"], "underdetermined_closed_radiation_enclosure")
                self.assertEqual(result["surfaces_without_temperature_path"], list(range(n)))
                self.assertNotIn("T_surface_K", result)
                self.assertNotIn("radiosity_W_m2", result)

                anchored = enclosure(factors, [340.0] + [None] * (n - 1))
                self.assertTrue(anchored["verified"])
                for temperature in anchored["T_surface_K"]:
                    self.assertAlmostEqual(temperature, 340.0, places=8)

    def test_disconnected_adiabatic_component_stays_free(self):
        factors = [[1.0, 0, 0, 0], *([[0, 1/3, 1/3, 1/3]] * 3)]
        for order in ((0, 1, 2, 3), (3, 2, 1, 0)):
            with self.subTest(order=order):
                matrix = [[factors[i][j] for j in order] for i in order]
                temperatures = [400.0 if i == 0 else None for i in order]
                result = enclosure(matrix, temperatures)
                self.assertFalse(result["verified"])
                self.assertEqual(result["surfaces_without_temperature_path"],
                                 [j for j, i in enumerate(order) if i != 0])

    def test_fixed_temperatures_overridden_by_adiabatic_indices(self):
        result = enclosure([[0, 1], [1, 0]], [350.0, 325.0], adiabatic=[0, 1])
        self.assertFalse(result["verified"])
        self.assertEqual(result["surfaces_without_temperature_path"], [0, 1])

    def test_multihop_anchor_and_existing_properties_guards(self):
        chain = [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        result = enclosure(chain, [None, None, 400.0])
        self.assertTrue(result["verified"])
        for temperature in result["T_surface_K"]:
            self.assertAlmostEqual(temperature, 400.0, places=8)
        self.assertEqual(enclosure([[0, 1], [1, 0]], [350, None], [0, .75])["error"],
                         "invalid_surface_properties")
        self.assertEqual(enclosure([[0, .8], [.8, 0]], [350, None])["error"],
                         "view_factor_enclosure_or_reciprocity_violation")
