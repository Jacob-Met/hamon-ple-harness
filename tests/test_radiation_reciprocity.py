"""Geometric controls for mixed-area diffuse radiation enclosures."""
import itertools
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def enclosure(reverse, scale=1., order=(0, 1, 2)):
    # The small patches have equal radiosity, but exchange with a hotter wall.
    # All rows close and both wall links are reciprocal for every reverse value.
    areas = [1e-4, 2e-4, 100.]
    to_first = 5e-7
    to_second = 2e-6 * (1 - reverse)
    factors = [[0., .5, .5], [reverse, 0., 1 - reverse],
               [to_first, to_second, 1 - to_first - to_second]]
    temperatures = [300., 300., 500.]
    return run_router_conservation_cad_verifier({
        "mode": "exchangers_radiation",
        "spec": {"subkind": "radiation_network",
                 "areas": [scale * areas[i] for i in order],
                 "F": [[factors[i][j] for j in order] for i in order],
                 "T_K": [temperatures[i] for i in order],
                 "emissivities": [1., 1., 1.]},
    })


class TestViewFactorReciprocity(unittest.TestCase):
    def test_mixed_area_enclosure(self):
        # Independent black-surface exchange law, Q_i=sum(A_i F_ij σ(T_i⁴−T_j⁴)).
        radiance_difference = 5.670374419e-8 * (500. ** 4 - 300. ** 4)
        expected = [-.5e-4 * radiance_difference, -1.5e-4 * radiance_difference,
                    2e-4 * radiance_difference]
        for scale, order in itertools.product((1e-3, 1., 1e3), itertools.permutations(range(3))):
            with self.subTest(scale=scale, order=order):
                valid = enclosure(.25, scale, order)
                self.assertTrue(valid["verified"])
                for index, heat in zip(order, valid["Q_net_W"]):
                    self.assertTrue(math.isclose(heat, scale * expected[index], rel_tol=1e-12))
                invalid = enclosure(.5, scale, order)
                self.assertFalse(invalid["verified"])
                self.assertEqual(invalid["error"], "view_factor_enclosure_or_reciprocity_violation")

    def test_local_fraction_tolerance(self):
        # (A2 / A1) * change in F21 is the mismatch as a fraction of A1.
        self.assertTrue(enclosure(.25 + 4e-6)["verified"])
        self.assertFalse(enclosure(.25 + 6e-6)["verified"])
