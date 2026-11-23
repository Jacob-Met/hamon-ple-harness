"""Analytic surface-energy controls for nearly reflecting diffuse-gray enclosures."""
import itertools
import unittest
from fractions import Fraction

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


SIGMA = Fraction(5.670374419e-8)


def enclosure(factors, temperatures, emissivities, areas=None):
    return run_router_conservation_cad_verifier({
        "mode": "exchangers_radiation",
        "spec": {
            "subkind": "radiation_network", "areas": areas or [1.] * len(temperatures),
            "emissivities": emissivities, "F": factors, "T_K": temperatures,
        },
    })


class WeakRadiationBalance(unittest.TestCase):
    def test_single_cavity_radiosity_is_blackbody_at_every_positive_emissivity(self):
        for eps in (1., 1e-5, 1e-12, 1e-16, 2.**-54, 5e-324):
            with self.subTest(eps=eps):
                result = enclosure([[1.]], [400.], [eps])
                self.assertTrue(result["verified"])
                self.assertAlmostEqual(result["radiosity_W_m2"][0], float(SIGMA * 400**4), places=8)
                self.assertEqual(result["Q_net_W"], [0.])

    def test_two_gray_surfaces_recover_heat_below_published_radiosity_resolution(self):
        # The radiation resistance for two unit-view surfaces of equal area
        # is (1/e1 + 1/e2 - 1) / A. This independently determines every flux.
        for temps, eps, area in itertools.product(
                ((300., 500.), (500., 300.)), ((1e-18, 3e-18), (1., 1e-16)), (1., 1e16)):
            with self.subTest(temps=temps, eps=eps, area=area):
                ea, eb = map(Fraction, eps)
                e1, e2 = [SIGMA * Fraction(t)**4 for t in temps]
                flux = (e1 - e2) / (1/ea + 1/eb - 1)
                j1, j2 = e1 - (1/ea - 1) * flux, e2 + (1/eb - 1) * flux
                result = enclosure([[0., 1.], [1., 0.]], temps, eps, [area, area])
                self.assertTrue(result["verified"])
                self.assertEqual(result["radiosity_W_m2"], [float(j1), float(j2)])
                self.assertEqual(result["Q_net_W"], [float(Fraction(area) * flux), float(-Fraction(area) * flux)])

    def test_adiabatic_surface_coupled_to_black_wall_and_free_component(self):
        # The 400 K black wall is the only emitter. A tiny nonzero aperture
        # fixes the equilibrium even when both self-view factors round to 1.
        opening = 2.**-80
        result = enclosure([[1., opening], [opening, 1.]], [400., None], [1., 0.75])
        self.assertTrue(result["verified"])
        self.assertEqual(result["T_surface_K"], [400., 400.])
        self.assertEqual(result["Q_net_W"], [0., 0.])
        missing = enclosure([[1., 0.], [0., 1.]], [400., None], [1., .75])
        self.assertFalse(missing["verified"])
        self.assertEqual(missing["surfaces_without_temperature_path"], [1])
        zero_emission = enclosure([[1.]], [400.], [0.])
        self.assertFalse(zero_emission["verified"])
        self.assertEqual(zero_emission["error"], "invalid_surface_properties")
