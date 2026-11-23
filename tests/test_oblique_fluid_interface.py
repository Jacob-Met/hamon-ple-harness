import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def interface(**values):
    spec = dict(subkind="acoustics", rho1=1, c1=1, rho2=1, c2=2)
    return run_router_conservation_cad_verifier(
        {"mode": "compressible_acoustics", "spec": dict(spec, **values)})


class ObliqueFluidInterface(unittest.TestCase):
    def test_evanescent_half_space_carries_no_mean_normal_power(self):
        result = interface(incidence_angle_deg=45)
        self.assertTrue(result["verified"], result)
        self.assertEqual(result["intensity_transmission_TI"], 0, result)

    def test_critical_contact_and_adjacent_declared_directions(self):
        at = interface(incidence_angle_deg=30)
        self.assertTrue(at["verified"], at)
        self.assertEqual(at["intensity_transmission_TI"], 0, at)
        below = interface(incidence_sine=math.nextafter(.5, 0))
        above = interface(incidence_sine=math.nextafter(.5, 1))
        self.assertGreater(below["intensity_transmission_TI"], 0, below)
        self.assertEqual(above["intensity_transmission_TI"], 0, above)

    def test_subcritical_refraction_is_different_from_normal_incidence(self):
        q = .25
        y1 = math.sqrt(1 - q*q)
        y2 = math.sqrt(1 - (2*q)**2) / 2
        expected = 4*y1*y2/(y1+y2)**2
        result = interface(incidence_sine=q)
        self.assertTrue(result["verified"], result)
        self.assertAlmostEqual(result["intensity_transmission_TI"], expected, places=14)
        normal = interface()
        self.assertEqual(normal["intensity_transmission_TI"], 8/9, normal)

    def test_grazing_has_no_incident_normal_power_to_normalize(self):
        for direction in (-90, 90):
            with self.subTest(direction=direction):
                result = interface(incidence_angle_deg=direction)
                self.assertFalse(result["verified"], result)
