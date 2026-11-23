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

    def test_evanescent_pressure_phase_decay_and_velocity_match(self):
        for angle in (-45, 45):
            result = interface(incidence_angle_deg=angle)
            self.assertTrue(result["verified"], result)
            self.assertEqual(result["regime"], "evanescent")
            reflected = complex(*result["pressure_reflection_complex"])
            transmitted = complex(*result["pressure_transmission_complex"])
            velocity = complex(*result["transmitted_normal_velocity_ratio"])
            self.assertAlmostEqual(reflected.real, 1/3)
            self.assertAlmostEqual(reflected.imag, -2*math.sqrt(2)/3)
            self.assertAlmostEqual(abs(1+reflected-transmitted), 0)
            self.assertAlmostEqual(abs(1-reflected-velocity), 0)
            self.assertAlmostEqual((transmitted*velocity.conjugate()).real, 0)
            self.assertEqual(result["transmitted_normal_slowness_s_m"], [0, .5])
            self.assertEqual(result["tangential_slowness_s_m"] > 0, angle > 0)
            self.assertTrue(result["transmission_loss_infinite"])
            self.assertIsNone(result["transmission_loss_dB"])

    def test_exact_critical_plus_signed_angular_neighbors(self):
        for angle in (-30.,30.):
            for direction, regime in ((math.nextafter(angle,0), "propagating"),
                                      (angle, "critical"),
                                      (math.nextafter(angle,2*angle), "evanescent")):
                result = interface(incidence_angle_deg=direction)
                self.assertTrue(result["verified"], result)
                self.assertEqual(result["regime"], regime)
            critical = interface(incidence_angle_deg=angle)
            self.assertEqual(critical["pressure_reflection_R"], 1)
            self.assertEqual(critical["pressure_transmission_T"], 2)
            self.assertEqual(critical["transmitted_normal_velocity_ratio"], [0,0])

    def test_matching_media_and_normal_cross_check(self):
        for angle in (0, 12.3, 45, 89.9):
            result = interface(incidence_angle_deg=angle, rho2=1, c2=1)
            self.assertTrue(result["verified"], result)
            self.assertEqual(result["pressure_reflection_complex"], [0,0])
            self.assertEqual(result["intensity_transmission_TI"], 1)
            self.assertEqual(result["transmission_loss_dB"], 0)
        normal = interface()
        declared = interface(incidence_sine=0)
        for key in ("pressure_reflection_R", "pressure_transmission_T", "intensity_transmission_TI"):
            self.assertEqual(normal[key], declared[key])
        resonator = dict(A_neck=.1, V_cavity=1, L_eff=.5)
        self.assertEqual(interface(**resonator)["helmholtz_f0_Hz"],
                         interface(**resonator, incidence_angle_deg=30)["helmholtz_f0_Hz"])
        self.assertFalse(interface(incidence_sine=.5, incidence_angle_deg=30)["verified"])
        self.assertFalse(interface(incidence_angle_deg=91)["verified"])
