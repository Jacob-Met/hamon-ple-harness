"""Physical impulse checks for the registered circular-orbit transfer route."""
import math
import unittest

from mpmath import mp

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


class OrbitalTransferTests(unittest.TestCase):
    def check_transfer(self, r1, r2, **tilt):
        mu = 3.986004418e14
        p = dict(subkind="orbital_transfer", mu=mu, r1=r1, r2=r2, isp_s=315, **tilt)
        result = dispatch_wave7_verifier("Hohmann", p)
        self.assertTrue(result["verified"], result)

        # Direct ellipse energy at both endpoints and ordinary Cartesian velocity
        # vectors. Extra precision here permits the subtractive formulation.
        c = mp.clone()
        c.dps = 145
        mu, r1, r2 = map(c.mpf, (mu, r1, r2))
        angle = (c.mpf(tilt["delta_inc_rad"]) if "delta_inc_rad" in tilt
                 else c.radians(math.remainder(tilt.get("delta_inc_deg", 0), 360)))
        expected = []
        for radius, tilted in ((r1, r1 > r2), (r2, r2 >= r1)):
            circular = c.sqrt(mu / radius)
            transfer = c.sqrt(mu * (2 / radius - 2 / (r1 + r2)))
            if r1 == r2:
                transfer = circular
            phi = angle if tilted else 0
            expected.append(c.hypot(transfer - circular * c.cos(phi), circular * c.sin(phi)))
        total = sum(expected)
        values = dict(dv1_m_s=expected[0], dv2_m_s=expected[1], dv_total_m_s=total,
                      propellant_mass_fraction=-c.expm1(-total / (315 * c.mpf(9.80665))),
                      transfer_time_s=c.pi * c.sqrt(((r1 + r2) / 2)**3 / mu))
        for name, value in values.items():
            self.assertTrue(math.isclose(result[name], float(value), rel_tol=3e-15), (name, result, value))
        return result

    def test_combined_burn_at_the_higher_apsis(self):
        for r1, r2 in [(7e6, 7e6), (7e6, math.nextafter(7e6, math.inf)),
                       (math.nextafter(7e6, math.inf), 7e6), (6678e3, 42164e3), (42164e3, 6678e3)]:
            for angle in [0.0, -1.e-8, 1.e-8, math.pi]:
                with self.subTest(r1=r1, r2=r2, angle=angle):
                    self.check_transfer(r1, r2, delta_inc_rad=angle)

    def test_degree_convention_and_tiny_propellant(self):
        for degrees in [0.0, 360.0, -720.0, 28.5, -28.5, 1.e250]:
            with self.subTest(degrees=degrees):
                self.check_transfer(7e6, 7e6, delta_inc_deg=degrees)
        tiny = self.check_transfer(7e6, 7e6, delta_inc_rad=1.e-80)
        self.assertGreater(tiny["propellant_mass_fraction"], 0.)

    def test_invalid_scalars_and_unrepresentable_output(self):
        base = dict(subkind="orbital_transfer", mu=1., r1=1., r2=1.)
        for change in [dict(mu=float("inf")), dict(r1=0), dict(r2=True), dict(isp_s=0),
                       dict(isp_s=1, g0=-1), dict(delta_inc_rad=float("nan")),
                       dict(r1=1.e-200, r2=1.e200)]:
            with self.subTest(change=change):
                self.assertFalse(dispatch_wave7_verifier("Hohmann", dict(base, **change))["verified"])
