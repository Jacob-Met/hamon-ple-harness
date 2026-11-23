"""Lamé wall geometry and radial/hoop stress invariants on the registered route."""
import math
import unittest
from fractions import Fraction

from hamon_wave6_physics_verifiers import dispatch_wave6_verifier


def vessel(**values):
    spec = dict(subkind="pressure_vessel", r_i=1, r_o=2, p_i=3e6, p_o=1e6)
    return dispatch_wave6_verifier("solid_mohr_vessel_buckling", dict(spec, **values))


class LameAnnularState(unittest.TestCase):
    def test_every_supplied_wall_dimension_belongs_to_one_annulus(self):
        for thickness in (-1, 0, .02, 1.01):
            with self.subTest(thickness=thickness):
                r = vessel(t=thickness)
                self.assertFalse(r["verified"], r)
        self.assertTrue(vessel(t=1)["verified"])

    def test_closed_end_equivalent_stress_survives_hydrostatic_shift(self):
        ri, ro = .001, 1000.
        geometry = Fraction(ri)**2 / (Fraction(ro)**2-Fraction(ri)**2)
        for pressure in (0, 100000):
            for excess in (-1, 1):
                with self.subTest(pressure=pressure, excess=excess):
                    r = vessel(r_i=ri, r_o=ro, r_eval=ro, p_i=pressure+excess,
                               p_o=pressure, closed_ends=True)
                    self.assertTrue(r["verified"], r)
                    expected = math.sqrt(3)*float(geometry)
                    self.assertTrue(math.isclose(r["sigma_von_mises"], expected,
                                                 rel_tol=3e-15), r)
                    if pressure:
                        self.assertEqual(r["sigma_radial_lame"], -pressure)
                        self.assertEqual(r["sigma_hoop_lame"], -pressure)
                        self.assertEqual(r["sigma_axial_lame"], -pressure)

    def test_joint_pressures_and_ends_have_distinct_equivalent_stress(self):
        # Identical pressure on both faces is a hydrostatic loading only when
        # the same pressure resultant acts on closed end caps.
        for closed, expected in ((True,0.), (False,100000.)):
            with self.subTest(closed=closed):
                result = vessel(p_i=100000, p_o=100000, closed_ends=closed, t=1)
                self.assertTrue(result["verified"], result)
                self.assertEqual(result["sigma_von_mises"], expected)
