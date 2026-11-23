"""Pressure-wall admissibility at the registered cylindrical shell route."""
from fractions import Fraction
import math
import unittest
from hamon_dual_verifier import run_dual_verify_spec


def shell(**changes):
    spec = dict(P_Pa=2e6, R_inner_m=1., S_allow_Pa=1e6, weld_efficiency_E=1.,
                corrosion_allowance_m=0., supplied_thickness_m=.01)
    spec.update(changes)
    return run_dual_verify_spec(dict(mode="domain", kind="asme_ug27_shell", spec=spec))


class ShellPressureAdmissibility(unittest.TestCase):
    def no_certified_pass(self, **changes):
        try:
            result = shell(**changes)
        except ValueError:
            return
        self.assertIsNot(result.get("passes_ug27"), True, result)
        if result.get("t_required_m") is not None:
            self.assertGreaterEqual(result["t_required_m"], 0., result)

    def test_no_negative_wall_solution_above_pressure_asymptote(self):
        self.no_certified_pass()

    def test_thick_wall_beyond_formula_cannot_certify(self):
        # At P=.4 SE the inferred load-bearing thickness exceeds R/2.
        self.no_certified_pass(P_Pa=400000., supplied_thickness_m=.6)

    def test_efficiency_and_corrosion_cannot_create_a_pressure_wall(self):
        for kw in [dict(P_Pa=-1000.), dict(P_Pa=1000.,weld_efficiency_E=2.),
                   dict(P_Pa=1000.,corrosion_allowance_m=-.1),
                   dict(P_Pa=0.,corrosion_allowance_m=.01)]:
            with self.subTest(kw=kw):
                self.no_certified_pass(**kw)

    def test_exact_pressure_wall_equality_and_adjacent_binary_inputs(self):
        # At R=1, SE=67 MPa, t=5/64 m the pressure rating is exactly 5 MPa.
        for thickness in (math.nextafter(.09375, 0.), .09375,
                          math.nextafter(.09375, 1.)):
            with self.subTest(thickness=thickness):
                r = shell(P_Pa=5e6, S_allow_Pa=134e6, weld_efficiency_E=.5,
                          supplied_thickness_m=thickness, corrosion_allowance_m=1/64)
                self.assertTrue(r["dual_verified"],r)
                self.assertEqual(r["passes_ug27"], thickness >= .09375)
                self.assertEqual(r["pressure_thickness_required_m"], 5/64)
                self.assertEqual(r["t_required_m"], .09375)
                t = Fraction(thickness)-Fraction(1,64)
                self.assertEqual(r["circumferential_pressure_capacity_Pa"],
                                 float(67_000_000*t/(1+Fraction(3,5)*t)))

    def test_inverse_sizing_and_rating_have_separate_applicability(self):
        for p in (math.nextafter(77e6,0.), 77e6, math.nextafter(77e6,math.inf)):
            with self.subTest(pressure=p):
                r = shell(P_Pa=p, S_allow_Pa=200e6, supplied_thickness_m=.75)
                self.assertIsNone(r["circumferential_pressure_capacity_Pa"])
                if p <= 77e6:
                    self.assertTrue(r["dual_verified"],r)
                    self.assertTrue(r["passes_ug27"],r)
                    self.assertEqual(r["t_required_m"],float(Fraction(p)/(200_000_000-Fraction(3,5)*Fraction(p))))
                else:
                    self.assertFalse(r["dual_verified"],r)
                    self.assertIsNone(r["passes_ug27"])
                    self.assertIsNone(r["t_required_m"])
        r = shell(P_Pa=2e6)
        self.assertTrue(r["dual_verified"],r)
        self.assertFalse(r["passes_ug27"])
        self.assertIsNone(r["t_required_m"])
        t = Fraction(.01)
        self.assertEqual(r["circumferential_pressure_capacity_Pa"],float(1_000_000*t/(1+Fraction(3,5)*t)))

    def test_zero_internal_pressure_still_requires_positive_remaining_metal(self):
        r = shell(P_Pa=0.,corrosion_allowance_m=.002,supplied_thickness_m=.01)
        self.assertTrue(r["dual_verified"],r)
        self.assertTrue(r["passes_ug27"])
        self.assertEqual(r["t_required_m"],.002)
        self.assertEqual(r["pressure_thickness_required_m"],0.)
