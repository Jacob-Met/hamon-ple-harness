"""Continuous displacement and reachability gates for registered beam designs."""
import math
import unittest
from fractions import Fraction
from unittest.mock import patch

import hamon_dual_verifier as dv


def design(support, **spec):
    return dv.solve_domain_problem("beam_design_check",
                                   {"span": 1, "E": 1, "I": 1, "support": support, **spec})


class BeamContinuousDeflectionTests(unittest.TestCase):
    def test_self_equilibrated_loads_between_nodes(self):
        # The fourth difference annihilates every cubic virtual displacement
        # on this element. Its integrated local bending displacement is nonzero.
        loads = [{"x": 4 + k/32, "P": 240*w} for k,w in enumerate((1,-4,6,-4,1))]
        for support in ("simply_supported", "fixed_fixed", "propped_cantilever", "cantilever"):
            for sign in (-1, 1):
                signed = [dict(p, P=sign*p["P"]) for p in loads]
                with self.subTest(support=support, sign=sign):
                    r = design(support, span=8, E=200e9, I=1e-10, point_loads=signed,
                               deflection_limit_ratio=40000)
                    self.assertTrue(r["dual_verified"])
                    self.assertEqual((r["delta_max_m"], r["delta_mid_m"]), (1/4096, 0))
                    self.assertEqual((r["M_max_Nm"],r["V_max_N"]), (15,720))
                    self.assertFalse(r["deflection_ok"])

    def test_exact_uniform_load_solutions(self):
        # Maximum and midpoint from the classical load integrals. For a
        # propped cantilever, the interior slope root solves 8t²-15t+6=0.
        t = (15-math.sqrt(33))/16
        prop = t*t/16 - 5*t**3/48 + t**4/24
        for support, peak, mid in (("simply_supported", 5/384, 5/384),
                                   ("fixed_fixed", 1/384, 1/384),
                                   ("cantilever", 1/8, 17/384),
                                   ("propped_cantilever", prop, 1/192)):
            with self.subTest(support=support):
                r = design(support, udl=-12, E=3)
                self.assertAlmostEqual(r["delta_max_m"], 4*peak, places=14)
                self.assertAlmostEqual(r["delta_mid_m"], 4*mid, places=14)

    def test_nondyadic_stationary_contact_and_adjacent_service_limits(self):
        # A unit cantilever under q=36 and tip P=-13 has its greatest
        # deflection at x=2/3. EI*v there is -22/81 and at L/2 it is -23/96.
        for ratio, admissible in ((math.nextafter(81,0), True),(81,True),
                                  (math.nextafter(81,math.inf),False)):
            with self.subTest(ratio=ratio):
                r = design("cantilever", udl=36, point_loads=[{"x":1,"P":-13}], E=22,
                           deflection_limit_ratio=ratio)
                self.assertEqual(r["delta_max_m"], 1/81)
                self.assertEqual(r["delta_mid_m"], 23/(96*22))
                self.assertEqual(r["deflection_ok"], admissible)

    def test_registered_fixed_alias_without_optional_symbolic_engine(self):
        with patch.object(dv, "HAS_SYMPY", False):
            r = dv.solve_domain_problem("beam_fixed_fixed", {"span":1,"udl":384,"E":1,"I":1})
            self.assertEqual((r["delta_max_m"],r["delta_mid_m"]),(1,1))

    def test_joint_material_and_section_rounding_at_service_limit(self):
        # The peaks round to (or adjacent to) the displayed limit. Both
        # independent input fields still determine the physical design decision.
        a = 2**-28
        b = 2**-27
        for modulus, inertia in ((1+a, 1-a), (1., 1.), (1+b, 1-b+2**-53)):
            exact = Fraction(1)/(Fraction(modulus)*Fraction(inertia))
            with self.subTest(modulus=modulus, inertia=inertia):
                r = design("fixed_fixed", udl=384, E=modulus, I=inertia, deflection_limit_ratio=1)
                self.assertEqual(r["delta_max_m"], float(exact))
                self.assertEqual(r["deflection_ok"], exact <= 1)
