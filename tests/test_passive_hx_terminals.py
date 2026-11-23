import math
import unittest

from hamon_dual_verifier import solve_domain_problem


BASE = dict(m_hot_kg_s=1., cp_hot_J_kgK=1000., Th_in_K=350.,
            m_cold_kg_s=1., cp_cold_J_kgK=1000., Tc_in_K=290.)


def sizing(outlet, arrangement="counterflow", **changes):
    spec = dict(BASE, Th_out_K=outlet, U_W_m2K=100., arrangement=arrangement)
    return solve_domain_problem("heat_exchanger_lmtd", dict(spec, **changes))


def rating(ua, arrangement="counterflow", **changes):
    return solve_domain_problem("heat_exchanger_ntu", dict(dict(BASE, UA_W_K=ua, arrangement=arrangement), **changes))


class PassiveHXTests(unittest.TestCase):
    def test_global_energy_closure_does_not_qualify_a_negative_area(self):
        r = sizing(270.)
        self.assertFalse(r["dual_verified"])
        self.assertNotIn("area_m2", r)

    def test_outlet_order_depends_on_arrangement(self):
        valid = sizing(305.)
        self.assertTrue(valid["dual_verified"])
        self.assertAlmostEqual(valid["area_m2"], 30.)
        self.assertFalse(sizing(305., "parallel")["dual_verified"])

    def test_negative_ua_cannot_refrigerate_from_cold_to_hot(self):
        self.assertFalse(rating(-250.)["dual_verified"])
