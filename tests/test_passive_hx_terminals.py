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

    def test_rating_and_sizing_replay_both_arrangements(self):
        for arrangement in ("parallel", "counterflow"):
            with self.subTest(arrangement=arrangement):
                base = dict(cp_hot_J_kgK=800., cp_cold_J_kgK=1250.)
                rated = rating(345., arrangement, **base)
                self.assertTrue(rated["dual_verified"], rated)
                inverse = sizing(rated["Th_out_K"], arrangement, **base)
                self.assertTrue(inverse["dual_verified"], inverse)
                self.assertAlmostEqual(inverse["area_m2"], 3.45, places=13)
                self.assertAlmostEqual(inverse["Tc_out_K"], rated["Tc_out_K"])
                entropy = 800*math.log(rated["Th_out_K"]/350)+1250*math.log(rated["Tc_out_K"]/290)
                self.assertGreater(entropy,0)
                self.assertAlmostEqual(rated["entropy_generation_W_K"], entropy)
                self.assertAlmostEqual(inverse["entropy_generation_W_K"], entropy)

    def test_isothermal_and_disconnected_limits(self):
        for arr in ("counterflow", "parallel", "shell_and_tube_1_2"):
            for ua in (0., 250., 1500.):
                r = rating(ua,arr,Tc_in_K=350.)
                self.assertTrue(r["dual_verified"], r)
                self.assertEqual(r["Q_duty_W"],0.)
                self.assertEqual(r["entropy_generation_W_K"],0.)
        sized = sizing(350., Tc_in_K=350.)
        self.assertTrue(sized["thermal_state_feasible"])
        self.assertFalse(sized["area_identifiable"])
        self.assertFalse(sized["dual_verified"])
        self.assertNotIn("area_m2", sized)
        self.assertEqual(sizing(350.)["area_m2"], 0.)
        # At this parallel outlet, the terminal approach is exactly zero.
        self.assertFalse(sizing(320., "parallel")["dual_verified"])

    def test_reverse_named_streams_and_resolved_small_change(self):
        r = sizing(335., Th_in_K=290., Tc_in_K=350.)
        self.assertTrue(r["dual_verified"],r)
        self.assertLess(r["Q_duty_W"],0.)
        self.assertGreater(r["entropy_generation_W_K"],0.)
        self.assertAlmostEqual(r["area_m2"],30.)
        cold = rating(1.,cp_cold_J_kgK=1e20)
        self.assertTrue(cold["dual_verified"],cold)
        self.assertEqual(cold["Tc_out_K"],290.)
        self.assertGreater(cold["cold_temperature_rise_K"],0.)
        self.assertAlmostEqual(1e20*cold["cold_temperature_rise_K"],cold["Q_duty_W"])

    def test_passive_inputs_and_finite_rating_at_a_rounded_pinch(self):
        for bad in (dict(cp_hot_J_kgK=-1000.,m_hot_kg_s=-1.), dict(Tc_in_K=-1.),
                    dict(m_cold_kg_s=0.), dict(U_W_m2K=0.)):
            with self.subTest(bad=bad):
                self.assertFalse(sizing(335., **bad)["dual_verified"])
        for arr in ("counterflow", "parallel"):
            near = rating(1e7,arr)
            self.assertTrue(near["dual_verified"],near)
            self.assertGreater(near["entropy_generation_W_K"],0.)

    def test_positive_entropy_survives_nearly_reversible_exchange(self):
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec = 400
            for hi, ci, ua in ((1e40,1.,1e180), (1e60,1.,1e180), (1e100,1.,1e180),
                               (290.,350.,1e240), (350.,290.,1e240)):
                with self.subTest(hi=hi,ci=ci,ua=ua):
                    a,b,n = map(Decimal.from_float,(hi,ci,ua))
                    gap_product = n*(a-b)**2/(a*b*(1+n)**2)
                    expected = float((1+gap_product).ln())
                    r = rating(ua,Th_in_K=hi,Tc_in_K=ci,cp_hot_J_kgK=1.,cp_cold_J_kgK=1.)
                    self.assertTrue(r["dual_verified"],r)
                    self.assertEqual(r["entropy_generation_W_K"],expected)
