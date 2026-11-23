"""Regenerative Brayton limits through the registered conservation/CAD callback."""
import math
import unittest
from decimal import Decimal, localcontext

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(**changes):
    spec = dict(subkind="brayton", T1=300., T3=1200., gamma=2., r_p=4., cp=1000.,
                eta_c=1., eta_t=1., regenerator_eff=1.)
    spec.update(changes)
    return run_router_conservation_cad_verifier({"mode": "thermo_cycles", "spec": spec})


class BraytonLimits(unittest.TestCase):
    def assert_relative(self, actual, expected):
        self.assertTrue(math.isclose(actual, float(expected), rel_tol=3e-14, abs_tol=0.),
                        (actual, expected))

    def test_regenerator_zero_driving_temperature(self):
        for eff in (0., .25, .75, 1.):
            with self.subTest(eff=eff):
                r = solve(regenerator_eff=eff)
                self.assertTrue(r["verified"])
                self.assertEqual([r[k] for k in ("T2_K","T2x_K","T4_K","T4y_K")], [600.]*4)
                self.assertEqual(r["q_regenerated_J_kg"], 0.)
                self.assertEqual(r["w_net_J_kg"], 300000.)
                self.assertEqual(r["eta_thermal"], .5)
                self.assertEqual(r["s_gen_J_kgK"], 0.)

    def test_zero_driving_temperature_with_irrational_station_temperature(self):
        for e in (0., .5, 1.):
            r = solve(r_p=3., T3=900., regenerator_eff=e)
            self.assertTrue(r["verified"])
            self.assertEqual(r["q_regenerated_J_kg"], 0.)
            self.assert_relative(r["T2_K"], 300*math.sqrt(3))
            self.assert_relative(r["eta_thermal"], 1-1/math.sqrt(3))

    def test_irreversible_equality_and_neighboring_directions(self):
        kwargs = dict(eta_t=.75, T3=960.)
        result = solve(**kwargs)
        self.assertTrue(result["verified"])
        self.assertEqual(result["q_regenerated_J_kg"], 0.)
        self.assertEqual(result["w_net_J_kg"], 60000.)
        self.assert_relative(result["eta_thermal"], 1/6)
        self.assertGreater(result["s_gen_J_kgK"], 0.)
        lower = math.nextafter(960., 0.)
        higher = math.nextafter(960., math.inf)
        self.assertEqual(solve(eta_t=.75,T3=lower)["error"], "regenerator_temperature_crossover")
        self.assertTrue(solve(eta_t=.75,T3=lower,regenerator_eff=0.)["verified"])
        accepted = solve(eta_t=.75, T3=higher)
        self.assertTrue(accepted["verified"])
        self.assertGreater(accepted["q_regenerated_J_kg"], 0.)

    def test_ideal_weak_compression_positive_work_and_efficiency(self):
        with localcontext() as ctx:
            ctx.prec = 100
            for ratio in (math.nextafter(1., math.inf), 1+2**-49, 1+1e-7):
                g = Decimal.from_float(1.4)
                tau = (Decimal.from_float(ratio).ln()*(g-1)/g).exp()
                wc = 300000*(tau-1)
                wt = 1200000*(1-1/tau)
                for eff in (0., .5, 1.):
                    with self.subTest(ratio=ratio,eff=eff):
                        r = solve(gamma=1.4, r_p=ratio, regenerator_eff=eff)
                        self.assertTrue(r["verified"])
                        self.assert_relative(r["w_compressor_J_kg"], wc)
                        self.assert_relative(r["w_turbine_J_kg"], wt)
                        self.assert_relative(r["w_net_J_kg"], wt-wc)
                        self.assert_relative(r["compressor_temperature_rise_K"], wc/1000)
                        self.assert_relative(r["turbine_temperature_drop_K"], wt/1000)
                        self.assertGreater(r["eta_thermal"], 0.)
                        if eff in (0.,1.):
                            self.assertEqual(r["s_gen_J_kgK"], 0.)
                        else:
                            self.assertGreater(r["s_gen_J_kgK"], 0.)

    def test_work_boundary_and_positive_specific_heat_scaling(self):
        self.assertEqual(solve(T3=600.,regenerator_eff=0.)["error"],"non_positive_net_work")
        self.assertEqual(solve(T3=math.nextafter(600.,0.),regenerator_eff=0.)["error"],"non_positive_net_work")
        for c in (2.**-50,1000.,2.**100):
            hot = math.nextafter(600.,math.inf)
            r = solve(T3=hot,cp=c,regenerator_eff=0.)
            self.assertTrue(r["verified"])
            self.assert_relative(r["w_net_J_kg"], (hot-600)*c/2)
            self.assertEqual(r["eta_thermal"],.5)

    def test_outside_supported_domain(self):
        for params in ({"cp":math.inf},{"r_p":1.},{"gamma":1.},{"T1":0.},{"eta_t":0.},{"regenerator_eff":1.1}):
            self.assertFalse(solve(**params)["verified"])
        self.assertFalse(solve(cp=1e308)["verified"])
