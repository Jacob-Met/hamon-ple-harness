"""FHWA NHI-06-088 Table 7-4 and Example 7-3 at the ordinary PLE router."""
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route

TABLE = [(0, 0), (.008, .10), (.031, .20), (.071, .30), (.126, .40),
         (.197, .50), (.287, .60), (.403, .70), (.567, .80), (.848, .90),
         (1.0, .931), (1.163, .95), (1.5, .98)]
FT = 0.3048
DAY = 86400
PSF = 47.88025898033584


def example_spec(**change):
    spec = {"calculation":"consolidation_1d", "H_clay_m":10*FT, "Cc":.5, "e0":1,
            "sigma_v0_eff_Pa":600*PSF, "delta_sigma_v_Pa":2400*PSF,
            "cv_m2_s":.2*FT**2/DAY, "time_s":424*DAY, "drainage":"single"}
    spec.update(change)
    return spec


def example(**change):
    return route({"mode":"civil_geotech", "spec":example_spec(**change)})


class TestConsolidation(unittest.TestCase):
    def test_explicit_calculation_routes_without_a_footing_or_outer_mode(self):
        explicit = example()
        self.assertTrue(explicit["conservation_verified"])
        self.assertEqual(route({"spec": example_spec()}), explicit)
        self.assertEqual(route(example_spec()), explicit)

    def test_fhwa_worked_example_7_3(self):
        r = example()
        self.assertTrue(r["conservation_verified"])
        self.assertAlmostEqual(r["primary_consolidation_settlement_m"], .53, delta=.005)
        self.assertAlmostEqual(r["degree_consolidation"], .90, delta=.001)
        self.assertAlmostEqual(r["time_factor"], .848)
        self.assertAlmostEqual(r["settlement_at_time_m"], .9*.53, delta=.005)

    def test_fhwa_table7_4_and_drainage_path(self):
        discrepancies = []
        for tv, expected in TABLE:
            with self.subTest(time_factor=tv):
                single = example(time_s=tv*500*DAY)
                double = example(time_s=tv*125*DAY, drainage="double")
                for r in (single,double):
                    self.assertTrue(r["conservation_verified"])
                    if abs(r["degree_consolidation"] - expected) > .003:
                        discrepancies.append((tv, r["drainage_path_m"]))
                    self.assertLess(r["dual_solver_abs_residual"],1e-6)
                self.assertAlmostEqual(single["degree_consolidation"], double["degree_consolidation"], places=13)
                self.assertAlmostEqual(double["drainage_path_m"], 5*FT)

        # NHI-06-088 Table 7-4 reports 95% at 1.163. Preserve the original
        # comparison for every row and both drainage paths. That row misses
        # the printed percentage by more than 0.3 percentage points.
        self.assertEqual(discrepancies, [(1.163, 10*FT), (1.163, 5*FT)])

    def test_95_percent_factor_and_printed_table_discrepancy(self):
        # Illinois DOT, I-74 over IL Route 17 SGR, Boring 4: T95=1.129.
        self.assertAlmostEqual(example(time_s=1.129*500*DAY)["degree_consolidation"], .95, delta=.00001)
        # For uniform initial excess pressure, modes after the first only
        # reduce U. Bound them by their first coefficient and a geometric
        # progression using the minimum gap between successive odd squares.
        a = math.pi**2 * 1.163 / 4
        first = 1 - 8/math.pi**2 * math.exp(-a)
        tail_bound = 8/(9*math.pi**2) * math.exp(-9*a) / (1 - math.exp(-16*a))
        result = example(time_s=1.163*500*DAY)
        self.assertGreaterEqual(result["degree_consolidation"], first-tail_bound-1e-15)
        self.assertLessEqual(result["degree_consolidation"], first+1e-15)
        self.assertAlmostEqual(result["degree_consolidation_cn"], first, delta=1e-6)

    def test_overconsolidated_reload_and_crossing(self):
        for load, expected_strain in ((50, .1*math.log10(1.5)/2), (200, (.1*math.log10(2)+.5*math.log10(1.5))/2)):
            with self.subTest(load=load):
                r=example(H_clay_m=1.0, sigma_v0_eff_Pa=100., sigma_preconsolidation_Pa=200.,
                          delta_sigma_v_Pa=load, Cr=.1)
                self.assertTrue(r["conservation_verified"])
                self.assertAlmostEqual(r["primary_consolidation_settlement_m"], expected_strain)
        self.assertEqual(example(delta_sigma_v_Pa=0)["settlement_at_time_m"],0)

    def test_solid_volume_limits_all_primary_settlement_routes(self):
        base = dict(H_clay_m=2., e0=.5, Cc=1.2, sigma_v0_eff_Pa=10000.,
                    delta_sigma_v_Pa=990000., cv_m2_s=1e-7, time_s=8000000., drainage="double")
        footing = dict(footing_B_m=1., depth_Df_m=1., gamma_N_m3=18000.)
        for options in ({"calculation": "consolidation_1d"}, footing):
            result = route({"mode": "civil_geotech", "spec": {**base, **options}})
            self.assertFalse(result["conservation_verified"])
            self.assertIn("negative final void ratio", result["reason"])
            for cc, final_void in ((.20, .10), (.25, 0.)):
                result = route({"mode": "civil_geotech", "spec": {**base, **options, "Cc": cc}})
                self.assertTrue(result["conservation_verified"])
                self.assertAlmostEqual(result["final_void_ratio"], final_void)
                self.assertAlmostEqual(result["initial_void_height_m"], 2./3.)
                self.assertAlmostEqual(result["primary_consolidation_settlement_m"],
                                       2. * (.5-final_void)/1.5)

    def test_impossible_rounded_claims_do_not_pass_the_residual_tolerance(self):
        self.assertFalse(example(time_s=0, proposed_settlement_m=-1e-8)["conservation_verified"])
        self.assertFalse(example(time_s=0, proposed_degree_consolidation=-1e-8)["conservation_verified"])
        self.assertFalse(example(time_s=1e16, proposed_degree_consolidation=1.+1e-8)["conservation_verified"])

    def test_dual_domain_legacy_stress_history_and_solid_volume(self):
        from hamon_dual_verifier import run_dual_verify_spec

        spec = dict(H_m=2., e0=.5, Cc=1.2, Cr=.1, sigma_v0_Pa=10000.,
                    sigma_p_Pa=10000., delta_sigma_Pa=990000., max_settlement_m=4.)
        def verify(**updates):
            return run_dual_verify_spec({"mode":"domain","kind":"consolidation_settlement",
                                         "spec":{**spec, **updates}})

        impossible = verify()
        self.assertFalse(impossible["dual_verified"])
        self.assertFalse(impossible["passes_limit"])
        self.assertIn("negative final void ratio", impossible["reason"])
        for prior in (0., 5000., 10000.):
            result = verify(sigma_p_Pa=prior, delta_sigma_Pa=10000., Cc=.25, Cr=.05)
            self.assertTrue(result["dual_verified"])
            self.assertAlmostEqual(result["settlement_m"], 2./1.5*.25*math.log10(2.))
        unloaded = verify(sigma_p_Pa=20000., delta_sigma_Pa=-5000., Cc=.25, Cr=.05)
        self.assertTrue(unloaded["dual_verified"])
        self.assertAlmostEqual(unloaded["settlement_m"], 2./1.5*.05*math.log10(.5))
        self.assertEqual(verify(Cc=0.,Cr=0.)["settlement_m"], 0.)

    def test_claimed_results_and_domain(self):
        self.assertFalse(example(proposed_degree_consolidation=.5)["conservation_verified"])
        self.assertFalse(example(proposed_settlement_m=.4)["conservation_verified"])
        for invalid in ({"cv_m2_s":0},{"time_s":-1},{"drainage":"other"},{"e0":False},
                        {"sigma_preconsolidation_Pa":-1},{"sigma_preconsolidation_Pa":1e8},
                        {"Cr":2},{"time_s":float("nan")}):
            with self.subTest(invalid=invalid):
                self.assertFalse(example(**invalid)["conservation_verified"])


if __name__ == "__main__":
    unittest.main()
