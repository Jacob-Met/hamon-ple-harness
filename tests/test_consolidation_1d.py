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


def example(**change):
    spec = {"calculation":"consolidation_1d", "H_clay_m":10*FT, "Cc":.5, "e0":1,
            "sigma_v0_eff_Pa":600*PSF, "delta_sigma_v_Pa":2400*PSF,
            "cv_m2_s":.2*FT**2/DAY, "time_s":424*DAY, "drainage":"single"}
    spec.update(change)
    return route({"mode":"civil_geotech", "spec":spec})


class TestConsolidation(unittest.TestCase):
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
