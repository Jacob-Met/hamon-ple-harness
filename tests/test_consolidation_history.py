"""Registered stage-history checks with independent load, diffusion and sign oracles."""
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(history, time, **changes):
    spec = dict(calculation="consolidation_history_1d", H_clay_m=1., cv_m2_s=1.,
                mv_per_Pa=1e-7, drainage="single", time_s=time,
                load_history=[dict(time_s=t, surcharge_Pa=q) for t, q in history])
    return run_router_conservation_cad_verifier({"mode": "auto", "spec": {**spec, **changes}})


class SettlementHistoryTests(unittest.TestCase):
    def test_step_recovers_both_drainage_paths_and_equilibrium(self):
        for drainage, tv in (("single", .25), ("double", 1.)):
            r = solve([(0, 10000)], .25, drainage=drainage)
            self.assertTrue(r["conservation_verified"], r)
            p = sum(8/math.pi**2/k**2 * math.exp(-k*k*math.pi**2*tv/4)
                    for k in range(1, 31, 2)) * 10000
            self.assertAlmostEqual(r["mean_excess_pore_pressure_Pa"], p)
            self.assertAlmostEqual(r["settlement_at_time_m"], (10000-p)*1e-7)
            self.assertLess(r["dual_solver_abs_residual_Pa"], .001)

    def test_linear_construction_ramp_and_drainage(self):
        hist = [(0, 0), (1, 10000)]
        at = solve(hist, 1.)
        self.assertTrue(at["conservation_verified"], at)
        # Integral of the pressure kernel over 0<T<1. The sum of its
        # infinite-time integrals is exactly 1/3 in drainage time units.
        residual = math.fsum(32/math.pi**4/k**4*math.exp(-k*k*math.pi**2/4)
                             for k in range(1, 31, 2))
        self.assertAlmostEqual(at["mean_excess_pore_pressure_Pa"], 10000*(1/3-residual))
        early = solve(hist, 1e-3)
        self.assertTrue(early["conservation_verified"], early)
        self.assertAlmostEqual(early["mean_effective_increment_Pa"],
                               (4/3)*math.sqrt(.001/math.pi)*10, places=12)
        late = solve(hist, 100.)
        self.assertAlmostEqual(late["settlement_at_time_m"], .001)
        self.assertGreater(late["mean_excess_pore_pressure_Pa"], 0.)

    def test_pulse_rebound_delayed_load_and_equivalent_subdivision(self):
        pulse = [(0, 10000), (1, 10000), (1, 0)]
        before = solve([(0,10000)], 1)
        after = solve(pulse, 1)
        self.assertTrue(after["conservation_verified"], after)
        self.assertAlmostEqual(before["settlement_at_time_m"], after["settlement_at_time_m"])
        self.assertAlmostEqual(before["mean_excess_pore_pressure_Pa"]-10000,
                               after["mean_excess_pore_pressure_Pa"])
        self.assertGreater(after["settlement_at_time_m"], 0.)
        self.assertLess(after["mean_excess_pore_pressure_Pa"], 0.)
        for t in (1.01, 2., 30.):
            r = solve(pulse, t)
            self.assertTrue(r["conservation_verified"], r)
            self.assertGreater(r["settlement_at_time_m"], 0.)
            self.assertLess(r["settlement_at_time_m"], after["settlement_at_time_m"])
            self.assertAlmostEqual(r["mean_excess_pore_pressure_Pa"],
                                   -r["mean_effective_increment_Pa"])
        later = [(0,0), (2,0), (2,10000), (3,10000), (3,0)]
        self.assertEqual(solve(later, 1)["settlement_at_time_m"], 0.)
        self.assertAlmostEqual(solve(later, 4)["settlement_at_time_m"], solve(pulse,2)["settlement_at_time_m"])
        ramp = [(0,0), (.2,3000), (.4,6000), (1,15000)]
        whole = solve([(0,0),(1,15000)], .3)
        self.assertAlmostEqual(solve(ramp, .3)["settlement_at_time_m"], whole["settlement_at_time_m"])

    def test_short_pulse_survives_when_endpoint_ages_round_together(self):
        duration = 1e-20
        pulse = [(0, 10000), (duration, 10000), (duration, 0)]
        self.assertEqual(1-duration, 1.)
        at = solve(pulse, 1)
        self.assertTrue(at["conservation_verified"], at)
        # First-order pulse response q*dt*U'(1); omitted O(dt) relative
        # corrections are orders of magnitude below binary64 precision.
        expected = 20000*duration*sum(math.exp(-(k*math.pi/2)**2)
                                      for k in range(1, 31, 2))
        self.assertTrue(math.isclose(at["mean_effective_increment_Pa"], expected, rel_tol=1e-13), at)
        self.assertTrue(math.isclose(at["mean_excess_pore_pressure_Pa"], -expected, rel_tol=1e-13), at)
        self.assertTrue(math.isclose(at["finite_volume_mean_effective_increment_Pa"], expected, rel_tol=1e-5), at)
        self.assertTrue(math.isclose(at["finite_volume_mean_excess_pore_pressure_Pa"], -expected, rel_tol=1e-5), at)

    def test_domain_claims_and_instantaneous_boundary(self):
        jump = solve([(0,10000)], 0)
        self.assertEqual(jump["settlement_at_time_m"], 0.)
        self.assertEqual(jump["mean_excess_pore_pressure_Pa"], 10000.)
        for history in ([], [(1,0)], [(0,-1)], [(0,0),(2,1),(1,3)], [(0,0),(0,1),(0,2)]):
            self.assertFalse(solve(history, 1)["conservation_verified"])
        for invalid in (dict(mv_per_Pa=0), dict(Cc=.2), dict(delta_sigma_v_Pa=10000),
                        dict(drainage="other"), dict(time_s=-1)):
            self.assertFalse(solve([(0,10000)], 1, **invalid)["conservation_verified"])
        self.assertFalse(solve([(0,10000)], 100., mv_per_Pa=.001)["conservation_verified"])
        for claim in (dict(proposed_settlement_m=-1e-14), dict(proposed_mean_excess_pore_pressure_Pa=0)):
            self.assertFalse(solve([(0,10000)], 1, **claim)["conservation_verified"])
        self.assertFalse(solve([(0,10000)], 1, calculation="consolidation_1d")["conservation_verified"])


if __name__ == "__main__":
    unittest.main()
