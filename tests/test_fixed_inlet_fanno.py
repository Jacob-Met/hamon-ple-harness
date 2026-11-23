import math
import unittest

from mpmath import mp
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def run(m1, length, gamma=1.4, friction=0.01, diameter=0.1, **conditions):
    return run_router_conservation_cad_verifier({
        "mode": "compressible_acoustics",
        "spec": {"subkind": "fanno", "M1": m1, "L": length, "gamma": gamma,
                 "f": friction, "D_h": diameter, **conditions}
    })


def length_between(m1, m2, gamma=1.4, friction=0.01, diameter=0.1):
    # Integrate the differential momentum/energy equation directly in Mach.
    ctx = mp.clone()
    ctx.dps = 75
    g, f, d, m1, m2 = map(ctx.mpf, (gamma, friction, diameter, m1, m2))
    def gradient(m):
        return 4 * (m-1) * (m+1) / (g * m**3 * (2 + (g-1)*m*m))
    scale = max(1, m1, m2)**2
    return float(d/(4*f*scale) * ctx.quad(lambda m: scale*gradient(m), [m2, m1]))


class FixedInletFannoTests(unittest.TestCase):
    def test_sonic_section_upstream_of_requested_supersonic_exit(self):
        critical = length_between(2., 1.)
        result = run(2., 2 * critical)
        self.assertFalse(result["verified"])
        self.assertFalse(result["requested_state_feasible"])
        self.assertTrue(result["choked"])
        self.assertNotIn("M2", result)
        self.assertNotIn("P2_over_P1", result)
        self.assertNotIn("T2_over_T1", result)
        self.assertEqual(result["reachable_sonic_state"]["M"], 1.)
        self.assertAlmostEqual(result["L_choke_m"], critical)
        self.assertAlmostEqual(result["residual"], 4*.01*critical/.1)

    def test_both_friction_branches_replay_mass_and_energy(self):
        for m1, wanted, g in ((.4, .7, 1.3), (4., 2., 1.67),
                               (1.000001, 1.0000004, 1.4), (1e100, 4e99, 1.4)):
            with self.subTest(m1=m1):
                L = length_between(m1, wanted, g)
                result = run(m1, L, gamma=g)
                self.assertTrue(result["verified"], result)
                self.assertFalse(result["choked"])
                self.assertTrue(math.isclose(result["M2"], wanted, rel_tol=3e-15))
                m2, temp, press = (result[key] for key in ("M2", "T2_over_T1", "P2_over_P1"))
                self.assertTrue(math.isclose(press * m2 / math.sqrt(temp), m1, rel_tol=3e-15))
                self.assertTrue(math.isclose(temp * (1 + (g-1)*m2*m2/2),
                                             1 + (g-1)*m1*m1/2, rel_tol=3e-15))
                self.assertTrue(math.isclose(length_between(m1, m2, g), L, rel_tol=2e-9))

    def test_exact_sonic_endpoint_and_distinct_adjacent_lengths(self):
        for m1 in (.65, 2.5):
            L = length_between(m1, 1.)
            previous = run(m1, math.nextafter(L, 0.))
            self.assertTrue(previous["verified"], previous)
            self.assertFalse(previous["choked"])
            self.assertNotEqual(previous["M2"], 1.)
            endpoint = run(m1, L)
            self.assertTrue(endpoint["verified"])
            self.assertTrue(endpoint["choked"])
            self.assertEqual(endpoint["M2"], 1.)
            beyond = run(m1, math.nextafter(L, math.inf))
            self.assertFalse(beyond["verified"])
            self.assertFalse(beyond["requested_state_feasible"])
        self.assertTrue(run(1., 0.)["verified"])
        self.assertFalse(run(1., 1e-20)["requested_state_feasible"])

    def test_fanning_and_darcy_conventions_and_fixed_gas(self):
        reference = run(2., .2)
        darcy = run(2., .2, friction=.04, friction_factor_convention="Darcy")
        for name in ("M2", "P2_over_P1", "T2_over_T1", "L_choke_m"):
            self.assertEqual(reference[name], darcy[name])
        consistent = run(2., .2, R=287., cp=1004.5)
        self.assertTrue(consistent["verified"])
        self.assertTrue(math.isclose(consistent["M2"], reference["M2"], rel_tol=1e-15))
        for bad in ({"R":287., "cp":500.}, {"branch":"subsonic"},
                    {"friction_factor_convention":"unspecified"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run(2., .2, **bad)
