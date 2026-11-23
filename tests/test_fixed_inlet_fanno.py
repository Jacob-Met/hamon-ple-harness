import math
import unittest

from mpmath import mp
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def run(m1, length, gamma=1.4, friction=0.01, diameter=0.1):
    return run_router_conservation_cad_verifier({
        "mode": "compressible_acoustics",
        "spec": {"subkind": "fanno", "M1": m1, "L": length, "gamma": gamma,
                 "f": friction, "D_h": diameter}
    })


def length_between(m1, m2, gamma=1.4, friction=0.01, diameter=0.1):
    # Integrate the differential momentum/energy equation directly in Mach.
    ctx = mp.clone()
    ctx.dps = 75
    g, f, d, m1, m2 = map(ctx.mpf, (gamma, friction, diameter, m1, m2))
    def gradient(m):
        return 4 * (m-1) * (m+1) / (g * m**3 * (2 + (g-1)*m*m))
    return float(d/(4*f) * ctx.quad(gradient, [m2, m1]))


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
