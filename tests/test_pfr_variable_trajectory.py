"""Independent flow-balance, closed-form and registered-CLI PFR controls."""
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from mpmath import mp
from scipy.integrate import solve_ivp
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route


def solve(**params):
    return route({"mode": "reactor_kinetics", "spec": {"reactor_type": "pfr", **params}})


class VariableRatePfr(unittest.TestCase):
    def test_heated_gas_trace_matches_independent_lambert_solution(self):
        ctx = mp.clone(); ctx.dps = 100
        for volume in (1e-26, .2, 5., 120., 1200.):
            with self.subTest(volume=volume):
                # V=2*(-log U)-1+U, hence U=-2*W0(-exp(-(V+1)/2)/2).
                u = -2*ctx.lambertw(-ctx.exp(-(ctx.mpf(volume)+1)/2)/2)
                r = solve(k=1., gas_phase=True, T0=300., delta_T_adiabatic=300., volume=volume)
                self.assertTrue(r["verified"], r)
                self.assertAlmostEqual(r["unreacted_fraction"]/float(u), 1., delta=2e-14)
                self.assertAlmostEqual(r["conversion_X"]/float(1-u), 1., delta=2e-14)
                self.assertAlmostEqual(r["exit_T_K"]/float(600-300*u), 1., delta=2e-15)

    def test_arrhenius_tiny_dose_remains_nonzero(self):
        for volume in (1e-26, 1e-300):
            with self.subTest(volume=volume):
                r = solve(k_ref=1., T_ref=300., E_a=5000., T0=300.,
                          delta_T_adiabatic=35., volume=volume)
                # Initial dX/dV=1; the smooth correction is O(V**2).
                self.assertTrue(r["verified"], r)
                self.assertAlmostEqual(r["conversion_X"]/volume, 1., delta=1e-14)

    def test_arrhenius_sizing_matches_forward_initial_value_problem(self):
        for gas, rise, expansion in ((False, 35., 0.), (True, 80., .8), (True, -350., -2.)):
            with self.subTest(gas=gas, rise=rise):
                p = dict(k_ref=1., T_ref=300., E_a=5000., T0=300.,
                         delta_T_adiabatic=rise, epsilon_A=expansion, gas_phase=gas)
                r = solve(conversion=.25, **p)
                def ode(v, state):
                    x = state[0]; temperature = 300+rise*x
                    k = math.exp(-5000/8.31446261815324*(1/temperature-1/300))
                    c = (1-x)/(1+expansion*x)/(temperature/300 if gas else 1)
                    return [k*c]
                reference = solve_ivp(ode, (0, r["volume_m3"]), [0.], method="DOP853", rtol=2e-12, atol=2e-14)
                self.assertTrue(reference.success)
                self.assertTrue(r["verified"], r)
                self.assertAlmostEqual(reference.y[0,-1], .25, delta=2e-11)
                inverse = solve(volume=r["volume_m3"], **p)
                self.assertTrue(inverse["verified"], inverse)
                self.assertAlmostEqual(inverse["conversion_X"], .25, delta=2e-14)

    def test_fractional_order_finite_depletion_is_not_an_infinite_tail(self):
        p = dict(k=1., order=.5, gas_phase=True, T0=300., delta_T_adiabatic=300.)
        # Integral_0^1 sqrt((1+X)/(1-X)) dX = 1 + pi/2.
        r = solve(conversion=1., **p)
        self.assertTrue(r["verified"], r)
        self.assertAlmostEqual(r["volume_m3"], 1+math.pi/2, delta=2e-14)
        plateau = solve(volume=3., **p)
        self.assertTrue(plateau["verified"], plateau)
        self.assertEqual(plateau["unreacted_fraction"], 0.)

    def test_nonpositive_order_cannot_continue_after_depletion(self):
        p = dict(k=1., order=0., gas_phase=True, T0=300., delta_T_adiabatic=300.)
        self.assertTrue(solve(volume=1., **p)["verified"])
        with self.assertRaisesRegex(ValueError, "depletion"):
            solve(volume=math.nextafter(1., math.inf), **p)

    def test_unrepresentable_and_nonphysical_outlets_refuse(self):
        p = dict(k=1., gas_phase=True, T0=300., delta_T_adiabatic=300.)
        with self.assertRaisesRegex(ValueError, "representable"):
            solve(volume=1500., **p)
        for values in (dict(conversion=.75, delta_T_adiabatic=-400.),
                       dict(conversion=.5, epsilon_A=-2.)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                solve(k_ref=1., T_ref=300., E_a=5000., T0=300., **values)

    def test_registered_json_cli_preserves_the_complement(self):
        payload = {"mode":"reactor_kinetics", "spec":{"reactor_type":"pfr", "k":1.,
                   "gas_phase":True, "T0":300., "delta_T_adiabatic":300., "volume":120.}}
        root = Path(__file__).resolve().parents[1]
        p = subprocess.run([sys.executable, str(root/"hamon_conservation_cad_verifier.py"), "--json-stdin"],
                           input=json.dumps(payload), text=True, capture_output=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        r = json.loads(p.stdout)
        self.assertTrue(r["verified"], r)
        self.assertEqual(r["conversion_X"], 1.)
        self.assertGreater(r["unreacted_fraction"], 0.)


if __name__ == "__main__":
    unittest.main()
