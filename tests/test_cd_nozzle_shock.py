"""Published NASA Glenn CDV benchmark plus independent prescribed-shock forward cases."""
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import hamon_pareto as router

ROOT = Path(__file__).resolve().parents[1]
BASE = {"mode": "compressible_acoustics", "spec": {
    "subkind": "cd_nozzle_shock", "gamma": 1.4, "exit_to_throat_area": 1.5,
    "back_to_total_pressure": .75}}
EMPTY = dict(schema="hamon.ple.intake.v1", scanned_files=0, engineering_file_count=0,
             by_extension={}, files=[], warnings=[])


def case(**kwargs):
    return {**BASE, "spec": {**BASE["spec"], **kwargs}}


def solve(**kwargs):
    return router.dispatch_ple_tool("hamon_conservation_cad_verifier", case(**kwargs))


def plan(text, inventory=EMPTY, **kwargs):
    return router.route_from_intake_prompt(json.dumps(inventory), instruction=text, **kwargs)


class CDNozzleShock(unittest.TestCase):
    def test_nasa_cdv_spatial_and_shock_state_registered_cli(self):
        run = subprocess.run([sys.executable, str(ROOT / "hamon_conservation_cad_verifier.py"),
                              "--json-stdin"], input=json.dumps(BASE), text=True,
                             capture_output=True, check=True)
        r = json.loads(run.stdout)
        self.assertTrue(r["verified"], r)
        # NASA's tabulated stations sit just to either side of the jump.
        x = 5*(1 + math.acos(4*(1.25-r["shock_area_over_throat"]))/math.pi)
        self.assertLess(7.562273, x)
        self.assertLess(x, 7.562293)
        for key, target in (("upstream_mach", 1.611728), ("downstream_mach", .6649400),
                            ("upstream_p_over_reservoir", .2312145),
                            ("downstream_p_over_reservoir", .6621843),
                            ("exit_mach", .5019145), ("exit_p_over_reservoir", .75)):
            self.assertAlmostEqual(r[key], target, delta=3e-6, msg=key)
        self.assertLess(r["residual"], 1e-14)
        print("NASA CDV internal shock: " + json.dumps({"x_in": x, **r}, sort_keys=True))

    def test_nasa_cdv_shock_free_controls(self):
        for back, side in ((.89, "high"), (.16, "low")):
            r = solve(back_to_total_pressure=back)
            self.assertFalse(r["verified"])
            self.assertEqual(r["regime"], "no_internal_shock_" + side + "_back_pressure")
            self.assertNotIn("shock_area_over_throat", r)

    def test_unrepresentable_mach_abstains(self):
        # With an extreme but finite supplied gamma, loss can remain above
        # the target throughout the entire representable supersonic branch.
        r = solve(gamma=1e100, back_to_total_pressure=.25)
        self.assertFalse(r["verified"])
        self.assertEqual(r["regime"], "unrepresentable_state")

    def test_independent_forward_prescribed_shock(self):
        for g in (1.2, 1.4, 5/3):
            a = (g-1)/2
            for u in (1.02, 1.35, 2., 4.):
                for expansion in (1.01, 1.6, 4):
                    with self.subTest(g=g, u=u, expansion=expansion):
                        def flux(m):
                            return m*(1+a*m*m)**(-(g+1)/(2*(g-1)))
                        ar = flux(1)/flux(u)
                        rho = (g+1)*u*u/((g-1)*u*u+2)
                        pj = 1+2*g*(u*u-1)/(g+1)
                        d = u/math.sqrt(rho*pj)
                        pd = pj*(1+a*u*u)**(-g/(g-1))
                        # March from the prescribed shock through the divergent
                        # subsonic outlet, conserving mass at both stations.
                        lo, hi = 0., d
                        for _ in range(65):
                            mid = (lo+hi)/2
                            if flux(mid)*expansion < flux(d):
                                lo = mid
                            else:
                                hi = mid
                        e = (lo+hi)/2
                        back = pd*((1+a*d*d)/(1+a*e*e))**(g/(g-1))
                        r = solve(gamma=g, back_to_total_pressure=back, exit_to_throat_area=ar*expansion)
                        self.assertTrue(r["verified"], r)
                        for key, expected in (("shock_area_over_throat", ar), ("upstream_mach", u),
                                              ("downstream_mach", d), ("exit_mach", e)):
                            self.assertAlmostEqual(r[key]/expected, 1, delta=4e-10)
                        self.assertLess(r["residual"], 1e-12)

    def test_complete_instruction_presolve_and_obligations(self):
        raw = json.dumps(BASE)
        p = plan("Run the calculation below and return its JSON result.\n" + raw)
        self.assertTrue(p["deliver_verified_json"])
        self.assertEqual(p["thinking_tier"], "low")
        self.assertEqual(json.loads(p["presolve_result_json"]), solve())
        for text in (raw + "\nMake a report.", json.dumps(case(back_to_total_pressure=.89)),
                     json.dumps(case(back_to_total_pressure=.16)), json.dumps(case(extra=3)),
                     raw.replace("0.75", "0.75000000000000001"),
                     raw.replace('"gamma":', '"gamma": 1.1, "gamma":')):
            self.assertFalse(plan(text)["presolve_verified"])
        self.assertFalse(plan(raw, mcp_server_count=1)["presolve_verified"])
        self.assertFalse(plan(raw, deliverable_count=1)["presolve_verified"])

    def test_tampered_receiver_output_abstains(self):
        good = solve()
        for key in ("upstream_mach", "downstream_p_over_reservoir", "downstream_mach",
                    "shock_area_over_throat", "exit_mach", "exit_p_over_reservoir",
                    "downstream_total_over_reservoir", "upstream_p_over_reservoir"):
            with self.subTest(key=key), patch.object(router, "dispatch_ple_tool",
                                                     return_value={**good, key: 1.003*good[key]}):
                self.assertFalse(plan(json.dumps(BASE))["presolve_verified"])
