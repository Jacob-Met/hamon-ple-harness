"""Replay the serialized shunt-match prescription using independent reflection waves."""
import json
import math
import unittest

from mpmath import mp

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def dispatch(spec):
    return run_router_conservation_cad_verifier({"mode": "rf_transmission_rlc", "spec": spec})


def replay(spec, solution, relative):
    ctx = mp.clone()
    scales = [math.log10(abs(x)) for x in (spec["R_L"], spec.get("X_L", 0), spec["Z0"]) if x]
    ctx.dps = 140 + int(4 * (max(scales) - min(scales)))
    def length(scalar, coordinate):
        if not relative:
            return ctx.mpf(solution[scalar])
        c = solution[coordinate]
        return ctx.mpf(c["quarter_waves"]) / 4 + ctx.mpf(c["offset_wavelengths"])

    d = length("d_over_lambda", "d_coordinate")
    l_stub = length("l_stub_over_lambda", "l_stub_coordinate")
    load = ctx.mpc(spec["R_L"], spec.get("X_L", 0)) / spec["Z0"]
    line_wave = (load - 1) / (load + 1) * ctx.expjpi(-4 * d)
    open_wave = 1 if spec["stub_type"] == "open" else -1
    stub_wave = open_wave * ctx.expjpi(-4 * l_stub)
    if line_wave == -1 or stub_wave == -1:
        return 1.0
    admittance = (1 - line_wave) / (1 + line_wave) + (1 - stub_wave) / (1 + stub_wave)
    return float(abs(1 / admittance - 1)) if admittance else math.inf


class SmithStubCoordinates(unittest.TestCase):
    def solve(self, **spec):
        return json.loads(json.dumps(dispatch({"subkind": "smith_stub", **spec}), allow_nan=False))

    def test_independent_replay_of_both_serialized_branches(self):
        for r in (1e-18, 1e-12, 1e-6, .01, 25., 50., 5000., 1e6, 1e12, 1e18):
            for x in (0., -50., 50.):
                for stub in ("open", "short"):
                    with self.subTest(resistance=r, reactance=x, stub=stub):
                        spec = {"Z0": 50., "R_L": r, "X_L": x, "stub_type": stub}
                        result = self.solve(**spec, length_representation="endpoint_relative")
                        self.assertEqual(len(result["all_branches"]), 1 if r == 50 and x == 0 else 2)
                        for solution in result["all_branches"]:
                            relative = replay(spec, solution, True)
                            plain = replay(spec, solution, False)
                            self.assertLess(relative, 1e-12)
                            self.assertTrue(solution["endpoint_relative_verified"])
                            self.assertEqual(solution["verified"], relative < 1e-10)
                            self.assertEqual(solution["absolute_lengths_verified"], plain < 1e-10)
                            self.assertTrue(math.isclose(solution["absolute_length_residual"], plain,
                                                        rel_tol=1e-10, abs_tol=1e-20))

    def test_old_scalar_and_opt_in_contracts(self):
        for branch in (0, 1):
            with self.subTest(branch=branch):
                spec = dict(Z0=50., R_L=1e-12, X_L=0., stub_type="short", branch=branch)
                plain = self.solve(**spec)
                relative = self.solve(**spec, length_representation="endpoint_relative")
                self.assertEqual(plain["length_representation"], "wavelengths")
                self.assertFalse(plain["verified"])
                self.assertTrue(relative["verified"])
                self.assertGreater(replay(spec, plain, False), 1e-3)
                self.assertLess(replay(spec, relative, True), 1e-12)
                self.assertEqual(relative["d_coordinate"], relative["all_branches"][branch]["d_coordinate"])
                self.assertEqual(plain["l_stub_coordinate"], relative["l_stub_coordinate"])

    def test_rounded_endpoint_preserves_nonzero_offset(self):
        spec = dict(Z0=1e200, R_L=1e-200, X_L=0., stub_type="open")
        result = self.solve(**spec, length_representation="endpoint_relative")
        self.assertEqual(len(result["all_branches"]), 2)
        near_half = result["all_branches"][1]
        self.assertEqual(near_half["d_over_lambda"], 0)
        self.assertEqual(near_half["d_coordinate"]["quarter_waves"], 2)
        self.assertLess(mp.mpf(near_half["d_coordinate"]["offset_wavelengths"]), 0)
        self.assertLess(replay(spec, near_half, True), 1e-12)
        self.assertFalse(near_half["absolute_lengths_verified"])
