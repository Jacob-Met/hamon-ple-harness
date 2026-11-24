"""Continuous negative-unity-feedback Nyquist winding through the registered verifier."""
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def verify(num, den, **claims):
    result = run_router_conservation_cad_verifier({
        "mode": "control", "spec": {"mode": "transfer_function", "num": num, "den": den, **claims}})
    return json.loads(json.dumps(result, allow_nan=False))


def from_roots(roots):
    coefficients = [1]
    for root in roots:
        coefficients = [a - root*b for a, b in zip(coefficients + [0], [0] + coefficients)]
    return coefficients


class NyquistWindingTests(unittest.TestCase):
    def test_prescribed_open_and_closed_roots(self):
        # Monic factorizations fix both root counts without a root solver.
        # Includes repeated, stable, unstable and unreduced shared poles.
        for n in range(1, 5):
            for p in range(n + 1):
                den = from_roots([1]*p + [-2]*(n-p))
                for z in range(n + 1):
                    with self.subTest(n=n, p=p, z=z):
                        closed = from_roots([3]*z + [-2]*(n-z))
                        num = [c-d for c, d in zip(closed, den)]
                        result = verify(num, den)
                        self.assertTrue(result["passed"], result)
                        self.assertEqual(result["nyquist_clockwise_encirclements"], z-p)
                        self.assertEqual(result["nyquist_closed_loop_rhp_poles"], z)

    def test_conjugate_pair_and_boundary(self):
        for gain, expected in [(7, 0), (8, None), (9, 2)]:
            result = verify([gain], [1, 3, 3, 1])
            self.assertEqual(result["nyquist_clockwise_encirclements"], expected)
            if gain == 9:
                self.assertEqual(result["nyquist_certificate"]["negative_ray_events"], [
                    {"omega_squared_interval": ["3", "3"], "clockwise_contribution": 2}])
            elif gain == 8:
                self.assertEqual(result["nyquist_certificate"]["reason"], "nyquist_locus_hits_minus_one")
        for gain, expected in [(math.nextafter(8.0, 0.0), 0), (math.nextafter(8.0, math.inf), 2)]:
            self.assertEqual(verify([gain], [1, 3, 3, 1])["nyquist_clockwise_encirclements"], expected)

    def test_tangent_flat_origin_and_real_locus(self):
        # Im(1+L)/w is proportional to (w²-1)² and F(j)=-1.
        result = verify([-8, -4, -4], [1, 3, 3, 1])
        events = result["nyquist_certificate"]["negative_ray_events"]
        self.assertEqual([event["clockwise_contribution"] for event in events], [1, 0])
        self.assertEqual([Fraction(x) for x in events[1]["omega_squared_interval"]], [1, 1])
        # Im(1+L) changes sign at 0 with vanishing first derivative.
        self.assertEqual(verify([-4, -2], [1, 2, 1])["nyquist_clockwise_encirclements"], 1)
        self.assertEqual(verify([1], [1, 0, -4])["nyquist_certificate"]["negative_ray_events"], [])

    def test_refusals_and_claims(self):
        for num, den, reason in [
            ([1], [1, 0], "open_loop_pole_on_contour"),
            ([1, 0, 1], [1, 1, 1, 1], "open_loop_pole_on_contour"),
            ([-1], [1, 1], "nyquist_locus_hits_minus_one"),
            ([1, 1], [1, 2], "requires_strictly_proper_loop"),
            ([1, 0, 0], [1, 2], "requires_strictly_proper_loop"),
        ]:
            result = verify(num, den, claimed_field="nyquist_clockwise_encirclements", claimed_value=0)
            self.assertFalse(result["passed"], result)
            self.assertIsNone(result["nyquist_clockwise_encirclements"])
            self.assertEqual(result["nyquist_certificate"]["reason"], reason)
        for scale in [1, -1, 2**-30, 2**30]:
            self.assertTrue(verify([-2*scale], [scale, scale], claimed_field="nyquist_clockwise_encirclements",
                                   claimed_value=1)["passed"])
        self.assertFalse(verify([-2], [1, 1], claimed_field="nyquist_clockwise_encirclements",
                                claimed_value=-1)["passed"])

    @unittest.skipUnless(importlib.util.find_spec("control"), "external python-control oracle")
    def test_external_control_library(self):
        import control
        import numpy as np

        for num, den in [([-2], [1, 1]), ([2], [1, -1]), ([7], [1, 3, 3, 1]),
                         ([12], [1, 3, 3, 1]), ([-8, -4, -4], [1, 3, 3, 1]),
                         ([100], [1, 6, 11, 6]), ([1], [1, 0, -4])]:
            with self.subTest(num=num, den=den):
                contour = control.nyquist_response(control.tf(num, den),
                    omega=np.r_[0, np.geomspace(1e-5, 1e5, 4001)])
                self.assertEqual(verify(num, den)["nyquist_clockwise_encirclements"], contour.count)

    def test_registered_node_callback(self):
        root = Path(__file__).resolve().parents[1]
        script = """
            import register from './task-tools.ts';
            const registry = new Map();
            register({registerTool(tool) { registry.set(tool.name, tool); }, on() {}});
            const tool = registry.get('hamon_conservation_cad_verifier');
            const result = await tool.execute('nyquist', {mode:'control', spec:{
              mode:'transfer_function',num:[12],den:[1,3,3,1],
              claimed_field:'nyquist_clockwise_encirclements',claimed_value:2}});
            console.log(JSON.stringify([tool.parameters.properties.spec.description, result.details]));
        """
        env = dict(os.environ, TASK_HARNESS_SRC=str(root), TASK_HARNESS_PYTHON=sys.executable,
                   TASK_RELAY_URL="http://127.0.0.1:1", OPENBLAS_NUM_THREADS="1")
        run = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                             cwd=root, env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        schema, result = json.loads(run.stdout)
        self.assertIn("nyquist_clockwise_encirclements", schema)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["nyquist_closed_loop_rhp_poles"], 2)
