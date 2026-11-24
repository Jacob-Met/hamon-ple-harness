"""Verify closed orbit requests through the participant and retain project obligations."""
import json
import math
from pathlib import Path
import unittest

from test_certified_inline_reply import certificate, instruction, task_course
from test_cr3bp_transfer import EARTH_MOON, SATURN_TITAN, orbit

MODE = "lagrangian_orbital"
LAMBERT = dict(subkind="lambert_transfer", mu=1, r1_m=[1, 0, 0],
               r2_m=[0, 1, 0], tof_s=math.pi / 2)


def request(spec):
    return {"mode": MODE, "spec": spec}


class OrbitalPresolve(unittest.IsolatedAsyncioTestCase):
    async def test_two_body_and_jpl_endpoints_deliver_without_model_configuration(self):
        for row in (EARTH_MOON, SATURN_TITAN):
            three_body = dict(subkind="cr3bp_transfer", **orbit(row))
            text = instruction([request(LAMBERT), request(three_body)])
            answer, events = await task_course(text)
            two, three = json.loads(answer)
            self.assertTrue(two["endpoint_verified"])
            self.assertAlmostEqual(two["v1_m_s"][1], 1)
            self.assertTrue(three["verified"])
            for actual, expected in zip(three["endpoint"], row[3], strict=True):
                self.assertAlmostEqual(actual, expected, delta=1e-8)
            self.assertAlmostEqual(three["jacobi"], row[2], delta=1e-11)
            self.assertEqual([event["type"] for event in events], ["exec", "done"])

    def test_incomplete_or_conflicting_inputs_and_obligations_keep_normal_route(self):
        three_body = dict(subkind="cr3bp_transfer", **orbit(EARTH_MOON))
        spec_text = instruction(request(LAMBERT))
        for bad in (
            {**LAMBERT, "r1_m": ["1", 0, 0]}, {**LAMBERT, "mu": 2**53 + 1},
            {**LAMBERT, "revolutions": 1}, {**LAMBERT, "proposed_v1_m_s": [0, -1, 0]},
            {**three_body, "target_state": [0] * 6}, {**three_body, "jacobi": -1},
            {**three_body, "flight_time_tu": math.tau + 0.01},
            {**three_body, "prompt": "write a report"}, {**three_body, "kind": "lagrangian_orbital"},
            {**LAMBERT, "mass_ratio": .01}, {**LAMBERT, "domain": "structural"},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(certificate(instruction(request(bad)))["presolve_verified"])
        rounded = spec_text.replace('"mu": 1,', '"mu": 1.00000000000000000000001,')
        self.assertNotEqual(rounded, spec_text)
        for text in (rounded, spec_text + "\nWrite a report.",
                     instruction([request(LAMBERT), request({**three_body, "jacobi": -1})])):
            self.assertFalse(certificate(text)["presolve_verified"])
        for options in ({"deliverable_count": 1}, {"mcp_server_count": 1}):
            self.assertFalse(certificate(spec_text, **options)["presolve_verified"])
        self.assertFalse(certificate(json.dumps(request(LAMBERT))).get("deliver_verified_json"))

    def test_original_project_instructions_do_not_enter_orbit_presolve(self):
        corpus = sorted((Path(__file__).resolve().parents[1] / "artifacts").glob("*/trajectory.json"))
        self.assertEqual(len(corpus), 29)
        for trajectory in corpus:
            with self.subTest(task=trajectory.parent.name):
                text = json.loads(trajectory.read_bytes())["steps"][0]["message"]
                self.assertFalse(certificate(text)["presolve_verified"])
