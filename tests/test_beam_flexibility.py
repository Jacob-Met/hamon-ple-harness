"""Independent virtual-work controls for a connected beam with an EI jump."""

from fractions import Fraction as F
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(spec):
    return run_router_conservation_cad_verifier({"mode": "fem", "spec": spec})


def beam(left, right, divisions=1):
    return {
        "mode": "euler_bernoulli_beam_1d",
        "x_nodes_m": [i / divisions for i in range(2 * divisions + 1)],
        "elements": [{"EI_Nm2": v} for v in [left] * divisions + [right] * divisions],
        "supports": [{"node": 0, "fix_w": True, "fix_theta": True}],
        "nodal_loads": [{"node": 2 * divisions, "fy_N": 1}],
    }


class BeamFlexibility(unittest.TestCase):
    def test_virtual_tip_load_and_reversed_stiffness_jump(self):
        for left, right in ((1, 2**60), (2**60, 1), (2**-60, 1), (1, 1)):
            # Unit tip force. The section moment is 2-x on either side.
            tip = F(7, 3) / F(left) + F(1, 3) / F(right)
            slope = F(3, 2) / F(left) + F(1, 2) / F(right)
            for divisions in (1, 8):
                with self.subTest(left=left, right=right, divisions=divisions):
                    r = solve(beam(left, right, divisions))
                    self.assertTrue(r["passed"], r)
                    self.assertEqual(r["tip_deflection_m"], float(tip))
                    self.assertEqual(r["rotations_rad"][-1], float(slope))
                    self.assertEqual(r["strain_energy_J"], float(tip / 2))
                    self.assertEqual(r["external_work_J"], float(tip / 2))
                    self.assertEqual(r["reactions_V_M"][0], [-1, -2])
                    self.assertEqual(r["reactions_V_M"][-1], [0, 0])

    def test_unequal_spans_redundant_support_recovers_compatibility(self):
        # A simply supported beam of lengths a and b has unit-force support
        # reactions -b/(a+b), -a/(a+b) for a load at the join. Integrating
        # M_unit^2/EI gives the compliance there. Prescribe that displacement
        # instead of the load and recover the unknown reaction.
        a, b, left, right = 1, 3, 1, 2**60
        spec = beam(left, right)
        spec["x_nodes_m"] = [0, a, a + b]
        spec["nodal_loads"] = []
        spec["supports"] = [{"node": i, "fix_w": True, "w_m": w} for i, w in enumerate((0, 0.125, 0))]
        r = solve(spec)
        self.assertTrue(r["passed"], r)
        compliance = F(a * a * b * b, 3 * (a + b) ** 2) * (F(a, left) + F(b, right))
        force = F(1, 8) / compliance
        self.assertEqual(
            r["reactions_V_M"],
            [[float(-force * b / (a + b)), 0], [float(force), 0], [float(-force * a / (a + b)), 0]],
        )
        self.assertEqual(r["strain_energy_J"], float(force / 16))
        self.assertEqual(r["support_work_J"], float(force / 16))
        self.assertEqual(r["external_work_J"], 0)

    def test_registered_typescript_engineering_tool(self):
        root = Path(__file__).resolve().parents[1]
        script = """
import register from './task-tools.ts';
const tools = new Map();
register({registerTool(tool) { tools.set(tool.name, tool); }, on() {}});
const tool = tools.get('hamon_conservation_cad_verifier');
const spec = JSON.parse(process.env.BEAM_CASE);
const response = await tool.execute('beam', {mode: 'fem', spec}, null, () => {}, {});
console.log(JSON.stringify(response.details));
"""
        env = {**os.environ, "TASK_HARNESS_SRC": str(root), "TASK_HARNESS_PYTHON": sys.executable,
               "TASK_RELAY_URL": "http://127.0.0.1:1", "BEAM_CASE": json.dumps(beam(1, 2**60))}
        result = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                                cwd=root, env=env, text=True, capture_output=True, timeout=120, check=True)
        answer = json.loads(result.stdout.splitlines()[-1])
        self.assertTrue(answer["passed"], answer)
        self.assertEqual(answer["tip_deflection_m"], float(F(7, 3) + F(1, 3*2**60)))

    def test_unrestrained_rotation_and_conflicting_supports(self):
        invalid = beam(1, 2**60)
        invalid["supports"] = [{"node": 0, "fix_w": True}]
        self.assertFalse(solve(invalid)["passed"])
        invalid["supports"].extend([{"node": 0, "fix_theta": True}, {"node": 0, "fix_w": True, "w_m": 1}])
        self.assertFalse(solve(invalid)["passed"])
