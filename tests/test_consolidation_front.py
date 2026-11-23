"""Compare the registered tool's drainage fronts with independent pressure and flux solutions."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from test_layered_consolidation_transient import column, layer

ROOT = Path(__file__).resolve().parents[1]
EXECUTE = """
import fs from 'node:fs';
import register from './task-tools.ts';
const tools = new Map();
register({ registerTool(tool) { tools.set(tool.name, tool); }, on() {} });
const verifier = tools.get('hamon_conservation_cad_verifier');
const answers = [];
for (const spec of JSON.parse(fs.readFileSync(0, 'utf8'))) {
    const reply = await verifier.execute('consolidation', {mode: 'civil_geotech', spec},
                                         null, () => {}, {});
    answers.push(reply.details);
}
console.log(JSON.stringify(answers));
"""


def registered(specs):
    env = dict(os.environ, TASK_HARNESS_SRC=str(ROOT), TASK_HARNESS_PYTHON=sys.executable,
               TASK_RELAY_URL="http://127.0.0.1:1", OPENBLAS_NUM_THREADS="1")
    run = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", EXECUTE],
                         cwd=ROOT, input=json.dumps(specs), capture_output=True, text=True, env=env,
                         check=True, timeout=240)
    return json.loads(run.stdout)


class FrontTests(unittest.TestCase):
    def test_single_double_and_mirrored_fronts(self):
        elapsed = .01
        ref = 2 * 1e-7 * 1e5 * math.sqrt(1e-6 * elapsed / math.pi)
        rows = [(column(layer(0, 1), layer(1, 2), time_s=elapsed, top_boundary=top,
                        bottom_boundary=bottom, proposed_settlement_m=ref*count), count)
                for top, bottom, count in (("drained", "impermeable", 1),
                                           ("impermeable", "drained", 1),
                                           ("drained", "drained", 2))]
        bad = {**rows[0][0], "proposed_settlement_m": ref * 1.00001}
        good, mirror, double, wrong = registered([r[0] for r in rows] + [bad])
        for answer, (_, count) in zip((good, mirror, double), rows):
            self.assertTrue(answer["conservation_verified"], answer)
            self.assertLess(answer["dual_solver_rel_residual"], 1e-6)
            self.assertAlmostEqual(answer["settlement_at_time_m"] / (ref * count), 1., delta=1e-6)
        self.assertTrue(wrong["ok"], wrong)
        self.assertFalse(wrong["conservation_verified"])

    def test_front_crosses_an_interface(self):
        # Equal mv*sqrt(cv) removes wave reflection in x = integral(dz/sqrt(cv)).
        # The far sealed boundary is more than twenty diffusion lengths away.
        times = (.05, .25, 1.)
        specs = [column(layer(0, .001), layer(.001, .04, 5e-8, 4e-6), time_s=t) for t in times]
        for t, result in zip(times, registered(specs)):
            self.assertTrue(result["conservation_verified"], result)
            root_t = math.sqrt(t)
            effusivity = 1e-10
            pressure = 1e5 * math.erf(1 / (2 * root_t))
            flux = -1e5 * effusivity * math.exp(-1 / (4 * t)) / math.sqrt(math.pi * t)
            motion = 2 * effusivity * 1e5 * math.sqrt(t / math.pi)
            def integral(x):
                return x * math.erf(x / (2 * root_t)) + 2 * math.sqrt(t / math.pi) * math.exp(-x*x/(4*t))
            face, = result["interfaces"]
            self.assertEqual(face["depth_m"], .001)
            computed = [result["settlement_at_time_m"], face["excess_pressure_Pa"],
                        face["downward_darcy_flux_m_s"], *result["layers_mean_excess_pressure_Pa"]]
            expected = [motion, pressure, flux, 1e5*(integral(1)-integral(0)),
                        1e5*(integral(20.5)-integral(1))/19.5]
            for actual, reference in zip(computed, expected):
                self.assertAlmostEqual(actual / reference, 1., delta=1e-6)

    def test_three_independent_resolution_limits(self):
        cases = [(column(layer(0, 1), layer(1, 2, 4e-7), time_s=1e-40), "storage precision"),
                 (column(layer(0, 1e-20), layer(1e-20, 1, 5e-8, 1.), time_s=.01), "ill-conditioned"),
                 (column(layer(0, .0001), layer(.0001, 1, 5e-8, 4e-6), time_s=.01), "did not converge")]
        for result, (_, limit) in zip(registered([s for s, _ in cases]), cases):
            self.assertFalse(result["ok"], result)
            self.assertFalse(result["conservation_verified"], result)
            self.assertIn(limit, result["reason"])
