"""Exercise the discoverable schema and actual registered control tool together."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RegisteredDiscreteControlTests(unittest.TestCase):
    def test_timebase_is_discoverable_and_reaches_discrete_solver(self):
        script = """
            import register from './task-tools.ts';
            const registry = new Map();
            register({ registerTool(tool) { registry.set(tool.name, tool); }, on() {} });
            const tool = registry.get('hamon_conservation_cad_verifier');
            const outputs = [];
            for (const A of [ [[0.9990, 0.0095], [-0.1903, 0.9039]], [[-2]] ]) {
                const result = await tool.execute('discrete-control', {
                    mode: 'control', spec: { mode: 'state_space', timebase: 'discrete',
                        sample_time_s: 0.01, A }
                });
                outputs.push(result.details);
            }
            console.log(JSON.stringify({ schema: tool.parameters.properties.spec, outputs }));
        """
        env = dict(os.environ, TASK_HARNESS_SRC=str(ROOT), TASK_HARNESS_PYTHON=sys.executable,
                   TASK_RELAY_URL='http://127.0.0.1:1', OPENBLAS_NUM_THREADS='1')
        proc = subprocess.run(
            ['node', '--experimental-strip-types', '--input-type=module', '-e', script],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn('timebase=discrete', payload['schema']['description'])
        self.assertIn('sample_time_s', payload['schema']['description'])
        for output, stable in zip(payload['outputs'], [True, False]):
            self.assertTrue(output['passed'], output)
            self.assertEqual(output['schur_stable'], stable)
            self.assertEqual(output['schur_recursion_stable'], stable)
            self.assertNotIn('hurwitz_stable', output)


if __name__ == '__main__':
    unittest.main()
