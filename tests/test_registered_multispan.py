"""Replay the production Pi tool callback with its real Python subprocess."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class RegisteredMultiSpan(unittest.TestCase):
    def test_structural_auto_and_conflicting_input(self):
        root = Path(__file__).resolve().parents[1]
        program = r"""
import assert from 'node:assert/strict';
import register from './task-tools.ts';
const tools = new Map();
register({ on() {}, registerTool(x) { tools.set(x.name, x); } });
const beam = tools.get('hamon_conservation_cad_verifier');
assert.ok(beam.description.includes('continuous_beam_multispan'));
const spec = { calculation:'continuous_beam_multispan',spans_m:[1,1,1],udl_N_m:[0,2,0],
               span_EI_Nm2:[1,1,1],support_conditions:{A:'compression_only',D:'compression_only'} };
for (const mode of ['structural','auto']) {
  const r=(await beam.execute('lift', {mode,spec})).details;
  assert.equal(r.M_max_Nm,.25);
  assert.equal(r.support_states.A,'lifted');
  assert.equal(r.support_states.D,'lifted');
  assert.equal(r.conservation_verified,true);
  console.log(mode, r.M_max_Nm, r.support_states);
}
const bad=(await beam.execute('invalid', { mode:'structural',
    spec:{ ...spec, span_m:3,E_Pa:1,I_m4:1,w_N_m:2 } })).details;
assert.equal(bad.ok,false);
console.log(bad.error);
"""
        env = dict(os.environ, TASK_HARNESS_SRC=str(root), TASK_HARNESS_PYTHON=sys.executable,
                   TASK_WORKDIR=str(root), TASK_RELAY_URL="http://127.0.0.1:0")
        proc = subprocess.run(["node", "--experimental-strip-types", "--input-type=module",
                               "-e", program], cwd=root, env=env, text=True,
                              capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
