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
assert.ok(beam.description.includes('mode=reference'));
const help = await beam.execute('help', {mode:'reference'});
assert.ok(help.content.some(item => item.text?.includes('continuous_beam_multispan')));
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
const {calculation, ...inputs} = spec;
const femSpec = {...inputs, mode: calculation};
const runFem = async (id, changes = {}) =>
  (await beam.execute(id, {mode: 'fem', spec: {...femSpec, ...changes}})).details;
const lifted = await runFem('fem-lift');
assert.equal(lifted.passed, true);
assert.deepEqual(lifted.support_gaps_m, {A: 1/12, B: 0, C: 0, D: 1/12});
assert.deepEqual(lifted.support_reactions_N, {A: 0, B: 1, C: 1, D: 0});
const reversed = await runFem('fem-reversal', {udl_N_m: [0, -2, 0]});
assert.equal(reversed.passed, true);
assert.deepEqual(reversed.support_gaps_m, {A: 0, B: 0, C: 0, D: 0});
assert.deepEqual(reversed.support_reactions_N, {A: 0.1, B: -1.1, C: -1.1, D: 0.1});
assert.deepEqual(reversed.support_states, {A: 'bearing', B: 'hold_down', C: 'hold_down', D: 'bearing'});
const unrestrained = await runFem('fem-no-equilibrium', {
  udl_N_m: [0, -2, 0], support_conditions: Array(4).fill('compression_only')
});
assert.equal(unrestrained.passed, false);
assert.match(unrestrained.reason, /no restrained equilibrium/);
assert.equal((await runFem('fem-conflict', {calculation: 'continuous_beam_2span'})).passed, false);
assert.equal((await runFem('fem-unsupported', {point_load_N: 10})).passed, false);
// Shi Lab CSA 03 gives the signed reactions (13, 22, -3) P/32 and support moment -3PL/32.
const point = {spans_m:[2,2], udl_N_m:[0,0], span_EI_Nm2:[16,16],
               point_loads:[{x_m:1, downward_N:32}], support_conditions:{}};
for (const family of ['structural', 'fem']) {
  const selector = family === 'fem' ? {mode: calculation} : {calculation};
  const fixed = (await beam.execute('point-' + family, {mode:family, spec:{...point, ...selector}})).details;
  assert.deepEqual(fixed.support_reactions_N, {A:13, B:22, C:-3});
  assert.equal(fixed.support_moments_Nm.B, -6);
  assert.equal(fixed.dual_verified, true);
  const free = (await beam.execute('free-' + family, {mode:family,
    spec:{...point, ...selector, support_conditions:{C:'compression_only'}}})).details;
  assert.deepEqual(free.support_reactions_N, {A:16, B:16, C:0});
  assert.equal(free.support_gaps_m.C, 1);
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
