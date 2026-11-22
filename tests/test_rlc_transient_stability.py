"""Independent endpoint and numerical-boundary regressions for series RLC."""
import json
import math
import os
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import mpmath as mp

import hamon_wave6_physics_verifiers as wave6


ROOT = Path(__file__).resolve().parents[1]


def reference(spec):
    with mp.workdps(90):
        r, l, c, vs, time_ = [mp.mpf(str(spec[k])) for k in ("R", "L", "C", "V_s", "t")]
        v0, i0 = mp.mpf(str(spec.get("v_c0", 0))), mp.mpf(str(spec.get("i_l0", 0)))
        # General affine generator in physical coordinates, independently
        # evaluated by mpmath rather than the receiver's modal equations.
        generator = mp.matrix([[0, 1/c, 0], [-1/l, -r/l, vs/l], [0, 0, 0]])
        state = mp.expm(generator*time_) * mp.matrix([v0, i0, 1])
        return float(state[0]), float(state[1])


class RlcTransientStabilityTests(unittest.TestCase):
    def assert_solution(self, spec, relative=5e-13):
        expected = reference(spec)
        actual = wave6.solve_transmission_line_rlc(dict(spec, subkind="rlc_transient"))
        self.assertTrue(actual["verified"], actual)
        for key, target in zip(("v_c_V", "i_l_A"), expected):
            if target:
                self.assertLessEqual(abs(actual[key]/target-1), relative, (key, actual, expected))
            else:
                self.assertEqual(actual[key], 0.)
        return actual

    def test_stiff_slow_tail_has_correct_voltage_and_current(self):
        for resistance, time_ in ((1e6, 1.), (1e9, 1000.)):
            with self.subTest(resistance=resistance):
                actual = self.assert_solution(dict(R=resistance, L=.001, C=1e-6, V_s=12., t=time_))
                self.assertEqual(actual["regime"], "overdamped")
                self.assertLess(actual["reference_squarings"], 64)

    def test_tiny_zero_state_voltage_survives_all_regimes(self):
        for resistance in (0., 1., 2., 3.):
            for time_ in (1e-8, 1e-25):
                with self.subTest(resistance=resistance, time=time_):
                    actual = self.assert_solution(dict(R=resistance, L=1., C=1., V_s=1., t=time_))
                    self.assertGreater(actual["v_c_V"], 0.)

    def test_fast_mode_can_evolve_before_small_source_response(self):
        self.assert_solution(dict(R=1e9, L=.001, C=1e-6, V_s=10., t=1e-12))

    def test_near_critical_models_are_not_snapped_to_critical(self):
        for ratio in (1.-1e-10, 1.+1e-10):
            with self.subTest(ratio=ratio):
                actual = self.assert_solution(dict(R=2*ratio, L=1., C=1., V_s=2.,
                                                   v_c0=-1., i_l0=-.02, t=1.))
                expected = "underdamped" if ratio < 1 else "overdamped"
                self.assertEqual(actual["regime"], expected)

    def test_initial_state_and_equilibrium_are_exact(self):
        spec = dict(subkind="rlc_transient", R=1e9, L=.001, C=1e-6,
                    V_s=3., v_c0=-2., i_l0=.7, t=0.)
        actual = wave6.solve_transmission_line_rlc(spec)
        self.assertEqual((actual["v_c_V"], actual["i_l_A"], actual["residual"]), (-2., .7, 0.))
        actual = wave6.solve_transmission_line_rlc(dict(spec, V_s=-2., i_l0=0., t=1000.))
        self.assertEqual((actual["v_c_V"], actual["i_l_A"], actual["residual"]), (-2., 0., 0.))

    def test_checker_rejects_erased_tiny_response(self):
        spec = dict(subkind="rlc_transient", R=1., L=1., C=1., V_s=1., t=1e-25)
        correct = wave6.solve_transmission_line_rlc(spec)
        with patch.object(wave6, "_rlc_analytic_state",
                          return_value=(0., correct["i_l_A"], "underdamped")):
            rejected = wave6.solve_transmission_line_rlc(spec)
        self.assertFalse(rejected["verified"])
        self.assertEqual(rejected["residual"], 1.)

    def test_invalid_passive_parameters_are_rejected(self):
        spec = dict(subkind="rlc_transient", R=1., L=1., C=1., V_s=1., t=1.)
        invalid = [("R", -1.), ("L", 0.), ("C", -1.), ("t", -1.),
                   *[(name, value) for name in ("R", "L", "C", "V_s", "v_c0", "i_l0", "t")
                     for value in (float("inf"), float("nan"))]]
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    wave6.solve_transmission_line_rlc(dict(spec, **{name: value}))

    def test_stiff_receiver_works_without_scipy(self):
        script = """
import importlib.abc,json,sys
class RejectScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("SciPy deliberately absent", name=fullname)
sys.meta_path.insert(0, RejectScipy())
from hamon_wave6_physics_verifiers import solve_transmission_line_rlc
print(json.dumps(solve_transmission_line_rlc(json.load(sys.stdin)), allow_nan=False))
"""
        spec = dict(subkind="rlc_transient", R=1e9, L=.001, C=1e-6, V_s=12., t=1000.)
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                input=json.dumps(spec), capture_output=True, text=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)
        self.assertTrue(actual["verified"])
        expected_v, expected_i = reference(spec)
        self.assertTrue(math.isclose(actual["v_c_V"], expected_v, rel_tol=5e-13))
        self.assertTrue(math.isclose(actual["i_l_A"], expected_i, rel_tol=5e-13))



class RegisteredRlcTransientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "node_modules/@earendil-works/pi-coding-agent").exists():
            raise unittest.SkipTest("registered Pi runtime is not installed")

    def test_registered_stiff_and_tiny_transients(self):
        valid = [
            dict(id="stiff", spec=dict(subkind="rlc_transient", R=1e9, L=.001,
                                      C=1e-6, V_s=12., t=1000.)),
            dict(id="tiny", spec=dict(subkind="rlc_transient", R=1., L=1.,
                                     C=1., V_s=1., t=1e-25)),
        ]
        calls = valid + [dict(id="invalid", spec=dict(valid[0]["spec"], L=0.))]
        source = r"""
import registerTaskTools from './task-tools.ts';
const tools = new Map();
registerTaskTools({ registerTool(t) { tools.set(t.name, t); }, on() {} });
const tool = tools.get('hamon_conservation_cad_verifier');
const results = {};
for (const call of JSON.parse(process.env.RLC_TOOL_CASES)) {
  try {
    const result = await tool.execute(call.id, {mode: 'rf_transmission_rlc', spec: call.spec},
                                      null, () => {}, {});
    results[call.id] = {completed: true, details: result.details};
  } catch (error) {
    results[call.id] = {completed: false, error: String(error)};
  }
}
console.log(JSON.stringify(results));
"""
        with tempfile.TemporaryDirectory() as work:
            env = dict(os.environ, TASK_RELAY_URL="http://127.0.0.1:1", TASK_WORKDIR=work,
                       TASK_HARNESS_SRC=str(ROOT), TASK_HARNESS_PYTHON=sys.executable,
                       RLC_TOOL_CASES=json.dumps(calls))
            process = subprocess.run(
                ["node", "--experimental-strip-types", "--input-type=module", "-e", source],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        actual = json.loads(process.stdout.strip().splitlines()[-1])
        self.assertFalse(actual["invalid"]["completed"])
        for case in valid:
            with self.subTest(case=case["id"]):
                observed = actual[case["id"]]
                self.assertTrue(observed["completed"], observed)
                self.assertTrue(observed["details"]["verified"])
                expected = reference(case["spec"])
                for key, target in zip(("v_c_V", "i_l_A"), expected):
                    self.assertLess(abs(observed["details"][key]/target-1), 5e-13)


if __name__ == "__main__":
    unittest.main()
