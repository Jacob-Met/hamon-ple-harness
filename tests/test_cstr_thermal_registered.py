"""Actual registered TypeScript tool controls for the coupled thermal model."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def model():
    return {"reactor_type": "cstr", "order": 1, "C_A0": 1000., "v0": .1/60,
            "volume": .1, "A_preexp": 7.2e10/60, "E_a": 72750., "T0": 350.,
            "delta_T_adiabatic": 50000/239., "steady_state_branch": "all",
            "heat_transfer": {"UA_W_K": 50000/60, "rho_kg_m3": 1000.,
                              "Cp_J_kg_K": 239., "coolant_T_K": 300.}}


class RegisteredThermalTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fold = {"reactor_type": "cstr", "order": 1, "C_A0": 1., "v0": 1., "volume": 1.,
                "k_ref": 1., "T_ref": 512., "E_a": 8.31446261815324*8192, "T0": 480.,
                "delta_T_adiabatic": 192., "steady_state_branch": "all",
                "heat_transfer": {"UA_W_K": .5, "rho_kg_m3": 1.,
                                  "Cp_J_kg_K": 1., "coolant_T_K": 384.}}
        cases = {"three_states": model(), "ambiguous": model(), "wrong_reactor": model(),
                 "incomplete_heat": model(), "numeric_string": model()}
        del cases["ambiguous"]["steady_state_branch"]
        cases["wrong_reactor"]["reactor_type"] = "pfr"
        del cases["incomplete_heat"]["heat_transfer"]["coolant_T_K"]
        cases["numeric_string"]["heat_transfer"]["Cp_J_kg_K"] = "239"
        for name, ua in (("below_fold", math.nextafter(.5, 0)), ("fold", .5),
                         ("above_fold", math.nextafter(.5, 1))):
            cases[name] = copy.deepcopy(fold)
            cases[name]["heat_transfer"]["UA_W_K"] = ua
        cases["hopf"] = copy.deepcopy(fold)
        cases["hopf"].update(T0=512., delta_T_adiabatic=320.)
        cases["hopf"]["heat_transfer"].update(UA_W_K=2., coolant_T_K=432.)
        source = r"""
import registerTaskTools from './task-tools.ts';
const registered = new Map();
registerTaskTools({registerTool(t) {registered.set(t.name,t);},on() {}});
const tool = registered.get('hamon_conservation_cad_verifier');
const results = {};
for (const [id,spec] of Object.entries(JSON.parse(process.env.CSTR_TOOL_CASES))) {
  try {
    const result = await tool.execute(id,{mode:'reactor_kinetics',spec},null,()=>{},{});
    results[id] = {completed:true,details:result.details};
  } catch (error) {results[id] = {completed:false,error:String(error)};}
}
console.log(JSON.stringify(results));
"""
        with tempfile.TemporaryDirectory() as workdir:
            env = {**os.environ, "TASK_RELAY_URL": "http://127.0.0.1:1",
                   "TASK_WORKDIR": workdir, "TASK_HARNESS_SRC": str(ROOT),
                   "TASK_HARNESS_PYTHON": sys.executable, "CSTR_TOOL_CASES": json.dumps(cases)}
            proc = subprocess.run(
                ["node", "--experimental-strip-types", "--input-type=module", "-e", source],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
        if proc.returncode:
            raise AssertionError(proc.stderr)
        cls.results = json.loads(proc.stdout.strip().splitlines()[-1])

    def test_original_heat_balance_and_full_dynamic_classification_reach_caller(self):
        observation = self.results["three_states"]
        self.assertTrue(observation["completed"], observation)
        result = observation["details"]
        self.assertTrue(result["verified"])
        self.assertEqual(result["steady_state_count"], 3)
        self.assertEqual([r["linear_stability"] for r in result["steady_states"]],
                         ["asymptotically_stable", "unstable_saddle", "unstable"])
        for row, temperature in zip(result["steady_states"],
                                    [324.48605305000467, 349.9623318873878, 369.7246520915924]):
            self.assertAlmostEqual(row["exit_T_K"], temperature, places=10)

    def test_fold_and_hopf_survive_json_and_typescript(self):
        counts = []
        for name in ("below_fold", "fold", "above_fold"):
            observation = self.results[name]
            self.assertTrue(observation["completed"], observation)
            self.assertTrue(observation["details"]["verified"])
            counts.append(observation["details"]["steady_state_count"])
        self.assertEqual(counts, [3, 2, 1])
        hopf = self.results["hopf"]
        self.assertTrue(hopf["completed"], hopf)
        self.assertEqual(hopf["details"]["steady_states"][0]["linear_stability"], "nonhyperbolic")

    def test_ambiguous_and_incompatible_inputs_do_not_return_a_certificate(self):
        for name in ("ambiguous", "wrong_reactor", "incomplete_heat", "numeric_string"):
            with self.subTest(name=name):
                self.assertFalse(self.results[name]["completed"], self.results[name])


if __name__ == "__main__":
    unittest.main()
