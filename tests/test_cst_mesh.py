"""CST patch, constitutive and admissibility controls through the registered Python receiver."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(spec):
    return run_router_conservation_cad_verifier({"mode": "fem", "spec": spec})


def patch(kinematics="plane_stress"):
    return {
        "mode": "cst_2d", "kinematics": kinematics, "E_Pa": 1000, "nu": 0.25, "thickness_m": 2,
        "nodes": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]],
        "elements": [[0, 1, 4], [1, 4, 2], [2, 3, 4], [3, 4, 0]],
        "supports": [{"node": 0, "ux_m": 0, "uy_m": 0}, {"node": 3, "ux_m": 0}],
        "loads": [{"node": 1, "fx_N": 100}, {"node": 2, "fx_N": 100}],
    }


class ConstantStrainTriangleTests(unittest.TestCase):
    def test_traction_patch_under_both_plane_assumptions(self):
        for mode, ex, ey, zz, energy in [
            ("plane_stress", 0.1, -0.025, 0, 10),
            ("plane_strain", 0.09375, -0.03125, 25, 9.375),
        ]:
            with self.subTest(mode=mode):
                spec = patch(mode)
                got = solve(spec)
                self.assertTrue(got["passed"], got)
                self.assertEqual(got["displacements_xy_m"], [[x*ex, y*ey] for x,y in spec["nodes"]])
                self.assertEqual(got["element_stress_xx_yy_xy_Pa"], [[100, 0, 0]]*4)
                self.assertEqual(got["element_sigma_zz_Pa"], [zz]*4)
                self.assertEqual(got["element_strain_xx_yy_gamma_xy"], [[ex, ey, 0]]*4)
                self.assertEqual(got["reactions_xy_N"], [[-100, 0], [0, 0], [0, 0], [-100, 0], [0, 0]])
                self.assertEqual(got["strain_energy_J"], energy)
                self.assertEqual(got["nodal_load_work_J"], energy)
                self.assertEqual(got["prescribed_support_work_J"], 0)
                self.assertFalse(solve({**spec, "claimed_field":"strain_energy_J",
                                        "claimed_value":0, "rtol":0, "atol":0})["passed"])

    def test_engineering_shear_support_work_and_rigid_motion(self):
        spec = {**patch(), "nodes": [[0,0],[2,0],[0,1]], "elements": [[2,1,0]],
                "E_Pa": 80, "loads": []}
        def prescribed(fn):
            return [{"node":i, "ux_m":x_disp, "uy_m":y_disp}
                    for i,(x,y) in enumerate(spec["nodes"]) for x_disp,y_disp in [fn(x,y)]]
        shear = solve({**spec, "supports":prescribed(lambda x,y: (y/16,0))})
        self.assertTrue(shear["passed"], shear)
        self.assertEqual(shear["element_stress_xx_yy_xy_Pa"], [[0,0,2]])
        self.assertEqual(shear["strain_energy_J"], 0.125)
        self.assertEqual(shear["prescribed_support_work_J"], 0.125)
        rigid = solve({**spec, "supports":prescribed(lambda x,y: (0.5-y/4, x/4))})
        self.assertTrue(rigid["passed"], rigid)
        self.assertEqual(rigid["strain_energy_J"], 0)
        self.assertEqual(rigid["reactions_xy_N"], [[0,0]]*3)

    def test_registered_tool_uses_mesh_and_documents_input_axes(self):
        script = """
            import fs from 'node:fs';
            import register from './task-tools.ts';
            const registry = new Map();
            register({registerTool(tool) {registry.set(tool.name, tool)}, on() {}});
            const tool = registry.get('hamon_conservation_cad_verifier');
            const help = await tool.execute('cst-reference', {mode:'reference'});
            const spec = JSON.parse(fs.readFileSync(0, 'utf8'));
            const response = await tool.execute('cst-patch', {mode:'fem', spec});
            console.log(JSON.stringify({help:help.content[0].text, result:response.details}));
        """
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, TASK_HARNESS_SRC=str(root), TASK_HARNESS_PYTHON=sys.executable,
                   TASK_RELAY_URL="http://127.0.0.1:1", OPENBLAS_NUM_THREADS="1")
        proc = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                              cwd=root, env=env, text=True, input=json.dumps(patch()),
                              capture_output=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        got = json.loads(proc.stdout)
        self.assertIn("cst_2d", got["help"])
        self.assertTrue(got["result"]["passed"], got["result"])
        self.assertEqual(got["result"]["element_stress_xx_yy_xy_Pa"], [[100, 0, 0]]*4)

    def test_mesh_and_support_refusals(self):
        spec = patch()
        cases = [
            {"supports":[]}, {"nu":0.5}, {"elements":[[0,1,2,3]]},
            {"elements":[[0,2,4]]}, {"elements":[[0,1,4]]},
            {"elements":[*spec["elements"], [1,4,0]]},
            {"elements":[[0,1,2],[0,4,3],[4,2,3]]},
            {"nodes":[[0,0],[2,0],[1,2],[0,1],[2,1],[1,-1]],
             "elements":[[0,1,2],[3,4,5]]},
            {"loads":[{"node":True,"fy_N":1}]}, {"edge_traction_Pa":1},
            {"supports":[*spec["supports"], {"node":0,"ux_m":1}]},
        ]
        for change in cases:
            with self.subTest(change=change):
                self.assertFalse(solve({**spec, **change})["passed"])


if __name__ == "__main__":
    unittest.main()
