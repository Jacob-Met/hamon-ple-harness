"""Whole-tube resistance through the actual thermal rating and sizing routes."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.integrate import solve_bvp

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_dual_verifier import solve_domain_problem


# Colorado State CBE101, "Team Exercise: The Effect of Fouling on Heat
# Exchanger Performance". The source gives radii 0.75 and 0.95 cm.
TUBE = dict(inner_diameter_m=0.015, outer_diameter_m=0.019, wall_k_W_mK=15.1,
            inner_h_W_m2K=800, outer_h_W_m2K=1200,
            inner_fouling_m2K_W=0.0004, outer_fouling_m2K_W=0.0001)
SOURCES = dict(m_hot_kg_s=1., cp_hot_J_kgK=220., m_cold_kg_s=1., cp_cold_J_kgK=280.,
               Th_in_K=430., Tc_in_K=300.)
REGISTERED = dict(subkind="heat_exchanger", C_h=220., C_c=280., T_h_in=430., T_c_in=300.)


def course(tube, **measurements):
    return run_router_conservation_cad_verifier({"mode": "exchangers_radiation",
        "spec": dict(REGISTERED, tube_resistance=tube, **measurements)})


def flux_oracle(tube):
    # Numerically solve radial wall conduction. Its two Robin conditions
    # enforce local film + fouling temperature jumps at different radii.
    ri, ro = tube["inner_diameter_m"]/2, tube["outer_diameter_m"]/2
    k = tube["wall_k_W_mK"]
    inner_jump = 1/tube["inner_h_W_m2K"] + tube["inner_fouling_m2K_W"]
    outer_jump = 1/tube["outer_h_W_m2K"] + tube["outer_fouling_m2K_W"]
    radius = np.linspace(ri, ro, 20)

    def differential(r, y):
        return np.array([y[1], -y[1]/r])

    def boundaries(a, b):
        return np.array([1 - a[0] + k*a[1]*inner_jump, b[0] + k*b[1]*outer_jump])

    field = solve_bvp(differential, boundaries, radius, np.zeros((2, len(radius))), tol=1e-9)
    if not field.success:
        raise AssertionError(field.message)
    fluxes = -2*math.pi*k*np.array([ri*field.y[1, 0], ro*field.y[1, -1]])
    np.testing.assert_allclose(fluxes[0], fluxes[1], rtol=2e-10)
    return float(fluxes[0])


class TubeFoulingTests(unittest.TestCase):
    def test_csu_clean_and_fouled_radial_boundary_value_problem(self):
        for fi, fo in [(0, 0), (.0004, 0), (0, .0001), (.0004, .0001)]:
            tube = dict(TUBE, inner_fouling_m2K_W=fi, outer_fouling_m2K_W=fo, total_length_m=1.)
            result = course(tube)
            self.assertTrue(result["verified"], result)
            reported = result["tube_resistance"]
            conductance = flux_oracle(tube)
            self.assertAlmostEqual(result["UA_W_K"], conductance, places=7)
            self.assertAlmostEqual(reported["service_U_inner_W_m2K"]*math.pi*.015, conductance, places=7)
            self.assertAlmostEqual(reported["service_U_outer_W_m2K"]*math.pi*.019, conductance, places=7)
            self.assertAlmostEqual(sum(reported["resistance_times_length_mK_W"].values()), 1/conductance, places=10)
        self.assertAlmostEqual(reported["clean_UA_W_K"], 23.2675363256, places=8)
        self.assertAlmostEqual(reported["rated_UA_W_K"], 18.8175378809, places=8)

    def test_service_measurements_and_sizing(self):
        tube = dict(TUBE, total_length_m=2.3)
        q = course(tube)
        self.assertTrue(q["verified"], q)
        for key, value in [("Q", q["Q_W"]), ("T_h_out", q["T_h_out_K"]), ("UA", q["UA_W_K"])]:
            result = course(TUBE, **{key: value})
            self.assertTrue(result["verified"], result)
            self.assertAlmostEqual(result["tube_resistance"]["required_total_length_m"], 2.3, places=10)
        wrong = course(tube, UA=q["tube_resistance"]["clean_UA_W_K"])
        self.assertFalse(wrong["verified"], wrong)
        self.assertGreater(wrong["observation_residuals"]["UA"], .1)
        self.assertFalse(course(tube, Q=q["Q_W"]*1.1)["verified"])
        self.assertFalse(course(TUBE)["verified"])

    def test_both_passive_routes_preserve_area_basis_and_validate_supplied_data(self):
        tube = dict(TUBE, total_length_m=3)
        rate = solve_domain_problem("heat_exchanger_ntu", dict(SOURCES, tube_resistance=tube))
        self.assertTrue(rate["dual_verified"], rate)
        inputs = dict(SOURCES, Th_out_K=rate["Th_out_K"], tube_resistance=TUBE)
        sized = solve_domain_problem("heat_exchanger_lmtd", inputs)
        self.assertTrue(sized["dual_verified"], sized)
        self.assertAlmostEqual(sized["area_m2"], math.pi*.019*3, places=12)
        self.assertAlmostEqual(sized["tube_resistance"]["required_inner_area_m2"], math.pi*.015*3, places=12)
        for changed in [dict(U_W_m2K=rate["tube_resistance"]["service_U_inner_W_m2K"]),
                        dict(UA_W_K=500), dict(tube_resistance=dict(TUBE, total_length_m=8))]:
            self.assertFalse(solve_domain_problem("heat_exchanger_lmtd", dict(inputs, **changed))["dual_verified"])
        self.assertFalse(solve_domain_problem("heat_exchanger_ntu", dict(SOURCES, tube_resistance=TUBE))["dual_verified"])

    def test_geometry_rejections_and_clean_limit(self):
        for changed in [dict(outer_diameter_m=.015), dict(inner_fouling_m2K_W=-.1),
                        dict(outer_h_W_m2K=True), dict(extra=1), dict(total_length_m=0),
                        dict(wall_k_W_mK="NaN")]:
            self.assertFalse(course(dict(dict(TUBE, total_length_m=1), **changed))["verified"])
        self.assertFalse(course({k:v for k,v in TUBE.items() if k != "outer_fouling_m2K_W"}, UA=1)["verified"])
        close = course(dict(TUBE, outer_diameter_m=math.nextafter(.015, math.inf),
                            inner_fouling_m2K_W=0, outer_fouling_m2K_W=0, total_length_m=1))
        self.assertTrue(close["verified"], close)
        cleaned = close["tube_resistance"]
        self.assertGreater(cleaned["resistance_times_length_mK_W"]["wall"], 0)
        self.assertEqual(cleaned["rated_UA_W_K"], cleaned["clean_UA_W_K"])

    def test_actual_registered_node_callback(self):
        root = Path(__file__).resolve().parents[1]
        spec = dict(REGISTERED, tube_resistance=dict(TUBE, total_length_m=1))
        script = """
            import register from './task-tools.ts';
            const tools = new Map();
            register({registerTool(tool) {tools.set(tool.name, tool)}, on() {}});
            const tool = tools.get('hamon_conservation_cad_verifier');
            const result = await tool.execute('tube', {mode:'exchangers_radiation', spec:JSON.parse(process.argv[1])});
            const reference = await tool.execute('tube-info', {mode:'reference'});
            console.log(JSON.stringify([reference.content[0].text, result.details]));
        """
        env = dict(os.environ, TASK_HARNESS_SRC=str(root), TASK_HARNESS_PYTHON=sys.executable,
                   TASK_RELAY_URL="http://127.0.0.1:1", OPENBLAS_NUM_THREADS="1")
        run = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script,
                              json.dumps(spec)], cwd=root, env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        description, result = json.loads(run.stdout)
        self.assertIn("tube_resistance", description)
        self.assertTrue(result["verified"], result)
        self.assertAlmostEqual(result["UA_W_K"], flux_oracle(TUBE), places=7)
