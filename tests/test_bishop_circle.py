"""Specified Bishop circles through the public registered endpoints."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hamon_bishop_circle import verify_bishop_circle

ROOT = Path(__file__).resolve().parents[1]


def slice_data(left, right, weight, cohesion, phi, pressure):
    return dict(left_x_m=left, right_x_m=right, weight_N_per_m=weight,
                cohesion_Pa=cohesion, friction_angle_deg=phi, pore_pressure_Pa=pressure)


def circle(slices, direction="right"):
    return dict(calculation="bishop_slip_circle", circle=dict(center_x_m=0, center_y_m=13, radius_m=10),
                slide_direction=direction, slices=slices)


def registered(cases):
    script = """
        import fs from 'node:fs';
        import register from './task-tools.ts';
        const registry = new Map();
        register({registerTool(tool) { registry.set(tool.name, tool); }, on() {}});
        const tool = registry.get('hamon_conservation_cad_verifier');
        const answers = [];
        for (const payload of JSON.parse(fs.readFileSync(0, 'utf8'))) {
            answers.push((await tool.execute('bishop', payload)).details);
        }
        console.log(JSON.stringify(answers));
    """
    env = dict(os.environ, TASK_HARNESS_SRC=str(ROOT), TASK_HARNESS_PYTHON=sys.executable,
               TASK_RELAY_URL="http://127.0.0.1:1", OPENBLAS_NUM_THREADS="1")
    run = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                         cwd=ROOT, input=json.dumps(cases), capture_output=True, text=True, env=env, timeout=120)
    if run.returncode:
        raise RuntimeError(run.stderr)
    return json.loads(run.stdout)


def manufactured(factor, near_pole=False):
    # Choose contact and water forces first. Determine the first shear from
    # signed moments, then invert the material and vertical force laws.
    edges = [-10., -4., 2., 8.] if near_pole else [-9., -5., -1., 3.]
    normal = [900., 720., 1.] if near_pole else [900., 720., 300.]
    water = [100., 200., 12.] if near_pole else [100., 200., 15.]
    phi = [0., 0., 45.] if near_pole else [32., 18., 22.]
    shear = [0., 100., 10.] if near_pole else [0., 280., 172.77283365678574]
    sine = [-(left+right)/20 for left, right in zip(edges, edges[1:])]
    cosine = [math.sqrt(1-v*v) for v in sine]
    shear[0] = (sum((n+u)*s*c for n,u,s,c in zip(normal,water,sine,cosine))
                - sum(q*c*c for q,c in zip(shear[1:],cosine[1:]))) / cosine[0]**2
    rows = []
    for i in range(3):
        length = (edges[i+1]-edges[i])/cosine[i]
        weight = (normal[i]+water[i])*cosine[i] + shear[i]*sine[i]
        strength = (factor*shear[i]-normal[i]*math.tan(math.radians(phi[i]))) / length
        rows.append(slice_data(edges[i], edges[i+1], weight, strength, phi[i], water[i]/length))
    return circle(rows), normal, shear


class BishopTests(unittest.TestCase):
    def test_registered_dry_wet_and_mirrored_single_slice(self):
        s = .5
        c = math.sqrt(1-s*s)
        wet = slice_data(-6, -4, 1000, 120, 30, 95)
        dry = dict(wet, pore_pressure_Pa=0)
        mirror = dict(wet, left_x_m=4, right_x_m=6)
        # With one slice, force resolution on the tangent and normal gives
        # the solution without any iterative factor-of-safety equation.
        setup = [circle([dry]), circle([wet]), circle([mirror], "left")]
        wrong = {**setup[1], "proposed_factor_of_safety": 20}
        out = registered([dict(mode="auto", spec=setup[0]), dict(mode="civil_geotech", spec=setup[1]),
                          dict(mode="civil", spec=setup[2]), dict(mode="geotech", spec=wrong)])
        for spec, result in zip(setup, out):
            with self.subTest(spec=spec):
                self.assertTrue(result["conservation_verified"], result)
                row, = spec["slices"]
                n = 1000*c - row["pore_pressure_Pa"]*2/c
                expected = (120*2/c+n*math.tan(math.pi/6))/(1000*s)
                self.assertAlmostEqual(result["factor_of_safety"], expected, places=12)
                self.assertAlmostEqual(result["slices"][0]["effective_normal_N_per_m"], n, places=10)
        self.assertLess(out[1]["factor_of_safety"], out[0]["factor_of_safety"])
        self.assertTrue(out[3]["ok"])
        self.assertFalse(out[3]["conservation_verified"])

    def test_real_dual_verify_cli(self):
        spec = circle([slice_data(-6, -4, 1000, 120, 30, 95)])
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "slope.json"
            for proposed, expected in ((1.299, False), (None, True)):
                inputs = dict(spec)
                if proposed is not None:
                    inputs["proposed_factor_of_safety"] = proposed
                file.write_text(json.dumps(dict(kind="bishop_slip_circle", spec=inputs)))
                proc = subprocess.run([sys.executable, str(ROOT/"engineering_toolkit.py"), "dual-verify", str(file)],
                                      capture_output=True, text=True, cwd=ROOT, timeout=60)
                result = json.loads(proc.stdout)
                self.assertEqual(result["dual_verified"], expected, result)

    def test_signed_forces_and_moments_across_uphill_toe(self):
        factor = 1.37
        spec, normals, shears = manufactured(factor)
        answer = verify_bishop_circle(spec)
        self.assertTrue(answer["conservation_verified"], answer)
        self.assertAlmostEqual(answer["factor_of_safety"], factor, places=12)
        torque = 0.
        for raw, row, n, q in zip(spec["slices"], answer["slices"], normals, shears):
            self.assertAlmostEqual(row["effective_normal_N_per_m"], n, places=9)
            self.assertAlmostEqual(row["mobilized_shear_N_per_m"], q, places=9)
            x = .5*(raw["left_x_m"]+raw["right_x_m"])
            y = -math.sqrt(100.-x*x)
            # Inward normal follows the radius. Resisting shear opposes motion.
            nx, ny = -x/10., -y/10.
            load = n + raw["pore_pressure_Pa"]*(raw["right_x_m"]-raw["left_x_m"])/ny
            fx = load*nx - q*ny
            fy = load*ny + q*nx - raw["weight_N_per_m"]
            self.assertAlmostEqual(fy, 0., places=10)
            torque += x*fy - y*fx
        self.assertAlmostEqual(torque, 0., places=9)
        self.assertLess(answer["slices"][-1]["downhill_base_angle_deg"], 0.)
        self.assertLess(answer["equilibrium_rel_residual"], 1e-10)

    def test_admissibility_and_pole_resolution(self):
        pole = 1 / math.sqrt(3)
        spec, normal, _ = manufactured(pole+1e-5, near_pole=True)
        near = verify_bishop_circle(spec)
        self.assertTrue(near["conservation_verified"], near)
        self.assertAlmostEqual(near["factor_of_safety"], pole+1e-5, places=11)
        self.assertAlmostEqual(near["slices"][-1]["effective_normal_N_per_m"], normal[-1], places=7)
        unresolved, _, _ = manufactured(pole+1e-15, near_pole=True)
        base = slice_data(-6,-4,1000,300,30,0)
        for invalid in [unresolved, circle([dict(base,pore_pressure_Pa=500)]),
                        circle([base],"left"), circle([dict(base,left_x_m=-11)]),
                        circle([base,dict(base,left_x_m=-3,right_x_m=-2)]),
                        {**circle([base]), "seismic_coefficient": .2},
                        circle([dict(base,friction_angle_deg=90)])]:
            with self.subTest(spec=invalid):
                result = verify_bishop_circle(invalid)
                self.assertFalse(result["conservation_verified"], result)
                self.assertFalse(result["ok"], result)
                self.assertTrue(result["reason"])
