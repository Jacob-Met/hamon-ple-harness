
"""Unilateral contact from independently integrated nonnegative pressure fields."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F

from hamon_bearing_capacity import verify_rigid_strip_contact
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_dual_verifier import run_dual_verify_spec
from hamon_rankine import verify_rankine_layers

ROOT = Path(__file__).resolve().parents[1]
BASE = {"calculation": "rigid_strip_contact", "base_width_m": 3,
        "vertical_resultant_N_m": 100000, "moment_Nm_m": 144000}


ACTION_BASE = {"calculation": "rigid_strip_contact", "base_width_m": 3}


def horizontal_pair(sign=1, upper_height=3):
    return {**ACTION_BASE, "applied_loads": [
        {"x_m": 0, "vertical_N_m": 100000},
        {"height_m": upper_height, "horizontal_N_m": sign*48000},
        {"height_m": 0, "horizontal_N_m": -sign*48000},
    ]}


def simpson_loads(left, right, p_left, p_right):
    """Exact forward integration, not the solver's inverse contact equations."""
    middle = (left + right) / 2
    p_middle = (p_left + p_right) / 2
    normal = (right-left) * (p_left + 4*p_middle + p_right) / 6
    moment = (right-left) * (left*p_left + 4*middle*p_middle + right*p_right) / 6
    return normal, moment


def from_pressure(width, left, right, p_left, p_right):
    normal, moment = simpson_loads(left, right, p_left, p_right)
    return {"calculation": "rigid_strip_contact", "base_width_m": float(width),
            "vertical_resultant_N_m": float(normal), "moment_Nm_m": float(moment)}


class RigidStripContact(unittest.TestCase):
    def assertClose(self, actual, expected, scale=1):
        self.assertTrue(math.isclose(actual, float(expected), rel_tol=2e-12,
                                     abs_tol=2e-13*scale), (actual, expected))

    def test_wall_couple_reaches_contact_without_a_horizontal_force_location(self):
        wall = verify_rankine_layers({
            "pressure_state": "active",
            "layers": [{"thickness_m": 6, "phi_deg": 30, "gamma_above_water_N_m3": 18000}],
            "front_layers": [{"thickness_m": 2, "phi_deg": 30, "gamma_above_water_N_m3": 18000}],
        })
        self.assertTrue(wall["conservation_verified"], wall)
        self.assertEqual(wall["thrust_N_m"], 0)
        self.assertEqual(wall["moment_Nm_m"], 144000)
        out = run_router_conservation_cad_verifier({**BASE, "moment_Nm_m": wall["moment_Nm_m"]})
        self.assertTrue(out["conservation_verified"], out)
        self.assertEqual(out["contact_regime"], "partial_contact")
        self.assertEqual(out["contact_width_m"], 0.18)
        self.assertEqual(out["contact_interval_m"], [1.32, 1.5])
        self.assertEqual(out["q_max_Pa"], float(F(10000000, 9)))
        self.assertEqual(out["q_left_Pa"], 0)
        self.assertFalse(verify_rigid_strip_contact({
            **BASE, "proposed_contact_width_m": 0.12})["conservation_verified"])

    def test_balanced_horizontal_actions_reduce_to_the_signed_couple(self):
        for sign in (-1, 1):
            for height in (0, 1, 3):
                with self.subTest(sign=sign, height=height):
                    reduced = run_router_conservation_cad_verifier(horizontal_pair(sign, height))
                    self.assertEqual(reduced.pop("horizontal_resultant_N_m"), 0)
                    expected = verify_rigid_strip_contact({**BASE, "moment_Nm_m": sign*48000*height})
                    self.assertEqual(reduced, expected)
                    self.assertTrue(reduced["conservation_verified"])
        dual = run_dual_verify_spec({"mode": "domain", "kind": "rigid_strip_contact",
                                     "spec": horizontal_pair(-1)})
        self.assertTrue(dual["dual_verified"], dual)
        self.assertEqual(dual["contact_interval_m"], [-1.5, -1.32])

    def test_force_components_origin_and_free_couple(self):
        # V = 100 kN/m. Clockwise M = 80 + 75 + 50 - 80 = 125 kN m/m.
        for dx, dz in ((0, 0), (-8, 100)):
            actions = [
                {"vertical_N_m": "80*kN/m", "x_m": 2+dx,
                 "horizontal_N_m": "25*kN/m", "height_m": 5+dz},
                {"vertical_N_m": "20*kN/m", "x_m": 1+dx},
                {"horizontal_N_m": "-25*kN/m", "height_m": dz},
                {"couple_Nm_m": "-80*kN*m/m"},
            ]
            for ordered in (actions, list(reversed(actions))):
                out = verify_rigid_strip_contact({
                    **ACTION_BASE, "base_center": {"x_m": 1+dx, "height_m": 2+dz},
                    "applied_loads": ordered})
                self.assertEqual(out.pop("horizontal_resultant_N_m"), 0)
                self.assertEqual(out, verify_rigid_strip_contact({**BASE, "moment_Nm_m": 125000}))
        # The large opposing horizontal contributions must not erase a smaller action.
        tiny = verify_rigid_strip_contact({**ACTION_BASE, "applied_loads": [
            {"horizontal_N_m": 2**60, "height_m": 1},
            {"vertical_N_m": 10, "x_m": 0.125},
            {"horizontal_N_m": -(2**60), "height_m": 1}]})
        self.assertTrue(tiny["conservation_verified"], tiny)
        self.assertEqual(tiny["moment_Nm_m"], 1.25)

    def test_unbalanced_actions_and_malformed_inputs_are_uncertified(self):
        for residual in (1, 2**-35):
            spec = horizontal_pair()
            spec["applied_loads"].append({"horizontal_N_m": residual, "height_m": 0})
            out = verify_rigid_strip_contact(spec)
            self.assertEqual(out["horizontal_resultant_N_m"], residual)
            self.assertEqual(out["moment_Nm_m"], 144000)
            self.assertFalse(out["conservation_verified"])
            self.assertFalse(out["ok"])
        for spec in (
            {**BASE, "applied_loads": []}, {**BASE, "base_center": {"x_m": 0, "height_m": 0}},
            {**horizontal_pair(), "moment_Nm_m": 0},
            {**horizontal_pair(), "base_center": {"x_m": 0}},
            *({**ACTION_BASE, "applied_loads": loads} for loads in (
                [], {}, [None], [{}], [{"horizontal_N_m": 0}], [{"height_m": 2}],
                [{"vertical_N_m": 1, "x_m": "2*Pa"}], [{"couple_Nm_m": True}],
                [{"x_m": 0, "vertical_N_m": 100, "unknown": 9}])),
        ):
            with self.subTest(spec=spec):
                self.assertFalse(verify_rigid_strip_contact(spec)["ok"])

    def test_independently_generated_pressure_fields_and_reflections(self):
        count = 0
        for width in (F(3, 2), F(3), F(6), F(12)):
            half = width / 2
            fields = [
                (-half, half, F(24), F(24)),
                (-half, half, F(0), F(24)),
                (-half, half, F(1, 2**20), F(24)-F(1, 2**20)),
            ]
            for extent in (width/4, width/2, 7*width/8, width-F(1, 2**20)):
                fields.append((half-extent, half, F(0), F(24)))
            for left, right, pa, pb in fields:
                for sign in (1, -1):
                    a, b, qa, qb = ((left, right, pa, pb) if sign > 0
                                     else (-right, -left, pb, pa))
                    for scale in (F(1, 16), F(1), F(128)):
                        qa_s, qb_s = qa*scale, qb*scale
                        spec = from_pressure(width, a, b, qa_s, qb_s)
                        out = verify_rigid_strip_contact(spec)
                        with self.subTest(spec=spec):
                            self.assertTrue(out["conservation_verified"], out)
                            self.assertClose(out["contact_width_m"], b-a, float(width))
                            self.assertClose(out["contact_interval_m"][0], a, float(width))
                            self.assertClose(out["contact_interval_m"][1], b, float(width))
                            self.assertClose(out["q_max_Pa"], max(qa_s, qb_s), float(scale))
                            actual_n, actual_m = simpson_loads(
                                *map(F, out["contact_interval_m"]),
                                *map(F, out["contact_edge_pressure_Pa"]))
                            self.assertClose(float(actual_n), spec["vertical_resultant_N_m"])
                            self.assertClose(float(actual_m), spec["moment_Nm_m"],
                                             abs(spec["vertical_resultant_N_m"]*float(width)))
                        count += 1
        self.assertEqual(count, 168)

    def test_exact_middle_third_and_its_neighbours(self):
        for sign in (-1, 1):
            for moment, regime in (
                (math.nextafter(72., 0.), "full_contact"),
                (72., "full_contact"),
                (math.nextafter(72., math.inf), "partial_contact"),
            ):
                out = verify_rigid_strip_contact({
                    **BASE, "base_width_m": 6, "vertical_resultant_N_m": 72,
                    "moment_Nm_m": sign*moment})
                self.assertTrue(out["conservation_verified"], out)
                self.assertEqual(out["contact_regime"], regime)
                if moment == 72:
                    self.assertEqual(out["contact_width_m"], 6)
                    self.assertEqual(out["q_min_Pa"], 0)
                    self.assertEqual(out["q_max_Pa"], 24)

    def test_thin_contact_and_no_finite_edge_equilibrium(self):
        width = F(1, 2**30)
        spec = from_pressure(F(6), F(3)-width, F(3), F(0), F(2**31))
        for sign in (-1, 1):
            out = verify_rigid_strip_contact({**spec, "moment_Nm_m": sign*spec["moment_Nm_m"]})
            self.assertTrue(out["conservation_verified"], out)
            self.assertEqual(out["contact_regime"], "partial_contact")
            self.assertGreater(out["contact_width_m"], 0)
            for edge in (3., math.nextafter(3., math.inf), 4.):
                refused = verify_rigid_strip_contact({
                    **spec, "moment_Nm_m": sign*edge})
                self.assertFalse(refused["conservation_verified"])
                self.assertEqual(refused["equilibrium_status"], "no_finite_equilibrium")
                self.assertNotIn("q_max_Pa", refused)

    def test_unloaded_tension_and_unbalanced_pure_couple(self):
        out = verify_rigid_strip_contact({**BASE, "vertical_resultant_N_m": 0,
                                         "moment_Nm_m": 0})
        self.assertTrue(out["conservation_verified"], out)
        self.assertEqual(out["contact_regime"], "unloaded")
        self.assertIsNone(out["contact_interval_m"])
        self.assertIsNone(out["contact_width_m"])
        self.assertIsNone(out["eccentricity_m"])
        self.assertEqual(out["q_max_Pa"], 0)
        for normal, moment in ((-1, 0), (0, 1), (0, -1)):
            refused = verify_rigid_strip_contact({
                **BASE, "vertical_resultant_N_m": normal, "moment_Nm_m": moment})
            self.assertFalse(refused["conservation_verified"], refused)
            self.assertEqual(refused["equilibrium_status"], "no_finite_equilibrium")

    def test_unit_conversion_and_dimension_rejection(self):
        converted = verify_rigid_strip_contact({
            **BASE, "base_width_m": "3000*mm",
            "vertical_resultant_N_m": "100*kN/m", "moment_Nm_m": "144*kN*m/m"})
        self.assertEqual(converted, verify_rigid_strip_contact(BASE))
        for key, value in (("base_width_m", "3*s"),
                           ("vertical_resultant_N_m", "100*kN"),
                           ("moment_Nm_m", "144*kN*m"),
                           ("base_width_m", "3"), ("base_width_m", "3*")):
            out = verify_rigid_strip_contact({**BASE, key: value})
            self.assertFalse(out["conservation_verified"], out)

    def test_claims_cannot_certify_the_wrong_contact_diagram(self):
        good = {"proposed_contact_regime": "partial_contact",
                "proposed_contact_interval_m": ["1320*mm", "1500*mm"],
                "proposed_contact_width_m": "180*mm",
                "proposed_q_max_Pa": "10000000*Pa/9"}
        self.assertTrue(verify_rigid_strip_contact({**BASE, **good})["conservation_verified"])
        for key, value in (("proposed_q_max_Pa", 33333.3333333333),
                           ("proposed_q_left_Pa", -62666.666666667),
                           ("proposed_contact_regime", "full_contact"),
                           ("proposed_contact_interval_m", [-1.5, -1.32]),
                           ("proposed_contact_width_m", 0.12),
                           ("proposed_eccentricity_m", -1.44),
                           ("proposed_q_min_Pa", True)):
            out = verify_rigid_strip_contact({**BASE, key: value})
            self.assertFalse(out["conservation_verified"], out)


    def test_near_edge_proposed_intervals_preserve_their_own_geometry(self):
        # Independent review: absolute position tolerance had swallowed an
        # entire contact span. The three bad spans carry 0, -1/3 and 1/6
        # of the required force under the returned triangular pressures.
        gap = 2.0**-34
        good = [1 - 3*gap, 1.]
        bad = [[1., 1.], [1., 1-gap], [1-gap/2, 1.],
               [good[0]-gap, 1-gap], [good[0]+gap, 1+gap]]
        for sign in (-1, 1):
            spec = {**BASE, "base_width_m": 2, "vertical_resultant_N_m": 1,
                    "moment_Nm_m": sign*(1-gap)}
            reflect = lambda pair: pair if sign > 0 else [-pair[1], -pair[0]]
            solved = verify_rigid_strip_contact(spec)
            self.assertTrue(solved["output_geometry_verified"], solved)
            valid = verify_rigid_strip_contact({
                **spec, "proposed_contact_interval_m": reflect(good),
                "proposed_contact_width_m": 3*gap,
                "proposed_q_max_Pa": solved["q_max_Pa"]})
            self.assertTrue(valid["conservation_verified"], valid)
            for pair in bad:
                with self.subTest(sign=sign, proposed=pair):
                    out = verify_rigid_strip_contact({
                        **spec, "proposed_contact_interval_m": reflect(pair)})
                    self.assertTrue(out["ok"], out)
                    self.assertFalse(out["claim_checks"]["proposed_contact_interval_m"], out)
                    self.assertFalse(out["conservation_verified"], out)

    def test_certified_returned_intervals_remain_valid_claims(self):
        count = 0
        for exponent in range(20, 51):
            for sign in (-1, 1):
                spec = {**BASE, "base_width_m": 2, "vertical_resultant_N_m": 1,
                        "moment_Nm_m": sign*(1-2.0**-exponent)}
                solved = verify_rigid_strip_contact(spec)
                self.assertTrue(solved["output_geometry_verified"], solved)
                out = verify_rigid_strip_contact({
                    **spec,
                    "proposed_contact_interval_m": solved["contact_interval_m"],
                    "proposed_contact_width_m": solved["contact_width_m"],
                    "proposed_q_left_Pa": solved["q_left_Pa"],
                    "proposed_q_right_Pa": solved["q_right_Pa"],
                    "proposed_q_max_Pa": solved["q_max_Pa"]})
                self.assertTrue(out["conservation_verified"], out)
                count += 1
        self.assertEqual(count, 62)

    def test_model_domain_and_nonfinite_inputs(self):
        for key, value in (("base_width_m", 0), ("base_width_m", -3),
                           ("base_width_m", True), ("moment_Nm_m", math.nan),
                           ("vertical_resultant_N_m", math.inf),
                           ("pressure_model", "meyerhof_effective_area"),
                           ("contact_patch", [1.39, 1.49]),
                           ("footing_L_m", 10), ("proposed_pressure", 1e6)):
            out = verify_rigid_strip_contact({**BASE, key: value})
            self.assertFalse(out["conservation_verified"], out)
        for key in ("base_width_m", "vertical_resultant_N_m", "moment_Nm_m"):
            missing = dict(BASE)
            del missing[key]
            self.assertFalse(verify_rigid_strip_contact(missing)["ok"])

    def test_pressure_range_and_displayed_geometry_are_honest(self):
        overflow = verify_rigid_strip_contact({
            **BASE, "base_width_m": 1e-300, "vertical_resultant_N_m": 1e300,
            "moment_Nm_m": 0})
        self.assertFalse(overflow["conservation_verified"], overflow)
        # M/N rounds to B/2, although exact parsed inputs have a finite strip.
        # Rounded endpoints lose ~1/6 of its width: do not certify that geometry.
        out = verify_rigid_strip_contact({
            **BASE, "base_width_m": 0.2, "vertical_resultant_N_m": 1e16,
            "moment_Nm_m": 1e15})
        self.assertEqual(out["equilibrium_status"], "finite")
        self.assertEqual(out["contact_regime"], "partial_contact")
        self.assertFalse(out["output_geometry_verified"], out)
        self.assertFalse(out["conservation_verified"], out)

    def test_router_and_provisioned_domain_route(self):
        expected = verify_rigid_strip_contact(BASE)
        for payload in (BASE, {"mode": "civil_geotech", "spec": BASE}):
            self.assertEqual(run_router_conservation_cad_verifier(payload), expected)
        dual = run_dual_verify_spec({"mode": "domain", "kind": "rigid_strip_contact",
                                     "spec": BASE})
        self.assertTrue(dual["dual_verified"], dual)
        self.assertEqual(dual["contact_interval_m"], expected["contact_interval_m"])

    def test_source_inline_without_the_authoring_checkout(self):
        names = ("hamon_bearing_capacity", "hamon_dual_verifier",
                 "hamon_conservation_cad_verifier")
        bundle = {"sources": {n: (ROOT/(n+".py")).read_text() for n in names},
                  "payload": {**BASE, "base_width_m": "3000*mm"}}
        runner = """
import json,sys,types
bundle=json.loads(sys.stdin.read())
for name,source in bundle['sources'].items():
    module=types.ModuleType(name);module.__file__='/inline/'+name+'.py'
    sys.modules[name]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
print(json.dumps(run_router_conservation_cad_verifier(bundle['payload']),allow_nan=False))
"""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-I", "-c", runner], cwd=tmp,
                                    input=json.dumps(bundle), text=True, capture_output=True,
                                    timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), verify_rigid_strip_contact(BASE))

    def test_real_registered_pi_tool(self):
        if not (ROOT/"node_modules/@earendil-works/pi-coding-agent").exists():
            self.skipTest("registered Pi runtime not installed")
        cases = [BASE, horizontal_pair(), horizontal_pair(-1),
                 {**ACTION_BASE, "applied_loads": [{"height_m": 4, "horizontal_N_m": 1}]},
                 {**BASE, "moment_Nm_m": -144000},
                 {**BASE, "moment_Nm_m": 0}, {**BASE, "moment_Nm_m": 150000},
                 {**BASE, "base_width_m": "3000*mm"},
                 {**BASE, "proposed_contact_width_m": .12},
                 {**BASE, "base_width_m": 2, "vertical_resultant_N_m": 1,
                  "moment_Nm_m": 1-2.0**-34, "proposed_contact_interval_m": [1., 1.]},
                 {**BASE, "base_width_m": 2, "vertical_resultant_N_m": 1,
                  "moment_Nm_m": -(1-2.0**-34), "proposed_contact_interval_m": [-1., -1.]}]
        script = """
import registerTaskTools from './task-tools.ts';
const tools=new Map();
registerTaskTools({registerTool(t){tools.set(t.name,t);},on(){}});
const tool=tools.get('hamon_conservation_cad_verifier');
const results=[];
for(const spec of JSON.parse(process.env.CONTACT_CASES)){
 const result=await tool.execute('contact',{mode:'civil_geotech',spec},null,()=>{},{});
 results.push(result.details);
}
console.log(JSON.stringify(results));
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "TASK_RELAY_URL": "http://127.0.0.1:1",
                   "TASK_WORKDIR": tmp, "TASK_HARNESS_SRC": str(ROOT),
                   "TASK_HARNESS_PYTHON": sys.executable, "CONTACT_CASES": json.dumps(cases)}
            result = subprocess.run(
                ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(actual, [verify_rigid_strip_contact(s) for s in cases])


if __name__ == "__main__":
    unittest.main()
