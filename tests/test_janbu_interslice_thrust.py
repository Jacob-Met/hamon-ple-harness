"""Free-end slice thrust on published geometry and independent two-slice roots."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mpmath import mp

from hamon_janbu_slices import _raw_interval_forces, verify_janbu_slices
from test_janbu_slices import ibaraki


def table(rows):
    return {"calculation": "janbu_slices", "slice_order": "toe_to_crest", "slices": [
        dict(id=str(i), width_m=1, weight_N_m=w, base_angle_deg=a, phi_deg=p,
             c_prime_Pa=c, pore_pressure_Pa=u)
        for i, (w, a, p, c, u) in enumerate(rows, 1)]}


def quadratic_reference(rows):
    # Multiply the two slice denominators; use the larger quadratic root.
    ctx = mp.clone()
    ctx.dps = 170
    terms = []
    for w, a, phi, c, u in rows:
        a = ctx.radians(ctx.mpf(a))
        tangent = ctx.tan(ctx.radians(ctx.mpf(phi)))
        k = (ctx.mpf(c) + (ctx.mpf(w) - ctx.mpf(u))*tangent) / ctx.cos(a)**2
        terms.append((k, -ctx.tan(a)*tangent, ctx.mpf(w)*ctx.tan(a)))
    (k1, p1, d1), (k2, p2, d2) = terms
    d = d1 + d2
    b = d*(p1+p2) + k1 + k2
    root = (b + ctx.sqrt(b*b - 4*d*(d*p1*p2 + k1*p2 + k2*p1))) / (2*d)
    return root, k1/(root-p1) - d1


class OrderedJanbuTests(unittest.TestCase):
    def test_dry_wet_and_reversed_physical_order(self):
        for count, water, expected_f, expected_verified in [
                (6, None, 1.8293049027220651, True), (33, 4., 1.4256398870992784, False)]:
            spec = dict(ibaraki(count, water), slice_order="toe_to_crest", required_factor_of_safety=1.4)
            out = verify_janbu_slices(spec)
            with self.subTest(count=count):
                self.assertTrue(out["solution_found"], out.get("reason"))
                self.assertTrue(out["base_contact_verified"])
                self.assertAlmostEqual(out["factor_of_safety"], expected_f, places=13)
                self.assertTrue(out["interslice_force_balance_verified"], out.get("reason"))
                self.assertEqual(out["verified"], expected_verified)
                self.assertEqual(out["meets_required_factor"], expected_verified)
                self.assertEqual(len(out["interfaces"]), count-1)
                reverse = verify_janbu_slices(dict(spec, slices=spec["slices"][::-1],
                                                   slice_order="crest_to_toe"))
                self.assertEqual(reverse["verified"], expected_verified)
                for a, b in zip(out["interfaces"], reversed(reverse["interfaces"])):
                    self.assertEqual(a["between_slice_ids"], b["between_slice_ids"][::-1])
                    self.assertEqual(a["sign"], b["sign"])
                    self.assertEqual(a["total_thrust_interval_N_m"], b["total_thrust_interval_N_m"])
                if water is not None:
                    face = out["interfaces"][-1]
                    self.assertEqual(face["between_slice_ids"], ["32", "33"])
                    self.assertEqual(face["sign"], "tension")
                    for endpoint in face["total_thrust_interval_N_m"]:
                        self.assertAlmostEqual(endpoint, -5131.370650288813, places=8)

    def test_tensile_or_unresolved_interfaces_cannot_certify(self):
        c = 100000000012 * math.ulp(0.)
        big = 2.**1000
        w = (10000000020 * math.ulp(0.))*big
        cases = [
            [(.5, 45, 0, c/2, 0), (1., 45, 0, c, 0)],
            [(math.nextafter(.5, 1), 45, 0, c/2, 0), (1., 45, 0, c, 0)],
            [(w/2, 45, 0, w/4, 0), (big, 0, 0, 0, 0), (w, 45, 0, w/2, 0)],
            [(1, 45, 89.999, 0, 0), (1, 45, 89.999, 14323.944876775484, .25)],
            [(1, 45, 89.999999, 0, 0), (1, 45, 89.999999, 14323944.9340238, .25)],
        ]
        for i, rows in enumerate(cases):
            with self.subTest(i=i):
                out = verify_janbu_slices(dict(table(rows), required_factor_of_safety=.1))
                self.assertFalse(out["verified"], out)
                self.assertFalse(out["meets_required_factor"])
                self.assertTrue(all(f["sign"] != "compression" for f in out["interfaces"]))
                json.dumps(out, allow_nan=False)

    def test_closed_form_two_slice_enclosures_and_claimed_signs(self):
        cases = [[(1, -a, 10, 1e-6, 0), (2, a, 10, 1, 0)]
                 for a in [80, 89.999, math.nextafter(90., 0.)]]
        cases += [[(0, -45, 45, 10.**-k, 0), (1, 45, 0, .01, 0)] for k in [0, 10, 50, 70]]
        cases += [[(s, 20, 30, .3*s, 0), (2*s, 40, 10, .4*s, .2*s)]
                  for s in [1e-302, 1e250, 1e300]]
        for i, rows in enumerate(cases):
            reference, thrust = quadratic_reference(rows)
            with self.subTest(i=i):
                root, (force,) = _raw_interval_forces(table(rows), float(reference))
                self.assertLessEqual(root[0], reference)
                self.assertGreaterEqual(root[1], reference)
                self.assertLessEqual(force[0], thrust)
                self.assertGreaterEqual(force[1], thrust)
                out = verify_janbu_slices(table(rows))
                for face in out.get("interfaces", []):
                    if face["sign"] == "compression":
                        self.assertGreater(thrust, 0)
                    if face["sign"] == "tension":
                        self.assertLess(thrust, 0)

    def test_optional_order_threshold_and_invalid_inputs(self):
        spec = ibaraki(6)
        self.assertNotIn("interfaces", verify_janbu_slices(spec))
        self.assertFalse(verify_janbu_slices(dict(spec, slice_order="unordered"))["verified"])
        ordered = dict(spec, slice_order="toe_to_crest")
        lo, hi = verify_janbu_slices(ordered)["factor_of_safety_interval"]
        self.assertTrue(verify_janbu_slices(dict(ordered, required_factor_of_safety=lo))["meets_required_factor"])
        self.assertFalse(verify_janbu_slices(dict(ordered, required_factor_of_safety=math.nextafter(lo, math.inf)))["meets_required_factor"])
        for invalid in (float("inf"), 10**400):
            out = verify_janbu_slices(table([(invalid, 45, 10, 1, 0)]))
            self.assertFalse(out["verified"])

    def test_registered_wet_surface(self):
        root = Path(__file__).resolve().parents[1]
        source = """
import registerTaskTools from './task-tools.ts';
const tools = new Map();
registerTaskTools({registerTool(t) { tools.set(t.name, t); }, on() {}});
const tool = tools.get('hamon_conservation_cad_verifier');
const r = await tool.execute('janbu-wet-33', {mode:'civil_geotech',
  spec: JSON.parse(process.env.JANBU_SPEC)}, null, () => {}, {});
console.log(JSON.stringify(r.details));
"""
        spec = dict(ibaraki(33, 4.), slice_order="toe_to_crest", required_factor_of_safety=1.4)
        with tempfile.TemporaryDirectory() as work:
            env = {**os.environ, "TASK_WORKDIR": work, "TASK_HARNESS_SRC": str(root),
                   "TASK_HARNESS_PYTHON": sys.executable, "TASK_RELAY_URL": "http://127.0.0.1:1",
                   "JANBU_SPEC": json.dumps(spec)}
            proc = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", source],
                                  cwd=root, env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout.splitlines()[-1])
        self.assertTrue(out["equilibrium_verified"], out.get("reason"))
        self.assertEqual(out["tensile_interfaces"], [["32", "33"]])
        self.assertFalse(out["meets_required_factor"])
        self.assertFalse(out["verified"])


if __name__ == "__main__":
    unittest.main()
