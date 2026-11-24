"""Check rational quadratic Body sweeps through the public IFC clash receiver."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import ifcopenshell

from hamon_ifc_spatial_clash import run_router_ifc_clash
from test_ifc_indexed_curves import fixture

POINTS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
          (0, -1), (1, -1), (1, 0)]


def model(weight, offsets=(0.,), mm=False):
    text = fixture([(0, 0), (1, 0), (1, 1), (0, 1)],
                   [("IfcLineIndex", (1, 2, 3, 4, 1))], offsets=offsets, mm=mm)
    f = ifcopenshell.file.from_string(text)
    scale = 1000.0 if mm else 1.0
    points = [f.create_entity("IfcCartesianPoint", tuple(x*scale for x in pair)) for pair in POINTS]
    curve = f.create_entity("IfcRationalBSplineCurveWithKnots", 2, points, "UNSPECIFIED",
                            True, False, [3, 2, 2, 2, 3], [0., 1., 2., 3., 4.],
                            "PIECEWISE_BEZIER_KNOTS", [1., weight]*4 + [1.])
    f.by_type("IfcArbitraryClosedProfileDef")[0].OuterCurve = curve
    return f


def public(f):
    return run_router_ifc_clash({"ifc_text": f.to_string(), "cross_discipline_only": False,
                                 "required_clearance_m": .04})


def reference_area(weight):
    # Sample the actual parametric curve in source order at 32,768 points.
    # This polygon integral is independent of the source-area calculation.
    points = []
    n = 8192
    for i in range(0, 8, 2):
        a, b, c = POINTS[i:i+3]
        for j in range(n):
            t = j/n
            weights = ((1.-t)**2, 2.*weight*t*(1.-t), t*t)
            points.append(tuple(sum(k*p[dim] for k, p in zip(weights, (a,b,c)))/sum(weights)
                                for dim in (0,1)))
    return .5*abs(math.fsum(a[0]*b[1] - a[1]*b[0] for a,b in zip(points, points[1:]+points[:1])))


class RationalProfile(unittest.TestCase):
    def test_weights_and_units_change_source_volume(self):
        for weight, mm in ((.5, False), (1., True), (2., False)):
            with self.subTest(weight=weight, mm=mm):
                report = public(model(weight, mm=mm))
                self.assertTrue(report["ok"], report)
                self.assertAlmostEqual(report["total_solid_volume_m3"], .2*reference_area(weight), delta=2e-8)

    def test_reverse_orientation_and_concave_control_polygon(self):
        f = model(.5)
        ring = f.by_type("IfcRationalBSplineCurveWithKnots")[0]
        ring.ControlPointsList = tuple(reversed(ring.ControlPointsList))
        ring.WeightsData = tuple(reversed(ring.WeightsData))
        result = public(f)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["total_solid_volume_m3"], .2*reference_area(.5), delta=2e-8)
        # Pull the north-west outer control into the diamond without opening
        # or crossing the ring. A control hull alone cannot certify its chords.
        ring.ControlPointsList[3].Coordinates = (-.1, -.1)
        result = public(f)
        self.assertFalse(result["ok"], result)
        self.assertIsNone(result["total_solid_volume_m3"])

    def test_hole_in_peer_profile_never_certifies_solid_material(self):
        f = model(.5)
        make = f.create_entity
        base = f.by_type("IfcExtrudedAreaSolid")[0]
        corners = [make("IfcCartesianPoint", xy) for xy in
                   ((-.1,-.1),(.1,-.1),(.1,.1),(-.1,.1))]
        hole = make("IfcPolyline", [*corners, corners[0]])
        profile = make("IfcArbitraryProfileDefWithVoids", "AREA", None, base.SweptArea.OuterCurve, [hole])
        body = make("IfcExtrudedAreaSolid", profile, base.Position, base.ExtrudedDirection, base.Depth)
        ctx = f.by_type("IfcProject")[0].RepresentationContexts[0]
        rep = make("IfcShapeRepresentation", ctx, "Body", "SweptSolid", [body])
        make("IfcColumn", ifcopenshell.guid.new(), None, "void peer", None, None,
             f.by_type("IfcSlab")[0].ObjectPlacement, make("IfcProductDefinitionShape",None,None,[rep]))
        result = public(f)
        self.assertFalse(result["ok"], result)
        self.assertIsNone(result["total_solid_volume_m3"])

    def test_nested_far_origin_keeps_a_small_authored_gap(self):
        f = model(.5, (0., .225))
        make = f.create_entity
        parent = make("IfcLocalPlacement", None, make("IfcAxis2Placement3D",
                      make("IfcCartesianPoint", (1e17, -1e17, 1e17))))
        for slab in f.by_type("IfcSlab"):
            slab.ObjectPlacement.PlacementRelTo = parent
        result = public(f)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["all_pairs"][0]["status"], "clearance_violation")
        self.assertAlmostEqual(result["min_clearance_m"], .025, places=12)

    def test_gap_contact_and_overlap(self):
        for gap, status in ((.025, "clearance_violation"), (.3, "clear"),
                            (0., "hard_clash"), (-.05, "hard_clash")):
            with self.subTest(gap=gap):
                report = public(model(.5, (0., .2+gap)))
                self.assertTrue(report["ok"], report)
                self.assertEqual(report["all_pairs"][0]["status"], status)
                self.assertAlmostEqual(report["min_clearance_m"], max(0., gap), places=8)

    def test_identical_control_polygon_changes_lateral_material(self):
        # At x=y the rational boundary is exactly (w+1/2)/(w+1).
        # The 10 mm probe lies outside the w=1/2 solid, inside w=2.
        for weight, expected in ((.5, "clear"), (2., "hard_clash")):
            f = model(weight)
            make = f.create_entity
            p = make("IfcCartesianPoint", (.75, .75, .05))
            loc = make("IfcLocalPlacement", None, make("IfcAxis2Placement3D", p))
            profile = make("IfcRectangleProfileDef", "AREA", None, None, .01, .01)
            body = make("IfcExtrudedAreaSolid", profile, None, make("IfcDirection", (0., 0., 1.)), .1)
            ctx = f.by_type("IfcProject")[0].RepresentationContexts[0]
            rep = make("IfcShapeRepresentation", ctx, "Body", "SweptSolid", [body])
            make("IfcColumn", ifcopenshell.guid.new(), None, "probe", None, None, loc,
                 make("IfcProductDefinitionShape", None, None, [rep]))
            with self.subTest(weight=weight):
                result = public(f)
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["all_pairs"][0]["status"], expected)
                boundary = (weight+.5)/(weight+1)
                distance = max(0., math.sqrt(2)*(.745-boundary))
                self.assertAlmostEqual(result["min_clearance_m"], distance, places=7)
                lo,hi = result["all_pairs"][0]["distance_bounds_m"]
                self.assertLessEqual(lo, distance+8*math.ulp(distance))
                self.assertGreaterEqual(hi, distance-8*math.ulp(distance))
                self.assertLessEqual(hi-lo, 1e-8)

    def test_open_crossing_or_unsupported_span_abstains(self):
        for change in ("open", "crossing", "weight-ratio", "knot-span"):
            f = model(.5)
            curve = f.by_type("IfcRationalBSplineCurveWithKnots")[0]
            if change == "open":
                curve.ControlPointsList[-1].Coordinates = (1., .125)
            elif change == "crossing":
                curve.ControlPointsList[4].Coordinates = (1.2, .3)
            elif change == "weight-ratio":
                curve.WeightsData = (1., .01, 1., .5, 1., .5, 1., .5, 1.)
            else:
                curve.KnotMultiplicities = (3, 1, 3, 2, 3)
            with self.subTest(change=change):
                result = public(f)
                self.assertFalse(result["ok"], result)
                self.assertIsNone(result["total_solid_volume_m3"])


    def test_registered_relay_reads_rational_curve(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "node_modules").is_dir():
            self.skipTest("registered Node receiver runs in the complete CI checkout")
        from test_wired_router_e2e import _start_mock_relay
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / "rational.ifc").write_text(model(.5).to_string())
            calls = []
            relay = _start_mock_relay(work, calls)
            try:
                script = """
import register from './task-tools.ts';
const tools = new Map();
register({registerTool(t) {tools.set(t.name,t)}, on() {}});
const r = await tools.get('hamon_ifc_spatial_clash').execute(
    'source-nurbs', {path:'rational.ifc'}, null, () => {}, {});
console.log(JSON.stringify(r.details));
"""
                env = {**os.environ, "TASK_WORKDIR": temp, "TASK_HARNESS_PYTHON": sys.executable,
                       "TASK_HARNESS_SRC": str(root),
                       "TASK_RELAY_URL": f"http://127.0.0.1:{relay.server_address[1]}"}
                run = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                                     env=env, cwd=root, text=True, capture_output=True, timeout=30, check=True)
                data = json.loads(run.stdout.strip().splitlines()[-1])
                self.assertTrue(data["ok"], data)
                self.assertAlmostEqual(data["total_solid_volume_m3"], .2*reference_area(.5), delta=2e-8)
                self.assertEqual(calls, ["read_file"])
            finally:
                relay.shutdown()
                relay.server_close()


if __name__ == "__main__":
    unittest.main()
