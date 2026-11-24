"""Check rational quadratic Body sweeps through the public IFC clash receiver."""
import math
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

    def test_gap_contact_and_overlap(self):
        for gap, status in ((.025, "clearance_violation"), (.3, "clear"),
                            (0., "hard_clash"), (-.05, "hard_clash")):
            with self.subTest(gap=gap):
                report = public(model(.5, (0., .2+gap)))
                self.assertTrue(report["ok"], report)
                self.assertEqual(report["all_pairs"][0]["status"], status)
                self.assertAlmostEqual(report["min_clearance_m"], max(0., gap), places=8)

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


if __name__ == "__main__":
    unittest.main()
