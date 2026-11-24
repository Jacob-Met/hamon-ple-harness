"""Check indexed arc extrusions through the registered clash engine's public receiver."""
import math
import unittest

import ifcopenshell
from hamon_ifc_spatial_clash import run_router_ifc_clash


def fixture(points, segments, offsets=(0.0,), *, mm=False, direction=(0., 0., 1.)):
    f = ifcopenshell.file(schema="IFC4")
    make = f.create_entity
    point = lambda xyz: make("IfcCartesianPoint", tuple(float(x) for x in xyz))
    place = lambda xyz: make("IfcAxis2Placement3D", point(xyz))
    pose = place((0, 0, 0))
    ctx = make("IfcGeometricRepresentationContext", None, "Model", 3, 1e-7, pose)
    unit = make("IfcSIUnit", None, "LENGTHUNIT", "MILLI" if mm else None, "METRE")
    make("IfcProject", ifcopenshell.guid.new(), None, "project", None, None, None, None,
         [ctx], make("IfcUnitAssignment", [unit]))
    factors = 1000 if mm else 1
    stored = [tuple(float(v*factors) for v in p) for p in points]
    parts = [make(kind, ids) for kind, ids in segments]
    curve = make("IfcIndexedPolyCurve", make("IfcCartesianPointList2D", stored), parts, False)
    profile = make("IfcArbitraryClosedProfileDef", "AREA", None, curve)
    sweep = make("IfcExtrudedAreaSolid", profile, pose, make("IfcDirection", direction), .2*factors)
    rep = make("IfcShapeRepresentation", ctx, "Body", "SweptSolid", [sweep])
    shape = make("IfcProductDefinitionShape", None, None, [rep])
    for off in offsets:
        loc = make("IfcLocalPlacement", None, place((0, 0, off*factors)))
        make("IfcSlab", ifcopenshell.guid.new(), None, "slab", None, None, loc, shape)
    return f.to_string()


POINTS = [(0, 0), (1, 0), (1.4, 2), (1, 4), (0, 4), (-.4, 2)]
EDGES = [("IfcLineIndex", (1, 2)), ("IfcArcIndex", (2, 3, 4)),
         ("IfcLineIndex", (4, 5)), ("IfcArcIndex", (5, 6, 1))]
RADIUS = 5.2
ANGLE = 2*math.asin(2/RADIUS)
AREA = 4 + RADIUS**2 * (ANGLE - math.sin(ANGLE))


def check(text, clearance=.05):
    return run_router_ifc_clash({"ifc_text": text, "cross_discipline_only": False,
                                 "required_clearance_m": clearance})


class IndexedCurves(unittest.TestCase):
    def test_authored_arc_direction_storage_permutation_and_units(self):
        for mm in (False, True):
            for direction in ((0., 0., 1.), (3., 0., -4.)):
                order = [4, 0, 3, 1, 5, 2]
                points = [POINTS[i] for i in order] + [(900., 600.)]
                remap = lambda i: order.index(i-1)+1
                for reverse in (False, True):
                    segments = EDGES if not reverse else [(k, ids[::-1]) for k, ids in EDGES[::-1]]
                    segments = [(k, tuple(remap(i) for i in ids)) for k, ids in segments]
                    report = check(fixture(points, segments, mm=mm, direction=direction))
                    self.assertTrue(report["ok"], report)
                    self.assertEqual(report["solid_count"], 1)
                    self.assertAlmostEqual(report["total_solid_volume_m3"],
                                           AREA*.2*abs(direction[2])/math.hypot(*direction), places=10)

    def test_three_physical_pair_outcomes_from_native_shapes(self):
        for offset, expected, distance in ((.1, "hard_clash", 0), (.21, "clearance_violation", .01),
                                            (1.2, "clear", 1)):
            r = check(fixture(POINTS, EDGES, (0., offset)))
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["all_pairs"][0]["status"], expected)
            self.assertAlmostEqual(r["min_clearance_m"], distance, places=9)

    def test_indexed_line_order_and_invalid_source_refusals(self):
        wrapped = [(1, 1), (0, 0), (17, 23), (1, 0), (0, 1)]
        straight = check(fixture(wrapped, [("IfcLineIndex", (2, 4, 1, 5, 2))]))
        self.assertAlmostEqual(straight["total_solid_volume_m3"], .2)
        for edges in (EDGES[:-1], [EDGES[0], ("IfcArcIndex", (3, 2, 4)), *EDGES[2:]],
                      [EDGES[0], ("IfcArcIndex", (2, 3, 17)), *EDGES[2:]]):
            r = check(fixture(POINTS, edges))
            self.assertFalse(r["ok"])
            self.assertIsNone(r["total_solid_volume_m3"])
        collinear = list(POINTS)
        collinear[2] = (1, 2)
        self.assertFalse(check(fixture(collinear, EDGES))["ok"])
