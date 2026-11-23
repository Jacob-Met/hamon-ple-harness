"""Compare every guarded stadium edge and contact against the analytic spine."""
from fractions import Fraction as F
import unittest

from shapely.geometry import LineString

from test_eagle_board_domain import board, dimension_records, edge, nominal_board_domain, rectangle, reverse_rows


def stadium(quarters, rotated):
    def world(x, y):
        return ((3*x-4*y)/5, (4*x+3*y)/5) if rotated else (x, y)
    vertices = [(-1,-1), (1,-1), (2,0), (1,1), (-1,1), (-2,0)]
    angles = [0, 90, 90, 0, 90, 90]
    if not quarters:
        vertices = [vertices[i] for i in (0,1,3,4)]
        angles = [0, 180, 0, 180]
    points = [world(*p) for p in vertices]
    return [edge(p, q, curve=s) for p,q,s in zip(points, points[1:]+points[:1], angles)]


class StadiumTests(unittest.TestCase):
    def test_full_guarded_rim_and_provenance(self):
        for quarters in (False, True):
            for rotated in (False, True):
                rows = stadium(quarters, rotated)
                rows = reverse_rows(rows)[::2] + reverse_rows(rows)[1::2]
                _, doc = board(rectangle(-4,4,-4,4) + rows)
                domain, proof = nominal_board_domain(dimension_records(doc))
                caps = proof["hole_caps"]
                spine = LineString([[float(F(v)) for v in cap["center_nm"]] for cap in caps])
                expected = [(-600000,-800000),(600000,800000)] if rotated else [(-1000000,0),(1000000,0)]
                self.assertCountEqual(list(spine.coords), expected)
                self.assertCountEqual([s[-1] for cap in caps for s in cap["arc_sources"]],
                                      [4+i for i,row in enumerate(rows) if F(row.get("curve"))])
                ring = list(domain.interiors[0].coords)
                for a,b in zip(ring, ring[1:]):
                    self.assertGreaterEqual(spine.distance(LineString([a,b])), 1_000_000)
                self.assertLessEqual(proof["hole_arc_error_upper_bound_nm"], 1)

    def test_unsupported_joins_and_nesting(self):
        opening = stadium(True, False)
        for key, value in (("curve","-90"), ("x2","2.1")):
            broken = stadium(True, False)
            broken[1].set(key, value)
            if key == "x2":
                broken[2].set("x1", value)
            _, doc = board(rectangle(-4,4,-4,4) + broken)
            with self.assertRaises(ValueError):
                nominal_board_domain(dimension_records(doc))
        for rows in (opening, rectangle(-4,4,-4,4)+opening+rectangle(-.1,.1,-.1,.1)):
            _, doc = board(rows)
            with self.assertRaises(ValueError):
                nominal_board_domain(dimension_records(doc))
