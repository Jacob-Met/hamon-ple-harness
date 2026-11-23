"""Independent whole-edge, trace and normal-route checks for circular cutouts."""
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET

from shapely.geometry import LineString, Point
from hamon_eagle_copper import inter_net_contact
from test_eagle_board_domain import board, dimension_records, edge, nominal_board_domain, rectangle, reverse_rows


def ring(steps, *, center=(0, 0), radius=1, rotated=False):
    axes = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def point(i):
        x, y = axes[i % 4]
        if rotated:
            x, y = F(3*x-4*y, 5), F(4*x+3*y, 5)
        return center[0]+radius*x, center[1]+radius*y

    rows = []
    i = 0
    for step in steps:
        rows.append(edge(point(i), point(i+step), curve=90*step))
        i += step
    return rows


class CircularOpeningTests(unittest.TestCase):
    def test_guarded_contours_and_trace_clearance(self):
        center = (F(1, 8), F(-3, 8))
        origin = Point(125_000, -375_000)
        for steps in ((2,2), (2,1,1), (1,1,1,1)):
            for rotated in (False, True):
                circle = ring(steps, center=center, rotated=rotated)
                for rows in (circle, reverse_rows(circle)):
                    _, drawing = board(rectangle(-4,4,-4,4) + rows[::2] + rows[1::2])
                    domain, proof = nominal_board_domain(dimension_records(drawing))
                    self.assertEqual(proof["hole_center_nm"], ["125000", "-375000"])
                    self.assertCountEqual([r[-1] for r in proof["hole_sources"]], range(4,4+len(rows)))
                    self.assertLessEqual(proof["hole_arc_error_upper_bound_nm"], 1)
                    boundary = list(domain.interiors[0].coords)
                    for a,b in zip(boundary,boundary[1:]):
                        self.assertGreaterEqual(origin.distance(LineString([a,b])), 1_000_000)
                    for offset in (0, 1_005_000, 1_010_000, 1_020_000):
                        # The nearest point is inside the segment, not an endpoint.
                        y = -375_000 + offset
                        spine = LineString([(-1_500_000,y),(1_500_000,y)])
                        expected = offset > 1_000_000 + 10_000
                        self.assertEqual(domain.covers(spine.buffer(10_000)), expected)

    def test_registered_contacts(self):
        for steps in ((2,2), (2,1,1), (1,1,1,1)):
            for center, count in (((F(1,2),F(1,2)),1), ((0,0),0)):
                for rotated in (False, True):
                    rows = ring(steps, center=center, radius=F(1,4), rotated=rotated)
                    root,_ = board(rectangle()+rows)
                    with TemporaryDirectory() as tmp:
                        path = Path(tmp)/"layout.brd"
                        ET.ElementTree(root).write(path)
                        result = inter_net_contact(path,("A","B"))
                    self.assertEqual(result["board_domain"]["status"], "bounded")
                    self.assertEqual(len(result["contacts"]), count)
                    for hit in result["contacts"]:
                        x,y = map(F,hit["center_nm"])
                        r = F(hit["disk_radius_nm"])+250_000
                        self.assertGreater((x-center[0]*1_000_000)**2+(y-center[1]*1_000_000)**2,r*r)

    def test_malformed_and_unqualified_contours(self):
        retraced = ring((2,2))
        retraced[1].set("curve","-180")
        inconsistent = ring((1,1,1,1))
        inconsistent[0].set("x2",".1")
        inconsistent[1].set("x1",".1")
        for rows in (rectangle()+retraced, rectangle()+inconsistent, rectangle()+ring((2,2),center=(1,0)),
                     ring((2,2)), rectangle()+ring((2,2))+rectangle(-.1,.1,-.1,.1)):
            _,drawing=board(rows)
            with self.assertRaises(ValueError):
                nominal_board_domain(dimension_records(drawing))
