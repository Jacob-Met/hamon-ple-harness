import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hamon_cad_takeoff import scan_dxf_takeoff

class DxfBulges(unittest.TestCase):
    def measured(self, vertices, closed=True, units=6):
        tags = [(0, 'SECTION'), (2, 'HEADER'), (9, '$INSUNITS'), (70, units),
                (0, 'ENDSEC'), (0, 'SECTION'), (2, 'ENTITIES'), (0, 'LWPOLYLINE'),
                (8, 'OUTLINE'), (90, len(vertices)), (70, int(closed))]
        for x, y, bulge in vertices:
            tags.extend([(10, x), (20, y)])
            if bulge:
                tags.append((42, bulge))
        tags.extend([(0, 'ENDSEC'), (0, 'EOF')])
        with TemporaryDirectory() as work:
            path = Path(work) / 'curve.dxf'
            path.write_text(''.join(f'{code}\n{value}\n' for code, value in tags))
            result = scan_dxf_takeoff(path)
        return (result['closed_area_m2_by_layer'].get('OUTLINE', 0.0),
                result['polyline_perimeter_m_by_layer']['OUTLINE'])

    def test_two_vertex_circle_both_windings(self):
        for sign in (-1, 1):
            area, length = self.measured([(12, 7, sign), (16, 7, sign)], units=4)
            self.assertAlmostEqual(area, 4 * math.pi / 1_000_000, places=12)
            self.assertAlmostEqual(length, 4 * math.pi / 1000, places=12)

    def test_clockwise_closing_major_arc(self):
        bend = -(1 + math.sqrt(2))
        area, length = self.measured([(0, 0, 0), (2, 0, bend)])
        self.assertAlmostEqual(area, 1 + 1.5 * math.pi)
        self.assertAlmostEqual(length, 2 + 1.5 * math.pi * math.sqrt(2))

    def test_mixed_signed_segments(self):
        # Equal opposite semicircles on parallel edges cancel their signed areas.
        area, length = self.measured([(0, 0, 1), (2, 0, 0), (2, 4, -1), (0, 4, 0)])
        self.assertAlmostEqual(area, 8)
        self.assertAlmostEqual(length, 8 + 2 * math.pi)

    def test_unused_open_final_bulge(self):
        area, length = self.measured([(0, 0, 1), (2, 0, 9)], closed=False)
        self.assertEqual(area, 0.0)
        self.assertAlmostEqual(length, math.pi)

    def test_tiny_curvature_on_large_drawing_offset(self):
        b = 1e-9
        area, length = self.measured([(1e7, 1e7, b), (1e7 + 3, 1e7, 0)])
        self.assertAlmostEqual(area, 3*b, delta=1e-20)
        self.assertAlmostEqual(length, 6, places=12)

    def test_ordinary_rectangle(self):
        self.assertEqual(self.measured([(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)]),
                         (12, 14))
