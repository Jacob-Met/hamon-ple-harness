"""Exercise the conservative qualification boundary for two curved cutouts."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET

from hamon_eagle_copper import inter_net_contact
from test_eagle_board_domain import board, dimension_records, edge, nominal_board_domain, rectangle
from test_eagle_circle import ring


class TwoCircleTests(unittest.TestCase):
    def test_guarded_clearance(self):
        radius = Fraction(1, 8)
        for shift in (0, 100):
            for gap_nm in (-1, 0, 11, 12, 13, 30):
                with self.subTest(shift=shift, gap_nm=gap_nm):
                    gap = Fraction(gap_nm, 1_000_000)
                    left = ring((2, 2), center=(shift-radius, shift), radius=radius)
                    right = ring((1, 1, 1, 1), center=(shift+radius+gap, shift), radius=radius)
                    border = rectangle(shift-2, shift+2, shift-2, shift+2)
                    _, document = board(border + left + right)
                    records = dimension_records(document)
                    if gap_nm <= 12:
                        with self.assertRaises(ValueError):
                            nominal_board_domain(records)
                    else:
                        domain, info = nominal_board_domain(records)
                        self.assertEqual(len(domain.interiors), 2)
                        self.assertGreater(info["guarded_hole_separation_nm"], 4)
                        self.assertEqual([hole["radius_squared_nm2"] for hole in info["dimension_holes"]],
                                         ["15625000000"] * 2)

    def test_noninteger_radii(self):
        # A pair of antipodal endpoints at (+/-0.1, +/-0.1) mm gives
        # r^2 = 20,000,000,000 nm^2. The exact integer upper radius is 141422 nm.
        def diagonal_ring(x):
            d = Fraction(1, 10)
            a, b = (x-d, -d), (x+d, d)
            return [edge(a, b, curve=180), edge(b, a, curve=180)]

        for distance in (282856, 282857):
            rows = rectangle() + diagonal_ring(0) + diagonal_ring(Fraction(distance, 1_000_000))
            _, document = board(rows)
            records = dimension_records(document)
            if distance == 282856:
                with self.assertRaises(ValueError):
                    nominal_board_domain(records)
            else:
                _, info = nominal_board_domain(records)
                self.assertEqual([hole["radius_squared_nm2"] for hole in info["dimension_holes"]],
                                 ["20000000000"] * 2)

    def test_contact_witness_in_each_surviving_corner(self):
        rows = rectangle() + ring((2, 2), radius=Fraction(1, 2))
        rows += ring((1, 1, 1, 1), center=(Fraction(85, 100), 0), radius=Fraction(16, 100))
        for x in (-380_000, 380_000):
            for y in (-380_000, 380_000):
                root, document = board(rows)
                for signal, (dx, dy) in zip(document.findall("signals/signal"), ((60_000, 0), (0, 60_000))):
                    signal.remove(signal.find("wire"))
                    a = (Fraction(x-dx, 1_000_000), Fraction(y-dy, 1_000_000))
                    b = (Fraction(x+dx, 1_000_000), Fraction(y+dy, 1_000_000))
                    signal.append(edge(a, b, layer=1, width=".016"))
                with TemporaryDirectory() as temp:
                    path = Path(temp)/"board.brd"
                    ET.ElementTree(root).write(path)
                    result = inter_net_contact(path, ("A", "B"))
                self.assertEqual(result["board_domain"]["status"], "bounded")
                hit, = result["contacts"]
                cx, cy = map(Fraction, hit["center_nm"])
                radius = Fraction(hit["disk_radius_nm"])
                self.assertGreater(radius, 0)
                self.assertLessEqual(abs(cx-x) + radius, 8_000)
                self.assertLessEqual(abs(cy-y) + radius, 8_000)
                self.assertGreater(cx*cx + cy*cy, (500_000 + radius)**2)
                self.assertGreater((cx-850_000)**2 + cy*cy, (160_000 + radius)**2)
