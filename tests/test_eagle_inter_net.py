"""Board-file controls for nominal inter-net wire overlap."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from hamon_eagle_copper import inter_net_contact


def wire(x1, y1, x2, y2, layer=1, **kwargs):
    fields = dict(x1=x1, y1=y1, x2=x2, y2=y2, width=.2, layer=layer, **kwargs)
    return ET.Element("wire", {key: str(value) for key, value in fields.items()})


class WireContactTests(unittest.TestCase):
    def query(self, left, right, holes=(), other=(), names=("A", "B")):
        root = ET.Element("eagle")
        board = ET.SubElement(ET.SubElement(root, "drawing"), "board")
        plain = ET.SubElement(board, "plain")
        for x, y, diameter in holes:
            ET.SubElement(plain, "hole", x=str(x), y=str(y), drill=str(diameter))
        signals = ET.SubElement(board, "signals")
        for name, rows in [("A", left), ("B", right), ("C", other)]:
            ET.SubElement(signals, "signal", name=name).extend(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.brd"
            ET.ElementTree(root).write(path, encoding="utf-8")
            result = inter_net_contact(path, names)
            self.assertEqual(result["board_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["clearance"], "not_certified")
        return result

    def test_crossing_area_and_interior_disk(self):
        for layer in (1, 16):
            left = [wire(-1, 0, 1, 0, layer)]
            right = [wire(0, -1, 0, 1, layer)]
            for names in (("B", "A"), ("A", "B")):
                result = self.query(left, right, names=names)
                self.assertEqual(result["status"], "contact_witnessed")
                contact, = result["contacts"]
                self.assertEqual(contact["layer"], layer)
                self.assertEqual(contact["sources"], [
                    {"signal": name, "primitive": ("wire", 0)} for name in names])
                x, y = contact["center_nm"]
                radius = contact["disk_radius_nm"]
                self.assertGreater(radius, 0)
                self.assertLess(max(abs(x), abs(y)) + radius, 100_000)
                self.assertTrue(.03999 < contact["overlap_lower_bound_mm2"] <= .04)

    def test_tangency_other_layer_and_unsupported_artwork_stay_unresolved(self):
        left = [wire(-1, 0, 1, 0)]
        for other in [wire(-1, .2, 1, .2), wire(0, -1, 0, 1, 16),
                      ET.Element("polygon", layer="1")]:
            result = self.query(left, [other])
            self.assertEqual((result["status"], result["contacts"]), ("unresolved", []))
            if other.tag == "polygon":
                self.assertEqual(result["unsupported"][0]["signal"], "B")
        with self.assertRaises(ValueError):
            self.query(left, [], names=("A", "A"))

    def test_board_and_third_signal_drills_remove_overlap(self):
        left, right = [wire(-1, 0, 1, 0)], [wire(0, -1, 0, 1)]
        self.assertFalse(self.query(left, right, holes=[(0, 0, .4)])["contacts"])
        via = ET.Element("via", x="0", y="0", drill=".4", extent="1-16")
        self.assertFalse(self.query(left, right, other=[via])["contacts"])
        self.assertTrue(self.query(left, right, holes=[(1, 1, .4)])["contacts"])

    def test_signed_arc_follows_bow_instead_of_chord(self):
        for sign in (-1, 1):
            arc = wire(-1, 0, 1, 0, curve=180 * sign)
            on_bow = self.query([arc], [wire(0, -.8 * sign, 0, -1.2 * sign)])
            contact, = on_bow["contacts"]
            x, y = contact["center_nm"]
            self.assertLess(abs((x*x + y*y)**.5 - 1e6) + contact["disk_radius_nm"], 1e5)
            self.assertFalse(self.query([arc], [wire(0, -.2, 0, .2)])["contacts"])
