"""Board-file controls for nominal inter-net wire overlap."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from hamon_eagle_copper import inter_net_contact, routed_continuity


def wire(x1, y1, x2, y2, layer=1, **kwargs):
    fields = dict(x1=x1, y1=y1, x2=x2, y2=y2, width=.2, layer=layer, **kwargs)
    return ET.Element("wire", {key: str(value) for key, value in fields.items()})


class WireContactTests(unittest.TestCase):
    def query(self, left, right, holes=(), other=(), names=("A", "B"), milling=(), layout=None):
        root = ET.Element("eagle")
        board = ET.SubElement(ET.SubElement(root, "drawing"), "board")
        plain = ET.SubElement(board, "plain")
        plain.extend(milling)
        if layout:
            layout(board)
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

    @staticmethod
    def outline(xmin, xmax, ymin=-.2, ymax=.2, layer=46):
        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        return [ET.Element("wire", x1=str(a[0]), y1=str(a[1]), x2=str(b[0]),
                           y2=str(b[1]), width="0", layer=str(layer))
                for a, b in zip(corners, corners[1:] + corners[:1])]

    def test_milled_intersection_is_clipped_before_measuring_the_witness_disk(self):
        left, right = [wire(-1, 0, 1, 0)], [wire(0, -1, 0, 1)]
        self.assertFalse(self.query(left, right, milling=self.outline(-.2, .2))["contacts"])
        self.assertFalse(self.query(left, right, milling=self.outline(-.2, .2, layer="046"))["contacts"])
        partial, = self.query(left, right, milling=self.outline(-.2, .05))["contacts"]
        self.assertGreater(partial["center_nm"][0] - partial["disk_radius_nm"], 50_000)
        self.assertAlmostEqual(partial["overlap_lower_bound_mm2"], .01, places=5)
        for outline in (self.outline(9, 10), self.outline(-.2, .2, layer=51)):
            row, = self.query(left, right, milling=outline)["contacts"]
            self.assertAlmostEqual(row["overlap_lower_bound_mm2"], .04, places=5)
        for outline in (self.outline(-.2, .2)[:-1], [ET.Element("circle", layer="46")]):
            result = self.query(left, right, milling=outline)
            self.assertEqual(result["contacts"], [])
            self.assertIn("layer 46", result["detail"])

    def test_only_placed_package_milling_removes_board_copper(self):
        def layout(placed):
            def insert(board):
                lib = ET.SubElement(ET.SubElement(board, "libraries"), "library", name="holes")
                package = ET.SubElement(ET.SubElement(lib, "packages"), "package", name="CUT")
                package.extend(self.outline(-.2, .2, 1.8, 2.2))
                if placed:
                    ET.SubElement(ET.SubElement(board, "elements"), "element", name="J1",
                                  library="holes", package="CUT", x="2", y="0", rot="R90")
            return insert
        for placed in (False, True):
            result = self.query([wire(-1, 0, 1, 0)], [wire(0, -1, 0, 1)], layout=layout(placed))
            self.assertEqual(len(result["contacts"]), 0 if placed else 1)

    def test_milling_blocks_the_same_net_centerline_and_finite_width_path(self):
        root = ET.fromstring("""<eagle><drawing><board><plain/><libraries><library name="Pads">
          <packages><package name="pair"><smd name="L" x="-1" y="0" dx=".2" dy=".2" layer="1"/>
          <smd name="R" x="1" y="0" dx=".2" dy=".2" layer="1"/></package></packages></library>
          </libraries><elements><element name="P" library="Pads" package="pair" x="0" y="0"/></elements>
          <signals><signal name="A"/></signals></board></drawing></eagle>""")
        root.find(".//signal").append(wire(-1, 0, 1, 0))
        prediction = {"physical_contacts": [{"part": "P", "pad": pin, "net": "A"} for pin in ("L", "R")],
                      "schematic_sha256": "fixture"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.brd"
            for x, expected in ((9, "witnessed"), (-.05, "unresolved")):
                plain = root.find(".//plain")
                plain.clear()
                plain.extend(self.outline(x, x+.1))
                ET.ElementTree(root).write(path, encoding="utf-8")
                self.assertEqual(routed_continuity(prediction, path, "A")["status"], expected)

    def test_milling_that_reaches_a_via_bore_breaks_its_full_collar_proof(self):
        from shapely.geometry import box
        from hamon_eagle_material import material_continuity

        pads = [("pad", "P", "top"), ("pad", "P", "bottom")]
        via = ("via", 0)
        primitives = [dict(source=via, layers=(1, 16), shape="disk", center=(0, 0),
                           diameter=400_000, plated=True)]
        for pad, layer in zip(pads, (1, 16)):
            primitives.append(dict(source=pad, terminal=pad, layers=(layer,), shape="rectangle",
                                   center=(-120_000, 0), size=(80_000, 80_000)))
        bores = [(0, 0, 100_000, via)]
        intact = material_continuity(pads, primitives, bores)
        self.assertEqual(len(intact["witnesses"]), 1)
        notch = box(40_000, -10_000, 210_000, 10_000)
        milled = material_continuity(pads, primitives, bores, notch)
        self.assertFalse(milled["witnesses"])
