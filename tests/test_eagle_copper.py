"""Independent synthetic EAGLE corner cases for routed copper."""

import tempfile
import unittest
from pathlib import Path
from hamon_eagle_copper import routed_continuity


def wire(x1, y1, x2, y2, layer=1):
    return f'<wire x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" width=".1" layer="{layer}"/>'


class CopperWitnessTests(unittest.TestCase):
    def routed(
        self, artwork, end=(2, 1), rotation="R0", pad='smd dx=".2" dy=".2" layer="1"'
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.brd"
            path.write_text(
                f"""<eagle><drawing><board><libraries><library name="parts"><packages>
              <package name="A"><smd name="p" x="0" y="0" dx=".2" dy=".2" layer="1"/></package>
              <package name="B"><{pad} name="p" x="0" y="0"/></package>
              </packages></library></libraries><elements>
              <element name="A" library="parts" package="A" x="0" y="0"/>
              <element name="B" library="parts" package="B" x="{end[0]}" y="{end[1]}" rot="{rotation}"/>
              </elements><signals><signal name="x">{artwork}</signal></signals></board></drawing></eagle>"""
            )
            prediction = {
                "schematic_sha256": "test",
                "physical_contacts": [
                    {"net": "x", "part": i, "pad": "p"} for i in ("A", "B")
                ],
            }
            return routed_continuity(prediction, path, "x")

    def test_witness_at_interior_routed_vertex(self):
        out = self.routed(wire(0, 0, 4, 0) + wire(2, 0, 2, 1))
        self.assertEqual(out["status"], "witnessed")
        steps = out["witnesses"][0]["steps"]
        self.assertEqual(
            [s["source"] for s in steps],
            [["terminal", "A", "p"], ["wire", 0], ["wire", 1], ["terminal", "B", "p"]],
        )
        self.assertEqual(steps[1]["end"], ["copper", 1, 2000000, 0])

    def test_layer_crossings_require_plating(self):
        traces = wire(0, 0, 2, 0) + wire(2, 0, 2, 1, 16)
        via = '<via x="2" y="0" drill=".25" extent="1-16"/>'
        self.assertEqual(self.routed(traces, rotation="MR0")["status"], "unresolved")
        self.assertEqual(
            self.routed(traces + via, rotation="MR0")["status"], "witnessed"
        )
        out = self.routed(traces + via.replace("1-16", "1-2"), rotation="MR0")
        self.assertEqual(out["status"], "unresolved")
        self.assertTrue(out["unsupported"])

    def test_drill_void_is_not_conductive_landing(self):
        out = self.routed(
            wire(0, 0, 0.1, 0), end=(0.1, 0), pad='pad drill="1" diameter="1.5"'
        )
        self.assertEqual(out["reason"], "trace_does_not_demonstrably_reach_plating")
        self.assertFalse(out["witnesses"])

    def test_unsupported_artwork_does_not_prove_continuity(self):
        for item in (
            wire(0, 0, 2, 1, 19),
            '<polygon layer="1" width=".1"/>',
            wire(0, 0, 2, 1).replace("/>", ' curve="90"/>'),
        ):
            with self.subTest(item=item):
                result = self.routed(item)
                self.assertEqual(result["status"], "unresolved")
                self.assertTrue(result["unsupported"])
        self.assertEqual(
            self.routed(wire(0, 0, 2, 1), rotation="R17")["status"], "unresolved"
        )


if __name__ == "__main__":
    unittest.main()
