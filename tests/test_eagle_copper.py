"""Independent synthetic EAGLE corner cases for routed copper."""

import tempfile
import unittest
from pathlib import Path
from hamon_eagle_copper import routed_continuity


def wire(x1, y1, x2, y2, layer=1):
    return f'<wire x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" width=".1" layer="{layer}"/>'


class CopperWitnessTests(unittest.TestCase):
    def routed(
        self, artwork, end=(2, 1), rotation="R0", pad='smd dx=".2" dy=".2" layer="1"',
        plain="", package="", other_signals=""
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.brd"
            path.write_text(
                f"""<eagle><drawing><board><plain>{plain}</plain><libraries><library name="parts"><packages>
              <package name="A"><smd name="p" x="0" y="0" dx=".2" dy=".2" layer="1"/></package>
              <package name="B"><{pad} name="p" x="0" y="0"/>{package}</package>
              </packages></library></libraries><elements>
              <element name="A" library="parts" package="A" x="0" y="0"/>
              <element name="B" library="parts" package="B" x="{end[0]}" y="{end[1]}" rot="{rotation}"/>
              </elements><signals><signal name="x">{artwork}</signal>{other_signals}</signals></board></drawing></eagle>"""
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

    def test_witness_requires_continuous_trace_artwork(self):
        direct = wire(0, 0, 40, 0)
        detour = wire(0, 0, 0, 2) + wire(0, 2, 40, 2) + wire(40, 2, 40, 0)
        for style in ("longdash", "shortdash", "dashdot"):
            with self.subTest(style=style):
                marked = direct.replace("/>", f' style="{style}"/>')
                result = self.routed(marked, end=(40, 0))
                self.assertEqual(result["status"], "unresolved")
                self.assertEqual(result["unsupported"][0]["source"], ("wire", 0))
                self.assertFalse(result["witnesses"])

                result = self.routed(marked + detour, end=(40, 0))
                self.assertEqual(result["status"], "witnessed")
                sources = [step["source"] for step in result["witnesses"][0]["steps"]]
                self.assertEqual([s for s in sources if s[0] == "wire"],
                                 [["wire", 1], ["wire", 2], ["wire", 3]])
        for continuous in (direct, direct.replace("/>", ' style="continuous"/>')):
            self.assertEqual(self.routed(continuous, end=(40, 0))["status"], "witnessed")

    def test_layer_crossings_require_plating(self):
        traces = wire(0, 0, 2, 0) + wire(2, 0, 2, 1, 16)
        via = '<via x="2" y="0" drill=".25" extent="1-16"/>'
        self.assertEqual(self.routed(traces, rotation="MR0")["status"], "unresolved")
        routed = self.routed(traces + via, rotation="MR0")
        self.assertEqual(routed["status"], "witnessed")
        self.assertEqual([step["plated_approach"] for step in routed["witnesses"][0]["steps"]
                          if "plated_approach" in step], [["via", 0], ["via", 0]])
        out = self.routed(traces + via.replace("1-16", "1-2"), rotation="MR0")
        self.assertEqual(out["status"], "unresolved")
        self.assertTrue(out["unsupported"])

    def test_drill_void_is_not_conductive_landing(self):
        out = self.routed(
            wire(0, 0, 0.1, 0), end=(0.1, 0), pad='pad drill="1" diameter="1.5"'
        )
        self.assertEqual(out["reason"], "trace_does_not_demonstrably_reach_plating")
        self.assertFalse(out["witnesses"])

    def test_board_drill_cut_tangent_remote_and_detour(self):
        direct = wire(0, 0, 4, 0)
        detour = wire(0, 0, 0, 1) + wire(0, 1, 4, 1) + wire(4, 1, 4, 0)
        # Tangency to the centerline still leaves half the trace conductive.
        for y, status in [(0, "unresolved"), (".30", "witnessed"),
                          (".300001", "witnessed")]:
            with self.subTest(y=y):
                hole = f'<hole x="2" y="{y}" drill=".6"/>'
                self.assertEqual(self.routed(direct, end=(4, 0), plain=hole)["status"], status)
        out = self.routed(direct + detour, end=(4, 0), plain='<hole x="2" y="0" drill=".6"/>')
        self.assertEqual(out["status"], "witnessed")
        self.assertEqual([step["source"][1] for step in out["witnesses"][0]["steps"]
                          if step["source"][0] == "wire"], [1, 2, 3])

    def test_placed_footprint_hole_and_other_net_drill(self):
        direct = wire(0, 0, 4, 0)
        # B is at (4, 0); R90 sends the local hole (0, 2) to (2, 0).
        hole = '<hole x="0" y="2" drill=".4"/>'
        self.assertEqual(self.routed(direct, end=(4, 0), rotation="R90", package=hole)["status"],
                         "unresolved")
        self.assertEqual(self.routed(direct, end=(4, 0), package=hole)["status"], "witnessed")
        other = '<signal name="other"><via x="2" y="0" drill=".4" extent="1-16"/></signal>'
        self.assertEqual(self.routed(direct, end=(4, 0), other_signals=other)["status"],
                         "unresolved")

    def test_inside_bore_shortcut_does_not_hide_valid_walk(self):
        short = (wire(0, 0, 2, 0) + wire(2, 0, 2.1, 0, 16) + wire(2.1, 0, 4, 0)
                 + '<via x="2" y="0" drill=".4" extent="1-16"/>'
                 + '<via x="2.1" y="0" drill=".4" extent="1-16"/>')
        self.assertEqual(self.routed(short, end=(4, 0))["status"], "unresolved")
        path = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 0)]
        detour = "".join(wire(*a, *b) for a, b in zip(path, path[1:]))
        out = self.routed(short + detour, end=(4, 0))
        self.assertEqual(out["status"], "witnessed")
        self.assertEqual([step["source"][1] for step in out["witnesses"][0]["steps"]
                          if step["source"][0] == "wire"], list(range(3, 9)))

    def test_drills_sever_plated_rim_without_crossing_trace_centerline(self):
        direct = (wire(0, 0, 1, 0) + wire(1, 0, 2, 0)
                  + '<via x="1" y="0" drill="1" diameter="1.2" extent="1-16"/>')
        splits = '<hole x="1" y=".55" drill=".30"/><hole x="1" y="-.55" drill=".30"/>'
        self.assertEqual(self.routed(direct, end=(2, 0))["status"], "witnessed")
        self.assertEqual(self.routed(direct, end=(2, 0), plain=splits)["status"], "unresolved")
        detour = wire(0, 0, 0, 1) + wire(0, 1, 2, 1) + wire(2, 1, 2, 0)
        out = self.routed(direct + detour, end=(2, 0), plain=splits)
        self.assertEqual(out["status"], "witnessed")
        self.assertEqual([s["source"][1] for s in out["witnesses"][0]["steps"]
                          if s["source"][0] == "wire"], [2, 3, 4])

    def test_shortcut_inside_a_bore_does_not_hide_a_valid_landing(self):
        # The old centerline proof needs the four-edge route. The SMD itself
        # overlaps the surviving annulus to the left of x=-.05 mm.
        detour = (
            wire(0, 0, -1, 0) + wire(-1, 0, -1, 1)
            + wire(-1, 1, 1, 1) + wire(1, 1, ".45", 0)
        )
        shortcut = wire(0, 0, ".45", 0)
        pad = dict(end=(".45", 0), pad='pad drill="1" diameter="1.5"')
        self.assertEqual(self.routed(detour, **pad)["status"], "witnessed")
        material = self.routed(shortcut, **pad)
        self.assertEqual(material["status"], "witnessed")
        self.assertEqual(material["material"]["model"], "finite_outer_layer_regions")
        for artwork in (detour + shortcut, shortcut + detour, shortcut + detour + shortcut):
            with self.subTest(artwork=artwork):
                result = self.routed(artwork, **pad)
                self.assertEqual(result["status"], "witnessed")
                steps = result["witnesses"][0]["steps"]
                self.assertEqual(sum(s["source"][0] == "wire" for s in steps), 4)
                nodes = [tuple(steps[0]["start"])] + [tuple(s["end"]) for s in steps]
                self.assertEqual(len(nodes), len(set(nodes)))

    def test_finite_width_bore_threshold_and_placed_cut(self):
        direct = wire(0, 0, 4, 0).replace('width=".1"', 'width=".254"')
        for diameter, expected in [(".127", "witnessed"), (".254", "unresolved"),
                                   (".28", "unresolved"), (".6", "unresolved")]:
            with self.subTest(diameter=diameter):
                out = self.routed(direct, end=(4, 0),
                                  plain=f'<hole x="2" y="0" drill="{diameter}"/>')
                self.assertEqual(out["status"], expected)
                if expected == "witnessed":
                    self.assertTrue(any(len(s["material"]["path_nm"]) > 2
                                        for s in out["witnesses"][0]["steps"]))
        out = self.routed(direct, end=(4, 0), rotation="R90",
                          package='<hole x="0" y="2" drill=".127"/>')
        self.assertEqual(out["status"], "witnessed")

    def test_rectangular_pad_edge_overlap_and_rotations(self):
        direct = wire(0, 0, 1.7, 0)
        for rotation, local in [("R0", "R0"), ("R90", "R90"), ("MR180", "R0")]:
            with self.subTest(rotation=rotation):
                pad = f'smd dx=".8" dy=".2" layer="{16 if rotation.startswith("M") else 1}" rot="{local}"'
                out = self.routed(direct, end=(2, 0), rotation=rotation, pad=pad)
                self.assertEqual(out["status"], "witnessed")
        for endpoint, expected in [("1.84", "unresolved"), ("1.85", "unresolved"),
                                   ("1.86", "witnessed")]:
            out = self.routed(wire(0, 0, endpoint, 0), end=(2, 0))
            self.assertEqual(out["status"], expected)

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
