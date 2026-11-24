"""Geometry controls for complete outlined labels on quarter-pitch drafting sites."""
import unittest
from xml.etree.ElementTree import Element
import hamon_eagle_outlined as outlined
from hamon_eagle_schematic import _outlined_supplies

PITCH = 10.0

def strokes(name, origin, baseline=-0.3):
    # Font shapes are existing fixtures; these tests isolate binding, not OCR.
    result = []
    cursor = origin
    for char in name:
        if char == "_":
            edges = [((0, -1), (4, -1))]
        else:
            edges = outlined._edges(outlined._GLYPHS[char])
        for a, b in edges:
            result.append(tuple((cursor + 1.05*x, baseline - 1.05*y) for x,y in (a,b)))
        width = max(x for edge in edges for x,y in edge)
        cursor += 1.05*(width+2)
    return result

def recognized(wires, ink):
    return [(i,n,r) for i,n,_,r in outlined.wire_words(wires,ink,PITCH)]

class OutlinedOriginTests(unittest.TestCase):
    def test_complete_long_label_can_extend_past_wire_end(self):
        words = recognized([[(0,0),(20,0)]],strokes("JTAG_TMS",10.3))
        self.assertIn((0,"JTAG_TMS",True),words)

    def test_quarter_origins_preserve_absolute_residual(self):
        for quarter in (0,.25,.5,.75):
            with self.subTest(quarter=quarter):
                ink=strokes("JTAG_TDI",10.3+10*quarter)
                self.assertIn((0,"JTAG_TDI",True),recognized([[(0,0),(30,0)]],ink))
                displaced=[tuple((x+.4,y) for x,y in line) for line in ink]
                self.assertNotIn((0,"JTAG_TDI",True),recognized([[(0,0),(30,0)]],displaced))

    def test_eighth_grid_incidental_text_does_not_gain_a_label(self):
        self.assertNotIn((0,"UOTGID",True),recognized([[(0,0),(20,0)]],strokes("UOTGID",4.05)))

    def test_vertical_projection_and_paint_order(self):
        wire=[(0,0),(20,0)]
        ink=strokes("JTAG_TCK",7.8)
        turn=lambda p:(p[1],-p[0])
        rotated=[tuple(map(turn,line[::-1])) for line in ink[::-1]]
        self.assertIn((0,"JTAG_TCK",True),recognized([list(map(turn,wire))],rotated))

    def test_erasing_underscore_does_not_create_a_prefix(self):
        ink=strokes("JTAG_TMS",7.8)
        damaged=[line for line in ink if not all(abs(p[1]-.75)<1e-8 for p in line)]
        self.assertEqual(recognized([[(0,0),(30,0)]],damaged),[])

    def test_contained_word_needs_no_drafting_grid_origin(self):
        self.assertIn((0,"D+",True),recognized([[(0,0),(30,0)]],strokes("D+",5.73)))

    def test_pen_styles_do_not_mix_into_one_broken_word(self):
        valid=strokes("JTAG_TMS",7.8)
        noise=[((9.1,-2.2),(11.8,-3.6))]
        self.assertIn((0,"JTAG_TMS",True),recognized([[(0,0),(20,0)]],{.65:valid,.36:noise}))
        self.assertNotIn((0,"JTAG_TMS",True),recognized([[(0,0),(20,0)]],valid+noise))

    def test_excessive_distinct_pen_styles_withhold_text(self):
        self.assertEqual(recognized([[(0,0),(20,0)]],{i+1:strokes("JTAG_TMS",7.8) for i in range(17)}),[])

    def test_one_occurrence_over_two_separate_wires_stays_unresolved(self):
        ink=strokes("TX",7.8)
        paths={"rgb(30%, 30%, 30%)":[]}
        for (x0,y0),(x1,y1) in ink:
            e=Element("path",{"d":f"M {x0} {y0} L {x1} {y1}","stroke-width":".65"})
            paths["rgb(30%, 30%, 30%)"].append((e,(1,0,0,1,0,0)))
        wires=[[(0,0),(12,0)],[(16,0),(28.6,0)]]
        groups,_,anchors=_outlined_supplies(paths,"rgb(0%, 50%, 0%)",PITCH,wires,[0,1])
        self.assertNotEqual(groups[0],groups[1])
        self.assertNotIn("TX",anchors)

if __name__=="__main__":
    unittest.main()
