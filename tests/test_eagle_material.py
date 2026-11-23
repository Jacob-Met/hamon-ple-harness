"""Physical controls for finite-width nominal copper and its explicit path proofs."""

import math
import unittest

from hamon_eagle_material import material_continuity


def pad(name, x, y, size=(40_000, 40_000), layer=1):
    return {"source": ("terminal", name, "p"), "terminal": ("pad", name, "p"),
            "shape": "rectangle", "center": (x, y), "size": size, "layers": (layer,)}


def wire(number, a, b, width, layer=1, cap="round"):
    return {"source": ("wire", number), "shape": "segment", "a": a, "b": b,
            "width": width, "layers": (layer,), "cap": cap}


def assess(primitives, bores=()):
    return material_continuity([p["terminal"] for p in primitives if "terminal" in p],
                               primitives, bores)


def segment_distance(a, b, center):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx + dy*dy
    u = max(0, min(1, ((center[0]-a[0])*dx + (center[1]-a[1])*dy)/length2)) if length2 else 0
    return math.hypot(a[0] + u*dx - center[0], a[1] + u*dy - center[1])


class FiniteCopperTests(unittest.TestCase):
    def test_small_centered_bore_has_a_real_nonstraight_route(self):
        boards = [pad("A", 0, 0), wire(0, (0, 0), (1_000_000, 0), 254_000),
                  pad("B", 1_000_000, 0)]
        result = assess(boards, [(500_000, 0, 127_000, None)])
        self.assertEqual(result["unreachable"], [])
        steps = result["witnesses"][0]["steps"]
        self.assertEqual([s["source"][0] for s in steps], ["terminal", "wire", "terminal"])
        self.assertTrue(all(a["end"] == b["start"] for a, b in zip(steps, steps[1:])))
        path = steps[1]["material"]["path_nm"]
        self.assertGreater(len(path), 2)
        for a, b in zip(path, path[1:]):
            self.assertGreaterEqual(segment_distance(a, b, (500_000, 0)), 63_501)
            self.assertLessEqual(max(abs(a[1]), abs(b[1])), 126_999)
        for step in steps[:-1]:
            self.assertGreater(step["material"]["exit_overlap"]["radius_nm"], 0)
        for diameter in [254_000, 280_000, 600_000]:
            with self.subTest(diameter=diameter):
                self.assertFalse(assess(boards, [(500_000, 0, diameter, None)])["witnesses"])

    def test_round_cap_area_and_exact_tangency(self):
        for gap, expected in [(234_000,True), (254_000,False), (274_000,False)]:
            for origin in [0, 9_000_000_000_000]:
                with self.subTest(gap=gap,origin=origin):
                    s = lambda x: (origin + x, origin)
                    primitives = [pad("A", *s(0)), wire(0, s(0), s(500_000), 254_000),
                                  wire(1, s(500_000 + gap), s(1_400_000), 254_000),
                                  pad("B", *s(1_400_000))]
                    self.assertEqual(bool(assess(primitives)["witnesses"]), expected)
        # A flat end cannot claim the conducting cap of a round-ended trace.
        prim = [pad("A",0,0), wire(0,(0,0),(500_000,0),254_000,cap="flat"),
                wire(1,(734_000,0),(1_400_000,0),254_000),pad("B",1_400_000,0)]
        self.assertFalse(assess(prim)["witnesses"])

    def test_cut_pad_is_never_a_virtual_bridge(self):
        a = pad("A", 0, 0, (1_000_000, 200_000))
        b, c = pad("B",-450_000,0), pad("C",450_000,0)
        hole = [(0,0,300_000,None)]
        self.assertTrue(assess([a,b],hole)["witnesses"])
        self.assertTrue(assess([a,c],hole)["witnesses"])
        self.assertFalse(assess([a,b,c],hole)["witnesses"])

    def test_offcenter_plated_contact_requires_a_real_collar(self):
        via = {"source": ("via",0), "layers": (1,16), "shape": "disk",
               "center": (0,0), "diameter":1_000_000, "plated":True}
        bore = [(0,0,400_000,("via",0))]
        for intrusion, expected in [(25_000,True), (0,False), (-25_000,False)]:
            edge = 500_000 + 88_900 - intrusion
            parts = [pad("A",0,1_000_000),wire(0,(0,1_000_000),(0,edge),177_800),
                     via,wire(1,(0,-400_000),(0,-1_000_000),177_800,16),
                     pad("B",0,-1_000_000,layer=16)]
            result = assess(parts,bore)
            self.assertEqual(bool(result["witnesses"]),expected)
            if expected:
                steps = result["witnesses"][0]["steps"]
                crossings = [s for s in steps if s["material"]["kind"]=="plated_barrel"]
                self.assertEqual(len(crossings),1)
                self.assertEqual(crossings[0]["material"]["barrel_wall_point_nm"],[200_000,0])
                self.assertEqual(crossings[0]["material"]["path"][1:3],
                                 [["barrel_wall",1,200_000,0],["barrel_wall",16,200_000,0]])
                self.assertTrue(all(s["end"]==t["start"] for s,t in zip(steps,steps[1:])))
                self.assertFalse(assess(parts,[(0,0,400_000,None)])["witnesses"])

    def test_plated_terminal_path_begins_at_its_real_collar(self):
        term = {"source": ("terminal","A","p"), "terminal":("pad","A","p"),
                "layers":(1,16),"shape":"disk","center":(0,0),"diameter":1_000_000,"plated":True}
        landing = [term,wire(0,(-450_000,0),(-1_000_000,0),100_000),
                   pad("B",-1_000_000,0)]
        out = assess(landing, [(0,0,400_000,term["source"])])
        first = out["witnesses"][0]["steps"][0]["material"]
        self.assertEqual(first["terminal_collar"]["point_nm"],first["path_nm"][0])
        for a,b in zip(first["path_nm"],first["path_nm"][1:]):
            self.assertGreaterEqual(segment_distance(a,b,(0,0)),200_001)
        # Disabling the barrel independently also disables access to this PTH
        # terminal, while its surviving copper may still be used in-plane.
        self.assertFalse(assess(landing,[(0,0,400_000,None)])["witnesses"])

    def test_unsupported_or_opposite_face_cannot_be_a_bridge(self):
        reverse = [pad("A",0,0),pad("B",0,0,layer=16)]
        self.assertFalse(assess(reverse)["witnesses"])
        unsupported = {"source": ("polygon",0), "layers":(1,16), "shape":"polygon"}
        self.assertFalse(assess(reverse+[unsupported])["witnesses"])


if __name__ == "__main__":
    unittest.main()
