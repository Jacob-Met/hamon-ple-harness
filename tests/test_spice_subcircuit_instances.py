import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def route(deck, **kwargs):
    return run_router_conservation_cad_verifier({"mode": "circuit", "spec": {"netlist": deck, **kwargs}})


class HierarchicalCircuits(unittest.TestCase):
    def test_shared_definition_has_two_independently_loaded_outputs(self):
        deck = ("Twin dividers\nVbias supply 0 DC 10\nXfirst supply outA div\nXsecond supply outB div\n"
                "RloadA outA 0 10k\nRloadB outB 0 2k\n.subckt div p o\n"
                "Rtop p o 1k\nRbottom o 0 1k\n.ends div\n.end")
        r = route(deck, proposed_node_voltages={"OUTa": 100/21, "OUTb": 4})
        self.assertTrue(r["conservation_verified"])
        self.assertEqual(set(r["node_voltages"]), {"0", "supply", "outA", "outB"})
        self.assertAlmostEqual(r["total_complex_power_delivered_VA"][0], 10*((10-100/21)/1000 + .006))
        self.assertFalse(route(deck, proposed_node_voltages={"outA": 0, "outB": 0})["conservation_verified"])

    def test_nested_calls_alias_formal_ports_and_isolate_private_nodes(self):
        # Each occurrence is two series stages, with the second input fed from
        # the first output. Solve the two loaded-divider resistors seen by V1.
        deck = (".subckt pair A O\nXfirst A local stage\nXlast LOCAL O stage\n.ends\n"
                ".subckt stage in out\nRhigh in out 1k\nRlow out 0 1k\n.ends\n"
                ".subckt unused o\nRghost o ghost 2\n.ends\nV1 in 0 12\n"
                "Xa in alpha PAIR\nXb IN beta pair\nR1 alpha 0 3k\nR2 beta 0 7k")
        r = route(deck)
        self.assertTrue(r["conservation_verified"])
        for name, load in (("alpha", 3000), ("beta", 7000)):
            rr = 1/(1/1000+1/load)
            stage = 1/(1/1000+1/(1000+rr))
            middle = 12*stage/(1000+stage)
            self.assertAlmostEqual(r["node_voltages"][name], middle*rr/(1000+rr))
        private = [k for k in r["node_voltages"] if k.endswith("/local")]
        self.assertEqual(len(private), 2)
        self.assertNotEqual(r["node_voltages"][private[0]], r["node_voltages"][private[1]])
        self.assertFalse(any("ghost" in k for k in r["node_voltages"]))

    def test_nested_current_controls_and_magnetic_references_stay_local(self):
        deck = ("Voltage divider example\n.subckt sense p o h\nVprobe p z 0\nRz z 0 4\n"
                "Fcopy 0 o vPROBE 2\nHread h 0 VProbe 3\n.ends\n"
                ".subckt coil a b\nKlocal lp ls 1\nLp a 0 1\nLs b 0 4\n.ends\n"
                ".subckt stage p o h sec\nXamp p o h sense\nXcoil h sec coil\n.ends\n"
                "Va pi 0 AC 8\nVb qi 0 AC -4\nXa pi a h1 s1 stage\nXb qi b h2 s2 stage\n"
                "RA a 0 7\nRB b 0 9\nR1 s1 0 100\nR2 s2 0 75\n"
                ".ac lin 1 60 60")
        r = route(deck)
        self.assertTrue(r["conservation_verified"])
        for name, expected in (("a",28),("b",-18),("h1",6),("h2",-3),("s1",12),("s2",-6)):
            self.assertAlmostEqual(complex(*r["node_voltages"][name]), expected)
        names = [b["name"].casefold() for b in r["branch_currents"]]
        self.assertEqual(len(names), len(set(names)))

    def test_instance_paths_cannot_alias_literal_names_or_each_other(self):
        deck = ("Xtop leaf leaf\nV1 leaf 0 4\nV2 @Xtop/m 0 9\n"
                ".subckt leaf a\nR1 a m 1\nR2 m 0 1\n.ends")
        r = route(deck)
        self.assertTrue(r["conservation_verified"])
        self.assertEqual(r["node_voltages"]["@Xtop/m"], 9)
        self.assertIn(2, r["node_voltages"].values())
        deck = ("Path names\nV1 a 0 3\nX1/X2 a unit\nX1 a parent\n"
                ".subckt parent p\nX2 p unit\n.ends\n.subckt unit p\nR1 p m 1\nR2 m 0 2\n.ends")
        r=route(deck)
        self.assertTrue(r["conservation_verified"])
        self.assertEqual(len([v for v in r["node_voltages"].values() if math.isclose(v,2)]), 2)

    def test_incomplete_or_unsupported_hierarchy_refuses(self):
        definition = ".subckt simple in out\nR1 in out 1k\nR2 out 0 1k\n.ends simple\n"
        for bad in ("Xbad p o missing", "Xbad p simple", "Xbad p o simple gain=2",
                    "Xbad p o simple\nxBAD p o simple", ".global extra", "D1 p o diode"):
            deck = "Test deck\nV1 p 0 1\nR0 o 0 1k\n" + bad + "\n" + definition
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                route(deck)
        for malformed in (".subckt s a\nXa a s\n.ends", ".subckt s a\n.ends other",
                          ".subckt s a\nR1 a 0 1k", ".subckt s a A\n.ends",
                          ".subckt s a\nD1 a 0 d\n.ends", ".subckt s a params: x=1\n.ends",
                          ".subckt s a\nR1 a 0 1 m=2\n.ends", ".subckt s a\nR1 a 0 1k*2\n.ends", ".subckt s a\n.subckt t b\n.ends t\n.ends s",
                          ".subckt s a\nXloop a other\n.ends\n.subckt other t\nXback t s\n.ends"):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                route("Test deck\nV1 p 0 1\nX1 p s\n"+malformed)
