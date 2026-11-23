"""Electrical queries through embedded EAGLE component, device and footprint identities."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hamon_eagle_netlist import compare_board, schematic_netlist


SCHEMATIC = """<eagle><drawing><schematic><libraries><library name="L">
<packages><package name="D"><smd name="1"/><pad name="2"/><pad name="3"/><pad name="4"/></package></packages>
<symbols><symbol name="logic"><pin name="IN"/><pin name="VCC" direction="pwr"/></symbol>
<symbol name="power"><pin name="GND@1" direction="pwr"/></symbol>
<symbol name="rail"><pin name="RAIL" direction="sup"/></symbol></symbols>
<devicesets><deviceset name="dual"><gates><gate name="A" symbol="logic"/><gate name="P" symbol="power"/></gates>
<devices><device name="X" package="D"><connects><connect gate="A" pin="IN" pad="1 2" route="any"/>
<connect gate="A" pin="VCC" pad="3"/><connect gate="P" pin="GND@1" pad="4"/></connects></device>
<device name="other" package="D"><connects><connect gate="A" pin="IN" pad="3"/></connects></device></devices></deviceset>
<deviceset name="rail"><gates><gate name="1" symbol="rail"/></gates><devices><device name=""/></devices></deviceset>
</devicesets></library></libraries><parts><part name="U1" library="L" deviceset="dual" device="X"/>
<part name="P1" library="L" deviceset="rail" device=""/></parts><sheets>
<sheet><instances><instance part="U1" gate="A"/></instances><nets><net name="SIG"><segment>
<pinref part="U1" gate="A" pin="IN"/></segment></net><net name="ALT"><segment>
<pinref part="U1" gate="A" pin="VCC"/></segment></net></nets></sheet>
<sheet><instances><instance part="P1" gate="1"/></instances><nets><net name="SIG"><segment>
<pinref part="P1" gate="1" pin="RAIL"/></segment></net></nets></sheet></sheets></schematic></drawing></eagle>"""

BOARD = """<eagle><drawing><board><libraries><library name="L"><packages><package name="D">
<smd name="1"/><pad name="2"/><pad name="3"/><pad name="4"/></package></packages></library></libraries>
<elements><element name="U1" library="L" package="D"/></elements><signals>
<signal name="SIG"><contactref element="U1" pad="1"/><contactref element="U1" pad="2"/></signal>
<signal name="ALT"><contactref element="U1" pad="3"/></signal>
<signal name="GND"><contactref element="U1" pad="4"/></signal></signals></board></drawing></eagle>"""


class EagleNetlistTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.sch, self.brd = root/"test.sch", root/"test.brd"
        self.sch.write_text(SCHEMATIC)
        self.brd.write_text(BOARD)

    def test_gate_device_pad_mapping_and_uninvoked_power(self):
        r = schematic_netlist(self.sch)
        self.assertEqual(r["sheet_count"], 2)
        self.assertEqual({p["pad"]: p["net"] for p in r["physical_contacts"]},
                         {"1":"SIG", "2":"SIG", "3":"ALT", "4":"GND"})
        logical = {p["pin"]:p for p in r["logical_pins"]}
        self.assertEqual(logical["GND@1"]["source"], "automatic_power")
        self.assertIsNone(logical["GND@1"]["sheet"])
        self.assertEqual(logical["VCC"]["source"], "pinref")
        self.assertEqual(logical["RAIL"]["pads"], [])
        c = compare_board(r, self.brd)
        self.assertTrue(c["agreement"])
        self.assertEqual(c["exact_contacts"], 4)
        self.assertEqual(c["exact_nets"], ["ALT", "GND", "SIG"])

    def test_mapped_auto_power_has_actual_sheet_and_explicit_wire_takes_priority(self):
        self.sch.write_text(SCHEMATIC.replace('<pinref part="U1" gate="A" pin="VCC"/>', ''))
        rows = schematic_netlist(self.sch)["logical_pins"]
        power = next(p for p in rows if p["pin"] == "VCC")
        self.assertEqual((power["net"],power["sheet"],power["source"]), ("VCC",1,"automatic_power"))

    def test_wrong_net_missing_pad_and_invalid_footprint_are_distinct(self):
        r = schematic_netlist(self.sch)
        self.brd.write_text(BOARD.replace('name="ALT"><contactref', 'name="OTHER"><contactref')
                            .replace('<contactref element="U1" pad="2"/>',''))
        c = compare_board(r, self.brd)
        self.assertFalse(c["agreement"])
        self.assertEqual(c["exact_nets"], ["GND"])
        self.assertEqual([(p["pad"],p["schematic_net"],p["board_net"]) for p in c["mismatches"]],
                         [("2","SIG",None), ("3","ALT","OTHER")])
        self.brd.write_text(BOARD.replace('element="U1" pad="4"', 'element="U1" pad="44"'))
        with self.assertRaises(ValueError):
            compare_board(r, self.brd)

    def test_conflicting_pins_invalid_devices_and_hierarchy_reject(self):
        for src in (SCHEMATIC.replace('pin="IN" pad="1 2"', 'pin="IN" pad="1 22"'),
                    SCHEMATIC.replace('pin="IN" pad="1 2"', 'pin="IN" pad="3"'),
                    SCHEMATIC.replace('<connect gate="P" pin="GND@1" pad="4"/>', ''),
                    SCHEMATIC.replace('gate="A" pin="IN"/></segment>', 'gate="Z" pin="IN"/></segment>'),
                    SCHEMATIC.replace('<parts>', '<modules><module name="Nested"/></modules><parts>')):
            with self.subTest(src=src), self.assertRaises(ValueError):
                self.sch.write_text(src)
                schematic_netlist(self.sch)

    def test_bad_instance_without_pinrefs_cannot_supply_automatic_power(self):
        extra_part = '<part name="U2" library="L" deviceset="dual" device="X"/>'
        bad_instance = '<instance part="U2" gate="NO_SUCH_GATE"/>'
        src = SCHEMATIC.replace("</parts>", extra_part+"</parts>").replace("<instances>", "<instances>"+bad_instance, 1)
        self.sch.write_text(src)
        with self.assertRaisesRegex(ValueError, "Unresolved EAGLE gate instance"):
            schematic_netlist(self.sch)

    def test_real_cli_returns_contact_evidence(self):
        root = Path(__file__).resolve().parents[1]
        p = subprocess.run([sys.executable,str(root/"engineering_toolkit.py"),"eagle-netlist",
                            str(self.sch),"--board",str(self.brd)],check=True,capture_output=True,text=True)
        out = json.loads(p.stdout)
        self.assertTrue(out["board_comparison"]["agreement"])
        self.assertEqual(len(out["physical_contacts"]),4)
