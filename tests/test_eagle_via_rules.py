"""Rule-derived outer annuli for round full-span EAGLE vias."""
import unittest
from xml.etree import ElementTree as ET

from hamon_eagle_copper import rule_via_annulus


def board(ratio, lower, upper):
    design = ET.Element("board")
    rules = ET.SubElement(design, "designrules")
    for name, value in zip(("rvViaOuter", "rlMinViaOuter", "rlMaxViaOuter"), (ratio, lower, upper)):
        ET.SubElement(rules, "param", name=name, value=value)
    return design


class ViaRuleTests(unittest.TestCase):
    def test_clamps_and_inward_subnanometre_bound(self):
        for ratio, lower, upper, expected in [
            ("0.25", "8mil", "20mil", 812_800),
            ("0.75", "0.2032mm", "20mil", 1_016_000),
            ("2", "8mil", "0.508mm", 1_422_400),
            ("0", "0.00000123mm", "20mil", 406_402),
        ]:
            with self.subTest(ratio=ratio):
                size, evidence = rule_via_annulus(board(ratio, lower, upper), 406_400)
                self.assertEqual(size, expected)
                self.assertEqual(evidence["drill_nm"], 406_400)
                self.assertEqual(evidence["outer_diameter_lower_bound_nm"], size)

    def test_incomplete_or_ambiguous_rules_stay_unresolved(self):
        for args in [("25%", "8mil", "20mil"), ("0.25", "8", "20mil"),
                     ("0.25", "-8mil", "20mil"), ("0.25", "8mil", "2mil"),
                     ("NaN", "8mil", "20mil"), ("0", "0mm", "20mil")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                rule_via_annulus(board(*args), 406_400)
        for repeated in (False, True):
            root = board("0.25", "8mil", "20mil")
            rules = root.find("designrules")
            if repeated:
                ET.SubElement(rules, "param", name="rvViaOuter", value="0.5")
            else:
                rules.remove(rules.find("param"))
            with self.assertRaises(ValueError):
                rule_via_annulus(root, 406_400)
        root = board("0.25", "8mil", "20mil")
        ET.SubElement(root, "designrules")
        with self.assertRaises(ValueError):
            rule_via_annulus(root, 406_400)
