"""Elemental conservation for complete mineral and water formula units."""
import unittest

from hamon_conservation_cad_verifier import parse_chemical_formula, verify_thermal_fluid_chemical_balance


class MineralFormulaBalance(unittest.TestCase):
    def test_crystal_water_scales_all_its_atoms(self):
        for formula, atoms in [
            ("CaSO4·2H2O", {"Ca": 1, "S": 1, "O": 6, "H": 4}),
            ("MgSO4·7H2O", {"Mg": 1, "S": 1, "O": 11, "H": 14}),
            ("KAl(SO4)2·12H2O", {"K": 1, "Al": 1, "S": 2, "O": 20, "H": 24}),
            ("CaSO4·0.5H2O", {"Ca": 1, "S": 1, "O": 4.5, "H": 1}),
            ("Ca0.5Mg0.5CO3", {"Ca": 0.5, "Mg": 0.5, "C": 1, "O": 3}),
            ("Ca0MgCO3", {"Ca": 0, "Mg": 1, "C": 1, "O": 3}),
        ]:
            with self.subTest(formula=formula):
                self.assertEqual(parse_chemical_formula(formula), atoms)

    def test_gypsum_water_balance_in_both_directions(self):
        for reverse in [False, True]:
            for water, expected in [(1, False), (2, True), (3, False)]:
                with self.subTest(reverse=reverse, water=water):
                    wet = [{"formula": "CaSO4·2H2O", "ndot_mol_s": 1}]
                    dry = [{"formula": "CaSO4", "ndot_mol_s": 1}, {"formula": "H2O", "ndot_mol_s": water}]
                    ins, outs = (wet, dry) if reverse else (dry, wet)
                    result = verify_thermal_fluid_chemical_balance({"inlets": ins, "outlets": outs})
                    self.assertEqual(result["species_balance_ok"], expected)
                    self.assertEqual(result["conservation_verified"], expected)

    def test_malformed_self_matches_fail_closed(self):
        for formula in ["Ca(OH", "Ca)SO4", "Ca()SO4", "Ca?SO4", "CaSO4·", "CaSO4··H2O", "CaSO4·2"]:
            with self.subTest(formula=formula):
                stream = [{"formula": formula, "ndot_mol_s": 1}]
                result = verify_thermal_fluid_chemical_balance({"inlets": stream, "outlets": stream})
                self.assertFalse(result["conservation_verified"])
                self.assertFalse(result["species_balance_ok"])
                self.assertTrue(result["species_balance_error"])
