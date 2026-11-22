import unittest

from hamon_pareto import dispatch_ple_tool


class StreamEvidenceTests(unittest.TestCase):
    def check_balance(self, inlets, outlets):
        return dispatch_ple_tool(
            "hamon_conservation_cad_verifier", {"inlets": inlets, "outlets": outlets}
        )

    def test_missing_flow_differs_from_explicit_zero(self):
        main = {"mdot_kg_s": 3}
        for bypass, expected in (({}, False), ({"mdot_kg_s": 0}, True)):
            with self.subTest(bypass=bypass):
                result = self.check_balance([main, bypass], [main])
                self.assertIs(result["mass_balance_ok"], expected)
                self.assertIs(result["conservation_verified"], expected)

    def test_species_balance_has_its_own_evidence(self):
        steam = {"formula": "H2O", "ndot_mol_s": 1}
        result = self.check_balance([steam], [steam])
        self.assertTrue(result["species_balance_ok"])
        self.assertNotIn("mass_balance_ok", result)
        self.assertTrue(result["conservation_verified"])

        steam = dict(steam, mdot_kg_s=1)
        partial = self.check_balance([steam, {"mdot_kg_s": 0, "formula": "O2"}], [steam])
        self.assertTrue(partial["mass_balance_ok"])
        self.assertFalse(partial["species_balance_ok"])
        self.assertFalse(partial["conservation_verified"])

    def test_overflowed_element_flow_cannot_verify(self):
        result = self.check_balance([{"formula": "H2", "ndot_mol_s": 1e308}],
                                    [{"formula": "H2", "ndot_mol_s": 1}])
        self.assertFalse(result["species_balance_ok"])
        self.assertFalse(result["conservation_verified"])
