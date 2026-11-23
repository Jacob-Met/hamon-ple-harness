import base64
import unittest

from hamon_conservation_cad_verifier import verify_circuit_conservation


def solve(cards, **spec):
    return verify_circuit_conservation({"netlist": "* batch node fixture\n" + cards + "\n.end", **spec})


class BatchNodeIdentityTests(unittest.TestCase):
    def test_supply_power_and_first_spelling(self):
        for cards, node in (("V1 VDD 0 10\nR1 vdd 0 1k", "VDD"),
                            ("R1 vdd 0 1k\nV1 VDD 0 10", "vdd")):
            with self.subTest(node=node):
                result = solve(cards)
                self.assertTrue(result["conservation_verified"])
                self.assertEqual(result["node_voltages"], {"0": 0., node: 10.})
                self.assertAlmostEqual(result["total_complex_power_delivered_VA"][0], .1)
                currents = {b["name"]: b["current_A"] for b in result["branch_currents"]}
                self.assertAlmostEqual(currents["V1"], -.01)
                self.assertAlmostEqual(currents["R1"], .01)

    def test_proposals_share_a_physical_reference(self):
        cards = "V1 Bus GnD 3\nR1 bUs gNd 600"
        good = solve(cards, proposed_node_voltages={"BUS": 3, "bus": 3, "gND": 0})
        bad = solve(cards, proposed_node_voltages={"BUS": 3, "bus": 0})
        self.assertEqual(good["node_voltages"], {"0": 0., "Bus": 3.})
        self.assertAlmostEqual(good["branch_currents"][0]["current_A"], -.005)
        self.assertTrue(good["candidate_voltages_valid"])
        self.assertFalse(bad["candidate_voltages_valid"])
        self.assertFalse(bad["conservation_verified"])

    def test_differential_control_pins(self):
        result = solve("V1 SeNsE 0 2\nV2 Ref 0 1\nE1 Out 0 sense rEF -3\nR1 oUt 0 2k")
        self.assertTrue(result["conservation_verified"])
        self.assertEqual(result["node_voltages"]["Out"], -3.)
        self.assertAlmostEqual(result["branch_currents"][2]["current_A"], .0015)
        self.assertAlmostEqual(result["total_complex_power_delivered_VA"][0], .0045)

    def test_ac_filter_voltage_and_complex_power(self):
        result = solve("V1 SOURCE GnD AC 1 0\nR1 source Tap 1k\n"
                       "C1 tAP 0 159.15494309189535n\n.ac lin 1 1000 1000")
        self.assertTrue(result["conservation_verified"])
        for actual, expected in zip(result["node_voltages"]["Tap"], [.5, -.5]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(result["total_complex_power_delivered_VA"], [.0005, -.0005]):
            self.assertAlmostEqual(actual, expected)

    def test_non_ascii_bytes_follow_batch_spelling(self):
        separate = solve("V1 Straße 0 5\nR1 STRASSE 0 1k")
        mapped = solve("V1 À 0 5\nR1 à 0 1k", proposed_node_voltages={"__": 5})
        self.assertEqual(separate["total_complex_power_delivered_VA"][0], 0.)
        self.assertEqual(set(separate["node_voltages"]), {"0", "Straße", "STRASSE"})
        self.assertEqual(mapped["node_voltages"], {"0": 0., "À": 5.})
        self.assertAlmostEqual(mapped["branch_currents"][0]["current_A"], -.005)
        self.assertTrue(mapped["conservation_verified"])

    def test_double_zero_is_its_own_node(self):
        result = solve("V1 00 gNd 2\nV2 Bus 00 3\nR1 bUs 0 1k")
        self.assertEqual(result["node_voltages"], {"0": 0., "00": 2., "Bus": 5.})
        self.assertAlmostEqual(result["total_complex_power_delivered_VA"][0], .025)

    def test_file_deck_and_structured_branch_contracts(self):
        raw = b"* netlist\nV1 Rail 0 10\nR1 rail 0 1k\n.end"
        from_file = verify_circuit_conservation({"file_b64": base64.b64encode(raw).decode()})
        self.assertAlmostEqual(from_file["total_complex_power_delivered_VA"][0], .1)
        structured = verify_circuit_conservation({"branches": [
            {"type": "V", "name": "V1", "n_pos": "Rail", "n_neg": "0", "value": 10.},
            {"type": "R", "name": "R1", "n_pos": "rail", "n_neg": "0", "value": 1000.}],
            "proposed_node_voltages": {"Rail": 10, "rail": 0}})
        self.assertTrue(structured["conservation_verified"])
        self.assertEqual(structured["node_voltages"], {"0": 0., "Rail": 10., "rail": 0.})
        self.assertEqual(structured["total_complex_power_delivered_VA"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
