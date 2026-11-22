import math
import unittest

from hamon_conservation_cad_verifier import verify_circuit_conservation


class IdealInductorOperatingPointTests(unittest.TestCase):
    def test_grounded_short_has_exact_terminal_voltage_and_signed_current(self):
        deck = "V1 rail 0 3\nR1 rail p 1\nL1 q p 0.001\nR2 q 0 2"
        result = verify_circuit_conservation({"netlist": deck})
        self.assertTrue(result["conservation_verified"])
        self.assertAlmostEqual(result["node_voltages"]["p"], 2.0)
        self.assertAlmostEqual(result["node_voltages"]["q"], 2.0)
        currents = {x["name"]: x["current_A"] for x in result["branch_currents"]}
        self.assertAlmostEqual(currents["L1"], -1.0)
        self.assertAlmostEqual(currents["V1"], -1.0)
        self.assertFalse(verify_circuit_conservation({
            "netlist": deck, "proposed_node_voltages": {"q": 1.0},
        })["conservation_verified"])

    def test_undefined_full_operating_points_are_rejected(self):
        decks = [
            "V1 pin 0 1\nL1 pin 0 1m",                   # Contradictory ideal voltages.
            "I1 0 pin 1\nL1 pin 0 1m\nL2 pin 0 2m",       # Individual currents are free.
            "I1 a b 1\nR1 b a 2\nL1 b c 1m",              # Only voltage differences are fixed.
        ]
        for deck in decks:
            with self.subTest(deck=deck), self.assertRaisesRegex(ValueError, "Singular MNA"):
                verify_circuit_conservation({"netlist": deck})

    def test_dc_inductor_still_requires_a_numeric_component_value(self):
        source = {"name": "I1", "type": "I", "n_pos": "0", "n_neg": "pin", "value": 1.0}
        coil = {"name": "L1", "type": "L", "n_pos": "pin", "n_neg": "0"}
        for fields in ({}, {"value": "invalid"}, {"value": float("nan")}):
            with self.subTest(fields=fields), self.assertRaises((KeyError, ValueError)):
                verify_circuit_conservation({"branches": [source, {**coil, **fields}]})

    def test_ac_inductor_retains_its_reactive_drop(self):
        freq = 50.0
        inductance = 0.1
        deck = f"V1 rail 0 AC 2\nR1 rail n 3\nL1 n 0 {inductance}"
        result = verify_circuit_conservation({"netlist": deck, "freq_hz": freq})
        self.assertTrue(result["conservation_verified"])
        actual = complex(*result["branch_currents"][2]["current_A"])
        self.assertAlmostEqual(abs(actual - 2 / (3 + 2j * math.pi * freq * inductance)), 0)
