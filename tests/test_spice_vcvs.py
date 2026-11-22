import math
import unittest

from hamon_conservation_cad_verifier import verify_circuit_conservation


class VoltageControlledVoltageSourceTests(unittest.TestCase):
    def test_floating_differential_constraint_current_and_power(self):
        for gain in (-2, 0, 3):
            with self.subTest(gain=gain):
                deck = f"""Differential voltage source
Vcp ca 0 DC 2
Vcm cb 0 DC -1
Eamp b a cb ca {gain}
Ra a 0 1k
Rb b 0 2k
Rload a b 600
.end"""
                result = verify_circuit_conservation({"netlist": deck})
                voltage = 3 * gain
                self.assertTrue(result["conservation_verified"])
                self.assertAlmostEqual(result["node_voltages"]["a"], voltage / 3)
                self.assertAlmostEqual(result["node_voltages"]["b"], -2 * voltage / 3)
                currents = {b["name"]: b["current_A"] for b in result["branch_currents"]}
                self.assertAlmostEqual(currents["Eamp"], voltage / 500)
                self.assertEqual((currents["Vcp"], currents["Vcm"]), (0, 0))
                power, reactive = result["total_complex_power_delivered_VA"]
                self.assertAlmostEqual(power, voltage**2 / 500)
                self.assertEqual(reactive, 0)

    def test_controlled_phasor_drives_capacitor(self):
        result = verify_circuit_conservation({"freq_hz": 1000, "branches": [
            {"name": "V1", "type": "V", "n_pos": "in", "n_neg": "0", "value": 2, "phase_deg": 90},
            {"name": "E1", "type": "E", "n_pos": "out", "n_neg": "0", "nc_pos": "in", "nc_neg": "0", "value": -3},
            {"name": "C1", "type": "C", "n_pos": "out", "n_neg": "0", "value": 1e-6},
        ]})
        self.assertTrue(result["conservation_verified"])
        vr, vi = result["node_voltages"]["out"]
        ir, ii = result["branch_currents"][1]["current_A"]
        self.assertAlmostEqual(vr, 0)
        self.assertAlmostEqual(vi, -6)
        self.assertAlmostEqual(ir, -0.012 * math.pi)
        self.assertAlmostEqual(ii, 0)

    def test_first_linear_card_and_ordinary_title(self):
        for title in ("", "Example of a voltage follower\n"):
            with self.subTest(title=title):
                deck = title + "Eamp out 0 in 0 2\nV1 in 0 DC 1\nR1 out 0 1k\n.end"
                result = verify_circuit_conservation({"netlist": deck})
                self.assertTrue(result["conservation_verified"])
                self.assertAlmostEqual(result["node_voltages"]["out"], 2)

    def test_refuses_unsupported_controlled_source(self):
        for card in (
            "Eamp out 0 in 0", "Eamp out 0 in 0 2 trailing", "Eamp out 0 POLY(1) in 0 1 2",
            "Eamp out 0 VALUE={2*V(in)}", "Eamp out 0 in 0 1.2.3", "Eamp out 0 in 0 1e999",
            "Eamp out 0 VALUE x 2", "Eamp out 0 POLY(1) x 2", "Eamp out 0 TABLE x 2",
        ):
            with self.subTest(card=card), self.assertRaises(ValueError):
                verify_circuit_conservation({"netlist": f"Amplifier\nV1 in 0 DC 1\n{card}\nR1 out 0 1k\n.end"})
