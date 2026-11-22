"""Branch identity and current direction at the two public circuit solver interfaces."""

import cmath
import math
import unittest

from hamon_conservation_cad_verifier import verify_circuit_conservation
from hamon_dual_verifier import solve_domain_problem


class BranchCurrentTests(unittest.TestCase):
    def test_parallel_rlc_printed_currents_and_source_phase(self):
        # Kuphaldt, Lessons in Electric Circuits II, ch. 5, parallel RLC.
        # The ideal current meters are collapsed. The inductor retains the
        # 1e-12 ohm series resistance used by the published SPICE example.
        ends = {"n1": "s", "n2": "gnd"}
        result = solve_domain_problem("ac_phasor_rlc_network", {
            "freq_Hz": 60, "voltage_sources_to_ground": {"s": 120},
            "resistors": [{**ends, "R_ohm": 250}],
            "inductors": [{**ends, "L_H": 0.650, "R_series_ohm": 1e-12}],
            "capacitors": [{**ends, "C_F": 1.5e-6}],
        })
        rows = result["branch_currents"]
        self.assertEqual([(r["collection"], r["index"]) for r in rows],
                         [("resistors", 0), ("inductors", 0), ("capacitors", 0)])
        self.assertEqual(result["I_branch_s__GND___A"], rows[0]["current_A"])
        measured = [complex(*r["current_A"]) for r in rows]
        supplied = complex(*result["prescribed_node_supply_A"]["s"])
        self.assertAlmostEqual(abs(supplied - sum(measured)), 0)
        expected = [(0.4800, 0), (0.4897, -90), (0.06786, 90), (0.6390, -41.31)]
        for value, (magnitude, angle) in zip([*measured, supplied], expected):
            self.assertAlmostEqual(abs(value), magnitude, places=5 if angle == 90 else 4)
            self.assertAlmostEqual(math.degrees(cmath.phase(value)), angle, places=2)

    def test_legacy_label_collision_preserves_both_oriented_elements(self):
        result = solve_domain_problem("dc_nodal_kcl", {
            "voltage_sources_to_ground": {"1": 10, "23": 0, "12": 8, "3": 2},
            "resistors": [{"n1": "1", "n2": "23", "R_ohm": 5},
                          {"n1": "12", "n2": "3", "R_ohm": 2}],
        })
        self.assertEqual(result["I_branch_123_A"], 2)
        self.assertEqual([(b["n1"], b["n2"], b["current_A"]) for b in result["branch_currents"]],
                         [("1", "23", 2), ("12", "3", 3)])

    def test_fixed_supply_excludes_independent_injection(self):
        result = solve_domain_problem("dc_nodal_kcl", {
            "voltage_sources_to_ground": {"hi": 12},
            "current_injections": [{"node": "hi", "I_A": [0.25, 1.5]}],
            "resistors": [{"n1": "low", "n2": "hi", "R_ohm": 3},
                          {"n1": "low", "n2": "0", "R_ohm": 3}],
        })
        self.assertEqual(result["branch_currents"][0]["current_A"], -2)
        self.assertEqual(result["prescribed_node_supply_A"]["hi"], [1.75, -1.5])

    def test_mna_reports_every_input_in_passive_sign_convention(self):
        branches = [{"name": "v", "type": "V", "n_pos": "s", "n_neg": "0", "value": 12},
                    {"name": "r1", "type": "R", "n_pos": "s", "n_neg": "0", "value": 3},
                    {"name": "r2", "type": "R", "n_pos": "0", "n_neg": "s", "value": 6}]
        result = verify_circuit_conservation({"branches": branches})
        self.assertTrue(result["conservation_verified"])
        rows = result["branch_currents"]
        self.assertEqual([(r["index"], r["name"], r["n1"], r["n2"]) for r in rows],
                         [(0, "v", "s", "0"), (1, "r1", "s", "0"), (2, "r2", "0", "s")])
        for row, expected in zip(rows, [-6, 4, -2]):
            self.assertAlmostEqual(row["current_A"], expected)
