"""Independent balances and boundary controls for jacketed CSTR branches."""
import copy
from fractions import Fraction
import math
import unittest
import numpy as np

from hamon_wave6_physics_verifiers import solve_reactor_kinetics
from hamon_cstr_thermal import R_GAS, _bounds


def benchmark(coolant=300):
    # Notre Dame CBE30338 / SEMD exothermic reactor, converted to SI.
    # This repository consistently uses R=8.31446261815324, rather than the
    # tutorial's rounded 8.314. The independent oracle uses the same inputs.
    return {"reactor_type": "cstr", "order": 1, "C_A0": 1000., "v0": .1/60,
            "volume": .1, "A_preexp": 7.2e10/60, "E_a": 72750., "T0": 350.,
            "delta_T_adiabatic": 50000/239., "steady_state_branch": "all",
            "heat_transfer": {"UA_W_K": 50000/60, "rho_kg_m3": 1000.,
                              "Cp_J_kg_K": 239., "coolant_T_K": coolant}}


def fold(ua=.5):
    return {"reactor_type": "cstr", "order": 1, "C_A0": 1., "v0": 1., "volume": 1.,
            "k_ref": 1., "T_ref": 512., "E_a": R_GAS*8192, "T0": 480.,
            "delta_T_adiabatic": 192., "steady_state_branch": "all",
            "heat_transfer": {"UA_W_K": ua, "rho_kg_m3": 1.,
                              "Cp_J_kg_K": 1., "coolant_T_K": 384.}}


class JacketedCstr(unittest.TestCase):
    def test_published_model_has_three_distinct_dynamical_states(self):
        p = benchmark()
        result = solve_reactor_kinetics(p)
        self.assertTrue(result["verified"])
        rows = result["steady_states"]
        self.assertEqual(len(rows), 3)
        # Frozen before implementation using the original two differential
        # balances and independent SciPy bracket searches in temperature.
        expected = [324.48605305000467, 349.9623318873878, 369.7246520915924]
        self.assertEqual([r["linear_stability"] for r in rows],
                         ["asymptotically_stable", "unstable_saddle", "unstable"])
        q, v, c0, rho, cp, ua = p["v0"], p["volume"], p["C_A0"], 1000., 239., 50000/60
        for row, expected_t in zip(rows, expected):
            with self.subTest(temperature=expected_t):
                t, c = row["exit_T_K"], row["C_A_mol_m3"]
                self.assertAlmostEqual(t, expected_t, places=10)
                k = p["A_preexp"]*math.exp(-p["E_a"]/(R_GAS*t))
                dk = k*p["E_a"]/(R_GAS*t*t)
                dc = q/v*(c0-c)-k*c
                dt = q/v*(350-t)+(50000/(rho*cp))*k*c+ua/(rho*cp*v)*(300-t)
                self.assertLess(abs(dc), 2e-11)
                self.assertLess(abs(dt), 2e-12)
                jacobian = [[-q/v-k, -dk*c],
                            [50000/(rho*cp)*k, -q/v-ua/(rho*cp*v)+50000/(rho*cp)*dk*c]]
                np.testing.assert_allclose(row["jacobian"], jacobian, rtol=2e-13, atol=1e-15)
                eigenvalues = sorted(np.linalg.eigvals(jacobian), key=lambda z: (z.real, z.imag))
                got = sorted((complex(*z) for z in row["eigenvalues_s_inv"]), key=lambda z: (z.real, z.imag))
                np.testing.assert_allclose(got, eigenvalues, rtol=2e-12, atol=1e-15)
        self.assertLess(result["temperature_bounds_K"][1], 384)
        # Being the hottest root is insufficient to establish stability.
        self.assertTrue(all(z[0] > 0 and z[1] != 0 for z in rows[-1]["eigenvalues_s_inv"]))

    def test_coolant_changes_branch_count_and_stability(self):
        expected = [(290, 312.6587596626876, "asymptotically_stable"),
                    (295, 317.74658211425844, "asymptotically_stable"),
                    (305, 378.07277063838217, "unstable"),
                    (310, 383.8921610143717, "asymptotically_stable")]
        for coolant, temperature, status in expected:
            p = benchmark(coolant)
            del p["steady_state_branch"]
            with self.subTest(coolant=coolant):
                r = solve_reactor_kinetics(p)
                self.assertEqual(r["steady_state_count"], 1)
                self.assertAlmostEqual(r["exit_T_K"], temperature, places=10)
                self.assertEqual(r["linear_stability"], status)

    def test_adiabatic_limit_preserves_legacy_branches(self):
        p = benchmark()
        p["heat_transfer"]["UA_W_K"] = 0
        heated = solve_reactor_kinetics(p)
        plain = solve_reactor_kinetics({k: v for k, v in p.items() if k != "heat_transfer"})
        self.assertEqual(heated["steady_state_count"], plain["steady_state_count"])
        for a, b in zip(heated["steady_states"], plain["steady_states"]):
            for key in ("conversion_X", "unreacted_fraction", "exit_T_K"):
                self.assertEqual(a[key], b[key])

    def test_exact_fold_and_adjacent_heat_transfer_inputs(self):
        counts = []
        for ua in (math.nextafter(.5, 0), .5, math.nextafter(.5, 1)):
            r = solve_reactor_kinetics(fold(ua))
            self.assertTrue(r["verified"])
            counts.append(r["steady_state_count"])
            if ua == .5:
                row = r["steady_states"][-1]
                self.assertEqual(row["conversion_X"], .5)
                self.assertEqual(row["exit_T_K"], 512)
                self.assertEqual(row["linear_stability"], "nonhyperbolic")
                self.assertEqual(row["determinant_s_inv2"], 0)
                self.assertEqual(row["trace_s_inv"], -.5)
        self.assertEqual(counts, [3, 2, 1])

    def test_exact_hopf_is_not_declared_asymptotically_stable(self):
        p = fold(2.)
        p.update(T0=512., delta_T_adiabatic=320.)
        p["heat_transfer"]["coolant_T_K"] = 432.
        r = solve_reactor_kinetics(p)["steady_states"][0]
        self.assertEqual(r["conversion_X"], .5)
        self.assertEqual(r["linear_stability"], "nonhyperbolic")
        self.assertEqual((r["trace_s_inv"], r["determinant_s_inv2"]), (0., 1.))
        self.assertEqual(r["eigenvalues_s_inv"], [[0., 1.], [0., -1.]])

    def test_equivalent_temperature_scaling_preserves_all_branches(self):
        original = solve_reactor_kinetics(benchmark())["steady_states"]
        p = benchmark()
        scale = 2.**-60
        for key in ("T0", "delta_T_adiabatic", "E_a"):
            p[key] *= scale
        p["heat_transfer"]["coolant_T_K"] *= scale
        result = solve_reactor_kinetics(p)
        self.assertTrue(result["verified"])
        self.assertEqual(result["steady_state_count"], 3)
        for a, b in zip(original, result["steady_states"]):
            self.assertEqual(a["conversion_X"], b["conversion_X"])
            self.assertEqual(a["exit_T_K"]*scale, b["exit_T_K"])
            self.assertEqual(a["linear_stability"], b["linear_stability"])

    def test_large_integer_kinetic_input_does_not_round_to_a_fold(self):
        magnitude = 2**53
        counts = []
        for offset in (-1, 0, 1, 2):
            p = fold(magnitude//2)
            p.update(v0=magnitude, k_ref=magnitude+offset)
            result = solve_reactor_kinetics(p)
            self.assertTrue(result["verified"])
            counts.append(result["steady_state_count"])
        self.assertEqual(counts, [1, 2, 3, 3])

    def test_explicit_branch_choice_and_ambiguity(self):
        p = benchmark()
        rows = solve_reactor_kinetics(p)["steady_states"]
        for index, row in enumerate(rows):
            self.assertEqual(solve_reactor_kinetics({**p, "steady_state_branch": index})["exit_T_K"], row["exit_T_K"])
        del p["steady_state_branch"]
        with self.assertRaisesRegex(ValueError, "multiple physical"):
            solve_reactor_kinetics(p)
        for bad in (True, 1., "1", -1, 3):
            with self.subTest(branch=bad), self.assertRaises(ValueError):
                solve_reactor_kinetics({**p, "steady_state_branch": bad})

    def test_second_order_and_endothermic_balance(self):
        p = {**benchmark(), "order": 2, "k": .001, "delta_T_adiabatic": -25.}
        del p["A_preexp"], p["E_a"]
        r = solve_reactor_kinetics(p)
        self.assertEqual(r["steady_state_count"], 1)
        row = r["steady_states"][0]
        c, t = row["C_A_mol_m3"], row["exit_T_K"]
        self.assertAlmostEqual(p["v0"]*(p["C_A0"]-c), p["volume"]*p["k"]*c*c, places=12)
        self.assertEqual(row["linear_stability"], "asymptotically_stable")
        self.assertTrue(r["temperature_bounds_K"][0] <= t <= r["temperature_bounds_K"][1])

    def test_thermal_contract_never_silently_ignores_incompatible_inputs(self):
        changes = [dict(reactor_type="pfr"), dict(gas_phase=True), dict(epsilon_A=.01),
                   dict(conversion=.5), dict(T=350), dict(volume=0), dict(order=-1),
                   dict(F_A0=100), dict(k=1), dict(heat_transfer={})]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                solve_reactor_kinetics({**benchmark(), **change})
        for value in (True, "239", float("nan"), float("inf"), 10**500, 0, -1):
            p = benchmark()
            p["heat_transfer"]["Cp_J_kg_K"] = value
            with self.subTest(value=str(value)[:24]), self.assertRaises(ValueError):
                solve_reactor_kinetics(p)
        p = benchmark()
        p["heat_transfer"]["unit"] = "watts"
        with self.assertRaises(ValueError):
            solve_reactor_kinetics(p)

    def test_serialized_stability_enclosures_round_outward(self):
        for value in (Fraction(1, 3), Fraction(-1, 3), Fraction(0)):
            low, high = _bounds([value, value], "control")
            self.assertLessEqual(Fraction(low), value)
            self.assertGreaterEqual(Fraction(high), value)


if __name__ == "__main__":
    unittest.main()
