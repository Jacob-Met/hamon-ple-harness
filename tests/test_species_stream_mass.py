import math
import unittest

from hamon_pareto import dispatch_ple_tool


def stream(formula, n, m):
    return {"formula": formula, "ndot_mol_s": n, "mdot_kg_s": m}


def check(ins, outs, **kwargs):
    return dispatch_ple_tool("hamon_conservation_cad_verifier", {
        "mode": "chemical", "spec": {"inlets": ins, "outlets": outs, **kwargs}
    })


class SpeciesStreamMassTests(unittest.TestCase):
    def test_swapped_mass_has_two_closing_global_balances(self):
        ins = [stream("H2O", 1, 0.018015), stream("NaCl", 1, 0.05844)]
        self.assertTrue(check(ins, ins)["conservation_verified"])
        outs = [stream("H2O", 1, 0.05844), stream("NaCl", 1, 0.018015)]
        for a, b, side in ((ins, outs, "outlets"), (outs, ins, "inlets")):
            result = check(a, b)
            self.assertTrue(result["mass_balance_ok"])
            self.assertTrue(result["species_balance_ok"])
            self.assertFalse(result["stream_mass_mole_ok"])
            self.assertFalse(result["conservation_verified"])
            self.assertEqual([(x["side"], x["index"]) for x in result["stream_mass_mole_errors"]],
                             [(side, 0), (side, 1)])

    def test_formula_unit_consistency_on_split_and_hydrated_streams(self):
        wet = [stream("CaSO4·2H2O", 2.5, 0.430415)]
        dry = [stream("CaSO4", 2.5, 0.340340), stream("H2O", 5, 0.090075)]
        for a, b in ((wet, dry), (dry, wet)):
            self.assertTrue(check(a, b)["conservation_verified"])

    def test_zero_unsupported_and_scaled_rates(self):
        for record, valid in ((stream("NaCl", 0, 0), True),
                              (stream("NaCl", 0, math.ulp(0.0)), False),
                              (stream("NaCl", math.ulp(0.0), 0), False),
                              (stream("NaCl", -1, -0.05844), False),
                              (stream("Tc", 1, 0.099), False),
                              (stream("Xx", 1, 0.1), False),
                              (stream("H2O", 1e-225, 1.8015e-227), True),
                              (stream("H2O", 1e-225, 1.85e-227), False),
                              (stream("H2O", 1, 1), False)):
            with self.subTest(record=record):
                self.assertIs(check([record], [record])["conservation_verified"], valid)

    def test_quoted_atomic_weight_uncertainty_and_user_tolerance(self):
        # The abridged normal-material Li interval is [6.88, 7.00].
        for mass, rtol, valid in ((0.006885, 0, True), (0.006875, 0, False),
                                  (0.006875, 0.001, True), (0.00701, 0, False)):
            record = stream("Li", 1, mass)
            with self.subTest(mass=mass, rtol=rtol):
                self.assertIs(check([record], [record], rtol=rtol)["conservation_verified"], valid)

    def test_sample_molar_mass_is_an_explicit_whole_formula_input(self):
        heavy = stream("H2O", 2, 0.0400552)
        self.assertFalse(check([heavy], [heavy])["conservation_verified"])
        sample = dict(heavy, molar_mass_g_mol=20.0276)
        result = check([sample], [sample])
        self.assertTrue(result["conservation_verified"])
        self.assertEqual(result["declared_molar_mass_streams"],
                         [{"side": "inlets", "index": 0}, {"side": "outlets", "index": 0}])
        for declared in (10.0138, True, None, "unknown", -20.0276, math.inf, math.nan):
            row = dict(heavy, molar_mass_g_mol=declared)
            self.assertFalse(check([row], [row])["conservation_verified"])
        for formula, expected in (("Tc", True), ("Xx", False)):
            row = dict(stream(formula, 1, 0.099), molar_mass_g_mol=99)
            self.assertIs(check([row], [row])["conservation_verified"], expected)

    def test_single_coordinate_contracts(self):
        for row in ({"mdot_kg_s": 1}, {"formula": "NaCl", "ndot_mol_s": 1}):
            result = check([row], [row])
            self.assertTrue(result["conservation_verified"])
            self.assertNotIn("stream_mass_mole_ok", result)
