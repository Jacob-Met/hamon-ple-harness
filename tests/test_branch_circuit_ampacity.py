"""Independent unadjusted and adjusted checks for an ordinary branch circuit."""
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def check(**changes):
    spec = dict(I_continuous_A=80., I_noncontinuous_A=0., base_ampacity_A=95.,
                temp_factor=1.08, bundling_factor=1.)
    spec.update(changes)
    return run_dual_verify_spec(dict(mode="domain", kind="ampacity_derating", spec=spec))


class BranchCircuitAmpacity(unittest.TestCase):
    def test_cold_ambient_does_not_replace_unadjusted_continuous_sizing(self):
        # 95 A < 1.25*80 A, even though 95*1.08 > 100 A.
        out = check()
        self.assertFalse(out["ampacity_ok"])

    def test_adjusted_insulation_rating_only_has_to_carry_actual_load(self):
        # Unadjusted 100 A meets 125% demand and adjusted 91 A exceeds 80 A.
        out = check(base_ampacity_A=100., temp_factor=.91)
        self.assertTrue(out["ampacity_ok"])

    def test_termination_and_insulation_limits_are_independent(self):
        # A sufficiently hot-rated insulation can pass its own derating check
        # while the cooler equipment termination prevents selecting this wire.
        r = check(base_ampacity_A=130.,termination_ampacity_A=95.,temp_factor=.8)
        self.assertEqual(r["I_unadjusted_available_A"],95.)
        self.assertEqual(r["I_actual_load_A"],80.)
        self.assertTrue(r["adjusted_load_ok"])
        self.assertFalse(r["unadjusted_sizing_ok"])
        self.assertFalse(r["ampacity_ok"])
        r = check(base_ampacity_A=130.,termination_ampacity_A=100.,
                  temp_factor=.8,bundling_factor=.75)
        self.assertTrue(r["unadjusted_sizing_ok"])
        self.assertFalse(r["adjusted_load_ok"])
        self.assertEqual(r["I_derated_allowable_A"],78.)

    def test_only_an_explicit_listed_assembly_uses_actual_continuous_demand(self):
        spec = dict(I_continuous_A=90.,base_ampacity_A=95.,temp_factor=1.)
        self.assertFalse(check(**spec)["ampacity_ok"])
        r = check(**spec,listed_100_percent_assembly=True)
        self.assertEqual(r["I_design_required_A"],90.)
        self.assertTrue(r["ampacity_ok"])
        self.assertFalse(check(**dict(spec,temp_factor=.875),listed_100_percent_assembly=True)["ampacity_ok"])

    def test_separate_contacts_are_inclusive_at_actual_inputs(self):
        import math
        spec = dict(I_continuous_A=64.,I_noncontinuous_A=32.,
                    base_ampacity_A=128.,termination_ampacity_A=112.,temp_factor=.75)
        r = check(**spec)
        self.assertTrue(r["ampacity_ok"])  # 112 A unadjusted, 96 A corrected.
        self.assertEqual(r["I_required_table_A"],128.)
        for terminal,expected in ((math.nextafter(112.,0.),False),(112.,True),
                                  (math.nextafter(112.,math.inf),True)):
            r = check(**dict(spec,termination_ampacity_A=terminal))
            self.assertEqual(r["unadjusted_sizing_ok"],expected)
            self.assertTrue(r["adjusted_load_ok"])
        for demand,expected in ((math.nextafter(32.,0.),True),(32.,True),
                                (math.nextafter(32.,math.inf),False)):
            r = check(**dict(spec,I_noncontinuous_A=demand,termination_ampacity_A=128.))
            self.assertEqual(r["adjusted_load_ok"],expected)
            self.assertTrue(r["unadjusted_sizing_ok"])

    def test_invalid_current_rating_and_factor_inputs(self):
        for cases in (dict(I_continuous_A=-1),dict(I_noncontinuous_A=-1),
                      dict(base_ampacity_A=0),dict(termination_ampacity_A=-1),
                      dict(temp_factor=0),dict(bundling_factor=1.1),
                      dict(listed_100_percent_assembly="false")):
            with self.assertRaises(ValueError):
                check(**cases)
