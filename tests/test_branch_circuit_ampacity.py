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
