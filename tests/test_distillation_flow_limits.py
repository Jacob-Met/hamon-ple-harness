"""Material balances and finite-stage limits for the registered binary column."""
from fractions import Fraction
import unittest

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


class DistillationLimits(unittest.TestCase):
    def solve(self, **overrides):
        spec = {"subkind": "distillation", "alpha": 2, "x_D": 0.6, "x_B": 0.4,
                "z_F": 0.5, "q": 1, "F_mol_s": 100, **overrides}
        return dispatch_wave7_verifier("McCabe-Thiele binary distillation", spec)

    def test_negative_rectifying_or_stripping_flow(self):
        for q, reflux, field, flow in (
            (1, -0.1, "L_rect_mol_s", -5),
            (0, 0.8, "V_strip_mol_s", -10),
            (-0.2, 1.2, "V_strip_mol_s", -10),
        ):
            with self.subTest(q=q, reflux=reflux):
                result = self.solve(q=q, R=reflux)
                self.assertIs(result["verified"], False)
                self.assertEqual(result["error"], "negative_internal_section_flow")
                self.assertEqual(result[field], flow)

    def test_zero_reflux_stripping_section(self):
        # At R=0, D=B=50, the sole operating line is y=2x-2/5.
        # The equilibrium abscissas are 3/7 and 8/27. The last step is fractional.
        for operating in ({"R": 0}, {}):
            with self.subTest(operating=operating):
                result = self.solve(**operating)
                self.assertIs(result["verified"], True)
                self.assertEqual(result["R_min"], 0)
                self.assertAlmostEqual(result["R_feed_pinch"], -0.4)
                self.assertEqual(result["L_rect_mol_s"], 0)
                self.assertEqual(result["V_strip_mol_s"], 50)
                self.assertEqual(result["section_mode"], "stripping_only")
                self.assertEqual(result["feed_stage"], 1)
                self.assertAlmostEqual(result["N_theoretical"], float(Fraction(152, 125)))

    def test_zero_boilup_rectifying_section(self):
        # The bottom balance fixes x=2/5. The active line is y=(x+3/5)/2.
        result = self.solve(q=0, R=1)
        self.assertIs(result["verified"], True)
        self.assertEqual(result["R_min"], 1)
        self.assertEqual(result["V_strip_mol_s"], 0)
        self.assertEqual(result["L_strip_mol_s"], result["B_mol_s"])
        self.assertEqual(result["section_mode"], "rectifying_only")
        self.assertEqual(result["feed_stage"], 2)
        self.assertAlmostEqual(result["N_theoretical"], float(Fraction(101, 75)))
        result = self.solve(q=0)
        self.assertIs(result["verified"], True)
        self.assertEqual(result["R_operating"], 1.25)
        self.assertEqual(result["V_strip_mol_s"], 12.5)

    def test_flash_requires_fixed_split_and_equilibrium(self):
        products = {"q": 0.5, "z_F": 0.375, "x_D": 0.5, "x_B": 0.25}
        for operating in ({"R": 0}, {}):
            result = self.solve(**products, alpha=3, **operating)
            self.assertIs(result["verified"], True)
            self.assertEqual(result["section_mode"], "single_equilibrium_flash")
            self.assertEqual(result["N_theoretical"], 1)
            self.assertEqual(result["L_rect_mol_s"], 0)
            self.assertEqual(result["V_strip_mol_s"], 0)
        for alpha in (2, 4):
            with self.subTest(alpha=alpha):
                result = self.solve(**products, R=0, alpha=alpha)
                self.assertIs(result["verified"], False)
                self.assertEqual(result["error"], "zero_circulation_flash_products_not_in_equilibrium")

    def test_one_equilibrium_contact_can_attain_fenske_equality(self):
        for q, reflux, expected_mode in ((1, 0, "stripping_only"), (0, 1, "rectifying_only"),
                                          (1, 0.25, "two_section")):
            with self.subTest(q=q, reflux=reflux):
                result = self.solve(q=q, R=reflux, alpha=3, x_D=0.5, x_B=0.25, z_F=0.375)
                self.assertIs(result["verified"], True)
                self.assertEqual(result["section_mode"], expected_mode)
                self.assertEqual(result["N_min_fenske"], 1)
                self.assertEqual(result["N_theoretical"], 1)
                self.assertEqual(result["N_stages_integer"], 1)

    def test_genuine_pinches_at_finite_and_single_section_boundaries(self):
        cases = ({"x_D": 0.95, "x_B": 0.05, "R": 1.7},
                 {"alpha": 3, "z_F": 0.25, "x_D": 0.5, "x_B": 0.125, "R": 0},
                 {"alpha": 3, "z_F": 0.5, "x_D": 0.75, "x_B": 0.25, "q": 0, "R": 1})
        for params in cases:
            with self.subTest(params=params):
                result = self.solve(**params)
                self.assertIs(result["verified"], False)
                self.assertEqual(result["error"], "reflux_below_minimum_pinch")

    def test_fractional_feed_and_energy_conditioned_section_flows(self):
        for q, reflux, levels, count in (
            (0.5, 0.5, (25, 75, 75, 25), Fraction(253, 200)),
            (-0.2, 1.8, (90, 140, 70, 20), Fraction(172, 135)),
            (1.2, 0.1, (5, 55, 125, 75), Fraction(963, 800)),
        ):
            with self.subTest(q=q):
                result = self.solve(q=q, R=reflux)
                self.assertIs(result["verified"], True)
                fields = ("L_rect_mol_s", "V_rect_mol_s", "L_strip_mol_s", "V_strip_mol_s")
                self.assertEqual(tuple(result[k] for k in fields), levels)
                self.assertAlmostEqual(result["N_theoretical"], float(count))
        self.assertIs(self.solve(R_mult=-1)["verified"], False)
