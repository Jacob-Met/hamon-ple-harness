"""Signed balanced shunt-bank selection at a specified load power factor."""
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def correction(**changes):
    spec = dict(P_kW=100., V_ll_V=400., freq_Hz=50., pf_initial=.95, pf_target=.8)
    spec.update(changes)
    return run_dual_verify_spec(dict(mode="domain",kind="pf_correction",spec=spec))


class ShuntBankSelection(unittest.TestCase):
    def test_larger_lagging_reactive_demand_requires_inductor(self):
        out = correction()
        self.assertTrue(out["dual_verified"])
        capacitance = out["C_equiv_delta_per_phase_F"]
        self.assertTrue(capacitance is None or capacitance >= 0., capacitance)
        self.assertEqual(out["bank_type"], "reactor")
        self.assertGreater(out["bank_inductance_per_leg_H"], 0.)

    def test_leading_load_to_unity_requires_positive_reactor(self):
        out = correction(pf_initial=.8,pf_target=1.,initial_pf_direction="leading")
        self.assertLess(out["Q_initial_kvar"], 0.)
        self.assertEqual(out["bank_type"], "reactor")
        self.assertAlmostEqual(out["Q_target_kvar"], 0.)

    def test_zero_pf_cannot_carry_specified_nonzero_real_power(self):
        for field in ("pf_initial","pf_target"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    correction(**{field:0.})
