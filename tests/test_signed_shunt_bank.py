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

    def test_balanced_leg_component_and_replay_for_both_connections(self):
        import math
        for connection in ("delta","wye"):
            leg = 400. if connection == "delta" else 400./math.sqrt(3.)
            for pi,pt,bank in ((.8,.95,"capacitor"),(.95,.8,"reactor")):
                out = correction(pf_initial=pi,pf_target=pt,bank_connection=connection)
                self.assertTrue(out["dual_verified"])
                self.assertEqual(out["bank_type"],bank)
                self.assertAlmostEqual(out["bank_leg_voltage_V"],leg)
                omega = 2*math.pi*50
                if bank == "capacitor":
                    self.assertIsNone(out["bank_inductance_per_leg_H"])
                    q = -3*omega*leg**2*out["bank_capacitance_per_leg_F"]/1000
                    self.assertGreater(out["C_equiv_delta_per_phase_F"],0.)
                else:
                    self.assertIsNone(out["bank_capacitance_per_leg_F"])
                    self.assertIsNone(out["C_equiv_delta_per_phase_F"])
                    q = 3*leg**2/(omega*out["bank_inductance_per_leg_H"])/1000
                self.assertAlmostEqual(q,out["Q_shunt_kvar"])
                self.assertAlmostEqual(q,-out["Qc_required_kvar"])
                self.assertAlmostEqual(out["Q_initial_kvar"]+q,out["Q_target_kvar"])

    def test_force_zero_and_cross_unity_with_equal_pf_magnitudes(self):
        for initial,target,bank in (("lagging","lagging","none"),("leading","leading","none"),
                                    ("lagging","leading","capacitor"),("leading","lagging","reactor")):
            out = correction(pf_initial=.8,pf_target=.8,initial_pf_direction=initial,target_pf_direction=target)
            self.assertEqual(out["bank_type"],bank)
            self.assertAlmostEqual(out["Q_shunt_kvar"], {"none":0., "capacitor":-150., "reactor":150.}[bank])
            if bank == "none":
                self.assertIsNone(out["bank_capacitance_per_leg_F"])
                self.assertIsNone(out["bank_inductance_per_leg_H"])
                self.assertEqual(out["C_equiv_delta_per_phase_F"],0.)

    def test_resolve_adjacent_pf_settings_and_preserve_weak_bank(self):
        import math
        pi = .8
        for target,bank in ((math.nextafter(pi,0.),"reactor"),(pi,"none"),
                            (math.nextafter(pi,1.),"capacitor")):
            out = correction(pf_initial=pi,pf_target=target)
            self.assertEqual(out["bank_type"],bank)
            if bank != "none":
                self.assertNotEqual(out["Q_shunt_kvar"],0.)
                self.assertLess(out["relative_replay_error"],1e-10)

    def test_refuse_undefined_load_or_device(self):
        for change in (dict(P_kW=-1),dict(P_kW=0),dict(V_ll_V=0),dict(freq_Hz=0),
                       dict(pf_target=1.1),dict(bank_connection="series"),
                       dict(initial_pf_direction="capacitor")):
            with self.assertRaises(ValueError):
                correction(**change)
