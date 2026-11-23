"""Datum invariance and physical output replay for a single-stage refrigerator."""
import math
import unittest
from decimal import Decimal as D, localcontext

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


def solve(**params):
    return dispatch_wave7_verifier("thermo_cycles", {"subkind": "vapor_compression", **params})


class RefrigerationDatumTests(unittest.TestCase):
    def test_both_carnot_decisions_and_every_state_difference(self):
        # COPr=6 violates the first reservoir pair's Carnot bound of 5.4.
        # COPr=8 meets the second pair's bound of 9.
        for efficiency, hot, decision in [(0.6, 320., False), (0.8, 300., True)]:
            for datum in (0., 2.**40, -(2.**40), 2.**53, -(2.**53)):
                with self.subTest(eta=efficiency, datum=datum), localcontext() as c:
                    c.prec = 120
                    p = dict(h1=datum, h2s=datum + 2., h3=datum - 20.,
                             eta_c=efficiency, m_dot=.375, T_L=270., T_H=hot)
                    r = solve(**p)
                    self.assertEqual(r["verified"], decision, r)
                    self.assertTrue(r["carnot_checked"])
                    # Independent compression definition and condenser/evaporator
                    # state differences, all on the same high-precision datum.
                    h1, h2s, h3 = map(D, (p["h1"], p["h2s"], p["h3"]))
                    h2 = (h2s - (1-D(efficiency))*h1) / D(efficiency)
                    work, cold, reject = h2-h1, h1-h3, h2-h3
                    reference = dict(h2=h2, h4=h3, w_comp=work, q_evap=cold,
                                     q_cond=reject, COP_R=cold/work, COP_HP=reject/work,
                                     Q_evap_W=D(.375)*cold, W_comp_W=D(.375)*work)
                    for field, value in reference.items():
                        self.assertEqual(r[field], float(value), field)
                    emitted_rise = D(r["h2"])-h1+D(r["h2_correction"])
                    self.assertLess(abs(emitted_rise-work)/work, D("1e-16"))

    def test_trace_lift_at_an_ordinary_enthalpy_reference(self):
        datum = 250000.
        step = math.ulp(datum)
        p = dict(h1=datum, h2s=datum+step, h3=datum-10*step, T_L=270.)
        for eta, hot, verified, cop in [(.6, 320., False, 6.), (.8, 300., True, 8.)]:
            with self.subTest(eta=eta):
                result = solve(**p, eta_c=eta, T_H=hot)
                self.assertEqual(result["verified"], verified)
                self.assertEqual(result["COP_R"], cop)
                self.assertEqual(result["w_comp"], step/eta)

    def test_zero_loss_throttle_and_ordinary_datum(self):
        p = dict(h1=250000., h2s=280000., h3=100000., eta_c=1., m_dot=2.)
        r = solve(**p)
        self.assertTrue(r["verified"], r)
        self.assertFalse(r["carnot_checked"])
        self.assertEqual((r["h2"], r["h2_correction"], r["h4"]), (280000., 0., 100000.))
        self.assertEqual((r["COP_R"], r["COP_HP"], r["W_comp_W"]), (5., 6., 60000.))

    def test_missing_input_differences_and_unrepresentable_public_power(self):
        datum = 2.**54
        self.assertEqual(float(datum + 1.), datum)
        for h2s, h3 in [(datum+1., datum-16.), (datum+8., datum-1.)]:
            with self.subTest(h2s=h2s,h3=h3):
                self.assertFalse(solve(h1=datum,h2s=h2s,h3=h3)["verified"])
        for mass, q in [(1.e305, 1.e5), (math.ulp(0.), .25)]:
            with self.subTest(mass=mass):
                self.assertFalse(solve(h1=0.,h2s=q,h3=-q,m_dot=mass)["verified"])
        for change in [dict(h1=float("inf")), dict(T_L=-5.,T_H=300.), dict(eta_c=0.),
                       dict(m_dot=float("nan"))]:
            self.assertFalse(solve(**dict(dict(h1=2.,h2s=3.,h3=1.), **change))["verified"])

    def test_specific_balance_does_not_set_equipment_capacity(self):
        states = dict(h1=250000., h2s=280000., h3=100000., eta_c=.75)
        result = solve(**states)
        self.assertTrue(result["verified"])
        self.assertFalse(result["rates_identified"])
        self.assertEqual((result["COP_R"], result["q_evap"], result["w_comp"]), (3.75, 150000., 40000.))
        for key in ("m_dot_kg_s", "Q_evap_W", "W_comp_W", "Q_cond_W", "rate_basis"):
            self.assertIsNone(result[key], key)

        observations = {"m_dot": .5, "Q_evap_W": 75000., "W_comp_W": 20000., "Q_cond_W": 95000.}
        for name, measurement in observations.items():
            with self.subTest(basis=name):
                rated = solve(**states, **{name: measurement})
                self.assertTrue(rated["verified"])
                self.assertTrue(rated["rates_identified"])
                self.assertEqual(rated["rate_basis"], name)
                self.assertEqual(rated["m_dot_kg_s"], .5)
                for field in ("Q_evap_W", "W_comp_W", "Q_cond_W"):
                    self.assertEqual(rated[field], observations[field])
        self.assertTrue(solve(**states, **observations)["verified"])

    def test_inconsistent_measurements_and_partial_reservoirs_fail(self):
        states = dict(h1=250000., h2s=280000., h3=100000., eta_c=.75)
        for supplied in ({"Q_evap_W": 75000., "W_comp_W": 21000.},
                         {"m_dot": 2., "Q_cond_W": 95000.},
                         {"Q_evap_W": False}, {"W_comp_W": float("nan")},
                         {"Q_cond_W": -1.}, {"T_L": 270.}, {"T_H": 310.}):
            with self.subTest(supplied=supplied):
                self.assertFalse(solve(**states, **supplied)["verified"])
