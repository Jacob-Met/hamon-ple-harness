"""Reached and asymptotic states of a passive RC precharge circuit."""
import math
import unittest
from hamon_dual_verifier import run_dual_verify_spec


def precharge(f=.5, **changes):
    spec = dict(V_bus_V=400.,C_uF=2000.,R_precharge_ohm=50.,target_fraction=f)
    spec.update(changes)
    return run_dual_verify_spec(dict(mode="domain",kind="rc_precharge",spec=spec))


class RcPrechargeEndpoint(unittest.TestCase):
    def test_finite_target_after_twenty_time_constants(self):
        f = 1.-2.**-32
        r = precharge(f)
        self.assertTrue(r["dual_verified"],r)
        self.assertAlmostEqual(r["t_target_s"],32*math.log(2)*.1,places=14)

    def test_passive_circuit_cannot_verify_negative_stored_energy(self):
        try:
            r = precharge(R_precharge_ohm=-50.,C_uF=-2000.)
        except ValueError:
            return
        self.assertFalse(r["dual_verified"],r)

    def test_target_state_closes_source_capacitor_and_resistor_energy(self):
        r = precharge(.5)
        self.assertEqual(r["stored_energy_J"],160.)  # Existing full-charge capacity
        self.assertEqual(r["target_capacitor_energy_J"],40.)
        self.assertEqual(r["source_work_J"],160.)
        self.assertEqual(r["resistor_heat_J"],120.)
