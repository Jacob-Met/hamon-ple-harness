"""Electrical feasibility and Faradaic target on the single constant-current cell."""
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def cell(**overrides):
    spec = dict(subkind="electrochemistry", mode="galvanic", E0_V=1, n_electrons=1,
                Q_reaction=1, current_A=2, time_s=10, R_internal_ohm=1,
                molar_mass_g_mol=1, target_mass_g=1)
    return run_router_conservation_cad_verifier(
        dict(mode="electrochem_separations", spec=dict(spec, **overrides)))


class ElectrochemicalPortState(unittest.TestCase):
    def test_passive_load_cannot_supply_the_requested_electrical_power(self):
        # At 2 A a 1 V reversible source releases 2 W; the 1 ohm resistor
        # already consumes 4 W. A passive external load cannot supply 2 W.
        result = cell(external_circuit="passive_load")
        self.assertFalse(result["verified"],result)
        self.assertEqual(result["V_cell_V"],-1.)
        self.assertFalse(result["operating_point_feasible"])
        self.assertEqual(result["target_mass_status"],"infeasible_operating_point")
        self.assertIsNone(result["target_time_s"])

    def test_prescribed_current_can_absorb_external_power_in_discharge_polarity(self):
        for options in ({}, {"external_circuit":"prescribed_current"}):
            result = cell(**options)
            self.assertTrue(result["verified"],result)
            self.assertEqual(result["terminal_power_out_W"],-2.)
            self.assertEqual(result["reversible_port_power_out_W"],2.)
            self.assertEqual(result["ohmic_dissipation_W"],4.)
            self.assertFalse(result["passive_load_feasible"])
            self.assertEqual(result["target_mass_status"],"unique")

    def test_passive_short_and_its_adjacent_current_states(self):
        for current, feasible in ((math.nextafter(1.,0.),True),(1.,True),
                                  (math.nextafter(1.,2.),False)):
            with self.subTest(current=current):
                result=cell(current_A=current,external_circuit="passive_load")
                self.assertEqual(result["verified"],feasible,result)
                self.assertEqual(result["passive_load_feasible"],feasible)
                if current==1.:
                    self.assertEqual(result["terminal_power_out_W"],0.)
                    self.assertEqual(result["electrical_operating_state"],"short_circuit")
