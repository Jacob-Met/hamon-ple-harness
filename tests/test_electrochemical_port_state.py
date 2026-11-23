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

    def test_both_polarities_obey_active_and_passive_port_balances(self):
        for mode, sign in (("galvanic",1),("electrolytic",-1)):
            for emf in (-1.5,.75):
                out=cell(mode=mode,E0_V=emf,current_A=2,R_internal_ohm=.125,
                         eta_activation_V=.125,faradaic_efficiency=.75)
                self.assertTrue(out["verified"],out)
                voltage=emf-sign*(.25+.125)
                self.assertEqual(out["V_cell_V"],voltage)
                self.assertEqual(out["terminal_power_out_W"],sign*2*voltage)
                self.assertEqual(out["ohmic_dissipation_W"],.5)
                self.assertEqual(out["activation_dissipation_W"],.25)
                self.assertEqual(out["reversible_port_power_out_W"],sign*2*emf)
                self.assertEqual(out["assigned_reaction_reversible_power_out_W"],sign*1.5*emf)
                self.assertEqual(out["unassigned_current_magnitude_A"],.5)
                self.assertAlmostEqual(-out["delta_G_J_mol"]*out["assigned_reference_extent_rate_mol_s"],
                                       out["assigned_reaction_reversible_power_out_W"])
                passive=cell(mode=mode,E0_V=emf,current_A=2,R_internal_ohm=.125,
                             eta_activation_V=.125,external_circuit="passive_load")
                self.assertEqual(passive["verified"],sign*voltage>=0)
                if not passive["verified"]:
                    self.assertNotIn("product_moles",passive)
                    self.assertIsNone(passive["target_time_s"])

    def test_open_circuit_and_initial_zero_target(self):
        for mode in ("galvanic","electrolytic"):
            out=cell(mode=mode,current_A=0,external_circuit="passive_load",target_mass_g=0)
            self.assertTrue(out["verified"],out)
            self.assertEqual(out["electrical_operating_state"],"open_circuit")
            self.assertEqual(out["terminal_power_out_W"],0)
            self.assertEqual(out["target_mass_status"],"all_nonnegative_times")
        blocked=cell(external_circuit="passive_load",target_mass_g=0)
        self.assertFalse(blocked["verified"])
        self.assertEqual(blocked["target_time_s"],0.)

    def test_nernst_term_resolves_a_loss_cancellation(self):
        # Cancellation of the exact ohmic term leaves the source's activity
        # contribution intact, even if the displayed Nernst potential ties E0.
        for q in (math.nextafter(1.,0.),math.nextafter(1.,2.)):
            out=cell(E0_V=1,current_A=1,R_internal_ohm=1,Q_reaction=q)
            self.assertTrue(out["verified"],out)
            self.assertEqual(out["E_nernst_V"],1.)
            self.assertEqual(out["terminal_power_out_W"]>0,q<1)
            self.assertNotEqual(out["V_cell_V"],0.)
            self.assertEqual(out["passive_load_feasible"],q<1)
        large=2**54
        out=cell(E0_V=large+1,current_A=1,R_internal_ohm=large)
        self.assertTrue(out["verified"],out)
        self.assertEqual(out["V_cell_V"],1.)
        self.assertEqual(out["terminal_power_out_W"],1.)
