"""Physical first-law, freezing-limit and participant-intake controls."""
import json
import unittest
from unittest.mock import patch

import hamon_pareto
from hamon_pde_control_psychro_fem import verify_psychrometric_process as solve


INVENTORY = json.dumps(dict(schema="hamon.ple.intake.v1", scanned_files=0,
                            engineering_file_count=0, files=[], warnings=[], by_extension={}))
FROST = dict(mode="cooling_coil", inlet={"t_db_c":-2., "rh":.9},
             outlet={"t_db_c":-12., "rh":1.}, m_da_kg_s=1., t_condensate_c=-12.)
WET = dict(mode="cooling_coil", inlet={"t_db_c":24.,"w_kg_per_kgda":.012},
           outlet={"t_db_c":8.,"w_kg_per_kgda":.006}, m_da_kg_s=2.)


def route(spec):
    instruction = json.dumps({"mode":"psychrometrics", "spec":spec})
    return hamon_pareto.route_from_intake_prompt(INVENTORY, instruction=instruction)


class CoilProductTest(unittest.TestCase):
    def test_frost_and_assumed_liquid(self):
        original = solve(FROST)
        frost = solve({**FROST, "condensate_phase":"ice"})
        self.assertTrue(original["passed"] and frost["passed"])
        self.assertEqual(original["phase_source"], "assumed_liquid")
        self.assertAlmostEqual(original["q_total_kw"],13.998915040544095,places=10)
        self.assertAlmostEqual(frost["q_total_kw"],14.47198881207644,places=10)
        self.assertAlmostEqual(frost["water_product_enthalpy_kj_kg"],-358.1384)
        self.assertAlmostEqual(frost["deposited_ice_kg_s"],.0015364207159459689)
        self.assertEqual(frost["drained_liquid_kg_s"],0)
        self.assertEqual(frost["phase_source"],"supplied")
        self.assertEqual(frost["water_temperature_source"],"supplied")

    def test_freezing_limits_and_known_mixture(self):
        def product(phase,t,fraction=None):
            spec = {**WET, "condensate_phase":phase, "t_condensate_c":t}
            if fraction is not None:
                spec["ice_mass_fraction"] = fraction
            result = solve(spec)
            self.assertTrue(result["passed"],result)
            return result
        liquid, ice = product("liquid",0), product("ice",0)
        mass = .012
        self.assertAlmostEqual(ice["q_total_kw"]-liquid["q_total_kw"],333.5*mass)
        for f in (0,.25,.5,.75,1):
            r = product("mixed",0,f)
            self.assertAlmostEqual(r["deposited_ice_kg_s"],mass*f)
            self.assertAlmostEqual(r["drained_liquid_kg_s"],mass*(1-f))
            self.assertAlmostEqual(r["q_total_kw"],liquid["q_total_kw"]+333.5*mass*f)
        self.assertLess(abs(product("ice",-1e-7)["q_total_kw"]-ice["q_total_kw"]),4e-9)
        self.assertLess(abs(product("liquid",1e-7)["q_total_kw"]-liquid["q_total_kw"]),6e-9)

    def test_air_state_crossing_and_volumetric_flow(self):
        rates = []
        for t in (-1e-7,0,1e-7):
            spec = {**FROST, "inlet":{"t_db_c":12,"rh":.8}, "outlet":{"t_db_c":t,"rh":.8},
                    "t_condensate_c":-5, "condensate_phase":"ice"}
            result = solve(spec)
            self.assertTrue(result["passed"] and route(spec)["presolve_verified"])
            rates.append(result["q_total_kw"])
        self.assertLess(max(rates)-min(rates),1e-6)
        spec = {**FROST, "condensate_phase":"ice", "v_dot_in_m3_s":.8}
        del spec["m_da_kg_s"]
        self.assertTrue(route(spec)["presolve_verified"])

    def test_any_liquid_share_obeys_local_boiling_condition(self):
        spec = dict(mode="cooling_coil", p_atm_pa=550, m_da_kg_s=1,
                    inlet={"t_db_c":-10,"rh":.65}, outlet={"t_db_c":-20,"rh":.95},
                    t_condensate_c=0)
        for fraction in (0,.5,1):
            mixed = {**spec,"condensate_phase":"mixed","ice_mass_fraction":fraction}
            self.assertEqual(solve(mixed)["passed"],fraction==1)
            self.assertEqual(route(mixed)["presolve_verified"],fraction==1)
        self.assertFalse(solve({**spec,"condensate_phase":"liquid"})["passed"])

    def test_dry_coil_and_flow_scaling(self):
        r = solve({**WET, "outlet":{"t_db_c":20.,"w_kg_per_kgda":.012},
                   "condensate_phase":"ice", "t_condensate_c":-8})
        self.assertTrue(r["passed"])
        self.assertEqual((r["deposited_ice_kg_s"],r["drained_liquid_kg_s"],r["q_latent_kw"]),(0,0,0))
        self.assertAlmostEqual(r["q_total_kw"],2*(1.006+1.86*.012)*4)
        for flow in (1e-15,1e12):
            result = solve({**FROST,"condensate_phase":"ice","m_da_kg_s":flow})
            self.assertAlmostEqual(result["q_total_kw"]/flow,14.47198881207644)
            self.assertLess(result["sensible_heat_ratio"],.9)

    def test_contradictory_and_missing_assumptions(self):
        for change in (dict(condensate_phase="ice",t_condensate_c=.01),dict(condensate_phase="liquid"),
                       dict(condensate_phase="ice",t_condensate_c=-61),dict(condensate_phase="mixed"),
                       dict(condensate_phase="mixed",t_condensate_c=0),
                       dict(condensate_phase="mixed",t_condensate_c=0,ice_mass_fraction=1.1),
                       dict(condensate_phase="ice",ice_mass_fraction=1),
                       dict(ice_mass_fraction=.5),dict(condensate_phase="guess"),
                       dict(condensate_phase="ice",v_dot_in_m3_s=1),
                       dict(condensate_phase="ice",load_split="ashrae_dew_point"),
                       dict(condensate_phase="ice",t_condensate_c=True),
                       dict(condensate_phase="ice",inlet={"t_db_c":-2.,"rh":.9,"w_kg_per_kgda":.1})):
            spec = {**FROST, **change}
            with self.subTest(change=change):
                self.assertFalse(solve(spec)["passed"])
                self.assertFalse(route(spec)["presolve_verified"])

    def test_participant_receives_full_registered_result(self):
        spec = {**FROST,"condensate_phase":"ice"}
        plan = route(spec)
        self.assertEqual((plan["thinking_tier"],plan["presolve_verified"]),("low",True))
        actual = json.loads(plan["presolve_result_json"])
        self.assertEqual(actual,solve(spec))
        for change in ({"phase_source":"assumed_liquid"},{"deposited_ice_kg_s":0.},
                       {"q_total_kw":13.998915040544095},{"drained_liquid_kg_s":.001},
                       {"water_product_enthalpy_kj_kg":0.},{"water_temperature_source":"outlet_dry_bulb"}):
            with patch.object(hamon_pareto,"dispatch_ple_tool",return_value={**actual,**change}):
                self.assertFalse(route(spec)["presolve_verified"])
        self.assertFalse(route(FROST)["presolve_verified"])
        self.assertFalse(route({**spec,"send_report":True})["presolve_verified"])
        self.assertTrue(route({**WET,"condensate_phase":"liquid",
                               "load_split":"ashrae_dew_point"})["presolve_verified"])
        self.assertTrue(route({**WET,"condensate_phase":"mixed","t_condensate_c":0,
                               "ice_mass_fraction":.25})["presolve_verified"])
