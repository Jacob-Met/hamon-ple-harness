import unittest
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def flow(inlets,outlets,**options):
    spec=dict(calculation="steady_flow_control_volume",property_reference="documented common basis",
              inlets=inlets,outlets=outlets,heat_reservoirs=[],shaft_work_out_W=0)
    return run_router_conservation_cad_verifier({"mode":"thermal_fluid","spec":dict(spec,**options)})


def stream(m,h,s,**extra):
    return dict(mdot_kg_s=m,h_J_kg=h,s_J_kgK=s,**extra)


class SteadyFlowThermodynamics(unittest.TestCase):
    def test_adiabatic_turbine_needs_nonnegative_entropy_and_first_law(self):
        inlet=[stream(1,1000000,4000)]
        valid=flow(inlet,[stream(1,800000,4000)],shaft_work_out_W=200000)
        self.assertTrue(valid["conservation_verified"],valid)
        for h,s,w in ((800000,3999,200000),(800000,4000,250000)):
            with self.subTest(h=h,s=s,w=w):
                r=flow(inlet,[stream(1,h,s)],shaft_work_out_W=w)
                self.assertFalse(r["conservation_verified"],r)

    def test_conflicting_property_basis_cannot_be_certified_by_mass_only(self):
        r=flow([stream(1,0,0)],[stream(1,0,0,property_reference="unrelated material table")])
        self.assertFalse(r["conservation_verified"],r)

    def test_reversible_device_and_multiple_reservoirs(self):
        # Two reservoirs transfer 600 W into the volume and 300 W out.
        # Their entropy flows cancel at 600 K and 300 K, respectively.
        reservoirs = [{"heat_into_W": 600, "temperature_K": 600},
                      {"heat_into_W": -300, "temperature_K": 300}]
        state = [stream(2, 40, 3)]
        result = flow(state, state, heat_reservoirs=reservoirs, shaft_work_out_W=300,
                      entropy_generation_W_K=0)
        self.assertTrue(result["thermodynamically_feasible"], result)
        self.assertEqual(result["entropy_heat_transfer_W_K"], 0)
        self.assertEqual(result["net_heat_into_W"], 300)
        result = flow(state, state, heat_reservoirs=reservoirs[::-1], shaft_work_out_W=300,
                      entropy_generation_W_K=-1e-25)
        self.assertFalse(result["conservation_verified"])
        self.assertTrue(result["second_law_ok"])
        self.assertFalse(result["entropy_generation_observation_ok"])

    def test_entropy_from_opposing_heat_flows_and_wrong_reservoir_direction(self):
        state = [stream(1, 10, 1)]
        for direction in (1, -1):
            result = flow(state, state, heat_reservoirs=[
                {"heat_into_W": 600 * direction, "temperature_K": 600},
                {"heat_into_W": -600 * direction, "temperature_K": 300}])
            self.assertTrue(result["first_law_ok"], result)
            self.assertEqual(result["net_heat_into_W"], 0)
            self.assertEqual(result["entropy_generation_W_K"], direction)
            self.assertEqual(result["conservation_verified"], direction > 0)

    def test_shaft_kinetic_and_elevation_terms_with_shifted_datums(self):
        for offset in (0, 10**100, -10**100):
            inlet = [stream(2, offset + 50, offset, speed_m_s=0, elevation_m=offset)]
            outlet = [stream(2, offset, offset + 1, speed_m_s=10, elevation_m=offset + 3)]
            result = flow(inlet, outlet, gravity_m_s2=10, shaft_work_out_W=-60,
                          entropy_generation_W_K=2)
            self.assertTrue(result["conservation_verified"], result)
            for field, value in (("enthalpy_flux_change_W", -100),
                                 ("kinetic_energy_flux_change_W", 100),
                                 ("potential_energy_flux_change_W", 60),
                                 ("energy_residual_W", 0), ("entropy_generation_W_K", 2)):
                self.assertEqual(result[field], value)
            wrong = flow(inlet, outlet, gravity_m_s2=10, shaft_work_out_W=60)
            self.assertEqual(wrong["energy_residual_W"], -120)
            self.assertFalse(wrong["conservation_verified"])

    def test_nominal_stream_mass_mixing_and_zero_flow(self):
        ins = [stream(.1, 100, 30), stream(.2, 400, 60), stream(0, -1e200, 1e200)]
        out = [stream(.3, 300, 51)]
        result = flow(ins, out, entropy_generation_W_K=.3)
        self.assertTrue(result["conservation_verified"], result)
        self.assertEqual(result["entropy_generation_W_K"], .3)
        result = flow(ins, [stream(.30000000000000004, 300, 51)])
        self.assertFalse(result["mass_balance_ok"])
        self.assertFalse(result["thermodynamically_feasible"])

    def test_bad_inputs_and_unrepresentable_outputs_refuse(self):
        state = [stream(1, 0, 0)]
        for options in ({"heat_reservoirs": [{"heat_into_W": 0, "temperature_K": 0}]},
                        {"heat_reservoirs": [{"heat_into_W": 0, "temperature_K": -3}]},
                        {"inlets": [stream(-1, 0, 0)]},
                        {"outlets": [stream(0, 0, 0)]},
                        {"inlets": [stream(1, 0, 0, speed_m_s=-2)]},
                        {"inlets": [stream(1, 0, 0, heat_W=3)]},
                        {"shaft_work_out_W": True}, {"property_reference": ""},
                        {"outlets": [stream(1, 0, 0, speed_m_s=1e300)]}):
            with self.subTest(options=options):
                defaults = dict(inlets=state, outlets=state)
                result = flow(**dict(defaults, **options))
                self.assertFalse(result["conservation_verified"])
                self.assertFalse(result["ok"])

    def test_auto_dispatch_and_existing_mass_only_remain_available(self):
        state = [stream(2, 10, 3)]
        spec = {"calculation": "steady_flow_control_volume", "inlets": state, "outlets": state,
                "property_reference": "water table A", "shaft_work_out_W": 0,
                "heat_reservoirs": []}
        result = run_router_conservation_cad_verifier({"spec": spec})
        self.assertTrue(result["second_law_ok"], result)
        bare = run_router_conservation_cad_verifier({"spec": {"inlets": [{"mdot_kg_s": 2}],
                                                             "outlets": [{"mdot_kg_s": 2}]}})
        self.assertTrue(bare["conservation_verified"], bare)
        self.assertNotIn("entropy_generation_W_K", bare)
