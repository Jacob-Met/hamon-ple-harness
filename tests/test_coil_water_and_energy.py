import unittest

from hamon_dual_verifier import solve_domain_problem


def coil(**changes):
    spec = dict(m_dry_air_kg_s=1., T_in_C=30., T_out_C=15.,
                w_in_kg_kg=.006, w_out_kg_kg=.010,
                duct_width_m=.5, duct_height_m=.4, duct_length_m=2.)
    return solve_domain_problem("hvac_coil_and_duct", dict(spec, **changes))


class CoilInventoryTests(unittest.TestCase):
    def test_air_acquires_water_without_any_water_input(self):
        result = coil()
        self.assertFalse(result["dual_verified"])
        self.assertFalse(result["coil_water_balance_ok"])
        self.assertAlmostEqual(result["coil_water_residual_kg_s"], .004)
        self.assertAlmostEqual(result["Q_total_cooling_W"], 5141.8)

    def test_wet_coil_air_and_drain_energy_are_distinct(self):
        # Enthalpy fluxes, calculated independently for the two streams.
        m, wi, wo, ti, to = 2., .013, .007, 28., 12.
        h_in = m * (1006*ti + wi*(2501000 + 1860*ti))
        h_out = m * (1006*to + wo*(2501000 + 1860*to))
        kw = dict(m_dry_air_kg_s=m, w_in_kg_kg=wi, w_out_kg_kg=wo, T_in_C=ti, T_out_C=to)
        air_only = coil(**kw)
        self.assertTrue(air_only["dual_verified"])
        self.assertNotIn("Q_coil_heat_removed_W", air_only)
        for enthalpy in (0., 10.*4186., -334000.):
            with self.subTest(enthalpy=enthalpy):
                r = coil(**kw, condensate_enthalpy_J_kg=enthalpy)
                self.assertTrue(r["dual_verified"])
                self.assertAlmostEqual(r["condensate_kg_s"], m*(wi-wo))
                self.assertAlmostEqual(r["Q_total_cooling_W"], h_in-h_out)
                self.assertAlmostEqual(r["Q_total_cooling_W"], r["Q_sensible_W"]+r["Q_vapor_transfer_W"])
                self.assertAlmostEqual(r["Q_coil_heat_removed_W"], h_in-h_out-m*(wi-wo)*enthalpy)
                self.assertTrue(r["coil_total_energy_balance_ok"])
                self.assertEqual(r["duct_pressure_drop_Pa"], air_only["duct_pressure_drop_Pa"])

    def test_dry_idle_and_infeasible_energy_controls(self):
        for changes in (dict(w_in_kg_kg=.008, w_out_kg_kg=.008),
                        dict(m_dry_air_kg_s=0.), dict(T_in_C=15., w_in_kg_kg=.010)):
            r = coil(**changes)
            self.assertTrue(r["dual_verified"])
            self.assertEqual(r["Q_coil_heat_removed_W"], r["Q_total_cooling_W"])
        heating = coil(T_in_C=10., T_out_C=25., w_out_kg_kg=.006)
        self.assertTrue(heating["coil_water_balance_ok"])
        self.assertFalse(heating["coil_air_energy_balance_ok"])
        self.assertFalse(heating["dual_verified"])
        r = coil(w_in_kg_kg=.012, condensate_enthalpy_J_kg=1e9)
        self.assertTrue(r["coil_air_energy_balance_ok"])
        self.assertFalse(r["coil_total_energy_balance_ok"])
        self.assertFalse(r["dual_verified"])

    def test_thermodynamic_state_inputs_must_be_finite_and_physical(self):
        for options in (dict(w_in_kg_kg=-.001), dict(T_out_C=-273.15),
                        dict(w_out_kg_kg=float('nan')), dict(condensate_enthalpy_J_kg=float("inf"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                coil(**options)
