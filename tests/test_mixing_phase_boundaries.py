"""Dry-air and water balances at clear, ice, and liquid mixing boundaries."""
import math
import unittest

from mpmath import mp
from hamon_pde_control_psychro_fem import verify_psychrometric_process


def mix(first, second, model):
    return verify_psychrometric_process({"mode": "adiabatic_mixing", "condensate_model": model,
                                         "stream1": first, "stream2": second})


class MixingBoundaries(unittest.TestCase):
    def test_saturated_identical_inlets_are_idempotent(self):
        for temp in (-20., -10., .01, 20., 60.):
            for model in ("liquid", "equilibrium_ice"):
                for flow in (1., 7.):
                    row = {"t_db_c": temp, "rh": 1., "m_da_kg_s": flow}
                    with self.subTest(temp=temp, model=model, flow=flow):
                        out = mix(row, {**row, "m_da_kg_s": 1.}, model)
                        self.assertTrue(out["passed"], out)
                        self.assertEqual(out["t_db_mix_c"], temp)
                        self.assertFalse(out["fog_saturated"])
                        self.assertEqual(out["w_condensate_kg_per_kgda"], 0.)
                        self.assertEqual(out["rh_mix"], 1.)

    def test_melting_endpoints_from_total_enthalpy(self):
        ctx = mp.clone()
        ctx.dps = 110
        n = ctx.mpf
        # Evaluate the ASHRAE saturated-ice correlation at the melting point.
        # Inlet vapor ratios are supplied directly, independent of RH inversion.
        tk = n("273.15")
        coefficients = list(map(n, ("6.3925247", "-.0096778430", "6.2215701e-7",
                                    "2.0747825e-9", "-9.4840240e-13")))
        logp = -n("5674.5359")/tk + ctx.polyval(coefficients[::-1], tk) + n("4.1635019")*ctx.log(tk)
        pws = ctx.exp(logp)
        wz = n(".621945")*pws/(101325-pws)
        wc, wh = map(n, (.0002, .009))
        hc = -25*n("1.006") + wc*(2501-25*n("1.86"))
        hh = 15*n("1.006") + wh*(2501+15*n("1.86"))
        fusion = n("333.5")
        for condensed_h in (0, -fusion):
            # At either endpoint H_in = 2501*w_sat + (w_total-w_sat)*h_phase.
            cold_gap = 2501*wz+(wc-wz)*condensed_h-hc
            hot_gap = 2501*wz+(wh-wz)*condensed_h-hh
            ratio = float(-cold_gap/hot_gap)
            for mass in (math.nextafter(ratio, -math.inf), ratio, math.nextafter(ratio, math.inf)):
                with self.subTest(condensed_h=float(condensed_h), mass=mass):
                    m = n(mass)
                    w = (wc+m*wh)/(1+m)
                    h = (hc+m*hh)/(1+m)
                    ice = (2501*wz-h)/fusion
                    liquid = w-wz-ice
                    cold = {"t_db_c": -25, "w_kg_per_kgda": float(wc), "m_da_kg_s": 1}
                    warm = {"t_db_c": 15, "w_kg_per_kgda": float(wh), "m_da_kg_s": mass}
                    out = mix(cold, warm, "equilibrium_ice")
                    self.assertTrue(out["passed"], out)
                    self.assertTrue(out["fog_saturated"])
                    self.assertEqual(out["h_mix_kj_per_kgda"], float(h))
                    self.assertEqual(out["w_mix_kg_per_kgda"], float(w))
                    if min(ice, liquid) > 0:
                        self.assertEqual(out["t_db_mix_c"], 0.)
                        self.assertEqual(out["w_ice_kg_per_kgda"], float(ice))
                        self.assertEqual(out["w_liquid_kg_per_kgda"], float(liquid))
                    elif ice < 0:
                        self.assertGreater(out["t_db_mix_c"], 0.)
                        self.assertEqual(out["w_ice_kg_per_kgda"], 0.)
                    else:
                        self.assertLess(out["t_db_mix_c"], 0.)
                        self.assertEqual(out["w_liquid_kg_per_kgda"], 0.)

    def test_volume_flow_and_dry_air_flow_describe_same_clear_mixture(self):
        # Independently compute specific volume per kg dry air for two distinct streams.
        streams = []
        for temp, w, mass in ((12., .003, 3.), (24., .008, 1.)):
            v = 287.042*(273.15+temp)*(1+1.607858*w)/101325.
            streams.append(({"t_db_c": temp, "w_kg_per_kgda": w, "m_da_kg_s": mass},
                            {"t_db_c": temp, "w_kg_per_kgda": w, "v_dot_m3_s": mass*v}))
        given_mass = mix(streams[0][0], streams[1][0], "liquid")
        given_volume = mix(streams[0][1], streams[1][1], "liquid")
        for field in ("m_da_total_kg_s", "h_mix_kj_per_kgda", "w_mix_kg_per_kgda", "t_db_mix_c"):
            self.assertTrue(math.isclose(given_mass[field], given_volume[field], rel_tol=1e-14))

    def test_saturation_correlation_jump_cannot_supply_a_missing_equilibrium(self):
        # At +0.01 C, the two ASHRAE pressure correlations differ by about
        # 3.544e-6 Pa. This mass ratio puts inlet enthalpy inside that jump.
        cold = {"t_db_c": -25, "w_kg_per_kgda": .0002, "m_da_kg_s": 1}
        warm = {"t_db_c": 15, "w_kg_per_kgda": .009, "m_da_kg_s": 1.201564845347281}
        out = mix(cold, warm, "equilibrium_ice")
        self.assertFalse(out["passed"], out)
        self.assertIn("correlation phase gap", out["reason"])
