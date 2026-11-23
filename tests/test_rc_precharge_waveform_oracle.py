"""Independent signed RC endpoint inventory from the original current waveform."""
import math
import unittest
from decimal import Decimal as D, localcontext

from hamon_dual_verifier import run_dual_verify_spec


def route(bus, c_micro, resistance, fraction):
    return run_dual_verify_spec({"mode":"domain","kind":"rc_precharge","spec":{
        "V_bus_V":bus,"C_uF":c_micro,"R_precharge_ohm":resistance,"target_fraction":fraction}})


class PassiveRcWaveformOracle(unittest.TestCase):
    def test_signed_energy_charge_and_unreached_charge_from_current(self):
        with localcontext() as ctx:
            ctx.prec = 180
            for bus,cap,resistance in ((-37.5,125.,16.), (7.25,3200000.,.125)):
                v,c,r = map(D.from_float, (bus,cap,resistance))
                c /= 10**6
                tau = r*c
                for f in (0.,2.**-200,.125,.5,.875,math.nextafter(1.,0.),1.):
                    with self.subTest(bus=bus,f=f):
                        out = route(bus,cap,resistance,f)
                        self.assertTrue(out["dual_verified"],out)
                        f_exact = D.from_float(f)
                        if f == 1:
                            self.assertIsNone(out["t_target_s"])
                            self.assertEqual(out["target_time_status"],"asymptotic")
                            residual_fraction = D(0)
                        else:
                            t = D.from_float(out["t_target_s"])
                            residual_fraction = (-t/tau).exp()
                            time_ref = -tau*(1-f_exact).ln()
                            self.assertEqual(out["t_target_s"],float(time_ref))
                        # I(t)=V/R exp(-t/RC). Integrate signed current and
                        # Ohmic power across the finite endpoint/limit.
                        current = v/r*residual_fraction
                        q_remaining = tau*current
                        q_reached = c*v*(1-residual_fraction)
                        source = v*q_reached
                        heat = (v*v/r)*(tau/2)*(1-residual_fraction**2)
                        cap_energy = q_reached*q_reached/(2*c)
                        checks = ((D.from_float(out["target_current_A"]),current),
                                  (c*D.from_float(out["target_resistor_voltage_V"]),q_remaining),
                                  (D.from_float(out["target_capacitor_charge_C"]),q_reached),
                                  (c*D.from_float(out["target_capacitor_voltage_V"]),q_reached),
                                  (D.from_float(out["source_work_J"]),source),
                                  (D.from_float(out["resistor_heat_J"]),heat),
                                  (D.from_float(out["target_capacitor_energy_J"]),cap_energy),
                                  (D.from_float(out["stored_energy_J"]),c*v*v/2))
                        for actual, expected in checks:
                            if expected:
                                self.assertLess(abs(actual/expected-1),D("4e-14"))
                            else:
                                self.assertEqual(actual,0)

    def test_full_capacity_energy_representation_at_zero_charge(self):
        for sign in (-1,1):
            for capacitance, accepted in ((2_000_000,True),(3_000_000,False)):
                with self.subTest(sign=sign,capacitance=capacitance):
                    try:
                        out = route(sign*2.**-537,capacitance,1.,0.)
                    except ValueError:
                        self.assertFalse(accepted)
                    else:
                        self.assertIs(out["dual_verified"],accepted)
                        if accepted:
                            self.assertEqual(out["stored_energy_J"],math.ulp(0.))
