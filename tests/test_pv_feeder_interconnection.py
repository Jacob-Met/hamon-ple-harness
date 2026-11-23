import cmath
import math
import unittest

from hamon_ac_power_flow import verify_ac_power_flow
from hamon_dual_verifier import solve_domain_problem


def feeder(**changes):
    values = dict(V_ll_V=480., P_export_W=80000., pf_absorbing=.8, R_feeder_ohm=.36,
                  X_feeder_ohm=.48, transformer_rating_kVA=125., max_vrise_pu=.02)
    return solve_domain_problem("pv_interconnection", dict(values, **changes))


class PVFeeder(unittest.TestCase):
    def test_resistive_and_reactive_terms_cancel_only_to_first_order(self):
        r = feeder()
        self.assertTrue(r["dual_verified"] and r["power_flow_feasible"])
        self.assertLess(abs(r["linearized_voltage_rise_pu"]), 1e-15)
        self.assertAlmostEqual(r["voltage_rise_pu"], -.0372803145888827)
        self.assertFalse(r["voltage_ok"] or r["thermal_and_voltage_ok"])
        self.assertTrue(r["thermal_ok"])
        # The exported 100 kVA is below 102 kVA, but actual feeder current
        # exceeds the transformer nameplate current at 480 V.
        thermal = feeder(transformer_rating_kVA=102.,max_vrise_pu=.05)
        self.assertTrue(thermal["voltage_ok"])
        self.assertFalse(thermal["thermal_ok"] or thermal["thermal_and_voltage_ok"])

    def test_independent_nodal_solve_and_series_power(self):
        for p, pf, resistance, reactance in [
            (80000., .8, .36, .48), (-32000.,.92,.12,.2),
            (41000.,1.,.43,0.), (37000.,.96,.03,-.19),
        ]:
            r = feeder(P_export_W=p,pf_absorbing=pf,R_feeder_ohm=resistance,X_feeder_ohm=reactance)
            scale = 100000.
            q = -p*math.sqrt((1-pf)*(1+pf))/pf
            zb = 480**2/scale
            independent = verify_ac_power_flow({
                "buses":[{"id":"grid","type":"slack","vm_pu":1,"va_deg":0},
                         {"id":"pcc","type":"PQ","p_injection_pu":p/scale,"q_injection_pu":q/scale}],
                "lines":[{"id":"feeder","from":"pcc","to":"grid","r_pu":resistance/zb,"x_pu":reactance/zb}],
            })
            self.assertTrue(independent["conservation_verified"], independent)
            v = independent["voltages"]["pcc"]
            u = cmath.rect(v["vm_pu"], math.radians(v["va_deg"]))
            self.assertAlmostEqual(complex(*r["pcc_voltage_pu"]),u,places=10)
            current = complex(*r["line_current_A"])
            z = complex(resistance, reactance)
            self.assertAlmostEqual(480/math.sqrt(3)*(u-1), z*current, places=8)
            self.assertAlmostEqual(complex(p,q),
                                   complex(r["grid_received_W"],r["grid_received_var"])+3*z*abs(current)**2,
                                   places=8)
            pin = independent["lines"]["feeder"]
            self.assertAlmostEqual(r["grid_received_W"], -pin["p_to_pu"]*scale, places=6)
            self.assertAlmostEqual(r["feeder_loss_W"], 3*resistance*abs(current)**2, places=9)

    def test_two_real_roots_coalesce_and_then_disappear(self):
        # With V_LL=100 V, P=10 kW, pf=1 and R=0 the imaginary voltage
        # equals X in per unit. The two real parts coalesce at X=1/2.
        for x, count in ((math.nextafter(.5,0.),2),(.5,1),(math.nextafter(.5,math.inf),0)):
            r = feeder(V_ll_V=100.,P_export_W=10000.,pf_absorbing=1.,R_feeder_ohm=0.,X_feeder_ohm=x)
            self.assertEqual(r["voltage_root_count"],count)
            self.assertEqual(r["power_flow_feasible"], bool(count))
            if count:
                self.assertTrue(r["dual_verified"])
                self.assertGreaterEqual(r["pcc_voltage_pu"][0],.5)
            else:
                self.assertFalse(r["thermal_and_voltage_ok"])
                self.assertNotIn("pcc_voltage_pu",r)

    def test_no_load_zero_impedance_and_exact_limit_contact(self):
        zero = feeder(P_export_W=0.)
        self.assertEqual((zero["pcc_voltage_pu"],zero["line_current_A"]), ([1,0],[0,0]))
        self.assertEqual(zero["voltage_root_count"],1)
        line = feeder(R_feeder_ohm=0.,X_feeder_ohm=0.,transformer_rating_kVA=100.,max_vrise_pu=0.)
        self.assertEqual(line["voltage_rise_pu"],0)
        self.assertTrue(line["thermal_and_voltage_ok"])
        # u=1.25, so a=u*(u-1)=5/16. Signed rise and rating are exact contacts.
        for change, ok in ((math.nextafter(.25,0.),False),(.25,True),(math.nextafter(.25,1),True)):
            r=feeder(V_ll_V=100.,P_export_W=10000.,pf_absorbing=1.,R_feeder_ohm=5/16,
                     X_feeder_ohm=0.,transformer_rating_kVA=8.,max_vrise_pu=change)
            self.assertEqual(r["voltage_rise_pu"],.25)
            self.assertEqual(r["thermal_and_voltage_ok"],ok)
        r=feeder(V_ll_V=100.,P_export_W=10000.,pf_absorbing=1.,R_feeder_ohm=5/16,
                 X_feeder_ohm=0.,transformer_rating_kVA=math.nextafter(8.,0.),max_vrise_pu=.25)
        self.assertFalse(r["thermal_ok"])

    def test_invalid_inputs_refuse(self):
        for bad in ({"pf_absorbing":0},{"pf_absorbing":1.01},{"V_ll_V":0},
                    {"R_feeder_ohm":-1},{"max_vrise_pu":-.1},{"P_export_W":float("inf")}):
            with self.assertRaises(ValueError): feeder(**bad)
