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

    def test_initial_state_asymptotic_limit_and_zero_bus(self):
        for f, status, time, voltage, current, heat in [
            (0., "already_at_target", 0., 0., 8., 0.),
            (1., "asymptotic", None, 400., 0., 160.),
        ]:
            r = precharge(f)
            self.assertTrue(r["dual_verified"], r)
            self.assertEqual((r["target_time_status"],r["t_target_s"]),(status,time))
            self.assertEqual((r["target_capacitor_voltage_V"],r["target_current_A"],r["resistor_heat_J"]),
                             (voltage,current,heat))
        for f in (0.,.65,1.):
            r = precharge(f,V_bus_V=0.)
            self.assertTrue(r["dual_verified"],r)
            self.assertEqual((r["target_time_status"],r["t_target_s"]),("already_at_target",0.))
            self.assertEqual((r["source_work_J"],r["resistor_heat_J"],r["target_capacitor_energy_J"]),(0.,0.,0.))

    def test_original_passive_ode_and_work_at_signed_or_weak_targets(self):
        from mpmath import mp
        ctx = mp.clone()
        ctx.dps = 100
        for vbus in (-24.,75.):
            for f in (2.**-480,.125,.8125,math.nextafter(1.,0.)):
                with self.subTest(vbus=vbus,f=f):
                    r = precharge(f,V_bus_V=vbus,C_uF=1250.,R_precharge_ohm=17.)
                    self.assertTrue(r["dual_verified"],r)
                    self.assertEqual(r["target_time_status"],"finite")
                    capacitance = ctx.mpf(1250)/1_000_000
                    tau = 17*capacitance
                    time = ctx.mpf(r["t_target_s"])
                    elapsed = time/tau
                    v = -vbus*ctx.expm1(-elapsed)
                    i = vbus*ctx.exp(-elapsed)/17
                    # Integrate the physical source power and resistor heating,
                    # using normalized time to retain accuracy at tiny fractions.
                    heat = ctx.quad(lambda s: (vbus**2/17)*ctx.exp(-2*s),[0,elapsed])*tau
                    work = ctx.quad(lambda s: (vbus**2/17)*ctx.exp(-s),[0,elapsed])*tau
                    for key,expected in (("target_capacitor_voltage_V",v),("target_current_A",i),
                            ("target_capacitor_charge_C",capacitance*v),("source_work_J",work),
                            ("resistor_heat_J",heat),("target_capacitor_energy_J",capacitance*v*v/2)):
                        self.assertLess(abs(ctx.mpf(r[key])/expected-1),ctx.mpf("1e-13"))
                    self.assertAlmostEqual(r["t_target_s"],float(-tau*ctx.log1p(-ctx.mpf(f))))

    def test_passive_parameter_and_fraction_domain(self):
        for changes in [dict(C_uF=0),dict(R_precharge_ohm=0),dict(target_fraction=-.1),
                        dict(target_fraction=1.1),dict(V_bus_V=float("inf"))]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                precharge(**changes)
