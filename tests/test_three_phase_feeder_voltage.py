"""Full terminal-voltage phasors for the balanced, short series feeder."""
import math
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def feeder(**changes):
    spec = dict(V_ll_V=400., I_A=100., pf=1., length_m=1000.,
                R_ohm_per_km=0., X_ohm_per_km=1.)
    spec.update(changes)
    return run_dual_verify_spec(dict(mode="domain", kind="three_phase_vdrop", spec=spec))


class ThreePhaseFeederVoltage(unittest.TestCase):
    def test_sending_reference_retains_pure_quadrature_drop(self):
        # V_s² = V_r² + 3 (I X)² for load current in phase with V_r.
        out = feeder()
        self.assertFalse(out["passes_3pct"])
        self.assertAlmostEqual(out["vdrop_ll_V"], 400. - math.sqrt(130000.))
        self.assertAlmostEqual(out["V_receiving_ll_V"], math.sqrt(130000.))

    def test_receiving_reference_retains_pure_quadrature_drop(self):
        out = feeder(voltage_reference="receiving")
        self.assertFalse(out["passes_3pct"])
        self.assertAlmostEqual(out["vdrop_ll_V"], math.sqrt(190000.) - 400.)
        self.assertAlmostEqual(out["V_sending_ll_V"], math.sqrt(190000.))

    def test_two_positive_branches_and_sign_of_sending_phasor(self):
        spec = dict(V_ll_V=1., I_A=1., X_ohm_per_km=2., pf=0., power_factor_direction="leading")
        lo, hi = 2*math.sqrt(3)-1, 2*math.sqrt(3)+1
        for branch, voltage, sending_real in (("low",lo,-1.),("high",hi,1.)):
            out = feeder(**spec,receiving_voltage_branch=branch)
            self.assertTrue(out["dual_verified"])
            self.assertEqual(out["selected_receiving_branch"],branch)
            self.assertEqual(out["V_sending_phasor_ll_V"],[sending_real,0.])
            self.assertAlmostEqual(out["V_receiving_ll_V"],voltage)
            self.assertAlmostEqual(out["vdrop_ll_V"],1-voltage)
            for actual, expected in zip(out["positive_receiving_voltages_ll_V"],(lo,hi)):
                self.assertAlmostEqual(actual,expected)

    def test_positive_voltage_and_real_root_feasibility(self):
        for spec,state in [(dict(V_ll_V=100.),"no_real_receiving_voltage"),
                           (dict(V_ll_V=3.,I_A=1.,R_ohm_per_km=2.,X_ohm_per_km=0.,pf=.5),
                            "no_positive_receiving_voltage")]:
            out = feeder(**spec)
            self.assertEqual(out["state"],state)
            self.assertFalse(out["feasible"])
            self.assertFalse(out["passes_3pct"])
        # The leading load gives a negative longitudinal term and a positive
        # tangent voltage. These binary64 values straddle the irrational fold.
        for v,possible in ((math.sqrt(3.),False),(math.nextafter(math.sqrt(3.),math.inf),True)):
            out = feeder(V_ll_V=v,I_A=1.,pf=0.,R_ohm_per_km=1.,X_ohm_per_km=2.,
                         power_factor_direction="leading")
            self.assertEqual(out["feasible"],possible)
            if possible:
                self.assertEqual(len(out["positive_receiving_voltages_ll_V"]),2)

    def test_signed_cancellation_and_limit_neighbors(self):
        # At receiving voltage 2, I=3, X=1, PF=.5 lagging, sending magnitude
        # is exactly 7. The exact percent change is 250.
        for limit,passed in ((math.nextafter(250.,0.),False),(250.,True),
                             (math.nextafter(250.,math.inf),True)):
            out = feeder(voltage_reference="receiving",V_ll_V=2.,I_A=3.,pf=.5,max_vdrop_pct=limit)
            self.assertEqual(out["V_sending_ll_V"],7.)
            self.assertEqual(out["passes_3pct"],passed)
        # The alternate leading solution has exactly zero change at I=V.
        for i,expected_sign in ((math.nextafter(1.,0.),1),(1.,0),(math.nextafter(1.,math.inf),-1)):
            out = feeder(V_ll_V=1.,I_A=i,pf=.5,power_factor_direction="leading",
                         receiving_voltage_branch="low",max_vdrop_pct=0.)
            v = out["vdrop_ll_V"]
            self.assertEqual((v>0)-(v<0),expected_sign)
            self.assertEqual(out["passes_3pct"],expected_sign<=0)

    def test_quadrature_headroom_smaller_than_one_terminal_voltage_ulp(self):
        for ref in ("sending","receiving"):
            out = feeder(voltage_reference=ref,V_ll_V=1.,I_A=2.**-80,max_vdrop_pct=0.)
            self.assertEqual(out["V_sending_ll_V"],out["V_receiving_ll_V"])
            self.assertEqual(out["vdrop_ll_V"],1.5*2.**-160)
            self.assertFalse(out["passes_3pct"])

    def test_zero_load_resistive_line_and_invalid_inputs(self):
        for ref in ("receiving","sending"):
            out = feeder(voltage_reference=ref,I_A=0.)
            self.assertEqual(out["vdrop_ll_V"],0.)
            self.assertTrue(out["passes_3pct"])
        out = feeder(V_ll_V=400.,I_A=2.,R_ohm_per_km=1.,X_ohm_per_km=0.)
        self.assertAlmostEqual(out["vdrop_ll_V"],2*math.sqrt(3))
        for spec in (dict(I_A=-1),dict(pf=1.1),dict(R_ohm_per_km=-1),
                     dict(voltage_reference="both"),dict(power_factor_direction="unknown"),
                     dict(voltage_reference="receiving",receiving_voltage_branch="low")):
            with self.assertRaises(ValueError):
                feeder(**spec)

    def test_each_reported_voltage_has_resolved_precision(self):
        # sqrt(3) times the minimum subnormal rounds to two minsubs, a 15%
        # error. A one-volt KVL residual cannot validate this subnormal
        # observable independently.
        with self.assertRaisesRegex(ValueError, "resolution"):
            feeder(voltage_reference="receiving",V_ll_V=1.,I_A=1.,
                   R_ohm_per_km=math.ulp(0.),X_ohm_per_km=0.)
        # A nearly pure six-minsub drop is representable to the usual relative
        # precision even though its terminal voltages round to the same number.
        for ref in ("sending","receiving"):
            r = feeder(voltage_reference=ref,V_ll_V=1.,I_A=1.,X_ohm_per_km=2.**-536)
            self.assertTrue(r["dual_verified"])
            self.assertEqual(r["vdrop_ll_V"],6*math.ulp(0.))
