"""Lossless standing-wave nodes and finite nearby port states."""
import json
import math
import unittest
from mpmath import mp
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


class TransmissionPortStates(unittest.TestCase):
    def solve(self, load_x, **kwargs):
        spec = dict(subkind="transmission_line", Z0=50, R_L=0, X_L=load_x, **kwargs)
        result = run_router_conservation_cad_verifier(dict(mode="rf_transmission_rlc", spec=spec))
        self.assertTrue(result["verified"], result)
        json.dumps(result, allow_nan=False)
        return result

    def test_exact_voltage_and_current_nodes(self):
        for turns, reactance, kind in [(.25, 0, "open"), (.5, 0, "short"),
                                      (.125, 50, "open"), (.125, -50, "short"),
                                      (.375, 50, "short"), (.375, -50, "open")]:
            for offset in (0., 8., -1.):
                with self.subTest(turns=turns, reactance=reactance, offset=offset):
                    r = self.solve(reactance, length_wavelengths=turns+offset)
                    self.assertEqual(r["Z_in_kind"], kind)
                    self.assertEqual((r["Z_in_real"], r["Z_in_imag"]),
                                     (None, None) if kind=="open" else (0., 0.))
                    self.assertEqual((r["Y_in_real"], r["Y_in_imag"]),
                                     (0., 0.) if kind=="open" else (None, None))
                    self.assertIsNone(r["VSWR"])
                    self.assertEqual(r["VSWR_status"], "unbounded")

    def test_finite_neighbors_loss_and_direct_radian_input(self):
        ctx = mp.clone()
        ctx.dps = 160
        for node, reactance in [(.125, 50), (.125, -50), (.25, 0)]:
            for turns, loss, radians in [(math.nextafter(node, 0.), 0., False),
                                         (math.nextafter(node, 1.), 0., False),
                                         (node, 1e-55, False), (node, .04, False),
                                         (node, 0., True)]:
                with self.subTest(node=node, reactance=reactance, turns=turns, loss=loss, radians=radians):
                    p = {"length_wavelengths": turns} if not radians else {"beta_rad_m": math.tau*turns}
                    r = self.solve(reactance, alpha_np_m=loss, length_m=1., **p)
                    self.assertEqual(r["Z_in_kind"], "finite")
                    # Offset d from a known node. At a current node, Z=Z0 coth(d);
                    # a voltage node gives Z=Z0 tanh(d). This also distinguishes
                    # binary radian input from an exact fractional wavelength.
                    angle = ctx.mpf(p["beta_rad_m"])-2*ctx.pi*node if radians else 2*ctx.pi*(ctx.mpf(turns)-node)
                    d = ctx.mpc(loss, angle)
                    z = 50/ctx.tanh(d) if reactance >= 0 else 50*ctx.tanh(d)
                    for key, ref in [("Z_in", z), ("Y_in", 1/z)]:
                        self.assertEqual(r[key+"_real"], float(ref.real))
                        self.assertEqual(r[key+"_imag"], float(ref.imag))

    def test_finite_port_when_only_reciprocal_chart_fits(self):
        for load in (1e-307, 1e-310):
            r = run_router_conservation_cad_verifier(dict(mode="rf_transmission_rlc", spec=dict(
                subkind="transmission_line", Z0=50, R_L=load, length_wavelengths=.25)))
            self.assertTrue(r["verified"],r)
            self.assertEqual(r["Z_in_kind"], "finite")
            self.assertEqual((r["Z_in_real"], r["Z_in_imag"]), (None, None))
            self.assertEqual((r["Y_in_real"], r["Y_in_imag"]), (load/2500., 0.))
            self.assertEqual(r["VSWR_status"], "outside_numeric_range")
            json.dumps(r, allow_nan=False)

    def test_purely_reactive_reflection_has_exact_zero_return_loss(self):
        for impedance, reactance in [(50,19), (37,1), (50,-19)]:
            r = run_router_conservation_cad_verifier(dict(mode="rf_transmission_rlc", spec=dict(
                subkind="transmission_line", Z0=impedance, R_L=0, X_L=reactance, length_wavelengths=.17)))
            self.assertTrue(r["verified"],r)
            self.assertEqual(r["VSWR_status"], "unbounded")
            self.assertEqual(r["gamma_load_mag"], 1.)
            self.assertEqual(r["return_loss_dB"], 0.)

    def test_reflection_and_admittance_of_matched_line(self):
        spec = dict(subkind="transmission_line", Z0=50, R_L=50, alpha_np_m=.01,
                    length_m=1, length_wavelengths=.317)
        r = run_router_conservation_cad_verifier(dict(mode="rf_transmission_rlc", spec=spec))
        self.assertTrue(r["verified"],r)
        self.assertEqual(r["Z_in_kind"], "finite")
        self.assertEqual((r["Z_in_real"], r["Z_in_imag"]), (50., 0.))
        self.assertEqual((r["Y_in_real"], r["Y_in_imag"]), (.02, 0.))
        self.assertEqual(r["VSWR"], 1.)
        self.assertEqual(r["return_loss_status"], "infinite")
        self.assertIsNone(r["return_loss_dB"])
