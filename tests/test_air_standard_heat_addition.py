import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def cycle(kind, **observations):
    spec = dict(subkind=kind, r=4., gamma=1.5, T1=300., cv=750., P1=100000.)
    return run_router_conservation_cad_verifier({"mode":"thermo_cycles",
                                                "spec":dict(spec,**observations)})


class AirStandardHeatAddition(unittest.TestCase):
    def test_otto_cannot_discard_co_supplied_constant_volume_heat(self):
        r = cycle("otto",T3=900.,q_in=90000.)
        self.assertFalse(r["verified"],r)
        self.assertTrue(cycle("otto",T3=900.,q_in=225000.)["verified"])

    def test_diesel_cannot_discard_temperature_or_constant_pressure_heat(self):
        for extra in (dict(T3=750.), dict(q_in=90000.), dict(T3=900.,q_in=-1.)):
            with self.subTest(extra=extra):
                r = cycle("diesel",r_c=1.5,**extra)
                self.assertFalse(r["verified"],r)
        self.assertTrue(cycle("diesel",r_c=1.5,T3=900.,q_in=337500.)["verified"])

    def test_each_single_coordinate_and_joint_cycle_agree(self):
        for kind,heat,variants in (("otto",225000.,[dict(T3=900.),dict(q_in=225000.)]),
                                  ("diesel",337500.,[dict(T3=900.),dict(q_in=337500.),dict(r_c=1.5)])):
            for observations in variants:
                r = cycle(kind, **observations)
                self.assertTrue(r["verified"],r)
                self.assertEqual(r["T2_K"],600.)
                self.assertEqual(r["T3_K"],900.)
                self.assertEqual(r["q_in_J_kg"],heat)
                self.assertEqual(r["heat_addition_delta_T_K"],300.)
                if kind == "diesel":
                    self.assertEqual(r["cutoff_ratio"],1.5)
                expected_t4=450. if kind=="otto" else 300.*1.5**1.5
                qout=750.*(expected_t4-300.)
                self.assertAlmostEqual(r["T4_K"],expected_t4)
                self.assertAlmostEqual(r["q_out_J_kg"],qout)
                self.assertAlmostEqual(r["w_net_J_kg"],heat-qout)

    def test_rounded_feasible_state_can_lie_between_all_centers(self):
        q = math.nextafter(math.nextafter(240000.,math.inf),math.inf)
        r=cycle("otto",cv=800.,T3=900.,q_in=q)
        self.assertTrue(r["verified"],r)
        self.assertEqual(r["T3_K"],900.)
        self.assertEqual(r["q_in_J_kg"],q)
        self.assertFalse(cycle("otto",cv=800.,T3=900.,q_in=math.nextafter(q,math.inf))["verified"])
        self.assertFalse(cycle("otto",cv=800.,T3=900,q_in=q)["verified"])

    def test_small_heat_has_its_own_coordinate_even_if_peak_rounds_to_compression(self):
        for kind in ("otto","diesel"):
            r=cycle(kind,T3=600.,q_in=2.**-200)
            self.assertTrue(r["verified"],r)
            self.assertEqual(r["T2_K"],r["T3_K"])
            self.assertEqual(r["q_in_J_kg"],2.**-200)
            self.assertGreater(r["heat_addition_delta_T_K"],0.)
            self.assertGreater(r["w_net_J_kg"],0.)

    def test_heat_and_diesel_expansion_domains(self):
        for kind, values in (("otto",dict(q_in=0.)),("otto",dict(T3=500.)),
                             ("diesel",dict(r_c=1.)),("diesel",dict(r_c=4.)),
                             ("diesel",dict(T3=2500.)),("diesel",dict(q_in=-1.)),
                             ("otto",dict(T3=900.,r_c=1.5))):
            self.assertFalse(cycle(kind,**values)["verified"])
