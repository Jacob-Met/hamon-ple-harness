import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def alternate(energy, q=2., width=1., slope=0., gravity=4., mode="civil_geotech"):
    return run_router_conservation_cad_verifier({"mode": mode, "spec": {
        "calculation": "channel_specific_energy", "bottom_width_m": width, "side_slope_z": slope,
        "Q_m3_s": q, "specific_energy_m": energy, "gravity_m_s2": gravity}})


class ChannelSpecificEnergy(unittest.TestCase):
    def test_two_depths_at_equal_energy_and_different_momentum(self):
        r = alternate(2., q=4., gravity=8.)
        self.assertTrue(r["conservation_verified"], r)
        self.assertEqual(r["state"], "two_depths")
        a,b = r["depth_solutions"]
        self.assertEqual([a["regime"],b["regime"]], ["supercritical","subcritical"])
        self.assertEqual(a["depth_m"], 1.)
        self.assertAlmostEqual(b["depth_m"], (1+math.sqrt(5))/2)
        for p in (a,b):
            self.assertAlmostEqual(p["depth_m"]+p["velocity_m_s"]**2/16, 2.)
        def momentum(y):
            return 16/(8*y)+y*y/2
        self.assertGreater(abs(momentum(a["depth_m"])-momentum(b["depth_m"])), .04)

    def test_critical_coalescence_and_adjacent_energies(self):
        for energy, count, label in ((math.nextafter(1.5,0.),0,"below_minimum"),
                                     (1.5,1,"critical"),(math.nextafter(1.5,math.inf),2,"two_depths")):
            with self.subTest(energy=energy):
                r = alternate(energy)
                self.assertTrue(r["ok"], r)
                self.assertEqual(r["state"], label)
                self.assertEqual(len(r["depth_solutions"]), count)
                if count == 1:
                    self.assertEqual(r["depth_solutions"][0]["depth_m"], 1.)
                    self.assertEqual(r["depth_solutions"][0]["regime"], "critical")

    def test_trapezoid_two_known_branches_and_reversed_discharge(self):
        for q in (-3., 3.):
            r = alternate(2.125, q=q, width=1., slope=1., gravity=1., mode="auto")
            self.assertTrue(r["conservation_verified"], r)
            for p,y in zip(r["depth_solutions"], (1.,2.)):
                self.assertEqual(p["depth_m"],y)
                self.assertEqual(p["area_m2"],(1+y)*y)
                self.assertEqual(p["velocity_m_s"],q/p["area_m2"])
                self.assertEqual(p["velocity_head_m"],2.125-y)
                fr2 = q*q*(1+2*y)/p["area_m2"]**3
                self.assertAlmostEqual(p["froude"],math.sqrt(fr2))
                self.assertAlmostEqual(p["froude_squared_minus_one"],fr2-1)

    def test_triangular_and_trapezoidal_critical_contacts(self):
        for width, slope, q, g, e in ((0.,1.,1.,2.,1.25),(2.,1.,4.5,3.,1.375)):
            for value,state in ((math.nextafter(e,0.),"below_minimum"),(e,"critical"),
                                (math.nextafter(e,math.inf),"two_depths")):
                r = alternate(value,q,width,slope,g)
                self.assertTrue(r["ok"], r)
                self.assertEqual(r["state"],state)
                if state == "critical":
                    p, = r["depth_solutions"]
                    self.assertEqual(p["depth_m"],1.)
                    self.assertEqual(p["froude_squared_minus_one"],0.)

    def test_complementary_kinetic_head_and_zero_discharge(self):
        tiny = 2.**-70
        r = alternate(2.,q=tiny,gravity=1.)
        self.assertTrue(r["conservation_verified"], r)
        deep = r["depth_solutions"][1]
        self.assertEqual(deep["depth_m"],2.)
        self.assertEqual(deep["velocity_head_m"],tiny**2/8)
        stagnant = alternate(2.,q=0.)
        self.assertEqual(stagnant["state"],"stagnant")
        self.assertEqual(stagnant["depth_solutions"][0]["velocity_m_s"],0.)
        for spec in ((2.,1.,0.,0.,4.),(-2.,1.,1.,0.,4.),(2.,1.,1.,-1.,4.)):
            self.assertFalse(alternate(*spec)["ok"])
