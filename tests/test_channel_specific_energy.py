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
