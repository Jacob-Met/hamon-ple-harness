import cmath
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def route(deck, **extra):
    return run_router_conservation_cad_verifier({"mode": "circuit", "spec": {"netlist": deck, **extra}})


class CurrentControlledSources(unittest.TestCase):
    def test_one_sensor_supplies_both_outputs_and_power(self):
        deck = ("Fcopy 0 out vSense 3\nHtrans h 0 VSENSE 500\nVbias in 0 DC 2\n"
                "Rin in s 1k\nVsense s 0 DC 0\nRload out 0 1k\nRout h 0 1k")
        titled = route("First circuit uses voltage feedback\n" + deck)
        self.assertTrue(titled["conservation_verified"])
        r = route(deck, proposed_node_voltages={"out": 6, "h": 1})
        self.assertTrue(r["conservation_verified"])
        self.assertEqual(r["node_voltages"]["out"], 6)
        self.assertEqual(r["node_voltages"]["h"], 1)
        currents = {br["name"]: br["current_A"] for br in r["branch_currents"]}
        self.assertAlmostEqual(currents["Vsense"], .002)
        self.assertAlmostEqual(currents["Fcopy"], .006)
        self.assertAlmostEqual(currents["Htrans"], -.001)
        self.assertAlmostEqual(r["total_complex_power_delivered_VA"][0], .004+.036+.001)
        self.assertFalse(route(deck, proposed_node_voltages={"out": 0, "h": 0})["conservation_verified"])

    def test_feedback_senses_the_solved_current_and_replays_terminal_power(self):
        # The CCCS returns a fraction of the sensed current to ground. Loading
        # and feedback jointly determine the sensed current, including its phase.
        for k in (-.7, .4):
            for direction in (-1, 1):
                sensor = "s out" if direction == 1 else "out s"
                deck = (f"Hread h 0 vsense -230\nVdrive src 0 AC 2 73\nRin src s 101\n"
                        f"Vsense {sensor} 0\nFfb out 0 vSENSE {k}\nRo out 0 77\n"
                        "Co out 0 3u\nRh h 0 210")
                w = math.tau * 780
                source = cmath.rect(2, math.radians(73))
                y = 1/77+1j*w*3e-6
                out = (1-k*direction)*source / (1-k*direction+101*y)
                current = direction*(source-out)/101
                h = -230*current
                power = abs(current)**2*101 + abs(out)**2*y.conjugate() + abs(h)**2/210
                r = route(deck, freq_hz=780)
                self.assertTrue(r["conservation_verified"])
                for node, expected in (("out", out), ("h", h)):
                    self.assertAlmostEqual(complex(*r["node_voltages"][node]), expected)
                currents = {br["name"]: complex(*br["current_A"]) for br in r["branch_currents"]}
                self.assertAlmostEqual(currents["Vsense"], current)
                self.assertAlmostEqual(currents["Ffb"], k*current)
                self.assertAlmostEqual(currents["Hread"], -h/210)
                self.assertAlmostEqual(complex(*r["total_complex_power_delivered_VA"]), power)

    def test_g_source_delivers_and_absorbs_power(self):
        for deck, watts in [
            ("Transconductance\nVbias in 0 2\nGsrc 0 out in 0 1m\nRo out 0 1k", .004),
            ("Absorbing source\nVbias in 0 2\nVhold out 0 3\nGsink out 0 in 0 1m\nRo out 0 1k", .009),
        ]:
            r = route(deck)
            self.assertTrue(r["conservation_verified"])
            self.assertAlmostEqual(r["total_complex_power_delivered_VA"][0], watts)

    def test_structured_control_against_nonzero_voltage_source(self):
        branches = [
            {"name":"F1","type":"F","n_pos":"out","n_neg":"0","control_source":"drive","value":2},
            {"name":"Drive","type":"V","n_pos":"in","n_neg":"0","value":3},
            {"name":"R1","type":"R","n_pos":"in","n_neg":"0","value":50},
            {"name":"R2","type":"R","n_pos":"out","n_neg":"0","value":10},
        ]
        r=run_router_conservation_cad_verifier({"mode":"circuit","spec":{"branches":branches}})
        self.assertTrue(r["conservation_verified"])
        self.assertAlmostEqual(r["node_voltages"]["out"], 1.2)

    def test_unknown_ambiguous_and_behavioral_controls_refuse(self):
        base="Sensor\nV1 in 0 2\nRin in 0 50\nRl out 0 10\n"
        for kind in ("F", "H"):
            for extra in ("", " v1", " v1 2 extra", " missing 1", " Rin 2",
                          " POLY(1) v1 0 1", " v1 1e999", " v1 nonsense"):
                with self.subTest(kind=kind,extra=extra), self.assertRaises(ValueError):
                    route(base + f"{kind}x out 0{extra}")
            with self.assertRaises(ValueError):
                route(base + f"v1 other 0 0\n{kind}x out 0 V1 1")
