import cmath
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def circuit(deck, **extra):
    spec = {"netlist": deck, **extra}
    return run_router_conservation_cad_verifier({"mode": "circuit", "spec": spec})


class TestCoupledInductors(unittest.TestCase):
    def test_loaded_secondary_winding_phasors_and_power(self):
        # Eliminate winding currents analytically with i_load = Vsecondary/R.
        for k in (0.3, 0.8, 1.0):
            for reversed_secondary in (False, True):
                with self.subTest(k=k, reversed=reversed_secondary):
                    sign = -1 if reversed_secondary else 1
                    dots = "0 sec" if reversed_secondary else "sec 0"
                    deck = (f"Kx lPRI Lsec {k}\nV1 pri 0 AC 1 30\nLpri pri 0 10m\n"
                            f"Lsec {dots} 40m\nRL sec 0 100\n.ac lin 1 1k 1k")
                    v1 = cmath.rect(1, math.pi / 6)
                    omega = math.tau * 1000
                    mutual = k * .02
                    v2 = sign * (mutual / .01) * v1 / (1 + 1j * omega * .04 * (1-k*k)/100)
                    i2 = -sign * v2/100
                    i1 = (v1/(1j*omega) - mutual*i2)/.01
                    result = circuit(deck, proposed_node_voltages={"sec": [v2.real, v2.imag]})
                    self.assertTrue(result["conservation_verified"])
                    currents = {br["name"]: complex(*br["current_A"]) for br in result["branch_currents"]}
                    self.assertAlmostEqual(currents["Lpri"], i1)
                    self.assertAlmostEqual(currents["Lsec"], i2)
                    self.assertAlmostEqual(complex(*result["node_voltages"]["sec"]), v2)
                    self.assertAlmostEqual(result["total_complex_power_delivered_VA"][0], abs(v2)**2 / 100)
                    self.assertFalse(circuit(deck, proposed_node_voltages={"sec": 0})["conservation_verified"])

    def test_three_winding_terminal_voltages_and_energy(self):
        # Independent current excitation prescribes winding currents without
        # solving a circuit. At omega=2*pi each port voltage is j*omega*flux.
        for kcards, coefficients in [
            ("Kall La Lb Lc .5", (.5, .5, .5)),
            ("Kab La Lb .4\nKbc Lb Lc .7", (.4, 0, .7)),
            ("Kall La Lb Lc 1", (1, 1, 1)),
        ]:
            with self.subTest(kcards=kcards):
                deck = ("Three loads\nLa a 0 1\nLb b 0 4\nLc c 0 9\nIa 0 a AC 1\n"
                        "Ib 0 b AC 2 90\nIc 0 c AC 3 180\n" + kcards)
                r = circuit(deck, freq_hz=1)
                self.assertTrue(r["conservation_verified"])
                ab, ac, bc = coefficients
                flux = (1 + 4j*ab - 9*ac, 2*ab + 8j - 18*bc, 3*ac + 12j*bc - 27)
                for node, linkage in zip("abc", flux):
                    self.assertAlmostEqual(complex(*r["node_voltages"][node]), 1j * math.tau * linkage)
                expected_q = math.tau * (1 + 16 + 81 - 18*ac)
                self.assertAlmostEqual(r["total_complex_power_delivered_VA"][1], expected_q)

    def test_dc_and_uncoupled_limit(self):
        r = circuit("DC coils\nV1 in 0 5\nR1 in a 2\nLa a 0 1\nLb b 0 2\n"
                    "R2 b 0 3\nK1 La Lb .8")
        self.assertTrue(r["conservation_verified"])
        self.assertEqual(r["node_voltages"]["a"], 0)
        self.assertEqual(r["node_voltages"]["b"], 0)
        self.assertEqual(r["branch_currents"][2]["current_A"], 2.5)
        r = circuit("No coupling\nV1 a 0 AC 1\nLa a 0 1\nLb b 0 2\nR2 b 0 3", freq_hz=60)
        self.assertTrue(r["conservation_verified"])
        self.assertEqual(complex(*r["node_voltages"]["b"]), 0)

    def test_bad_couplings_refused_before_certification(self):
        deck = "Coils\nLa a 0 1\nLb b 0 2\nLc c 0 3\nIa 0 a AC 1\n"
        for cards in ("K1 La Lmissing .5", "K1 La Ia .5", "K1 La La .5",
                      "K1 La Lb .8\nK2 lb la .7", "K1 La Lb NaN", "K1 La Lb 1.001",
                      "K1 La Lb 0", "K1 La Lb -.3", "K1 La", "K1 La Lb .9\nK2 Lb Lc .9",
                      "K1 La Lb 1\nK2 La Lc .5", "K1 La Lb .3\nK1 La Lc .3",
                      "Ld d 0 2 m=3\nK1 La Ld .5", "lA d 0 4\nK1 La Lb .5"):
            with self.subTest(cards=cards), self.assertRaises(ValueError):
                circuit(deck + cards, freq_hz=60)

    def test_structured_cards_keep_original_branch_current_indices(self):
        branches = [
            {"type": "L", "name": "L1", "n_pos": "a", "n_neg": "0", "value": 1},
            {"type": "K", "name": "K1", "inductors": ["L1", "L2"], "value": .5},
            {"type": "L", "name": "L2", "n_pos": "b", "n_neg": "0", "value": 4},
            {"type": "I", "name": "I1", "n_pos": "0", "n_neg": "a", "value": 3},
        ]
        r = run_router_conservation_cad_verifier({"mode": "circuit", "spec": {"branches": branches, "freq_hz": 1}})
        self.assertTrue(r["conservation_verified"])
        self.assertEqual([br["index"] for br in r["branch_currents"]], [0,2,3])
        self.assertAlmostEqual(complex(*r["node_voltages"]["b"]), 3j * math.tau)
