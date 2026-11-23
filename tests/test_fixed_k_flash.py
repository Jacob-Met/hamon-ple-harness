import math
import unittest
from fractions import Fraction

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def flash(feed, ratios=(2, 1, .5), **extra):
    return run_router_conservation_cad_verifier({
        "mode": "electrochem_separations",
        "spec": {"subkind": "equilibrium_flash", "feed_mol_s": list(feed),
                 "K_values": list(ratios), **extra},
    })


class FixedKFlashTests(unittest.TestCase):
    def test_ternary_interior_and_incipient_compositions(self):
        for feed, state, beta in (([2, 2, 4], "bubble_point", 0),
                                  ([3, 2, 3], "two_phase", .5),
                                  ([4, 2, 2], "dew_point", 1)):
            with self.subTest(state=state):
                r = flash(feed)
                self.assertTrue(r["verified"], r)
                self.assertEqual(r["phase_state"], state)
                self.assertEqual(r["vapor_fraction"], beta)
                self.assertEqual(r["liquid_mole_fractions"], [.25, .25, .5])
                self.assertEqual(r["vapor_mole_fractions"], [.5, .25, .25])
                self.assertEqual(r["liquid_component_mol_s"], [8*(1-beta)*x for x in [.25,.25,.5]])
                self.assertEqual(r["vapor_component_mol_s"], [8*beta*y for y in [.5,.25,.25]])

    def test_single_phase_has_no_absent_phase_composition(self):
        for feed, state, absent in (([1, 2, 5], "liquid", "vapor"),
                                    ([5, 2, 1], "vapor", "liquid")):
            r = flash(feed)
            self.assertTrue(r["verified"], r)
            self.assertEqual(r["phase_state"], state)
            self.assertEqual(r[absent+"_component_mol_s"], [0, 0, 0])
            self.assertIsNone(r[absent+"_mole_fractions"])

    def test_zero_feed_species_and_unidentified_split(self):
        r = flash([0, 2, 3, 0], [1000, 1, 1, .0001])
        self.assertTrue(r["verified"], r)
        self.assertEqual(r["phase_state"], "indeterminate")
        self.assertFalse(r["phase_fraction_identifiable"])
        self.assertEqual(r["vapor_fraction_interval"], [0, 1])
        self.assertIsNone(r["vapor_fraction"])
        self.assertEqual(r["vapor_mole_fractions"], [0, .4, .6, 0])
        s = flash([0,3,3], [100,2,.5])
        self.assertEqual(s["vapor_component_mol_s"], [0,2,1])
        self.assertEqual(s["liquid_component_mol_s"], [0,1,2])

    def test_adjacent_component_rate_decides_endpoint_contact(self):
        for b in (2., 4.):
            for moved in (math.nextafter(b, 0.), math.nextafter(b, math.inf)):
                # At K=(2,1/2), n=(a,c), solving the two original component
                # balances gives V=2a-c and L=2c-a wherever both are positive.
                a, c = (moved, 4.) if b == 2. else (moved, 2.)
                v, l = 2*Fraction(a)-Fraction(c), 2*Fraction(c)-Fraction(a)
                r = flash([a,c], [2,.5])
                self.assertTrue(r["verified"], r)
                if min(v, l) < 0:
                    self.assertEqual(r["phase_state"], "liquid" if v < 0 else "vapor")
                else:
                    self.assertEqual(r["phase_state"], "two_phase")
                    self.assertEqual(r["vapor_mol_s"], float(v))
                    self.assertEqual(r["liquid_mol_s"], float(l))
                    self.assertEqual(r["vapor_fraction"], float(v/(v+l)))
                    self.assertEqual(r["liquid_fraction"], float(l/(v+l)))

    def test_phase_inventory_rounding_tie_with_a_nondyadic_phase_fraction(self):
        a = 2.**52
        for c in (a-.5, a-1.5):
            vapor = 2*Fraction(a)-Fraction(c)
            result = flash([a,c], [2,.5])
            self.assertTrue(result["verified"], result)
            self.assertEqual(result["vapor_mol_s"], float(vapor))
            self.assertEqual(result["vapor_fraction"], float(vapor/(Fraction(a)+Fraction(c))))

    def test_four_components_replay_both_phase_normalizations_and_conservation(self):
        ratios = [4., 2., .5, .25]
        for scale in (1., 2.**140, 2.**-140):
            feed = [n*scale for n in (2,3,4,1)]
            result = flash(feed, ratios)
            self.assertTrue(result["verified"], result)
            self.assertEqual(result["phase_state"], "two_phase")
            x, y = result["liquid_mole_fractions"], result["vapor_mole_fractions"]
            self.assertAlmostEqual(sum(x), 1.)
            self.assertAlmostEqual(sum(y), 1.)
            for i, (n, ratio) in enumerate(zip(feed,ratios)):
                self.assertAlmostEqual(y[i]/x[i], ratio)
                l, v = result["liquid_component_mol_s"][i], result["vapor_component_mol_s"][i]
                self.assertTrue(math.isclose(l+v,n,rel_tol=3e-16))
                self.assertTrue(math.isclose(l/result["liquid_mol_s"],x[i],rel_tol=3e-16))
                self.assertTrue(math.isclose(v/result["vapor_mol_s"],y[i],rel_tol=3e-16))

    def test_invalid_or_extra_thermodynamic_inputs(self):
        for feed, ratios, extra in (([0,0], [2,.5], {}), ([1], [2,.5], {}),
                ([1,-1], [2,.5], {}), ([1,2], [2,0], {}),
                ([1,2], [math.inf,.5], {}), ([1,2], [2,.5], {"T_K":300})):
            with self.subTest(feed=feed, ratios=ratios, extra=extra):
                self.assertFalse(flash(feed, ratios, **extra)["verified"])
