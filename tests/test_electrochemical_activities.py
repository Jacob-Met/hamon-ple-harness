"""Activity-resolved Nernst calculations using the registered tool entry."""
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def term(species, stoich, activity):
    return dict(species=species, stoich=stoich, activity=activity)


def solve(**params):
    spec = dict(subkind="electrochemistry", E0_V=0, n_electrons=2)
    spec.update(params)
    return run_router_conservation_cad_verifier(dict(mode="electrochem_separations", spec=spec))


class ElectrochemicalActivities(unittest.TestCase):
    def test_published_cells(self):
        for e0, rows, digits, published in [
            (-.17, [term("Co(II)", 1, .15), term("Fe(II)", -1, 1.94)], 2, -.14),
            (0, [term("Zn(II) anode", 1, .1), term("Zn(II) cathode", -1, .5)], 3, .021),
        ]:
            r = solve(E0_V=e0, activity_terms=rows)
            self.assertTrue(r["verified"])
            self.assertEqual(round(r["E_nernst_V"], digits), published)

    def test_legacy_equivalence_scaling_and_inversion(self):
        rows = [term("product", 1, .1), term("reactant", -1, .5)]
        base = solve(activity_terms=rows, Q_reaction=.2)
        self.assertTrue(base["verified"])
        self.assertEqual(base["E_nernst_V"], solve(Q_reaction=.2)["E_nernst_V"])
        self.assertNotIn("ln_Q_reaction", solve(Q_reaction=.2))
        self.assertEqual(solve()["E_nernst_V"], 0)
        doubled = [dict(t, stoich=2*t["stoich"]) for t in reversed(rows)]
        self.assertEqual(base["E_nernst_V"], solve(activity_terms=doubled, n_electrons=4)["E_nernst_V"])
        self.assertEqual(-base["E_nernst_V"], solve(activity_terms=[dict(t, stoich=-t["stoich"]) for t in rows])["E_nernst_V"])

    def test_linear_quotient_overflow_and_underflow(self):
        for sign in (1, -1):
            rows = [term("product", 3*sign, 1e200), term("reactant", -2*sign, 1e-150)]
            r = solve(activity_terms=rows)
            self.assertTrue(r["verified"])
            self.assertAlmostEqual(r["ln_Q_reaction"], sign*900*math.log(10), places=11)

    def test_equilibrium_and_cancellation(self):
        equilibrium = [term("A", .5, 4), term("B", -1, 2)]
        self.assertEqual(solve(activity_terms=equilibrium)["ln_Q_reaction"], 0)
        rows = [term("A", 1e250, 2), term("B", 1e-220, 3), term("C", -1e250, 2)]
        result = solve(activity_terms=rows)
        self.assertTrue(result["verified"])
        self.assertAlmostEqual(result["ln_Q_reaction"] / 1e-220, math.log(3), places=14)

    def test_rejects_conflicts_or_invalid_source_terms(self):
        t = term("Zn", 1, .1)
        bad = [[], {}, [t, dict(t, species=" Zn ")], [dict(t, stoich=0)], [dict(t, activity=0)],
               [dict(t, activity=True)], [dict(t, activity=float("nan"))], [dict(t, species=" ")],
               [dict(t, activity=1, concentration_M=.1)]]
        for rows in bad:
            with self.subTest(rows=rows):
                self.assertFalse(solve(activity_terms=rows)["verified"])
        self.assertFalse(solve(activity_terms=[t], Q_reaction=1)["verified"])
