import math
import unittest
from fractions import Fraction

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_pde_control_psychro_fem import verify_control_system


def solve(a, **fields):
    return verify_control_system({"mode": "state_space", "timebase": "discrete", "A": a, **fields})


class DiscreteTimeControlTest(unittest.TestCase):
    def test_published_state_space_examples(self):
        # Reference P values from scipy.linalg.solve_discrete_lyapunov(A.T, I).
        examples = (
            ("CTMS mass-spring-damper", [[.9990, .0095], [-.1903, .9039]], True,
             [[130.42842793063753, 2.49331871948051], [2.49331871948051, 5.763905014953808]]),
            ("SciPy transposed example", [[.2, .7], [.5, -.9]], False,
             [[.7087289269727466, 1.4351882212069762], [1.4351882212069762, -2.4266314973902343]]),
        )
        for name, a, stable, reference in examples:
            with self.subTest(name=name):
                spec = {"mode": "state_space", "timebase": "discrete", "A": a}
                result = run_router_conservation_cad_verifier({"mode": "control", "spec": spec})
                self.assertTrue(result["passed"])
                self.assertEqual(result["schur_stable"], stable)
                self.assertEqual(result["schur_recursion_stable"], stable)
                self.assertEqual(result["lyapunov_pd"], stable)
                self.assertEqual(result["lyapunov_solution_status"], "unique")
                p = [[Fraction(v) for v in row] for row in result["lyapunov_P_exact"]]
                for i in range(2):
                    for j in range(2):
                        self.assertAlmostEqual(float(p[i][j]), reference[i][j], places=10)
                        propagated = sum(Fraction(a[k][i]) * p[k][l] * Fraction(a[l][j])
                                         for k in range(2) for l in range(2))
                        self.assertEqual(p[i][j] - propagated, int(i == j))

    def test_scalar_recurrences(self):
        for a in (-2, -.5, 0, .5, 2):
            with self.subTest(a=a):
                result = solve([[a]], sample_time_s=.125)
                self.assertTrue(result["passed"])
                self.assertEqual(result["schur_stable"], abs(a) < 1)
                self.assertEqual(result["schur_recursion_stable"], abs(a) < 1)
                self.assertNotIn("hurwitz_stable", result)
                self.assertEqual(result["lyapunov_P_exact"], [[str(1 / (1 - Fraction(a)**2))]])

    def test_complex_and_real_unit_circle_boundaries(self):
        for m in (1, -1, math.nextafter(1, 0), math.nextafter(1, math.inf)):
            for a in ([[m]], [[0, -m], [m, 0]], [[m, 25], [0, m]]):
                with self.subTest(m=m, a=a):
                    result = solve(a)
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["schur_stable"], abs(m) < 1)
                    self.assertEqual(result["schur_recursion_stable"], abs(m) < 1)

    def test_nonnormal_multistate_identity_and_custom_forcing(self):
        a = [[.5, 8, -2], [0, .25, 2], [0, 0, -.75]]
        q = [[0, 1, 2], [1, -2, 0], [2, 0, 1]]
        result = solve(a, Q=q, B=[[0], [0], [1]], C=[[1, 0, 0]])
        self.assertTrue(result["passed"])
        self.assertTrue(result["schur_stable"])
        self.assertTrue(result["observable"])
        self.assertTrue(result["controllable"])
        self.assertTrue(result["stability_certificate"]["positive_definite"])
        self.assertFalse(result["lyapunov_pd"])
        for p_text, forcing in ((result["lyapunov_P_exact"], q),
                                (result["stability_certificate"]["P_exact"],
                                 [[int(i == j) for j in range(3)] for i in range(3)])):
            p = [[Fraction(v) for v in row] for row in p_text]
            for i in range(3):
                for j in range(3):
                    propagated = sum(Fraction(a[k][i]) * p[k][l] * Fraction(a[l][j])
                                     for k in range(3) for l in range(3))
                    self.assertEqual(p[i][j] - propagated, forcing[i][j])

    def test_singular_stein_operator(self):
        result = solve([[2, 0], [0, .5]])
        self.assertTrue(result["passed"])
        self.assertFalse(result["schur_stable"])
        self.assertEqual(result["lyapunov_solution_status"], "nonunique")
        result = solve([[0, -1], [1, 0]])
        self.assertFalse(result["schur_stable"])
        self.assertEqual(result["lyapunov_solution_status"], "inconsistent")

    def test_continuous_output_and_timebase_validation(self):
        default = verify_control_system({"mode": "state_space", "A": [[-2]]})
        explicit = verify_control_system({"mode": "state_space", "A": [[-2]], "timebase": "continuous"})
        self.assertEqual(default, explicit)
        self.assertTrue(default["hurwitz_stable"])
        self.assertNotIn("schur_stable", default)
        for timebase, period in (("continuous", 1), ("sampled_typo", 1), ("discrete", 0),
                                 ("discrete", -1), ("discrete", True), ("discrete", "1"),
                                 ("discrete", math.inf), ("discrete", math.nan)):
            with self.subTest(timebase=timebase, period=period):
                self.assertFalse(solve([[.5]], timebase=timebase, sample_time_s=period)["passed"])
        self.assertFalse(verify_control_system({"A": [[.5]], "sample_time_s": 1})["passed"])
        self.assertFalse(verify_control_system({"den": [1, .5], "timebase": "discrete"})["passed"])
        for period, claim in ((1, 1), (2**53 + 1, 2**53), (10**400, 10**400)):
            result = solve([[.5]], sample_time_s=period, claimed_field="sample_time_s",
                           claimed_value=claim, atol=0, rtol=0)
            self.assertFalse(result["passed"])
            self.assertIn("metadata", result["reason"])

    def test_cross_domain_or_incompatible_claim(self):
        for field, claim in (("hurwitz_stable", True), ("schur_stable", False),
                             ("schur_stable", .99), ("schur_recursion_stable", False),
                             ("lyapunov_pd", False), ("lyapunov_residual_exact", False)):
            with self.subTest(field=field, claim=claim):
                self.assertFalse(solve([[.5]], claimed_field=field, claimed_value=claim, atol=100)["passed"])
        for tol in (-1, math.inf, math.nan):
            self.assertFalse(solve([[.5]], claimed_field="schur_stable", claimed_value=True, rtol=tol)["passed"])
        self.assertTrue(solve([[.5]], claimed_field="schur_stable", claimed_value=True)["passed"])
        self.assertFalse(verify_control_system({"A": [[-.5]], "claimed_field": "schur_stable",
                                               "claimed_value": True})["passed"])


if __name__ == "__main__":
    unittest.main()
