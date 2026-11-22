import math
import unittest
from _scipy_compat import brentq_fallback


class BrentBracketTest(unittest.TestCase):
    def test_previous_iterate_updates_the_bracket(self):
        for fn, expected in ((lambda x: x*x-2, math.sqrt(2)),
                             (lambda x: math.exp(-x)-0.1, math.log(10))):
            with self.subTest(root=expected):
                actual=brentq_fallback(fn,0,10)
                self.assertAlmostEqual(actual,expected,places=10)
                self.assertAlmostEqual(fn(actual),0,places=10)

    def test_embedded_gauss_weights_match_their_nodes(self):
        from _scipy_compat import _gk15_panel
        for degree in (0, 2, 10):
            value, error = _gk15_panel(lambda x: x**degree, -1.0, 1.0, ())
            self.assertAlmostEqual(value, 2 / (degree + 1), places=13)
            self.assertLess(error, 1e-12)
