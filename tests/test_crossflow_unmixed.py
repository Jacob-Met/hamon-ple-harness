import math
import unittest

from hamon_wave7_engineering_verifiers import (
    _hx_eps_from_ntu,
    _unmixed_crossflow_deficit,
    _unmixed_crossflow_ntu,
    solve_exchanger_or_radiation,
)


class UnmixedCrossflowTests(unittest.TestCase):
    def test_moderate_rating_and_sizing_on_both_stream_orientations(self):
        # Independent exact both-unmixed quadrature (ht 1.2.0).
        references = [(1.0, 0.3, 0.22850837737858395),
                      (0.3, 2.0, 0.7841838391085876),
                      (0.3, 20.0, 0.9995232886306167)]
        for ratio, ntu, eps in references:
            for hot, cold in [(2000., 2000./ratio), (2000./ratio, 2000.)]:
                base = dict(subkind="heat_exchanger", flow_arrangement="crossflow_unmixed",
                            C_h=hot, C_c=cold, T_h_in=410., T_c_in=290.)
                rating = solve_exchanger_or_radiation(dict(base, UA=2000.*ntu))
                sizing = solve_exchanger_or_radiation(dict(base, effectiveness=eps))
                with self.subTest(ratio=ratio, ntu=ntu, hot=hot):
                    self.assertTrue(rating["verified"] and sizing["verified"])
                    self.assertAlmostEqual(rating["effectiveness"], eps, places=13)
                    self.assertTrue(math.isclose(sizing["NTU"], ntu, rel_tol=3e-13))
                    lo, hi = sizing["NTU_numerical_bracket"]
                    self.assertLessEqual(lo, sizing["NTU"])
                    self.assertLessEqual(sizing["NTU"], hi)
                    self.assertLessEqual(hi - lo, ntu * 3e-14)
                    self.assertTrue(math.isclose(rating["Q_W"], eps * 240000., rel_tol=3e-13))

    def test_sizing_resolves_rounded_effectiveness_plateau(self):
        # At equal capacity, 1-epsilon ~ 1/sqrt(pi*NTU). The relative
        # correction is O(1/NTU) at this input's NTU > 1e30.
        eps = math.nextafter(1.0, 0.0)
        ntu, _ = _unmixed_crossflow_ntu(eps, 1.0)
        self.assertTrue(math.isclose(ntu, 1.0/(math.pi*(1.0-eps)**2), rel_tol=2e-13))
        self.assertTrue(math.isclose(_unmixed_crossflow_deficit(ntu, 1.0), 1.0-eps, rel_tol=2e-13))
        invalid = solve_exchanger_or_radiation(dict(subkind="heat_exchanger", flow_arrangement="crossflow_unmixed",
                                                    T_h_in=410., T_c_in=290., C_h=2., C_c=2., effectiveness=1.))
        self.assertFalse(invalid["verified"])

    def test_weak_transfer_and_one_infinite_capacity_limit(self):
        report = lambda ntu, cr: _hx_eps_from_ntu(ntu, cr, "crossflow_unmixed")
        for cr in (0.0, 1e-200, 1.0):
            self.assertEqual(report(1e-300, cr), 1e-300)
        for ntu in (0.001, 0.9, 1.1, 15.0):
            self.assertTrue(math.isclose(report(ntu, 0.0), -math.expm1(-ntu), rel_tol=3e-15))

    def test_relation_is_continuous_across_implementation_boundary(self):
        for cr in (0.001, 0.3, math.nextafter(1.0, 0.0), 1.0):
            limits = [_hx_eps_from_ntu(math.nextafter(1.0, side), cr, "crossflow_unmixed")
                      for side in (0.0, 2.0)]
            self.assertLess(abs(limits[0] - limits[1]), 2e-14)
