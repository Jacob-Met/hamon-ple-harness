"""Pressure-wall admissibility at the registered cylindrical shell route."""
import unittest
from hamon_dual_verifier import run_dual_verify_spec


def shell(**changes):
    spec = dict(P_Pa=2e6, R_inner_m=1., S_allow_Pa=1e6, weld_efficiency_E=1.,
                corrosion_allowance_m=0., supplied_thickness_m=.01)
    spec.update(changes)
    return run_dual_verify_spec(dict(mode="domain", kind="asme_ug27_shell", spec=spec))


class ShellPressureAdmissibility(unittest.TestCase):
    def no_certified_pass(self, **changes):
        try:
            result = shell(**changes)
        except ValueError:
            return
        self.assertIsNot(result.get("passes_ug27"), True, result)
        if result.get("t_required_m") is not None:
            self.assertGreaterEqual(result["t_required_m"], 0., result)

    def test_no_negative_wall_solution_above_pressure_asymptote(self):
        self.no_certified_pass()

    def test_thick_wall_beyond_formula_cannot_certify(self):
        # At P=.4 SE the inferred load-bearing thickness exceeds R/2.
        self.no_certified_pass(P_Pa=400000., supplied_thickness_m=.6)

    def test_efficiency_and_corrosion_cannot_create_a_pressure_wall(self):
        for kw in [dict(P_Pa=-1000.), dict(P_Pa=1000.,weld_efficiency_E=2.),
                   dict(P_Pa=1000.,corrosion_allowance_m=-.1),
                   dict(P_Pa=0.,corrosion_allowance_m=.01)]:
            with self.subTest(kw=kw):
                self.no_certified_pass(**kw)
