"""Full terminal-voltage phasors for the balanced, short series feeder."""
import math
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def feeder(**changes):
    spec = dict(V_ll_V=400., I_A=100., pf=1., length_m=1000.,
                R_ohm_per_km=0., X_ohm_per_km=1.)
    spec.update(changes)
    return run_dual_verify_spec(dict(mode="domain", kind="three_phase_vdrop", spec=spec))


class ThreePhaseFeederVoltage(unittest.TestCase):
    def test_sending_reference_retains_pure_quadrature_drop(self):
        # V_s² = V_r² + 3 (I X)² for load current in phase with V_r.
        out = feeder()
        self.assertFalse(out["passes_3pct"])
        self.assertAlmostEqual(out["vdrop_ll_V"], 400. - math.sqrt(130000.))
        self.assertAlmostEqual(out["V_receiving_ll_V"], math.sqrt(130000.))

    def test_receiving_reference_retains_pure_quadrature_drop(self):
        out = feeder(voltage_reference="receiving")
        self.assertFalse(out["passes_3pct"])
        self.assertAlmostEqual(out["vdrop_ll_V"], math.sqrt(190000.) - 400.)
        self.assertAlmostEqual(out["V_sending_ll_V"], math.sqrt(190000.))
