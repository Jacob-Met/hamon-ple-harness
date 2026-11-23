"""Contrast checks for two independently supported rectangular patches."""
import copy
import unittest

from numpy.testing import assert_allclose

from hamon_dual_verifier import run_dual_verify_spec


def coupled_panels():
    return {
        "nodes_m": [[0, 0], [0, 1], [-1, 1], [-1, 0],
                    [0, 3], [0, 4], [-1, 4], [-1, 3]],
        "panels": [{"nodes": nodes, "Dx_N_m": 2e11, "Dy_N_m": 1e11,
                    "G_N_m": 1e11/3} for nodes in ([3, 0, 1, 2], [7, 4, 5, 6])],
        "bars": [{"nodes": nodes, "EA_N": ea} for nodes, ea in
                 [([1, 4], 1e13), ([2, 7], 110.), ([1, 7], 220.)]],
        "springs": [{"node": i, "axis": axis, "k_N_m": k}
                    for i, axis, k in [(0, "y", 1.), (0, "x", 2.), (1, "x", 3.),
                                       (4, "y", 1.), (4, "x", 2.), (5, "x", 3.)]],
        "loads": [{"node": 1, "force_N": [.013, .021]},
                  {"node": 6, "force_N": [-.017, .009]}],
    }


def solve(spec):
    return run_dual_verify_spec({"kind": "diaphragm_membrane", "spec": spec})


class ConnectorForceTests(unittest.TestCase):
    def test_force_contrast_under_off_axis_loading(self):
        result = solve(coupled_panels())
        self.assertTrue(result["dual_verified"])
        # Independent, 70-decimal-digit Cartesian Q4 energy Hessian, using
        # four integration points per panel and axial/ground spring energies.
        # No patch rigid-motion transform enters these reference values.
        assert_allclose(result["bar_tension_N"],
                        [-.007803029848427894, -.005405982362245012, .008059920676505681],
                        rtol=1e-10, atol=1e-14)
        assert_allclose(result["panel_center_shear_N_m"],
                        [.005405982362245012, -.007196970151875893], rtol=1e-10)
        assert_allclose(result["spring_transmitted_N"],
                        [.01499999999969621, .005405982362245012, .003989511532570428,
                         .01500000000030379, -.003592464046691334, -.009803029848124109],
                        rtol=1e-10)
        assert_allclose(result["strain_energy_J"], .0002547976922303322, rtol=1e-10)
        assert_allclose(result["displacements_m"][6],
                        [-.00326767661628073, .01352855540774161], rtol=1e-10)

    def test_redundant_parallels_share_tension_and_retain_external_response(self):
        spec = coupled_panels()
        one = solve(spec)
        spec["bars"].append(copy.deepcopy(spec["bars"][0]))
        spec["bars"][0]["EA_N"] *= .25
        spec["bars"][-1]["EA_N"] *= .75
        # Reversing the endpoint order preserves signed tension.
        spec["bars"][-1]["nodes"].reverse()
        two = solve(spec)
        self.assertTrue(two["dual_verified"])
        for key in ("displacements_m", "spring_transmitted_N", "panel_center_shear_N_m"):
            assert_allclose(two[key], one[key], atol=1e-14, rtol=1e-10)
        first, second, third = one["bar_tension_N"]
        assert_allclose(two["bar_tension_N"], [.25*first, second, third, .75*first], atol=1e-14, rtol=1e-10)
