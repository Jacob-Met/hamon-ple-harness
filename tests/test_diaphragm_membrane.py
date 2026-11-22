import copy

import unittest

from numpy.testing import assert_allclose

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_dual_verifier import run_dual_verify_spec
from hamon_pareto import select_domain_tools


def strip():
    # Three adjacent panels have a continuous three-segment collector along their south edge.
    return {"nodes_m": [[i, j] for j in (0, 1) for i in range(4)],
            "panels": [{"nodes": [i, i+1, i+5, i+4], "Dx_N_m": 700.,
                        "Dy_N_m": 500., "G_N_m": 600.} for i in range(3)],
            "bars": [{"nodes": [i, i+1], "EA_N": 300.} for i in range(3)],
            "springs": [{"node": i, "axis": axis, "k_N_m": 10000.}
                        for i in (0, 3) for axis in ("x", "y")],
            "loads": [{"node": i+4, "force_N": [f, 0.]} for i, f in enumerate((5, 10, 10, 5))]}


class DiaphragmTests(unittest.TestCase):
    def test_independent_uniform_extension_patch_with_spring_settlement_and_bar(self):
        spec = {"nodes_m": [[0, 0], [2, 0], [2, 1], [0, 1]],
                "panels": [{"nodes": [0, 1, 2, 3], "Dx_N_m": 3000, "Dy_N_m": 500,
                            "G_N_m": 400}],
                "bars": [{"nodes": pair, "EA_N": 500} for pair in ([0, 1], [3, 2])],
                "springs": [{"node": i, "axis": "x", "k_N_m": 1000} for i in (0, 3)] +
                           [{"node": 0, "axis": "y", "k_N_m": 900}],
                "loads": [{"node": i, "force_N": [20, 0]} for i in (1, 2)]}
        result = run_dual_verify_spec({"mode": "domain", "kind": "diaphragm_membrane", "spec": spec})
        assert result["dual_verified"]
        assert_allclose(result["bar_tension_N"], [5, 5], atol=1e-12)
        assert_allclose(result["spring_transmitted_N"], [20, 20, 0], atol=1e-12)
        assert_allclose(result["panel_center_shear_N_m"], [0], atol=1e-11)
        assert_allclose([u[0] for u in result["displacements_m"]], [.02, .04, .04, .02], atol=1e-12)
        assert_allclose(result["external_work_J"], .8, atol=1e-12)
        assert_allclose(result["strain_energy_J"], .8, atol=1e-12)


    def test_center_shear_sign_and_reaction_moment_for_pure_shear_patch(self):
        spec = {"nodes_m": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "panels": [{"nodes": [0, 1, 2, 3], "Dx_N_m": 450, "Dy_N_m": 700,
                            "G_N_m": 200}],
                "springs": [{"node": 0, "axis": "x", "k_N_m": 1000},
                            {"node": 0, "axis": "y", "k_N_m": 1000},
                            {"node": 1, "axis": "y", "k_N_m": 1000}],
                "loads": [{"node": n, "force_N": f} for n, f in enumerate(
                    ([-1, -1], [-1, 1], [1, 1], [1, -1]))]}
        result = run_router_conservation_cad_verifier({"spec": {"calculation": "diaphragm_membrane", **spec}})
        assert result["conservation_verified"]
        assert_allclose(result["panel_center_shear_N_m"], [2], atol=1e-12)
        assert_allclose(result["spring_transmitted_N"], [0, 0, 0], atol=1e-12)


    def test_collector_reversal_keeps_global_equilibrium(self):
        assert "hamon_conservation_cad_verifier" in select_domain_tools("Check a roof diaphragm")
        source = strip()
        source["bars"][0].update(allowable_tension_N=11, allowable_compression_N=10, demand_factor=2)
        changed = copy.deepcopy(source)
        changed["panels"][0]["G_N_m"] /= 100
        cases = [run_router_conservation_cad_verifier({"mode": "structural", "spec": {
            "calculation": "diaphragm_membrane", **s}}) for s in (source, changed)]
        for result in cases:
            assert result["conservation_verified"]
            assert_allclose(sum(result["spring_transmitted_N"][::2]), 30, atol=1e-12)
        assert cases[0]["bar_capacity_checks"][0]["passes"]
        assert not cases[1]["bar_capacity_checks"][0]["passes"]


    def test_tied_wall_stations_retain_both_physical_values(self):
        spec = strip()
        spec["panels"][0]["G_N_m"] /= 100
        spec["loads"] = [{"node": 5, "force_N": [0., 25.]}]
        spec["stations"] = [
            {"edge_nodes": edge, "fraction": fraction, **rule}
            for edge, fraction, rule in (
                ([0, 3], 1/3, {}), ([0, 3], 1/3, {"tie_rule": "first_from_start"}),
                ([3, 0], 2/3, {"tie_rule": "first_from_start"}),
                ([0, 3], 1/3, {"tie_rule": "max_abs"}))]
        result = run_dual_verify_spec({"kind": "diaphragm_membrane", "spec": spec})
        unknown, first, reversed_wall, envelope = result["stations"]
        self.assertEqual(unknown["panels"], [0, 1])
        self.assertIsNone(unknown["magnitude_N_m"])
        expected = list(map(abs, result["panel_center_shear_N_m"][:2]))
        assert_allclose(unknown["signed_shear_N_m"], [50/3, -25/3], atol=1e-11)
        self.assertEqual(first["magnitude_N_m"], expected[0])
        self.assertEqual(reversed_wall["magnitude_N_m"], expected[1])
        self.assertEqual(envelope["magnitude_N_m"], max(expected))
        spec["stations"] = [{"edge_nodes": [1, 5], "fraction": .5}]
        result = run_router_conservation_cad_verifier({"spec": {"calculation": "diaphragm_membrane", **spec}})
        self.assertFalse(result["ok"])

    def test_invalid_mesh_cannot_verify(self):
        for fault in ("inverted", "duplicate", "singular", "nan", "disconnected", "hanging", "overlap"):
            with self.subTest(fault=fault):
                self.check_invalid_mesh(fault)


    def check_invalid_mesh(self, fault):
        spec = strip()
        if fault == "inverted":
            spec["panels"][0]["nodes"].reverse()
        elif fault == "duplicate":
            spec["panels"].append(spec["panels"][0])
        elif fault == "singular":
            spec["springs"] = spec["springs"][:2]
            spec["loads"] = []  # a balanced zero RHS cannot disguise a rigid-body rotation
        elif fault == "nan":
            spec["springs"][0]["k_N_m"] = float("nan")
        elif fault == "disconnected":
            spec["nodes_m"].append([4, 8])
        elif fault == "hanging":
            spec["nodes_m"] += [[0.5, 1], [0.5, 2], [0, 2]]
            spec["panels"].append({**spec["panels"][0], "nodes": [4, 8, 9, 10]})
        else:
            spec["nodes_m"] += [[0.5, 0], [1.5, 0], [1.5, 1], [0.5, 1]]
            spec["panels"].append({**spec["panels"][0], "nodes": [8, 9, 10, 11]})
        result = run_router_conservation_cad_verifier({"mode": "structural", "spec": {
            "calculation": "diaphragm_membrane", **spec}})
        assert not result["ok"] and not result["conservation_verified"]
