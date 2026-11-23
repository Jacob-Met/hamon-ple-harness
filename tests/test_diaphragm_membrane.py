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


    def test_rigid_displacement_with_soft_restraints_and_stiff_membrane(self):
        xy = [[0, 0], [1, 0], [1, 1], [0, 1]]
        for stiffness in (1e9, 1e13):
            for weak_node in (0, 1):
                with self.subTest(stiffness=stiffness, weak_node=weak_node):
                    strong_node = 1 - weak_node
                    angle = .002
                    weak_force = 1e-14 * angle * (weak_node - strong_node)
                    springs = [{"node": 0, "axis": "x", "k_N_m": 1.0},
                               {"node": strong_node, "axis": "y", "k_N_m": 1.0},
                               {"node": weak_node, "axis": "y", "k_N_m": 1e-14}]
                    spec = {"nodes_m": xy,
                            "panels": [{"nodes": [0, 1, 2, 3], "Dx_N_m": stiffness,
                                        "Dy_N_m": stiffness, "G_N_m": stiffness/3}],
                            "springs": springs,
                            "loads": [{"node": 0, "force_N": [.004, 0]},
                                      {"node": weak_node, "force_N": [0, weak_force]}]}
                    result = run_dual_verify_spec({"kind": "diaphragm_membrane", "spec": spec})
                    self.assertTrue(result["dual_verified"])
                    expected = [[.004 - angle*y, angle*(x-strong_node)] for x, y in xy]
                    assert_allclose(result["displacements_m"], expected, rtol=1e-12, atol=1e-17)
                    assert_allclose(result["panel_center_shear_N_m"], [0], atol=1e-15)
                    assert_allclose(result["spring_transmitted_N"][2], weak_force, rtol=1e-12)
                    spec["springs"] = springs[:2]
                    with self.assertRaisesRegex(ValueError, "unrestrained rotation"):
                        run_dual_verify_spec({"kind": "diaphragm_membrane", "spec": spec})

    def test_small_elastic_shear_survives_large_rigid_translation(self):
        xy = [[0, 0], [1, 0], [1, 1], [0, 1]]
        for stiffness in (1e9, 1e13):
            with self.subTest(stiffness=stiffness):
                shear = .002
                shear_strain = shear / (stiffness/3)
                loads = [[.004 - shear/2, -shear/2], [-shear/2, shear/2],
                         [shear/2, shear/2], [shear/2, -shear/2]]
                spec = {"nodes_m": xy,
                        "panels": [{"nodes": [0, 1, 2, 3], "Dx_N_m": stiffness,
                                    "Dy_N_m": stiffness, "G_N_m": stiffness/3}],
                        "springs": [{"node": 0, "axis": "x", "k_N_m": 1},
                                    {"node": 0, "axis": "y", "k_N_m": 1},
                                    {"node": 1, "axis": "y", "k_N_m": 1}],
                        "loads": [{"node": i, "force_N": f} for i, f in enumerate(loads)]}
                result = run_router_conservation_cad_verifier(
                    {"spec": {"calculation": "diaphragm_membrane", **spec}})
                self.assertTrue(result["conservation_verified"], result)
                assert_allclose(result["displacements_m"],
                                [[.004 + shear_strain*y, 0] for _, y in xy], rtol=1e-12, atol=1e-17)
                assert_allclose(result["panel_center_shear_N_m"], [shear], rtol=1e-12)
                assert_allclose(result["spring_transmitted_N"], [.004, 0, 0], atol=1e-16)
                self.assertLess(result["dual_rel_discrepancy"], 1e-12)


    def test_stiff_patches_joined_by_axial_bars(self):
        xy = [[0, 0], [1, 0], [1, 1], [0, 1], [3, 0], [4, 0], [4, 1], [3, 1]]
        panels = [{"nodes": nodes, "Dx_N_m": 1e13, "Dy_N_m": 1e13, "G_N_m": 3e12}
                  for nodes in ([0, 1, 2, 3], [4, 5, 6, 7])]
        supports = [(0, "x", 1), (0, "y", 2), (1, "y", 1e-14),
                    (4, "x", 2), (4, "y", 2), (5, "x", 2), (5, "y", 2)]
        def solve(links, active, displacement, axial_forces):
            forces = [[0., 0.] for _ in xy]
            for node, axis, stiffness in active:
                dim = int(axis == "y")
                forces[node][dim] += stiffness*displacement[node][dim]
            bars = []
            for (a, b, ea), tension in zip(links, axial_forces):
                dx, dy = xy[b][0]-xy[a][0], xy[b][1]-xy[a][1]
                norm = (dx*dx + dy*dy)**.5
                for dim, component in enumerate((dx/norm, dy/norm)):
                    forces[b][dim] += tension*component
                    forces[a][dim] -= tension*component
                bars.append({"nodes": [a, b], "EA_N": ea})
            spec = {"nodes_m": xy, "panels": panels, "bars": bars,
                    "springs": [{"node": n, "axis": axis, "k_N_m": k} for n, axis, k in active],
                    "loads": [{"node": i, "force_N": f} for i, f in enumerate(forces)]}
            return run_dual_verify_spec({"kind": "diaphragm_membrane", "spec": spec})

        rigid = [[.004-.002*y, .002*x] for x, y in xy]
        one = [(1, 7, 3e11)]
        result = solve(one, supports, rigid, [0])
        assert_allclose(result["displacements_m"], rigid, atol=2e-15)
        assert_allclose(result["bar_tension_N"], [0], atol=1e-14)
        with self.assertRaises(ValueError):
            solve(one, supports[:3], rigid, [0])
        # Three independent connector extensions restrain the second patch,
        # which has no direct springs. All bars are unstretched by this motion.
        three = [(1, 4, 3e4), (2, 7, 5e4), (1, 7, 2e4)]
        result = solve(three, supports[:3], rigid, [0, 0, 0])
        assert_allclose(result["displacements_m"], rigid, atol=2e-15)
        # Relative patch translation stretches a real horizontal connector.
        # These nodal loads include its equal and opposite end forces.
        movement = [[.004 + (1e-5 if i >= 4 else 0), 0] for i in range(8)]
        horizontal = [(1, 4, 600.)]
        tension = 600.*1e-5/2
        result = solve(horizontal, supports, movement, [tension])
        assert_allclose(result["displacements_m"], movement, atol=2e-14)
        assert_allclose(result["bar_tension_N"], [tension], rtol=1e-12)

    def test_corner_hinge_compatibility_and_redundant_brace(self):
        xy = [[0, 0], [1, 0], [1, 1], [0, 1], [2, 1], [2, 2], [1, 2]]
        spec = {"nodes_m": xy, "panels": [
            {"nodes": ns, "Dx_N_m": 1e13, "Dy_N_m": 1e13, "G_N_m": 3e12}
            for ns in ([0, 1, 2, 3], [2, 4, 5, 6])],
            "springs": [{"node": 0, "axis": "x", "k_N_m": 1},
                        {"node": 0, "axis": "y", "k_N_m": 1},
                        {"node": 1, "axis": "y", "k_N_m": 1},
                        {"node": 4, "axis": "y", "k_N_m": 1e-14}],
            "loads": [{"node": 0, "force_N": [.004, 0]},
                      {"node": 4, "force_N": [0, 2e-17]}]}
        result = run_dual_verify_spec({"kind": "diaphragm_membrane", "spec": spec})
        expected = [[.004, 0] if i < 4 else [.004-.002*(y-1), .002*(x-1)]
                    for i, (x, y) in enumerate(xy)]
        assert_allclose(result["displacements_m"], expected, atol=2e-15)
        # This bar passes through the shared pin; its first-order extension
        # cannot restrain the second panel's otherwise free hinge rotation.
        spec["springs"].pop()
        spec["bars"] = [{"nodes": [0, 5], "EA_N": 1e4}]
        with self.assertRaises(ValueError):
            run_dual_verify_spec({"kind": "diaphragm_membrane", "spec": spec})

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
