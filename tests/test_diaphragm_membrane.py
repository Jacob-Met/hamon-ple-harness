import copy

import pytest

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


def test_independent_uniform_extension_patch_with_spring_settlement_and_bar():
    spec = {"nodes_m": [[0, 0], [2, 0], [2, 1], [0, 1]],
            "panels": [{"nodes": [0, 1, 2, 3], "Dx_N_m": 3000, "Dy_N_m": 500,
                        "G_N_m": 400}],
            "bars": [{"nodes": pair, "EA_N": 500} for pair in ([0, 1], [3, 2])],
            "springs": [{"node": i, "axis": "x", "k_N_m": 1000} for i in (0, 3)] +
                       [{"node": 0, "axis": "y", "k_N_m": 900}],
            "loads": [{"node": i, "force_N": [20, 0]} for i in (1, 2)]}
    result = run_dual_verify_spec({"mode": "domain", "kind": "diaphragm_membrane", "spec": spec})
    assert result["dual_verified"]
    assert result["bar_tension_N"] == pytest.approx([5, 5])
    assert result["spring_transmitted_N"] == pytest.approx([20, 20, 0])
    assert result["panel_center_shear_N_m"] == pytest.approx([0], abs=1e-11)
    assert [u[0] for u in result["displacements_m"]] == pytest.approx([.02, .04, .04, .02])
    assert result["external_work_J"] == pytest.approx(.8)
    assert result["strain_energy_J"] == pytest.approx(.8)


def test_center_shear_sign_and_reaction_moment_for_pure_shear_patch():
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
    assert result["panel_center_shear_N_m"] == pytest.approx([2])
    assert result["spring_transmitted_N"] == pytest.approx([0, 0, 0], abs=1e-12)


def test_collector_reversal_keeps_global_equilibrium():
    assert "hamon_conservation_cad_verifier" in select_domain_tools("Check a roof diaphragm")
    source = strip()
    source["bars"][0].update(allowable_tension_N=11, allowable_compression_N=10, demand_factor=2)
    changed = copy.deepcopy(source)
    changed["panels"][0]["G_N_m"] /= 100
    cases = [run_router_conservation_cad_verifier({"mode": "structural", "spec": {
        "calculation": "diaphragm_membrane", **s}}) for s in (source, changed)]
    for result in cases:
        assert result["conservation_verified"]
        assert sum(result["spring_transmitted_N"][::2]) == pytest.approx(30)
    assert cases[0]["bar_capacity_checks"][0]["passes"]
    assert not cases[1]["bar_capacity_checks"][0]["passes"]


@pytest.mark.parametrize("fault", ["inverted", "duplicate", "singular", "nan", "disconnected", "hanging", "overlap"])
def test_invalid_mesh_cannot_verify(fault):
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
