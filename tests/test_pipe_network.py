"""Physical references for the passive-network conservation gate."""

import asyncio
import json
import math
import subprocess
import sys
from pathlib import Path

import engineering_intake

import pytest

from hamon_pipe_network import verify_pipe_network
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_dual_verifier import run_dual_verify_spec
from hamon_pareto import select_domain_tools


def public_three_loop_case():
    # E. Maurer, hydraulics R package vignette, §7, variable-friction example.
    # https://cran.r-project.org/web/packages/hydraulics/vignettes/hydraulics_vignette.html
    endpoints = ["RA", "AB", "BC", "CD", "DR", "CE", "EF", "FD", "BG", "GE"]
    lengths = [250, 100, 125, 125, 100, 100, 125, 100, 100, 125]
    diameters = [0.30, 0.20, 0.20, 0.20, 0.20, 0.15, 0.25, 0.15, 0.15, 0.25]
    demands = {"C": 0.04, "F": 0.02, "G": 0.02}
    return {
        "calculation": "pipe_network",
        "nodes": [{"id": "R", "fixed_head_m": 70}]
        + [{"id": node, "demand_m3_s": demands.get(node, 0)} for node in "ABCDEFG"],
        "pipes": [
            {
                "id": str(i),
                "from": a,
                "to": b,
                "length_m": length,
                "diameter_m": diameter,
                "roughness_m": 0.00025,
            }
            for i, ((a, b), length, diameter) in enumerate(
                zip(endpoints, lengths, diameters), 1
            )
        ],
    }


def test_published_three_loop_benchmark():
    # The published solution stops after three iterations and prints four decimals.
    expected = [
        0.0382,
        0.0382,
        0.0230,
        -0.0258,
        -0.0418,
        0.0088,
        0.0040,
        -0.0160,
        0.0152,
        -0.0048,
    ]
    result = verify_pipe_network(public_three_loop_case())
    assert result["conservation_verified"], result
    assert result["independent_cycles"] == 3
    assert result["reservoir_supply_m3_s"] == pytest.approx(0.08)
    assert list(result["flows_m3_s"].values()) == pytest.approx(expected, abs=0.0003)
    assert result["max_mass_residual_m3_s"] < 1e-9
    assert result["max_energy_residual_m"] < 1e-7


def parallel_case(demand):
    return {
        "calculation": "pipe_network",
        "nodes": [{"id": "J", "demand_m3_s": demand}, {"id": "R", "fixed_head_m": 50}],
        "pipes": [
            {"id": "a", "from": "R", "to": "J", "length_m": 20, "diameter_m": 0.1},
            {"id": "b", "from": "J", "to": "R", "length_m": 80, "diameter_m": 0.1},
        ],
    }


@pytest.mark.parametrize("demand", [0, 1e-12, 1e-7, -1e-6])
def test_signed_laminar_parallel_solution(demand):
    case = parallel_case(demand)
    expected = {"a": 0.8 * demand, "b": -0.2 * demand}
    case["proposed_flows_m3_s"] = expected
    result = verify_pipe_network(case)
    assert result["conservation_verified"], result
    assert result["flows_m3_s"] == pytest.approx(expected, abs=1e-20)
    drop = 128 * 1e-6 * 20 * expected["a"] / (math.pi * 9.80665 * 0.1**4)
    assert result["heads_m"]["J"] == pytest.approx(50 - drop, abs=1e-11)


def test_incorrect_or_partial_claim_and_disconnected_node():
    case = parallel_case(1e-10)
    case["proposed_flows_m3_s"] = {"a": 1e-8, "b": 0}
    assert not verify_pipe_network(case)["conservation_verified"]
    case["proposed_flows_m3_s"] = {"a": 0}
    assert not verify_pipe_network(case)["ok"]
    del case["proposed_flows_m3_s"]
    case["nodes"].append({"id": "Island"})
    assert not verify_pipe_network(case)["ok"]


def test_unrepresented_pump_never_receives_certificate():
    case = parallel_case(0.01)
    case["pipes"][0]["pump_head_m"] = 3
    assert not verify_pipe_network(case)["ok"]
    del case["pipes"][0]["pump_head_m"]
    case["pumps"] = [{"from": "R", "to": "J"}]
    assert not verify_pipe_network(case)["ok"]


def test_normal_auto_and_legacy_routes_check_answer():
    case = parallel_case(1e-7)
    assert "hamon_conservation_cad_verifier" in select_domain_tools("Solve the pipe network")
    for valid in (True, False):
        case["proposed_flows_m3_s"] = {"a": (0.8 if valid else 0.7)*1e-7, "b": -0.2e-7}
        assert run_router_conservation_cad_verifier({"spec": case})["conservation_verified"] is valid
        assert run_dual_verify_spec({"mode": "domain", "kind": "pipe_network", "spec": case})["dual_verified"] is valid


def test_isolated_participant_dual_verifier_cli(tmp_path):
    class TaskContainer:
        async def write_file(self, path, content):
            return (tmp_path / Path(path).name).write_bytes(content)

    note = asyncio.run(engineering_intake.provision_engineering_cli(TaskContainer()))
    assert "dual-verify" in note
    water = {"mode": "domain", "kind": "pipe_network", "spec": parallel_case(1e-7)}
    soil = {
        "mode": "domain", "kind": "consolidation_settlement",
        "spec": {"H_m": 2, "e0": .5, "Cc": .2, "Cr": .1, "sigma_v0_Pa": 10000,
                 "sigma_p_Pa": 10000, "delta_sigma_Pa": 10000, "max_settlement_m": .5},
    }
    multi = {"mode": "domain", "kind": "pipe_network", "spec": three_reservoir_case()}
    for problem in (water, multi, soil):
        source = tmp_path / "problem.json"
        source.write_text(json.dumps(problem))
        run = subprocess.run(
            [sys.executable, "-E", "-s", str(tmp_path / "hamon-engineering-toolkit.py"),
             "dual-verify", str(source)], cwd=tmp_path, capture_output=True, text=True,
        )
        assert run.returncode == 0, (run.stdout, run.stderr)
        assert json.loads(run.stdout)["dual_verified"] is True


def test_tiny_head_gradient_and_mixed_scale_branch():
    # Laminar Hagen–Poiseuille head from two equally loaded short pipes.
    demand = 1e-8
    loss = 128e-6 * (demand / 2) / (math.pi * 9.80665 * .1**4)
    case = {
        "nodes": [{"id": "R", "fixed_head_m": 0}, {"id": "J", "demand_m3_s": demand}],
        "pipes": [{"id": i, "from": "R", "to": "J", "length_m": 1, "diameter_m": .1}
                  for i in ("a", "b")],
    }
    for sign, expected in [(-1, True), (1, False)]:
        case["proposed_heads_m"] = {"R": 0, "J": sign * loss}
        result = verify_pipe_network(case)
        assert result["conservation_verified"] is expected
        assert result["proposed_checks"]["pipe_head_drops_match"] is expected
    case = {
        "nodes": [{"id": "R", "fixed_head_m": 10},
                  {"id": "M", "demand_m3_s": .01}, {"id": "T", "demand_m3_s": 5e-10}],
        "pipes": [{"id": i, "from": "R", "to": i, "length_m": 1, "diameter_m": .1}
                  for i in ("M","T")],
    }
    for tap, expected in ((5e-10, True), (0, False)):
        case["proposed_flows_m3_s"] = {"M": .01, "T": tap}
        result = verify_pipe_network(case)
        assert result["conservation_verified"] is expected
        assert result["proposed_checks"]["junction_demands_match"] is expected

@pytest.mark.parametrize("direction", [1, -1])
def test_radial_capillary_and_low_resistance_tap(direction):
    # Independent laminar Hagen–Poiseuille solution, with the tap pipe reversed.
    feed, tap = direction * 5.1e-7, direction * 1e-8
    coefficient = 128e-6 / (math.pi * 9.80665)
    mid = 100 - coefficient * 10 * feed / 0.001**4
    end = mid - coefficient * tap / 0.02**4
    case = {
        "nodes": [{"id": "M", "demand_m3_s": feed - tap}, {"id": "T", "demand_m3_s": tap},
                  {"id": "R", "fixed_head_m": 100}],
        "pipes": [{"id": "tap", "from": "T", "to": "M", "length_m": 1, "diameter_m": 0.02},
                  {"id": "feed", "from": "R", "to": "M", "length_m": 10, "diameter_m": 0.001}],
        "proposed_flows_m3_s": {"tap": -tap, "feed": feed},
        "proposed_heads_m": {"R": 100., "M": mid, "T": end},
    }
    solved = verify_pipe_network(case)
    assert solved["conservation_verified"], solved
    assert solved["heads_m"] == pytest.approx(case["proposed_heads_m"], abs=1e-12, rel=0)
    assert solved["flows_m3_s"] == pytest.approx(case["proposed_flows_m3_s"], abs=1e-20, rel=0)
    assert max(solved["reynolds"].values()) < 2000
    assert solved["max_nodal_cycle_flow_difference_m3_s"] < 1e-20
    assert run_dual_verify_spec({"mode": "domain", "kind": "pipe_network", "spec": case})["dual_verified"]
    case["proposed_heads_m"]["T"] = mid
    assert not verify_pipe_network(case)["proposed_checks"]["pipe_head_drops_match"]
    case["proposed_heads_m"]["T"] = end
    case["proposed_flows_m3_s"]["tap"] = tap
    assert not verify_pipe_network(case)["proposed_checks"]["junction_demands_match"]


@pytest.mark.parametrize("left_length,right_length", [(200.0, 200.0), (100.0, 300.0)])
@pytest.mark.parametrize("gravity", [9.80665, 1.62])
def test_prescribed_heads_laminar_series(left_length, right_length, gravity):
    # Hagen–Poiseuille gives resistance proportional to length for equal diameters.
    case = {
        "nodes": [{"id": "H", "fixed_head_m": 10.02}, {"id": "M"},
                  {"id": "L", "fixed_head_m": 10.0}],
        "pipes": [{"id": "a", "from": "H", "to": "M", "length_m": left_length, "diameter_m": .02},
                  {"id": "b", "from": "L", "to": "M", "length_m": right_length, "diameter_m": .02}],
    }
    total_length = left_length + right_length
    case["gravity_m_s2"] = gravity
    flow = math.pi * gravity * .02**4 * .02 / (128e-6 * total_length)
    midpoint = 10.02 - .02 * left_length / total_length
    case["proposed_heads_m"] = {"H": 10.02, "M": midpoint, "L": 10.0}
    case["proposed_flows_m3_s"] = {"a": flow, "b": -flow}
    result = run_dual_verify_spec({"mode": "domain", "kind": "pipe_network", "spec": case})
    assert result["dual_verified"], result
    solved = verify_pipe_network(case)
    assert solved["heads_m"]["M"] == pytest.approx(midpoint, abs=1e-13)
    assert solved["flows_m3_s"] == pytest.approx(case["proposed_flows_m3_s"], rel=1e-11)
    assert solved["reservoir_supplies_m3_s"] == pytest.approx({"H": flow, "L": -flow}, rel=1e-11)
    assert solved["independent_cycles"] == 0
    assert max(solved["reynolds"].values()) < 2000
    case["proposed_heads_m"]["L"] += .001
    assert not verify_pipe_network(case)["proposed_checks"]["reservoir_head_matches"]


@pytest.mark.parametrize("gravity", [1.62, 9.780, 9.832])
def test_site_gravity_checks_local_head_and_rejects_standard_gravity_claim(gravity):
    length, diameter, viscosity, demand, minor = 34.0, .014, 1e-5, 1e-5, 3.5
    speed = demand / (math.pi * diameter**2 / 4)
    # Hagen–Poiseuille wall friction and the fitting's minor loss, in J/kg.
    specific_loss = 32 * viscosity * length * speed / diameter**2 + minor * speed**2 / 2
    case = {
        "calculation": "pipe_network", "kinematic_viscosity_m2_s": viscosity,
        "gravity_m_s2": gravity,
        "nodes": [{"id": "R", "fixed_head_m": 50}, {"id": "J", "demand_m3_s": demand}],
        "pipes": [{"id": "p", "from": "R", "to": "J", "length_m": length,
                   "diameter_m": diameter, "minor_loss_K": minor}],
        "proposed_heads_m": {"R": 50, "J": 50 - specific_loss / gravity},
    }
    solved = run_router_conservation_cad_verifier({"spec": case})
    assert solved["conservation_verified"], solved
    assert solved["heads_m"]["J"] == pytest.approx(case["proposed_heads_m"]["J"], abs=1e-11)
    case["proposed_heads_m"]["J"] = 50 - specific_loss / 9.80665
    assert not run_dual_verify_spec(
        {"mode": "domain", "kind": "pipe_network", "spec": case}
    )["dual_verified"]


@pytest.mark.parametrize("gravity", [0, -1, True, "9.81", None, math.inf, math.nan])
def test_invalid_site_gravity_is_rejected(gravity):
    assert not verify_pipe_network({**parallel_case(1e-7), "gravity_m_s2": gravity})["ok"]


def three_reservoir_case():
    # Independent linear network: sum((Hi - Hj) / Ri) = demand.
    lengths = [100, 200, 300]
    imposed = [30.025, 30.01, 29.995]
    resistance = [128e-6 * length / (math.pi * 9.80665 * .02**4) for length in lengths]
    demand = 1.2e-6
    junction = (sum(h / r for h, r in zip(imposed, resistance)) - demand) / sum(1/r for r in resistance)
    supply = [(h - junction)/r for h, r in zip(imposed, resistance)]
    names = ["upper", "middle", "lower"]
    return {
        "calculation": "pipe_network",
        "nodes": [{"id": "tap", "demand_m3_s": demand}]
                 + [{"id": name, "fixed_head_m": h} for name, h in zip(names, imposed)],
        "pipes": [{"id": name, "from": name, "to": "tap", "length_m": length, "diameter_m": .02}
                  for name, length in zip(names, lengths)],
        "proposed_heads_m": dict(zip(names, imposed), tap=junction),
        "proposed_flows_m3_s": dict(zip(names, supply)),
    }


def test_three_boundary_supplies_and_signed_demand():
    case = three_reservoir_case()
    result = run_router_conservation_cad_verifier({"spec": case})
    assert result["conservation_verified"], result
    supplied = result["reservoir_supplies_m3_s"]
    assert supplied == pytest.approx(case["proposed_flows_m3_s"], rel=1e-10)
    assert min(supplied.values()) < 0 < max(supplied.values())
    assert sum(supplied.values()) == pytest.approx(1.2e-6, abs=1e-16)
    assert result["max_nodal_cycle_flow_difference_m3_s"] < 1e-14


def test_all_heads_prescribed_and_disconnected_reservoir_components():
    nodes = [{"id": "a", "fixed_head_m": 2.0}, {"id": "b", "fixed_head_m": 2.0}]
    pipe = {"id": "link", "from": "a", "to": "b", "length_m": 20, "diameter_m": .1}
    result = verify_pipe_network({"nodes": nodes, "pipes": [pipe]})
    assert result["conservation_verified"], result
    assert result["flows_m3_s"] == {"link": 0.0}
    nodes.append({"id": "c", "fixed_head_m": 1.0})
    assert not verify_pipe_network({"nodes": nodes, "pipes": [pipe]})["ok"]


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("tap_first", [True, False])
@pytest.mark.parametrize("separate_reservoirs", [True, False])
def test_parallel_capillary_feed_keeps_downstream_minor_gradient(direction, tap_first, separate_reservoirs):
    # Two laminar feed pipes have 1:5 resistance. The terminal tap drops
    # 0.26 micrometres while the feed drops over twenty metres.
    feed, tap = direction * 5.1e-7, direction * 1e-8
    coefficient = 128e-6 / (math.pi * 9.80665)
    left = 100.
    right = left + direction * .025 if separate_reservoirs else left
    resistance = coefficient * 12 / .001**4
    mid = (left + right / 5 - resistance * feed) / 1.2
    short_flow = (left - mid) / resistance
    long_flow = (mid - right) / (5 * resistance)
    end = mid - coefficient * tap / .02**4
    nodes = [{"id": "M", "demand_m3_s": feed - tap}, {"id": "T", "demand_m3_s": tap},
             {"id": "R", "fixed_head_m": left}]
    heads = {"M": mid, "T": end, "R": left}
    right_name = "R"
    if separate_reservoirs:
        right_name = "S"
        nodes.append({"id": "S", "fixed_head_m": right})
        heads["S"] = right
    pipes = [
        {"id": "short", "from": "R", "to": "M", "length_m": 12, "diameter_m": .001},
        {"id": "long", "from": "M", "to": right_name, "length_m": 60, "diameter_m": .001},
    ]
    tap_pipe = {"id": "tap", "from": "T", "to": "M", "length_m": 1, "diameter_m": .02}
    pipes.insert(0 if tap_first else len(pipes), tap_pipe)
    case = {
        "calculation": "pipe_network", "nodes": nodes, "pipes": pipes,
        "proposed_heads_m": heads,
        "proposed_flows_m3_s": {"short": short_flow, "long": long_flow, "tap": -tap},
    }
    result = verify_pipe_network(case)
    assert result["conservation_verified"], result
    assert result["independent_cycles"] == (0 if separate_reservoirs else 1)
    assert result["max_nodal_cycle_flow_difference_m3_s"] < 1e-20
    assert result["heads_m"] == pytest.approx(heads, rel=0, abs=1e-12)
    assert result["flows_m3_s"] == pytest.approx(case["proposed_flows_m3_s"], rel=0, abs=1e-20)
    case["proposed_heads_m"]["T"] = mid
    assert not verify_pipe_network(case)["proposed_checks"]["pipe_head_drops_match"]
