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
    for problem in (water, soil):
        source = tmp_path / "problem.json"
        source.write_text(json.dumps(problem))
        run = subprocess.run(
            [sys.executable, "-E", "-s", str(tmp_path / "hamon-engineering-toolkit.py"),
             "dual-verify", str(source)], cwd=tmp_path, capture_output=True, text=True,
        )
        assert run.returncode == 0, (run.stdout, run.stderr)
        assert json.loads(run.stdout)["dual_verified"] is True
