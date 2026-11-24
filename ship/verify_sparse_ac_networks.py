"""Compare public AC grids against independent PYPOWER Newton-Raphson solutions.

Fetch sources separately; the receiver executes no MATLAB and embeds no case answers.
From a harness checkout:
  git clone https://github.com/MATPOWER/matpower /tmp/matpower
  git -C /tmp/matpower checkout b6c85b0323d20fb801cab3377c803a285d730847
  git clone https://github.com/rwl/PYPOWER /tmp/pypower
  git -C /tmp/pypower checkout c7f1cf50bedd34012b92fae4f092ec55617546a8
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python ship/verify_sparse_ac_networks.py \
      /tmp/matpower /tmp/pypower

PEGASE is a fictitious European-scale grid. The Polish case represents the
1999-2000 winter peak; the IEEE cases are standard benchmarks. See each case's
upstream header for attribution and modeling assumptions. This course covers
fixed PQ loads, fixed PV magnitudes, and one slack bus, without Q-limit switching.
"""
import argparse
import cmath
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

MATPOWER_REV = "b6c85b0323d20fb801cab3377c803a285d730847"
PYPOWER_REV = "c7f1cf50bedd34012b92fae4f092ec55617546a8"
CASES = ("case14", "case118", "case300", "case1354pegase", "case2383wp")
TERMINALS = ("p_from_pu", "q_from_pu", "p_to_pu", "q_to_pu")


def pinned(directory, rev, files):
    head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if head != rev:
        raise ValueError(f"{directory} must be checked out at {rev}")
    subprocess.run(["git", "-C", str(directory), "diff", "--quiet", "HEAD", "--", *files], check=True)


def source_case(path):
    """Read literal public MATPOWER matrices for the external solver."""
    source = path.read_text()
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.Ee+-]+)", source)[1])
    ppc = {"version": "2", "baseMVA": base}
    for name in ("bus", "branch", "gen"):
        body = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", source, re.S)[1]
        body = re.sub(r"%[^\n]*", "", body)
        ppc[name] = np.array([[float(v) for v in row.split()] for row in body.split(";") if row.strip()])
    return ppc


def oracle(ppc):
    from pypower.ppoption import ppoption
    from pypower.runpf import runpf

    case = copy.deepcopy(ppc)
    nonref = case["bus"][:, 1] != 3
    case["bus"][nonref, 7:9] = [1.0, 0.0]
    # Preserve the file's slack angle and all generator voltage setpoints.
    opts = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-11, PF_MAX_IT=30, ENFORCE_Q_LIMS=0)
    result, success = runpf(case, opts)
    if not success:
        raise ValueError("External flat-start power flow failed")
    voltages = {str(int(b[0])): {"vm_pu": b[7], "va_deg": b[8]} for b in result["bus"]}
    lines = {str(i): b[13:17] / ppc["baseMVA"] for i, b in enumerate(result["branch"]) if b[10] > 0}
    if not all(np.isfinite(v).all() for v in lines.values()):
        raise ValueError("Nonfinite external branch terminal")
    return voltages, lines


def checked_answer(run, spec, expected):
    start = time.perf_counter()
    actual = run(spec)
    sec = time.perf_counter() - start
    volts, lines = expected
    if set(actual.get("voltages", {})) != set(volts) or set(actual.get("lines", {})) != set(lines):
        raise AssertionError(f"Missing buses or lines: {actual.get('reason')}")

    def phasor(row):
        return cmath.rect(row["vm_pu"], math.radians(row["va_deg"]))

    dv = max(abs(phasor(actual["voltages"][k]) - phasor(v)) for k, v in volts.items())
    ds = max(abs(actual["lines"][k][field] - v[i]) for k, v in lines.items() for i, field in enumerate(TERMINALS))
    return {"verified": bool(actual.get("conservation_verified")), "seconds": round(sec, 5),
            "max_phasor_delta_pu": float(dv), "max_terminal_component_delta_pu": float(ds),
            "max_rectangular_residual_pu": actual.get("max_rectangular_mismatch_pu")}


def qualify(path, harness_case, run):
    ppc = source_case(path)
    spec = harness_case(path)
    spec["calculation"] = "ac_power_flow"
    # Ordinary PYPOWER PF uses unlimited PV generation unless enforcement is requested.
    for bus in spec["buses"]:
        bus.pop("q_max_injection_pu", None)
        bus.pop("q_min_injection_pu", None)
    expected = oracle(ppc)
    original = checked_answer(run, spec, expected)

    rotated_spec = copy.deepcopy(spec)
    slack = next(b for b in rotated_spec["buses"] if b["type"] == "slack")
    slack["va_deg"] = slack.get("va_deg", 0) + 17
    rotated_spec["buses"].reverse()
    rotated_spec["lines"].reverse()
    rotated_voltages = {k: {**row, "va_deg": row["va_deg"] + 17} for k, row in expected[0].items()}
    transformed = checked_answer(run, rotated_spec, (rotated_voltages, expected[1]))

    load = next(b for b in ppc["bus"] if b[1] == 1 and b[2] > 0 and b[3] > 0)
    key = str(int(load[0]))
    delta = 0.2 * load[2:4]
    load[2:4] += delta
    node = next(b for b in spec["buses"] if b["id"] == key)
    node["p_injection_pu"] -= float(delta[0] / ppc["baseMVA"])
    node["q_injection_pu"] -= float(delta[1] / ppc["baseMVA"])

    changed = checked_answer(run, spec, oracle(ppc))
    spec["proposed_voltages"] = expected[0]
    stale = run(spec)
    if not stale.get("ok"):
        raise AssertionError(f"Stale claim did not reach the physical verifier: {stale.get('reason')}")
    return {"case": path.stem, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "buses": len(spec["buses"]), "in_service_branches": len(spec["lines"]),
            "original": original, "rotated_and_reversed": transformed, "changed": changed, "changed_load_bus": key,
            "stale_claim_accepted": bool(stale["conservation_verified"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matpower", type=Path)
    parser.add_argument("pypower", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    paths = [Path("data") / (name + ".m") for name in args.cases]
    pinned(args.matpower, MATPOWER_REV, [str(path) for path in paths])
    pinned(args.pypower, PYPOWER_REV, ["pypower"])
    sys.path[:0] = [str(args.pypower.resolve()), str(args.repo.resolve()), str(args.repo.resolve() / "ship")]
    from verify_ac_regulation import matpower
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier

    reports = [qualify(args.matpower / path, matpower, run_router_conservation_cad_verifier) for path in paths]
    print(json.dumps({"matpower_revision": MATPOWER_REV, "pypower_revision": PYPOWER_REV,
                      "relative_load_increase": 0.2, "q_limits_enforced": False, "cases": reports}, indent=2))
    for row in reports:
        for state in ("original", "changed", "rotated_and_reversed"):
            checked = row[state]
            if not (checked["verified"] and checked["max_phasor_delta_pu"] < 1e-7
                    and checked["max_terminal_component_delta_pu"] < 1e-7):
                raise SystemExit(f"{row['case']} failed {state} verification")
        if row["stale_claim_accepted"]:
            raise SystemExit(f"{row['case']} accepted the obsolete operating point")


if __name__ == "__main__":
    main()
