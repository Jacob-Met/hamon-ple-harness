"""Compare AC implementations on public MATPOWER cases and coupled-limit controls.

Usage: python ship/verify_ac_regulation.py BASELINE_PY CASE14_M CASE30_M
Public-case source: MATPOWER/matpower@b6c85b0323d20fb801cab3377c803a285d730847.
The case parser reads literal numeric matrices; it does not execute MATLAB.
"""
import argparse
import cmath
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import time
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline", type=Path)
parser.add_argument("cases", type=Path, nargs="+")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

baseline = load("ac_baseline", args.baseline)
candidate = load("hamon_ac_power_flow", root / "hamon_ac_power_flow.py")
controls = load("ac_controls", root / "tests/test_ac_power_flow.py")


def matpower(path):
    source = path.read_text()
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", source)[1])
    def matrix(name):
        section = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", source, re.S)[1]
        section = re.sub(r"%[^\n]*", "", section)
        return [[float(v) for v in row.split()] for row in section.split(";") if row.strip()]
    raw_buses, generators, branches = map(matrix, ("bus", "gen", "branch"))
    buses = []
    for row in raw_buses:
        units = [g for g in generators if g[0] == row[0] and g[7] > 0]
        typ = {1: "PQ", 2: "PV", 3: "slack"}[int(row[1])]
        bus = {"id": str(int(row[0])), "type": typ,
               "shunt_g_pu": row[4] / base, "shunt_b_pu": row[5] / base}
        if typ != "slack":
            bus["p_injection_pu"] = (sum(g[1] for g in units) - row[2]) / base
        if typ == "PQ":
            bus["q_injection_pu"] = (sum(g[2] for g in units) - row[3]) / base
        else:
            bus["vm_pu"] = units[0][5] if units else row[7]
        if typ == "slack":
            bus["va_deg"] = row[8]
        if typ == "PV":
            bus["q_min_injection_pu"] = (sum(g[4] for g in units) - row[3]) / base
            bus["q_max_injection_pu"] = (sum(g[3] for g in units) - row[3]) / base
        buses.append(bus)
    lines = [{"id": str(i), "from": str(int(r[0])), "to": str(int(r[1])),
              "r_pu": r[2], "x_pu": r[3], "line_charging_b_pu": r[4],
              "tap_ratio": r[8] or 1, "phase_shift_deg": r[9]}
             for i, r in enumerate(branches) if r[10] > 0]
    return {"buses": buses, "lines": lines}


def timed(module, spec):
    start = time.perf_counter()
    result = module.verify_ac_power_flow(copy.deepcopy(spec))
    return result, time.perf_counter() - start


rows = []
for path in args.cases:
    name = path.name
    spec = matpower(path)
    before, bt = timed(baseline, spec)
    after, at = timed(candidate, spec)
    delta = None
    if before.get("voltages") and after.get("voltages"):
        def voltage(row):
            return cmath.rect(row["vm_pu"], math.radians(row["va_deg"]))
        delta = max(abs(voltage(before["voltages"][k]) - voltage(after["voltages"][k]))
                    for k in before["voltages"])
    rows.append({"case": name, "buses": len(spec["buses"]),
                 "baseline_verified": before["conservation_verified"],
                 "candidate_verified": after["conservation_verified"],
                 "baseline_seconds": bt, "candidate_seconds": at,
                 "max_voltage_phasor_change_pu": delta,
                 "candidate_terminal_power_residual": after.get("tellegen_residual_pu"),
                 "candidate_current_balance_residual": after.get("max_rectangular_mismatch_pu"),
                 "baseline_reason": before.get("reason"), "candidate_reason": after.get("reason")})

samples = []
def capture(spec):
    samples.append(copy.deepcopy(spec))
    return candidate.verify_ac_power_flow(spec)

controls.verify_ac_power_flow = capture
positive = []
for resistance in (10, 100):
    for angle, reverse in ((0, False), (17, False), (17, True)):
        samples.clear()
        controls.test_cross_tie_requires_joint_mixed_reactive_limits(resistance, angle, reverse)
        positive.append(samples[1])
for count in (5, 12, 24):
    samples.clear()
    controls.test_mixed_limits_on_one_connected_generator_chain(count)
    positive.append(samples[0])
bad_claims = []
for upper in (False, True):
    for angle in (0, 17):
        samples.clear()
        controls.test_saturated_claim_requires_reactive_power_at_its_bound(upper, angle)
        bad_claims.append(samples[1])
for label, module in (("baseline", baseline), ("candidate", candidate)):
    run = [timed(module, spec) for spec in positive]
    rows.append({"implementation": label, "connected_cases": len(positive),
                 "verified": sum(r["conservation_verified"] for r, _ in run),
                 "seconds": sum(t for _, t in run),
                 "worst_generators": 24,
                 "incorrect_saturation_claims_accepted": sum(
                     module.verify_ac_power_flow(spec)["conservation_verified"] for spec in bad_claims),
                 "bad_claim_count": len(bad_claims)})
print(json.dumps(rows, indent=2))
