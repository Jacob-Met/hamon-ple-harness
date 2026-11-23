"""Independent connected-equilibrium and public-grid controls.

Usage: python ship/verify_ac_equilibria.py BASELINE_PY CANDIDATE_PY [CASE_M ...]
Add --require-equilibria when qualifying a candidate. No MATLAB code is executed.
Known witnesses establish feasibility; a different independently verified root
also qualifies. Agreement with the supplied witness is reported separately.
"""
import argparse
import cmath
import copy
import json
import math
from pathlib import Path

from verify_ac_regulation import load, matpower


def terminal_power(spec, voltages):
    """Sum explicit terminal currents; never construct the solver's Y matrix."""
    current = {b["id"]: complex(b.get("shunt_g_pu", 0), b.get("shunt_b_pu", 0))
               * voltages[b["id"]] for b in spec["buses"]}
    for line in spec["lines"]:
        f, t = line["from"], line["to"]
        tap = cmath.rect(line.get("tap_ratio", 1), math.radians(line.get("phase_shift_deg", 0)))
        source = voltages[f] / tap
        series = (source - voltages[t]) / complex(line["r_pu"], line["x_pu"])
        charging = 0.5j * line.get("line_charging_b_pu", 0)
        current[f] += (series + charging * source) / tap.conjugate()
        current[t] += -series + charging * voltages[t]
    return {key: value * current[key].conjugate() for key, value in voltages.items()}


def witness(groups, upper, heterogeneous=False):
    """Build input powers from a physical equilibrium before calling either solver."""
    buses = [{"id": "S", "type": "slack", "vm_pu": 1, "va_deg": 0}]
    lines, voltage, settings = [], {"S": 1 + 0j}, {}
    for group in range(groups):
        a, b = "A" + str(group), "B" + str(group)
        va = 0.94 + (0.002 * math.sin(group) if heterogeneous else 0)
        vb = (0.98 if upper else 1.04) + (0.003 * math.cos(group) if heterogeneous else 0)
        angle = (5 if upper else 2) + (0.2 * math.sin(group) if heterogeneous else 0)
        voltage[a], voltage[b] = cmath.rect(va, math.radians(angle)), cmath.rect(vb, -math.radians(angle))
        settings[a], settings[b] = va + (0.07 if upper else -0.07), vb
        buses.extend([{"id": a, "type": "PV"}, {"id": b, "type": "PV"}])
        for f, t, x in [("S", a, 3 if upper else 5), (a, b, 0.1), (b, "S", 3 if upper else 5)]:
            lines.append({"id": str(len(lines)), "from": f, "to": t, "r_pu": 0, "x_pu": x})
        if group:
            lines.append({"id": str(len(lines)), "from": "A" + str(group - 1),
                          "to": b if heterogeneous else a, "r_pu": 0.01, "x_pu": 0.2})
    spec = {"buses": buses, "lines": lines}
    power = terminal_power(spec, voltage)
    for bus in buses[1:]:
        key = bus["id"]
        q = power[key].imag
        if key.startswith("A"):
            lower, higher = (q - 1.5, q) if upper else (q, q + 1.5)
        else:
            lower, higher = (q - 0.001, q + 1.5) if upper else (q - 1.5, q + 0.001)
        bus.update(vm_pu=settings[key], p_injection_pu=power[key].real,
                   q_min_injection_pu=lower, q_max_injection_pu=higher)
    spec["proposed_voltages"] = {key: {"vm_pu": abs(v), "va_deg": math.degrees(cmath.phase(v))}
                                 for key, v in voltage.items()}
    return spec


def transform(spec, angle, reverse):
    result = copy.deepcopy(spec)
    for bus in result["buses"]:
        if bus["type"] == "slack":
            bus["va_deg"] = bus.get("va_deg", 0) + angle
    for claim in result.get("proposed_voltages", {}).values():
        claim["va_deg"] += angle
    if reverse:
        result["buses"].reverse()
        result["lines"].reverse()
    return result


def proposed_residual(spec):
    values = {key: cmath.rect(row["vm_pu"], math.radians(row["va_deg"]))
              for key, row in spec["proposed_voltages"].items()}
    power = terminal_power(spec, values)
    errors = []
    for bus in spec["buses"]:
        if bus["type"] == "slack":
            continue
        key = bus["id"]
        errors.append(abs(power[key].real - bus["p_injection_pu"]))
        lower, upper = bus["q_min_injection_pu"], bus["q_max_injection_pu"]
        q = power[key].imag
        errors.append(max(lower - q, q - upper, 0))
        if key.startswith("A"):
            bound = upper if bus["vm_pu"] > abs(values[key]) else lower
            errors.append(abs(q - bound))
        else:
            errors.append(abs(abs(values[key]) - bus["vm_pu"]))
    return max(errors, default=0)


def phasors(answer):
    return {key: cmath.rect(row["vm_pu"], math.radians(row["va_deg"]))
            for key, row in answer.get("voltages", {}).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("cases", type=Path, nargs="*")
    parser.add_argument("--require-equilibria", action="store_true")
    args = parser.parse_args()
    modules = {name: load("equilibrium_" + name, path)
               for name, path in (("baseline", args.baseline), ("candidate", args.candidate))}
    output = {"witnesses": [], "public": []}
    transformations = [(0, False), (17, False), (17, True)]
    for heterogeneous in (False, True):
        for groups in (3, 6):
            for upper in (False, True):
                original = witness(groups, upper, heterogeneous)
                for angle, reverse in transformations:
                    spec = transform(original, angle, reverse)
                    residual = proposed_residual(spec)
                    assert residual < 1e-12, (groups, upper, residual)
                    row = {"generators": 2 * groups, "upper": upper,
                           "heterogeneous": heterogeneous, "rotation": angle,
                           "reversed": reverse, "physical_witness_residual": residual}
                    for name, module in modules.items():
                        plain = copy.deepcopy(spec)
                        del plain["proposed_voltages"]
                        actual = module.verify_ac_power_flow(plain)
                        claimed = module.verify_ac_power_flow(spec)
                        bad = copy.deepcopy(spec)
                        self_claim = None
                        if actual["conservation_verified"]:
                            bad["proposed_voltages"] = copy.deepcopy(actual["voltages"])
                            self_claim = module.verify_ac_power_flow(bad)["conservation_verified"]
                        bad["proposed_voltages"]["A0"]["va_deg"] += 0.1
                        wrong = module.verify_ac_power_flow(bad)
                        row[name] = {"native_verified": actual["conservation_verified"],
                                     "witness_verified": claimed["conservation_verified"],
                                     "self_claim_verified": self_claim,
                                     "wrong_claim_accepted": wrong["conservation_verified"],
                                     "reason": actual.get("reason"),
                                     "rectangular_residual": actual.get("max_rectangular_mismatch_pu")}
                    output["witnesses"].append(row)
    for path in args.cases:
        original = matpower(path)
        references = {name: phasors(module.verify_ac_power_flow(original))
                      for name, module in modules.items()}
        for angle, reverse in transformations:
            spec = transform(original, angle, reverse)
            row = {"case": path.name, "buses": len(spec["buses"]), "rotation": angle, "reversed": reverse}
            for name, module in modules.items():
                actual = module.verify_ac_power_flow(spec)
                v = phasors(actual)
                expected = {key: value * cmath.exp(1j * math.radians(angle)) for key, value in references[name].items()}
                change = max((abs(v[key] - value) for key, value in expected.items()), default=None) if v else None
                negative = None
                if v:
                    bad = copy.deepcopy(spec)
                    bad["proposed_voltages"] = copy.deepcopy(actual["voltages"])
                    key = next(b["id"] for b in spec["buses"] if b["type"] != "slack")
                    bad["proposed_voltages"][key]["vm_pu"] += 0.001
                    negative = module.verify_ac_power_flow(bad)["conservation_verified"]
                row[name] = {"verified": actual["conservation_verified"],
                             "rotation_order_error": change, "wrong_claim_accepted": negative,
                             "terminal_residual": actual.get("tellegen_residual_pu"),
                             "rectangular_residual": actual.get("max_rectangular_mismatch_pu"),
                             "reason": actual.get("reason")}
            output["public"].append(row)
    output["summary"] = {name: {
        "witnesses": sum(row[name]["witness_verified"] for row in output["witnesses"]),
        "native_equilibria": sum(row[name]["native_verified"] for row in output["witnesses"]),
        "self_claims_verified": sum(bool(row[name]["self_claim_verified"]) for row in output["witnesses"]),
        "witness_total": len(output["witnesses"]),
        "public_verified": sum(row[name]["verified"] for row in output["public"]),
        "public_total": len(output["public"]),
        "wrong_claims_accepted": sum(bool(row[name]["wrong_claim_accepted"]) for rows in (output["witnesses"], output["public"]) for row in rows),
    } for name in modules}
    print(json.dumps(output, indent=2))
    if args.require_equilibria:
        result = output["summary"]["candidate"]
        assert result["native_equilibria"] == result["witness_total"]
        assert result["self_claims_verified"] == result["witness_total"]
        assert result["public_verified"] == result["public_total"]
        assert result["wrong_claims_accepted"] == 0
        assert all(row["candidate"]["rotation_order_error"] is not None
                   and row["candidate"]["rotation_order_error"] < 1e-6
                   for row in output["public"])


if __name__ == "__main__":
    main()
