"""Compare received loopback requests from the 29 public examples against the pinned full-guide baseline."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = "3efe0f4b626d15fe4dc01a4ad36d3429fd534327"
FULL = "d0a5d7315453152c60f50948df11c9215354cd1443aae83fc4bd65d6a292dd8f"
POSITIVE = "synthetic-zinc-cell"
INCOMPLETE = "synthetic-zinc-incomplete"
REFERENCE = "synthetic-reference"


def receive(path, labels):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [row["label"] for row in rows] != labels:
        raise ValueError(f"{path}: missing, duplicated or out-of-order cases")
    for row in rows:
        if len(row["requests"]) != (2 if row["label"] == REFERENCE else 1):
            raise ValueError(f"{row['label']}: request count changed")
    return {row["label"]: row for row in rows}


def check(condition, label):
    if not condition:
        raise ValueError(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    args = parser.parse_args()
    tasks = list(json.loads((args.archive / "manifest.json").read_bytes())["tasks"])
    check(len(tasks) == len(set(tasks)) == 29, "public task roster")
    both = tasks + [POSITIVE, INCOMPLETE]
    stock = receive(args.stock, both)
    candidate = receive(args.candidate, both + [REFERENCE])
    differences = []
    stock_names = stock[tasks[0]]["requests"][0]["tool_names"]
    check(len(stock_names) == len(set(stock_names)) and "hamon_conservation_cad_verifier" in stock_names, "stock tools")
    for key in both:
        a, b = stock[key], candidate[key]
        check(a["instruction_sha256"] == b["instruction_sha256"], f"{key}: instruction drift")
        x, y = a["requests"][0], b["requests"][0]
        check(x["tool_names"] == y["tool_names"] == stock_names, f"{key}: tools")
        check(x["guide_sha256"] == FULL and x["guide_bytes"] == 11872, f"{key}: stock guide")
        check(x["max_tokens"] == y["max_tokens"], f"{key}: output limit")
        if key == POSITIVE:
            check(x["presolved"] is False and y["presolved"] is True
                  and (x["budget"], y["budget"]) == (8192, 2048), f"{key}: positive control")
        else:
            check(x["presolved"] is False and y["presolved"] is False and x["budget"] == y["budget"],
                  f"{key}: historical or incomplete request")
        if key in tasks:
            history = json.loads((args.archive / key / "trajectory.json").read_bytes())
            original = next(step["message"] for step in history["steps"] if step["source"] == "user")
            check(hashlib.sha256(original.encode()).hexdigest() == a["instruction_sha256"], f"{key}: original prompt")
            delta = {unit: x[unit] - y[unit] for unit in ("request_bytes", "request_bpe")}
            check(min(delta.values()) > 0, f"{key}: no measured first-request reduction")
            differences.append(delta)
    first, second = candidate[REFERENCE]["requests"]
    check(first["tool_names"] == second["tool_names"] == stock_names
          and first["guide_sha256"] == second["guide_sha256"] == candidate[tasks[0]]["requests"][0]["guide_sha256"],
          "reference route")
    check(second["returned_reference_sha256"] == FULL and second["returned_reference_bytes"] == 11872,
          "reference payload differs from the registered original")
    summary = {unit: {"min": min(d[unit] for d in differences), "max": max(d[unit] for d in differences),
                      "sum": sum(d[unit] for d in differences)} for unit in differences[0]}
    print(json.dumps(dict(stock_commit=BASE, historical_requests=len(tasks), initial_request_savings=summary,
                          positive_budget=[8192, 2048], reference_round_trip_requests=candidate[REFERENCE]["requests"],
                          measurement="Captured localhost Pi HTTP body sizes and cl100k_base token proxy; "
                                      "responses are fixtures. No model billing or task-solution scores."), sort_keys=True))


if __name__ == "__main__":
    main()
