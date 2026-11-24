"""Record genuine Pi loopback HTTP requests for public archive prompts and jacket controls."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from replay_engineering_provider_wire import replay


def controls():
    common = {"reactor_type": "cstr", "order": 1, "C_A0": 1000, "v0": .1 / 60,
              "volume": .1, "T0": 350, "delta_T_adiabatic": 50000 / 239,
              "A_preexp": 7.2e10 / 60, "E_a": 72750 * (8.31446261815324 / 8.314),
              "steady_state_branch": "all"}
    heat = {"UA_W_K": 50000 / 60, "rho_kg_m3": 1000, "Cp_J_kg_K": 239, "coolant_T_K": 300}
    for label, spec in (
        ("dimensional_all_three", {**common, "heat_transfer": heat}),
        ("dimensional_ambiguous", {**common, "heat_transfer": heat, "steady_state_branch": None}),
        ("dimensional_unknown_jacket_inventory", {**common, "heat_transfer": {**heat, "volume_m3": 1}}),
    ):
        yield label, json.dumps({"mode": "reactor_kinetics", "spec": spec})


def record(root):
    sys.path[:0] = [str(root), str(root / "tests")]
    task_ids = json.loads((root / "artifacts/manifest.json").read_bytes())["tasks"]
    assert len(task_ids) == 29
    prompts = {}
    for task in task_ids:
        steps = json.loads((root / "artifacts" / task / "trajectory.json").read_bytes())["steps"]
        prompts[task] = next(s["message"] for s in steps if s["source"] == "user")
    prompts.update(controls())
    for label, prompt in prompts.items():
        result = asyncio.run(replay(root, label, prompt, False, None, "claude-opus-5-5"))
        print(json.dumps(result, sort_keys=True), flush=True)


def compare(paths):
    def read(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 32 and len({r["label"] for r in rows}) == len(rows)
        assert all(len(r["requests"]) == 1 for r in rows)
        return {r["label"]: r for r in rows}
    stock, candidate = map(read, paths)
    assert stock.keys() == candidate.keys()
    summary = {}
    for label in stock:
        assert stock[label]["instruction_sha256"] == candidate[label]["instruction_sha256"]
        a, b = (side[label]["requests"][0] for side in (stock, candidate))
        assert a["tool_names"] == b["tool_names"]
        if label == "dimensional_all_three":
            assert not a["presolved"] and b["presolved"]
            assert a["output_config"]["effort"] != "low" and b["output_config"]["effort"] == "low"
        else:
            assert (a["thinking"], a["output_config"], a["presolved"]) == (b["thinking"], b["output_config"], b["presolved"])
        summary[label] = {"bytes_before": a["request_bytes"], "bytes_after": b["request_bytes"],
                          "stock_effort": a["output_config"], "candidate_effort": b["output_config"]}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root", type=Path)
    mode.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.root:
        record(args.root.resolve())
    else:
        compare(args.compare)
