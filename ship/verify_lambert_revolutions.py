"""Qualify the published multi-revolution Lambert tool against Gooding and Cartesian dynamics.

The caller supplies a checkout of jorgepiloto/lamberthub at ORACLE_SHA. Its
example table credits Der, The Superior Lambert Algorithm, example 1. Compare
the two returned velocity pairs without assuming Gooding's branch labels.
"""
import argparse
import asyncio
from itertools import permutations
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp


STOCK_SHA = "a3e4866b6970205e360e6420a48e3967d8ff187e"
ORACLE_SHA = "5010709b000bb6b29831ad628fa4b1ce1f519036"
BRANCHES = ("lower_x", "upper_x")
NODE = """
import register from './task-tools.ts';
import {readFileSync} from 'node:fs';
const tools = new Map();
register({registerTool(t) {tools.set(t.name,t)}, on() {}});
const tool = tools.get('hamon_conservation_cad_verifier');
const rows = [];
for (const [i,payload] of JSON.parse(readFileSync(0,'utf8')).entries()) {
    const result = await tool.execute('multirev-'+i,payload,null,()=>{},{});
    if (result.isError) throw Error(JSON.stringify(result));
    rows.push({details:result.details, text:JSON.parse(result.content[0].text)});
}
process.stdout.write(JSON.stringify(rows));
"""


def head(directory):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()


def registered(root, specs):
    calls = [{"mode": "lagrangian_orbital", "spec": spec} for spec in specs]
    with tempfile.TemporaryDirectory() as work:
        env = dict(os.environ, TASK_RELAY_URL="http://127.0.0.1:1", TASK_WORKDIR=work,
                   TASK_HARNESS_SRC=str(root), TASK_HARNESS_PYTHON=sys.executable)
        proc = subprocess.run(["node", "--experimental-strip-types", "--input-type=module",
                               "-e", NODE], cwd=root, env=env, input=json.dumps(calls),
                              text=True, capture_output=True, timeout=240)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        assert not list(Path(work).iterdir())
    rows = json.loads(proc.stdout)
    assert len(rows) == len(calls)
    assert all(row["text"] == row["details"] for row in rows)
    return [row["text"] for row in rows]


async def pi_round_trip(root, spec, expected):
    sys.path[:0] = [str(root), str(root / "tests"), str(root / "ship")]
    import replay_engineering_provider_wire as wire

    events = wire.tool_events
    digest = wire.digest
    returned = []
    def issue_call(model):
        for kind, body in events(model):
            if kind == "content_block_delta":
                body["delta"]["partial_json"] = json.dumps({"mode": "lagrangian_orbital", "spec": spec})
            yield kind, body

    def observe(data):
        if data.startswith(b"{"):
            obj = json.loads(data)
            if obj.get("subkind") == "lambert_transfer":
                returned.append(obj)
        return digest(data)

    with patch.object(wire, "tool_events", issue_call), patch.object(wire, "digest", observe):
        trial = await wire.replay(root, "multirev", "Evaluate the supplied orbital transfer.",
                                  True, None, "claude-opus-5-5")
    assert len(trial["requests"]) == 2 and returned == [expected], (trial, returned)
    return {"requests": 2, "tool_result_sha256": trial["requests"][1]["returned_reference_sha256"]}


def paired_error(actual, expected, scale):
    """Maximum endpoint vector error after the best bijection of the two complete solutions."""
    return min(max(float(np.linalg.norm(a - b)) / scale
                   for pair, idx in zip(actual, order)
                   for a, b in zip(pair, expected[idx]))
               for order in permutations(range(2)))


def propagate(spec, pair):
    """Replay the rounded velocities; independently count forward crossings of the start ray."""
    length = np.linalg.norm(spec["r1_m"])
    speed = math.sqrt(spec["mu"] / length)
    tof = spec["tof_s"] * speed / length
    r = np.asarray(spec["r1_m"]) / length
    target = np.asarray(spec["r2_m"]) / length
    v = np.asarray(pair[0]) / speed
    h = np.cross(r, v)
    momentum = np.linalg.norm(h)
    transverse = np.cross(h / momentum, r)
    short = spec["transfer_path"] == "short"
    assert (np.dot(h, np.cross(r, target)) > 0) == short

    def rhs(_, y):
        radius = np.linalg.norm(y[:3])
        return np.r_[y[3:6], -y[:3] / radius ** 3, momentum / radius ** 2]

    def start_ray(_, y):
        return np.dot(y[:3], transverse)
    start_ray.direction = 1

    count = spec["revolutions"]
    ivp = solve_ivp(rhs, (0, tof), np.r_[r, v, 0.0], method="DOP853",
                    rtol=3e-13, atol=2e-15, max_step=tof / (48 * (count + 1)),
                    events=start_ray)
    assert ivp.success, ivp.message
    final = ivp.y[:, -1]
    crossings = sum(t > 1e-9 * tof for t in ivp.t_events[0])
    angle = math.acos(float(np.clip(np.dot(r, target) / np.linalg.norm(target), -1, 1)))
    sweep = math.tau * count + (angle if short else math.tau - angle)
    replay = {"count": int(crossings), "sweep_error_rad": float(abs(final[-1] - sweep)),
              "position_relative": float(np.linalg.norm(final[:3] - target) / np.linalg.norm(target)),
              "velocity_scaled": float(np.linalg.norm(final[3:6] - pair[1] / speed))}
    assert crossings == count and max(replay[k] for k in
          ("sweep_error_rad", "position_relative", "velocity_scaled")) < 1e-6, replay
    return replay


def cases():
    der = dict(mu=3.986004418e14, r1_m=[22592145.603, -1599915.239, -19783950.506],
               r2_m=[1922067.697, 4054157.051, -8925727.465], tof_s=36000, revolutions=1)
    for path in ("short", "long"):
        yield "der", dict(der, subkind="lambert_transfer", transfer_path=path)
    for count in (2, 4, 32):
        for path, arc in (("short", math.pi / 2), ("long", 3 * math.pi / 2)):
            yield "inclined", dict(subkind="lambert_transfer", mu=1.0, r1_m=[1, 0, 0],
                                  r2_m=[0, -0.6, 0.8], tof_s=math.tau * count + arc,
                                  revolutions=count, transfer_path=path)


def qualify(args):
    root, oracle = args.root.resolve(), args.oracle_source.resolve()
    assert head(oracle) == ORACLE_SHA
    if args.stock:
        assert head(root) == STOCK_SHA
    sys.path.insert(0, str(oracle / "src"))
    from lamberthub import gooding1990
    assert Path(inspect.getfile(gooding1990)).resolve().is_relative_to(oracle / "src")
    published = runpy.run_path(str(oracle / "tests/test_multirev_solvers.py"))["TABLE_OF_TRANSFERS"]

    controls = list(cases())
    specs = [dict(spec, revolution_branch=b) for _, spec in controls for b in BRANCHES]
    infeasible = dict(controls[0][1], tof_s=18000)
    invalid = [dict(infeasible, revolution_branch=b) for b in BRANCHES]
    circular = dict(subkind="lambert_transfer", mu=1., r1_m=[1, 0, 0],
                    r2_m=[0, 1, 0], tof_s=math.pi / 2)
    rows = registered(root, specs + invalid + [circular])
    assert rows[-1]["verified"] and np.allclose(rows[-1]["v1_m_s"], [0, 1, 0], atol=1e-12)
    for row in rows[-3:-1]:
        assert row["verified"] is False and "v1_m_s" not in row and "v2_m_s" not in row, row
    for low in (False, True):
        try:
            gooding1990(infeasible["mu"], np.array(infeasible["r1_m"]), np.array(infeasible["r2_m"]),
                        infeasible["tof_s"], M=1, is_prograde=True, is_low_path=low)
        except ValueError as exc:
            assert "No feasible solution" in str(exc), str(exc)
        else:
            raise AssertionError("Gooding accepted the infeasible five-hour case")
    receipts = []
    for i, (geometry, spec) in enumerate(controls):
        r1, r2 = np.asarray(spec["r1_m"], dtype=float), np.asarray(spec["r2_m"], dtype=float)
        speed = math.sqrt(spec["mu"] / np.linalg.norm(r1))
        prograde = bool((np.cross(r1, r2)[2] > 0) == (spec["transfer_path"] == "short"))
        expected = [gooding1990(spec["mu"], r1, r2, spec["tof_s"], M=spec["revolutions"],
                                is_prograde=prograde, is_low_path=low) for low in (False, True)]
        assert np.linalg.norm(expected[0][0] - expected[1][0]) / speed > 1e-4
        two = rows[2*i:2*i+2]
        receipt = dict(geometry=geometry, spec=spec, registered=two)
        if args.stock:
            assert all(r["verified"] is False and "v1_m_s" not in r and "v2_m_s" not in r for r in two)
            replay_pairs = expected
        else:
            for branch, row in zip(BRANCHES, two):
                assert row["verified"] and row["endpoint_verified"] and row["revolution_count_verified"], row
                assert row["revolution_branch"] == branch and row["propagated_revolutions"] == spec["revolutions"]
            assert two[0]["root_parameter_x"] < two[1]["root_parameter_x"]
            replay_pairs = [[np.asarray(row[k]) for k in ("v1_m_s", "v2_m_s")] for row in two]
            receipt["gooding_scaled_error"] = paired_error(replay_pairs, expected, speed)
            assert receipt["gooding_scaled_error"] < 2e-8, receipt
        if geometry == "der":
            sense = "prograde" if prograde else "retrograde"
            table = [np.array(published[f"M1_{sense}_{suffix}"]) * 1000 for suffix in ("high", "low")]
            receipt["der_scaled_error"] = paired_error(replay_pairs, table, speed)
            assert receipt["der_scaled_error"] < 5e-8, receipt
        receipt["dop853"] = [propagate(spec, pair) for pair in replay_pairs]
        receipts.append(receipt)
    return dict(harness_sha=head(root), oracle_sha=ORACLE_SHA, stock=args.stock,
                registered_calls=len(rows), positive_cases=receipts, infeasible=rows[-3:-1],
                pi_loopback=asyncio.run(pi_round_trip(root, specs[0], rows[0])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--oracle-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock", action="store_true")
    options = parser.parse_args()
    report = qualify(options)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['harness_sha']} stock={report['stock']} {report['registered_calls']} calls, "
          f"{sum(len(c['dop853']) for c in report['positive_cases'])} propagated solutions")
