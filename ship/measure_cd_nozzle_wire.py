"""Paired real Pi loopback requests on the historical archive and NASA CDV controls."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def nozzle(back=.75):
    return json.dumps({"mode": "compressible_acoustics", "spec": {"subkind": "cd_nozzle_shock",
                       "gamma": 1.4, "exit_to_throat_area": 1.5, "back_to_total_pressure": back}})


async def covered(instruction):
    import harness
    import taskbridge
    from test_direct_json_delivery import exchange, FALLBACK
    from mock_sse_proxy import MockStream
    from websockets.asyncio.server import serve

    async def entry(ws):
        await taskbridge._session(ws, harness.solve)

    stream = MockStream(reply_text=FALLBACK)
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        folder = Path(tmp) / "task"
        home.mkdir()
        folder.mkdir()
        async with await asyncio.start_server(stream.respond, "127.0.0.1", 0) as model:
            url = f"http://127.0.0.1:{model.sockets[0].getsockname()[1]}"
            env = {"HOME": str(home), "PATH": os.environ["PATH"], "NO_PROXY": "127.0.0.1,localhost",
                   "MODEL_NAME": "claude-sonnet-4-5", "ANTHROPIC_API_KEY": "fixture", "ANTHROPIC_BASE_URL": url}
            with patch.dict(os.environ, env, clear=True):
                async with serve(entry, "127.0.0.1", 0) as server:
                    answer, counts = await exchange(server.sockets[0].getsockname()[1], folder, instruction)
    result = None if answer == FALLBACK else json.loads(answer)
    return {"requests": stream.requests, "counts": counts, "terminal_result": result}


def record(root, shard):
    sys.path[:0] = [str(root), str(root / "tests"), str(root / "ship")]
    from replay_engineering_provider_wire import replay
    tasks = json.loads((root / "artifacts/manifest.json").read_bytes())["tasks"]
    if len(tasks) != 29 or len(set(tasks)) != 29:
        raise ValueError("Expected exactly 29 archived public instructions")
    corpus = {}
    for index, task in enumerate(sorted(tasks)):
        if index % 2 == shard:
            steps = json.loads((root / "artifacts" / task / "trajectory.json").read_bytes())["steps"]
            corpus[task] = next(step["message"] for step in steps if step["source"] == "user")
    if shard == 0:
        corpus.update(nasa_cdv_bare=nozzle(), nasa_cdv_shock_free=nozzle(.89))
    rows = {}
    for label, text in corpus.items():
        rows[label] = asyncio.run(replay(root, label, text, False, None, "claude-opus-5-5"))
    if shard == 1:
        text = "Run the calculation below and return its JSON result.\n" + nozzle()
        rows["nasa_cdv_covered"] = {"instruction_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                    **asyncio.run(covered(text))}
    return rows


def compare(path):
    arms = {}
    for arm in ("stock", "candidate"):
        first, second = (json.loads((path / f"{arm}-{i}.json").read_bytes()) for i in (0, 1))
        assert first.keys().isdisjoint(second)
        arms[arm] = {**first, **second}
    stock, candidate = arms["stock"], arms["candidate"]
    assert stock.keys() == candidate.keys() and len(stock) == 32
    summary = {}
    for label in sorted(stock):
        a, b = stock[label], candidate[label]
        assert a["instruction_sha256"] == b["instruction_sha256"]
        if label == "nasa_cdv_covered":
            assert (a["requests"], b["requests"]) == (1, 0)
            assert a["terminal_result"] is None and b["counts"] == {"probes": 1, "writes": 0, "usage": 0}
            result = b["terminal_result"]
            assert result["verified"] and result["residual"] < 1e-12
            x = 5*(1+math.acos(4*(1.25-result["shock_area_over_throat"]))/math.pi)
            assert 7.562273 < x < 7.562293
            summary[label] = {"stock_requests": 1, "candidate_requests": 0, "shock_x_in": x,
                              "max_residual": result["residual"], "candidate_counts": b["counts"]}
            continue
        (ar,), (br,) = a["requests"], b["requests"]
        assert ar["tool_names"] == br["tool_names"]
        if label == "nasa_cdv_bare":
            assert not ar["presolved"] and br["presolved"]
            assert ar["output_config"]["effort"] != "low" and br["output_config"]["effort"] == "low"
        else:
            assert (ar["thinking"], ar["output_config"], ar["presolved"]) == (br["thinking"], br["output_config"], br["presolved"])
        summary[label] = {"stock_effort": ar["output_config"], "candidate_effort": br["output_config"],
                          "stock_bytes": ar["request_bytes"], "candidate_bytes": br["request_bytes"]}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = compare(args.compare) if args.compare else record(args.root.resolve(), args.shard)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(data, sort_keys=True))
