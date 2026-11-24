"""Replay all public prompts over real Pi SSE with a deliberately empty evaluator.

Run with the submitted Pi installed: python tests/replay_empty_worlds.py
This adversarial false-completion control never executes participant commands or
writes task artifacts. It measures the terminal contract, not engineering scores.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from unittest import mock

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import delivery_guard
import harness
from mock_sse_proxy import MockStream


CLAIM = "I delivered every file."


PROMPT_COURSE = json.loads((Path(__file__).with_name("archived_prompt_course.json")).read_bytes())


def load_cases(artifacts, requested=()):
    """Resolve the complete archived course or an exact, explicitly requested subset."""
    roster = PROMPT_COURSE["prompts"]
    available = {path.parent.name: path for path in artifacts.glob("*/trajectory.json")}
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate requested task ID")
    selected = set(requested) if requested else set(roster)
    unknown = selected - roster.keys() if requested else available.keys() - roster.keys()
    missing = selected - available.keys()
    if unknown or missing:
        raise ValueError(f"archived prompt roster mismatch: unknown={sorted(unknown)}, missing={sorted(missing)}")
    cases = []
    for task in sorted(selected):
        trace = json.loads(available[task].read_bytes())
        steps = trace.get("steps") if isinstance(trace, dict) else None
        first = steps[0] if isinstance(steps, list) and steps else None
        if (not isinstance(first, dict) or first.get("source") != "user"
                or type(first.get("step_id")) is not int or first["step_id"] != 1
                or not isinstance(first.get("message"), str)):
            raise ValueError(f"{task}: first trajectory step is not the archived user prompt")
        message = first["message"]
        if hashlib.sha256(message.encode("utf-8")).hexdigest() != roster[task]["sha256"]:
            raise ValueError(f"{task}: first user prompt digest differs from the archive")
        cases.append((task, message, roster[task]["workdir"]))
    return cases


async def run_case(port, task, instruction, cwd):
    targets = delivery_guard.extract_targets(instruction, cwd)
    terminal = delivery_guard.TerminalDelivery(instruction, targets)
    terminal.add_working_copy(cwd)
    if not terminal.paths:
        raise ValueError(f"{task}: public deliverable contract was not recognized")
    probed = set()
    async with connect(f"ws://127.0.0.1:{port}") as socket:
        await socket.send(json.dumps({"type": "task", "instruction": instruction, "workdir": cwd,
                                      "limits": {"agent_timeout_sec": 400}}))
        async with asyncio.timeout(380):
            async for frame in socket:
                msg = json.loads(frame)
                if msg["type"] == "done":
                    summary = msg.get("final_message", "")
                    assert CLAIM not in summary and all(p in summary for p in terminal.paths), (task, summary)
                    assert set(terminal.paths) <= probed, (task, probed)
                    return {"task": task, "guarded_paths": len(terminal.paths),
                            "done": True, "summary": summary}
                if msg["type"] == "error":
                    raise RuntimeError((task, msg))
                if "id" not in msg:
                    continue
                # A missing file fact is supplied only for the registered read-only
                # status probe. No shell command from the participant is executed.
                reply = {"type": "result", "id": msg["id"], "ok": False, "error": "empty fixture"}
                if msg["type"] == "exec":
                    parts = shlex.split(msg["command"])
                    if len(parts) == 4 and parts[:3] == ["python3", "-c", delivery_guard.REMOTE_PROBE]:
                        paths = [t["path"] for t in json.loads(parts[3])]
                        probed.update(paths)
                        reply = {"type": "result", "id": msg["id"], "ok": True, "exit_code": 0,
                                 "output": json.dumps({p: {"state": "missing"} for p in paths})}
                await socket.send(json.dumps(reply))
    raise RuntimeError(f"{task}: no terminal message")


async def replay(cases):
    stream = MockStream(reply_text=CLAIM)
    observations = {}
    async def entry(socket):
        await harness.taskbridge._session(socket, harness.solve)
    with tempfile.TemporaryDirectory() as home:
        async with await asyncio.start_server(stream.respond, "127.0.0.1", 0) as proxy:
            url = f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
            env = {"PATH": os.environ["PATH"], "HOME": home, "NO_PROXY": "127.0.0.1,localhost",
                   "MODEL_API": "google-cloud/claude-opus-5-5", "MODEL_NAME": "claude-opus-5-5",
                   "ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "fixture",
                   "MODEL_BASE_URL": url + "/v1", "MODEL_API_KEY": "fixture"}
            with mock.patch.dict(os.environ, env, clear=True):
                async with serve(entry, "127.0.0.1", 0) as server:
                    port = server.sockets[0].getsockname()[1]
                    for i in range(0, len(cases), 2):
                        results = await asyncio.gather(*(run_case(port, *case) for case in cases[i:i + 2]))
                        for row in results:
                            observations[row["task"]] = row
                            print(json.dumps({k:v for k,v in row.items() if k != "summary"}), flush=True)
    assert stream.requests >= len(cases)
    return {"prompts": len(cases), "sse_requests": stream.requests, "observations": observations}


async def main(cases):
    # Compare the ordinary model path and the direct-answer candidate through
    # the same real Pi/SSE/evaluator protocol. These public prompts have other
    # work and delivery obligations; none is a closed inline calculation.
    with mock.patch.object(harness, "_certified_inline_reply", return_value=(None, None)):
        baseline = await replay(cases)
    direct_admissions = 0
    original = harness._certified_inline_reply
    async def observe(*args, **kwargs):
        nonlocal direct_admissions
        result = await original(*args, **kwargs)
        if result[1] is not None:
            direct_admissions += 1
        return result
    with mock.patch.object(harness, "_certified_inline_reply", side_effect=observe):
        candidate = await replay(cases)
    assert baseline["observations"] == candidate["observations"]
    assert baseline["sse_requests"] == candidate["sse_requests"]
    assert direct_admissions == 0
    print(json.dumps({"public_prompts": len(cases),
                      "archive_commit": PROMPT_COURSE["archive_commit"],
                      "full_course": len(cases) == len(PROMPT_COURSE["prompts"]),
                      "task_ids": [task for task, _, _ in cases],
                      "baseline_sse_requests": baseline["sse_requests"],
                      "candidate_sse_requests": candidate["sse_requests"],
                      "terminal_summaries_equal": True,
                      "direct_admissions": direct_admissions}), flush=True)


if __name__ == "__main__":
    try:
        cases = load_cases(ROOT / "artifacts", sys.argv[1:])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    asyncio.run(main(cases))
