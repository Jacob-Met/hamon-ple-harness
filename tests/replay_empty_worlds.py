"""Replay all public prompts over real Pi SSE with a deliberately empty evaluator.

Run with the submitted Pi installed: python tests/replay_empty_worlds.py
This adversarial false-completion control never executes participant commands or
writes task artifacts. It measures the terminal contract, not engineering scores.
"""
import asyncio
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
from ship.audit_ple_delivery import task_workdir


CLAIM = "I delivered every file."


async def run_case(port, folder):
    instruction = json.loads((folder / "trajectory.json").read_bytes())["steps"][0]["message"]
    exports = ["/" + str(p.relative_to(folder / "files")) for p in (folder / "files").rglob("*") if p.is_file()]
    cwd = task_workdir(instruction, exports)
    targets = delivery_guard.extract_targets(instruction, cwd)
    terminal = delivery_guard.TerminalDelivery(instruction, targets)
    terminal.add_working_copy(cwd)
    if not terminal.paths:
        raise ValueError(f"{folder.name}: public deliverable contract was not recognized")
    probed = set()
    async with connect(f"ws://127.0.0.1:{port}") as socket:
        await socket.send(json.dumps({"type": "task", "instruction": instruction, "workdir": cwd,
                                      "limits": {"agent_timeout_sec": 400}}))
        async with asyncio.timeout(380):
            async for frame in socket:
                msg = json.loads(frame)
                if msg["type"] == "done":
                    summary = msg.get("final_message", "")
                    assert CLAIM not in summary and all(p in summary for p in terminal.paths), (folder.name, summary)
                    assert set(terminal.paths) <= probed, (folder.name, probed)
                    return {"task": folder.name, "guarded_paths": len(terminal.paths), "done": True}
                if msg["type"] == "error":
                    raise RuntimeError((folder.name, msg))
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
    raise RuntimeError(f"{folder.name}: no terminal message")


async def main(folders):
    stream = MockStream(reply_text=CLAIM)
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
                    for i in range(0, len(folders), 2):
                        results = await asyncio.gather(*(run_case(port, f) for f in folders[i:i + 2]))
                        for row in results:
                            print(json.dumps(row), flush=True)
    assert stream.requests >= len(folders)
    print(json.dumps({"prompts": len(folders), "sse_requests": stream.requests}))


if __name__ == "__main__":
    cases = sorted(p.parent for p in (ROOT / "artifacts").glob("*/trajectory.json"))
    if sys.argv[1:]:
        cases = [p for p in cases if p.name in sys.argv[1:]]
    if not cases:
        raise SystemExit("No matching public prompts.")
    asyncio.run(main(cases))
