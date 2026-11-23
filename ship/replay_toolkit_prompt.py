"""Capture the first real Pi request with the toolkit provisioned into a local fixture.

Run once per checkout with --root and compare the resulting JSONL. An answer from a
loopback SSE fixture ends each trial. No deliverables, model accuracy or billed cost
are measured. The candidate's restored hash must equal the stock request hash.
"""
import argparse
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

HEADER = "\nRun each indented example as `python3 /tmp/hamon-engineering-toolkit.py` followed by the arguments shown."
COMMAND = "python3 /tmp/hamon-engineering-toolkit.py "
END = "every engineering calculation and deliverable."


def restored(text):
    if HEADER not in text:
        return text
    start, rest = text.split(HEADER, 1)
    guide, end, suffix = rest.partition(END)
    if not end:
        raise ValueError("Missing end of provisioned command guide")
    return start + guide.replace("\n  ", "\n  " + COMMAND) + end + suffix


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


async def run(root, task, short, count_tokens):
    import engineering_intake
    import harness
    from fin_provider_replay import events

    folder = root / "artifacts" / task
    instruction = json.loads((folder / "trajectory.json").read_bytes())["steps"][0]["message"]
    captured = {}
    raw_request = b""
    runtime_cwd = ""
    async def request(reader, writer):
        nonlocal raw_request
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode().split("\r\n")
            method, path, _ = headers[0].split()
            assert method == "POST" and path.partition("?")[0] == "/v1/messages"
            length = next(int(h.split(":", 1)[1]) for h in headers if h.lower().startswith("content-length:"))
            raw = await reader.readexactly(length)
            raw_request = raw
            body = json.loads(raw)
            assert body["stream"] and not captured
            content = body["messages"][0]["content"]
            prompt = content if isinstance(content, str) else "\n".join(
                part["text"] for part in content if part["type"] == "text")
            assert instruction in prompt
            # Pi writes the ephemeral private scratch cwd into its own system
            # text. Normalize only the generated basename, then restore the
            # invocation prefix before comparing complete structured requests.
            normalized = json.loads(json.dumps(body))
            prefixes = 0
            guide_count = 0
            for part in normalized["system"]:
                value = part["text"]
                guide_count += value.count("DETERMINISTIC ENGINEERING TOOLS (in the task container")
                prefixes += value.count(COMMAND.rstrip())
                value = value.replace(runtime_cwd, "/tmp/pi-FIXTURE/work")
                part["text"] = restored(value)
            assert guide_count == (0 if short else 1), (task, guide_count)
            captured.update(task=task, raw_bytes=length, cli_path_mentions=prefixes,
                            restored_request_sha256=digest(normalized), tools=len(body["tools"]),
                            thinking=body.get("thinking"), max_tokens=body["max_tokens"])
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            for name, data in events(body["model"], "Fixture response"):
                message = json.dumps({"type": name, **data})
                writer.write(f"event: {name}\ndata: {message}\n\n".encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    class FirstCallComplete(Exception):
        pass
    normal_process = harness.taskbridge.run_process
    async def first_call(*args, **kwargs):
        nonlocal runtime_cwd
        runtime_cwd = kwargs["cwd"]
        finished = await normal_process(*args, **kwargs)
        if not captured:
            raise RuntimeError(f"Pi exited before sending its first request: {finished.stderr[-500:]}")
        raise FirstCallComplete()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        async def write(path, data):
            if short:
                return len(data) - 1
            target = work / Path(path).name
            return target.write_bytes(data)
        async def execute(command, **kwargs):
            if engineering_intake.REMOTE_PROBE in shlex.split(command):
                record = dict(schema="hamon.ple.intake.v1", scanned_files=0, engineering_file_count=0,
                              by_extension={}, files=[], metadata={}, warnings=[])
                return dict(exit_code=0, output=json.dumps(record))
            return dict(exit_code=1, output="Empty fixture")
        session = SimpleNamespace(instruction=instruction, workdir="/app", mcp_servers=[], final_message="",
                                  write_file=write, exec=execute, time_left=lambda: None,
                                  log=AsyncMock(), usage=AsyncMock())
        async with await asyncio.start_server(request, "127.0.0.1", 0) as server:
            endpoint = "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1])
            env = dict(PATH=os.environ["PATH"], HOME="/tmp", NO_PROXY="127.0.0.1,localhost",
                       NODE_NO_WARNINGS="1", OPENBLAS_NUM_THREADS="1", NODE_OPTIONS="--max-old-space-size=256",
                       ANTHROPIC_BASE_URL=endpoint, ANTHROPIC_API_KEY="fixture", MODEL_NAME="claude-opus-5-5")
            with patch.dict(os.environ, env, clear=True), patch.object(harness.taskbridge, "run_process", first_call):
                try:
                    await asyncio.wait_for(harness.solve(session), timeout=120)
                except FirstCallComplete:
                    pass
        if not short:
            assert (work / "hamon-engineering-toolkit.py").read_bytes() == (root / "engineering_toolkit.py").read_bytes()
    if count_tokens:
        # Release Pi before loading the tokenizer. Only the count survives this trial.
        code = ("import sys,tiktoken; "
                "print(len(tiktoken.get_encoding('cl100k_base').encode_ordinary(sys.stdin.buffer.read().decode())))")
        captured["cl100k_base"] = int(subprocess.check_output([sys.executable, "-c", code], input=raw_request))
    return captured

def compare(paths, tasks, short_write):
    def index(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["task"] for r in rows] == tasks, "Missing, duplicate or reordered task"
        return rows
    old_rows, new_rows = map(index, paths)
    reductions = []
    for stock, candidate in zip(old_rows, new_rows):
        for key in ("restored_request_sha256", "tools", "thinking", "max_tokens"):
            assert stock[key] == candidate[key], (stock["task"], key)
        if short_write:
            assert stock["cli_path_mentions"] == candidate["cli_path_mentions"] == 0
            assert stock["raw_bytes"] == candidate["raw_bytes"]
        else:
            assert stock["cli_path_mentions"] > 1 and candidate["cli_path_mentions"] == 1
            assert stock["raw_bytes"] > candidate["raw_bytes"]
        row = dict(task=stock["task"], wire_bytes_saved=stock["raw_bytes"] - candidate["raw_bytes"])
        if "cl100k_base" in stock:
            row["cl100k_base_saved"] = stock["cl100k_base"] - candidate["cl100k_base"]
            if not short_write:
                assert row["cl100k_base_saved"] > 0
        reductions.append(row)
    print(json.dumps(dict(prompts=len(reductions), reductions=reductions)))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("STOCK_JSONL", "CANDIDATE_JSONL"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--task", action="append")
    parser.add_argument("--short-write", action="store_true")
    parser.add_argument("--bpe", action="store_true", help="Also count cl100k_base proxy tokens with tiktoken")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / "tests")]
    tasks = json.loads((root / "artifacts/manifest.json").read_bytes())["tasks"]
    if args.task and set(args.task) - set(tasks):
        parser.error("Unknown public task")
    if args.compare:
        compare(args.compare, args.task or tasks, args.short_write)
        return
    for name in (args.task or tasks):
        print(json.dumps(await run(root, name, args.short_write, args.bpe)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
