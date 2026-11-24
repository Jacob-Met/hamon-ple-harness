"""Exercise the real submitted Pi's Bash tool, task wire and cancellation on local fixtures."""
import asyncio
import json
import os
from pathlib import Path
import shlex
from unittest import mock

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import harness
from mock_sse_proxy import MockStream
from test_final_materialize_stream import RepairStream


class WriteAndWait:
    def __init__(self, command):
        self.first = RepairStream("", command=command)
        self.pending = MockStream()

    async def respond(self, reader, writer):
        if self.first.requests == 0:
            await self.first.respond(reader, writer)
        else:
            await self.pending.respond(reader, writer)


async def produce(task_id, path, source_bytes, scratch):
    """Only the registered copy command can materialize the file sent to the organizer."""
    work = scratch / task_id
    work.mkdir(parents=True)
    fixture = work / "input.bin"
    fixture.write_bytes(source_bytes)
    actual = work / "container" / path.lstrip("/")
    actual.parent.mkdir(parents=True)
    if actual.exists():
        raise AssertionError("the target must start absent")
    command = shlex.join(["cp", "--", "/fixture/input.bin", path])
    stream = WriteAndWait(command)
    acknowledged = asyncio.Event()
    cancelled = asyncio.Event()
    writes = 0
    terminals = []

    async def entry(socket):
        await harness.taskbridge._session(socket, harness.solve)

    async def cancel_after_write(socket):
        await acknowledged.wait()
        await stream.pending.started.wait()
        await socket.send(json.dumps({"type": "cancel"}))
        cancelled.set()

    async with await asyncio.start_server(stream.respond, "127.0.0.1", 0) as proxy:
        url = f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
        env = {"PATH":os.environ["PATH"], "HOME":str(work), "NO_PROXY":"127.0.0.1,localhost",
               "MODEL_API":"google-cloud/claude-opus-5-5", "MODEL_NAME":"claude-opus-5-5",
               "ANTHROPIC_BASE_URL":url, "ANTHROPIC_API_KEY":"fixture",
               "MODEL_BASE_URL":url+"/v1", "MODEL_API_KEY":"fixture"}
        with mock.patch.dict(os.environ, env, clear=True):
            async with serve(entry, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as ws:
                    await ws.send(json.dumps({"type":"task", "workdir":"/filesystem/work",
                                              "instruction":f"Copy the supplied binary file /fixture/input.bin to {path}.",
                                              "limits":{"agent_timeout_sec":150}}))
                    stopper = asyncio.create_task(cancel_after_write(ws))
                    try:
                        async with asyncio.timeout(60):
                            async for raw in ws:
                                msg = json.loads(raw)
                                if msg["type"] in ("done","error"):
                                    terminals.append(msg)
                                    continue
                                if "id" not in msg:
                                    continue
                                # Ignore setup/probe requests. Only a precisely matched
                                # registered copy can mutate the temporary task filesystem.
                                answer = {"type":"result","id":msg["id"],"ok":False,
                                          "error":"no such input in this fixture"}
                                lines = msg.get("command", "").splitlines()
                                if msg["type"] == "exec" and lines and lines[-1] == command:
                                    if writes or cancelled.is_set():
                                        raise AssertionError("duplicate or late tool write")
                                    proc = await asyncio.create_subprocess_exec("cp","--",fixture,actual)
                                    if await proc.wait() or actual.read_bytes() != source_bytes:
                                        raise AssertionError("the real task-side copy changed the bytes")
                                    writes += 1
                                    answer.update(ok=True,exit_code=0,output="")
                                    answer.pop("error")
                                await ws.send(json.dumps(answer))
                                if writes:
                                    acknowledged.set()
                            await stopper
                            await stream.pending.disconnected.wait()
                    finally:
                        stopper.cancel()
                        await asyncio.gather(stopper,return_exceptions=True)
    if (not cancelled.is_set() or writes != 1 or stream.first.requests != 1
            or "bash" not in stream.first.tools or len(terminals) != 1 or terminals[0]["type"] != "done"):
        raise AssertionError(f"{task_id}: no acknowledged registered write and single cancel terminal")
    message = terminals[0].get("final_message")
    if not isinstance(message,str) or not message.strip() or actual.read_bytes() != source_bytes:
        raise AssertionError(f"{task_id}: missing terminal or edited target after cancellation")
    return actual, message
