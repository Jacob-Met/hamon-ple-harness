"""Run the organizer cancel check with an ephemeral, offline Anthropic SSE fixture.

Run with the organizer Python environment:
    python tests/mock_sse_proxy.py /path/to/organizer/kit-checkout
The Docker host gateway must be able to reach host loopback.
"""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import sys


class MockStream:
    """Return a real partial Anthropic event stream, then wait for client close."""

    def __init__(self):
        self.started = asyncio.Event()
        self.disconnected = asyncio.Event()

    async def respond(self, reader, writer):
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii").split("\r\n")
            fields = dict(line.lower().split(": ", 1) for line in headers[1:] if ": " in line)
            request = json.loads(await reader.readexactly(int(fields["content-length"])))
            method, target, _ = headers[0].split()
            if (method, target.split("?", 1)[0]) != ("POST", "/v1/messages") or request.get("stream") is not True:
                writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            events = [
                ("message_start", {"message": {"id": "local", "type": "message", "role": "assistant",
                    "model": request["model"], "content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}}),
                ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Partial"}}),
            ]
            for kind, event in events:
                payload = json.dumps({"type": kind, **event})
                writer.write(f"event: {kind}\ndata: {payload}\n\n".encode())
            await writer.drain()
            self.started.set()
            await reader.read()
            self.disconnected.set()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


async def run_kit(kit_root):
    stream = MockStream()
    async with await asyncio.start_server(stream.respond, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        env = {**os.environ, "MODEL_PROXY_URL": f"http://127.0.0.1:{port}",
               "MODEL_PROXY_KEY": "local-fixture"}
        check = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "kit", "check", str(Path(__file__).resolve().parents[1]),
            "--cancel-test", "--no-smoke", cwd=kit_root, env=env)
        status = await check.wait()
        if status == 0 and not (stream.started.is_set() and stream.disconnected.is_set()):
            print("The local model stream did not start and disconnect.", file=sys.stderr)
            return 1
        return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kit_root", type=Path)
    sys.exit(asyncio.run(run_kit(parser.parse_args().kit_root)))
