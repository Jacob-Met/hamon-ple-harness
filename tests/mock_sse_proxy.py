"""Run the organizer cancel check with an ephemeral, offline Anthropic SSE fixture.

Run with the organizer Python environment:
    python tests/mock_sse_proxy.py /path/to/organizer/kit-checkout
The host probe uses loopback. On Linux the organizer's isolated container
reaches the same fixture through Docker's local bridge gateway.
"""
import argparse
import asyncio
import contextlib
import json
import ipaddress
import os
import secrets
from pathlib import Path
import sys
import subprocess


class MockStream:
    """Return a real partial Anthropic event stream, then wait for client close."""

    def __init__(self, reply_text=None, *, auth_token=None):
        self.reply_text = reply_text
        self.auth_token = auth_token
        self.requests = 0
        self.probed = asyncio.Event()
        self.started = asyncio.Event()
        self.disconnected = asyncio.Event()

    async def respond(self, reader, writer):
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii").split("\r\n")
            fields = {name.lower(): value.strip() for name, value in
                      (line.split(":", 1) for line in headers[1:] if ":" in line)}
            if self.auth_token and fields.get("x-api-key") != self.auth_token and (
                fields.get("authorization") != "Bearer " + self.auth_token
            ):
                writer.write(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            request = json.loads(await reader.readexactly(int(fields["content-length"])))
            method, target, _ = headers[0].split()
            route = (method, target.split("?", 1)[0])
            if (route == ("POST", "/v1/chat/completions")
                    and request.get("stream", False) is False and request.get("messages")):
                # The organizer checks connectivity through its OpenAI-compatible
                # endpoint before using the native Anthropic SSE stream for Pi.
                body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "OK"}}]}).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
                await writer.drain()
                self.probed.set()
                return
            if route != ("POST", "/v1/messages") or request.get("stream") is not True:
                writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            events = [
                ("message_start", {"message": {"id": "local", "type": "message", "role": "assistant",
                    "model": request["model"], "content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}}),
                ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": self.reply_text or "Partial"}}),
            ]
            self.requests += 1
            if self.reply_text is not None:
                events.extend([
                    ("content_block_stop", {"index": 0}),
                    ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                       "usage": {"output_tokens": 1}}),
                    ("message_stop", {}),
                ])
            for kind, event in events:
                payload = json.dumps({"type": kind, **event})
                writer.write(f"event: {kind}\ndata: {payload}\n\n".encode())
            await writer.drain()
            self.started.set()
            if self.reply_text is None:
                await reader.read()
            self.disconnected.set()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


def docker_bridge_gateway():
    """Find the local address the organizer maps to host.docker.internal."""
    if sys.platform != "linux":
        return None
    raw = subprocess.run(
        ["docker", "network", "inspect", "bridge"], check=True,
        capture_output=True, text=True, timeout=15,
    ).stdout
    for config in json.loads(raw)[0]["IPAM"]["Config"]:
        value = config.get("Gateway")
        if value:
            gateway = ipaddress.ip_address(value)
            if gateway.version == 4 and not gateway.is_unspecified:
                return str(gateway)
    raise RuntimeError("Docker's local bridge has no IPv4 gateway")


@contextlib.asynccontextmanager
async def serve_kit_stream(stream, gateway):
    async with contextlib.AsyncExitStack() as stack:
        host = await stack.enter_async_context(
            await asyncio.start_server(stream.respond, "127.0.0.1", 0)
        )
        port = host.sockets[0].getsockname()[1]
        if gateway and gateway != "127.0.0.1":
            await stack.enter_async_context(
                await asyncio.start_server(stream.respond, gateway, port)
            )
        yield port


async def run_kit(kit_root):
    token = secrets.token_hex(24)
    stream = MockStream(auth_token=token)
    async with serve_kit_stream(stream, docker_bridge_gateway()) as port:
        env = {**os.environ, "MODEL_PROXY_URL": f"http://127.0.0.1:{port}",
               "MODEL_PROXY_KEY": token}
        check = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "kit", "check", str(Path(__file__).resolve().parents[1]),
            "--cancel-test", "--no-smoke", cwd=kit_root, env=env)
        status = await check.wait()
        if status == 0 and not (stream.probed.is_set() and stream.started.is_set() and stream.disconnected.is_set()):
            print("The local model probe, stream start, or disconnect was missing.", file=sys.stderr)
            return 1
        return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kit_root", type=Path)
    sys.exit(asyncio.run(run_kit(parser.parse_args().kit_root)))
