"""Exercise the submitted Pi process over loopback SSE and evaluator WebSocket."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import harness
import taskbridge


from mock_sse_proxy import MockStream


@unittest.skipUnless(Path(harness.PI).is_file(), "install the submitted Pi via npm ci")
class LoopbackCancel(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_an_active_model_stream(self):
        stream = MockStream()
        terminals = []
        async def evaluator(socket):
            async for frame in socket:
                msg = json.loads(frame)
                if msg["type"] in {"done", "error"}:
                    terminals.append(msg)
                elif "id" in msg:
                    # No task-container filesystem for this transport-only check.
                    await socket.send(json.dumps({"type": "result", "id": msg["id"],
                        "ok": False, "error": "fixture has no files"}))

        async def entry(socket):
            await taskbridge._session(socket, harness.solve)

        with tempfile.TemporaryDirectory() as home:
            async with (await asyncio.start_server(stream.respond, "127.0.0.1", 0)) as model:
                url = f"http://127.0.0.1:{model.sockets[0].getsockname()[1]}"
                env = {"PATH": os.environ["PATH"], "HOME": home, "NO_PROXY": "127.0.0.1,localhost",
                       "MODEL_API": "google-cloud/claude-opus-5-5", "MODEL_NAME": "claude-opus-5-5",
                       "ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "local-fixture",
                       "MODEL_BASE_URL": url + "/v1", "MODEL_API_KEY": "local-fixture"}
                with mock.patch.dict(os.environ, env, clear=True):
                    async with serve(entry, "127.0.0.1", 0) as server:
                        port = server.sockets[0].getsockname()[1]
                        async with connect(f"ws://127.0.0.1:{port}") as socket:
                            reader = asyncio.create_task(evaluator(socket))
                            try:
                                await socket.send(json.dumps({"type": "task", "instruction":
                                    "Write output/report.docx.", "workdir": "/app",
                                    "limits": {"agent_timeout_sec": 180}}))
                                started = asyncio.create_task(stream.started.wait())
                                try:
                                    done, _ = await asyncio.wait({started, reader}, timeout=30,
                                                                return_when=asyncio.FIRST_COMPLETED)
                                    self.assertIn(started, done, f"stream never started: {terminals}")
                                finally:
                                    started.cancel()
                                    await asyncio.gather(started, return_exceptions=True)
                                await socket.send(json.dumps({"type": "cancel"}))
                                async with asyncio.timeout(10):
                                    await reader
                                    await stream.disconnected.wait()
                            finally:
                                reader.cancel()
                                await asyncio.gather(reader, return_exceptions=True)
        self.assertEqual([msg["type"] for msg in terminals], ["done"])
        self.assertIn("/app/output/report.docx", terminals[0]["final_message"])
        self.assertIn("could not be verified", terminals[0]["final_message"])
