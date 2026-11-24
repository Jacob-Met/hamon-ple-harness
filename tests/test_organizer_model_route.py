"""Exercise the two addresses used by the organizer's local Docker course."""
import asyncio
import json
import sys
import unittest

from mock_sse_proxy import MockStream, serve_kit_stream


@unittest.skipUnless(sys.platform == "linux", "Linux Docker bridge route")
class OrganizerModelRoute(unittest.IsolatedAsyncioTestCase):
    async def test_probe_and_stream_share_the_reserved_port(self):
        stream = MockStream(reply_text="finished", auth_token="MixedCase")
        async with serve_kit_stream(stream, "127.0.0.2") as port:
            for host, route, body, credential, status in [
                ("127.0.0.2", "/v1/messages", {"model": "fixture", "stream": True},
                 "x-api-key: mixedcase", 401),
                ("127.0.0.1", "/v1/chat/completions", {"messages": [{"role": "user", "content": "ok"}]},
                 "Authorization: Bearer MixedCase", 200),
                ("127.0.0.2", "/v1/messages", {"model": "fixture", "stream": True},
                 "x-api-key: MixedCase", 200),
            ]:
                reader, writer = await asyncio.open_connection(host, port)
                payload = json.dumps(body).encode()
                head = f"POST {route} HTTP/1.1\r\n{credential}\r\nContent-Length: {len(payload)}\r\n\r\n"
                writer.write(head.encode() + payload)
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), 3)
                writer.close()
                await writer.wait_closed()
                self.assertTrue(response.startswith(f"HTTP/1.1 {status} ".encode()), response)
            self.assertTrue(stream.probed.is_set() and stream.started.is_set() and stream.disconnected.is_set())
            self.assertEqual(stream.requests, 1)
        for host in ("127.0.0.1", "127.0.0.2"):
            with self.assertRaises(OSError):
                await asyncio.open_connection(host, port)
