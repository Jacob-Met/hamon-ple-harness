"""Check the two distinct model routes used by the organizer cancel check."""
import asyncio
import http.client
import json
import unittest

from mock_sse_proxy import MockStream


class OrganizerModelProbe(unittest.IsolatedAsyncioTestCase):
    async def test_only_connectivity_probe_returns_without_opening_a_stream(self):
        mock = MockStream()
        async with await asyncio.start_server(mock.respond, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]

            def request(path, payload):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    connection.request("POST", path, json.dumps(payload),
                                       headers={"Content-Type": "application/json"})
                    response = connection.getresponse()
                    return response.status, response.read()
                finally:
                    connection.close()

            prompt = {"messages": [{"role": "user", "content": "Reply OK."}]}
            for path, payload in [
                ("/v1/messages", prompt),
                ("/v1/chat/completions", {**prompt, "stream": True}),
            ]:
                status, _ = await asyncio.to_thread(request, path, payload)
                self.assertEqual(status, 400)
                self.assertFalse(mock.probed.is_set())

            status, body = await asyncio.to_thread(request, "/v1/chat/completions", prompt)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "OK")
            self.assertTrue(mock.probed.is_set())
            self.assertFalse(mock.started.is_set())
            self.assertFalse(mock.disconnected.is_set())
