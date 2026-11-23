"""Real Pi -> SSE -> registered write -> task exec -> final receipt.

Synthetic calculation/format fixtures measure delivery and shutdown, not grades.
All connections stay on loopback; no Docker or Harbor.
"""
import asyncio
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock
import zipfile
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import archive_contract
import delivery_guard
import delivery_materialize
import harness
from mock_sse_proxy import MockStream

GENERATOR = '''import io, json, sys, zipfile
from pathlib import Path
value = sum(json.loads(Path('input.json').read_text()))
document = io.BytesIO()
with zipfile.ZipFile(document, 'w') as z:
    z.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    z.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="r1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Computed total: %s</w:t></w:r></w:p></w:body></w:document>' % value)
target = Path(sys.argv[1])
if target.suffix == '.docx':
    target.write_bytes(document.getvalue())
else:
    with zipfile.ZipFile(target, 'w') as z:
        z.writestr('report.docx', document.getvalue())
        z.writestr('result.json', json.dumps({'total': value}))
        z.writestr('checks.csv', 'quantity,value\\ntotal,%s\\n' % value)
        z.writestr('README.md', '# Calculation fixture\\nSum of supplied values.\\n')
        z.writestr('notes.txt', 'Synthetic runtime control.\\n')
        z.writestr('audit.json', json.dumps({'inputs': 3}))
'''
MEMBERS = ['report.docx', 'result.json', 'checks.csv', 'README.md', 'notes.txt', 'audit.json']

class RepairStream:
    """The second model turn writes a generator through Pi's actual write tool."""
    def __init__(self, script):
        self.script = str(script)
        self.requests = 0
        self.tools = set()

    async def respond(self, reader, writer):
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode('ascii').split('\r\n')
            fields = dict(line.lower().split(': ', 1) for line in headers[1:] if ': ' in line)
            request = json.loads(await reader.readexactly(int(fields['content-length'])))
            method, target, _ = headers[0].split()
            if method != 'POST' or target.split('?')[0] != '/v1/messages' or request.get('stream') is not True:
                writer.write(b'HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n')
                return
            self.requests += 1
            self.tools.update(t['name'] for t in request.get('tools', []))
            events = [('message_start', {'message': {'id': f'local-{self.requests}', 'type': 'message',
                'role': 'assistant', 'model': request['model'], 'content': [],
                'usage': {'input_tokens': 1, 'output_tokens': 0}}})]
            if self.requests == 2:
                events += [
                    ('content_block_start', {'index': 0, 'content_block': {'type': 'tool_use',
                        'id': 'write-generator', 'name': 'write', 'input': {}}}),
                    ('content_block_delta', {'index': 0, 'delta': {'type': 'input_json_delta',
                        'partial_json': json.dumps({'path': self.script, 'content': GENERATOR})}}),
                ]
                reason = 'tool_use'
            else:
                events += [
                    ('content_block_start', {'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
                    ('content_block_delta', {'index': 0, 'delta': {'type': 'text_delta', 'text': 'Calculation prepared.'}}),
                ]
                reason = 'end_turn'
            events += [('content_block_stop', {'index': 0}),
                       ('message_delta', {'delta': {'stop_reason': reason, 'stop_sequence': None},
                                          'usage': {'output_tokens': 1}}), ('message_stop', {})]
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n')
            for kind, event in events:
                writer.write(f'event: {kind}\ndata: {json.dumps({"type": kind, **event})}\n\n'.encode())
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


@unittest.skipUnless(Path(harness.PI).is_file(), 'actual installed Pi CLI required')
class FinalMaterializationStream(unittest.IsolatedAsyncioTestCase):
    async def course(self, extension, *, cancel=False, stream_factory=RepairStream,
                     max_steps=None, expected_requests=3):
        # CI provisions this empty task root, separate from the harness.
        with (tempfile.TemporaryDirectory(prefix='ple-recovery-', dir='/filesystem') as directory,
              tempfile.TemporaryDirectory(prefix='ple-home-') as home):
            root = Path(directory)
            (root / 'input.json').write_text('[2, 3, 5]')
            output, script = root / ('report' + extension), root / 'generate.py'
            prompt = f'Compute the total in input.json and write the final result to {output}.'
            if extension == '.zip':
                prompt += '\nWrite a first complete version, then improve it. Produce exactly these six files at the archive root:\n' + '\n'.join(
                    f'{i}. `{name}`' for i, name in enumerate(MEMBERS, 1))
                self.assertEqual(sorted(archive_contract.member_names(prompt)), sorted(MEMBERS))
            stream = MockStream() if cancel else stream_factory(script)
            writes, materializations, children = [], [], []
            spawn = asyncio.create_subprocess_exec
            loop = asyncio.get_running_loop()
            spawn_transport = loop.subprocess_exec

            async def record_spawn(*args, **kwargs):
                # run_process uses a custom SubprocessStreamProtocol so leader
                # exit is observable even if a descendant holds an output pipe.
                transport, protocol = await spawn_transport(*args, **kwargs)
                if str(args[1]) == harness.PI:
                    children.append(transport)
                return transport, protocol

            async def entry(socket):
                await harness.taskbridge._session(socket, harness.solve)

            async with await asyncio.start_server(stream.respond, '127.0.0.1', 0) as proxy:
                url = f'http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}'
                env = {'PATH': os.environ['PATH'], 'HOME': home, 'NO_PROXY': '127.0.0.1,localhost',
                       'MODEL_API': 'google-cloud/claude-opus-5-5', 'MODEL_NAME': 'claude-opus-5-5',
                       'ANTHROPIC_BASE_URL': url, 'ANTHROPIC_API_KEY': 'fixture',
                       'MODEL_BASE_URL': url + '/v1', 'MODEL_API_KEY': 'fixture'}
                with (mock.patch.dict(os.environ, env, clear=True),
                      mock.patch.object(loop, 'subprocess_exec', side_effect=record_spawn),
                      mock.patch.object(harness.engineering_intake, 'intake_for_prompt', mock.AsyncMock(return_value='')),
                      mock.patch.object(harness.engineering_intake, 'provision_engineering_cli', mock.AsyncMock(return_value='')),
                      mock.patch.object(harness.archive_contract, 'provision', mock.AsyncMock(return_value=None))):
                    async with serve(entry, '127.0.0.1', 0) as server:
                        port = server.sockets[0].getsockname()[1]
                        async with connect(f'ws://127.0.0.1:{port}') as socket:
                            await socket.send(json.dumps({'type': 'task', 'instruction': prompt, 'workdir': str(root),
                                                          'limits': {'agent_timeout_sec': 150, **({'max_steps': max_steps} if max_steps is not None else {})}}))
                            async def stop():
                                await stream.started.wait()
                                await socket.send(json.dumps({'type': 'cancel'}))
                            stopper = asyncio.create_task(stop()) if cancel else None
                            try:
                                async with asyncio.timeout(45):
                                    async for frame in socket:
                                        message = json.loads(frame)
                                        if message['type'] == 'done':
                                            summary = message.get('final_message', '')
                                            break
                                        self.assertNotEqual(message['type'], 'error', message)
                                        if 'id' not in message:
                                            continue
                                        result = {'type': 'result', 'id': message['id'], 'ok': True}
                                        if message['type'] == 'read_file':
                                            self.assertEqual(message['path'], str(root / 'input.json'))
                                            data = (root / 'input.json').read_bytes()
                                            result.update(content_b64=base64.b64encode(data).decode(), size=len(data), truncated=False)
                                        elif message['type'] == 'write_file':
                                            self.assertEqual(message['path'], str(script))
                                            self.assertFalse(message.get('append'))
                                            data = base64.b64decode(message['content_b64'])
                                            self.assertEqual(data.decode(), GENERATOR)
                                            script.write_bytes(data)
                                            writes.append(str(script))
                                            result['size'] = len(data)
                                        elif message['type'] == 'exec':
                                            args = shlex.split(message['command'])
                                            self.assertEqual(args[:2], ['python3', '-c'])
                                            self.assertEqual(len(args), 4)
                                            self.assertIn(args[2], [delivery_guard.REMOTE_PROBE, delivery_materialize.REMOTE_MATERIALIZE])
                                            spec = json.loads(args[3])
                                            if isinstance(spec, dict):
                                                self.assertEqual(spec['script'], str(script))
                                                self.assertEqual(spec['target'], str(output))
                                                materializations.append(spec)
                                            else:
                                                self.assertTrue(all(Path(row['path']).is_relative_to(root) for row in spec))
                                            process = await spawn(*args, cwd=root, stdout=asyncio.subprocess.PIPE,
                                                                  stderr=asyncio.subprocess.PIPE)
                                            stdout, stderr = await asyncio.wait_for(process.communicate(), message['timeout_sec'] + 2)
                                            self.assertEqual(process.returncode, 0, stderr.decode())
                                            result.update(exit_code=0, output=stdout.decode(), timed_out=False, truncated=False)
                                        else:
                                            self.fail(f'Unexpected task action: {message["type"]}')
                                        await socket.send(json.dumps(result))
                                    else:
                                        self.fail('No terminal done frame')
                            finally:
                                if stopper:
                                    stopper.cancel()
                                    await asyncio.gather(stopper, return_exceptions=True)
            self.assertTrue(children)
            self.assertTrue(all(child.get_returncode() is not None for child in children))
            self.assertFalse(list(Path(home).glob('pi-*')))
            if cancel:
                await asyncio.wait_for(stream.disconnected.wait(), 3)
                self.assertEqual(writes, [])
                self.assertEqual(materializations, [])
                self.assertFalse(output.exists())
                self.assertIn('could not be verified', summary)
            else:
                self.assertIn('write', stream.tools)
                self.assertEqual(len(stream.requests) if isinstance(stream.requests, list) else stream.requests, expected_requests)
                self.assertEqual(writes, [str(script)])
                self.assertEqual(len(materializations), 1)
                self.assertNotIn('Incomplete', summary)
                self.assertNotIn('could not be verified', summary)
                with zipfile.ZipFile(output) as archive:
                    if extension == '.zip':
                        self.assertEqual(sorted(archive.namelist()), sorted(MEMBERS))
                        self.assertEqual(json.loads(archive.read('result.json')), {'total': 10})
                        with zipfile.ZipFile(io.BytesIO(archive.read('report.docx'))) as document:
                            self.assertIn(b'Computed total: 10', document.read('word/document.xml'))
                    else:
                        self.assertIn(b'Computed total: 10', archive.read('word/document.xml'))

    async def test_docx_from_generator_written_in_final_repair(self):
        await self.course('.docx')

    async def test_six_member_zip_from_generator_written_in_final_repair(self):
        await self.course('.zip')

    async def test_live_sse_cancel_reaps_pi_and_leaves_no_deliverable(self):
        await self.course('.docx', cancel=True)


if __name__ == '__main__':
    unittest.main()
