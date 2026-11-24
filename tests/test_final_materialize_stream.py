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
import shutil
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
    members = {'report.docx': document.getvalue(), 'result.json': json.dumps({'total': value}).encode(),
               'checks.csv': ('quantity,value\\ntotal,%s\\n' % value).encode(),
               'README.md': b'# Calculation fixture\\nSum of supplied values.\\n',
               'notes.txt': b'Synthetic runtime control.\\n', 'audit.json': b'{"inputs":3}'}
    write_loose = False
    if write_loose:
        for name, data in members.items():
            (target.parent / name).write_bytes(data)
    else:
        with zipfile.ZipFile(target, 'w') as z:
            for name, data in members.items():
                z.writestr(name, data)
'''
MEMBERS = ['report.docx', 'result.json', 'checks.csv', 'README.md', 'notes.txt', 'audit.json']

class RepairStream:
    """The second model turn writes a generator through Pi's actual write tool."""
    def __init__(self, script, *, command=None):
        self.script = str(script)
        self.generator = GENERATOR
        self.command = command
        self.early_command = None
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
            early = self.early_command if self.requests == 1 else None
            write_turn = 3 if self.early_command else 2
            if early or self.requests == (1 if self.command else write_turn):
                command = early or self.command
                tool = 'bash' if command else 'write'
                args = {'command': command} if command else {'path': self.script, 'content': self.generator}
                events += [
                    ('content_block_start', {'index': 0, 'content_block': {'type': 'tool_use',
                        'id': 'produce', 'name': tool, 'input': {}}}),
                    ('content_block_delta', {'index': 0, 'delta': {'type': 'input_json_delta',
                        'partial_json': json.dumps(args)}}),
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
    async def course(self, extension, *, cancel=False, loose=False, stream_factory=RepairStream,
                     max_steps=None, expected_requests=3, late_loose=False, split=False, inherited=False, template_case=None):
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
            members = MEMBERS
            if template_case:
                from test_template_workbooks import authored_bytes, install_templates
                templates = install_templates(root)
                members = [*templates, *MEMBERS[2:]]
                prompt = prompt.replace("report.docx`", "foundation_schedule.xlsx` on the owner's template, one row per foundation for all thirty-six, with the columns the template defines.")
                prompt = prompt.replace("result.json`", "constraint_register.xlsx` on the owner's template, with the columns the template defines.")
                for name, template in templates.items():
                    (root / (name + ".fixture")).write_bytes(authored_bytes(template, 36 if name == members[0] else 1))
                self.assertEqual(archive_contract.table_roles(prompt, members), dict(zip(members, (36, 1))))
            command = None
            if loose:
                command = 'python3 write_members.py'
                (root / 'write_members.py').write_text(
                    "import sys\nsys.argv.append('.fixture-pack.zip')\n" + GENERATOR
                    + "\nwith zipfile.ZipFile(target) as z: z.extractall('.')\n"
                    + "target.unlink()\nPath('report.zip').touch()\n")
            stream = (MockStream() if cancel else RepairStream(script, command=command) if loose
                      else stream_factory(script))
            if late_loose:
                stream.generator = GENERATOR.replace('write_loose = False', 'write_loose = True')
            if template_case:
                first = members[1] if template_case == "swapped" else members[0]
                stream.generator = stream.generator.replace(
                    "'report.docx': document.getvalue(), 'result.json': json.dumps({'total': value}).encode()",
                    f"{members[0]!r}: Path({first + '.fixture'!r}).read_bytes(), "
                    f"{members[1]!r}: Path({members[1] + '.fixture'!r}).read_bytes()")
            if split:
                self.assertTrue(late_loose)
                # Only three members come from the final generator. The other
                # three must come from the same task's earlier command.
                stream.generator = stream.generator.replace(
                    '(target.parent / name).write_bytes(data)',
                    'if name in ("report.docx", "result.json", "checks.csv"): (target.parent / name).write_bytes(data)')
                earlier = {'README.md': '# early calculation', 'notes.txt': 'early notes', 'audit.json': '{"early":1}'}
                if inherited:
                    (root / 'notes.txt').write_text(earlier.pop('notes.txt'))
                (root / 'early.py').write_text('from pathlib import Path\n' + '\n'.join(
                    f'Path({name!r}).write_text({body!r})' for name, body in earlier.items()))
                stream.early_command = 'python3 early.py'
            use_publisher = loose or late_loose
            helpers = set()
            observed_recovery = []
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
                      (contextlib.nullcontext() if use_publisher else
                       mock.patch.object(harness.archive_contract, 'provision', mock.AsyncMock(return_value=None)))):
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
                                            if use_publisher and message['path'] != str(script):
                                                helper = Path(message['path'])
                                                self.assertEqual(helper.parent, Path('/tmp'))
                                                self.assertRegex(helper.name, r'^ple-archive-[0-9a-f]{16}\.pyz$')
                                                raw = base64.b64decode(message['content_b64'])
                                                helper.write_bytes(raw)
                                                helpers.add(helper)
                                                self.addCleanup(helper.unlink, missing_ok=True)
                                                result['size'] = len(raw)
                                                await socket.send(json.dumps(result))
                                                continue
                                            self.assertEqual(message['path'], str(script))
                                            self.assertFalse(message.get('append'))
                                            data = base64.b64decode(message['content_b64'])
                                            self.assertEqual(data.decode(), stream.generator if late_loose else GENERATOR)
                                            script.write_bytes(data)
                                            writes.append(str(script))
                                            result['size'] = len(data)
                                        elif message['type'] == 'exec':
                                            args = shlex.split(message['command'])
                                            if split and message['command'].endswith('\n' + stream.early_command):
                                                args = shlex.split(stream.early_command)
                                            elif loose and message['command'].endswith('\n' + command):
                                                self.assertEqual(message['command'].splitlines()[-1], command)
                                                args = shlex.split(command)
                                            elif use_publisher and len(args) >= 3 and Path(args[1]) in helpers:
                                                stage, dest, member_spec = json.loads(args[2])
                                                self.assertEqual(dest, str(output))
                                                self.assertEqual(list(member_spec), members)
                                                self.assertRegex(stage, r'^/tmp/ple-members-[0-9a-f]{16}$')
                                                self.addCleanup(shutil.rmtree, stage, ignore_errors=True)
                                                self.addCleanup(Path(stage + '.checkpoint.zip').unlink, missing_ok=True)
                                                observed_recovery.append(args[3:])
                                            else:
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
                                            if not use_publisher or args[1:2] == ['-c']:
                                                self.assertEqual(process.returncode, 0, stderr.decode())
                                            result.update(exit_code=process.returncode, output=stdout.decode(), timed_out=False, truncated=False)
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
            elif loose:
                self.assertIn('bash', stream.tools)
                self.assertGreater(output.stat().st_size, 0, summary)
                self.assertIn(['--after-empty'], observed_recovery)
                self.assertNotIn('Incomplete', summary)
                with zipfile.ZipFile(output) as archive:
                    self.assertEqual(sorted(archive.namelist()), sorted(MEMBERS))
                    for member in MEMBERS:
                        self.assertEqual(archive.read(member), (root / member).read_bytes())
            else:
                self.assertIn('write', stream.tools)
                self.assertEqual(len(stream.requests) if isinstance(stream.requests, list) else stream.requests, expected_requests)
                self.assertEqual(writes, [str(script)])
                self.assertEqual(len(materializations), 1)
                if late_loose:
                    self.assertTrue(any(args[:1] == ['--from-generated-output'] for args in observed_recovery))
                if split:
                    self.assertTrue(all((root / name).is_file() for name in MEMBERS))
                if template_case:
                    self.assertTrue(any(args[:1] == ['--from-generated-output'] for args in observed_recovery))
                    if template_case == "swapped":
                        self.assertIn("Incomplete delivery", summary)
                        self.assertFalse(output.exists())
                    else:
                        self.assertNotIn("Incomplete", summary)
                        self.assertNotIn("could not be verified", summary)
                        with zipfile.ZipFile(output) as archive:
                            self.assertEqual(sorted(archive.namelist()), sorted(members))
                            for name in members[:2]:
                                self.assertEqual(archive.read(name), (root / (name + ".fixture")).read_bytes())
                    return
                if inherited:
                    self.assertIn('Incomplete delivery', summary)
                    self.assertFalse(output.exists(), summary)
                    self.assertEqual((root / 'notes.txt').read_text(), 'early notes')
                    return
                self.assertNotIn('Incomplete', summary)
                self.assertNotIn('could not be verified', summary)
                with zipfile.ZipFile(output) as archive:
                    if extension == '.zip':
                        self.assertEqual(sorted(archive.namelist()), sorted(MEMBERS))
                        if split:
                            self.assertEqual(archive.read('notes.txt'), b'early notes')
                            self.assertEqual(json.loads(archive.read('audit.json')), {'early': 1})
                        self.assertEqual(json.loads(archive.read('result.json')), {'total': 10})
                        with zipfile.ZipFile(io.BytesIO(archive.read('report.docx'))) as document:
                            self.assertIn(b'Computed total: 10', document.read('word/document.xml'))
                    else:
                        self.assertIn(b'Computed total: 10', archive.read('word/document.xml'))

    async def test_actual_pi_template_role_pair(self):
        for case in ("completed", "swapped"):
            with self.subTest(case=case):
                await self.course(".zip", late_loose=True, template_case=case)

    async def test_docx_from_generator_written_in_final_repair(self):
        await self.course('.docx')

    async def test_six_member_zip_from_generator_written_in_final_repair(self):
        await self.course('.zip')

    async def test_empty_output_from_registered_bash_republishes_complete_loose_members(self):
        await self.course('.zip', loose=True)

    async def test_six_loose_members_from_the_final_generator(self):
        await self.course('.zip', late_loose=True)

    async def test_earlier_tool_and_final_generator_complete_one_archive(self):
        await self.course('.zip', late_loose=True, split=True, expected_requests=4)

    async def test_unchanged_inherited_member_cannot_complete_split_archive(self):
        await self.course('.zip', late_loose=True, split=True, inherited=True, expected_requests=4)

    async def test_live_sse_cancel_reaps_pi_and_leaves_no_deliverable(self):
        await self.course('.docx', cancel=True)


if __name__ == '__main__':
    unittest.main()
