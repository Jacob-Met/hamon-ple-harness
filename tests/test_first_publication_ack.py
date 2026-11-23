"""Exercise the actual publisher and its Pi tool-result boundary."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
import zipfile

import archive_contract


class FirstPublicationAck(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_publication_does_not_repeat_the_remote_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "out.zip"

            async def write_file(path, body):
                return (root / Path(path).name).write_bytes(body)

            session = SimpleNamespace(
                instruction=("Build the first complete version, then improve. Write out.zip "
                             "containing two files at the archive root:\n1. `a.json`\n2. `b.csv`"),
                write_file=write_file, log=AsyncMock(), exec=AsyncMock(),
            )
            # Use the same standalone helper that runs inside the task container.
            _, provisioned = await archive_contract.provision(
                session, [{"kind": "file", "path": str(target)}])
            _, helper, _ = shlex.split(provisioned)
            stage = root / "staged"
            stage.mkdir()
            (stage / "a.json").write_text('{"pressure": 12}')
            (stage / "b.csv").write_text("value\n12\n")
            command = shlex.join([sys.executable, "-I", "-S", str(root / Path(helper).name),
                                  json.dumps([str(stage), str(target), ["a.json", "b.csv"]])])
            watch = archive_contract.FirstPublication((str(stage), command))
            await watch.observe(session, {"type": "tool_execution_start", "toolCallId": "publish",
                                          "toolName": "bash", "args": {"command": command}})
            result = subprocess.run(shlex.split(command), capture_output=True, text=True, check=True)
            await watch.observe(session, {"type": "message_end", "message": {
                "role": "toolResult", "toolCallId": "publish", "toolName": "bash",
                "isError": False, "content": [{"type": "text", "text": result.stdout}]}})
            self.assertTrue(watch.published)
            session.exec.assert_not_awaited()
            with zipfile.ZipFile(target) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.read("a.json"), (stage / "a.json").read_bytes())
                self.assertEqual(archive.read("b.csv"), (stage / "b.csv").read_bytes())

    async def test_other_commands_and_ambiguous_results_use_existing_publication(self):
        command = "publisher /tmp/stage"
        for tool, actual, output, error, followups in (
            ("bash", command, "", False, 1),
            ("bash", command + " && echo more", "PLE_ARCHIVE_PUBLISHED", False, 1),
            ("write", command, "PLE_ARCHIVE_PUBLISHED", False, 1),
            ("bash", command, "PLE_ARCHIVE_PUBLISHED", True, 0),
        ):
            with self.subTest(tool=tool, actual=actual, output=output, error=error):
                session = SimpleNamespace(log=AsyncMock(), exec=AsyncMock(return_value={"exit_code": 0}))
                watch = archive_contract.FirstPublication(("/tmp/stage", command))
                await watch.observe(session, {"type": "tool_execution_start", "toolCallId": "a",
                                              "toolName": tool, "args": {"command": actual}})
                await watch.observe(session, {"type": "message_end", "message": {
                    "role": "toolResult", "toolCallId": "a", "toolName": tool, "isError": error,
                    "content": [{"type": "text", "text": output}]}})
                self.assertEqual(session.exec.await_count, followups)
                self.assertEqual(watch.published, not error)
