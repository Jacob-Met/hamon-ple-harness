"""Exercise final-turn generator recovery through the real solve controller.

Task I/O is mapped to an isolated directory; the calculation itself executes.
These generic fixtures do not assert an organizer engineering score.
"""
import contextlib
import asyncio
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest import mock

import delivery_materialize
import harness


GENERATOR = """import json, sys
from pathlib import Path
numbers = json.loads(Path('input.json').read_text())
Path(sys.argv[1]).write_text(json.dumps({'answer': sum(numbers)}))
"""


ARCHIVE_NAMES = [f"region_{n}.json" for n in range(6)]
ARCHIVE_GENERATOR = """import json
from pathlib import Path
numbers = json.loads(Path('input.json').read_text())
for name in %r:
    Path(name).write_text(json.dumps({'answer': sum(numbers)}))
""" % ARCHIVE_NAMES


class FinalMaterialization(unittest.IsolatedAsyncioTestCase):
    async def course(self, *, inherited=False, remaining=90,
                     early_generator=False, rewrite=False, first_unknown=False,
                     final_log_failure=None, archive=False, unchanged_member=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "filesystem/work"
            work.mkdir(parents=True)
            (work / "input.json").write_text("[2, 3, 5]")
            if inherited:
                (work / "generate.py").write_text(GENERATOR)
            target_name = "submission.zip" if archive else "report.json"
            generator = ARCHIVE_GENERATOR if archive else GENERATOR
            plan = None
            if archive:
                if unchanged_member:
                    (work / ARCHIVE_NAMES[0]).write_text('{"answer": 10}')
                plan = ("/tmp/loose-stage", "python3 archive_checkpoint.py " +
                        shlex.quote(json.dumps(["/tmp/loose-stage", "/filesystem/work/submission.zip", ARCHIVE_NAMES])))
            left = 700
            turns, executions = [], []
            session = SimpleNamespace(
                instruction="Write the final result to /filesystem/work/report.json.",
                workdir="/filesystem/work", mcp_servers=[], final_message="",
                usage=mock.AsyncMock(), log=mock.AsyncMock(), time_left=lambda: left,
            )
            if archive:
                session.instruction = ("Create a complete submission early, then improve it in place. "
                    "Write the final deliverable to /filesystem/work/submission.zip. "
                    "Write exactly these six files:\n" + "\n".join(
                        f"{i}. **`{name}`**" for i, name in enumerate(ARCHIVE_NAMES, 1)))

            async def log(text):
                if text.startswith("delivery_gate: final_materializers="):
                    if final_log_failure == "stall":
                        await asyncio.Event().wait()
                    elif final_log_failure:
                        raise RuntimeError("log unavailable")
            session.log.side_effect = log

            def real_path(value):
                path = Path(value)
                return path if path.is_relative_to(root) else root / str(path).lstrip("/")

            async def remote(command, timeout_sec):
                parts = shlex.split(command)
                if archive and parts[1] == "archive_checkpoint.py":
                    stage, target, names = json.loads(parts[2])
                    mapped = json.dumps([str(real_path(stage)), str(real_path(target)), names])
                    proc = subprocess.run([sys.executable, harness.HERE / "archive_checkpoint.py",
                                           mapped, *parts[3:]], capture_output=True, timeout=timeout_sec)
                    return {"exit_code": proc.returncode, "output": proc.stdout.decode()}
                _, _, source, arguments = parts
                spec = json.loads(arguments)
                if isinstance(spec, dict):
                    self.assertEqual(source, delivery_materialize.REMOTE_MATERIALIZE)
                    self.assertEqual(spec["script"], "/filesystem/work/generate.py")
                    self.assertEqual(spec["target"], "/filesystem/work/" + target_name)
                    executions.append(spec)
                    if first_unknown and len(executions) == 1:
                        raise harness.taskbridge.TaskError("generator result unavailable")
                    result = subprocess.run(
                        [sys.executable, str(real_path(spec["script"])), str(real_path(spec["target"]))],
                        cwd=work, capture_output=True, timeout=timeout_sec,
                    )
                    return {"exit_code": 0, "output": json.dumps({
                        "state": "executed" if result.returncode == 0 else "nonzero",
                    })}
                stdout = io.StringIO()
                with (mock.patch("pathlib.Path", side_effect=real_path),
                      mock.patch.object(sys, "argv", ["-c", arguments]),
                      contextlib.redirect_stdout(stdout)):
                    exec(source, {})
                return {"exit_code": 0, "output": stdout.getvalue()}

            session.exec = remote

            async def participant(*args, on_line, log_prefix, **kwargs):
                nonlocal left
                turns.append(log_prefix)
                if log_prefix == "pi-delivery-repair: ":
                    if not inherited:
                        (work / "generate.py").write_text(generator + ("\n# Revised calculation\n" if rewrite else ""))
                    left = remaining
                else:
                    if early_generator:
                        (work / "generate.py").write_text(GENERATOR)
                    left = 230
                await on_line(json.dumps({"type": "message_end", "message": {
                    "role": "assistant", "content": [{"type": "text", "text": "The calculation is ready."}],
                }}).encode())
                return SimpleNamespace(code=0, stderr="")

            relay = mock.MagicMock()
            relay.__aenter__.return_value = SimpleNamespace(env=lambda: dict(os.environ))
            with contextlib.ExitStack() as patches:
                for owner, name, value in [
                    (harness.taskbridge, "Relay", mock.Mock(return_value=relay)),
                    (harness.taskbridge, "run_process", mock.AsyncMock(side_effect=participant)),
                    (harness, "configure", mock.Mock(return_value=("model", "fixture"))),
                    (harness.engineering_intake, "intake_for_prompt", mock.AsyncMock(return_value="")),
                    (harness.engineering_intake, "provision_engineering_cli", mock.AsyncMock(return_value="")),
                    (harness.archive_contract, "provision", mock.AsyncMock(return_value=plan)),
                ]:
                    patches.enter_context(mock.patch.object(owner, name, value))
                summary = await harness.solve(session)
            output = work / target_name
            if archive and output.exists():
                with zipfile.ZipFile(output) as result:
                    return summary, {n: json.loads(result.read(n)) for n in result.namelist()}, turns, executions
            return summary, json.loads(output.read_text()) if output.exists() else None, turns, executions

    async def test_new_generator_after_repair_finishes(self):
        summary, output, turns, executions = await self.course()
        self.assertEqual(turns, ["pi: ", "pi-delivery-repair: "])
        self.assertEqual(output, {"answer": 10})
        self.assertNotIn("Incomplete", summary)
        self.assertEqual(len(executions), 1)

    async def test_last_generator_publishes_verified_loose_archive_members(self):
        summary, output, _, executions = await self.course(archive=True)
        self.assertEqual(len(executions), 1)
        self.assertEqual(output, {name: {"answer": 10} for name in ARCHIVE_NAMES})
        self.assertNotIn("Incomplete", summary)

    async def test_unchanged_member_bytes_do_not_complete_a_new_archive(self):
        summary, output, _, executions = await self.course(archive=True, unchanged_member=True)
        self.assertEqual(len(executions), 1)
        self.assertIsNone(output)
        self.assertIn("Incomplete", summary)

    async def test_unchanged_inherited_generator_stays_unexecuted(self):
        summary, output, _, executions = await self.course(inherited=True)
        self.assertIsNone(output)
        self.assertEqual(executions, [])
        self.assertIn("Incomplete", summary)

    async def test_expiring_task_stays_unexecuted(self):
        summary, output, _, executions = await self.course(remaining=5)
        self.assertIsNone(output)
        self.assertEqual(executions, [])
        self.assertIn("Incomplete", summary)

    async def test_unknown_attempt_is_not_repeated_without_a_new_version(self):
        summary, output, _, executions = await self.course(early_generator=True, first_unknown=True)
        self.assertIsNone(output)
        self.assertEqual(len(executions), 1)
        self.assertIn("Incomplete", summary)

    async def test_repair_can_revise_an_unsuccessful_generator(self):
        summary, output, _, executions = await self.course(early_generator=True, first_unknown=True, rewrite=True)
        self.assertEqual(output, {"answer": 10})
        self.assertEqual(len(executions), 2)
        self.assertNotIn("Incomplete", summary)

    async def test_telemetry_failure_does_not_hide_the_delivered_file(self):
        summary, output, _, _ = await self.course(final_log_failure="error")
        self.assertEqual(output, {"answer": 10})
        self.assertNotIn("Incomplete", summary)

    async def test_telemetry_stall_cannot_hold_the_final_receipt(self):
        summary, output, _, _ = await asyncio.wait_for(self.course(final_log_failure="stall"), 8)
        self.assertEqual(output, {"answer": 10})
        self.assertNotIn("Incomplete", summary)


if __name__ == "__main__":
    unittest.main()
