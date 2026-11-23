"""A complete graded archive must survive an interrupted optional refinement."""
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import archive_checkpoint
import harness


class ArchiveRefinementRecovery(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, change, *, tool_tail=False, checkpoint_ok=True, after_checkpoint=900,
                       refinement_error=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage, target = root / "members", root / "answer.zip"
            members = ["report.md", "result.json"]
            targets = [{"kind": "file", "path": str(target)}]
            plan = (str(stage), "fixture-publisher")
            calls, commands = [], []
            first = None
            remaining = 900

            async def execute(command, timeout_sec):
                nonlocal remaining
                commands.append(command)
                try:
                    if command.endswith(" --verify"):
                        archive_checkpoint.verify_checkpoint(stage, target, members)
                        action = None
                    elif command.endswith(" --checkpoint"):
                        if not checkpoint_ok:
                            return {"exit_code": 1, "output": "checkpoint unavailable"}
                        archive_checkpoint.checkpoint(stage, target, members)
                        remaining = after_checkpoint
                        action = None
                    elif command.endswith(" --recover"):
                        action = archive_checkpoint.recover(stage, target, members)
                    else:
                        archive_checkpoint.publish(stage, target, members)
                        action = None
                    return {"exit_code": 0, "output": json.dumps({"action": action})}
                except (ValueError, OSError, zipfile.BadZipFile) as exc:
                    return {"exit_code": 1, "output": str(exc)}

            session = SimpleNamespace(
                workdir=str(root), instruction="Write an early complete archive, then improve it.",
                mcp_servers=[], final_message="", log=mock.AsyncMock(), usage=mock.AsyncMock(),
                time_left=lambda: remaining, exec=execute,
            )

            async def process(argv, _, *, on_line, log_prefix, **kwargs):
                nonlocal first
                calls.append(log_prefix)
                if log_prefix == "pi: ":
                    stage.mkdir()
                    (stage / "report.md").write_text("# First complete engineering report\n")
                    (stage / "result.json").write_text('{"result": 12}\n')
                    archive_checkpoint.publish(stage, target, members)
                    first = target.read_bytes()
                elif log_prefix == "pi-archive-refine: ":
                    if change in {"corrupt", "both_invalid"}:
                        target.write_bytes(b"PK\x03\x04interrupted replacement")
                    elif change == "missing":
                        target.unlink()
                    if change in {"stage_invalid", "both_invalid"}:
                        (stage / "result.json").write_text('{"unfinished":')
                    elif change == "improved":
                        (stage / "report.md").write_text("# Improved engineering report\n")
                        archive_checkpoint.publish(stage, target, members)
                else:
                    self.fail("The optional refinement must not start an unproved third model run")
                if len(calls) > 1 and refinement_error is not None:
                    raise refinement_error
                if tool_tail and log_prefix != "pi: ":
                    message = {"role": "assistant", "stopReason": "toolUse", "content": [
                        {"type": "toolCall", "id": "pending", "name": "bash", "arguments": {}}]}
                else:
                    message = {"role": "assistant", "stopReason": "stop", "content": [
                        {"type": "text", "text": "Refinement complete" if len(calls) > 1 else "First complete"}]}
                await on_line(json.dumps({"type": "message_end", "message": message}).encode())
                return SimpleNamespace(code=0, stderr="")

            async def snapshot(_, requested, *, deadline=None):
                if requested != targets:
                    return {}
                if not target.exists():
                    row = {"state": "missing"}
                else:
                    try:
                        archive_checkpoint.verify(target, members)
                        row = {"state": "ready", "fingerprint": hashlib.sha256(target.read_bytes()).hexdigest()}
                    except (ValueError, zipfile.BadZipFile):
                        row = {"state": "invalid", "reason": "invalid archive"}
                return {str(target): row}

            relay = mock.MagicMock()
            relay.__aenter__.return_value = SimpleNamespace(env=lambda: dict(os.environ))
            with contextlib.ExitStack() as stack:
                for obj, name, options in (
                    (harness.taskbridge, "Relay", {"return_value": relay}),
                    (harness.taskbridge, "run_process", {"side_effect": process}),
                    (harness, "configure", {"return_value": ("fixture", "fixture")}),
                    (harness.engineering_intake, "intake_for_prompt", {"return_value": ""}),
                    (harness.engineering_intake, "provision_engineering_cli", {"return_value": ""}),
                    (harness.delivery_guard, "extract_targets", {"return_value": targets}),
                    (harness.delivery_guard, "snapshot", {"side_effect": snapshot}),
                    (harness.archive_contract, "provision", {"return_value": plan}),
                ):
                    stack.enter_context(mock.patch.object(obj, name, **options))
                reply = await harness.solve(session)
            final = target.read_bytes() if target.exists() else None
            state = (await snapshot(session, targets))[str(target)]["state"]
            return dict(calls=calls, commands=commands, first=first, final=final, state=state, reply=reply)

    async def test_failed_refinement_recovers_the_complete_archive(self):
        for change in ("corrupt", "missing", "both_invalid"):
            with self.subTest(change=change):
                result = await self.exercise(change)
                self.assertEqual(result["state"], "ready")
                self.assertEqual(result["final"], result["first"])
                self.assertEqual(len(result["calls"]), 2)
                self.assertIn("Recovered a complete archive", result["reply"])
                self.assertNotIn("Refinement complete", result["reply"])

    async def test_successful_or_unpublished_refinement_keeps_the_valid_output(self):
        for change in ("unchanged", "stage_invalid", "improved"):
            with self.subTest(change=change):
                result = await self.exercise(change)
                self.assertEqual(result["state"], "ready")
                self.assertEqual(result["final"] == result["first"], change != "improved")

    async def test_unacknowledged_refinement_tail_can_recover_without_model_replay(self):
        result = await self.exercise("both_invalid", tool_tail=True)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["final"], result["first"])
        self.assertEqual(len(result["calls"]), 2)
        self.assertIn("before giving a final answer", result["reply"])

    async def test_failed_checkpoint_preserves_the_first_pass(self):
        result = await self.exercise("both_invalid", checkpoint_ok=False)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["final"], result["first"])
        self.assertEqual(len(result["calls"]), 1)


    async def test_ordinary_refinement_error_enters_recovery(self):
        result = await self.exercise("both_invalid", refinement_error=OSError("fixture"))
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["final"], result["first"])
        self.assertEqual(len(result["calls"]), 2)
        self.assertIn("Recovered a complete archive", result["reply"])

    async def test_task_cancellation_does_not_start_recovery(self):
        with mock.patch.object(harness.archive_contract, "recover_after_refinement") as recover:
            with self.assertRaises(harness.taskbridge.Cancelled):
                await self.exercise("both_invalid", refinement_error=harness.taskbridge.Cancelled("fixture"))
            recover.assert_not_awaited()

    def test_csv_parser_failure_restores_the_saved_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, target = Path(folder) / "members", Path(folder) / "answer.zip"
            stage.mkdir()
            (stage / "data.csv").write_text("value\n12\n")
            archive_checkpoint.publish(stage, target, ["data.csv"])
            archive_checkpoint.checkpoint(stage, target, ["data.csv"])
            first = target.read_bytes()
            invalid = "x" * 80 + "\n"
            (stage / "data.csv").write_text(invalid)
            with zipfile.ZipFile(target, "w") as output:
                output.writestr("data.csv", invalid)
            old_limit = csv.field_size_limit(48)
            try:
                self.assertEqual(archive_checkpoint.recover(stage, target, ["data.csv"]), "restored")
            finally:
                csv.field_size_limit(old_limit)
            self.assertEqual(target.read_bytes(), first)

    def test_checkpoint_validates_the_copied_version_before_replacing_the_saved_one(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, target = Path(folder) / "members", Path(folder) / "answer.zip"
            stage.mkdir()
            (stage / "report.md").write_text("# First version\n")
            archive_checkpoint.publish(stage, target, ["report.md"])
            archive_checkpoint.checkpoint(stage, target, ["report.md"])
            first = target.read_bytes()
            copy_file = archive_checkpoint.shutil.copyfile
            def replace_then_copy(source, destination):
                with zipfile.ZipFile(target, "w") as output:
                    output.writestr("report.md", "# Different valid version\n")
                return copy_file(source, destination)
            with mock.patch.object(archive_checkpoint.shutil, "copyfile", side_effect=replace_then_copy):
                with self.assertRaisesRegex(ValueError, "staged and published versions differ"):
                    archive_checkpoint.checkpoint(stage, target, ["report.md"])
            target.write_bytes(b"incomplete")
            (stage / "report.md").write_text("")
            self.assertEqual(archive_checkpoint.recover(stage, target, ["report.md"]), "restored")
            self.assertEqual(target.read_bytes(), first)

    async def test_standalone_task_helper_restores_the_saved_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target, stage = root / "answer.zip", root / "members"
            prompt = ("First create a complete version, then improve it. Write answer.zip "
                      "containing exactly these two files at the archive root:\n"
                      "1. `report.md`\n2. `result.json`\n")
            async def write_file(path, body):
                return (root / Path(path).name).write_bytes(body)
            session = SimpleNamespace(instruction=prompt, write_file=write_file)
            plan = await harness.archive_contract.provision(session, [{"kind": "file", "path": str(target)}])
            self.assertIsNotNone(plan)
            _, helper, arguments = shlex.split(plan[1])
            _, _, members = json.loads(arguments)
            stage.mkdir()
            (stage / "report.md").write_text("# Complete report\n")
            (stage / "result.json").write_text('{"result":12}')
            command = [sys.executable, "-I", "-S", str(root / Path(helper).name),
                       json.dumps([str(stage), str(target), members])]
            subprocess.run(command, check=True, capture_output=True)
            first = target.read_bytes()
            subprocess.run(command + ["--checkpoint"], check=True, capture_output=True)
            target.write_bytes(b"incomplete ZIP")
            (stage / "result.json").write_text('{"unfinished":')
            restored = subprocess.run(command + ["--recover"], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(restored.stdout), {"action": "restored"})
            self.assertEqual(target.read_bytes(), first)
            archive_checkpoint.verify(target, members)

    async def test_checkpoint_time_is_removed_from_the_refinement_budget(self):
        for remaining, expected_runs in ((79.999, 1), (80, 2)):
            with self.subTest(remaining=remaining):
                result = await self.exercise("both_invalid", after_checkpoint=remaining)
                self.assertEqual(result["state"], "ready")
                self.assertEqual(result["final"], result["first"])
                self.assertEqual(len(result["calls"]), expected_runs)

    async def test_unknown_recovery_and_cancellation_do_not_report_success(self):
        for result in ({"exit_code": 1}, {"exit_code": 0, "output": "not json"},
                       {"exit_code": 0, "output": '{"action":"restored"}', "truncated": True},
                       {"exit_code": 0, "output": '{"action":"unknown"}'},
                       {"exit_code": 0, "output": '{"action":[]}'}):
            with self.subTest(result=result):
                session = SimpleNamespace(exec=mock.AsyncMock(return_value=result))
                self.assertIsNone(await harness.archive_contract.recover_after_refinement(session, "publisher"))
        session = SimpleNamespace(exec=mock.AsyncMock(side_effect=harness.taskbridge.Cancelled("fixture")))
        with self.assertRaises(harness.taskbridge.Cancelled):
            await harness.archive_contract.recover_after_refinement(session, "publisher")


if __name__ == "__main__":
    unittest.main()
