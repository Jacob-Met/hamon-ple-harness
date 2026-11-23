"""A complete graded archive must survive an interrupted optional refinement."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import archive_checkpoint
import harness


class ArchiveRefinementRecovery(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, change, *, tool_tail=False, checkpoint_ok=True):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage, target = root / "members", root / "answer.zip"
            members = ["report.md", "result.json"]
            targets = [{"kind": "file", "path": str(target)}]
            plan = (str(stage), "fixture-publisher")
            calls, commands = [], []
            first = None

            async def execute(command, timeout_sec):
                commands.append(command)
                try:
                    if command.endswith(" --verify"):
                        archive_checkpoint.verify_checkpoint(stage, target, members)
                        action = None
                    elif command.endswith(" --checkpoint"):
                        if not checkpoint_ok:
                            return {"exit_code": 1, "output": "checkpoint unavailable"}
                        archive_checkpoint.checkpoint(stage, target, members)
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
                time_left=lambda: 900, exec=execute,
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


if __name__ == "__main__":
    unittest.main()
