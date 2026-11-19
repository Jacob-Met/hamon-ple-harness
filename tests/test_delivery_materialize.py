"""Locked, task-independent generator promotion and negative controls.

The full historical IFC generators also run in ship/replay_public_generators.py
under the separate offline CI workflow, in scratch with the real IFC parser.
"""
import asyncio
import json
import shlex
import unittest
import delivery_materialize as m


class CandidateContracts(unittest.TestCase):
    def test_mep_ifc_prioritized_once_per_directory(self):
        items = m.candidates([
            {"path": "/app/files/hall/output/reconciliation.json", "kind": "file"},
            {"path": "/app/files/hall/output/model.ifc", "kind": "file"},
        ])
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0], {
            "script": "/app/files/hall/output/solution.py",
            "target": "/app/files/hall/output/model.ifc",
        })
        self.assertEqual(items[1]["script"], "/app/files/hall/output/generate.py")
        self.assertEqual(len(m.snapshot_targets(items)), 3)

    def test_distinct_projects_never_coalesce(self):
        paths = ["/app/files/a/output/model.ifc", "/app/files/b/output/model.ifc"]
        items = m.candidates([{"path": p, "kind": "file"} for p in paths])
        self.assertEqual({x["target"] for x in items}, set(paths))

    def test_directory_and_relative_paths_are_never_run(self):
        targets = [
            {"path": "/filesystem/output", "kind": "dir"},
            {"path": "/filesystem/output/../../keys.zip", "kind": "file"},
            {"path": "/var/secret/model.ifc", "kind": "file"},
            {"path": "output/model.ifc", "kind": "file"},
            {"path": "/app/files/task/output/model.py", "kind": "file"},
        ]
        self.assertEqual(m.candidates(targets), [])

    def test_unchanged_inherited_script_does_not_run(self):
        s = "/app/files/project/output/solution.py"
        old = {s: {"state": "ready", "fingerprint": "a"}}
        self.assertFalse(m.newly_written(old, old, s))
        self.assertFalse(m.newly_written(None, old, s))
        self.assertFalse(m.newly_written({s: {"state": "unknown"}}, old, s))
        self.assertTrue(m.newly_written(old, {
            s: {"state": "ready", "fingerprint": "b"}}, s))

    def test_new_script_materializes_missing_exact_target(self):
        dest = "/app/files/hall/output/model.ifc"
        items = m.candidates([{"path": dest, "kind": "file"}])
        chosen = m.plan(
            items, [{"path": dest, "reason": "missing"}],
            {x["script"]: {"state": "missing"} for x in items},
            {x["script"]: {"state": "ready", "fingerprint": "new"}
             for x in items[:1]},
        )
        self.assertEqual(chosen, [items[0]])

    def test_no_remaining_failure_does_not_run(self):
        dest = "/app/files/hall/output/model.ifc"
        items = m.candidates([{"path": dest, "kind": "file"}])
        before = {x["script"]: {"state": "missing"} for x in items}
        after = {items[0]["script"]: {"state": "ready", "fingerprint": "a"}}
        self.assertEqual(m.plan(items, [], before, after), [])

    def test_failed_first_script_can_use_second(self):
        dest = "/app/files/hall/output/model.ifc"
        items = m.candidates([{"path": dest, "kind": "file"}])
        before = {x["script"]: {"state": "missing"} for x in items}
        after = {items[1]["script"]: {"state": "ready", "fingerprint": "fresh"}}
        self.assertEqual(m.plan(
            items, [{"path": dest, "reason": "missing"}], before, after), [items[1]])

    def test_new_unknown_or_symlink_is_not_sufficient(self):
        dest = "/app/files/hall/output/model.ifc"
        items = m.candidates([{"path": dest, "kind": "file"}])
        bad = {items[0]["script"]: {"state": "invalid", "reason": "symlink_path"}}
        self.assertEqual(m.plan(
            items, [{"path": dest, "reason": "missing"}],
            {items[0]["script"]: {"state": "missing"}}, bad), [])

    def test_one_generator_can_emit_multiple_sibling_outputs(self):
        targets = [{"path": f"/app/files/task/output/{name}", "kind": "file"}
                   for name in ("model.ifc", "reconciliation.json")]
        items = m.candidates(targets)
        before = {i["script"]: {"state": "missing"} for i in items}
        after = {items[0]["script"]: {"state": "ready", "fingerprint": "new"}}
        chosen = m.plan(items, [
            {"path": t["path"], "reason": "missing"} for t in targets
        ], before, after)
        self.assertEqual(len(chosen), 1)


class RemoteProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_uses_task_exec_not_host_subprocess(self):
        class FakeSession:
            def __init__(self):
                self.commands = []
            def time_left(self):
                return 1000
            async def exec(self, command, timeout_sec):
                self.commands.append((command, timeout_sec))
                return {"exit_code": 0,
                        "output": '{"state":"executed","target_created":true}'}
        s = FakeSession()
        item = {"script": "/app/files/hall/output/solution.py",
                "target": "/app/files/hall/output/model.ifc"}
        result = await m.materialize(s, [item])
        self.assertEqual(result[0]["state"], "executed")
        self.assertEqual(len(s.commands), 1)
        argv = shlex.split(s.commands[0][0])
        self.assertEqual(argv[:2], ["python3", "-c"])
        options = json.loads(argv[3])
        self.assertEqual(options["target"], item["target"])
        self.assertIn("subprocess.run(", argv[2])
        self.assertIn("shell=False", argv[2])

    async def test_protocol_task_error_fails_open(self):
        class NotAvailable(Exception):
            pass
        class FakeSession:
            def time_left(self):
                return 1000
            async def exec(self, command, timeout_sec):
                raise NotAvailable("offline evaluator task protocol")
        item = {"script": "/app/files/hall/output/solution.py",
                "target": "/app/files/hall/output/model.ifc"}
        self.assertEqual(await m.materialize(FakeSession(), [item]),
                         [{"target": item["target"], "state": "unknown"}])

    async def test_short_deadline_does_not_execute(self):
        class FakeSession:
            def time_left(self):
                return 99
            async def exec(self, command, timeout_sec):
                self.fail("No execution should occur after deadline gate")
        self.assertEqual(await m.materialize(
            FakeSession(), [{"script": "/app/files/hall/output/solution.py",
                             "target": "/app/files/hall/output/model.ifc"}]), [])


if __name__ == "__main__":
    unittest.main()
