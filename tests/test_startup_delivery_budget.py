"""Run the real delivery preflights against a deterministic in-memory task protocol."""
import contextlib
import json
import shlex
from types import SimpleNamespace
import unittest
from unittest import mock

import archive_contract
import delivery_guard
import harness
import taskbridge


IFC = "Write the model to /app/output/model.ifc"
ARCHIVE = ("Save the final deliverable as `/app/output/submission.zip`. Write one file, "
           "`/app/output/submission.zip`, containing at its root:\n"
           "### 1. `summary.txt`\n### 2. `data.csv`\n\nFinish a first complete version, then improve.")


class FirstRun(Exception):
    """Stop as soon as the participant receives the available budget."""


class TaskProtocol:
    def __init__(self, instruction, left, costs=(), grade_ok=True):
        self.instruction, self.left = instruction, left
        self.workdir, self.mcp_servers = "/app", []
        self.now, self.origin = 1000.0, 1000.0
        self.costs, self.grade_ok = iter(costs), grade_ok
        self.calls, self.allowance = [], None

    def time_left(self):
        return None if self.left is None else self.left - (self.now - self.origin)

    async def wait_for(self, pending, timeout):
        self.allowance = timeout
        try:
            return await pending
        finally:
            self.allowance = None

    def charge(self, kind, remote=None):
        self.calls.append((kind, remote, self.allowance))
        planned = next(self.costs)
        permitted = self.allowance if remote is None else min(remote, self.allowance)
        self.now += min(planned, permitted)
        if planned > permitted:
            raise TimeoutError

    async def exec(self, command, *, timeout_sec):
        targets = json.loads(shlex.split(command)[-1])
        kind = "generators" if targets[0]["path"].endswith(".py") else "graded"
        self.charge(kind, timeout_sec)
        # An empty JSON object is a known observation. An error is unknown.
        return {"exit_code": 0 if self.grade_ok else 1, "output": "{}"}

    async def write_file(self, path, data):
        self.charge("archive")
        return len(data)


class StartupDeliveryBudget(unittest.IsolatedAsyncioTestCase):
    def clock(self, stack, task):
        for module in (harness, archive_contract, delivery_guard):
            stack.enter_context(mock.patch.object(module, "monotonic", side_effect=lambda: task.now))
        for module in (archive_contract, delivery_guard):
            stack.enter_context(mock.patch.object(module, "asyncio", SimpleNamespace(wait_for=task.wait_for)))

    async def first_run(self, task):
        async def participant(_, session, *, stop_before_sec, **kwargs):
            raise FirstRun(taskbridge.time_share(session.time_left(), stop_before_sec))
        relay = mock.MagicMock()
        relay.__aenter__.return_value = SimpleNamespace(env=lambda: {})
        with contextlib.ExitStack() as stack:
            self.clock(stack, task)
            stack.enter_context(mock.patch.object(taskbridge, "Relay", return_value=relay))
            stack.enter_context(mock.patch.object(taskbridge, "run_process", new=participant))
            stack.enter_context(mock.patch.object(harness, "configure", return_value=("fixture", "fixture")))
            for name in ("intake_for_prompt", "provision_engineering_cli"):
                stack.enter_context(mock.patch.object(harness.engineering_intake, name, return_value=""))
            with self.assertRaises(FirstRun) as stopped:
                await harness.solve(task)
        return stopped.exception.args[0]

    async def test_consecutive_startup_operations_leave_the_participant_time(self):
        cases = [
            (30, IFC, (15, 15), [("graded", 18, 18), ("generators", 3, 3)], 9),
            (35, IFC, (20, 20), [("graded", 23, 23), ("generators", 3, 3)], 9),
            (45, ARCHIVE, (20, 8, 20), [("graded", 25, 29), ("archive", None, 9),
                                        ("generators", 1, 1)], 12),
        ]
        for left, instruction, costs, expected, share in cases:
            with self.subTest(left=left):
                task = TaskProtocol(instruction, left, costs)
                self.assertEqual(await self.first_run(task), share)
                self.assertEqual(task.calls, expected)

    async def test_unknown_clock_shares_one_29_second_allowance(self):
        task = TaskProtocol(ARCHIVE, None, (19, 9.5, 50))
        self.assertIsNone(await self.first_run(task))
        # Half a second remains after upload. The integer remote limit
        # cannot represent it, so there is no generator exec.
        self.assertEqual(task.calls, [("graded", 25, 29), ("archive", None, 10)])
        self.assertEqual(task.now - task.origin, 28.5)

    async def test_failed_graded_observation_cannot_start_other_preflights(self):
        for instruction in (IFC, ARCHIVE):
            with self.subTest(instruction=instruction):
                task = TaskProtocol(instruction, 35, (1, 1, 1), grade_ok=False)
                self.assertGreater(await self.first_run(task), 0)
                self.assertEqual(task.calls, [("graded", 23, 23)])

    async def test_shortest_deadlines_and_text_have_no_remote_startup_call(self):
        for instruction, left in ((IFC, 12.75), (ARCHIVE, 11), ("Explain the result in text", 30)):
            with self.subTest(left=left):
                task = TaskProtocol(instruction, left)
                self.assertGreater(await self.first_run(task), 0)
                self.assertEqual(task.calls, [])

    async def test_fractional_snapshot_and_post_run_defaults(self):
        task = TaskProtocol(ARCHIVE, None, (0, 0, 0))
        targets = delivery_guard.extract_targets(ARCHIVE, task.workdir)
        with contextlib.ExitStack() as stack:
            self.clock(stack, task)
            self.assertEqual(await delivery_guard.snapshot(task, [], deadline=task.now), {})
            self.assertIsNone(await delivery_guard.snapshot(task, targets, deadline=task.now + 0.5))
            self.assertEqual(await delivery_guard.snapshot(task, targets, deadline=task.now + 3.75), {})
            self.assertEqual(await delivery_guard.snapshot(task, targets), {})
            self.assertIsNotNone(await archive_contract.provision(task, targets))
        self.assertEqual(task.calls, [("graded", 3, 3.75), ("graded", 25, 29), ("archive", None, 12)])
