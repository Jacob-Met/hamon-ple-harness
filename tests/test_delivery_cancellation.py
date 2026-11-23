"""Optional delivery operations preserve cancellation and ordinary failure."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import archive_contract
import delivery_guard
import delivery_materialize
import taskbridge


class DeliveryCancellation(unittest.IsolatedAsyncioTestCase):
    def operations(self, request):
        session = SimpleNamespace(
            instruction=(
                "Write a first complete version of output/submission.zip, then improve it. "
                "Produce exactly these two files at the archive root:\n"
                "1. `one.md`\n2. `two.md`"
            ),
            workdir="/app", time_left=lambda: None,
            exec=request, write_file=request,
        )
        targets = [{"path": "/app/output/submission.zip", "kind": "file"}]
        selected = [{"script": "/app/output/build.py", "target": "/app/output/model.ifc"}]
        return (
            ("snapshot", lambda: delivery_guard.snapshot(session, targets), None),
            ("archive provision", lambda: archive_contract.provision(session, targets), None),
            ("archive publication", lambda: archive_contract.run_command(session, "fixture"), False),
            ("materialize", lambda: delivery_materialize.materialize(session, selected),
             [{"target": "/app/output/model.ifc", "state": "unknown"}]),
        )

    async def test_both_cancellation_types_propagate(self):
        for exception in (taskbridge.Cancelled, asyncio.CancelledError):
            request = AsyncMock(side_effect=exception())
            for name, operation, _ in self.operations(request):
                with self.subTest(operation=name, exception=exception.__name__):
                    request.reset_mock()
                    with self.assertRaises(exception):
                        await operation()
                    request.assert_awaited_once()

    async def test_ordinary_protocol_failures_remain_optional(self):
        request = AsyncMock(side_effect=taskbridge.TaskError("operation unavailable"))
        for name, operation, fallback in self.operations(request):
            with self.subTest(operation=name):
                request.reset_mock()
                self.assertEqual(await operation(), fallback)
                request.assert_awaited_once()

    async def test_cancel_wins_when_the_handler_already_failed_during_shutdown(self):
        import json
        from unittest.mock import patch

        socket = AsyncMock()
        socket.recv.return_value = json.dumps({"type": "task", "instruction": "Write output/report.docx."})

        async def cancel(session):
            session.cancelled.set()

        async def failed_cleanup(session):
            await session.cancelled.wait()
            session.fallback_summary = lambda _: "Unverified report after cancellation"
            raise RuntimeError("stream closed")

        with patch.object(taskbridge.Session, "_read", cancel):
            await taskbridge._session(socket, failed_cleanup)
        messages = [json.loads(call.args[0]) for call in socket.send.await_args_list]
        self.assertEqual(messages, [{"type": "done", "final_message": "Unverified report after cancellation"}])
