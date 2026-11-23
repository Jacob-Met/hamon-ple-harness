"""Exercise the remote materializer lifecycle with an instrumented POSIX child."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from delivery_materialize import REMOTE_MATERIALIZE


@unittest.skipUnless(hasattr(os, "WNOWAIT"), "the task materializer runs on POSIX")
class MaterializerLifecycle(unittest.TestCase):
    def exercise(self, exit_status, *, launch_error=False):
        script = "/app/files/project/solution.py"
        target = "/app/files/project/result.pdf"
        argv = [sys.executable, script, target]
        events = []
        now = 0.0
        pid = 18421

        def spawn(args, **kwargs):
            self.assertEqual(args, argv)
            self.assertEqual(kwargs["cwd"], str(Path(script).parent))
            self.assertTrue(kwargs.get("start_new_session", False))
            self.assertFalse(kwargs["shell"])
            for stream in ("stdin", "stdout", "stderr"):
                self.assertEqual(kwargs[stream], subprocess.DEVNULL)
            events.append("spawn")
            if launch_error:
                raise OSError("cannot start interpreter")
            return SimpleNamespace(pid=pid, args=args, wait=lambda: events.append("reap"))

        def observe(kind, ident, flags):
            self.assertEqual((kind, ident), (os.P_PID, pid))
            self.assertEqual(flags, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exit_status is None:
                return None
            code, status = exit_status
            events.append("observe")
            return SimpleNamespace(si_code=code, si_status=status)

        def signal_group(ident, sig):
            self.assertEqual(ident, pid)
            events.append(sig)

        def record_file_check(path):
            if str(path) == target:
                events.append("check target")
            return True

        def advance(seconds):
            nonlocal now
            self.assertGreater(seconds, 0)
            now += seconds

        stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(subprocess, "Popen", spawn))
            stack.enter_context(patch.object(os, "waitid", observe))
            stack.enter_context(patch.object(os, "killpg", signal_group))
            stack.enter_context(patch.object(time, "monotonic", lambda: now))
            stack.enter_context(patch.object(time, "sleep", advance))
            stack.enter_context(patch.object(Path, "is_file", record_file_check))
            stack.enter_context(patch.object(Path, "is_symlink", return_value=False))
            stack.enter_context(patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1)))
            spec = json.dumps({"script": script, "target": target, "timeout": 1})
            stack.enter_context(patch.object(sys, "argv", ["probe", spec]))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            exec(compile(REMOTE_MATERIALIZE, "<remote materializer>", "exec"), {})
        return json.loads(stdout.getvalue()), events

    def assert_cleanup_precedes_result(self, events):
        self.assertEqual(events.count("spawn"), 1)
        self.assertEqual(events.count("reap"), 1)
        self.assertLess(events.index(signal.SIGTERM), events.index(signal.SIGKILL))
        self.assertLess(events.index(signal.SIGKILL), events.index("reap"))
        if "check target" in events:
            self.assertLess(events.index("reap"), events.index("check target"))

    def test_completion_is_observed_before_group_cleanup_and_target_check(self):
        for signal_code, status, expected_code in (
            (os.CLD_EXITED, 0, 0),
            (os.CLD_EXITED, 7, 7),
            (os.CLD_KILLED, signal.SIGTERM, -signal.SIGTERM),
        ):
            with self.subTest(status=status):
                result, events = self.exercise((signal_code, status))
                self.assertEqual(result["state"], "executed" if expected_code == 0 else "nonzero")
                self.assertEqual(result["exit_code"], expected_code)
                self.assertTrue(result["target_created"])
                self.assertLess(events.index("observe"), events.index(signal.SIGTERM))
                self.assert_cleanup_precedes_result(events)

    def test_timeout_escalates_and_reaps_before_returning(self):
        result, events = self.exercise(None)
        self.assertEqual(result["state"], "timeout")
        self.assertGreaterEqual(result["wall_ms"], 1000)
        self.assertLessEqual(result["wall_ms"], 2000)
        self.assertNotIn("check target", events)
        self.assert_cleanup_precedes_result(events)

    def test_failed_launch_does_not_signal_a_group(self):
        result, events = self.exercise(None, launch_error=True)
        self.assertEqual(result, {"state": "exec_error"})
        self.assertEqual(events, ["spawn"])


if __name__ == "__main__":
    unittest.main()
