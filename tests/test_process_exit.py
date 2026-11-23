"""Real process-group controls for taskbridge.run_process (POSIX)."""

import asyncio
import contextlib
import fcntl
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import taskbridge


# Only the descendant keeps the locked file and any selected output pipes.
LEADER = """
import fcntl, os, subprocess, sys, time
lock_path, stdout, stderr, exit_code, delay = sys.argv[1:]
lock = open(lock_path, "w")
fcntl.flock(lock, fcntl.LOCK_EX)
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(30)"],
    stdout=None if stdout == "1" else subprocess.DEVNULL,
    stderr=None if stderr == "1" else subprocess.DEVNULL,
    pass_fds=(lock.fileno(),),
)
lock.close()
print("ready", flush=True)
print("stderr tail", file=sys.stderr, flush=True)
time.sleep(float(delay))
sys.exit(int(exit_code))
"""


class Session:
    def __init__(self, budget):
        self.budget = budget
        self.logs = []

    def time_left(self):
        return self.budget

    async def log(self, text):
        self.logs.append(text)


@unittest.skipUnless(os.name == "posix", "the harness uses POSIX process groups")
class ProcessExitTests(unittest.IsolatedAsyncioTestCase):
    async def run_group(self, stdout, stderr, code=0, delay=0, cancel=False):
        session = Session(1.0)
        ready = asyncio.Event()
        output = []

        async def receive(line):
            output.append(line)
            ready.set()

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            lock = Path(temp) / "child.lock"
            argv = [sys.executable, "-c", LEADER, str(lock), str(stdout), str(stderr),
                    str(code), str(delay)]
            task = asyncio.create_task(taskbridge.run_process(
                argv, session, on_line=receive, stop_before_sec=0))
            if cancel:
                await asyncio.wait_for(ready.wait(), 3)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 8)
                result = None
            else:
                result = await asyncio.wait_for(task, 8)
                self.assertEqual(result.stderr, "stderr tail")
            self.assertEqual(output, [b"ready"])
            # A terminated descendant releases its file lock even if init has
            # not reaped its zombie yet. Give SIGKILL a bounded delivery period.
            with lock.open() as probe:
                for attempt in range(100):
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if attempt == 99:
                            self.fail("background descendant still holds its lock")
                        await asyncio.sleep(.01)
            return result

    async def test_inherited_streams(self):
        for stdout, stderr in ((1, 0), (0, 1), (1, 1), (0, 0)):
            with self.subTest(stdout=stdout, stderr=stderr):
                result = await self.run_group(stdout, stderr)
                self.assertEqual((result.code, result.stopped), (0, False))

    async def test_nonzero_leader_with_inherited_pipes(self):
        result = await self.run_group(1, 1, code=17)
        self.assertEqual((result.code, result.stopped), (17, False))

    async def test_live_leader_times_out(self):
        result = await self.run_group(1, 1, delay=30)
        self.assertEqual((result.code, result.stopped), (-1, True))

    async def test_cancel_cleans_group(self):
        await self.run_group(1, 1, delay=30, cancel=True)

    async def test_detached_pipe_owner_survives_bounded_drain(self):
        # The detached observer inherited both streams. It proves it survived
        # cleanup by writing an acknowledgement only after the harness returns.
        child = (
            "from pathlib import Path; import sys,time\n"
            "release,ack = map(Path,sys.argv[1:])\n"
            "for _ in range(500):\n"
            " if release.exists(): ack.touch(); break\n"
            " time.sleep(.01)\n"
        )
        leader = ("import subprocess,sys; "
                  "subprocess.Popen([sys.executable,'-c',*sys.argv[1:]], "
                  "start_new_session=True); print('tail',flush=True)")
        output = []

        async def receive(line):
            output.append(line)

        with tempfile.TemporaryDirectory() as temp:
            release, ack = (Path(temp) / name for name in ("release", "ack"))
            with mock.patch.object(taskbridge, "PROCESS_DRAIN_SEC", .2):
                try:
                    result = await asyncio.wait_for(taskbridge.run_process(
                        [sys.executable, "-c", leader, child, str(release), str(ack)],
                        Session(None), on_line=receive), 2)
                    self.assertEqual((output, result.code, result.stopped), ([b"tail"], 0, False))
                finally:
                    release.touch()
            for _ in range(100):
                if ack.exists():
                    break
                await asyncio.sleep(.01)
            self.assertTrue(ack.exists(), "detached observer was terminated")

    async def test_tail_and_stdin(self):
        output = []
        async def receive(line):
            output.append(line)

        argv = [sys.executable, "-c",
                "import sys; data=sys.stdin.buffer.read(); "
                "sys.stdout.buffer.write(data); sys.stderr.write('tail')"]
        with contextlib.redirect_stderr(io.StringIO()):
            result = await taskbridge.run_process(argv, Session(None), stdin=b"last line",
                                                  on_line=receive)
        self.assertEqual((output, result.code, result.stderr, result.stopped),
                         ([b"last line"], 0, "tail", False))
