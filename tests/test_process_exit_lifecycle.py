"""Independent real-process checks for inherited output handles."""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import unittest
import taskbridge

class Session:
    def __init__(self, budget=2):
        self.deadline=time.monotonic()+budget
        self.logs=[]
    def time_left(self):
        return self.deadline-time.monotonic()
    async def log(self, text):
        self.logs.append(text)

def live(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[0]!="Z"
    except FileNotFoundError:
        return False

@unittest.skipUnless(sys.platform.startswith("linux"),"Linux process-group deployment")
class ExitLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.children=[]
    async def asyncTearDown(self):
        for pid,group in self.children:
            with contextlib.suppress(ProcessLookupError):
                if live(pid) and os.getpgid(pid)==group:
                    os.killpg(group,signal.SIGKILL)
    def source(self, pipes="both", code=0, wait=False):
        out="None" if pipes in ("stdout","both") else "subprocess.DEVNULL"
        err="None" if pipes in ("stderr","both") else "subprocess.DEVNULL"
        return ("import subprocess,sys,os,json,time\n"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
            f"stdout={out},stderr={err})\n"
            "print(json.dumps([p.pid,os.getpid()]),flush=True)\n"
            "print('answer-tail',flush=True)\n"+
            ("time.sleep(30)\n" if wait else "")+f"sys.exit({code})\n")
    def callback(self, output, ready=None):
        async def line(value):
            text=value.decode()
            output.append(text)
            if text.startswith("["):
                self.children.append(json.loads(text))
                if ready:
                    ready.set()
        return line
    async def stopped(self):
        deadline=time.monotonic()+1
        while any(live(pid) for pid,_ in self.children):
            if time.monotonic()>deadline:
                self.fail("original child remains executing")
            await asyncio.sleep(.01)
    async def run_case(self,pipes="both",code=0,wait=False,budget=2):
        output=[]; session=Session(budget)
        result=await asyncio.wait_for(taskbridge.run_process(
            [sys.executable,"-c",self.source(pipes,code,wait)],session,
            on_line=self.callback(output),stop_before_sec=.5),9)
        await self.stopped()
        self.assertIn("answer-tail",output)
        return result,session
    async def test_all_completed_pipe_variants(self):
        for pipes in ("stdout","stderr","both","neither"):
            with self.subTest(pipes=pipes):
                result,session=await self.run_case(pipes)
                self.assertEqual((result.code,result.stopped),(0,False))
                self.assertEqual(session.logs,[])
    async def test_nonzero_exit_is_not_a_timeout(self):
        result,_=await self.run_case(code=7)
        self.assertEqual((result.code,result.stopped),(7,False))
    async def test_real_deadline_still_stops_running_processes(self):
        result,_=await self.run_case(wait=True,budget=.4)
        self.assertEqual((result.code,result.stopped),(-1,True))
    async def test_stdin_and_unterminated_output_tail(self):
        output=[]
        async def line(value): output.append(value)
        result=await taskbridge.run_process(
            [sys.executable,"-c","import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())"],
            Session(),stdin=b"first\nlast-without-newline",on_line=line,stop_before_sec=.5)
        self.assertEqual((result.code,result.stopped),(0,False))
        self.assertEqual(output,[b"first",b"last-without-newline"])
    async def test_cancellation_cleans_entire_original_group(self):
        output=[]; ready=asyncio.Event()
        work=asyncio.create_task(taskbridge.run_process(
            [sys.executable,"-c",self.source(wait=True)],Session(30),
            on_line=self.callback(output,ready),stop_before_sec=.5))
        try:
            await asyncio.wait_for(ready.wait(),3)
            work.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(work,8)
            await self.stopped()
        finally:
            if not work.done():
                work.cancel()
                with contextlib.suppress(asyncio.CancelledError): await work
    async def test_actual_websocket_finishes_with_completed_result(self):
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve
        async def handler(session):
            output=[]
            result=await taskbridge.run_process(
                [sys.executable,"-c",self.source()],session,
                on_line=self.callback(output),stop_before_sec=.5)
            return json.dumps([result.code,result.stopped,output[-1]])
        async def connected(ws): await taskbridge._session(ws,handler)
        async with serve(connected,"127.0.0.1",0,ping_interval=None) as server:
            port=server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({"type":"task","instruction":"Local lifecycle control",
                    "limits":{"agent_timeout_sec":2}}))
                while True:
                    frame=json.loads(await asyncio.wait_for(ws.recv(),9))
                    if frame["type"]=="log": continue
                    self.assertEqual(frame["type"],"done")
                    self.assertEqual(json.loads(frame["final_message"]),[0,False,"answer-tail"])
                    break
        await self.stopped()

if __name__=="__main__": unittest.main()
