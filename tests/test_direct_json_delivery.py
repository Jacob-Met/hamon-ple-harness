"""Observe real terminal messages and loopback-model requests for complete JSON instructions."""
import asyncio
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import engineering_intake
import harness
import taskbridge
from mock_sse_proxy import MockStream


SINGLE = "Run the calculation below and return its JSON result."
BATCH = "Run the calculations below and return their JSON results in the same order."
FALLBACK = "loopback participant response"


def case(delta):
    return {"mode": "civil_geotech", "spec": {
        "calculation": "consolidation_1d", "H_clay_m": 5, "e0": 0.8,
        "Cc": 0.25, "sigma_v0_eff_Pa": 100000, "delta_sigma_v_Pa": delta,
        "cv_m2_s": 1e-7, "time_s": 31536000, "drainage": "double",
    }}


def control_case(a, **extra):
    return {"mode": "control", "spec": {
        "mode": "state_space", "timebase": "discrete", "A": a, **extra,
    }}


async def exchange(port, workdir, instruction):
    probe = "cd " + shlex.quote(str(workdir)) + " && python3 -c " + shlex.quote(engineering_intake.REMOTE_PROBE)
    counts = {"probes": 0, "writes": 0, "usage": 0}
    async with connect(f"ws://127.0.0.1:{port}") as ws, asyncio.timeout(90):
        await ws.send(json.dumps({"type": "task", "instruction": instruction, "workdir": str(workdir),
                                  "limits": {"agent_timeout_sec": 400}}))
        async for wire in ws:
            msg = json.loads(wire)
            kind = msg["type"]
            if kind == "done":
                return msg["final_message"], counts
            if kind == "error":
                raise AssertionError(msg)
            if kind == "usage":
                counts["usage"] += 1
            if "id" not in msg:
                continue
            reply = {"type": "result", "id": msg["id"], "ok": False, "error": "unneeded fixture operation"}
            if kind == "exec" and msg["command"] == probe:
                counts["probes"] += 1
                proc = await asyncio.create_subprocess_exec(sys.executable, "-c", engineering_intake.REMOTE_PROBE,
                    cwd=workdir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await proc.communicate()
                if proc.returncode:
                    raise AssertionError(err.decode())
                reply.update(ok=True, exit_code=0, output=out.decode(), timed_out=False)
            if kind == "write_file":
                # Refuse optional tool provisioning in this fixture. The fallback must still call Pi.
                counts["writes"] += 1
            await ws.send(json.dumps(reply))
    raise AssertionError("missing terminal message")


class DirectJsonDelivery(unittest.IsolatedAsyncioTestCase):
    async def test_answer_and_provider_calls(self):
        async def entry(ws):
            await taskbridge._session(ws, harness.solve)

        stream = MockStream(reply_text=FALLBACK)
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            folder = Path(tmp) / "task"
            home.mkdir()
            folder.mkdir()
            async with await asyncio.start_server(stream.respond, "127.0.0.1", 0) as model:
                url = f"http://127.0.0.1:{model.sockets[0].getsockname()[1]}"
                env = {"HOME": str(home), "PATH": os.environ["PATH"], "NO_PROXY": "127.0.0.1,localhost",
                       "MODEL_NAME": "claude-sonnet-4-5", "ANTHROPIC_API_KEY": "fixture", "ANTHROPIC_BASE_URL": url}
                with patch.dict(os.environ, env, clear=True):
                    async with serve(entry, "127.0.0.1", 0) as server:
                        port = server.sockets[0].getsockname()[1]
                        for cover, loads in ((SINGLE, [50000]), (BATCH, [60000, 50000]), (BATCH, [50000, 50000])):
                            inputs = [case(d) for d in loads]
                            data = inputs[0] if cover == SINGLE else inputs
                            answer, counts = await exchange(port, folder, cover + "\n" + json.dumps(data))
                            self.assertEqual(counts, {"probes": 1, "writes": 0, "usage": 0})
                            self.assertEqual(stream.requests, 0)
                            results = [json.loads(answer)] if cover == SINGLE else json.loads(answer)
                            self.assertEqual(len(results), len(loads))
                            # Independent double-drainage Terzaghi series and logarithmic compression.
                            tv = 1e-7 * 31536000 / 2.5 ** 2
                            u = 1 - sum(8 / ((2*k + 1)*math.pi)**2 *
                                        math.exp(-((2*k + 1)*math.pi)**2 * tv/4) for k in range(40))
                            for delta, row in zip(loads, results):
                                self.assertIs(row["conservation_verified"], True)
                                settlement = u*5*0.25/1.8*math.log10(1 + delta/100000)
                                self.assertTrue(math.isclose(row["settlement_at_time_m"], settlement, rel_tol=1e-8))

                        # Published CTMS sampled plant and SciPy's documented
                        # discrete Lyapunov example, transposed for A^T P A.
                        import numpy as np
                        from scipy.linalg import solve_discrete_lyapunov
                        ctms = control_case([[.9990, .0095], [-.1903, .9039]], sample_time_s=.01)
                        scipy_case = control_case([[.2, .7], [.5, -.9]])
                        for data, stable in ((ctms, True), (scipy_case, False)):
                            answer, counts = await exchange(port, folder, SINGLE + "\n" + json.dumps(data))
                            self.assertEqual(counts, {"probes": 1, "writes": 0, "usage": 0})
                            self.assertEqual(stream.requests, 0)
                            row = json.loads(answer)
                            self.assertIs(row["schur_stable"], stable)
                            self.assertNotIn("hurwitz_stable", row)
                            expected = solve_discrete_lyapunov(np.array(data["spec"]["A"]).T, np.eye(2))
                            np.testing.assert_allclose(row["lyapunov_P"], expected, rtol=1e-10, atol=1e-12)
                        answer, counts = await exchange(port, folder, BATCH + "\n" + json.dumps([scipy_case, ctms]))
                        self.assertEqual(counts, {"probes": 1, "writes": 0, "usage": 0})
                        self.assertEqual(stream.requests, 0)
                        self.assertEqual([r["schur_stable"] for r in json.loads(answer)], [False, True])

                        fallback_controls = (
                            control_case([[.1, .9], [.3, .7]]),  # decimal unit pole
                            control_case([[2, 0], [0, .5]]),  # nonunique Stein solution
                            control_case([[.5]], timebase="continuous"),
                            control_case([[.5]], claimed_field="schur_stable", claimed_value=False),
                        )
                        for number, data in enumerate(fallback_controls, 1):
                            answer, counts = await exchange(port, folder, SINGLE + "\n" + json.dumps(data))
                            self.assertEqual(answer, FALLBACK)
                            self.assertEqual(stream.requests, number)
                            self.assertGreater(counts["usage"], 0)

                        previous_requests = stream.requests
                        for number, text in enumerate((
                                json.dumps(case(50000)),
                                SINGLE + "\n" + json.dumps(case(50000)) + "\nExplain the assumptions.",
                                BATCH + "\n" + json.dumps([case(50000), {"mode": "civil_geotech", "spec": {"calculation": "unregistered"}}]),
                                SINGLE + "\n" + json.dumps(case(50000))), 1):
                            if number == 4:
                                (folder / "survey.txt").write_text("independent project input\n")
                            answer, counts = await exchange(port, folder, text)
                            self.assertEqual(answer, FALLBACK)
                            self.assertEqual(stream.requests, previous_requests + number)
                            self.assertGreater(counts["usage"], 0)

                        answer, counts = await exchange(port, folder, SINGLE + "\n" + json.dumps(ctms))
                        self.assertEqual(answer, FALLBACK)
                        self.assertEqual(stream.requests, previous_requests + 5)
                        self.assertGreater(counts["usage"], 0)

