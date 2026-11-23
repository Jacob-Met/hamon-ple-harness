"""Replay a structured fin input through harness.solve, the locked Pi CLI and loopback SSE."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

SPEC = dict(mode="convective_fin_1d", k_w_mk=200., area_m2=.00015, perimeter_m=.05,
            h_w_m2k=40., length_m=.12, t_base_c=125., t_inf_c=25., tip_bc="adiabatic")
INSTRUCTION = "```wave5_spec\n" + json.dumps(SPEC) + "\n```"
EMPTY = dict(schema="hamon.ple.intake.v1", scanned_files=0, engineering_file_count=0,
             by_extension={}, files=[], warnings=[])


def independent_flux(s):
    axial = s["k_w_mk"] * s["area_m2"]
    lateral = s["h_w_m2k"] * s["perimeter_m"]
    return (math.sqrt(axial * lateral) * (s["t_base_c"] - s["t_inf_c"])
            * math.tanh(s["length_m"] * math.sqrt(lateral / axial)))


def events(model, answer):
    yield "message_start", {"message": {"id": "local", "role": "assistant", "type": "message",
            "model": model, "content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}}
    yield "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
    yield "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": answer}}
    yield "content_block_stop", {"index": 0}
    yield "message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": 1}}
    yield "message_stop", {}


async def replay(root, expected):
    expected_flux = independent_flux(SPEC)
    answer = format(expected_flux, ".15g") + " W"
    refusal = expected == "refused"
    instruction = INSTRUCTION.replace('"adiabatic"', '"unspecified"') if refusal else INSTRUCTION
    if refusal:
        answer = "Unspecified tip boundary"
    def inspect(prompt, result, sha):
        assert sha in prompt and result["passed"] is True
        assert math.isclose(result["q_fin_w"], expected_flux, rel_tol=1e-13)
        assert math.isclose(result["q_fvm_w"], expected_flux, rel_tol=1e-3)
        return {k: result[k] for k in ("q_fin_w", "q_fvm_w")}

    return await exchange(root, instruction, answer, expected == "verified", inspect,
                          "unsupported or missing fin tip boundary" if refusal else None)


async def exchange(root, instruction, answer, certified, inspect, refusal_log=None):
    sys.path.insert(0, str(root))
    import harness
    observation = {}
    response_error = []
    async def respond(reader, writer):
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("ascii").split("\r\n")
            method, target, protocol = head[0].split()
            assert method == "POST" and target.split("?", 1)[0] == "/v1/messages"
            headers = dict(x.split(": ", 1) for x in head[1:] if ": " in x)
            headers = {k.lower(): v for k, v in headers.items()}
            size = int(headers["content-length"])
            request = json.loads(await reader.readexactly(size))
            assert request["stream"] is True
            content = request["messages"][-1]["content"]
            prompt = content if isinstance(content, str) else "\n".join(x["text"] for x in content if x["type"] == "text")
            assert instruction in prompt
            sha = hashlib.sha256(instruction.encode()).hexdigest()
            marker = "Deterministic engineering calculation for the inline request (JSON tool output):\n"
            observed = dict(request_bytes=size, thinking=request.get("thinking"),
                            output_config=request.get("output_config"), max_tokens=request.get("max_tokens"),
                            instruction_sha256=sha, certificate=marker in prompt)
            if marker in prompt:
                observed.update(inspect(prompt, json.loads(prompt.split(marker, 1)[1].splitlines()[0]), sha))
            assert not observation
            observation.update(observed)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            for name, fields in events(request["model"], answer):
                payload = json.dumps({"type": name, **fields})
                writer.write(f"event: {name}\ndata: {payload}\n\n".encode())
            await writer.drain()
        except Exception as exc:
            response_error.append(repr(exc))
            writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        finally:
            writer.close()
            await writer.wait_closed()

    with tempfile.TemporaryDirectory() as home:
        async with await asyncio.start_server(respond, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            env = dict(HOME=home, PATH=os.environ["PATH"], ANTHROPIC_BASE_URL=f"http://127.0.0.1:{port}",
                       ANTHROPIC_API_KEY="loopback-fixture", MODEL_NAME="claude-sonnet-4-5")
            session = SimpleNamespace(instruction=instruction, workdir=home, mcp_servers=[], final_message="",
                                      time_left=lambda: None, usage=AsyncMock(), log=AsyncMock())
            with (patch.dict(os.environ, env, clear=True),
                  patch.object(harness.engineering_intake, "intake_for_prompt", AsyncMock(return_value=json.dumps(EMPTY))),
                  patch.object(harness.engineering_intake, "provision_engineering_cli", AsyncMock(return_value=""))):
                start = perf_counter()
                received = await asyncio.wait_for(harness.solve(session), timeout=45)
                elapsed = perf_counter() - start
            assert not response_error, response_error
            assert observation["thinking"]["type"] == "enabled"
            assert observation["thinking"]["budget_tokens"] == (2048 if certified else 8192)
            if refusal_log:
                assert any(refusal_log in call.args[0]
                           for call in session.log.await_args_list)
            assert received == answer, (received, answer)
            assert observation["certificate"] == certified, observation
            assert session.usage.await_args.kwargs["output_tokens"] == 1
    return {**observation, "final_answer": received, "elapsed_seconds": elapsed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expect", choices=("original", "verified", "refused"), default="verified")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(replay(args.root.resolve(), args.expect)), sort_keys=True))
