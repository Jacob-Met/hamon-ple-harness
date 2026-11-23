"""Measure the actual initial Pi messages for committed public examples using loopback SSE.

Run with the locked npm dependencies and optional tiktoken==0.12.0 for the
named cl100k_base comparison. Each response is a fixture. This measures
request representation and requested thinking limits, not model billing,
answers to the original tasks, or the effects of future reference requests.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


TOOL = "hamon_conservation_cad_verifier"
EMPTY = dict(schema="hamon.ple.intake.v1", scanned_files=0, engineering_file_count=0,
             files=[], by_extension={}, warnings=[])
CELL = dict(mode="electrochem_separations", spec=dict(
    subkind="electrochemistry", E0_V=0, n_electrons=2, T_K=298.15,
    activity_terms=[dict(species="Zn anode", stoich=1, activity=.1),
                    dict(species="Zn cathode", stoich=-1, activity=.5)]))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tool_events(model):
    yield "message_start", {"message": {"id": "ref", "role": "assistant", "type": "message",
        "model": model, "content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}}
    yield "content_block_start", {"index": 0, "content_block": {"type": "tool_use",
        "id": "ref-1", "name": TOOL, "input": {}}}
    yield "content_block_delta", {"index": 0, "delta": {"type": "input_json_delta",
        "partial_json": '{"mode":"reference"}'}}
    yield "content_block_stop", {"index": 0}
    yield "message_delta", {"delta": {"stop_reason": "tool_use", "stop_sequence": None},
                            "usage": {"output_tokens": 1}}
    yield "message_stop", {}


async def replay(root, label, instruction, ask_reference, tokenizer):
    import harness
    from fin_provider_replay import events
    observations, errors = [], []

    async def respond(reader, writer):
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("ascii").split("\r\n")
            method, target, _ = head[0].split()
            assert method == "POST" and target.split("?",1)[0] == "/v1/messages"
            headers = dict(line.lower().split(": ",1) for line in head[1:] if ": " in line)
            raw = await reader.readexactly(int(headers["content-length"]))
            req = json.loads(raw)
            assert req["stream"] is True
            msg = req["messages"][0]["content"]
            prompt = msg if isinstance(msg,str) else "\n".join(p["text"] for p in msg if p["type"]=="text")
            assert instruction in prompt
            tool = next(t for t in req["tools"] if t["name"] == TOOL)
            guide = tool["description"]
            record = dict(request_bytes=len(raw), guide_bytes=len(guide.encode()),
                          guide_sha256=digest(guide.encode()), budget=req["thinking"]["budget_tokens"],
                          max_tokens=req["max_tokens"], tool_names=[t["name"] for t in req["tools"]],
                          presolved="Deterministic engineering calculation for the inline request" in prompt)
            if tokenizer:
                record.update(request_bpe=len(tokenizer.encode_ordinary(raw.decode("utf-8"))),
                              guide_bpe=len(tokenizer.encode_ordinary(guide)))
            if observations and ask_reference:
                blocks = [b for m in req["messages"] if isinstance(m["content"],list)
                          for b in m["content"] if b["type"]=="tool_result"]
                assert len(blocks)==1 and not blocks[0].get("is_error")
                content=blocks[0]["content"]
                full=content if isinstance(content,str) else "\n".join(b["text"] for b in content if b["type"]=="text")
                record.update(returned_reference_bytes=len(full.encode()), returned_reference_sha256=digest(full.encode()))
            observations.append(record)
            out = tool_events(req["model"]) if ask_reference and len(observations)==1 else events(req["model"],"Replay received.")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            for name, body in out:
                data=json.dumps(dict(type=name,**body))
                writer.write(f"event: {name}\ndata: {data}\n\n".encode())
            await writer.drain()
        except Exception as exc:
            errors.append(repr(exc))
            writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
        finally:
            writer.close()
            await writer.wait_closed()

    with tempfile.TemporaryDirectory() as home:
        async with await asyncio.start_server(respond, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            env=dict(HOME=home,PATH=os.environ["PATH"],ANTHROPIC_BASE_URL=f"http://127.0.0.1:{port}",
                     ANTHROPIC_API_KEY="loopback-fixture",MODEL_NAME="claude-sonnet-4-5")
            session=SimpleNamespace(instruction=instruction,workdir="/workspace/task",mcp_servers=[],
                final_message="",time_left=lambda:None,usage=AsyncMock(),log=AsyncMock())
            # The measured condition has no live project files. Delivery tools
            # retain their normal settings but cannot attest any produced file.
            with (patch.dict(os.environ,env,clear=True),
                  patch.object(harness.engineering_intake,"intake_for_prompt",AsyncMock(return_value=json.dumps(EMPTY))),
                  patch.object(harness.engineering_intake,"provision_engineering_cli",AsyncMock(return_value="")),
                  patch.object(harness.delivery_guard,"snapshot",AsyncMock(return_value=None))):
                try:
                    await asyncio.wait_for(harness.solve(session),timeout=85)
                except TimeoutError as exc:
                    raise TimeoutError(f"{label}: Pi timed out after {len(observations)} requests; provider errors={errors}") from exc
    assert not errors,errors
    assert len(observations)==(2 if ask_reference else 1),(label,observations)
    return dict(label=label,instruction_sha256=digest(instruction.encode()),requests=observations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bpe",action="store_true")
    parser.add_argument("--reference",action="store_true")
    parser.add_argument("--only",help="Run one named archive or synthetic control")
    args=parser.parse_args()
    root=args.root.resolve()
    sys.path[:0]=[str(root),str(root/"tests")]
    tokenizer=None
    if args.bpe:
        import tiktoken
        tokenizer=tiktoken.get_encoding("cl100k_base")
    tasks=json.loads((root/"artifacts/manifest.json").read_text())["tasks"]
    assert len(tasks)==29 and len(set(tasks))==len(tasks)
    corpus={}
    for task in tasks:
        history=json.loads((root/"artifacts"/task/"trajectory.json").read_text())
        corpus[task]=next(s["message"] for s in history["steps"] if s["source"]=="user")
    corpus["synthetic-zinc-cell"]=json.dumps(CELL)
    corpus["synthetic-zinc-incomplete"]=json.dumps(CELL)+"\nAlso deliver a safety memo."
    if args.reference:
        corpus["synthetic-reference"]="Return the engineering tool reference."
    if args.only and args.only not in corpus:
        parser.error("unknown --only label")
    for label,instruction in corpus.items():
        if args.only and label!=args.only:
            continue
        result=asyncio.run(replay(root,label,instruction,label=="synthetic-reference",tokenizer))
        print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
