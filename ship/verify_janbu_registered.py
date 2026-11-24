"""Replay the Seequent slice table through harness.solve, Pi and the registered tool.

Run from the entrant checkout after installing its locked dependencies:
    python ship/verify_janbu_registered.py
Only the model response stream is a fixture. No sponsored endpoint is called.
"""
import asyncio
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import replay_engineering_provider_wire as wire
from test_janbu_slices import seequent


async def registered_answer(payload, label):
    reference_events = wire.tool_events
    original_digest = wire.digest
    answers = []

    def issue_slope_call(model):
        for kind, body in reference_events(model):
            if kind == "content_block_delta":
                body["delta"]["partial_json"] = json.dumps(payload)
            yield kind, body

    def observe_result(data):
        # The driver hashes the exact registered tool text returned in request two.
        # Observe those bytes before the digest without changing its fingerprint.
        if data.startswith(b"{"):
            candidate = json.loads(data)
            if candidate.get("calculation") == payload["spec"]["calculation"]:
                answers.append(candidate)
        return original_digest(data)

    with (patch.object(wire, "tool_events", issue_slope_call),
          patch.object(wire, "digest", observe_result)):
        trial = await wire.replay(ROOT, label, "Check the specified slope.",
                                  True, None, "claude-opus-5-5")
    assert len(trial["requests"]) == 2 and len(answers) == 1, trial
    return answers[0], len(trial["requests"])


async def main():
    payload = {"mode": "civil_geotech", "spec": {**seequent(), "required_factor_of_safety": 1.25}}
    answer, requests = await registered_answer(payload, "seequent-janbu")
    assert answer["method"] == "janbu_simplified_uncorrected"
    assert answer["verified"] and answer["base_contact_verified"] and answer["conservation_verified"]
    assert not answer["meets_required_factor"] and len(answer["slices"]) == 29
    assert math.isclose(answer["factor_of_safety"], 1.1247783031418455, rel_tol=0, abs_tol=1e-12)
    print(json.dumps({"requests": requests, "slices": 29, "factor_of_safety": answer["factor_of_safety"],
                      "meets_required_factor": answer["meets_required_factor"]}))


if __name__ == "__main__":
    asyncio.run(main())
