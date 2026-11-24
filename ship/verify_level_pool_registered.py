"""Receive the actual registered level-pool tool result through a two-request Pi loopback."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


async def main(root, stock):
    # Execute the selected participant and its locked Pi, while keeping the
    # same published hydrograph and observer across both checkouts.
    source = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root/"tests"), str(root/"ship"), str(source/"tests")]
    import replay_engineering_provider_wire as wire
    from test_level_pool_routing import county_case, county_oracle

    params = {"mode": "civil_geotech", "spec": county_case()}
    cases = json.dumps(params, separators=(",", ":"))
    original_events, original_digest = wire.tool_events, wire.digest
    returned = []

    def issue(model):
        for event, data in original_events(model):
            if event == "content_block_delta":
                data["delta"]["partial_json"] = cases
            yield event, data

    def read_return(data):
        if data.startswith(b"{"):
            result = json.loads(data)
            if result.get("tool") == "hamon_conservation_cad_verifier":
                returned.append(result)
        return original_digest(data)

    with patch.object(wire, "tool_events", issue), patch.object(wire, "digest", read_return):
        receipt = await wire.replay(root, "county-reservoir", "Route the detention basin inflow using the supplied rating.",
                                    True, None, "claude-opus-5-5")
    assert len(receipt["requests"]) == 2 and len(returned) == 1, receipt
    result = returned[0]
    if stock:
        assert result["conservation_verified"] is False and "hydrograph" not in result, result
        summary = {"baseline_reason": result["reason"]}
    else:
        summary = county_oracle(result)
    return {"input_sha256": hashlib.sha256(cases.encode()).hexdigest(),
            "received_result_sha256": receipt["requests"][1]["returned_reference_sha256"],
            "received_requests": len(receipt["requests"]), **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stock", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = asyncio.run(main(args.root.resolve(), args.stock))
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
