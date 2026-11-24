"""Replay the published ACADS slope geometry through the registered Pi tool."""
import asyncio
import json

from verify_janbu_registered import registered_answer
from test_critical_slip_grid import acads


async def main():
    for method in ("bishop", "janbu_uncorrected"):
        spec = dict(acads(), method=method)
        answer, requests = await registered_answer({"mode": "civil_geotech", "spec": spec},
                                                   "acads-grid-" + method)
        winner = answer["minimum"]
        assert answer["conservation_verified"] and answer["counts"]["qualified"] > 1
        assert sum(answer["counts"].values()) == answer["trials"]
        assert not answer["continuous_global_minimum_verified"]
        if method == "bishop":
            assert abs(winner["factor_of_safety"] - 0.987) < 0.004
        print(json.dumps({"method": method, "requests": requests, "trials": answer["trials"],
                          "counts": answer["counts"], "circle": winner["circle"],
                          "factor_of_safety": winner["factor_of_safety"]}))


if __name__ == "__main__":
    asyncio.run(main())
