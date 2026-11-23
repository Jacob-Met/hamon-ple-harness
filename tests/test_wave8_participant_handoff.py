import asyncio
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import harness
import taskbridge


def test_solver_receipt_reaches_low_effort_participant_with_original_request(tmp_path):
    spec = {
        "calculation": "bearing_capacity",
        "bearing_method": "terzaghi",
        "footing_B_m": 2,
        "depth_Df_m": 1,
        "gamma_N_m3": 18000,
        "phi_deg": 30,
        "FS": 3,
    }
    inventory = json.dumps(
        {
            "schema": "hamon.ple.intake.v1",
            "scanned_files": 0,
            "engineering_file_count": 0,
            "by_extension": {},
            "files": [],
            "warnings": [],
        }
    )
    relay = mock.MagicMock()
    relay.__aenter__.return_value = SimpleNamespace(env=lambda: {})
    single = json.dumps({"mode": "civil_geotech", "spec": spec})
    alt = {"mode": "civil_geotech", "spec": {**spec, "footing_B_m": 3}}
    pair = [json.loads(single), alt]
    fence = "```json\n" + json.dumps(pair) + "\n```"
    history = {"mode": "civil_geotech", "spec": {
        "calculation": "consolidation_history_1d", "H_clay_m": 1, "cv_m2_s": 1,
        "mv_per_Pa": 1e-7, "drainage": "single", "time_s": 1,
        "load_history": [{"time_s": 0, "surcharge_Pa": 0}, {"time_s": 1, "surcharge_Pa": 10000}],
    }}
    staged = json.dumps(history)
    for instruction, tier, count in (
        (single, "low", 1),
        (json.dumps({"mode": "civil_geotech", "spec": {**spec, "applied_gross_pressure_Pa": 1e12}}),
         "medium", 0),
        ("Run the calculations below and return their JSON results in the same order.\n" + fence,
         "low", 2),
        (fence + "\nCompare against the project specification.", "medium", 0),
        (staged, "low", 1),
        (json.dumps({**history, "spec": {**history["spec"], "proposed_settlement_m": 1}}),
         "medium", 0),
    ):
        session = SimpleNamespace(
            instruction=instruction,
            workdir=str(tmp_path),
            mcp_servers=[],
            final_message="",
            time_left=lambda: None,
            usage=mock.AsyncMock(),
            log=mock.AsyncMock(),
        )

        async def fake_process(argv, session, **kwargs):
            assert argv[argv.index("--thinking") + 1] == tier
            prompt = kwargs["stdin"].decode()
            if not count:
                assert prompt == instruction
            else:
                assert prompt.startswith(instruction + "\n\n")
                answer = json.loads(prompt.splitlines()[-1])
                results = answer if count == 2 else [answer]
                assert len(results) == count
                assert all(result["conservation_verified"] for result in results)
                if instruction == staged:
                    assert results[0]["calculation"] == "consolidation_history_1d"
                    assert 0 < results[0]["mean_excess_pore_pressure_Pa"] < 10000
                else:
                    assert all(result["bearing_method"] == "terzaghi" for result in results)
                if count == 2:
                    assert results[0] != results[1]
            await kwargs["on_line"](
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "fixture answer"}],
                        },
                    }
                ).encode()
            )
            return SimpleNamespace(code=0, stderr="")

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(taskbridge, "Relay", return_value=relay)
            )
            stack.enter_context(
                mock.patch.object(
                    harness, "configure", return_value=("fixture", "fixture-model")
                )
            )
            stack.enter_context(
                mock.patch.object(
                    harness.engineering_intake,
                    "intake_for_prompt",
                    new=mock.AsyncMock(return_value=inventory),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    harness.engineering_intake,
                    "provision_engineering_cli",
                    new=mock.AsyncMock(return_value=""),
                )
            )
            stack.enter_context(
                mock.patch.object(taskbridge, "run_process", new=fake_process)
            )
            assert asyncio.run(harness.solve(session)) == "fixture answer"
