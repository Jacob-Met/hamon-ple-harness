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
    for overloaded, tier in ((False, "low"), (True, "medium")):
        if overloaded:
            spec["applied_gross_pressure_Pa"] = 1e12
        instruction = json.dumps({"mode": "civil_geotech", "spec": spec})
        session = SimpleNamespace(
            instruction=instruction,
            workdir=str(tmp_path),
            mcp_servers=[],
            final_message="",
            usage=mock.AsyncMock(),
            log=mock.AsyncMock(),
        )

        async def fake_process(argv, session, **kwargs):
            assert argv[argv.index("--thinking") + 1] == tier
            prompt = kwargs["stdin"].decode()
            if overloaded:
                assert prompt == instruction
            else:
                assert prompt.startswith(instruction + "\n\n")
                answer = json.loads(prompt.splitlines()[-1])
                assert (
                    answer["conservation_verified"]
                    and answer["bearing_method"] == "terzaghi"
                )
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
