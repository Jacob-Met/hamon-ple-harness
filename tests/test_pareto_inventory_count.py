import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import engineering_intake
import hamon_pareto


class InventorySession:
    def __init__(self, workdir):
        self.workdir = str(workdir)

    def time_left(self):
        return 120

    async def exec(self, command, timeout_sec):
        result = subprocess.run(command, shell=True, text=True, capture_output=True,
                                timeout=timeout_sec, check=False)
        return {"exit_code": result.returncode, "output": result.stdout}


class CountedInventoryTests(unittest.TestCase):
    def test_budget_uses_count_before_prompt_shortening(self):
        for count in (25, 26):
            for long_names in (False, True):
                with self.subTest(count=count, long_names=long_names), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    names = ["界" * 56] * 3 if long_names else ["a", "b", "c"]
                    directory = root.joinpath(*names)
                    directory.mkdir(parents=True)
                    for index in range(count):
                        (directory / f"item-{index}.txt").touch()
                    prompt = asyncio.run(engineering_intake.intake_for_prompt(InventorySession(root)))
                    rec = json.loads(next(line for line in prompt.splitlines() if line.startswith("{")))
                    self.assertEqual(rec["engineering_file_count"], count)
                    self.assertEqual(rec["by_extension"], {".txt": count})
                    self.assertEqual(len(rec["files"]), 0 if long_names else count)
                    plan = hamon_pareto.route_from_intake_prompt(prompt)
                    expected = ("medium", 0, 600) if count == 25 else ("high", 2, 960)
                    self.assertEqual((plan["thinking_tier"], plan["complexity_points"],
                                      plan["delivery_reserve_sec"]), expected)

    def test_previous_inventory_with_extension_totals(self):
        record = {"schema": "hamon.ple.intake.v1", "files": [], "by_extension": {".txt": 26}}
        self.assertEqual(hamon_pareto.route_from_intake_prompt(json.dumps(record))["thinking_tier"], "high")
