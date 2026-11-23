"""Terrain volumes through the task-provisioned ordinary engineering CLI."""

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engineering_intake


EXISTING = [
    [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
    [(0, 0, 0), (1, 1, 0), (0, 1, 0)],
]
PROPOSED = [
    [(0, 0, -0.5), (1, 0, 0.5), (0, 1, -0.5)],
    [(1, 0, 0.5), (1, 1, 0.5), (0, 1, -0.5)],
]


def drawing(proposed=PROPOSED):
    pairs = [(0, "SECTION"), (2, "HEADER"), (9, "$INSUNITS"),
             (70, 6), (0, "ENDSEC"), (0, "SECTION"), (2, "ENTITIES")]
    for layer, triangles in [("EG", EXISTING), ("FG", proposed)]:
        for triangle in triangles:
            pairs += [(0, "3DFACE"), (8, layer)]
            for i, point in enumerate([*triangle, triangle[-1]]):
                pairs += [(10 + i, point[0]), (20 + i, point[1]), (30 + i, point[2])]
    pairs += [(0, "ENDSEC"), (0, "EOF")]
    return "".join(f"{code}\n{value}\n" for code, value in pairs).encode()


class ProvisionedTerrainEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ple-terrain-entry-")
        self.addCleanup(self.tmp.cleanup)
        self.task = Path(self.tmp.name)
        task = self.task

        class Receiver:
            async def write_file(self, remote_path, data):
                return (task / Path(remote_path).name).write_bytes(data)

        note = asyncio.run(engineering_intake.provision_engineering_cli(Receiver()))
        self.assertIn("--existing-layer", note)
        self.assertTrue((self.task / "hamon_terrain_volume.py").is_file())
        self.source = self.task / "site with spaces.dxf"
        self.source.write_bytes(drawing())

    def run_cli(self, *args):
        result = subprocess.run(
            [sys.executable, "-E", "-s", "-S",
             str(self.task / "hamon-engineering-toolkit.py"),
             "dxf-takeoff", str(self.source), *args],
            cwd=self.task, capture_output=True, text=True, timeout=20,
        )
        self.assertFalse(result.stderr, result.stderr)
        return result.returncode, json.loads(result.stdout)

    def test_provisioned_crossing_surfaces_keep_both_signs_and_source_bytes(self):
        original = hashlib.sha256(self.source.read_bytes()).hexdigest()
        code, reply = self.run_cli("--existing-layer", "EG", "--proposed-layer", "FG")
        self.assertEqual(code, 0, reply)
        self.assertIs(reply["qualified"], True)
        self.assertAlmostEqual(reply["cut_m3"], 0.125)
        self.assertAlmostEqual(reply["fill_m3"], 0.125)
        self.assertEqual(reply["net_fill_m3"], 0)
        self.assertEqual(reply["common_area_m2"], 1)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), original)

    def test_partial_surface_keeps_diagnostics_and_withholds_site_total(self):
        self.source.write_bytes(drawing(PROPOSED[:1]))
        code, reply = self.run_cli("--existing-layer", "EG", "--proposed-layer", "FG")
        self.assertEqual(code, 3, reply)
        self.assertIs(reply["qualified"], False)
        self.assertEqual(reply["common_area_m2"], 0.5)
        self.assertEqual(reply["uncovered_existing_area_m2"], 0.5)
        for key in ["cut_m3", "fill_m3", "net_fill_m3"]:
            self.assertIsNone(reply[key])

    def test_one_layer_flag_cannot_silently_fall_back_to_surface_takeoff(self):
        code, reply = self.run_cli("--existing-layer", "EG")
        self.assertEqual(code, 2, reply)
        self.assertFalse(reply["ok"])
        self.assertIn("both", str(reply))

    def test_missing_staged_dependency_cannot_use_the_development_checkout(self):
        (self.task / "hamon_terrain_volume.py").unlink()
        code, reply = self.run_cli("--existing-layer", "EG", "--proposed-layer", "FG")
        self.assertEqual(code, 2, reply)
        self.assertFalse(reply["ok"])


if __name__ == "__main__":
    unittest.main()
