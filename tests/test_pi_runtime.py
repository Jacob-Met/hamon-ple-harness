"""The submitted lockfile selects the Pi CLI for both catalog and solve."""
import importlib.util
import pathlib
import subprocess
import unittest
from unittest.mock import patch


class PiRuntimeTests(unittest.TestCase):
    def test_catalog_uses_entry_local_cli_even_when_runner_has_pi(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("entry_harness", root / "harness.py")
        entry = importlib.util.module_from_spec(spec)
        with patch("shutil.which", return_value="/opt/runner/bin/pi"):
            spec.loader.exec_module(entry)

        result = subprocess.CompletedProcess([], 0, stdout="provider model\n")
        with patch.object(entry.subprocess, "run", return_value=result) as run:
            self.assertTrue(entry.in_catalog("provider", "model", {}))
        self.assertEqual(
            run.call_args.args[0],
            [str(root / "node_modules" / ".bin" / "pi"), "--offline", "--list-models", "model"],
        )
