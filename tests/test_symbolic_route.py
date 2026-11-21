from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import engineering_intake
import symbolic_route


class Session:
    def __init__(self, root):
        self.workdir = str(root)
        self.tools = {}
        self.commands = []

    def time_left(self):
        return 90

    async def write_file(self, path, data):
        self.tools[path] = data
        return len(data)

    async def exec(self, command, timeout_sec):
        self.commands.append(command)
        result = subprocess.run(command, shell=True, cwd=self.workdir, capture_output=True,
                                text=True, timeout=timeout_sec)
        return {"exit_code": result.returncode, "output": result.stdout}


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_workbook_evidence_and_lossless_guide(self):
        with tempfile.TemporaryDirectory() as root:
            session = Session(root)
            guide = await engineering_intake.provision_engineering_cli(session)
            path = Path(root) / "input 'A'.xlsx"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("xl/worksheets/sheet1.xml",
                           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                           '<sheetData><row><c r="A1"><f>1+1</f></c>'
                           '<c r="B1" t="e"><v>#REF!</v></c></row></sheetData></worksheet>')
            record = {}
            await engineering_intake.intake_for_prompt(session, discovered=record)
            cli = Path(__file__).resolve().parents[1] / "engineering_toolkit.py"
            with patch.object(symbolic_route, "TOOL", str(cli)):
                note = await symbolic_route.prepare(session, record, guide)
            self.assertIn('"formulas_without_cached_values": 1', note)
            self.assertIn('"error_count": 1', note)
            self.assertIn("xlsx-audit", note)
            self.assertEqual(session.tools[symbolic_route.GUIDE],guide.encode())
            self.assertLess(len(note.encode()), len(guide.encode()) // 2)
            self.assertEqual(len(session.commands),2)

    async def test_ambiguous_workbooks_use_model_without_extra_audit(self):
        session = Session(".")
        record = {"by_extension": {".xlsx": 2}, "files": []}
        note = await symbolic_route.prepare(session, record, "complete guide")
        self.assertIn("xlsx-audit", note)
        self.assertFalse(session.commands)

    async def test_delivery_failure_returns_full_guide(self):
        session = Session(".")
        async def short_write(*args):
            return 0
        session.write_file = short_write
        self.assertEqual(await symbolic_route.prepare(session, {}, "guide"), "guide")
