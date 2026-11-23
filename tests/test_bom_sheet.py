"""Read physically rendered PDF schedules through the entrant's ordinary drawing command."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from hamon_drawing_reader import build_minimal_vector_pdf

CLI = Path(__file__).resolve().parents[1] / "engineering_toolkit.py"


def page(sheet, title, cells):
    block = [(600, 350, "SHEET NAME"), (600, 332, title),
             (600, 295, "SHEET SIZE"), (600, 266, "SHEET NUMBER"), (615, 243, sheet)]
    stream = "\n".join(f"BT /F1 9 Tf {x} {y} Td ({label}) Tj ET" for x, y, label in block + cells)
    return {"content_stream": stream}


TABLE = [(150, 530, "BILL OF MATERIALS"), (60, 510, "EQUIPMENT"),
         (180, 510, "QTY"), (290, 510, "DESCRIPTION"), (60, 488, "INVERTER"),
         (183, 488, "6"), (238, 488, "ACME INV-200 inverter")]


def query(path, model="INV-200", title="LINE DIAGRAM"):
    return json.loads(subprocess.check_output([
        sys.executable, str(CLI), "pdf-bom-sheet", str(path),
        "--equipment", "INVERTER", "--part-id", model, "--title", title]))


@unittest.skipUnless(shutil.which("pdftotext") and shutil.which("pdfinfo"), "requires Poppler")
class BomSheets(unittest.TestCase):
    def test_exact_row_and_sheet_and_ambiguities(self):
        source = page("PV-2", "STRING LAYOUT", TABLE)
        destination = page("PV-4", "LINE DIAGRAM", [(65, 445, "INV-200 CONNECTION")])
        cases = [
            ([source, destination], "resolved"),
            ([source, destination, destination], "ambiguous_destination"),
            ([source, destination, page("PV-4A", "LINE DIAGRAM", [])], "ambiguous_destination"),
            ([source, destination, page("PV-4", "ELEVATION", [])], "ambiguous_destination"),
            ([source, source, destination], "ambiguous_bom_row"),
            ([source, page("PV-4", "LINE DIAGRAM", [(65, 445, "INV-200A")])], "part_not_on_destination"),
            ([source, page("PV-4", "UNRELATED", [(65, 445, "INV-200")])], "destination_title_not_found"),
            ([page("PV-2", "STRING LAYOUT", TABLE[:-1]), destination], "bom_row_not_found"),
            ([page("PV-2", "STRING LAYOUT", TABLE[:-1] + [(238, 459, "INV-200")]), destination],
             "bom_row_not_found"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"plan.pdf"
            for pages, expected in cases:
                with self.subTest(expected=expected):
                    path.write_bytes(build_minimal_vector_pdf(pages))
                    result = query(path)
                    self.assertEqual(result["status"], expected)
                    if expected == "resolved":
                        self.assertEqual(result["source_rows"][0]["quantity"]["text"], "6")
                        self.assertEqual(result["source_sheet"]["sheet"], "PV-2")
                        self.assertEqual((result["destination"]["page"], result["destination"]["sheet"]),
                                         (2, "PV-4"))
                    else:
                        self.assertIsNone(result["destination"])

    @unittest.skipUnless(os.environ.get("HAMON_SOLAR_PLAN_PDF"), "supply the public New Bedford plan set")
    def test_original_public_plan(self):
        # City of New Bedford, Isaksen Plan Set, 17 July 2026.
        path = Path(os.environ["HAMON_SOLAR_PLAN_PDF"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "e931d4a4b6c42a7be8516a6f98367967bd99da345e426446777a91217bbfa2c6")
        result = query(path, "SE50KUS", "ELECTRICAL LINE DIAGRAM")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual((result["source_sheet"]["sheet"], result["source_rows"][0]["quantity"]["text"]),
                         ("PV-2A", "6"))
        dest = result["destination"]
        self.assertEqual((dest["page"], dest["sheet"], len(dest["part_witnesses"])), (7, "PV-4", 7))
        ambiguous = query(path, "SE50KUS", "EQUIPMENT SPECIFICATIONS")
        self.assertEqual(ambiguous["status"], "ambiguous_destination")
        self.assertIsNone(ambiguous["destination"])
