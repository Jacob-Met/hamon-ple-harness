"""Structural Word document checks for the isolated engineering CLI."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import archive_checkpoint
import delivery_guard

from engineering_toolkit import audit_local_deliverable


TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
TRANSITIONAL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
STRICT = "http://purl.oclc.org/ooxml/officeDocument/relationships"
WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
STRICT_WORD = "http://purl.oclc.org/ooxml/wordprocessingml/main"
MAIN = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
SHEET_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml."


def package(path, target="document.xml", part="document.xml", word=WORD,
            relationship=TRANSITIONAL, declaration=None, mode="Internal",
            main_xml=None, rel_xml=None):
    if declaration is None:
        declaration = f'<Override PartName="/{part}" ContentType="{MAIN}"/>'
    if rel_xml is None:
        rel_xml = (f'<Relationships xmlns="{RELS}"><Relationship Id="start" '
                   f'Type="{relationship}/officeDocument" Target="{target}" '
                   f'TargetMode="{mode}"/></Relationships>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", f'<Types xmlns="{TYPES}">'
                   '<Default Extension="rels" ContentType="application/'
                   'vnd.openxmlformats-package.relationships+xml"/>'
                   f'{declaration}</Types>')
        z.writestr("_rels/.rels", rel_xml)
        z.writestr(part, main_xml if main_xml is not None else
                   f'<w:document xmlns:w="{word}"><w:body/></w:document>')


class DocxFormatGuard(unittest.TestCase):
    def test_relocated_strict_and_default_content_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "sample.docx"
            for options in ({}, {"target": "/Custom/Part.XML", "part": "custom/part.xml",
                                 "word": STRICT_WORD, "relationship": STRICT,
                                 "declaration": f'<Default Extension="XML" ContentType="{MAIN}"/>'}):
                with self.subTest(options=options):
                    package(doc, **options)
                    self.assertTrue(audit_local_deliverable(doc)["ok"])

    def test_invalid_package_parts_and_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "sample.docx"
            for options in ({"target": "other.xml"}, {"mode": "External"},
                            {"declaration": '<Default Extension="xml" ContentType="application/xml"/>'},
                            {"declaration": '<Override PartName="/document.xml" ContentType="application/xml"/>'},
                            {"rel_xml": "<Relationships/>"},
                            {"main_xml": '<document xmlns="urn:wrong"/>'},
                            {"main_xml": f'<document xmlns="{WORD}"/><extra/>'}):
                with self.subTest(options=options):
                    package(doc, **options)
                    result = audit_local_deliverable(doc)
                    self.assertFalse(result["ok"])
                    self.assertTrue(result["reason"].startswith("invalid_docx:"), result)

    def test_general_zip_and_mislabeled_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ext, valid in (("zip", True), ("docx", False)):
                path = Path(tmp) / ("archive." + ext)
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("readme.txt", "example")
                self.assertEqual(audit_local_deliverable(path)["ok"], valid)


class DeliveryPackageChecks(unittest.TestCase):
    @staticmethod
    def workbook(path, main_type, *, kind="worksheet", body=None, omit_sheet=False):
        part = "custom/books/register.xml"
        sheet = "custom/pages/pump.data"
        types = (f'<Override PartName="/{part}" ContentType="{main_type}"/>'
                 f'<Default Extension="data" ContentType="{SHEET_TYPE}{kind}+xml"/>'
                 '<Override PartName="/custom/preview.xml" ContentType="application/octet-stream"/>')
        main = (f'<workbook xmlns="{SHEET_NS}" xmlns:r="{TRANSITIONAL}"><sheets>'
                '<sheet name="Pump" sheetId="1" r:id="pump"/></sheets></workbook>')
        package(path, target="/" + part, part=part, declaration=types, main_xml=main)
        with zipfile.ZipFile(path, "a") as book:
            book.writestr("custom/books/_rels/register.xml.rels", f'<Relationships xmlns="{RELS}">'
                          f'<Relationship Id="pump" Type="{TRANSITIONAL}/{kind}" '
                          'Target="../pages/pump.data"/></Relationships>')
            if not omit_sheet:
                book.writestr(sheet, body if body is not None else f'<{kind} xmlns="{SHEET_NS}"/>')
            book.writestr("custom/preview.xml", b"binary preview")

    def test_relocated_main_parts_standalone_and_inside_deliverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            word = root / "report.docx"
            package(word, target="word/output.xml", part="word/output.xml")
            names = [word.name]
            for suffix, content_type in (
                ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"),
                ("xlsm", "application/vnd.ms-excel.sheet.macroEnabled.main+xml"),
            ):
                book = root / ("register." + suffix)
                self.workbook(book, content_type)
                names.append(book.name)
            submitted = root / "submission.zip"
            with zipfile.ZipFile(submitted, "w") as outer:
                for name in names:
                    outer.write(root / name, name)
            archive_checkpoint.verify(submitted, names)
            targets = [{"path": "/filesystem/" + name, "kind": "file"} for name in names]
            targets.append({"path": "/filesystem/submission.zip", "kind": "file", "root_members": names})
            self.assertEqual(self.probe(root, targets), {item["path"]: "ready" for item in targets})

    @staticmethod
    def probe(root, targets):
        observed = io.StringIO()
        with (mock.patch("pathlib.Path", side_effect=lambda path: root / Path(path).name),
              mock.patch.object(sys, "argv", ["-c", json.dumps(targets)]),
              contextlib.redirect_stdout(observed)):
            exec(delivery_guard.REMOTE_PROBE, {})
        return {k: v["state"] for k, v in json.loads(observed.getvalue()).items()}

    def test_secondary_parts_follow_declared_xml_types_through_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            book = root / "register.xlsx"
            submitted = root / "submission.zip"
            targets = [{"path": "/filesystem/register.xlsx", "kind": "file"},
                       {"path": "/filesystem/submission.zip", "kind": "file", "root_members": [book.name]}]
            cases = [({}, "ready"), ({"body": "<worksheet>"}, "invalid"),
                     ({"omit_sheet": True}, "invalid"), ({"kind": "chartsheet"}, "ready"),
                     ({"kind": "chartsheet", "body": f'<worksheet xmlns="{SHEET_NS}"/>'}, "invalid")]
            for options, expected in cases:
                with self.subTest(options=options):
                    self.workbook(book, SHEET_TYPE + "sheet.main+xml", **options)
                    with zipfile.ZipFile(submitted, "w") as outer:
                        outer.write(book, book.name)
                    self.assertEqual(self.probe(root, targets), {x["path"]: expected for x in targets})
                    if expected == "ready":
                        archive_checkpoint.verify(submitted, [book.name])
                    else:
                        with self.assertRaises(ValueError):
                            archive_checkpoint.verify(submitted, [book.name])
