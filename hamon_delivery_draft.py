"""Export completed participant data into an exact ZIP or a reconciled Word document."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from xml.dom import minidom
from zipfile import ZIP_DEFLATED, ZipFile

from archive_checkpoint import verify_typed_members
from hamon_witness_package import example, paragraphs, write_docx
from office_package import workbook_parts

TASK_ROOTS = (Path("/filesystem"), Path("/app"))


def safe_file(path):
    return (
        path.is_absolute()
        and any(path.is_relative_to(root) for root in TASK_ROOTS)
        and ".." not in path.parts
        and all(not parent.is_symlink() for parent in (path, *path.parents))
        and path.is_file()
    )


def coordinates(reference):
    match = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]{0,6})", reference)
    if not match:
        raise ValueError("invalid A1 coordinate")
    column = 0
    for character in match[1]:
        column = column * 26 + ord(character) - ord("A") + 1
    row = int(match[2])
    if column > 16384 or row > 1048576:
        raise ValueError("cell outside worksheet")
    return row, column


def children(node, name):
    return [item for item in node.childNodes
            if item.nodeType == item.ELEMENT_NODE and item.localName == name]


def element(parent, name):
    prefix = parent.prefix + ":" if parent.prefix else ""
    return parent.ownerDocument.createElementNS(parent.namespaceURI, prefix + name)


def patch_sheet(data, changes):
    if not isinstance(changes, dict) or not changes:
        raise ValueError("empty worksheet update")
    document = minidom.parseString(data)
    root = document.documentElement
    areas = children(root, "sheetData")
    if len(areas) != 1:
        raise ValueError("expected one worksheet data table")
    area = areas[0]
    rows = {int(row.getAttribute("r")): row for row in children(area, "row")}

    for reference, value in changes.items():
        number, column = coordinates(reference)
        if (not isinstance(value, (str, int, float, bool))
                or isinstance(value, float) and not math.isfinite(value)
                or isinstance(value, str) and not value.strip()):
            raise ValueError("expected a nonempty text or finite literal cell")
        row = rows.get(number)
        if row is None:
            row = element(area, "row")
            row.setAttribute("r", str(number))
            later = next((r for r in children(area, "row") if int(r.getAttribute("r")) > number), None)
            area.insertBefore(row, later)
            rows[number] = row

        cells = children(row, "c")
        cell = next((c for c in cells if c.getAttribute("r") == reference), None)
        if cell is None:
            cell = element(row, "c")
            cell.setAttribute("r", reference)
            later = next((c for c in cells if coordinates(c.getAttribute("r"))[1] > column), None)
            if later is None:
                later = next(iter(children(row, "extLst")), None)
            row.insertBefore(cell, later)
        if children(cell, "f"):
            raise ValueError("refusing to overwrite a template formula")
        for child in children(cell, "v") + children(cell, "is"):
            cell.removeChild(child)
        if isinstance(value, str):
            cell.setAttribute("t", "inlineStr")
            inline = element(cell, "is")
            node = element(inline, "t")
            node.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve")
            node.appendChild(document.createTextNode(value))
            inline.appendChild(node)
            new_content = inline
        else:
            cell.setAttribute("t", "b" if isinstance(value, bool) else "n")
            new_content = element(cell, "v")
            text = str(int(value)) if isinstance(value, bool) else str(value)
            new_content.appendChild(document.createTextNode(text))
        cell.insertBefore(new_content, next(iter(children(cell, "extLst")), None))
    # Let the reader recalculate the optional used range after inserting cells.
    for dimensions in children(root, "dimension"):
        root.removeChild(dimensions)
    return document.toxml(encoding="utf-8")


def invalidate_formula_cache(data):
    document = minidom.parseString(data)
    formulas = document.getElementsByTagNameNS(document.documentElement.namespaceURI, "f")
    changed = False
    for formula in formulas:
        for cached in children(formula.parentNode, "v"):
            formula.parentNode.removeChild(cached)
            changed = True
    return (document.toxml(encoding="utf-8") if changed else data), bool(formulas)


def require_recalculation(data):
    document = minidom.parseString(data)
    root = document.documentElement
    existing = children(root, "calcPr")
    settings = existing[0] if existing else element(root, "calcPr")
    if not existing:
        following = {"oleSize", "customWorkbookViews", "pivotCaches", "smartTagPr",
                     "smartTagTypes", "webPublishing", "fileRecoveryPr",
                     "webPublishObjects", "extLst"}
        later = next((c for c in root.childNodes
                      if c.nodeType == c.ELEMENT_NODE and c.localName in following), None)
        root.insertBefore(settings, later)
    settings.setAttribute("fullCalcOnLoad", "1")
    settings.setAttribute("forceFullCalc", "1")
    settings.setAttribute("calcMode", "auto")
    return document.toxml(encoding="utf-8")


def workbook(template, changes):
    path = Path(template)
    if not safe_file(path) or path.suffix.lower() != ".xlsx":
        raise ValueError("expected a task-local XLSX template")
    if not isinstance(changes, dict) or not changes:
        raise ValueError("expected worksheet names and cell values")
    output = io.BytesIO()
    with ZipFile(path) as source, ZipFile(output, "w", ZIP_DEFLATED) as destination:
        main, sheets = workbook_parts(source)
        selected = {row["part"]: row["name"] for row in sheets
                    if row["kind"] == "worksheet" and row["name"] in changes}
        if main is None or set(selected.values()) != set(changes):
            raise ValueError("unrecognized worksheet or template")
        worksheets = {row["part"] for row in sheets if row["kind"] == "worksheet"}
        has_formulas = False
        for item in source.infolist():
            if item.filename == main:
                continue
            data = source.read(item)
            if item.filename in selected:
                data = patch_sheet(data, changes[selected[item.filename]])
            if item.filename in worksheets:
                data, formulas = invalidate_formula_cache(data)
                has_formulas |= formulas
            destination.writestr(item, data)
        book = source.read(main)
        destination.writestr(source.getinfo(main), require_recalculation(book) if has_formulas else book)
    return output.getvalue()


def member_bytes(name, row):
    if Path(name).name != name or name in (".", "..") or not isinstance(row, dict):
        raise ValueError("invalid declared member")
    suffix = Path(name).suffix.lower()
    if suffix == ".json" and set(row) == {"json"} and isinstance(row["json"], (dict, list)) and row["json"]:
        return json.dumps(row["json"], allow_nan=False, ensure_ascii=False, indent=2).encode()
    if suffix in (".md", ".txt", ".csv") and set(row) == {"text"} and isinstance(row["text"], str) and row["text"].strip():
        return row["text"].encode()
    if suffix == ".xlsx" and set(row) == {"template", "cells"}:
        return workbook(row["template"], row["cells"])
    raise ValueError("unsupported or empty member: " + name)


def render(draft, target, names, fingerprint):
    draft, target = Path(draft), Path(target)
    if (draft != target.with_name(target.name + ".draft.json") or not safe_file(draft)
            or target.exists() or target.is_symlink()):
        raise ValueError("expected a fresh sibling draft and a missing destination")
    raw = draft.read_bytes()
    if hashlib.sha256(str(len(raw)).encode() + raw).hexdigest() != fingerprint:
        raise ValueError("draft changed after observation")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("invalid archive contract")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("expected an authored object")
    if target.suffix.lower() == ".docx":
        if names or manifest == example():
            raise ValueError("supply a project witness record")
        result, content = paragraphs(manifest)
        if not result["schedule_matches"]:
            raise ValueError("case and schedule do not reconcile")
        write_docx(target, content)
        return {"state": "executed"}

    members = manifest.get("members")
    if (target.suffix.lower() != ".zip" or not names or not isinstance(members, dict)
            or len(names) != len(set(names)) or set(names) != set(members)):
        raise ValueError("draft must supply every declared archive member")
    with tempfile.NamedTemporaryFile(dir=target.parent) as temporary:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name in names:
                archive.writestr(name, member_bytes(name, members[name]))
        temporary.flush()
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("archive checksum")
            verify_typed_members(archive, names)
        os.link(temporary.name, target)
    return {"state": "executed"}
