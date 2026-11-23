"""Compare an outlined schematic PDF to its native wires after freezing every PDF prediction."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from verify_eagle_schematic import REPO, align, counts, require


PDF_SHA = "a45cd2cf823f47bb4219870ad808a672140b1d82a7cdd534906c55c763d16557"
ARCHIVE_SHA = "6ec205e4ea1410109c1a960a56d18190b597bebaf3f4f62b3a6bfbed920838b9"
SCHEMATIC_SHA = "60f449078bea22dc298a0cad5b9733affb1214f6052f778e54babb08531e9734"


def read_pdf(path):
    return json.loads(subprocess.check_output([
        sys.executable, str(REPO / "engineering_toolkit.py"), "pdf-eagle-connectivity",
        str(path), "--flat-document",
    ]))


def edited_pdf(source, destination, edit):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ContentStream, FloatObject, NameObject, NumberObject

    reader, writer = PdfReader(source), PdfWriter()
    writer.append(reader)
    writer.add_metadata({"/Creator": "EAGLE" if edit == "recolor" else "PDF assembler"})
    for i, page in enumerate(writer.pages):
        content = ContentStream(page.get_contents(), writer)
        green = False
        dot_count = 0
        operations = []
        for operands, op in content.operations:
            if op in (b"scn", b"SCN") and len(operands) == 3:
                is_green = tuple(round(float(v), 3) for v in operands) == (0, .502, 0)
                if op == b"SCN":
                    green = is_green
                if edit == "recolor" and is_green:
                    operands = [NumberObject(0), NumberObject(0), FloatObject(.502)]
            if green and op == b"B*" and i < 2:
                # Mark 0 is at a lone wire end. Mark 3 is the first singly
                # printed dot with three distinct incident wire sections.
                erase_index = {"erase_one": 3, "erase_terminal_mark": 0}.get(edit)
                if edit == "erase_dots" or (i == 0 and dot_count == erase_index):
                    op = b"n"
                dot_count += 1
            operations.append((operands, op))
        if i < 2:
            require(dot_count == (94, 90)[i], "Physical source junction count changed")
        if i == 2 and edit == "appendix_stroke":
            operations.extend([([], b"q"), ([NumberObject(0)], b"G"),
                               ([NumberObject(60), NumberObject(90)], b"m"),
                               ([NumberObject(200), NumberObject(90)], b"l"),
                               ([], b"S"), ([], b"Q")])
        content.operations = operations
        page[NameObject("/Contents")] = writer._add_object(content)
    with destination.open("wb") as output:
        writer.write(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("vendor_archive", type=Path)
    parser.add_argument("--controls", action="store_true", help="Apply physical PDF edits with pypdf")
    args = parser.parse_args()
    require(sha256(args.pdf.read_bytes()).hexdigest() == PDF_SHA, "Original PDF required")
    baseline = read_pdf(args.pdf)
    require(baseline["status"] == "candidate_evidence" and
            [p["status"] for p in baseline["pages"]] == ["candidate_evidence"] * 2 + ["text_only_license"]
            and baseline["document_wire_groups"][2] == [] and not baseline["inter_sheet_anchors"],
            "Unexpected sheet or appendix classification")
    require([len(p["wires"]) for p in baseline["pages"]] == [468, 407, 0], "Unexpected wire roster")
    controls = {}
    if args.controls:
        with tempfile.TemporaryDirectory() as directory:
            for edit in ("recolor", "erase_one", "erase_terminal_mark", "erase_dots", "appendix_stroke"):
                path = Path(directory, edit + ".pdf")
                edited_pdf(args.pdf, path, edit)
                controls[edit] = read_pdf(path)
        blue, one, erased, extra = (controls[k] for k in ("recolor", "erase_one", "erase_dots", "appendix_stroke"))
        require(blue["document_wire_groups"] == baseline["document_wire_groups"]
                and all(x["wires"] == y["wires"] for x, y in zip(blue["pages"], baseline["pages"])),
                "Substituting blue ink and EAGLE metadata changed the prediction")
        require(all(x["wires"] == y["wires"] for x, y in zip(one["pages"], baseline["pages"]))
                and len(set(one["document_wire_groups"][0])) > len(set(baseline["document_wire_groups"][0])),
                "Erasing one physical junction did not split its incident wires")
        before, after = baseline["document_wire_groups"][0], one["document_wire_groups"][0]
        lost, gained = set(), set()
        for i in range(len(before)):
            for j in range(i):
                if (before[i] == before[j]) != (after[i] == after[j]):
                    (lost if before[i] == before[j] else gained).add((i, j))
        require(lost == {(i, j) for i in (12, 13, 14) for j in (9, 10, 11)} | {(14, 12), (14, 13)}
                and not gained and one["pages"][1]["wire_groups"] == baseline["pages"][1]["wire_groups"],
                "Single junction deletion changed unexpected wire pairs")
        require(controls["erase_terminal_mark"]["document_wire_groups"] == baseline["document_wire_groups"],
                "Erasing a mark at a single wire end changed membership")
        require(erased["document_wire_groups"] is None and extra["document_wire_groups"] is None,
                "Unmarked network or graphical appendix was silently admitted")
    require(sha256(args.vendor_archive.read_bytes()).hexdigest() == ARCHIVE_SHA, "Unexpected CAD archive")
    with ZipFile(args.vendor_archive) as archive:
        cad = archive.read("A000056-cad-files/DUE_V02g.sch")
    require(sha256(cad).hexdigest() == SCHEMATIC_SHA, "Unexpected native schematic")
    truth = ET.fromstring(cad).findall(".//sheets/sheet")
    require(len(truth) == 2, "Native sheet count differs")
    document = []
    for index, (sheet, expected) in enumerate(zip(truth, ((138, 128, 0, 8411), (125, 108, 0, 6587)))):
        page = baseline["pages"][index]
        names = align(page["wires"], sheet)
        rows = list(zip(names, baseline["document_wire_groups"][index], strict=True))
        observed = counts(rows)
        require(tuple(observed.values()) == expected, f"Sheet metric changed: {observed}")
        document.extend(rows)
        print(json.dumps({"page": index + 1, "wire_sections": len(names), **observed}))
    result = counts(document)
    require(result == {"nets": 176, "exact": 77, "false_connected": 0, "false_separated": 29874},
            f"Document metric changed: {result}")
    pairs = sum(n * (n - 1) // 2 for n in Counter(name for name, _ in document).values())
    print(json.dumps({"document": result, "correct_pairs": pairs - result["false_separated"],
                      "native_pairs": pairs, "physical_controls_passed": sorted(controls)}))


if __name__ == "__main__":
    main()
