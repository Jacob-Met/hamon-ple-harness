"""Compare an outlined schematic PDF to its native wires after freezing every PDF prediction."""
import argparse
from collections import Counter, defaultdict
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


def partition(groups):
    islands = defaultdict(list)
    for index, group in enumerate(groups):
        islands[group].append(index)
    return sorted(islands.values())


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
        ground_edits = 0
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
            if op == b"S" and len(operations) >= 2:
                (a, move), (b, line) = operations[-2:]
                if move == b"m" and line == b"l":
                    pair = (tuple(a), tuple(b))
                    targets = {
                        "break_xref_pointer": (1, ((6539, 4282), (6467, 4354))),
                        "erase_xref_border_A": (0, ((818, 2237), (894, 2237))),
                        "erase_mirrored_D": (0, ((3029, 2762), (3029, 2839))),
                        "erase_quarter_J": (1, ((5073, 12064), (5086, 12077))),
                        "erase_packed_plus": (0, ((11715, 8402), (11766, 8402))),
                        "erase_packed_V": (0, ((11754, 8414), (11779, 8440))),
                        "erase_wire_voltage_plus": (0, ((12182, 2081), (12234, 2081))),
                        "erase_corroborating_voltage_plus": (0, ((16622, 2561), (16673, 2561))),
                        "erase_vin_I_stem": (0, ((9033, 1922), (9033, 1999))),
                        "erase_small_reset_M": (1, ((5342, 3605), (5342, 3679))),
                    }
                    if edit in targets and i == targets[edit][0] and set(pair) == set(targets[edit][1]):
                        op = b"n"
                        ground_edits += 1
                    if (edit == "add_packed_foreign_stroke" and i == 0 and
                            set(pair) == {(11715, 8402), (11766, 8402)}):
                        # Paint the original plus and then a separate off-grid
                        # stroke in the shared caption margin using the same pen.
                        operations.extend([([], b"S"),
                                           ([NumberObject(11725), NumberObject(8406)], b"m"),
                                           ([NumberObject(11752), NumberObject(8420)], b"l")])
                        ground_edits += 1
                    if (i == 0 and edit == "xref_wrong_zone" and
                            all(2794 <= x <= 2845 and 1484 <= y <= 1561 for x,y in pair)):
                        # Physically replace the last B in XUSB/2.3B by a complete
                        # C in the same pen and lattice. Both captions still decode.
                        if ground_edits == 0:
                            operations.append(([], b"n"))
                            letter = [(4,1),(3,0),(1,0),(0,1),(0,5),(1,6),(3,6),(4,5)]
                            def to_pdf(point):
                                x,y = point
                                return [NumberObject(round(2794+51*x/4)), NumberObject(round(1561-77*y/6))]
                            for first,last in zip(letter,letter[1:]):
                                operations.extend([(to_pdf(first), b"m"), (to_pdf(last), b"l"), ([], b"S")])
                        op = b"n"
                        ground_edits += 1
                    deleted = {
                        "erase_voltage_stroke": ((16898, 2273), (16949, 2273)),
                        "erase_signal_underscore": ((1453, 11490), (1504, 11490)),
                    }
                    if i == 0 and edit in deleted and set(pair) == set(deleted[edit]):
                        op = b"n"
                        ground_edits += 1
                    if i == 0 and ((edit == "erase_ground_bar" and pair == ((16258, 7492), (16258, 7312))) or
                            (edit == "break_ground_G" and pair == ((16311, 7467), (16298, 7480)))):
                        op = b"n"
                        ground_edits += 1
                    if i == 0 and edit == "move_ground_word_toward_MOSI" and all(
                            16298 <= x <= 16375 and 7312 <= y <= 7518 for x, y in pair):
                        # The adjacent MOSI wire lies one 120-unit grid step to the left.
                        for n, (x, y) in zip((-2, -1), pair):
                            operations[n] = ([NumberObject(x - 120), NumberObject(y)], operations[n][1])
                        ground_edits += 1
            operations.append((operands, op))
        if i == 0 and edit in ("erase_ground_bar", "break_ground_G", "move_ground_word_toward_MOSI",
                               "erase_voltage_stroke", "erase_signal_underscore"):
            require(ground_edits == (18 if edit == "move_ground_word_toward_MOSI" else 1),
                    "Expected original painted ground strokes missing")
        if (edit, i) in (("erase_mirrored_D",0),("erase_quarter_J",1),("erase_small_reset_M",1),
                         ("erase_wire_voltage_plus",0), ("erase_corroborating_voltage_plus",0),
                         ("erase_vin_I_stem",0), ("erase_packed_plus",0), ("erase_packed_V",0),
                         ("add_packed_foreign_stroke",0), ("break_xref_pointer",1),
                         ("erase_xref_border_A",0)):
            require(ground_edits == 1, "Expected complete-word source stroke missing")
        if i == 0 and edit == "xref_wrong_zone":
            require(ground_edits == 10, "Expected all ten B strokes in the outlined target-zone caption")
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
            and baseline["document_wire_groups"][2] == []
            and {"GND", "+3V3", "JTAG_TMS", "JTAG_TCK"} <= {link["name"] for link in baseline["inter_sheet_anchors"]},
            "Unexpected sheet or appendix classification")
    require([len(p["wires"]) for p in baseline["pages"]] == [468, 407, 0], "Unexpected wire roster")
    positive = ([41, 42, 43, 45, 48, 56, 60, 63, 66, 67, 71, 102, 108, 111, 113, 118, 119,
                 121, 122, 123, 125, 127, 155, 156, 157, 160, 162, 165, 168, 169, 170, 171],
                [3, 5, 10, 11, 12, 15, 34, 35, 36, 51, 72, 73, 75, 76, 77, 78, 80])
    for page, indices in zip(baseline["pages"], positive):
        require(page["printed_equivalence_anchors"].get("GND") == indices, "Ground anchor roster changed")
    page1, page2 = baseline["pages"][:2]
    require(page1["printed_equivalence_anchors"]["+5V"] == [1, 2, 4, 14, 16, 24, 28, 31, 40]
            and page1["printed_equivalence_anchors"]["+3V3"] == [191, 194, 196, 198, 202]
            and page1["printed_equivalence_anchors"]["VIN"] == [172, 173, 175, 184]
            and page2["printed_equivalence_anchors"]["+3V3"] == [104, 109, 138, 142, 147, 148, 150, 153, 158, 160],
            "Complete voltage arrows no longer have the same physical endpoints")
    for page, tms, tck in ((page1, 230, 231), (page2, 178, 182)):
        anchors = page["printed_equivalence_anchors"]
        require(anchors["JTAG_TMS"] == ([tms] if page is page1 else [tms, 179])
                and anchors["JTAG_TCK"] == ([tck] if page is page1 else [tck, 183])
                and "JTAG" not in anchors and page["wire_groups"][tms] != page["wire_groups"][tck],
                "A printed underscore, following suffix, or distinct wire was lost")
    def reference_rows(page):
        return [(r["name"], r["wire_index"], r["zone"], r["target_page"], r["target_zone"])
                for r in page.get("outlined_cross_references", [])]
    require(reference_rows(page1) == [("XUSB",264,"1A",2,"3B")]
            and reference_rows(page2) == [("XUSB",234,"3B",1,"1A")],
            "Printed source and destination frame evidence changed")
    controls = {}
    if args.controls:
        with tempfile.TemporaryDirectory() as directory:
            for edit in ("recolor", "erase_one", "erase_terminal_mark", "erase_dots", "appendix_stroke",
                         "erase_ground_bar", "break_ground_G", "move_ground_word_toward_MOSI",
                         "erase_voltage_stroke", "erase_signal_underscore", "erase_mirrored_D",
                         "erase_quarter_J", "erase_small_reset_M", "erase_wire_voltage_plus",
                         "erase_corroborating_voltage_plus", "erase_vin_I_stem",
                         "erase_packed_plus", "erase_packed_V", "add_packed_foreign_stroke",
                         "break_xref_pointer", "erase_xref_border_A", "xref_wrong_zone"):
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
        expected = sum(baseline["document_wire_groups"], [])
        for i in (9, 10, 11):
            expected[i] = -1
        for i in (12, 13):
            expected[i] = -2
        require(partition(sum(one["document_wire_groups"], [])) == partition(expected),
                "Deleting the junction must detach only the three- and two-wire branches from +5V")
        require(controls["erase_terminal_mark"]["document_wire_groups"] == baseline["document_wire_groups"],
                "Erasing a mark at a single wire end changed membership")
        require(erased["document_wire_groups"] is None and extra["document_wire_groups"] is None,
                "Unmarked network or graphical appendix was silently admitted")
        for edit in ("erase_ground_bar", "break_ground_G", "move_ground_word_toward_MOSI"):
            changed = controls[edit]
            require(all(x["wires"] == y["wires"] for x, y in zip(changed["pages"], baseline["pages"])),
                    f"{edit} changed visible wire geometry")
            expected_groups = [*baseline["document_wire_groups"][0], *baseline["document_wire_groups"][1]]
            expected_groups[41] = -1  # Removing either witness isolates just this one-wire component.
            actual = [*changed["document_wire_groups"][0], *changed["document_wire_groups"][1]]
            require(partition(actual) == partition(expected_groups),
                    f"{edit} failed to isolate only wire 41 or joined a decoy")
        for edit, name, index in (("erase_voltage_stroke", "+3V3", 196),
                                  ("erase_signal_underscore", "JTAG_TMS", 230)):
            changed = controls[edit]
            anchors = changed["pages"][0]["printed_equivalence_anchors"]
            require(all(a["wires"] == b["wires"] for a, b in zip(changed["pages"], baseline["pages"]))
                    and index not in anchors.get(name, []) and "JTAG" not in anchors
                    and anchors["JTAG_TCK"] == [231], f"{edit} retained the erased evidence or joined a prefix")
            expected = sum(baseline["document_wire_groups"], [])
            separated = range(195, 202) if edit == "erase_voltage_stroke" else [230]
            for wire in separated:
                expected[wire] = -1
            if edit == "erase_voltage_stroke":
                expected[194] = -2  # Its peer's island has lost the independent supply symbol.
            require(partition(sum(changed["document_wire_groups"], [])) == partition(expected),
                    f"{edit} changed an unrelated pair or retained the erased witness")
        for edit in ("erase_packed_plus", "erase_packed_V", "add_packed_foreign_stroke"):
            changed = controls[edit]
            expected = sum(baseline["document_wire_groups"], [])
            for index in (0, 1):
                expected[index] = -1
            for index in range(189, 194):
                expected[index] = -2
            require(all(a["wires"] == b["wires"] for a, b in zip(changed["pages"], baseline["pages"])) and
                    partition(sum(changed["document_wire_groups"], [])) == partition(expected),
                    f"{edit} admitted unexplained ink or changed an unrelated physical connection")
        for edit in ("break_xref_pointer", "erase_xref_border_A", "xref_wrong_zone"):
            changed = controls[edit]
            expected = sum(baseline["document_wire_groups"], [])
            expected[468+234] = -1
            require(all(a["wires"] == b["wires"] for a,b in zip(changed["pages"], baseline["pages"]))
                    and partition(sum(changed["document_wire_groups"], [])) == partition(expected),
                    f"{edit} retained an uncorroborated XUSB join or changed an unrelated wire")
        require(reference_rows(controls["xref_wrong_zone"]["pages"][0]) == [("XUSB",264,"1A",2,"3C")]
                and reference_rows(controls["xref_wrong_zone"]["pages"][1]) == reference_rows(page2),
                "Wrong-zone control must preserve two legible captions with incompatible destinations")
        for edit, separated in (
                ("erase_wire_voltage_plus", [194]),
                ("erase_corroborating_voltage_plus", [194]),
                ("erase_vin_I_stem", list(range(175, 181))),
                ("erase_mirrored_D", list(range(224, 230))),
                ("erase_quarter_J", [468 + i for i in range(179, 182)]),
                ("erase_small_reset_M", [468 + i for i in range(348, 354)])):
            changed = controls[edit]
            require(all(a["wires"] == b["wires"] for a,b in zip(changed["pages"],baseline["pages"])),
                    f"{edit} changed wire geometry")
            expected = sum(baseline["document_wire_groups"], [])
            for index in separated:
                expected[index] = -1
            require(partition(sum(changed["document_wire_groups"], [])) == partition(expected),
                    f"{edit} retained an incomplete word or changed an unrelated connection")
    require(sha256(args.vendor_archive.read_bytes()).hexdigest() == ARCHIVE_SHA, "Unexpected CAD archive")
    with ZipFile(args.vendor_archive) as archive:
        cad = archive.read("A000056-cad-files/DUE_V02g.sch")
    require(sha256(cad).hexdigest() == SCHEMATIC_SHA, "Unexpected native schematic")
    truth = ET.fromstring(cad).findall(".//sheets/sheet")
    require(len(truth) == 2, "Native sheet count differs")
    document = []
    other_islands = defaultdict(list)
    for index, (sheet, expected) in enumerate(zip(truth, ((138, 136, 0, 6), (125, 125, 0, 0)))):
        page = baseline["pages"][index]
        names = align(page["wires"], sheet)
        rows = list(zip(names, baseline["document_wire_groups"][index], strict=True))
        observed = counts(rows)
        require(tuple(observed.values()) == expected, f"Sheet metric changed: {observed}")
        document.extend(rows)
        for i, (name, group) in enumerate(rows):
            if name != "GND":
                other_islands[group].append([index + 1, i])
        print(json.dumps({"page": index + 1, "wire_sections": len(names), **observed}))
    # Frozen after the PDF-only voltage and signal run, independent of numeric group IDs.
    other_roster = json.dumps(sorted(other_islands.values()), separators=(",", ":")).encode()
    require(sha256(other_roster).hexdigest() == "794cc14d871e996b4ca9c59e43826e234a5751d383fd5d11b643371c600c275c",
            "The partition of the 640 non-ground wire sections changed")
    result = counts(document)
    require(result == {"nets": 176, "exact": 174, "false_connected": 0, "false_separated": 9},
            f"Document metric changed: {result}")
    pairs = sum(n * (n - 1) // 2 for n in Counter(name for name, _ in document).values())
    print(json.dumps({"document": result, "correct_pairs": pairs - result["false_separated"],
                      "native_pairs": pairs, "physical_controls_passed": sorted(controls)}))


if __name__ == "__main__":
    main()

