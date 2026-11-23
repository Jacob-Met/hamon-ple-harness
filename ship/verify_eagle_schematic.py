"""Replay the prospective Mega 2560 Rev3e PDF results against the manufacturer's XML."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[1]
PDF_SHA = "4ce9d8ae628593d0bae73f98c653d60dcde1429d5a74bc9f17505e89e5570b23"
ZIP_SHA = "ba10b7df3afbd5086f801bb330f7fe951c2899b44a7ea364fd4783935c49d3f0"
SCHEMATIC_SHA = "1ff439fc6e1053903fdbaa46b061c150e043d84810e7629091519f918ded83bd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def align(wires, sheet):
    truth = [(n.get("name"), [(float(w.get("x1")), float(w.get("y1"))),
                              (float(w.get("x2")), float(w.get("y2")))])
             for n in sheet.findall("nets/net") for w in n.findall("segment/wire")]
    require(len(wires) == len(truth), "Visible wire count differs")
    offsets = []
    for axis, sign in ((0, 1), (1, -1)):
        ink = sorted(p[axis] for edge in wires for p in edge)
        design = sorted(sign * 72 / 25.4 * p[axis] for _, edge in truth for p in edge)
        diffs = sorted(x - y for x, y in zip(ink, design))
        offsets.append(diffs[len(diffs) // 2])
    assigned = {}
    for name, edge in truth:
        projected = [(offsets[0] + x * 72 / 25.4, offsets[1] - y * 72 / 25.4) for x, y in edge]
        possible = []
        for i, path in enumerate(wires):
            error = min(max(math.dist(a, b) for a, b in zip(projected, ordered))
                        for ordered in (path, path[::-1]))
            if error < .15:
                possible.append(i)
        require(len(possible) == 1 and possible[0] not in assigned, "Wire mapping is missing or ambiguous")
        assigned[possible[0]] = name
    return [assigned[i] for i in range(len(wires))]


def counts(rows):
    nets, groups = defaultdict(set), defaultdict(set)
    for i, (name, group) in enumerate(rows):
        nets[name].add(i)
        groups[group].add(i)
    pair = lambda n: n * (n - 1) // 2
    true_positive = sum(pair(n) for n in Counter(rows).values())
    return {
        "nets": len(nets),
        "exact": sum(v in groups.values() for v in nets.values()),
        "false_connected": sum(pair(len(s)) for s in groups.values()) - true_positive,
        "false_separated": sum(pair(len(s)) for s in nets.values()) - true_positive,
    }


def check_quarter_turns(pdf, original):
    """Compare physical endpoints and both wire partitions after rotating the PDF itself."""
    from pypdf import PdfWriter

    with tempfile.TemporaryDirectory() as tmp:
        for turn in (1, 2, 3):
            writer = PdfWriter(clone_from=pdf)
            sizes = [(float(p.cropbox.width), float(p.cropbox.height)) for p in writer.pages]
            for page in writer.pages:
                page.rotate(90 * turn)
            target = Path(tmp) / f"turned-{turn}.pdf"
            writer.write(target)
            actual = json.loads(subprocess.check_output([
                sys.executable, str(REPO / "engineering_toolkit.py"), "pdf-eagle-connectivity",
                str(target), "--flat-document",
            ]))
            require(actual["document_wire_groups"] == original["document_wire_groups"]
                    and actual["inter_sheet_anchors"] == original["inter_sheet_anchors"],
                    "Rotated document changed qualified joins or their witness groups")
            for old, new, (width, height) in zip(original["pages"], actual["pages"], sizes, strict=True):
                require(old["wire_groups"] == new["wire_groups"], "Rotated page changed local groups")
                for previous, current in zip(old["wires"], new["wires"], strict=True):
                    for (x, y), observed in zip(previous, current, strict=True):
                        expected = ((height - y, x), (width - x, height - y), (y, width - x))[turn - 1]
                        require(math.dist(expected, observed) < .01, "Rotated wire endpoint differs")
            print(json.dumps({"rotation_degrees": turn * 90, "covariant": True}))


def check_power_junction_controls(pdf, original):
    """Remove one physical feature at a time from the hash-verified manufacturer PDF."""
    from pypdf import PdfWriter
    from pypdf.generic import NumberObject

    old = original["document_wire_groups"][0]
    require(len({old[i] for i in range(147, 153)}) == 1, "Intact rail did not join both islands")
    reference_pairs = {(i, j) for i in range(len(old)) for j in range(i) if old[i] == old[j]}
    junctions = [w for w in original["pages"][0]["wire_end_label_witnesses"] if "junction" in w]
    require(len(junctions) == 1 and (junctions[0]["name"], junctions[0]["wire_index"],
            junctions[0]["stem_wire_index"]) == ("+3V3", 151, 150), "Unexpected physical junction")
    reference_pairs = {(i, j) for i in range(len(old)) for j in range(i) if old[i] == old[j]}
    junctions = [w for w in original["pages"][0]["wire_end_label_witnesses"] if "junction" in w]
    require(len(junctions) == 1 and (junctions[0]["name"], junctions[0]["wire_index"],
            junctions[0]["stem_wire_index"]) == ("+3V3", 151, 150), "Unexpected physical junction")
    with tempfile.TemporaryDirectory() as tmp:
        for edit in ("side_label", "stem_label", "stem_gap"):
            writer = PdfWriter(clone_from=pdf)
            page = writer.pages[0]
            contents = page.get_contents()
            ops = contents.operations
            if edit == "stem_gap":
                # First half of the T to capacitor lead; leave both label runs alone.
                moves = [i for i, (args, op) in enumerate(ops[:-1])
                         if op == b"m" and list(args) == [7919, 11600]
                         and ops[i+1] == ([7919, 11960], b"l")]
                require(len(moves) == 1, "Cannot localize the manufacturer branch")
                ops[moves[0]][0][1] = NumberObject(11630)
            else:
                # EAGLE emits one PDF text-show operation per glyph. These two
                # independent +3V3 runs are the capacitor lead and the open arm.
                glyphs = [i for i, (_, op) in enumerate(ops) if op == b"Tj"]
                start = 2842 if edit == "side_label" else 2838
                remove = glyphs[start:start+4]
                require(len(remove) == 4, "Cannot localize the manufacturer label")
                ops = [row for i, row in enumerate(ops) if i not in remove]
            contents.operations = ops
            page.replace_contents(contents)
            edited = Path(tmp) / f"{edit}.pdf"
            writer.write(edited)
            actual = json.loads(subprocess.check_output([
                sys.executable, str(REPO / "engineering_toolkit.py"), "pdf-eagle-connectivity",
                str(edited), "--flat-document",
            ]))
            require(actual["status"] == "candidate_evidence", "Edited PDF fell outside the supported profile")
            groups = actual["document_wire_groups"][0]
            actual_pairs = {(i, j) for i in range(len(groups)) for j in range(i) if groups[i] == groups[j]}
            withdrawn = {(i, j) for i in range(150, 153) for j in range(147, 150)}
            if edit == "stem_gap":
                withdrawn.update(((151, 150), (152, 150)))
            require(actual_pairs == reference_pairs - withdrawn, f"{edit} changed an unexpected connection")
            printed = actual["pages"][0]["wire_end_label_witnesses"]
            require(not any("junction" in w for w in printed), f"{edit} kept the junction witness")
            wires = actual["pages"][0]["wires"]
            stock = original["pages"][0]["wires"]
            changed = [i for i, (a, b) in enumerate(zip(wires, stock, strict=True)) if a != b]
            require(changed == ([150] if edit == "stem_gap" else []), "Unrelated native wire geometry changed")
            print(json.dumps({"power_junction_edit": edit, "join_withdrawn": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("vendor_archive", type=Path)
    parser.add_argument("--quarter-turns", action="store_true", help="Also replay all quarter-turn rotations using pypdf")
    parser.add_argument("--junction-controls", action="store_true", help="Erase either power label or disconnect the tee in the original PDF")
    args = parser.parse_args()
    require(sha256(args.pdf.read_bytes()).hexdigest() == PDF_SHA, "Unexpected vendor PDF")
    predictions = []
    for page in (1, 2):
        cmd = [sys.executable, str(REPO / "engineering_toolkit.py"), "pdf-eagle-connectivity",
               str(args.pdf), "--page", str(page)]
        output = json.loads(subprocess.check_output(cmd))
        require(output["status"] == "candidate_evidence" and output["pdf_sha256"] == PDF_SHA,
                "Published drawing profile could not be read")
        predictions.append((output["wires"], output["wire_groups"]))
    flat = json.loads(subprocess.check_output([
        sys.executable, str(REPO / "engineering_toolkit.py"), "pdf-eagle-connectivity",
        str(args.pdf), "--flat-document",
    ]))
    require(flat["status"] == "candidate_evidence" and flat["pdf_sha256"] == PDF_SHA,
            "Document reader returned an unsupported profile or different drawing")
    require([(p["wires"], p["wire_groups"]) for p in flat["pages"]] == predictions,
            "Document mode changed page geometry or local membership")
    require(len(flat["document_wire_groups"]) == len(predictions), "Document memberships omit sheets")
    terminals = [(page, w) for page, parsed in enumerate(flat["pages"], 1)
                 for w in parsed["wire_end_label_witnesses"] if "pin_bank_terminal" in w]
    require(len(terminals) == 1, "Expected one corroborated numbered bank")
    page, label = terminals[0]
    terminal = label["pin_bank_terminal"]
    require((page, label["name"], terminal["pin"], terminal["count"]) == (1, "VIN", 8, 8)
            and len(terminal["supporting_wire_islands"]) == 2
            and terminal["incidental_singletons"] == ["D1"],
            "Numbered terminal and independent text witnesses differ")
    if args.junction_controls:
        check_power_junction_controls(args.pdf, flat)
    if args.quarter_turns:
        check_quarter_turns(args.pdf, flat)
    require(sha256(args.vendor_archive.read_bytes()).hexdigest() == ZIP_SHA, "Unexpected vendor archive")
    with ZipFile(args.vendor_archive) as archive:
        cad = archive.read("MEGA2560_Rev3e.sch")
    require(sha256(cad).hexdigest() == SCHEMATIC_SHA, "Unexpected native schematic")
    sheets = ET.fromstring(cad).findall(".//sheets/sheet")
    require(len(sheets) == len(predictions), "Source sheet count differs")
    document = []
    flat_rows = []
    for page, ((wires, groups), sheet, expected) in enumerate(
            zip(predictions, sheets, ((86, 86, 0, 0), (31, 31, 0, 0))), 1):
        names = align(wires, sheet)
        rows = list(zip(names, [(page, group) for group in groups]))
        observed = counts(rows)
        require(tuple(observed[k] for k in ("nets", "exact", "false_connected", "false_separated")) == expected,
                f"Page {page} differs from the locked external result: {observed}")
        print(json.dumps({"page": page, "wire_sections": len(wires), **observed}))
        document.extend(rows)
        stitched = flat["document_wire_groups"][page - 1]
        require(len(stitched) == len(names), "Document memberships omit wires")
        flat_rows.extend(zip(names, stitched))
    result = counts(document)
    require(result == {"nets": 111, "exact": 105, "false_connected": 0, "false_separated": 1966},
            "Complete document differs from the independent recount")
    print(json.dumps({"combined_pages_without_inter_sheet_stitching": result}))
    result = counts(flat_rows)
    require(result == {"nets": 111, "exact": 111, "false_connected": 0, "false_separated": 0},
            f"Flat-document comparison differs: {result}")
    pairs_by_name = defaultdict(list)
    for name, group in flat_rows:
        pairs_by_name[name].append((name, group))
    missing = {name: total for name, rows in pairs_by_name.items()
               if (total := counts(rows)["false_separated"])}
    require(not missing,
            f"Remaining per-net separations differ: {missing}")
    print(json.dumps({"flat_document": result, "anchors": flat["inter_sheet_anchors"], "missing_by_net": missing}))


if __name__ == "__main__":
    main()
