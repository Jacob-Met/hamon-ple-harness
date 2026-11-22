"""Replay the prospective Mega 2560 Rev3e PDF results against the manufacturer's XML."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("vendor_archive", type=Path)
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
    require(sha256(args.vendor_archive.read_bytes()).hexdigest() == ZIP_SHA, "Unexpected vendor archive")
    with ZipFile(args.vendor_archive) as archive:
        cad = archive.read("MEGA2560_Rev3e.sch")
    require(sha256(cad).hexdigest() == SCHEMATIC_SHA, "Unexpected native schematic")
    sheets = ET.fromstring(cad).findall(".//sheets/sheet")
    require(len(sheets) == len(predictions), "Source sheet count differs")
    document = []
    flat_rows = []
    for page, ((wires, groups), sheet, expected) in enumerate(
            zip(predictions, sheets, ((86, 82, 0, 74), (31, 31, 0, 0))), 1):
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
    require(result == {"nets": 111, "exact": 102, "false_connected": 0, "false_separated": 2040},
            "Complete document differs from the independent recount")
    print(json.dumps({"combined_pages_without_inter_sheet_stitching": result}))
    result = counts(flat_rows)
    require(result == {"nets": 111, "exact": 106, "false_connected": 0, "false_separated": 165},
            f"Flat-document comparison differs: {result}")
    pairs_by_name = defaultdict(list)
    for name, group in flat_rows:
        pairs_by_name[name].append((name, group))
    missing = {name: total for name, rows in pairs_by_name.items()
               if (total := counts(rows)["false_separated"])}
    require(missing == {"RESET": 78, "GND": 66, "+3V3": 11, "VIN": 8, "AREF": 2},
            f"Remaining per-net separations differ: {missing}")
    print(json.dumps({"flat_document": result, "anchors": flat["inter_sheet_anchors"], "missing_by_net": missing}))


if __name__ == "__main__":
    main()
