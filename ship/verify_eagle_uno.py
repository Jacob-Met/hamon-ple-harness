"""Replay Uno TH Rev3e visible-wire connectivity against its manufacturer CAD."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from verify_eagle_schematic import align, check_quarter_turns, counts, require

REPO = Path(__file__).resolve().parents[1]
PDF_SHA = "4f09f63b8ec0246cf54a99335673ccd9e8e7e62d4665d6b11f70534420b2952b"
ZIP_SHA = "52ee88291eae55d2939bf62ea0700a6817dad0e9e3326005b161a54d473dfbe3"
SCHEMATIC_SHA = "f44b5a90b74757e91477fb3adb776ffc556d9ea5dd8cc6ef3c79c1bbba0aee23"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("vendor_archive", type=Path)
    parser.add_argument("--quarter-turns", action="store_true")
    args = parser.parse_args()
    require(sha256(args.pdf.read_bytes()).hexdigest() == PDF_SHA, "Unexpected vendor PDF")
    prediction = json.loads(subprocess.check_output([
        sys.executable, str(REPO / "engineering_toolkit.py"), "pdf-eagle-connectivity",
        str(args.pdf), "--flat-document",
    ]))
    require(prediction["status"] == "candidate_evidence" and prediction["pdf_sha256"] == PDF_SHA,
            "Published drawing profile could not be read")
    require(len(prediction["pages"]) == 1 and len(prediction["document_wire_groups"]) == 1,
            "Unexpected sheet count")
    page = prediction["pages"][0]
    require(len(page["wires"]) == 447, "Unexpected visible wire count")
    require(page["wire_groups"] == prediction["document_wire_groups"][0],
            "One-sheet document changed local wire membership")
    if args.quarter_turns:
        check_quarter_turns(args.pdf, prediction)
    # All PDF-only predictions, including rotations, precede opening the native CAD.
    require(sha256(args.vendor_archive.read_bytes()).hexdigest() == ZIP_SHA, "Unexpected vendor archive")
    with ZipFile(args.vendor_archive) as archive:
        native = archive.read("UNO-TH_Rev3e.sch")
    require(sha256(native).hexdigest() == SCHEMATIC_SHA, "Unexpected native schematic")
    sheets = ET.fromstring(native).findall(".//schematic/sheets/sheet")
    require(len(sheets) == 1, "Unexpected native sheet count")
    names = align(page["wires"], sheets[0])
    observed = counts(list(zip(names, prediction["document_wire_groups"][0], strict=True)))
    require(observed == {"nets": 61, "exact": 61, "false_connected": 0, "false_separated": 0},
            f"Uno visible-wire comparison differs: {observed}")
    side = [m for m in page["supply_glyphs"] if m.get("label_placement") == "side"]
    require(len(side) == 2 and {m["name"] for m in side} == {"+5V"},
            "Expected two independently witnessed side-arrow values")
    print(json.dumps({
        "drawing": "Arduino Uno TH Rev3e", "wire_sections": len(names), **observed,
        "side_arrow_values": len(side), "pdf_sha256": PDF_SHA,
        "reader_sha256": sha256((REPO / "hamon_eagle_schematic.py").read_bytes()).hexdigest(),
        "prediction_sha256": sha256(json.dumps(prediction, sort_keys=True).encode()).hexdigest(),
        "scope": "visible wire memberships; device pins and a PLE hosted score are not measured",
    }))


if __name__ == "__main__":
    main()
