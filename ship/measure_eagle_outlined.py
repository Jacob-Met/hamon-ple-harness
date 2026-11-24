"""Measure a frozen outlined-PDF candidate without changing its qualification gate."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from verify_eagle_outlined import ARCHIVE_SHA, PDF_SHA, SCHEMATIC_SHA, edited_pdf, read_pdf
from verify_eagle_schematic import align, counts, require


EDITS = ("recolor", "erase_one", "erase_terminal_mark", "erase_dots", "appendix_stroke",
         "erase_ground_bar", "break_ground_G", "move_ground_word_toward_MOSI",
         "erase_voltage_stroke", "erase_signal_underscore")


def measure(forecast, truth):
    if forecast.get("document_wire_groups") is None:
        return {"withheld": True}
    document, pages = [], []
    non_ground, fragments = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for index, sheet in enumerate(truth):
        page = forecast["pages"][index]
        names = align(page["wires"], sheet)
        rows = list(zip(names, forecast["document_wire_groups"][index], strict=True))
        pages.append(counts(rows))
        document.extend(rows)
        for wire, (name, group) in enumerate(rows):
            fragments[name][group].append([index + 1, wire])
            if name != "GND":
                non_ground[group].append([index + 1, wire])
    roster = json.dumps(sorted(non_ground.values()), separators=(",", ":")).encode()
    return {"document": counts(document), "pages": pages,
            "non_ground_partition_sha256": sha256(roster).hexdigest(),
            "incomplete": {name: list(groups.values()) for name, groups in fragments.items()
                           if len(groups) > 1}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("vendor_archive", type=Path)
    args = parser.parse_args()
    require(sha256(args.pdf.read_bytes()).hexdigest() == PDF_SHA, "Original PDF required")
    forecasts = {"original": read_pdf(args.pdf)}
    with tempfile.TemporaryDirectory() as directory:
        for edit in EDITS:
            target = Path(directory, edit + ".pdf")
            edited_pdf(args.pdf, target, edit)
            forecasts[edit] = read_pdf(target)
    # Freeze every unedited and edited prediction before opening the CAD oracle.
    print("PREDICTIONS_FROZEN " + json.dumps({
        name: sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        for name, value in forecasts.items()}), flush=True)
    require(sha256(args.vendor_archive.read_bytes()).hexdigest() == ARCHIVE_SHA, "Unexpected CAD archive")
    with ZipFile(args.vendor_archive) as archive:
        cad = archive.read("A000056-cad-files/DUE_V02g.sch")
    require(sha256(cad).hexdigest() == SCHEMATIC_SHA, "Unexpected native schematic")
    truth = ET.fromstring(cad).findall(".//sheets/sheet")
    require(len(truth) == 2, "Native sheet count differs")
    for name, forecast in forecasts.items():
        result = measure(forecast, truth)
        if name != "original":
            result.pop("incomplete", None)
        else:
            result["anchors"] = [p["printed_equivalence_anchors"] for p in forecast["pages"][:2]]
        print("OBSERVED " + json.dumps({"variant": name, **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
