"""Inspect complete outlined words before opening the paired native schematic."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hamon_drawing as d
import hamon_eagle_outlined as o
from hamon_eagle_schematic import document_connectivity
from verify_eagle_schematic import align, counts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    predicted = document_connectivity(args.pdf)
    for pg in predicted["pages"][:2]:
        with tempfile.TemporaryDirectory() as folder:
            svg = d._page_svg(args.pdf, pg["page"], folder)
        paths = defaultdict(list)
        for e in ET.fromstring(svg):
            if d.local(e.tag) == "path" and e.get("fill") == "none" and e.get("stroke"):
                paths[e.get("stroke")].append((e, d.parse_transform(e.get("transform"))))
        pitch = max(pg["junction_grid"]["pitch_pt"])
        _, segments = o.letter_strokes(paths, pg["junction_grid"]["stroke"], pitch)
        print(json.dumps({"page": pg["page"], "words": [(i,n,r) for i,n,_,r in o.wire_words(pg["wires"],segments,pitch)]}), flush=True)
    with ZipFile(args.archive) as archive:
        cad = archive.read("A000056-cad-files/DUE_V02g.sch")
    sheets = ET.fromstring(cad).findall(".//sheets/sheet")
    rows, groups = [], defaultdict(lambda:defaultdict(list))
    for page,(sheet,pred) in enumerate(zip(sheets,predicted["pages"]),1):
        names=align(pred["wires"],sheet)
        for i,(name,g) in enumerate(zip(names,predicted["document_wire_groups"][page-1])):
            rows.append((name,g)); groups[name][g].append((page,i))
    print(json.dumps({"metric":counts(rows),"incomplete":{n:list(g.values()) for n,g in groups.items() if len(g)>1}}),flush=True)

if __name__ == "__main__":
    main()
