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
        details = []
        for i,n,occ,r in o.wire_words(pg["wires"],segments,pitch):
            if r and n not in ("TX","16U2-TX","UOTGID","XUSB","MASTER-RESET"):
                continue
            vertical, edges = occ
            def project(point):
                return (-point[1],point[0]) if vertical else point
            wire = list(map(project, pg["wires"][i]))
            lo,hi = sorted(p[0] for p in wire)
            start,end = min(p[0] for e in edges for p in e),max(p[0] for e in edges for p in e)
            details.append({"wire":i,"name":n,"registered":r,"line":wire,"ink_x":[start,end],"ink_y":[min(p[1] for e in edges for p in e),max(p[1] for e in edges for p in e)],"pitch":pitch,
                            "left_offset":(start-lo)/pitch,"right_offset":(hi-end)/pitch})
        print(json.dumps({"page":pg["page"],"unregistered":details}),flush=True)
    with ZipFile(args.archive) as archive:
        cad = archive.read("A000056-cad-files/DUE_V02g.sch")
    sheets = ET.fromstring(cad).findall(".//sheets/sheet")
    rows, groups = [], defaultdict(lambda:defaultdict(list))
    for page,(sheet,pred) in enumerate(zip(sheets,predicted["pages"]),1):
        names=align(pred["wires"],sheet)
        print(json.dumps({"page":page,"native_unregistered":[(k,names[k]) for k in range(len(names)) if k in ({173,208,227,256,261,267,268,302,303,304,318,327,367,368,390,397,400} if page==1 else {179,183,187,189,404,244})]}),flush=True)
        for i,(name,g) in enumerate(zip(names,predicted["document_wire_groups"][page-1])):
            rows.append((name,g)); groups[name][g].append((page,i))
    print(json.dumps({"metric":counts(rows),"incomplete":{n:list(g.values()) for n,g in groups.items() if len(g)>1}}),flush=True)
    for i,sheet in enumerate(sheets,1):
        print(json.dumps({"native_page":i,"incomplete_labels":{net.get("name"):[dict(label.attrib) for label in net.findall("segment/label")] for net in sheet.findall("nets/net") if len(groups[net.get("name")])>1}}),flush=True)

if __name__ == "__main__":
    main()
