"""Independent manufacturer-board artwork controls for the public eagle-netlist CLI."""
import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

SOURCES = {
    "due": ("due-0.sch", "due-1.brd", "N$3", 605, 163,
            "60f449078bea22dc298a0cad5b9733affb1214f6052f778e54babb08531e9734",
            "fd7cc5f9a14623ff62a7555c826373fe8f842cca400f05ec4f7b4c5589e1780a"),
    "feather": ("feather/Adafruit Feather RP2040 Original.sch", "feather/Adafruit Feather RP2040 Original.brd",
                "MISO", 207, 52,
                "01aae2acd843c9f626d7119e77142e2897c8e98c5529a7c0faea2ce45402e422",
                "ed5eff88e1ab8f1cc5ae8db1a1188c1932a65c35a5df9f06b8910f73ff65b0d7"),
}
# Each entry fixes its independent expected result before candidate execution.
# Numbers name source-model dimensions in micrometres.
CASES = {
    "due-original": True, "feather-original": True,
    "due-bore-127": True, "due-bore-204": True, "due-bore-254": False, "due-bore-280": False,
    "due-footprint-127": True, "due-footprint-600": False,
    "due-caps-234": True, "due-caps-254": False, "due-caps-274": False,
    "due-pad-30": True, "due-pad-0": False, "due-pad--10": False,
    "due-detour-open": True, "due-detour-cut": False,
    "feather-ring-25": True, "feather-ring-0": False, "feather-ring--25": False,
}
C = (12.4049, 28.4999)
VIA = (32.4485, 6.731)
EPS = 1e-9  # mm; float roundoff only, never a physical contact tolerance


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fmt(v):
    return f"{v:.7f}".rstrip("0").rstrip(".")


def set_end(wire, which, p):
    for dim, v in zip("xy", p):
        wire.set(dim + str(which), fmt(v))


def start_end(wire):
    return tuple(tuple(float(wire.get(c + i)) for c in "xy") for i in "12")


def add_hole(parent, p, size):
    ET.SubElement(parent, "hole", x=fmt(p[0]), y=fmt(p[1]), drill=fmt(size))


def make(raw, case):
    donor, kind, *params = case.split("-")
    sch, brd, net, contacts, nets, sch_digest, brd_digest = SOURCES[donor]
    original = (raw / brd).read_bytes()
    assert sha(original) == brd_digest and sha((raw / sch).read_bytes()) == sch_digest
    root = ET.fromstring(original)
    board = root.find(".//board")
    signal = board.find(f'./signals/signal[@name="{net}"]')
    wires = signal.findall("wire")
    bores = []
    if kind == "bore":
        d = int(params[0]) / 1000
        add_hole(board.find("plain"), C, d)
        bores.append((C, d / 2))
    elif kind == "footprint":
        d = int(params[0]) / 1000
        element = board.find('./elements/element[@name="RN3"]')
        packages = board.find(f'./libraries/library[@name="{element.get("library")}"]/packages')
        package = copy.deepcopy(packages.find(f'package[@name="{element.get("package")}"]'))
        package.set("name", "CAY16-CONTROL")
        element.set("package", package.get("name"))
        packages.append(package)
        add_hole(package, (2.4373, -1.4889), d)
        bores.append((C, d / 2))
    elif kind == "caps":
        gap = int(params[0]) / 1000
        a, b = start_end(wires[1])
        set_end(wires[1], 1, b)
        set_end(wires[1], 2, (C[0] - gap / 2, C[1]))
        new = copy.deepcopy(wires[1])
        set_end(new, 1, (C[0] + gap / 2, C[1]))
        set_end(new, 2, a)
        signal.append(new)
    elif kind == "pad":
        overlap = int("-".join(params)) / 1000
        # RN3.4 lower edge lies at 29.4872 on its symmetry axis after R270.
        set_end(wires[0], 1, (13.2188, 29.4872 - .127 + overlap))
    elif kind == "detour":
        add_hole(board.find("plain"), C, .6)
        offset = (C[0] - .45, C[1] + .45)
        bores.append((C, .3))
        if params[0] == "cut":
            add_hole(board.find("plain"), offset, .6)
            bores.append((offset, .3))
        corners = [(C[0]-.48, C[1]-.48), (offset[0]-.48, offset[1]-.48),
                   (offset[0]+.48, offset[1]+.48), (C[0]+.48, C[1]+.48)]
        for a, b in zip(corners, corners[1:]):
            w = copy.deepcopy(wires[1])
            set_end(w, 1, a)
            set_end(w, 2, b)
            signal.append(w)
    elif kind == "ring":
        penetration = int("-".join(params)) / 1000
        signal.find("via").set("diameter", "1.0")
        signal.remove(wires[1])
        set_end(wires[0], 2, (VIA[0], VIA[1] + .5 + .0889 - penetration))
    elif kind != "original":
        raise ValueError(kind)
    data = original if kind == "original" else ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return data, signal, bores


def point_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0])*dx + (p[1] - a[1])*dy)/(dx*dx+dy*dy) if dx or dy else 0
    t = min(1., max(0., t))
    return math.dist(p, (a[0] + t*dx, a[1] + t*dy))


def xy_mm(xy):
    assert len(xy) == 2 and all(math.isfinite(q) for q in xy)
    return tuple(q/1e6 for q in xy)


def check_witness(case, witness, signal, bores):
    steps = witness["steps"]
    terminals = {("pad", v.get("element"), v.get("pad")) for v in signal.findall("contactref")}
    assert {tuple(witness["start"]), tuple(witness["end"])} == terminals
    assert steps and steps[0]["start"] == witness["start"] and steps[-1]["end"] == witness["end"]
    assert all(a["end"] == b["start"] for a, b in zip(steps, steps[1:]))
    sources = [tuple(s["source"]) for s in steps]
    asserted_barrels = 0
    overlaps = []
    for i, step in enumerate(steps):
        src = tuple(step["source"])
        mat = step.get("material")
        if not mat:
            # The alternate trace has a genuine clear centerline and may use the maintained proof.
            assert case.endswith("original") or case == "due-detour-open", (case, src, "missing finite material witness")
            if case == "due-detour-open" and src[0] == "wire":
                a, b = (xy_mm(step[k][2:]) for k in ("start","end"))
                wire = signal.findall("wire")[src[1]]
                assert all(point_segment(v,*start_end(wire)) < EPS for v in (a,b))
                assert all(point_segment(center,a,b) > radius for center,radius in bores)
            continue
        if mat["kind"] == "plated_barrel":
            assert src[0] in ("via", "terminal") and src == tuple(mat["source"])
            assert {step["start"][1], step["end"][1]} == {1,16}
            center, wall = map(xy_mm, (mat["drill_center_nm"], mat["barrel_wall_point_nm"]))
            assert abs(math.dist(center,wall)-mat["drill_radius_nm"]/1e6) < EPS
            assert mat["clear_collar_outer_radius_nm"] > mat["drill_radius_nm"]
            if src == ("via",0):
                asserted_barrels += 1
            continue
        assert mat["kind"] == "component" and mat["in_plane_clearance_nm"] > 0
        poly = list(map(xy_mm, mat["path_nm"]))
        assert poly
        for key, point in (("start", poly[0]), ("end", poly[-1])):
            node = step[key]
            if node[0] == "copper":
                assert mat["layer"] == node[1] and math.dist(point, xy_mm(node[2:])) < EPS
        if src[0] == "wire":
            wire = signal.findall("wire")[src[1]]
            a,b = start_end(wire)
            radius = float(wire.get("width"))/2
            assert int(wire.get("layer")) == mat["layer"]
            # The capsule is convex, so certified vertices also bound the connecting segment.
            assert all(point_segment(p,a,b) < radius for p in poly), (case,src,"outside wire")
        for c,r in bores:
            for p,q in zip(poly, poly[1:] or poly):
                assert point_segment(c,p,q) > r, (case,src,"polyline intersects explicit drilled void")
        if "exit_overlap" in mat:
            disk = mat["exit_overlap"]
            radius, p = disk["radius_nm"]/1e6, xy_mm(disk["point_nm"])
            nxt = tuple(steps[i+1]["source"])
            assert disk["kind"] == "overlap" and radius > 0 and math.dist(p,poly[-1]) < EPS
            for owner in (src,nxt):
                if owner[0] == "wire":
                    w = signal.findall("wire")[owner[1]]
                    assert point_segment(p,*start_end(w)) + radius <= float(w.get("width"))/2 + EPS
                if owner == ("via",0) and case.startswith("feather-ring"):
                    dist = math.dist(p,VIA)
                    assert dist+radius <= .5+EPS and dist-radius >= .2-EPS
                if owner == ("terminal","RN3","4") and case.startswith("due-pad"):
                    assert abs(p[1]-29.7372)+radius <= .25+EPS
                    assert abs(p[0]-13.2188)+radius <= .325+EPS
            overlaps.append(frozenset((src,nxt)))
    if case.startswith("due-caps"):
        assert frozenset((("wire",1),("wire",2))) in overlaps
    if case.startswith("due-pad"):
        assert frozenset((("terminal","RN3","4"),("wire",0))) in overlaps
    if case.startswith("feather-ring"):
        assert asserted_barrels == 1 and frozenset((("via",0),("wire",0))) in overlaps
    if case.startswith("due-detour"):
        assert {("wire",i) for i in (2,3,4)} <= set(sources)
    return dict(steps=len(steps), overlap_disks=len(overlaps), via_barrels=asserted_barrels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--frozen", type=Path)
    p.add_argument("--repo", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--case", action="append", choices=CASES)
    args = p.parse_args()
    reg = {}
    for name, expected in CASES.items():
        data, _, _ = make(args.raw,name)
        reg[name] = {"board_sha256":sha(data), "expected":"witnessed" if expected else "unresolved"}
    if args.freeze:
        with args.freeze.open("x") as out:
            json.dump(reg,out,indent=2,sort_keys=True)
            out.write("\n")
        return
    assert reg == json.loads(args.frozen.read_text())
    report = {}
    with tempfile.TemporaryDirectory() as td:
        for name in args.case or CASES:
            data, signal, bores = make(args.raw,name)
            sch, _, net, contacts, nets, *_ = SOURCES[name.split("-")[0]]
            board = Path(td) / (name+".brd")
            board.write_bytes(data)
            result = subprocess.run([sys.executable,str(args.repo/"engineering_toolkit.py"),"eagle-netlist",
                                     str(args.raw/sch),"--board",str(board),"--routed-net",net],
                                    text=True, capture_output=True, timeout=120)
            row = {**reg[name],"returncode":result.returncode}
            try:
                assert result.returncode == 0, result.stderr[-900:]
                doc = json.loads(result.stdout)
                flat, routed = doc["board_comparison"], doc["routed_continuity"]
                assert flat["board_sha256"] == routed["board_sha256"] == sha(data)
                assert flat["agreement"] and (flat["exact_contacts"],len(flat["exact_nets"])) == (contacts,nets)
                row["observed"] = routed["status"]
                assert routed["status"] == row["expected"], (routed["status"],row["expected"])
                if CASES[name]:
                    only, = routed["witnesses"]
                    row["witness"] = check_witness(name,only,signal,bores)
                    row["source_witness"] = only
                else:
                    assert not routed["witnesses"]
                row["passed"] = True
            except (AssertionError, KeyError, ValueError, IndexError) as exc:
                import traceback
                row.update(passed=False,error=repr(exc),location=traceback.format_exc(limit=2))
            report[name]=row
            args.out.write_text(json.dumps(report,indent=2)+"\n")
            print(name, row.get("observed"), row.get("passed"), row.get("error",""), flush=True)
    return 0 if all(row["passed"] for row in report.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
