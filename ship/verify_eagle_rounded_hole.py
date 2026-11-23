"""Replay finite copper beside a rounded internal EAGLE Dimension cutout."""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from eagle_board_domain_oracle import P, Q, rect, rounded, tube
from verify_eagle_board_domain import rectangle, run_cli, small_fixture, wire


A, B, C, D, E = [P(*v) for v in [
    (-.30, -.18), (.30, -.18), (.30, .06), (.18, .18), (-.30, .18)]]
CENTER, SHARP = P(.18, .06), P(.30, .18)
CUT = rounded([A, B, SHARP, E], {SHARP: (C, CENTER, D)})
# All tested witnesses lie in the central 1 mm of the unchanged 4 mm board,
# even after rotation. Its local inscribed square gives an independent bound.
CORE = rect(-1, -1, 1, 1)
MATERIAL = (CORE[0] + CUT[0], CUT[1], lambda p: CORE[2](p) and not CUT[2](p))


def turn(point, oblique, inverse=False):
    x, y = point
    if not oblique:
        return point
    s = -4 if inverse else 4
    return (3*x-s*y)/5, (s*x+3*y)/5


def decimal_point(point):
    values = tuple(str(float(v)) for v in point)
    assert P(*values) == point, "source coordinates must have exact finite decimals"
    return values


def control(args, label, location, expected, oblique, route, counts):
    def world(point):
        return decimal_point(turn(point, oblique))

    def local(point_nm):
        return turn(tuple(Q(v)/1_000_000 for v in point_nm), oblique, inverse=True)

    # Flip every inner wire and permute its records under the rotated course.
    edges = [(A,B,0), (B,C,0), (C,D,90), (D,E,0), (E,A,0)]
    if oblique:
        edges = [(b,a,-s) for a,b,s in edges[::2]+edges[1::2]]
    inner = [wire(world(a), world(b), curve=s) for a,b,s in edges]
    if route:
        ends = location
    else:
        ends = P(-.65,0), P(.65,0)
    pads = [(name,*world(p),1) for name,p in zip(("L","R"),ends)]
    board, schematic = small_fixture(pads, rectangle()+inner)
    for pad in board.findall(".//smd"):
        pad.set("dx", ".012")
        pad.set("dy", ".012")
    signals = board.find(".//signals")
    signal = signals.find("signal")
    if route:
        signal.append(wire(world(ends[0]), world(ends[1]), layer=1, width=".008"))
        query = ["--routed-net","A"]
    else:
        x, y = location
        reach = Q(".03")
        signal.append(wire(world((x-reach,y)), world((x+reach,y)), layer=1, width=".008"))
        other = ET.SubElement(signals,"signal",name="B")
        other.append(wire(world((x,y-reach)), world((x,y+reach)), layer=1, width=".008"))
        query = ["--contact-nets","A","B"]
    key = ("oblique-" if oblique else "axis-") + label
    sch = args.out/(key+".sch")
    ET.ElementTree(schematic).write(sch, encoding="utf-8")
    doc = run_cli(args,key,board,sch,query,(2,1))
    report = doc["routed_continuity" if route else "inter_net_contact"]
    info = report["board_domain"]
    assert info["status"] == "bounded" and info["hole_quarter_count"] == 1, key
    assert 0 < info["hole_arc_error_upper_bound_nm"] <= 1
    assert {tuple(v) for v in info["hole_sources"]} == {("plain","wire",i) for i in range(4,9)}
    witnesses = report["witnesses" if route else "contacts"]
    assert bool(witnesses) == expected, (key,report)

    def path_clear(p, q, radius):
        assert tube(MATERIAL,local(p),local(q),Q(radius)/1_000_000), (key,p,q,radius)

    for witness in witnesses:
        if not route:
            radius = witness["disk_radius_nm"]
            assert radius > 0
            path_clear(witness["center_nm"],witness["center_nm"],radius)
            counts["contact_disks"] += 1
            continue
        steps = witness["steps"]
        assert steps[0]["start"] == witness["start"] and steps[-1]["end"] == witness["end"]
        assert all(a["end"] == b["start"] for a,b in zip(steps,steps[1:]))
        for step in steps:
            part = step["material"]
            path = part["path_nm"]
            assert part["kind"] == "component" and part["in_plane_clearance_nm"] == 1 and path
            for a,b in zip(path, path[1:] or path):
                path_clear(a,b,1)
                counts["landings" if a == b else "path_segments"] += 1
            if "exit_overlap" in part:
                overlap = part["exit_overlap"]
                assert overlap["radius_nm"] > 0
                path_clear(overlap["point_nm"],overlap["point_nm"],overlap["radius_nm"])
                counts["route_disks"] += 1
    return dict(witnesses=len(witnesses), arc_error_nm=info["hole_arc_error_upper_bound_nm"],
                board_sha256=doc["board_comparison"]["board_sha256"])


def course(args):
    cases = [
        ("deep-removed",P(.22,.10),False), ("chord-sliver",P(.258,.138),False),
        ("square-sliver",P(.276,.156),True), ("outside",P(.34,.22),True),
        # The far corner of tangent's 8 um square is exactly the 3:4:5 circle point.
        ("tangent",P(.248,.152),False), ("straddle",P(.252,.156),True)]
    routes = [("skirt",(P(.246,.186),P(.312,.120)),True),
              ("cut",(P(.184,.212),P(.332,.064)),False)]
    # Both cut endpoints clear the circle and their connecting segment enters it.
    a,b = routes[1][1]
    assert tube(MATERIAL,a,a,F(1,1_000_000)) and tube(MATERIAL,b,b,F(1,1_000_000))
    assert not tube(MATERIAL,a,b,0)
    # Center of this disk is retained, but the rim enters the cut.
    p = P(.2526,.1568)
    assert tube(MATERIAL,p,p,0) and not tube(MATERIAL,p,p,Q(".002"))
    counts = Counter()
    results = {}
    for oblique in (False,True):
        for route,group in ((False,cases),(True,routes)):
            for label,position,expected in group:
                key = ("oblique-" if oblique else "axis-") + label
                results[key] = control(args,label,position,expected,oblique,route,counts)
    return dict(cases=results, verified=dict(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.out.mkdir(parents=True)
    report = course(args)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
