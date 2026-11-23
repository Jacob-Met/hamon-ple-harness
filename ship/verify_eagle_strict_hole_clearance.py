"""Replay exact near-threshold source gaps through the ordinary EAGLE contact CLI."""
import argparse
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

from verify_eagle_board_domain import small_fixture, wire


def rectangle(origin, u, v, left, right, bottom, top):
    return [(origin[0]+u[0]*x+v[0]*y, origin[1]+u[1]*x+v[1]*y)
            for x, y in ((left, bottom), (right, bottom), (right, top), (left, top))]


def mm(point):
    return tuple(str(Decimal(x)/1000000) for x in point)


def replay(repo, out):
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for t in (0, 1000000, 100000000):
        for u, shifts in (((3, 4), ((1, 9), (0, 10), (3, 8))),
                          ((4, 3), ((9, 1), (10, 0), (8, 3)))):
            v = (-u[1], u[0])
            for delta in shifts:
                gap = Fraction(u[0]*delta[0] + u[1]*delta[1], 5)
                outside = rectangle((t,t), (1,0), (0,1), -2000000, 2000000, -2000000, 2000000)
                left = rectangle((t,t), u, v, -80000, 0, -70000, 70000)
                right = rectangle((t+delta[0],t+delta[1]), u, v, 0, 80000, -70000, 70000)
                dims = [wire(mm(a), mm(b)) for poly in (outside, left, right)
                        for a, b in zip(poly, poly[1:]+poly[:1])]
                board, schematic = small_fixture([(k,*mm((t+x,t)),1)
                                                  for k,x in (("L",-850000),("R",850000))], dims)
                signals = board.find(".//signals")
                other = ET.SubElement(signals, "signal", name="B")
                x = y = t + 1000000
                signals.find("signal").append(wire(mm((x-30000,y)), mm((x+30000,y)), layer=1, width=".008"))
                other.append(wire(mm((x,y-30000)), mm((x,y+30000)), layer=1, width=".008"))
                name = f"gap-{t}-{u[0]}-{gap.numerator}_{gap.denominator}"
                sch, brd = out/(name+".sch"), out/(name+".brd")
                ET.ElementTree(schematic).write(sch)
                ET.ElementTree(board).write(brd)
                command = [sys.executable, str(repo/"engineering_toolkit.py"), "eagle-netlist",
                           str(sch), "--board", str(brd), "--contact-nets", "A", "B"]
                doc = json.loads(subprocess.check_output(command))
                (out/(name+".json")).write_text(json.dumps(doc, indent=2)+"\n")
                compare = doc["board_comparison"]
                assert compare["agreement"] and compare["exact_contacts"] == 2, name
                report = doc["inter_net_contact"]
                info = report["board_domain"]
                expected = gap > 8
                assert info["status"] == ("bounded" if expected else "unresolved"), (name, info)
                assert bool(report["contacts"]) == expected, name
                results.append(dict(case=name, source_gap_nm=str(gap), status=info["status"],
                                    contacts=len(report["contacts"]), board_sha256=compare["board_sha256"]))
    (out/"summary.json").write_text(json.dumps(results, indent=2)+"\n")
    print(f"{len(results)} drawings, {sum(bool(r['contacts']) for r in results)} remote crossings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    replay(args.repo.resolve(), args.out.resolve())
