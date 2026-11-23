"""Replay physically identical rounded-hole queries after exact tangent-wire subdivision."""
import argparse
from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET


def endpoint(wire, end):
    return tuple(Decimal(wire.get(axis + str(end))) for axis in ("x", "y"))


def split_bridge(plain, pieces):
    arcs = [w for w in plain.findall("wire") if Decimal(w.get("curve", "0"))]
    endpoints = {endpoint(w, end) for w in arcs for end in (1, 2)}
    bridges = [w for w in plain.findall("wire") if not Decimal(w.get("curve", "0"))
               and endpoint(w, 1) in endpoints and endpoint(w, 2) in endpoints]
    assert len(arcs) == 4 and len(bridges) == 4
    original = bridges[0]
    index, a, b = list(plain).index(original), endpoint(original, 1), endpoint(original, 2)
    plain.remove(original)
    # The exact Decimal interpolation terminates for the registered 1, 2 and 4 cases.
    for j in range(pieces):
        edge = deepcopy(original)
        for k, axis in enumerate(("x","y")):
            edge.set(axis+"1", str(a[k] + (b[k]-a[k])*j/pieces))
            edge.set(axis+"2", str(a[k] + (b[k]-a[k])*(j+1)/pieces))
        plain.insert(index+j, edge)


def replay(repo, drawings, output):
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for frame in (0, 1):
        for control in ("clear-contact","inside-contact","clear-trace","clipped-trace"):
            key = f"corner-1-1-{frame}-{control}"
            sch = drawings/(key+".sch")
            routed = control.endswith("trace")
            query = ["--routed-net","A"] if routed else ["--contact-nets","A","B"]
            for pieces in (1, 2, 4):
                tree = ET.parse(drawings/(key+".brd"))
                plain = tree.getroot().find("./drawing/board/plain")
                split_bridge(plain, pieces)
                prefix = f"{key}-{pieces}"
                board = output/(prefix+".brd")
                tree.write(board)
                args = [sys.executable, str(repo/"engineering_toolkit.py"), "eagle-netlist",
                        str(sch), "--board", str(board), *query]
                doc = json.loads(subprocess.check_output(args))
                (output/(prefix+".json")).write_text(json.dumps(doc, indent=2)+"\n")
                comparison = doc["board_comparison"]
                assert comparison["agreement"] and comparison["board_sha256"] == sha256(board.read_bytes()).hexdigest()
                received = doc["routed_continuity" if routed else "inter_net_contact"]
                domain = received["board_domain"]
                witnesses = received["witnesses" if routed else "contacts"]
                assert domain["status"] == "bounded" and bool(witnesses) == control.startswith("clear"), prefix
                assert domain["hole_wire_count"] == 7+pieces and domain["wire_count"] == 11+pieces, prefix
                assert len(set(map(tuple, domain["hole_sources"]))) == 7+pieces
                assert len(set(map(tuple, domain["hole_quarter_sources"]))) == 4
                report.append(dict(case=prefix, witnesses=len(witnesses), hole_wires=domain["hole_wire_count"],
                                   board_sha256=comparison["board_sha256"]))
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(len(report), "registered queries,", sum(bool(r["witnesses"]) for r in report), "positive witnesses")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--drawings", required=True, type=Path,
                        help="original output directory from verify_eagle_rounded_rectangle_hole.py")
    parser.add_argument("--out", required=True, type=Path)
    args=parser.parse_args()
    replay(args.repo.resolve(), args.drawings.resolve(), args.out.resolve())
