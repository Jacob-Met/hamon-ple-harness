"""Cold public-CLI checks for the original Arduino Due through vias and controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from xml.etree import ElementTree as ET

from verify_eagle_arc_copper import BOARD_SHA, SCH_SHA, PAIRS


def run(repo, schematic, board, net, pads=None):
    argv = [sys.executable, str(repo / "engineering_toolkit.py"), "eagle-netlist",
            str(schematic), "--board", str(board), "--routed-net", net]
    if pads:
        argv.extend(["--routed-pads", *pads])
    result = json.loads(subprocess.check_output(argv))
    return result["routed_continuity"]


def check_original(repo, sch, path, net):
    result = run(repo, sch, path, net)
    assert result["board_sha256"] == BOARD_SHA and result["status"] == "witnessed"
    assert len(result["terminals"]) == 5 and len(result["witnesses"]) == 4
    terminals = {tuple(t) for t in result["terminals"]}
    assert {tuple(w["end"]) for w in result["witnesses"]} == terminals - {tuple(result["terminals"][0])}
    rule, = result["material"]["rule_derived_vias"]
    assert rule["rules"] == {"rvViaOuter": "0.25", "rlMinViaOuter": "8mil", "rlMaxViaOuter": "20mil"}
    assert (rule["drill_nm"], rule["outer_diameter_lower_bound_nm"]) == (406_400, 812_800)
    transition, = [s["material"] for w in result["witnesses"] for s in w["steps"]
                   if s["material"]["kind"] == "plated_barrel"]
    assert transition["source"] == ["via", 0]
    path = transition["path"]
    assert [p[0] for p in path] == ["copper", "barrel_wall", "barrel_wall", "copper"]
    assert {path[0][1], path[-1][1]} == {1, 16}
    assert path[0][1] == path[1][1] and path[2][1] == path[3][1]
    center = transition["drill_center_nm"]
    assert math.isclose(math.dist(center, path[1][2:]), 203_200, abs_tol=.01)
    assert path[1][2:] == path[2][2:]
    assert 203_200 < math.dist(center, path[0][2:]) < transition["clear_collar_outer_radius_nm"] <= 406_400
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "raw", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    board, sch = args.raw / "due-1.brd", args.raw / "due-0.sch"
    assert hashlib.sha256(board.read_bytes()).hexdigest() == BOARD_SHA
    assert hashlib.sha256(sch.read_bytes()).hexdigest() == SCH_SHA
    rows = []
    for net, pin, usb in PAIRS:
        proof = check_original(args.repo, sch, board, net)
        (args.out / (net + "-full.json")).write_text(json.dumps(proof, indent=2))
        controls = {}
        for case in ("severed", "notched", "remote", "blind", "unplated", "missing", "ambiguous", "bad_units"):
            dom = ET.parse(board)
            layer = dom.find(".//board")
            signal = next(s for s in layer.findall("signals/signal") if s.get("name") == net)
            via, = signal.findall("via")
            x, y = float(via.get("x")), float(via.get("y"))
            if case in ("severed", "notched", "remote", "unplated"):
                if case == "unplated":
                    signal.remove(via)
                shift = {"severed": 0, "notched": .28, "remote": 10, "unplated": 0}[case]
                diameter = {"severed": "1.2", "notched": "0.18", "remote": "1.2", "unplated": "0.4064"}[case]
                ET.SubElement(layer.find("plain"), "hole", x=str(x + shift), y=str(y), drill=diameter)
            elif case == "blind":
                via.set("extent", "1-2")
            else:
                rules = layer.find("designrules")
                param = rules.find("param[@name='rlMinViaOuter']")
                if case == "missing":
                    rules.remove(param)
                elif case == "ambiguous":
                    ET.SubElement(rules, "param", name="rlMinViaOuter", value="2mil")
                else:
                    param.set("value", "8furlongs")
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "control.brd"
                dom.write(path, encoding="utf-8", xml_declaration=True)
                controls[case] = run(args.repo, sch, path, net)["status"]
                if case in ("missing", "blind"):
                    assert run(args.repo, sch, path, net, ("U1", pin, "USB1", usb))["status"] == "witnessed"
            assert controls[case] == ("witnessed" if case == "remote" else "unresolved"), (net, case)
        rows.append(dict(net=net, terminal_count=5, paths=4, controls=controls))
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
