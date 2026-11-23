"""Replay the registered Arduino Due bridge and translated control through the normal CLI."""
import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET


BOARD_SHA = "fd7cc5f9a14623ff62a7555c826373fe8f842cca400f05ec4f7b4c5589e1780a"
SCH_SHA = "60f449078bea22dc298a0cad5b9733affb1214f6052f778e54babb08531e9734"
ENDS = dict(x1="34.766335779", y1="35.465539929", x2="37.648255673", y2="35.813338579")


def run(repo, raw):
    sch, board = raw / "due-0.sch", raw / "due-1.brd"
    assert sha256(board.read_bytes()).hexdigest() == BOARD_SHA
    assert sha256(sch.read_bytes()).hexdigest() == SCH_SHA
    report = {}
    with TemporaryDirectory() as temp:
        for case, shift in [("original", None), ("bridge", 0), ("remote", 10)]:
            path = board
            if shift is not None:
                dom = ET.parse(board)
                signal = dom.find(".//board/signals/signal[@name='D-']")
                assert len(signal.findall("wire")) == 47
                points = {axis: str(Decimal(value) + shift) for axis, value in ENDS.items()}
                ET.SubElement(signal, "wire", **points, width="0.1", layer="1")
                path = Path(temp) / f"{case}.brd"
                dom.write(path, encoding="utf-8")
            counts = {}
            for name in ("D+", "D-"):
                doc = json.loads(subprocess.check_output([
                    sys.executable, str(repo / "engineering_toolkit.py"), "eagle-netlist", str(sch),
                    "--board", str(path), "--routed-net", name, "--contact-nets", "D+", "D-"]))
                routed = doc["routed_continuity"]
                assert routed["signal"] == name and routed["status"] == "witnessed"
                counts[name] = len(routed["witnesses"])
                assert counts[name] == 4
                contact = doc["inter_net_contact"]
                assert contact["board_sha256"] == sha256(path.read_bytes()).hexdigest()
                assert contact["clearance"] == "not_certified"
                found = {}
                for row in contact["contacts"]:
                    left, right = row["sources"]
                    assert left["signal"] == "D+" and left["primitive"][0] == "wire"
                    assert right == {"signal": "D-", "primitive": ["wire", 47]}
                    assert row["layer"] == 1 and row["disk_radius_nm"] > 0
                    index = left["primitive"][1]
                    assert index not in found
                    found[index] = row
                if case == "bridge":
                    assert contact["status"] == "contact_witnessed" and set(found) == {38, 42, 53, 54}
                    assert .0324 < found[53]["overlap_lower_bound_mm2"] < .0325
                else:
                    assert contact["status"] == "unresolved" and not found
            report[case] = dict(board_sha256=contact["board_sha256"], same_net_paths=counts,
                                status=contact["status"], contacts=contact["contacts"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = run(args.repo, args.raw)
    (args.out / "inter-net.json").write_text(json.dumps(report, indent=2) + "\n")
    print({case: len(row["contacts"]) for case, row in report.items()})
