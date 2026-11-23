"""Check ordinary EAGLE queries for rigidly transformed copies of one quarter recess."""

import argparse
import json
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path
from xml.etree import ElementTree as ET

from verify_eagle_board_domain import small_fixture, wire


def fmt(value):
    text = f"{float(value):.9f}".rstrip("0").rstrip(".") or "0"
    assert F(text) == value
    return text


def fixture(site, rotation, mirror, reverse):
    a, b, d = rotation

    def transform(point):
        x, y = (F(str(v)) * F(13, 10) for v in point)
        x *= mirror
        return (
            (a * x - b * y) / d + (17 if mirror < 0 else 0),
            (b * x + a * y) / d - (23 if mirror < 0 else 0),
        )

    def p(point):
        return tuple(fmt(x) for x in transform(point))

    corners = [(0, 0), (8, 0), (8, 8), (5, 8), (5, 5), (4, 4), (0, 4), (-1, 2)]
    edges = [
        (p(u), p(v), -90 * mirror if i == 4 else 0)
        for i, (u, v) in enumerate(zip(corners, corners[1:] + corners[:1]))
    ]
    if reverse:
        edges = [(v, u, -sweep) for u, v, sweep in reversed(edges)]
    outline = [wire(u, v, curve=sweep) for u, v, sweep in edges]
    x, y = map(F, site)
    span = F(1, 20)
    ends = [p(u) for u in [(x - span, y), (x + span, y), (x, y - span), (x, y + span)]]
    root, sch = small_fixture(
        [(name, *loc, 1) for name, loc in zip(("L", "R"), ends)], outline
    )
    for package in (root.find(".//package"), sch.find(".//package")):
        for pad in package.findall("smd"):
            pad.tag = "pad"
            for attr in ("layer", "dx", "dy"):
                del pad.attrib[attr]
            pad.attrib.update(shape="round", diameter=".05", drill=".008")
    signals = root.find("./drawing/board/signals")
    signals.find("signal").append(wire(*ends[:2], layer=1, width=".020"))
    ET.SubElement(signals, "signal", name="B").append(
        wire(*ends[2:], layer=1, width=".020")
    )
    return root, sch, transform((4, 5))


def min_radius2(a, b):
    v = tuple(y - x for x, y in zip(a, b))
    length = sum(x * x for x in v)
    t = max(0, min(1, -sum(x * y for x, y in zip(a, v)) / length)) if length else 0
    return sum((x + t * y) ** 2 for x, y in zip(a, v))


def verify(result, retained, center, rotation, mirror):
    """Certify each serialized disk and full planar segment in the true curved patch."""
    contact = result["inter_net_contact"]
    route = result["routed_continuity"]
    assert result["board_comparison"]["agreement"]
    assert contact["board_domain"]["status"] == "bounded", contact.get("detail")
    assert route["board_domain"]["status"] == "bounded"
    assert bool(contact["contacts"]) == retained
    assert (route["status"] == "witnessed") == retained
    a, b, den = rotation
    center = tuple(x * 1_000_000 for x in center)
    nominal_radius = 1_300_000
    counts = {"disks": 0, "segments": 0, "collars": 0}

    def local(point):
        u, v = (F(str(x)) - y for x, y in zip(point, center))
        return (mirror * (a * u + b * v) / den, (-b * u + a * v) / den)

    def inside_square(point, radius):
        x, y = point
        assert min(x, nominal_radius - x, -y, nominal_radius + y) >= radius

    def certify_disk(point, radius):
        p = local(point)
        radius = F(str(radius))
        inside_square(p, radius)
        assert sum(x * x for x in p) >= (nominal_radius + radius) ** 2

    for disk in contact["contacts"]:
        certify_disk(disk["center_nm"], disk["disk_radius_nm"])
        assert disk["disk_radius_nm"] > 4000
        counts["disks"] += 1
    for witness in route["witnesses"]:
        assert witness["steps"]
        for step in witness["steps"]:
            material = step["material"]
            path = list(map(local, material["path_nm"]))
            assert path and material["in_plane_clearance_nm"] == 1
            for point in path:
                inside_square(point, 1)
            for u, v in zip(path, path[1:]):
                assert min_radius2(u, v) >= (nominal_radius + 1) ** 2
                counts["segments"] += 1
            if "exit_overlap" in material:
                disk = material["exit_overlap"]
                certify_disk(disk["point_nm"], disk["radius_nm"])
                counts["disks"] += 1
            if "terminal_collar" in material:
                collar = material["terminal_collar"]
                certify_disk(
                    collar["drill_center_nm"], collar["clear_collar_outer_radius_nm"]
                )
                counts["collars"] += 1
    if retained:
        assert counts["segments"] > 0 and counts["collars"] == 2
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--observe", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rot in [(1, 0, 1), (3, 4, 5), (5, 12, 13)]:
        for mirror, reverse in [(1, False), (-1, False), (1, True), (-1, True)]:
            for label, site in [
                ("retained", ("4.85", "4.15")),
                ("removed", ("4.6", "4.4")),
            ]:
                brd, sch, center = fixture(site, rot, mirror, reverse)
                name = f"{rot[2]}-{mirror}-{int(reverse)}-{label}"
                bp, sp = (args.out / (name + suffix) for suffix in (".brd", ".sch"))
                ET.ElementTree(brd).write(bp, encoding="utf-8")
                ET.ElementTree(sch).write(sp, encoding="utf-8")
                cmd = [
                    sys.executable,
                    str(args.repo / "engineering_toolkit.py"),
                    "eagle-netlist",
                    str(sp),
                    "--board",
                    str(bp),
                    "--routed-net",
                    "A",
                    "--contact-nets",
                    "A",
                    "B",
                ]
                result = json.loads(subprocess.check_output(cmd, text=True))
                (args.out / (name + ".json")).write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                counts = (
                    {}
                    if args.observe
                    else verify(result, label == "retained", center, rot, mirror)
                )
                c = result["inter_net_contact"]
                r = result["routed_continuity"]
                rows.append(
                    dict(
                        verified_geometry=counts,
                        case=name,
                        domain=c["board_domain"]["status"],
                        contact=c["status"],
                        route=r["status"],
                        reason=c.get("detail"),
                        radius=[d["disk_radius_nm"] for d in c["contacts"]],
                        membership=result["board_comparison"]["agreement"],
                    )
                )
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for den in [1, 5, 13]:
        group = [r for r in rows if int(r["case"].split("-")[0]) == den]
        print(
            den,
            "bounded",
            sum(r["domain"] == "bounded" for r in group),
            "contacts",
            sum(r["contact"] == "contact_witnessed" for r in group),
            "routes",
            sum(r["route"] == "witnessed" for r in group),
            sorted({r["reason"] or "" for r in group}),
        )


if __name__ == "__main__":
    main()
