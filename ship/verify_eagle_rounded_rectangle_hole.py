"""Replay known corner controls and manufacturer traces against a four-quarter cutout."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from eagle_board_domain_oracle import P, Q, authentic_domains, rect, rounded, tube
from verify_eagle_board_domain import COUNTS, inspect_walk, run_cli, small_fixture, wire
from verify_eagle_finite_copper import SOURCES
from verify_eagle_rounded_hole import decimal_point, turn


def drawing(center=P(0, 0), scale=Q(1)):
    """Return the eight authored edges and an independent four-circle oracle."""
    cx, cy = center

    def at(x, y):
        return cx + scale * x, cy + scale * y

    vertices = [at(2, -2), at(2, 2), at(-2, 2), at(-2, -2)]
    corners = [
        (at(1, -2), at(1, -1), at(2, -1)),
        (at(2, 1), at(1, 1), at(1, 2)),
        (at(-1, 2), at(-1, 1), at(-2, 1)),
        (at(-2, -1), at(-1, -1), at(-1, -2)),
    ]
    edges = []
    for i, (entry, _, exit_) in enumerate(corners):
        edges.extend([(corners[i - 1][2], entry, 0), (entry, exit_, 90)])
    return edges, rounded(vertices, dict(zip(vertices, corners)))


def combined(outer, hole):
    return outer[0] + hole[0], outer[1] + hole[1], lambda p: outer[2](p) and not hole[2](p)


def synthetic(args):
    edges, hole = drawing()
    material = combined(rect(-4, -4, 4, 4), hole)
    controls = {}
    for oblique in (False, True):
        # Reflect first, then apply an exact 3:4:5 rotation, then scramble the wires.
        def world(p):
            return turn((-p[0], p[1]), True) if oblique else p

        def local(p):
            x, y = turn(tuple(Q(v) / 1_000_000 for v in p), oblique, inverse=True)
            return (-x, y) if oblique else (x, y)

        def edge(a, b, s=0):
            return wire(decimal_point(world(a)), decimal_point(world(b)), curve=-s if oblique else s)

        vs = [P(-4, -4), P(4, -4), P(4, 4), P(-4, 4)]
        outer = [edge(a, b) for a, b in zip(vs, vs[1:] + vs[:1])]
        hole_edges = (
            [edge(b, a, -s) for a, b, s in edges[::2] + edges[1::2]]
            if oblique
            else [edge(*r) for r in edges]
        )
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:

            def mirror(p):
                return sx * p[0], sy * p[1]

            points = [
                (P(1.8, 1.8), Q(".05"), "clear"),
                (P(1.6, 1.6), Q(".025"), "inside"),
                (P(1.72, 1.72), Q(".025"), "straddling"),
            ]
            for point, radius, label in points:
                p = mirror(point)
                # An exact circle test on this corner independently fixes each disk class.
                d2 = (point[0] - 1) ** 2 + (point[1] - 1) ** 2
                assert (d2 > (1 + radius) ** 2) == (label == "clear")
                assert (d2 < (1 - radius) ** 2) == (label == "inside")
                assert tube(material, p, p, radius) == (label == "clear")
            cases = [
                ("clear-contact", P(1.8, 1.8), True),
                ("inside-contact", P(1.6, 1.6), False),
                ("clear-trace", (P(1.65, 1.95), P(1.95, 1.65)), True),
                ("clipped-trace", (P(1.35, 2.03), P(2.03, 1.35)), False),
            ]
            for label, where, expect in cases:
                route = label.endswith("trace")
                ends = [mirror(p) for p in where] if route else [P(-3, 0), P(3, 0)]
                pads = [(name, *decimal_point(world(p)), 1) for name, p in zip(("L", "R"), ends)]
                doc, sch = small_fixture(pads, outer + hole_edges)
                for pad in doc.findall(".//smd"):
                    pad.attrib.update(dx=".02", dy=".02")
                signals = doc.find(".//signals")
                signal = signals.find("signal")
                if route:
                    for p in ends:
                        assert tube(material, p, p, Q(".006"))
                    assert tube(material, *ends, Q(".006")) == expect
                    signal.append(wire(*[decimal_point(world(p)) for p in ends], layer=1, width=".012"))
                    query = ["--routed-net", "A"]
                else:
                    x, y = mirror(where)
                    for target, dx, dy in [
                        (signal, Q(".05"), 0),
                        (ET.SubElement(signals, "signal", name="B"), 0, Q(".05")),
                    ]:
                        target.append(
                            wire(
                                decimal_point(world((x - dx, y - dy))),
                                decimal_point(world((x + dx, y + dy))),
                                layer=1,
                                width=".012",
                            )
                        )
                    query = ["--contact-nets", "A", "B"]
                name = f"corner-{sx}-{sy}-{int(oblique)}-{label}"
                schfile = args.out / (name + ".sch")
                ET.ElementTree(sch).write(schfile, encoding="utf-8")
                data = run_cli(args, name, doc, schfile, query, (2, 1))
                report = data["routed_continuity" if route else "inter_net_contact"]
                witnesses = report["witnesses" if route else "contacts"]
                status = report["board_domain"]
                if args.baseline:
                    assert status["status"] == "unresolved" and not witnesses
                else:
                    assert (
                        status["hole_quarter_count"] == 4
                        and len(set(map(tuple, status["hole_quarter_sources"]))) == 4
                    )
                    assert status["hole_arc_error_upper_bound_nm"] <= 1 and bool(witnesses) == expect, (
                        name,
                        report,
                    )
                    for witness in witnesses:
                        if route:
                            steps = witness["steps"]
                            assert all(a["end"] == b["start"] for a, b in zip(steps, steps[1:]))
                            for step in steps:
                                piece = step["material"]
                                path = piece["path_nm"]
                                for a, b in zip(path, path[1:] or path):
                                    assert tube(material, local(a), local(b), Q("0.000001"))
                                    COUNTS["corner_path_segments"] += 1
                                if "exit_overlap" in piece:
                                    port = piece["exit_overlap"]
                                    p = local(port["point_nm"])
                                    assert tube(material, p, p, Q(port["radius_nm"]) / 1_000_000)
                                    COUNTS["corner_route_disks"] += 1
                        else:
                            p = local(witness["center_nm"])
                            r = Q(witness["disk_radius_nm"]) / 1_000_000
                            assert r > 0 and tube(material, p, p, r)
                            COUNTS["corner_contact_disks"] += 1
                controls[name] = len(witnesses)
    return controls


def manufacturer(args):
    results = {}
    for donor, title, pivot, remote in [
        ("due", "Due", P("12.4049", "28.4999"), P(34, 28)),
        ("feather", "Feather", P("32.4485", "6.731"), P(22, 12)),
    ]:
        sch, brd, net, nc, nn, sch_hash, brd_hash = SOURCES[donor]
        assert sha256((args.raw / sch).read_bytes()).hexdigest() == sch_hash
        assert sha256((args.raw / brd).read_bytes()).hexdigest() == brd_hash
        original = authentic_domains()[title]
        for position, center, expect in [
            ("original", None, True),
            ("clear", remote, True),
            ("clipped", pivot, False),
        ]:
            root = ET.parse(args.raw / brd).getroot()
            material = original
            if center:
                edges, hole = drawing(center, Q(".2"))
                root.find("./drawing/board/plain").extend(
                    wire(decimal_point(a), decimal_point(b), curve=s) for a, b, s in edges
                )
                material = combined(original, hole)
            name = f"{donor}-{position}"
            data = run_cli(args, name, root, args.raw / sch, ["--routed-net", net], (nc, nn))
            report = data["routed_continuity"]
            ws = report["witnesses"]
            if args.baseline and center:
                assert report["board_domain"]["status"] == "unresolved" and not ws
            else:
                assert report["board_domain"]["status"] == "bounded" and bool(ws) == expect
                if center:
                    assert report["board_domain"]["hole_quarter_count"] == 4
                for w in ws:
                    inspect_walk(w, material, feather=donor == "feather")
            results[name] = len(ws)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--raw", type=Path)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    output = {"corner_controls": synthetic(args)}
    if args.raw:
        output["manufacturer"] = manufacturer(args)
    output["verified"] = dict(COUNTS)
    (args.out / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["verified"]))
