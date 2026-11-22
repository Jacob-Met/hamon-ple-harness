"""Wire connectivity evidence for two-color EAGLE vector schematics."""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

import hamon_drawing as drawing


class Groups:
    def __init__(self, size):
        self.parent = list(range(size))

    def root(self, index):
        while index != self.parent[index]:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, a, b):
        self.parent[self.root(a)] = self.root(b)


class Points:
    """Spatial index with an exact Euclidean check after bucket selection."""
    def __init__(self, radius):
        self.radius = radius
        self.buckets = defaultdict(list)

    def cell(self, point):
        return tuple(math.floor(x / self.radius) for x in point)

    def add(self, point, value):
        self.buckets[self.cell(point)].append((point, value))

    def near(self, point):
        x, y = self.cell(point)
        for u in range(x - 1, x + 2):
            for v in range(y - 1, y + 2):
                for other, value in self.buckets.get((u, v), ()):
                    if math.dist(point, other) <= self.radius:
                        yield value


def _color(value):
    match = re.fullmatch(r"rgb\(([0-9.]+)%,\s*([0-9.]+)%,\s*([0-9.]+)%\)", value or "")
    if match:
        red, green, blue = map(float, match.groups())
        if green > 12 and max(red, blue) < 1:
            return "green"
        if red > 12 and max(green, blue) < 1:
            return "red"
    return ""


def _ink(svg):
    wires, symbol_lines = [], []
    unsupported = Counter()
    fills = 0
    stack = [(ElementTree.fromstring(svg), drawing._ID, "none", "black", True)]
    while stack:
        element, matrix, stroke, fill, visible = stack.pop()
        tag = drawing.local(element.tag)
        if tag in ("defs", "clipPath", "symbol"):
            continue
        transform = drawing.parse_transform(element.get("transform"))
        matrix = drawing.mmul(matrix, transform) if matrix is not None and transform is not None else None
        stroke = element.get("stroke", stroke)
        fill = element.get("fill", fill)
        visible = visible and all(element.get(k, "1") == "1" for k in ("opacity", "stroke-opacity"))
        stack.extend((child, matrix, stroke, fill, visible) for child in reversed(list(element)))
        color = _color(stroke)
        if not color or tag not in ("path", "line", "circle", "ellipse", "polygon", "polyline"):
            continue
        if not visible or matrix is None:
            unsupported[color + "_appearance_or_transform"] += 1
            continue
        if tag == "path" and fill == "none":
            points = drawing._path_points(element.get("d", ""))
            if points and len(points) == 2:
                ends = [tuple(round(x, 4) for x in drawing.transform_point(matrix, p)) for p in points]
                if color == "red":
                    symbol_lines.append(ends)
                elif ends[0] != ends[1]:
                    wires.append(ends)
                else:
                    unsupported["degenerate_green"] += 1
            elif color == "green":
                unsupported["green_nonsegment"] += 1
        elif color == "green" and tag == "path" and drawing._svg_curve_bbox(element.get("d", ""), matrix):
            fills += 1
        elif color == "green":
            unsupported["green_unknown_shape"] += 1
    return wires, symbol_lines, fills, dict(unsupported)


def _label_anchors(wires, roots, words):
    anchors = defaultdict(set)
    horizontal = [(roots[i], min(a[0], b[0]), max(a[0], b[0]), a[1])
                  for i, (a, b) in enumerate(wires) if abs(a[1] - b[1]) < .08]
    for word in words:
        name = word["text"]
        if not re.fullmatch(r"[+A-Z][A-Z0-9_+/\-]{1,16}", name):
            continue
        left, top, right, bottom = word["page_bbox_pt"]
        height = bottom - top
        hits = {root for root, start, end, y in horizontal
                if -.22 * height < y - bottom < .07 * height
                and start - .35 * height <= left <= end - .35 * height
                and min(right, end) - max(left, start) > .55 * min(right - left, end - start)}
        if len(hits) == 1:
            anchors[name].update(hits)
    return anchors


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


_GROUND_NAME = re.compile(r"(?:[AVDS]?GND|VSS)", re.I)
_POWER_NAME = re.compile(r"(?:[+\-][A-Z0-9.]+|VCC|VDD|VIN|VBUS)", re.I)


def _supply_anchors(wires, roots, lines, words):
    endpoints, ends, centers = Points(.13), Points(.13), Points(.13)
    for i, (a, b) in enumerate(wires):
        endpoints.add(a, roots[i])
        endpoints.add(b, roots[i])
    for a, b in lines:
        ends.add(a, b)
        ends.add(b, a)
        centers.add(_midpoint(a, b), a)
    supply_words = []
    for word in words:
        name = word["text"]
        ground = bool(_GROUND_NAME.fullmatch(name))
        power = bool(_POWER_NAME.fullmatch(name))
        if ground or power:
            box = word["page_bbox_pt"]
            supply_words.append((name, ground, _midpoint(box[:2], box[2:])))
    supplies, marks = defaultdict(set), []
    for a, b in lines:
        for foot, tip in ((a, b), (b, a)):
            length = math.dist(foot, tip)
            touched = set(endpoints.near(foot))
            if length < 1 or len(touched) != 1:
                continue
            ux, uy = (tip[0] - foot[0]) / length, (tip[1] - foot[1]) / length

            def project(p):
                dx, dy = (p[0] - tip[0]) / length, (p[1] - tip[1]) / length
                return dx * ux + dy * uy, dy * ux - dx * uy

            transverse = any(abs(u) < .04 and .64 < abs(v) < .86
                             for u, v in map(project, centers.near(tip)))
            wings = [v for u, v in map(project, ends.near(tip))
                     if -.84 < u < -.66 and .42 < abs(v) < .58]
            arrow = any(v < 0 for v in wings) and any(v > 0 for v in wings)
            if transverse == arrow:
                continue
            names = {name for name, ground, center in supply_words if ground == transverse
                     and .18 < project(center)[0] < 1.3 and abs(project(center)[1]) < .65}
            if len(names) == 1:
                name, root = next(iter(names)), next(iter(touched))
                supplies[name].add(root)
                marks.append({"name": name, "wire_index": root, "foot": foot, "tip": tip,
                              "shape": "ground_bar" if transverse else "supply_arrow"})
    return supplies, marks


def _graph(wires, lines, words):
    groups, endpoints = Groups(len(wires)), Points(.12)
    for i, (a, b) in enumerate(wires):
        for p in (a, b):
            for j in endpoints.near(p):
                groups.join(i, j)
            endpoints.add(p, i)
    roots = [groups.root(i) for i in range(len(wires))]
    labels = _label_anchors(wires, roots, words)
    supplies, marks = _supply_anchors(wires, roots, lines, words)
    # A printed supply word alone can land on a different wire. The full
    # glyph and its stem take precedence everywhere the symbol is observed.
    anchors = {name: hits for name, hits in labels.items() if name not in supplies}
    anchors.update(supplies)
    for hits in anchors.values():
        first = next(iter(hits))
        for other in hits:
            groups.join(first, other)
    group_ids = {}
    membership = []
    for i in range(len(wires)):
        root = groups.root(i)
        group_ids.setdefault(root, len(group_ids))
        membership.append(group_ids[root])
    return membership, marks, {k: sorted(v) for k, v in anchors.items()}


def page_connectivity(filename, page=1):
    """Return visible page-local candidate wire groups with geometric evidence."""
    source = Path(filename)
    stock = drawing.scan_pdf(source, page, page, limit_paths=1, limit_words=5000)["pages"][0]
    with tempfile.TemporaryDirectory(prefix="hamon-eagle-") as directory:
        wires, red, fills, unsupported = _ink(drawing._page_svg(source, page, directory))
    if len(wires) > 2000 or len(red) > 5000 or len(stock["words"]) >= 5000:
        raise ValueError("EAGLE schematic exceeds the registered page processing bounds")
    result = {"kind": "eagle-schematic-page-connectivity", "page": page,
              "pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "scope": "PDF-only, two-color EAGLE page-local wire groups; device pins and inter-sheet nets are outside the result.",
              "svg_coordinate_frame": stock["svg_coordinate_frame"], "wires": wires,
              "green_filled_marks": fills, "unsupported_ink": unsupported}
    metadata = subprocess.check_output(["pdfinfo", str(source)], text=True, timeout=20)
    match = re.search(r"(?m)^Creator:\s*(.*)$", metadata)
    creator = match.group(1).strip() if match else ""
    result["pdf_creator"] = creator
    if not re.search(r"\bEAGLE\b", creator, re.I) or not wires or not red or unsupported or stock["embedded_images"]:
        return {**result, "status": "unresolved_profile", "wire_groups": None}
    membership, marks, anchors = _graph(wires, red, stock["words"])
    return {**result, "status": "candidate_evidence", "wire_groups": membership,
            "group_count": len(set(membership)), "supply_glyphs": marks,
            "printed_equivalence_anchors": anchors}


def _ground_glyph_precedence(name, components, glyphs, labels, occurrences):
    """Allow incidental text on a witnessed ground island only when it cannot identify another island."""
    if not _GROUND_NAME.fullmatch(name):
        return False
    for component in components:
        marks = glyphs[component]
        if set(marks) != {(name, "ground_bar")}:
            return False
        roots = marks[name, "ground_bar"]
        if len(roots) < 2:
            return False
        for other, anchors in labels[component].items():
            if other == name:
                continue
            if (_GROUND_NAME.fullmatch(other) or _POWER_NAME.fullmatch(other)
                    or len(occurrences[other]) != 1 or not anchors <= roots):
                return False
    return True


def _stitch_pages(pages):
    """Map sheet-local components onto flat-design nets using witnessed exact names."""
    by_name = defaultdict(set)
    by_component = defaultdict(lambda: defaultdict(set))
    glyph_evidence = defaultdict(lambda: defaultdict(set))
    occurrences = defaultdict(set)
    for page in pages:
        membership = page["wire_groups"]
        number = page["page"]
        glyphs = {(mark["name"], membership[mark["wire_index"]])
                  for mark in page["supply_glyphs"]}
        for mark in page["supply_glyphs"]:
            root = mark["wire_index"]
            component = (number, membership[root])
            glyph_evidence[component][mark["name"], mark["shape"]].add(root)
        for name, anchors in page["printed_equivalence_anchors"].items():
            for index in anchors:
                group = membership[index]
                component = (number, group)
                by_component[component][name].add(index)
                occurrences[name].add((number, index))
                # A printed supply word can lie on a different wire. Require
                # physical glyph evidence on each sheet for a supply join.
                supply_name = _GROUND_NAME.fullmatch(name) or _POWER_NAME.fullmatch(name)
                if supply_name and (name, group) not in glyphs:
                    continue
                by_name[name].add(component)

    canonical = {}
    links = []
    for name, components in sorted(by_name.items()):
        if len(components) < 2 or len({page for page, _ in components}) != len(components):
            continue
        conflicting_text = any(set(by_component[component]) != {name} for component in components)
        if conflicting_text and not _ground_glyph_precedence(
                name, components, glyph_evidence, by_component, occurrences):
            continue
        canonical.update(dict.fromkeys(components, min(components)))
        link = {"name": name, "page_groups": sorted(components)}
        if conflicting_text:
            link["ground_bar_witnesses"] = [
                {"page": page, "group": group,
                 "wire_islands": sorted(glyph_evidence[page, group][name, "ground_bar"]),
                 "incidental_labels": sorted(set(by_component[page, group]) - {name})}
                for page, group in sorted(components)
            ]
        links.append(link)

    group_ids = {}
    memberships = []
    for page in pages:
        wire_groups = []
        for group in page["wire_groups"]:
            key = (page["page"], group)
            key = canonical.get(key, key)
            wire_groups.append(group_ids.setdefault(key, len(group_ids)))
        memberships.append(wire_groups)
    return memberships, links


def document_connectivity(filename):
    """Connect witnessed names across sheets of a single flat EAGLE design."""
    source = Path(filename)
    total, _ = drawing._page_geometry(source, 1, 1)
    if not 1 <= total <= 20:
        raise ValueError("EAGLE document exceeds the 20-sheet processing bound")
    pages = [page_connectivity(source, page) for page in range(1, total + 1)]
    result = {"kind": "eagle-schematic-document-connectivity", "pages": pages,
              "pdf_sha256": pages[0]["pdf_sha256"],
              "scope": "Candidate visible wire groups for one flat EAGLE design. Device pins and hierarchical names are outside the result."}
    if any(page["status"] != "candidate_evidence" for page in pages):
        return {**result, "status": "unresolved_profile", "document_wire_groups": None}
    memberships, links = _stitch_pages(pages)
    return {**result, "status": "candidate_evidence", "document_wire_groups": memberships,
            "inter_sheet_anchors": links, "group_count": len({g for sheet in memberships for g in sheet})}
