"""Wire connectivity evidence for two-color EAGLE vector schematics."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
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


def _red_shape_bounds(tag, element, matrix):
    """Conservative page-space bounds for symbol paths outside the straight-stroke list."""
    if tag == "path":
        return drawing._svg_curve_bbox(element.get("d", ""), matrix)
    if tag in ("circle", "ellipse"):
        return drawing._svg_ellipse_bbox(tag, element.attrib, matrix)
    points = drawing._shape_points(tag, element.attrib)
    return drawing._bounds(points, matrix) if points else None


def _ink(svg, glyph_origins=None, symbol_obstacles=None):
    wires, symbol_lines = [], []
    unsupported = Counter()
    fills = 0
    stack = [(ElementTree.fromstring(svg), drawing._ID, "none", "black", True, True)]
    while stack:
        element, matrix, stroke, fill, visible, glyph_visible = stack.pop()
        tag = drawing.local(element.tag)
        if tag in ("defs", "clipPath", "symbol"):
            continue
        transform = drawing.parse_transform(element.get("transform"))
        matrix = drawing.mmul(matrix, transform) if matrix is not None and transform is not None else None
        stroke = element.get("stroke", stroke)
        fill = element.get("fill", fill)
        visible = visible and all(element.get(k, "1") == "1" for k in ("opacity", "stroke-opacity"))
        glyph_visible = glyph_visible and element.get("fill-opacity", "1") == "1"
        stack.extend((child, matrix, stroke, fill, visible, glyph_visible) for child in reversed(list(element)))
        if (tag == "use" and glyph_origins is not None and matrix is not None
                and visible and glyph_visible and fill != "none"):
            ref = element.get("{http://www.w3.org/1999/xlink}href", element.get("href", ""))
            if re.fullmatch(r"#(?:glyph-?\d+|symbol\d+)-\d+", ref):
                point = drawing.transform_point(matrix, (float(element.get("x", 0)),
                                                        float(element.get("y", 0))))
                glyph_origins.append((point, ref.rsplit("-", 1)[0]))
        if tag not in ("path", "line", "rect", "circle", "ellipse", "polygon", "polyline"):
            continue
        color = _color(stroke)
        red_ink = color == "red" or _color(fill) == "red"
        if not color and not (red_ink and symbol_obstacles is not None):
            continue
        points = drawing._path_points(element.get("d", "")) if tag == "path" and fill == "none" else None
        if symbol_obstacles is not None and red_ink and len(points or ()) != 2:
            extra_lines, bounds = symbol_obstacles
            vertices = (points if tag == "path" else drawing._shape_points(tag, element.attrib))
            pairs = (drawing._straight_segment_vertex_pairs(tag, element.attrib, vertices)
                     if fill == "none" and matrix is not None else None)
            if pairs is not None:
                page_points = [drawing.transform_point(matrix, v) for v in vertices]
                extra_lines.extend((page_points[i], page_points[j]) for i, j in pairs)
            else:
                bounds.append(_red_shape_bounds(tag, element, matrix))
        if not color or tag == "rect":
            continue
        if not visible or matrix is None:
            unsupported[color + "_appearance_or_transform"] += 1
            continue
        if tag == "path" and fill == "none":
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


def _word_directions(words, glyphs):
    """Orient native word boxes from contiguous Cairo glyph origins in paint order."""
    x_order = sorted((point[0], i) for i, (point, _) in enumerate(glyphs))
    xs = [x for x, _ in x_order]
    for word in words:
        name = word["text"]
        ordinary_name = bool(re.fullmatch(r"[+A-Z][A-Z0-9_+/\-]{1,16}", name))
        if not ordinary_name and not _POWER_NAME.fullmatch(name):
            continue
        left, top, right, bottom = word["page_bbox_pt"]
        # Poppler reports boxes and Cairo glyph origins to different decimal precision.
        tol = .01
        lo, hi = bisect_left(xs, left - tol), bisect_right(xs, right + tol)
        indices = sorted(i for _, i in x_order[lo:hi]
                         if top - tol <= glyphs[i][0][1] <= bottom + tol)
        if len(indices) != len(name) or indices[-1] - indices[0] + 1 != len(indices):
            continue
        run = [glyphs[i] for i in indices]
        if len({font for _, font in run}) != 1:
            continue
        steps = [(b[0][0] - a[0][0], b[0][1] - a[0][1]) for a, b in zip(run, run[1:])]
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            if all(dx * x + dy * y > tol and abs(dx * y - dy * x) <= tol for x, y in steps):
                prefix = "" if ordinary_name else "power_"
                word[prefix + "baseline_direction"] = (dx, dy)
                word[prefix + "baseline_origin"] = run[0][0]
                word[prefix + "glyph_span"] = (indices[0], indices[-1] + 1)
                break


def _label_frame(point, direction):
    x, y = point
    dx, dy = direction
    return (dx * x + dy * y, dx * y - dy * x)


def _label_anchors(wires, roots, words, *, word_indices=None, word_roots=None):
    anchors = defaultdict(set)
    parallel = defaultdict(list)
    for direction in {w["baseline_direction"] for w in words if "baseline_direction" in w}:
        for root, (a, b) in zip(roots, wires):
            a, b = _label_frame(a, direction), _label_frame(b, direction)
            if abs(a[1] - b[1]) < .08:
                parallel[direction].append((root, min(a[0], b[0]), max(a[0], b[0]), a[1]))
    for word_index, word in enumerate(words):
        direction = word.get("baseline_direction")
        if not direction:
            continue
        x, y, u, v = word["page_bbox_pt"]
        ax, ay = _label_frame((x, y), direction)
        bx, by = _label_frame((u, v), direction)
        left, right = sorted((ax, bx))
        top, bottom = sorted((ay, by))
        height = bottom - top
        hits = {root for root, start, end, y in parallel[direction]
                if -.22 * height < y - bottom < .07 * height
                and start - .35 * height <= left <= end - .35 * height
                and min(right, end) - max(left, start) > .55 * min(right - left, end - start)}
        if word_roots is not None:
            word_roots[word_index].update(hits)
        if len(hits) == 1:
            anchors[word["text"]].update(hits)
            if word_indices is not None:
                word_indices.add(word_index)
    return anchors


def _touches_segment(point, segment, tolerance):
    a, b = segment
    if any(p < min(q, r) - tolerance or p > max(q, r) + tolerance
           for p, q, r in zip(point, a, b)):
        return False
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if not length2:
        return math.dist(point, a) <= tolerance
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2
    t = min(1, max(0, t))
    return math.dist(point, (a[0] + t * dx, a[1] + t * dy)) <= tolerance


def _numbered_pin_terminals(lines, words, direction):
    """Index red pin stubs in closed boxes with one complete, numbered pin bank."""
    vertical = defaultdict(set)
    horizontal = defaultdict(set)
    for a, b in lines:
        (x, y), (u, v) = _label_frame(a, direction), _label_frame(b, direction)
        if y == v and x != u:
            horizontal[tuple(sorted((x, u)))].add(y)
        elif x == u and y != v:
            vertical[tuple(sorted((y, v)))].add(x)
    numbers = []
    for word in words:
        if not re.fullmatch("[1-9][0-9]{0,2}", word["text"]):
            continue
        x, y, u, v = word["page_bbox_pt"]
        (a, b), (c, d) = _label_frame((x, y), direction), _label_frame((u, v), direction)
        numbers.append((min(a, c), max(a, c), min(b, d), max(b, d), int(word["text"])))
    banks = defaultdict(list)
    for (left, right), edges in horizontal.items():
        if len(edges) < 2:
            continue
        for (top, bottom), sides in vertical.items():
            if top not in edges or bottom not in edges or left not in sides or right not in sides:
                continue
            for (start, stop), ys in horizontal.items():
                if not left < start < stop < right:
                    continue
                rows = sorted(y for y in ys if top < y < bottom)
                if len(rows) < 3:
                    continue
                pitch = (rows[-1] - rows[0]) / (len(rows) - 1)
                if pitch <= stop - start or any(abs(y - rows[0] - j * pitch) > .12
                                                for j, y in enumerate(rows)):
                    continue
                pins = []
                for y in rows:
                    ids = [n for x0, x1, y0, y1, n in numbers
                           if left < x0 <= start <= x1 < right and y0 <= y <= y1 + .12
                           and top < y0 < y1 < bottom and y1 - y0 < pitch]
                    if len(ids) != 1:
                        break
                    pins.append(ids[0])
                consecutive = list(range(1, len(rows) + 1))
                if pins != consecutive and pins != consecutive[::-1]:
                    continue
                for y, pin in zip(rows, pins):
                    banks[start, y].append((pin, len(rows), left))
    return banks


def _wire_end_labels(wires, roots, endpoints, lines, words, symbol_bounds=(), pin_lines=()):
    """Find collinear native text at an electrically free end of a single green wire."""
    # Only scan ink for endpoints actually overlapped by a qualifying text run.
    # A symbol stroke or a different wire can meet an endpoint at its interior.
    if any(bound is None for bound in symbol_bounds):
        return []
    free = {}

    def is_free(index, point):
        if point not in free:
            x, y = point
            free[point] = (len(set(endpoints.near(point))) == 1
                           and not any(x0 - .13 <= x <= x1 + .13 and y0 - .13 <= y <= y1 + .13
                                       for x0, y0, x1, y1 in symbol_bounds)
                           and not any(_touches_segment(point, line, .13) for line in lines)
                           and not any(i != index and _touches_segment(point, wire, .12)
                                       for i, wire in enumerate(wires)))
        return free[point]

    parallel = defaultdict(list)
    for direction in {w["baseline_direction"] for w in words if "baseline_origin" in w}:
        for index, (a, b) in enumerate(wires):
            start, y = _label_frame(a, direction)
            end, other_y = _label_frame(b, direction)
            if abs(y - other_y) <= .08:
                parallel[direction].append((index, min(start, end), max(start, end), y, other_y,
                                            ((start, a), (end, b))))
    candidates = []
    banks = {}
    for word_index, word in enumerate(words):
        if "baseline_origin" not in word:
            continue
        direction = word["baseline_direction"]
        x, y, u, v = word["page_bbox_pt"]
        ax, ay = _label_frame((x, y), direction)
        bx, by = _label_frame((u, v), direction)
        left, right = sorted((ax, bx))
        height = abs(by - ay)
        origin, baseline = _label_frame(word["baseline_origin"], direction)
        hits, witnesses = set(), []
        for index, start, end, y0, y1, loose in parallel[direction]:
            if min(right, end) - max(left, start) < .5 * (right - left):
                continue
            if max(abs(y0 - baseline), abs(y1 - baseline)) > .12:
                # A manually offset loose label must start at the physical end.
                # The opposite end must enter the numbered terminal in its box.
                if not (abs(origin - start) <= .12 and min(ay, by) <= y0 <= max(ay, by)):
                    continue
                if direction not in banks:
                    banks[direction] = _numbered_pin_terminals(pin_lines, words, direction)
                (_, free_end), (_, pin_end) = sorted(loose)
                matches = [pin for pin in banks[direction].get(_label_frame(pin_end, direction), ())
                           if right < pin[2]]
                if (len(matches) != 1 or not is_free(index, free_end)
                        or set(endpoints.near(pin_end)) != {index}
                        or any(i != index and _touches_segment(pin_end, wire, .12)
                               for i, wire in enumerate(wires))):
                    continue
                pin = matches[0]
                hits.add(roots[index])
                witnesses.append({"name": word["text"], "wire_index": index, "endpoint": free_end,
                                  "baseline_origin": word["baseline_origin"], "_word_index": word_index,
                                  "pin_bank_terminal": {"endpoint": pin_end, "pin": pin[0], "count": pin[1]}})
                continue
            hits.add(roots[index])
            for pos, point in loose:
                if left - .5 * height <= pos <= right and is_free(index, point):
                    witnesses.append({"name": word["text"], "wire_index": index, "endpoint": point,
                                      "baseline_origin": word["baseline_origin"],
                                      "_word_index": word_index})
        if len(hits) == 1 and witnesses:
            candidates.append((hits.pop(), witnesses[0]))
    return candidates


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


_GROUND_NAME = re.compile(r"(?:[AVDS]?GND|VSS)", re.I)
_POWER_NAME = re.compile(r"(?:[+\-][A-Z0-9.]+|VCC|VDD|VIN|VBUS)", re.I)


def _supply_anchors(wires, roots, lines, words, *, all_glyphs=None):
    endpoints, ends, centers = Points(.13), Points(.13), Points(.13)
    for i, (a, b) in enumerate(wires):
        endpoints.add(a, roots[i])
        endpoints.add(b, roots[i])
    for a, b in lines:
        ends.add(a, b)
        ends.add(b, a)
        centers.add(_midpoint(a, b), a)
    supply_words = []
    for word_index, word in enumerate(words):
        name = word["text"]
        ground = bool(_GROUND_NAME.fullmatch(name))
        power = bool(_POWER_NAME.fullmatch(name))
        if ground or power:
            box = word["page_bbox_pt"]
            supply_words.append((word_index, name, ground, _midpoint(box[:2], box[2:])))
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
            printed = [(index, name) for index, name, ground, center in supply_words
                       if ground == transverse and .18 < project(center)[0] < 1.3
                       and abs(project(center)[1]) < .65]
            names = {name for _, name in printed}
            if all_glyphs is not None:
                all_glyphs.append({"wire_index": next(iter(touched)), "foot": foot, "tip": tip,
                                   "shape": "ground_bar" if transverse else "supply_arrow",
                                   "ordinary_names": sorted(names),
                                   "ordinary_word_indices": [i for i, _ in printed]})
            if len(names) == 1:
                name, root = next(iter(names)), next(iter(touched))
                supplies[name].add(root)
                marks.append({"name": name, "wire_index": root, "foot": foot, "tip": tip,
                              "shape": "ground_bar" if transverse else "supply_arrow"})
    return supplies, marks


def _side_supply_labels(glyphs, words):
    """Pair native rotated power text with a complete adjacent arrow glyph."""
    pairs = []
    word_hits, glyph_hits = defaultdict(set), defaultdict(set)
    ordinary_words = {i for glyph in glyphs for i in glyph["ordinary_word_indices"]}
    for glyph_index, glyph in enumerate(glyphs):
        if glyph["shape"] != "supply_arrow":
            continue
        foot, tip = glyph["foot"], glyph["tip"]
        length = math.dist(foot, tip)
        ux, uy = (tip[0] - foot[0]) / length, (tip[1] - foot[1]) / length
        for word_index, word in enumerate(words):
            direction = word.get("baseline_direction", word.get("power_baseline_direction"))
            origin = word.get("baseline_origin", word.get("power_baseline_origin"))
            if (not _POWER_NAME.fullmatch(word["text"])
                    or (direction and abs(direction[0] * ux + direction[1] * uy) < 1 - 1e-6)):
                continue
            x, y, u, v = word["page_bbox_pt"]
            projected = [(((a - tip[0]) * ux + (b - tip[1]) * uy) / length,
                          ((b - tip[1]) * ux - (a - tip[0]) * uy) / length)
                         for a, b in ((x, y), (u, y), (u, v), (x, v))]
            along, across = zip(*projected)
            # In glyph-length units, the stem is [-1, 0]. The nearest word
            # edge is .1..1 lengths away, the far edge at most 2, and the
            # projected word covers at least half of the stem's run.
            if (min(across) * max(across) <= 0
                    or not .1 < min(map(abs, across)) < 1
                    or max(map(abs, across)) > 2
                    or min(0, max(along)) - max(-1, min(along)) < .5):
                continue
            # Unoriented power text can veto ambiguity, but cannot supply a name.
            witnessed = origin is not None and bool(direction)
            pairs.append((glyph_index, word_index, witnessed))
            glyph_hits[glyph_index].add(word_index)
            word_hits[word_index].add(glyph_index)
    candidates = []
    for glyph_index, word_index, witnessed in pairs:
        glyph, word = glyphs[glyph_index], words[word_index]
        candidates.append((glyph["wire_index"], {
            "name": word["text"], "wire_index": glyph["wire_index"],
            "foot": glyph["foot"], "tip": glyph["tip"], "shape": glyph["shape"],
            "label_placement": "side", "label_bbox_pt": word["page_bbox_pt"],
            "baseline_origin": word.get("baseline_origin", word.get("power_baseline_origin")),
            "baseline_direction": word.get("baseline_direction", word.get("power_baseline_direction")),
            "_word_index": word_index,
            "_admissible": (witnessed and not glyph["ordinary_names"]
                            and word_index not in ordinary_words
                            and len(glyph_hits[glyph_index]) == len(word_hits[word_index]) == 1),
        }))
    return candidates


def _corroborated_bank_label(witness, root, target, words, anchors, label_words,
                             word_roots, label_roots, occurrences, glyphs):
    """Require a contiguous native paint run and independent existing wire anchors."""
    name = witness["name"]
    same = [(i, w) for i, w in enumerate(words) if w["text"] == name]
    if len(same) < 3 or any("glyph_span" not in w for _, w in same):
        return False
    same.sort(key=lambda pair: pair[1]["glyph_span"][0])
    if any(a["glyph_span"][1] != b["glyph_span"][0] for (_, a), (_, b) in zip(same, same[1:])):
        return False
    supporting = set()
    for index, _ in same:
        if index == witness["_word_index"]:
            continue
        if index not in label_words or len(word_roots[index]) != 1:
            return False
        supporting.update(word_roots[index])
    if len(supporting - {root}) < 2 or supporting != anchors.get(name):
        return False
    competing = set(label_roots[target]) - {name}
    if any(_POWER_NAME.fullmatch(other) or _GROUND_NAME.fullmatch(other)
           or sum(w["text"] == other for w in words) != 1
           or len(occurrences[other]) != 1 or len(label_roots[target][other]) != 1
           for other in competing):
        return False
    if any(mark[0] != name for mark in glyphs[target]):
        return False
    occupied = set().union(*(label_roots[target][other] for other in competing))
    clean = supporting - occupied
    if not clean:
        return False
    witness["pin_bank_terminal"]["supporting_wire_islands"] = sorted(supporting)
    witness["pin_bank_terminal"]["text_glyph_spans"] = [w["glyph_span"] for _, w in same]
    witness["pin_bank_terminal"]["incidental_singletons"] = sorted(competing)
    return True


def _graph(wires, lines, words, *, wire_end_witnesses=None, symbol_obstacles=((), ())):
    groups, endpoints = Groups(len(wires)), Points(.12)
    for i, (a, b) in enumerate(wires):
        for p in (a, b):
            for j in endpoints.near(p):
                groups.join(i, j)
            endpoints.add(p, i)
    roots = [groups.root(i) for i in range(len(wires))]
    label_words, all_glyphs = set(), []
    word_roots = defaultdict(set)
    labels = _label_anchors(wires, roots, words, word_indices=label_words, word_roots=word_roots)
    supplies, marks = _supply_anchors(wires, roots, lines, words, all_glyphs=all_glyphs)
    # A printed supply word alone can land on a different wire. The full
    # glyph and its stem take precedence everywhere the symbol is observed.
    anchors = {name: hits for name, hits in labels.items() if name not in supplies}
    anchors.update(supplies)
    for hits in anchors.values():
        first = next(iter(hits))
        for other in hits:
            groups.join(first, other)
    # A loose-end label only extends an independently observed same-page anchor.
    # Snapshot existing groups before adding joins so inference cannot bootstrap.
    extra_lines, symbol_bounds = symbol_obstacles
    clearance_lines = [*lines, *extra_lines] if extra_lines else lines
    candidates = (_wire_end_labels(wires, roots, endpoints, clearance_lines, words, symbol_bounds, lines)
                  + _side_supply_labels(all_glyphs, words))
    # An ordinary glyph and a proposed wire end can overlap the same printed
    # occurrence. Track its actual island for every role before admitting joins.
    for glyph in all_glyphs:
        for index in glyph["ordinary_word_indices"]:
            word_roots[index].add(glyph["wire_index"])
    for root, witness in candidates:
        word_roots[witness["_word_index"]].add(root)
    label_roots = defaultdict(lambda: defaultdict(set))
    occurrences = defaultdict(set)
    for name, hits in anchors.items():
        for root in hits:
            label_roots[groups.root(root)][name].add(root)
            occurrences[name].add(root)
    for root, witness in candidates:
        name = witness["name"]
        label_roots[groups.root(root)][name].add(root)
        occurrences[name].add(root)
    glyphs = defaultdict(lambda: defaultdict(set))
    for mark in marks:
        root = mark["wire_index"]
        glyphs[groups.root(root)][mark["name"], mark["shape"]].add(root)
    accepted = []
    for root, witness in candidates:
        # A single printed occurrence cannot name two disconnected candidates.
        # This check is symmetric between loose-end and side-glyph proposals.
        word_index = witness["_word_index"]
        if len(word_roots[word_index]) != 1:
            continue
        if witness.get("label_placement") == "side":
            if not witness["_admissible"] or word_index in label_words:
                continue
        name = witness["name"]
        existing = anchors.get(name)
        if not existing or set(label_roots[groups.root(root)]) != {name}:
            continue
        target = groups.root(next(iter(existing)))
        if "pin_bank_terminal" in witness:
            if not _corroborated_bank_label(witness, root, target, words, anchors, label_words,
                                            word_roots, label_roots, occurrences, glyphs):
                continue
        else:
            if (_GROUND_NAME.fullmatch(name) or _POWER_NAME.fullmatch(name)) and name not in supplies:
                continue
            if (set(label_roots[target]) != {name} and
                    not _ground_glyph_precedence(name, [target], glyphs, label_roots, occurrences)):
                continue
        if target != groups.root(root):
            accepted.append((root, witness))
    for root, witness in accepted:
        name = witness["name"]
        groups.join(next(iter(anchors[name])), root)
        anchors[name].add(root)
        public_witness = {key: value for key, value in witness.items() if not key.startswith("_")}
        if witness.get("label_placement") == "side":
            marks.append(public_witness)
        elif wire_end_witnesses is not None:
            wire_end_witnesses.append(public_witness)
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
        glyphs, symbol_obstacles = [], ([], [])
        wires, red, fills, unsupported = _ink(drawing._page_svg(source, page, directory), glyphs, symbol_obstacles)
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
    _word_directions(stock["words"], glyphs)
    wire_end_witnesses = []
    membership, marks, anchors = _graph(wires, red, stock["words"],
                                        wire_end_witnesses=wire_end_witnesses, symbol_obstacles=symbol_obstacles)
    return {**result, "status": "candidate_evidence", "wire_groups": membership,
            "group_count": len(set(membership)), "supply_glyphs": marks,
            "wire_end_label_witnesses": wire_end_witnesses,
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
        traced_ground = _ground_glyph_precedence(
            name, components, glyph_evidence, by_component, occurrences)
        if conflicting_text and not traced_ground:
            continue
        canonical.update(dict.fromkeys(components, min(components)))
        link = {"name": name, "page_groups": sorted(components)}
        if traced_ground:
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
