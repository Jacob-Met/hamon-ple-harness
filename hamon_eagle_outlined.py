"""Recognize complete framed stroke words at EAGLE symbols and signal wires."""
from collections import defaultdict
import math
import re

import hamon_drawing as drawing


# Roman vector strokes in the EAGLE four-by-six letter cell, with y upward.
# Every edge of every letter must be present. Digits or suffixes cannot be
# discarded just because a shorter supply name would be recognizable.
_GLYPHS = {
    "0": ["45 41 30 10 01 05 16 36 45 01"],
    "1": ["00 40", "04 26 20"],
    "2": ["05 16 36 45 44 00 40"],
    "3": ["01 10 30 41 42 33 23", "05 16 36 45 44 33"],
    "4": ["43 03 36 30"],
    "5": ["01 10 30 41 43 34 24 03 06 46"],
    "6": ["03 33 42 41 30 10 01 03 25 46"],
    "7": ["00 01 45 46 06"],
    "8": ["13 33 42 41 30 10 01 02 13 04 05 16 36 45 44 33"],
    "9": ["01 10 30 41 45 36 16 05 04 13 43"],
    "(": ["20 02 04 26"],
    ")": ["00 22 24 06"],
    "+": ["03 43", "21 25"],
    "-": ["03 43"],
    ".": ["10 00 01 11 10"],
    "/": ["00 46"],
    "A": ["00 04 26 44 40", "03 43"],
    "B": ["03 33 42 41 30 00 06 36 45 44 33"],
    "C": ["41 30 10 01 05 16 36 45"],
    "D": ["41 30 00 06 36 45 41"],
    "E": ["40 00 06 46", "03 23"],
    "F": ["00 06 46", "03 23"],
    "G": ["23 43 41 30 10 01 05 16 36 45"],
    "H": ["00 06", "03 43", "40 46"],
    "I": ["00 20", "06 26", "10 16"],
    "J": ["01 10 20 31 36", "26 46"],
    "K": ["00 06", "02 46", "13 40"],
    "L": ["40 00 06"],
    "M": ["00 06 24 46 40"],
    "N": ["00 06 40 46"],
    "O": ["41 30 10 01 05 16 36 45 41"],
    "P": ["00 06 36 45 43 32 02"],
    "R": ["00 06 36 45 43 32 02", "22 40"],
    "S": ["01 10 30 41 42 33 13 04 05 16 36 45"],
    "T": ["06 46", "20 26"],
    "U": ["46 41 30 10 01 06"],
    "V": ["46 42 20 02 06"],
    "W": ["46 40 22 00 06"],
    "X": ["00 46", "06 40"],
    "x": ["00 44", "04 40"],
}


def _edges(strokes):
    result = []
    for stroke in strokes:
        points = [tuple(map(int, pair)) for pair in stroke.split()]
        result.extend(tuple(sorted((a, b))) for a, b in zip(points, points[1:]))
    return tuple(sorted(result))


_ALPHABET = {_edges(strokes): letter for letter, strokes in _GLYPHS.items()}
_RGB = re.compile(r"rgb\(([\d.]+)%,\s*([\d.]+)%,\s*([\d.]+)%\)")


def letter_strokes(paths, wire_ink, pitch):
    """Use solid, visible neutral ink. Signal letters share the wire pen width."""
    letters, signal = [], []
    for ink, elements in paths.items():
        rgb = _RGB.fullmatch(ink)
        if ink == wire_ink or not rgb:
            continue
        channels = tuple(map(float, rgb.groups()))
        if not (5 < min(channels) <= max(channels) < 75 and max(channels)-min(channels) < .1):
            continue
        for element, matrix in elements:
            points = drawing._path_points(element.get("d", ""))
            if not points or len(points) != 2:
                continue
            edge = tuple(drawing.transform_point(matrix, p) for p in points)
            if math.dist(*edge) < .02:
                continue
            letters.append(edge)
            try:
                pen = float(element.get("stroke-width", "1")) * math.hypot(*matrix[:2])
            except ValueError:
                continue
            if .060 * pitch < pen < .073 * pitch:
                signal.append(edge)
    return LetterStrokes(letters), signal


class LetterStrokes:
    """Bound nearby searches without depending on SVG path order."""
    def __init__(self, segments):
        self.cells = defaultdict(list)
        for a, b in segments:
            x = (a[0] + b[0]) / 2
            y = (a[1] + b[1]) / 2
            self.cells[math.floor(x / 16), math.floor(y / 16)].append((a, b))

    def nearby(self, tip, radius):
        for x in range(math.floor((tip[0] - radius) / 16), math.floor((tip[0] + radius) / 16) + 1):
            for y in range(math.floor((tip[1] - radius) / 16), math.floor((tip[1] + radius) / 16) + 1):
                yield from self.cells[x, y]


def _word(lines, direction):
    points = [(direction*x, direction*y) for segment in lines for x, y in segment]
    left, right = min(x for x, _ in points), max(x for x, _ in points)
    bottom, top = min(y for _, y in points), max(y for _, y in points)
    height = top - bottom
    if not .54 < height < .77:
        return None
    step_y = height / 6
    length = round(((right - left) / step_y - 4) / 6) + 1
    if not 2 <= length <= 6:
        return None
    step_x = (right - left) / (6 * length - 2)
    if not .95 < step_x / step_y < 1.05:
        return None
    letters = defaultdict(list)
    for i in range(0, len(points), 2):
        ends = []
        for x, y in points[i:i+2]:
            u, v = (x-left)/step_x, (y-bottom)/step_y
            if max(abs(u-round(u)), abs(v-round(v))) > .22:
                return None
            ends.append((round(u), round(v)))
        pos = ends[0][0] // 6
        if any(x // 6 != pos or x % 6 > 4 or not 0 <= y <= 6 for x, y in ends):
            return None
        letters[pos].append(tuple(sorted((x%6, y) for x,y in ends)))
    result = [_ALPHABET.get(tuple(sorted(letters[i]))) for i in range(length)]
    return "".join(result) if all(result) else None


def supply_word(mark, lettering):
    foot, tip = mark["foot"], mark["tip"]
    length = math.dist(foot, tip)
    ux, uy = (tip[0]-foot[0])/length, (tip[1]-foot[1])/length
    def project(p):
        dx, dy = (p[0]-tip[0])/length, (p[1]-tip[1])/length
        return -dx*uy + dy*ux, dx*ux + dy*uy
    nearby = [(a, b, project(a), project(b)) for a, b in lettering.nearby(tip, 4*length)]
    def read_region(inside, turn):
        segments, physical, outside = [], [], []
        for a, b, p, q in nearby:
            hits = [inside(*t) for t in (p, q)]
            local = tuple((y, -x) if turn else (x, y) for x,y in (p,q))
            if all(hits):
                segments.append(local)
                physical.append(tuple(sorted((a,b))))
            elif any(hits):
                return None
            else:
                outside.append(local)
        if not 5 <= len(segments) <= 55:
            return None
        words = {_word(segments, sense) for sense in (-1,1)} - {None}
        if len(words) != 1:
            return None
        word = words.pop()
        edges = [v for line in segments for v in line]
        xmin, xmax = min(x for x,_ in edges), max(x for x,_ in edges)
        ymin, ymax = min(y for _,y in edges), max(y for _,y in edges)
        # A word can continue beyond the collection window without any
        # individual segment crossing its boundary. Guard both letter ends.
        for line in outside:
            if any(xmin-.65 < x < xmax+.65 and ymin-.02 < y < ymax+.02 for x,y in line):
                return None
        if word in {"+5V", "+3V3", "VCC"}:
            return word, tuple(sorted(physical))
        return None
    sources = [read_region(lambda x,y: abs(x) < 2.1 and -.035 < y < 1.18, False)]
    if mark["shape"] == "supply_arrow":
        for side in (-1,1):
            sources.append(read_region(lambda x,y: .85 < side*x < 1.88 and -2.2 < y < .70, True))
    unique = set(sources) - {None}
    return unique.pop() if len(unique) == 1 else None


def _signal_word(edges, baseline, pitch):
    """Decode an entire baseline run, including the underscore below the wire."""
    points = [p for edge in edges for p in edge]
    left, right = min(x for x, _ in points), max(x for x, _ in points)
    top = min(y for _, y in points)
    step_y = (baseline - top) / 6
    if not .095 < step_y / pitch < .12:
        return None
    columns = round((right - left) / step_y)
    if not 4 <= columns <= 120:
        return None
    step_x = (right - left) / columns
    if not .975 < step_x / step_y < 1.025:
        return None
    quantized = []
    for line in edges:
        points = [((x-left)/step_x, (baseline-y)/step_y) for x, y in line]
        if any(abs(v-round(v)) > .235 for p in points for v in p):
            return None
        quantized.append(tuple(sorted((round(x), round(y)) for x, y in points)))
    cells = []
    for edge in sorted(quantized):
        low, high = edge[0][0], edge[1][0]
        if cells and low <= cells[-1][1]:
            cells[-1][1] = max(cells[-1][1], high)
            cells[-1][2].append(edge)
        else:
            cells.append([low, high, [edge]])
    if not 2 <= len(cells) <= 24 or any(b[0]-a[1] != 2 for a, b in zip(cells, cells[1:])):
        return None
    word = []
    for low, _, lines in cells:
        shape = tuple(sorted(tuple((x-low, y) for x, y in edge) for edge in lines))
        letter = "_" if shape == (((0, -1), (4, -1)),) else _ALPHABET.get(shape)
        if letter is None:
            return None
        word.append(letter)
    name = "".join(word)
    return name if re.fullmatch(r"[A-Z0-9][A-Z0-9_+/().-]{1,23}", name) and re.search("[A-Z]", name) else None


def wire_words(wires, segments, pitch):
    """Read a complete word only when its baseline and pen register to one wire."""
    for vertical in (False, True):
        def project(p):
            return (-p[1], p[0]) if vertical else p

        buckets = defaultdict(list)
        for edge in segments:
            a, b = map(project, edge)
            buckets[math.floor((a[1]+b[1])/(2*pitch))].append((a, b))
        for index, (a, b) in enumerate(wires):
            (x0, y), (x1, y1) = project(a), project(b)
            if abs(y-y1) > .10:
                continue
            left, right = sorted((x0, x1))
            # Collect past the wire ends before testing whole-word containment.
            # Cutting at a wire end could make different longer names identical.
            near = []
            for k in range(math.floor(y/pitch)-1, math.floor(y/pitch)+2):
                for edge in buckets[k]:
                    xmin, xmax = sorted(p[0] for p in edge)
                    ymin, ymax = sorted(p[1] for p in edge)
                    if (ymin < y+.13*pitch and ymax > y-.76*pitch
                            and xmax >= left-pitch and xmin <= right+pitch):
                        near.append((xmin, xmax, edge))
            groups = []
            for xmin, xmax, edge in sorted(near):
                if groups and xmin-groups[-1][1] < .44*pitch:
                    groups[-1][1] = max(groups[-1][1], xmax)
                    groups[-1][2].append(edge)
                else:
                    groups.append([xmin, xmax, [edge]])
            for group_index, (start, end, edges) in enumerate(groups):
                # An erased letter (particularly an underscore below the line)
                # can leave two otherwise valid shorter words one cell apart.
                before = group_index and start-groups[group_index-1][1] < .96*pitch
                after = group_index+1 < len(groups) and groups[group_index+1][0]-end < .96*pitch
                if before or after:
                    continue
                if start < left-.08*pitch or end > right+.08*pitch or not 4 <= len(edges) <= 140:
                    continue
                name = _signal_word(edges, y-.03*pitch, pitch)
                if name:
                    occurrence = tuple(sorted(tuple(sorted(line)) for line in edges))
                    # A horizontal label origin can be left- or right-aligned.
                    offsets = ((start-left)/pitch-.03, (right-end)/pitch-.03)
                    registered = min(abs(v-round(v)) for v in offsets) < .025
                    yield index, name, (vertical, occurrence), registered
