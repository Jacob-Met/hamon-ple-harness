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
    "w": ["04 01 10 21 30 41 44"],
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
_GLYPH_SHAPES = {
    tuple(sorted(tuple((x-min(p[0] for line in shape for p in line),
                       y-min(p[1] for line in shape for p in line)) for x, y in edge)
                 for edge in shape))
    for shape in _ALPHABET
}
_RGB = re.compile(r"rgb\(([\d.]+)%,\s*([\d.]+)%,\s*([\d.]+)%\)")


def letter_strokes(paths, wire_ink, pitch):
    """Keep visible neutral lettering separated by its physical pen width."""
    letters, signal = [], defaultdict(list)
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
            if math.isfinite(pen) and pen > 0:
                signal[round(pen, 6)].append(edge)
    return LetterStrokes(letters), signal


class LetterStrokes:
    """Bound nearby searches without depending on SVG path order."""
    def __init__(self, segments):
        self.cells = defaultdict(list)
        self.lines = [tuple(tuple(p) for p in edge) for edge in segments]
        touching = defaultdict(list)
        for index, (a, b) in enumerate(self.lines):
            x = (a[0] + b[0]) / 2
            y = (a[1] + b[1]) / 2
            self.cells[math.floor(x / 16), math.floor(y / 16)].append(index)
            touching[a].append(index)
            touching[b].append(index)
        # Follow whole endpoint components before any spatial crop. A clipped
        # fragment of a neighboring glyph cannot explain a word's extra ink.
        self.component_ids, self.components = {}, []
        for first in range(len(self.lines)):
            if first in self.component_ids:
                continue
            component, pending = [], [first]
            self.component_ids[first] = len(self.components)
            while pending:
                index = pending.pop()
                component.append(self.lines[index])
                for point in self.lines[index]:
                    for other in touching[point]:
                        if other not in self.component_ids:
                            self.component_ids[other] = len(self.components)
                            pending.append(other)
            self.components.append(tuple(component))

    def _nearby_indices(self, tip, radius):
        for x in range(math.floor((tip[0] - radius) / 16), math.floor((tip[0] + radius) / 16) + 1):
            for y in range(math.floor((tip[1] - radius) / 16), math.floor((tip[1] + radius) / 16) + 1):
                yield from self.cells[x, y]

    def nearby(self, tip, radius):
        return (self.lines[index] for index in self._nearby_indices(tip, radius))

    def connected_nearby(self, tip, radius):
        return (self.components[index] for index in
                sorted({self.component_ids[i] for i in self._nearby_indices(tip, radius)}))


def _glyph_shape(lines, direction, units):
    dx, dy = direction
    sx, sy = units
    projected = [tuple(((dx*x+dy*y)/sx, (dy*x-dx*y)/sy) for x, y in edge) for edge in lines]
    left = min(x for edge in projected for x, _ in edge)
    low = min(y for edge in projected for _, y in edge)
    shifted = [tuple((x-left, y-low) for x, y in edge) for edge in projected]
    if any(abs(v-round(v)) > .32 for edge in shifted for point in edge for v in point):
        return None
    return tuple(sorted(tuple(sorted((round(x), round(y)) for x, y in edge)) for edge in shifted))


def _orthogonal_glyph_ink(components, direction, units):
    """Explain ink with complete, distinguishably orthogonal font glyphs only.

    This gives no electrical name or attachment to an annotation. Collinear
    fragments and glyphs that also fit the word's own orientation are excluded.
    """
    perpendicular = (0, -1) if direction[0] else (1, 0)
    maximum = max(map(len, _GLYPH_SHAPES))
    ink = set()
    for lines in components:
        if len(lines) > maximum:
            continue
        shape = _glyph_shape(lines, perpendicular, units[::-1])
        if shape not in _GLYPH_SHAPES or _glyph_shape(lines, direction, units) in _GLYPH_SHAPES:
            continue
        points = [point for edge in shape for point in edge]
        a, b = points[0], next(point for point in points if point != points[0])
        if all((x-a[0])*(b[1]-a[1]) == (y-a[1])*(b[0]-a[0]) for x, y in points):
            continue
        ink.update(tuple(sorted(edge)) for edge in lines)
    return frozenset(ink)


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
    # Horizontal stroke pitch can differ slightly from vertical pitch. A long
    # word can drift by a full grid cell when using the vertical estimate.
    names = set()
    for width in range(max(4, columns-3), min(120, columns+3)+1):
        name = _decode_signal_grid(edges, left, baseline, (right-left)/width, step_y)
        if name is not None:
            names.add(name)
    return names.pop() if len(names) == 1 else None


def _decode_signal_grid(edges, left, baseline, step_x, step_y):
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
    return name if re.fullmatch(r"[+A-Z0-9][A-Z0-9_+/().-]{1,23}", name) and re.search("[A-Z]", name) else None


def wire_words(wires, segments, pitch):
    """Bind a complete style-consistent word by containment or a registered origin."""
    if isinstance(segments, dict):
        if len(segments) > 16:
            return
        for ink in segments.values():
            yield from wire_words(wires, ink, pitch)
        return
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
            # Upright lettering can register either its baseline or its cap line to the wire.
            for scan_y in (y, y + .70*pitch):
                # Collect past the wire ends before testing whole-word containment.
                # Cutting at a wire end could make different longer names identical.
                near = []
                for k in range(math.floor(scan_y/pitch)-1, math.floor(scan_y/pitch)+2):
                    for edge in buckets[k]:
                        xmin, xmax = sorted(p[0] for p in edge)
                        ymin, ymax = sorted(p[1] for p in edge)
                        if (ymin >= scan_y-.76*pitch and ymax <= scan_y+.13*pitch
                                and xmax >= left-16*pitch and xmin <= right+16*pitch):
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
                    # A complete label may extend beyond its wire. Its exact
                    # grid-registered origin and unique physical occurrence bind
                    # it below; a percentage of word width is not that evidence.
                    overlap = min(end, right) - max(start, left)
                    if overlap <= 0 or not 4 <= len(edges) <= 140:
                        continue
                    name = _signal_word(edges, scan_y-.03*pitch, pitch)
                    if name:
                        occurrence = tuple(sorted(tuple(sorted(line)) for line in edges))
                        # A horizontal label origin can be left- or right-aligned.
                        offsets = []
                        if left-.08*pitch < start < right+.08*pitch:
                            offsets.append((start-left)/pitch-.03)
                        if left-.08*pitch < end < right+.08*pitch:
                            offsets.append((right-end)/pitch-.03)
                        # Label origins can occupy quarter-pitch drafting sites.
                        # Keep the same absolute coordinate residual at each site.
                        # A whole word contained by one wire also has a direct
                        # witness, including mirrored placements off that grid.
                        registered = (any(abs(4*v-round(4*v)) < .100 for v in offsets)
                                      or (start >= left and end <= right))
                        yield index, name, (vertical, occurrence), registered

def _fit_voltage(nearby, center, direction, units, name, orthogonal=()):
    """Fit a complete word, retaining ink without a full orthogonal glyph witness."""
    dx, dy = direction
    sx, sy = units
    extent = 6*len(name)-2
    expected = {
        tuple((6*i+x, y) for x, y in edge)
        for i, letter in enumerate(name) for edge in _edges(_GLYPHS[letter])
    }
    physical, actual, extra = [], [], []
    adjacent = {offset: [] for offset in (-6, extent+2)}
    for a, b in nearby:
        edge = []
        for point in (a, b):
            x, y = point[0]-center[0], point[1]-center[1]
            edge.append((2+(dx*x+dy*y)/sx, 3+(dy*x-dx*y)/sy))
        # Include the entire previous and following character cell. A lone
        # surviving stroke of a damaged suffix must also remain unexplained.
        if not any(-6.4 < x < extent+6.4 and -1.32 < y < 6.32 for x, y in edge):
            continue
        stroke = tuple(sorted((a, b)))
        rounded = tuple(sorted((round(x), round(y)) for x, y in edge))
        on_grid = all(abs(v-round(v)) <= .32 for p in edge for v in p)
        if on_grid:
            for offset, strokes in adjacent.items():
                if all(offset <= x <= offset+4 and 0 <= y <= 6 for x, y in rounded):
                    strokes.append(tuple((x-offset, y) for x, y in rounded))
        if on_grid and rounded in expected:
            if stroke in orthogonal:
                return None
            actual.append(rounded)
            physical.append(stroke)
        elif stroke not in orthogonal:
            extra.append(stroke)
    # Another arrow cannot turn a continuous name into two shorter names.
    # Additional neighboring ink cannot hide a complete continuation glyph.
    if any(set(shape) <= set(strokes) for shape in _ALPHABET for strokes in adjacent.values()):
        return None
    if len(actual) == len(expected) and set(actual) == expected:
        return tuple(sorted(physical)), frozenset(extra)


def _voltage_candidates(mark, lettering):
    """Two perpendicular strokes of a printed plus fix the font origin and scale."""
    foot, tip = mark["foot"], mark["tip"]
    stem = math.dist(foot, tip)
    along = tuple((b-a)/stem for a, b in zip(foot, tip))
    nearby = list(lettering.nearby(tip, 4*stem))
    components = tuple(lettering.connected_nearby(tip, 4*stem))
    crosses = defaultdict(list)
    for a, b in nearby:
        delta = tuple(y-x for x, y in zip(a, b))
        length = math.hypot(*delta)
        if .39*stem < length < .46*stem:
            vertical = abs(delta[0]) < .02
            if vertical or abs(delta[1]) < .02:
                crosses[vertical].append((tuple((x+y)/2 for x, y in zip(a, b)), length))
    matches = set()
    for center, height in crosses[True]:
        if math.dist(center, tip) > 2.9*stem:
            continue
        for cross_center, width in crosses[False]:
            if math.dist(center, cross_center) > .10:
                continue
            for dx, dy in ((1, 0), (0, -1)):
                sx, sy = (width/4, height/4) if dx else (height/4, width/4)
                orthogonal = _orthogonal_glyph_ink(components, (dx, dy), (sx, sy))
                for name in ("+5V", "+3V3"):
                    distance = (3*len(name)-3)*sx
                    mx, my = center[0]+dx*distance-tip[0], center[1]+dy*distance-tip[1]
                    forward = (mx*along[0]+my*along[1])/stem
                    across = (mx*along[1]-my*along[0])/stem
                    if abs(dx*along[0]+dy*along[1]) < .05:
                        placed = .02 < forward < 1.20 and abs(across) < .45
                    else:
                        placed = -.90 < forward < .55 and .85 < abs(across) < 1.90
                    if placed:
                        occurrence = _fit_voltage(nearby, center, (dx, dy), (sx, sy), name, orthogonal)
                        if occurrence:
                            matches.add((name, *occurrence))
    return matches



def paired_voltages(marks, lettering, components):
    """Keep isolated words and recover two crowded arrow legends with mutually explained ink."""
    candidates = [_voltage_candidates(mark, lettering) for mark in marks]
    readings = [{(name, ink) for name, ink, extra in choices if not extra} for choices in candidates]
    # The isolated-word pass above remains independent. A new crowded reading
    # has a stricter ownership requirement: no alternative candidate may claim
    # the same strokes for another word or electrical component.
    owners = defaultdict(set)
    for choices, component in zip(candidates, components, strict=True):
        for name, ink, _ in choices:
            for stroke in ink:
                owners[stroke].add((name, ink, component))
    crowded = []
    for index, choices in enumerate(candidates):
        if len(choices) == 1:
            name, ink, extra = next(iter(choices))
            if extra and all(len(owners[stroke]) == 1 for stroke in ink):
                crowded.append((index, name, ink, extra))
    partners = defaultdict(list)
    for pos, (i, name, ink, extra) in enumerate(crowded):
        for j, other_name, other_ink, other_extra in crowded[:pos]:
            if (components[i] != components[j] and extra.issubset(other_ink)
                    and other_extra.issubset(ink) and set(ink).isdisjoint(other_ink)):
                partners[i].append((j, name, ink))
                partners[j].append((i, other_name, other_ink))
    for i, matches in partners.items():
        if len(matches) == 1:
            j, name, ink = matches[0]
            if len(partners[j]) == 1:
                readings[i].add((name, ink))
    return [next(iter(options)) if len(options) == 1 else None for options in readings]


def anchored_voltage(mark, lettering):
    """Read a single arrow only when its complete caption has no unexplained ink."""
    return paired_voltages([mark], lettering, [0])[0]


def _border_character(edges, bounds):
    left, top, right, bottom = bounds
    strokes = [edge for edge in edges if all(
        left < x < right and top < y < bottom for x, y in edge)]
    if not 1 <= len(strokes) <= 20:
        return None
    xs, ys = zip(*(p for edge in strokes for p in edge))
    unit = (max(ys) - min(ys)) / 6
    if unit <= 0:
        return None
    origin, baseline = min(xs), max(ys)
    lattice = [tuple(((x-origin)/unit, (baseline-y)/unit) for x, y in edge) for edge in strokes]
    if any(abs(v-round(v)) > .20 for edge in lattice for p in edge for v in p):
        return None
    shape = tuple(sorted(tuple(sorted(tuple(round(v) for v in point) for point in edge)) for edge in lattice))
    return _ALPHABET.get(shape)


def _border_divisions(horizontal, span, borders, edges):
    lo, hi = span
    ticks = defaultdict(lambda: defaultdict(set))
    for a, b in edges:
        (x, y), (u, v) = (a, b) if horizontal else (a[::-1], b[::-1])
        if x != u or not lo < x < hi:
            continue
        for border, direction in zip(borders, (-1, 1)):
            if y == border and (v-y)*direction > 0:
                ticks[x][direction].add(round(abs(v-y), 1))
            if v == border and (y-v)*direction > 0:
                ticks[x][direction].add(round(abs(y-v), 1))
    sizes = defaultdict(list)
    for coordinate, sides in ticks.items():
        for size in sides[-1] & sides[1]:
            sizes[size].append(coordinate)
    for size, coordinates in sizes.items():
        bounds = [lo, *sorted(coordinates), hi]
        widths = [b-a for a,b in zip(bounds, bounds[1:])]
        if 2 <= len(widths) <= 26 and max(widths)-min(widths) < .02*min(widths) and size < .25*min(widths):
            yield size, bounds


def _sheet_frames(paths, wires):
    pens = defaultdict(list)
    for color, items in paths.items():
        for element, matrix in items:
            points = drawing._path_points(element.get("d", ""))
            if not points or len(points) != 2:
                continue
            edge = tuple(tuple(round(v, 2) for v in drawing.transform_point(matrix,p)) for p in points)
            pens[color, element.get("stroke-width")].append(edge)
    xs, ys = zip(*(p for wire in wires for p in wire))
    min_width, min_height = .6*(max(xs)-min(xs)), .6*(max(ys)-min(ys))
    for (color, _), edges in pens.items():
        segments = {tuple(sorted(edge)) for edge in edges}
        wide = [(a,b) for a,b in segments if a[1] == b[1] and b[0]-a[0] > min_width]
        if len(wide) > 32:
            continue
        for a, b in wide:
            for c, d in wide:
                if (c[0] != a[0] or d[0] != b[0] or c[1]-a[1] <= min_height
                        or (a,c) not in segments or (b,d) not in segments):
                    continue
                x_span, y_span = (a[0], b[0]), (a[1], c[1])
                columns = list(_border_divisions(True, x_span, y_span, edges))
                rows = list(_border_divisions(False, y_span, x_span, edges))
                if len(columns) == len(rows) == 1:
                    labels = [edge for (ink,_), strokes in pens.items() if ink == color for edge in strokes]
                    yield labels, columns[0], rows[0]


def _printed_zone(frame, tip):
    labels, (height, xs), (width, ys) = frame
    def interval(bounds, value):
        return next(((a,b) for a,b in zip(bounds,bounds[1:]) if a+.1 < value < b-.1), None)
    column, row = interval(xs, tip[0]), interval(ys, tip[1])
    if not column or not row:
        return None
    left, right = column
    top, bottom = row
    digits = [_border_character(labels, (left,a,right,b))
              for a,b in ((ys[0]-height,ys[0]),(ys[-1],ys[-1]+height))]
    letters = [_border_character(labels, (a,top,b,bottom))
               for a,b in ((xs[0]-width,xs[0]),(xs[-1],xs[-1]+width))]
    if (digits[0] and digits[0].isdigit() and digits[0] == digits[1]
            and letters[0] and letters[0].isalpha() and letters[0] == letters[1]):
        return digits[0] + letters[0]
    return None


def _pointed_frames(strokes, endpoints, pitch):
    for lines in strokes.values():
        neighbors = defaultdict(set)
        for a, b in lines:
            a, b = (tuple(round(v,2) for v in point) for point in (a,b))
            neighbors[a].add(b)
            neighbors[b].add(a)
        for tip in endpoints.keys() & neighbors.keys():
            if len(endpoints[tip]) != 1 or len(neighbors[tip]) != 2:
                continue
            a, b = sorted(neighbors[tip])
            if (a[0] != b[0] or abs(tip[0]-a[0]) < .3*pitch
                    or abs(a[1]+b[1]-2*tip[1]) > .04
                    or any(abs(abs(tip[0]-x)-abs(tip[1]-y)) > .04 for x,y in (a,b))):
                continue
            a_back, b_back = neighbors[a]-{tip}, neighbors[b]-{tip}
            if len(a_back) != 1 or len(b_back) != 1:
                continue
            c, d = next(iter(a_back)), next(iter(b_back))
            if (neighbors[c] != {a,d} or neighbors[d] != {b,c} or c[0] != d[0]
                    or c[1] != a[1] or d[1] != b[1] or (tip[0]-a[0])*(a[0]-c[0]) <= 0
                    or not 2*pitch < abs(a[0]-c[0]) < 28*pitch):
                continue
            yield tip, (min(a[0],c[0]), min(a[1],b[1]), max(a[0],c[0]), max(a[1],b[1]))


def cross_references(paths, wire_ink, pitch, wires):
    """Read a pointed wire-terminal caption and independently read its own border zone."""
    if sum(map(len, paths.values())) > 65000:
        return []
    _, strokes = letter_strokes(paths, wire_ink, pitch)
    endpoints = defaultdict(list)
    for index, line in enumerate(wires):
        for point in line:
            endpoints[tuple(round(v,2) for v in point)].append(index)
    pointed = list(_pointed_frames(strokes, endpoints, pitch))
    if not pointed or len(pointed) > 128:
        return []
    frames = list(_sheet_frames(paths, wires))
    if len(frames) != 1:
        return []
    references = []
    for tip, (left,top,right,bottom) in pointed:
        # Mixed pens, an unreadable stroke, or an incomplete glyph withhold the whole caption.
        in_frame = [[line for line in pen if all(left+.03<x<right-.03 and top+.03<y<bottom-.03
                                                  for x,y in line)] for pen in strokes.values()]
        words = [word for word in in_frame if word]
        if len(words) != 1 or not 4 <= len(words[0]) <= 140:
            continue
        edges = words[0]
        caption = _signal_word(edges, max(y for line in edges for _,y in line), pitch)
        parsed = re.fullmatch(r"([A-Z][A-Z0-9_().+-]*)/([1-9][0-9]?)\.([1-9][A-Z])", caption or "")
        zone = _printed_zone(frames[0], tip)
        if parsed and zone:
            name, destination, target_zone = parsed.groups()
            references.append({"name": name, "wire_index": endpoints[tip][0], "tip": list(tip),
                               "zone": zone, "target_page": int(destination), "target_zone": target_zone})
    return references

