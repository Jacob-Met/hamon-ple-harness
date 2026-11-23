"""Fixed independent X6 layer-46 witness oracle. All box and evidence units are nm."""
from collections import Counter
from fractions import Fraction
from xml.etree import ElementTree

from eagle_board_domain_oracle import P, Q, seg2


BOXES_NM = tuple(
    (left - 2, bottom - 2, right + 2, top + 2)
    for left, right in [(793500, 1993500), (4723500, 6423500)]
    for bottom, top in [(6810000, 7410000), (15450000, 16050000)]
)


def verify_original(path):
    board = ElementTree.parse(path).find("./drawing/board")
    element, = board.findall("./elements/element[@name='X6']")
    assert tuple(element.get(k) for k in ("library", "package", "x", "y", "rot")) == (
        "microbuilder", "USB_C_CUSB31-CFM2AX-01-X", "2.6035", "11.43", "R270"
    )
    package, = board.findall("./libraries/library[@name='microbuilder']/packages/"
                             "package[@name='USB_C_CUSB31-CFM2AX-01-X']")
    actual = Counter()
    for wire in package.iter():
        if wire.get("layer") == "46":
            assert wire.tag == "wire" and Q(wire.get("width")) == 0
            actual[(P(wire.get("x1"), wire.get("y1")), P(wire.get("x2"), wire.get("y2")),
                    Q(wire.get("curve", "0")))] += 1
    expected = Counter()
    boxes = []
    radius = Q(".30")
    for cx in map(Q, ["-4.32", "4.32"]):
        for lower, upper in [P("-1.51", "-.91"), P("2.42", "3.52")]:
            vertices = [(cx-radius, upper), (cx, upper+radius), (cx+radius, upper),
                        (cx+radius, lower), (cx, lower-radius), (cx-radius, lower)]
            for a, b, curve in zip(vertices, vertices[1:]+vertices[:1], [-90,-90,0,-90,-90,0]):
                expected[a, b, curve] += 1
            xs, ys = zip(*[((Q("2.6035")+y)*10**6, (Q("11.43")-x)*10**6)
                           for x, y in vertices])
            boxes.append((min(xs)-2, min(ys)-2, max(xs)+2, max(ys)+2))
    assert actual == expected and sorted(boxes) == sorted(BOXES_NM)
    return BOXES_NM


def _segment_clears(a, b, radius):
    for x0, y0, x1, y1 in BOXES_NM:
        if x0 <= a[0] <= x1 and y0 <= a[1] <= y1:
            return False
        corners = [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]
        if any(seg2(a, b, c, d) <= radius**2
               for c, d in zip(corners, corners[1:]+corners[:1])):
            return False
    return True


def path_clears_x6(vertices_nm):
    points = [P(*point) for point in vertices_nm]
    return len(set(points)) >= 2 and all(
        _segment_clears(a, b, Fraction(1)) for a, b in zip(points, points[1:])
    )


def disk_clears_x6(center_nm, radius_nm):
    center, radius = P(*center_nm), Q(radius_nm)
    return radius > 0 and _segment_clears(center, center, radius)
