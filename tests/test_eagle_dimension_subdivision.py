"""A straight Dimension wire may be authored as several forward-collinear pieces."""
from fractions import Fraction as F
import unittest

from hamon_eagle_material import nominal_board_domain


def loop(vertices, sweeps=()):
    return [(p, vertices[(i+1) % len(vertices)], sweeps[i] if sweeps else 0)
            for i, p in enumerate(vertices)]


OUTER = loop([(-400, -400), (400, -400), (400, 400), (-400, 400)])
OPENING = loop([(-80, -100), (80, -100), (100, -80), (100, 80),
                (80, 100), (-80, 100), (-100, 80), (-100, -80)],
               [0, 90, 0, 90, 0, 90, 0, 90])


def records(edges, rotate=False):
    def world(p):
        x, y = map(F, p)
        return ((3*x-4*y)/5, (4*x+3*y)/5) if rotate else (x, y)
    return [dict(a=world(a), b=world(b), curve=s, source=("plain", "wire", i))
            for i, (a, b, s) in enumerate(edges)]


def partition(edges, fractions):
    result = []
    for a, b, sweep in edges:
        if sweep:
            result.append((a, b, sweep))
            continue
        pts = [tuple(a[k]+f*(b[k]-a[k]) for k in range(2)) for f in fractions]
        result.extend((p, q, 0) for p, q in zip(pts, pts[1:]))
    return result


class DimensionSubdivisionTests(unittest.TestCase):
    def test_all_tangents_and_outer_edges_preserve_geometry_and_wire_sources(self):
        # Some pieces are less than the 8 nm unresolved-bridge threshold;
        # their complete physical bridge is 160 nm long.
        fractions = [F(0), F(1, 160), F(1, 2), F(1)]
        for rotate in (False, True):
            reference, _ = nominal_board_domain(records(OUTER + OPENING, rotate))
            outer, inner = partition(OUTER, fractions), partition(OPENING, fractions)
            raw = records(outer + inner, rotate)
            for rows in (raw, raw[::2]+raw[1::2],
                         [dict(r, a=r["b"], b=r["a"], curve=-r["curve"]) for r in raw[::-1]]):
                geometry, proof = nominal_board_domain(rows)
                self.assertTrue(geometry.equals(reference))
                self.assertEqual(proof["wire_count"], len(raw))
                self.assertEqual(proof["hole_wire_count"], len(inner))
                self.assertCountEqual(map(tuple, proof["sources"]), (r["source"] for r in raw))
                self.assertCountEqual(map(tuple, proof["hole_sources"]), (r["source"] for r in raw[len(outer):]))
                self.assertCountEqual(map(tuple, proof["hole_quarter_sources"]), (r["source"] for r in raw if r["curve"]))

    def test_kinks_and_doubled_back_paths_remain_unresolved(self):
        a, b, _ = OPENING[0]
        for chain in ([a, (0, -99), b], [a, (20, -100), (-20, -100), b]):
            changed = [(p,q,0) for p,q in zip(chain, chain[1:])] + OPENING[1:]
            with self.assertRaises(ValueError):
                nominal_board_domain(records(OUTER + changed))

    def test_true_short_bridge_cannot_be_lengthened_by_subdivision(self):
        # Replace each horizontal tangent by a true 8 nm bridge. Its four
        # 2 nm authored pieces must receive the same refusal as the original.
        narrow = [((x//abs(x))*(abs(x)-76), y) for (x,y), _, _ in OPENING]
        edges = loop(narrow, [s for _,_,s in OPENING])
        for inner in (edges, partition(edges,[F(0),F(1,4),F(1,2),F(3,4),F(1)])):
            with self.assertRaises(ValueError):
                nominal_board_domain(records(OUTER+inner))
