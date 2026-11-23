"""Source-exact controls for the strict separation of two guarded Dimension cutouts."""
import unittest
from fractions import Fraction as F

from hamon_eagle_material import nominal_board_domain


def rectangle(origin, u, v, x0, x1, y0, y1):
    return [(origin[0] + u[0]*x + v[0]*y, origin[1] + u[1]*x + v[1]*y)
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def records(loops):
    result = []
    for loop in loops:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            result.append(dict(a=tuple(map(F, a)), b=tuple(map(F, b)), curve=0,
                               source=("plain", "wire", len(result))))
    return result


class ExactHoleSeparationTests(unittest.TestCase):
    def test_strict_gap_before_and_after_rotation_and_translation(self):
        # The face-to-face gap is dot(u, delta)/|u|. Each guard consumes 2 nm.
        rotations = (
            ((1, 0), (0, 1), ((7, 0), (8, 0), (9, 0)), (F(7), F(8), F(9))),
            ((3, 4), (-4, 3), ((1, 9), (0, 10), (3, 8)), (F(39, 5), F(8), F(41, 5))),
            ((4, 3), (-3, 4), ((9, 1), (10, 0), (8, 3)), (F(39, 5), F(8), F(41, 5))),
        )
        for t in (0, 10**6, 10**8, 10**10, 999999*10**6, 9999990*10**6):
            origin = (t, t)
            outside = rectangle(origin, (1, 0), (0, 1), -2000000, 2000000, -2000000, 2000000)
            for u, v, shifts, gaps in rotations:
                left = rectangle(origin, u, v, -80000, 0, -70000, 70000)
                for (dx, dy), gap in zip(shifts, gaps):
                    with self.subTest(origin=t, normal=u, gap=gap):
                        self.assertEqual((u[0]*dx + u[1]*dy)**2, gap**2 * (u[0]**2 + u[1]**2))
                        right = rectangle((t+dx, t+dy), u, v, 0, 80000, -70000, 70000)
                        source = records([outside, left, right])
                        if gap <= 8:
                            with self.assertRaisesRegex(ValueError, "unresolved separation"):
                                nominal_board_domain(source)
                        else:
                            _, info = nominal_board_domain(source)
                            self.assertEqual(info["hole_count"], 2)

    def test_concave_interlocking_holes_and_nested_holes(self):
        outside = [(-500, -500), (500, -500), (500, 500), (-500, 500)]
        left = [(-400, -200), (100, -200), (100, -100), (-200, -100),
                (-200, 100), (100, 100), (100, 200), (-400, 200)]
        # The convex hulls of the holes intersect, but the finger clears both arms.
        finger = [(-100, -40), (300, -40), (300, 40), (-100, 40)]
        _, info = nominal_board_domain(records([outside, left, finger]))
        self.assertEqual(info["hole_count"], 2)
        nested = [(-360, -60), (-240, -60), (-240, 60), (-360, 60)]
        with self.assertRaisesRegex(ValueError, "unresolved separation"):
            nominal_board_domain(records([outside, left, nested]))
