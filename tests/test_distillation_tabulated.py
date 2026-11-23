import math
import unittest

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


TABLE = [[0, 0], [.1, .2], [.3, .4], [.5, .75], [.7, .78], [.9, .95], [1, 1]]


def solve(**fields):
    params = {"x_B": .1, "z_F": .5, "x_D": .9, "equilibrium_points": TABLE, **fields}
    return dispatch_wave7_verifier("electrochem_separations", params)


def staircase(table, q, reflux):
    # Independent double-precision receiver: find horizontal contacts by
    # bisection and the physical feed switch by intersecting two straight lines.
    def equilibrium(x):
        for (a, ya), (b, yb) in zip(table, table[1:]):
            if x <= b:
                return ya + (yb-ya)*(x-a)/(b-a)
        raise AssertionError(x)

    d = .5
    s = ((reflux+1)*d+q-d)/((reflux+1)*d+q-1)
    r = reflux/(reflux+1)
    b = .9*(1-r)
    contact = (b-(1-s)*.1)/(s-r)
    x, y, feed = .9, .9, None
    for count in range(1, 401):
        lo, hi = 0., 1.
        for _ in range(58):
            mid = (lo+hi)*.5
            if equilibrium(mid) < y:
                lo = mid
            else:
                hi = mid
        prev, x = x, (lo+hi)*.5
        if x <= contact and feed is None:
            feed = count
        if x <= .1:
            return count, count-1+(prev-.1)/(prev-x), feed or count
        y = r*x+b if x >= contact else s*x+(1-s)*.1
    raise AssertionError("receiver exceeded its stage budget")


class TabulatedDistillation(unittest.TestCase):
    def test_hidden_rectifier_knot_and_limits(self):
        stock = dispatch_wave7_verifier("electrochem_separations",
                dict(x_B=.1, z_F=.5, x_D=.9, alpha=3, R=1))
        self.assertTrue(stock["verified"])
        self.assertEqual(stock["R_min"], .6)
        for r, gap in [(1,-.02), (1.5,0), (2,1/75)]:
            row = solve(R=r, alpha=3)
            self.assertEqual(row["R_min"], 1.5)
            self.assertEqual(row["limiting_composition"], .7)
            self.assertEqual(row["min_gap_composition"], .7)
            self.assertAlmostEqual(row["min_operating_gap"], gap)
            self.assertEqual(row["verified"], r>1.5)
            self.assertNotIn("underwood_theta", row)
        row = solve(q=0, R=3)
        self.assertFalse(row["verified"])
        self.assertEqual((row["R_min"], row["limiting_composition"]), (3, .3))

    def test_feed_crossing_and_independent_staircase(self):
        at = solve(q=5, R=1)
        self.assertEqual(at["R_min"], 9/11)
        self.assertEqual(at["limiting_composition"], 31/40)
        for q, threshold in [(-1,5), (0,3), (.25,2.5), (.5,2), (1,1.5), (2,1.5), (5,9/11)]:
            with self.subTest(q=q):
                below = solve(q=q, R=threshold-.01)
                self.assertFalse(below["verified"])
                self.assertLess(below["min_operating_gap"], 0)
                r = threshold + 1
                actual = solve(q=q, R=r)
                self.assertTrue(actual["verified"], actual)
                n, fraction, feed = staircase(TABLE, q, r)
                self.assertEqual(actual["N_stages_integer"], n)
                self.assertAlmostEqual(actual["N_theoretical"], fraction, places=11)
                self.assertEqual(actual["feed_stage"], feed)

    def test_azeotrope_needle_and_multiple_inverses(self):
        for y in (.695, .70):
            table = [[0,0],[.1,.21],[.69,.8],[.7,y],[.71,.82],[.9,.96],[1,1]]
            self.assertEqual(solve(equilibrium_points=table, R=100)["error"],
                             "separation crosses a total-reflux equilibrium barrier")
        for first, second in [(.92,.88),(.90,.90)]:
            table = [[0,0],[.1,.2],[.4,first],[.6,second],[.9,.97],[1,1]]
            row = solve(equilibrium_points=table, R=25)
            self.assertGreater(row["min_operating_gap"], 0)
            self.assertEqual(row["error"], "ambiguous_equilibrium_inverse")
        table = [[0,0],[.01,.015],[.02,.012],*TABLE[1:]]
        self.assertTrue(solve(equilibrium_points=table, R=2)["verified"])
        table = [[0,0],[.1,.2],[.5,.75],[.699999,.82],[.7,.779999],
                 [.700001,.82],[.9,.95],[1,1]]
        row = solve(equilibrium_points=table, R=1.5)
        self.assertLess(row["min_operating_gap"], 0)
        self.assertEqual(row["min_gap_composition"], .7)

    def test_table_refinement_and_three_feed_line_crossings(self):
        stock = dispatch_wave7_verifier("electrochem_separations",
                dict(x_B=.1, z_F=.5, x_D=.9, alpha=3, q=.6, R=2))
        errors = []
        for n in (16, 64, 256):
            table = [[i/n, 3*(i/n)/(1+2*(i/n))] for i in range(n+1)]
            row = solve(equilibrium_points=table, q=.6, R=2)
            self.assertEqual((row["N_stages_integer"], row["feed_stage"]), (6,3))
            errors.append((row["R_min"]-stock["R_min"],
                           row["N_theoretical"]-stock["N_theoretical"]))
        for values in zip(*errors):
            self.assertTrue(values[0] > values[1] > values[2] > 0)
            self.assertLess(values[2], .0002)
        table = [[0,0],[.1,.23],[.5,.65],[.6,.69],[.65,.70],[.7,.85],
                 [.8,.90],[.9,.96],[1,1]]
        row = solve(equilibrium_points=table, q=3, R=4)
        self.assertEqual((row["R_min"], row["min_gap_composition"], row["min_operating_gap"]),
                         (4, .65, 0))
        self.assertFalse(row["verified"])
        row = solve(equilibrium_points=table, q=3, R=5)
        self.assertTrue(row["verified"])
        n, fractional, feed = staircase(table, 3, 5)
        self.assertEqual((row["N_stages_integer"], row["feed_stage"]), (n,feed))
        self.assertAlmostEqual(row["N_theoretical"], fractional, places=11)

    def test_flow_floor_flash_and_single_section(self):
        args = dict(x_B=.4,z_F=.5,x_D=.6,equilibrium_points=[[0,0],[.4,.6],[.6,.8],[1,1]])
        flash = solve(**args, q=.5, R=0)
        self.assertEqual((flash["section_mode"], flash["N_stages_integer"]),
                         ("single_equilibrium_flash",1))
        self.assertEqual(solve(**args, q=0, R=1)["section_mode"],"rectifying_only")
        self.assertEqual(solve(**args, q=1, R=0)["section_mode"],"stripping_only")
        self.assertEqual(solve(**args, q=0, R=.9)["error"],"negative internal section flow")
        args["equilibrium_points"] = [[0,0],[.4,.61],[.6,.8],[1,1]]
        self.assertEqual(solve(**args, q=.5, R=0)["error"],
                         "zero-circulation flash products are not in equilibrium")

    def test_malformed_and_partial_table(self):
        for table in [None, [], [[.2,.3],[.9,.95]], [[0,0],[.7,.7],[.6,.8]],
                      [[0,0],[1,math.nan]], [[0,0],[True,1]], [[0,0],[1,2]]]:
            row = solve(equilibrium_points=table, alpha=3)
            self.assertFalse(row["verified"])
            self.assertEqual(row["equilibrium_model"], "piecewise_linear_table")
        self.assertFalse(solve(equilibrium_table=TABLE, R=2)["verified"])
        row = solve(equilibrium_points=[[.1,.29],[.5,.75],[.9,.95]], R=5)
        self.assertEqual(row["error"],"stage_outside_equilibrium_table")


if __name__ == "__main__":
    unittest.main()
