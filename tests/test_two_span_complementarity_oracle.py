"""Independent exact six-DOF element assembly for signed two-span contact."""
import itertools
import unittest
from fractions import Fraction as F
from hamon_dual_verifier import run_dual_verify_spec


def solve_linear(a, b):
    # Eliminate against the augmented stiffness equations, keeping all pivots exact.
    n = len(b)
    rows = [list(row)+[rhs] for row,rhs in zip(a,b)]
    for k in range(n):
        p = next(i for i in range(k,n) if rows[i][k])
        rows[k],rows[p] = rows[p],rows[k]
        pivot = rows[k][k]
        rows[k] = [v/pivot for v in rows[k]]
        for i in range(n):
            if i != k:
                h = rows[i][k]
                rows[i] = [v-h*w for v,w in zip(rows[i],rows[k])]
    return [r[-1] for r in rows]


def reference(lengths, loads, ei, unilateral):
    k = [[F(0) for _ in range(6)] for _ in range(6)]
    f = [F(0) for _ in range(6)]
    for i,(l,q) in enumerate(zip(lengths,loads)):
        ke = [[12,6*l,-12,6*l],[6*l,4*l*l,-6*l,2*l*l],
              [-12,-6*l,12,-6*l],[6*l,2*l*l,-6*l,4*l*l]]
        fe = [-q*l/2,-q*l*l/12,-q*l/2,q*l*l/12]
        for a in range(4):
            f[2*i+a] += fe[a]
            for b in range(4):
                k[2*i+a][2*i+b] += ei*ke[a][b]/l**3
    candidates = set()
    for flags in itertools.product((False,True), repeat=len(unilateral)):
        fixed = {2*i for i in range(3) if i not in unilateral}
        fixed.update(2*i for i,on in zip(unilateral,flags) if on)
        if len(fixed) < 2:
            continue
        free = [i for i in range(6) if i not in fixed]
        d = [F(0) for _ in range(6)]
        u = solve_linear([[k[i][j] for j in free] for i in free],[f[i] for i in free])
        for i,value in zip(free,u):
            d[i] = value
        reactions = [sum(k[2*i][j]*d[j] for j in range(6))-f[2*i] for i in range(3)]
        vertical = d[::2]
        if all(reactions[i]>=0 and vertical[i]>=0 and reactions[i]*vertical[i]==0 for i in unilateral):
            candidates.add((tuple(reactions),tuple(vertical)))
    return candidates


class TwoSpanComplementarityOracle(unittest.TestCase):
    def test_nonuniform_spans_signed_loads_and_mixed_bearings(self):
        count = 0
        for lengths, loads in [((2,3),(8,2)), ((3,2),(0,12)), ((2,5),(-3,7)), ((5,3),(1,-9))]:
            for bearings in [(),(0,),(1,),(2,),(0,2),(1,2),(0,1,2)]:
                with self.subTest(lengths=lengths,loads=loads,bearings=bearings):
                    ll,qq = tuple(map(F,lengths)),tuple(map(F,loads))
                    candidates = reference(ll,qq,F(128),bearings)
                    spec = dict(span1=lengths[0],span2=lengths[1],udl1=loads[0],udl2=loads[1],
                                E=256,I=.5,support_conditions={"ABC"[i]:"compression_only" for i in bearings})
                    payload = dict(mode="domain",kind="continuous_beam_2span",spec=spec)
                    if not candidates:
                        with self.assertRaises(ValueError):
                            run_dual_verify_spec(payload)
                        continue
                    self.assertEqual(len(candidates),1)
                    reactions, gap = next(iter(candidates))
                    # Cases with at least two bilateral/strict contact pins have
                    # no remaining free rigid translation or tilt.
                    if sum(i not in bearings or reactions[i]>0 for i in range(3))<2:
                        continue
                    r = run_dual_verify_spec(payload)
                    self.assertTrue(r["dual_verified"],r)
                    for i,n in enumerate("ABC"):
                        self.assertEqual(r[f"R_{n}_N"],float(reactions[i]))
                        self.assertEqual(r["support_displacements_m"][n],float(gap[i]))
                        expected = ("hold_down" if reactions[i]<0 else "bilateral") if i not in bearings else (
                            "bearing" if reactions[i]>0 else "lifted" if gap[i]>0 else "touching")
                        self.assertEqual(r["support_states"][n],expected)
                    middle = reactions[0]*ll[0]-qq[0]*ll[0]**2/2
                    moments = [F(0),middle]
                    for l,q,force in zip(ll,qq,(reactions[0],reactions[2])):
                        if q and 0<force/q<l:
                            moments.append(force**2/(2*q))
                    self.assertEqual(r["M_B_Nm"],float(middle))
                    self.assertEqual(r["M_max_Nm"],float(max(map(abs,moments))))
                    count += 1
        self.assertGreaterEqual(count,14)
