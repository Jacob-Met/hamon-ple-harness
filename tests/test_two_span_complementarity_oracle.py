"""Independent exact six-DOF element assembly for signed two-span contact."""
import itertools
import math
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


def reference(lengths, loads, ei, unilateral, levels=(F(0), F(0), F(0))):
    k = [[F(0) for _ in range(6)] for _ in range(6)]
    f = [F(0) for _ in range(6)]
    for i,(l,q) in enumerate(zip(lengths,loads)):
        member_ei = ei[i] if isinstance(ei, tuple) else ei
        ke = [[12,6*l,-12,6*l],[6*l,4*l*l,-6*l,2*l*l],
              [-12,-6*l,12,-6*l],[6*l,2*l*l,-6*l,4*l*l]]
        fe = [-q*l/2,-q*l*l/12,-q*l/2,q*l*l/12]
        for a in range(4):
            f[2*i+a] += fe[a]
            for b in range(4):
                k[2*i+a][2*i+b] += member_ei*ke[a][b]/l**3
    candidates = set()
    for flags in itertools.product((False,True), repeat=len(unilateral)):
        fixed = {2*i for i in range(3) if i not in unilateral}
        fixed.update(2*i for i,on in zip(unilateral,flags) if on)
        if len(fixed) < 2:
            continue
        free = [i for i in range(6) if i not in fixed]
        d = [F(0) for _ in range(6)]
        for j in fixed:
            d[j] = levels[j // 2]
        rhs = [f[i] - sum(k[i][j]*d[j] for j in fixed) for i in free]
        u = solve_linear([[k[i][j] for j in free] for i in free], rhs)
        for i,value in zip(free,u):
            d[i] = value
        reactions = [sum(k[2*i][j]*d[j] for j in range(6))-f[2*i] for i in range(3)]
        vertical = d[::2]
        gaps = [v-z for v,z in zip(vertical,levels)]
        if all(reactions[i]>=0 and gaps[i]>=0 and reactions[i]*gaps[i]==0 for i in unilateral):
            candidates.add((tuple(reactions),tuple(vertical)))
    return candidates


class TwoSpanComplementarityOracle(unittest.TestCase):
    def test_prescribed_levels_against_partitioned_stiffness(self):
        lengths = F(2), F(3)
        rigidity = F(128), F(384)
        cases = 0
        states = set()
        for loads in [(F(8), F(2)), (F(-3), F(7)), (F(0), F(0))]:
            for levels in [(F(0), -F(1, 16), F(0)), (F(1, 8), F(0), -F(1, 4))]:
                # Two bilateral supports always restrain rigid translation and tilt.
                for bearings in [(), (0,), (1,), (2,)]:
                    with self.subTest(loads=loads, levels=levels, bearings=bearings):
                        expected, = reference(lengths, loads, rigidity, bearings, levels)
                        forces, displacements = expected
                        spec = dict(span1=2, span2=3, span_EI=[128, 384],
                                    udl1=float(loads[0]), udl2=float(loads[1]),
                                    support_conditions={"ABC"[i]: "compression_only" for i in bearings},
                                    support_settlements_m=dict(zip("ABC", [-float(z) for z in levels])))
                        result = run_dual_verify_spec(dict(mode="domain", kind="continuous_beam_2span", spec=spec))
                        self.assertTrue(result["dual_verified"])
                        self.assertEqual(result["M_B_Nm"], float(forces[0]*lengths[0] - loads[0]*lengths[0]**2/2))
                        for i, name in enumerate("ABC"):
                            self.assertEqual(result[f"R_{name}_N"], float(forces[i]))
                            self.assertEqual(result["support_displacements_m"][name], float(displacements[i]))
                            self.assertEqual(result["support_gaps_m"][name], float(displacements[i]-levels[i]))
                            states.add(result["support_states"][name])
                        cases += 1
        self.assertEqual(cases, 24)
        self.assertTrue({"lifted", "bearing", "hold_down"}.issubset(states))

    def test_settled_bearing_boundary(self):
        # An end-supported, level beam under uniform load deflects 5*q*L**4/(384*EI)
        # at midspan. Here this is exactly 1/16 m. Change the bearing depth by one ULP.
        free = 1/16
        for depth, state in [(math.nextafter(free, 0), "bearing"), (free, "touching"),
                             (math.nextafter(free, math.inf), "lifted")]:
            with self.subTest(depth=depth):
                spec = dict(span1=2, span2=2, udl1=24, udl2=24, span_EI=[1280, 1280],
                            support_settlements_m={"B": depth}, support_conditions={"B": "compression_only"})
                result = run_dual_verify_spec(dict(mode="domain", kind="continuous_beam_2span", spec=spec))
                self.assertEqual(result["support_states"]["B"], state)
                self.assertEqual(result["support_displacements_m"]["B"], -min(depth, free))
                self.assertEqual(result["support_gaps_m"]["B"], max(0., depth-free))

    def test_settlement_units_and_rigid_tilt(self):
        # These supports lie on a straight line, so the unloaded beam follows them stress free.
        spec = dict(span1=2, span2=3, span_EI=[128, 384],
                    support_settlements_m={"A": "125*mm", "B": "-125*mm", "C": "-500*mm"})
        payload = dict(mode="domain", kind="continuous_beam_2span", spec=spec)
        result = run_dual_verify_spec(payload)
        self.assertEqual(result["support_displacements_m"], dict(A=-.125, B=.125, C=.5))
        self.assertEqual([result[f"R_{n}_N"] for n in "ABC"], [0., 0., 0.])
        for invalid in ({"D": 0}, {"A": "1*N"}, [0, 0, 0], None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "support_settlements_m"):
                run_dual_verify_spec(dict(payload, spec=dict(spec, support_settlements_m=invalid)))



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

    def test_variable_span_rigidity_against_assembled_contact_system(self):
        # Length, load and rigidity of span AB and BC change together on reversal.
        for lengths, loads, stiffness in [
            ((2, 2), (1000, 150), (10_000_000, 40_000_000)),
            ((2, 2), (19, 4), (11, 44)),  # exact touching at the right end
            ((2, 5), (7, -3), (128, 11)),
            ((2, 3), (-2, -7), (17, 83)),
        ]:
            for order in (slice(None), slice(None, None, -1)):
                ll, qq, ei = (tuple(map(F, values[order])) for values in (lengths, loads, stiffness))
                for bearings in [(), (0,), (1,), (2,), (0, 2), (0, 1, 2)]:
                    with self.subTest(lengths=ll, loads=qq, rigidities=ei, bearings=bearings):
                        spec = dict(span1=int(ll[0]), span2=int(ll[1]), udl1=int(qq[0]),
                                    udl2=int(qq[1]), span_EI=list(map(int, ei)),
                                    support_conditions={"ABC"[i]: "compression_only" for i in bearings})
                        payload = dict(mode="domain", kind="continuous_beam_2span", spec=spec)
                        candidates = reference(ll, qq, ei, bearings)
                        if not candidates:
                            with self.assertRaises(ValueError):
                                run_dual_verify_spec(payload)
                            continue
                        self.assertEqual(len(candidates), 1)
                        forces, gaps = next(iter(candidates))
                        result = run_dual_verify_spec(payload)
                        self.assertTrue(result["dual_verified"])
                        self.assertEqual(result["M_B_Nm"], float(forces[0]*ll[0] - qq[0]*ll[0]**2/2))
                        for i, name in enumerate("ABC"):
                            self.assertEqual(result[f"R_{name}_N"], float(forces[i]))
                            self.assertEqual(result["support_displacements_m"][name], float(gaps[i]))
                            if i in bearings:
                                state = "bearing" if forces[i] else "lifted" if gaps[i] else "touching"
                                self.assertEqual(result["support_states"][name], state)

    def test_unequal_rigidity_touching_and_adjacent_binary64_loads(self):
        for load, expected in ((math.nextafter(19., 0.), "bearing"), (19., "touching"),
                               (math.nextafter(19., math.inf), "lifted")):
            spec = dict(span1=2, span2=2, udl1=load, udl2=4, span_EI=[11, 44],
                        support_conditions={"C":"compression_only"})
            result = run_dual_verify_spec(dict(mode="domain", kind="continuous_beam_2span", spec=spec))
            (forces, gaps), = reference((F(2), F(2)), (F(load), F(4)), (F(11), F(44)), (2,))
            self.assertEqual(result["support_states"]["C"], expected)
            self.assertEqual(result["R_C_N"], float(forces[2]))
            self.assertEqual(result["support_displacements_m"]["C"], float(gaps[2]))

    def test_span_rigidity_input_contract_and_units(self):
        spec = dict(span1=2, span2=2, udl1=1000, udl2=150, support_conditions={"C":"compression_only"})
        def solve(**kwargs):
            return run_dual_verify_spec(dict(mode="domain", kind="continuous_beam_2span", spec={**spec, **kwargs}))

        self.assertEqual(solve(E=20e6, I=.5), solve(span_EI=[10e6, 10e6]))
        self.assertEqual(solve(), solve(span_EI=[20e6, 20e6]))
        result = solve(span_EI=["10*MN*m**2", "40000*kN*m**2"])
        self.assertEqual(result, solve(span_EI=[10e6, 40e6]))
        self.assertEqual(result["support_states"]["C"], "lifted")
        self.assertEqual(result["support_displacements_m"]["C"], float(F(23, 1_200_000)))
        for bad in ([], [1], [1, 3, 5], [1, 0], [-1, 1], [True, 1], [1, "2*Pa"], "1,2", [1, None]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                solve(span_EI=bad)
        for other in ({"E":20e6}, {"I":.5}):
            with self.assertRaisesRegex(ValueError, "span_EI"):
                solve(span_EI=[1, 2], **other)
