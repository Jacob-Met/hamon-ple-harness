"""Independent pressure-volume paths and simultaneous observations of one Diesel volume."""

import math
import unittest
from fractions import Fraction

from mpmath import mp

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def cycle(kind, **fields):
    spec = dict(subkind=kind, r=16, T1=321, gamma=1.25, cv=733, P1=94321)
    return run_router_conservation_cad_verifier(
        {"mode": "thermo_cycles", "spec": dict(spec, **fields)}
    )


def observation_cell(value):
    center = Fraction(value)
    if isinstance(value, int):
        return (center, center)
    return tuple((center + Fraction(math.nextafter(value, toward))) / 2
                 for toward in (-math.inf, math.inf))


class AirStandardVolumeOracle(unittest.TestCase):
    def test_three_observations_select_a_common_off_center_diesel_volume(self):
        # Use V1=1, V2=1/16. The pressure of the heating stroke is P2.
        # Its ideal-gas temperature is T3=642*16*V3; supplied heat is
        # cp*(642*16*V3 - 642). Intersect observations in V3/V1.
        cp = Fraction(3665, 4)
        q0 = 245669.53125
        qm1 = math.nextafter(q0, -math.inf)
        qm2 = math.nextafter(qm1, -math.inf)
        qm3 = math.nextafter(qm2, -math.inf)
        observations = dict(T3=910.125, r_c=1.4176401869158877)
        for q, feasible in ((qm1, True), (qm2, True), (qm3, False),
                            (245669, False), (-1, False)):
            with self.subTest(q=q):
                constraints = [(observations["T3"], lambda x: x / (642*16)),
                               (observations["r_c"], lambda x: x / 16),
                               (q, lambda x: (642+x/cp) / (642*16))]
                lows, highs, centers = [], [], []
                for value, volume in constraints:
                    lo, hi = observation_cell(value)
                    lows.append(volume(lo))
                    highs.append(volume(hi))
                    centers.append(volume(Fraction(value)))
                low, high = max(lows), min(highs)
                self.assertNotEqual(low, high)
                self.assertEqual(low < high, feasible)
                if feasible:
                    self.assertFalse(any(low <= x <= high for x in centers))
                ans = cycle("diesel", q_in=q, **observations)
                self.assertEqual(ans["verified"], feasible, ans)
                if feasible:
                    for field, expected in (("T3_K", observations["T3"]),
                                            ("cutoff_ratio", observations["r_c"]),
                                            ("q_in_J_kg", q)):
                        self.assertEqual(ans[field], expected)

    def test_original_pressure_integrals_and_diesel_inverse_inputs(self):
        with mp.workdps(90):
            for kind in ("otto", "diesel"):
                for r, gamma, t3 in ((16, 1.25, 1250), (8, 1.625, 1900)):
                    g = mp.mpf(gamma)
                    t1, p1, cv = map(mp.mpf, (321, 94321, 733))
                    gas = cv*(g-1)
                    v1 = gas*t1/p1
                    v2 = v1/r
                    p2 = p1*(v1/v2)**g
                    t2 = p2*v2/gas
                    v3 = v2 if kind == "otto" else gas*t3/p2
                    p3 = gas*t3/v3
                    t4 = (p3*(v3/v1)**g)*v1/gas
                    compression = mp.quad(lambda v: p1*(v1/v)**g, [v1, v2])
                    expansion = mp.quad(lambda v: p3*(v3/v)**g, [v3, v1])
                    heating = (p2*(v3-v2)) if kind == "diesel" else 0
                    work = compression+heating+expansion
                    qin = cv*(t3-t2)+heating
                    heat_rejected = cv*(t4-t1)
                    inputs = [dict(T3=t3), dict(q_in=float(qin))]
                    if kind == "diesel":
                        inputs += [dict(r_c=float(v3/v2)),
                                   dict(T3=t3, q_in=float(qin), r_c=float(v3/v2))]
                    for extra in inputs:
                        with self.subTest(kind=kind, r=r, gamma=gamma, extra=extra):
                            ans = cycle(kind, r=r, gamma=gamma, **extra)
                            self.assertTrue(ans["verified"], ans)
                            for field, reference in (("T2_K", t2), ("T3_K", t3),
                                    ("T4_K", t4), ("w_net_J_kg", work),
                                    ("q_in_J_kg", qin), ("q_out_J_kg", heat_rejected),
                                    ("eta_thermal", work/qin), ("MEP_Pa", work/(v1-v2))):
                                self.assertLess(abs(mp.mpf(ans[field])/reference-1),
                                                mp.mpf("2e-15"), field)

    def test_diesel_original_volume_contacts_exclude_zero_length_strokes(self):
        for rc, allowed in ((math.nextafter(1., -math.inf), False), (1, False),
                            (math.nextafter(1., math.inf), True),
                            (math.nextafter(16., -math.inf), True), (16, False),
                            (math.nextafter(16., math.inf), False)):
            with self.subTest(rc=rc):
                result = cycle("diesel", r_c=rc)
                self.assertEqual(result["verified"], allowed, result)
                if allowed:
                    self.assertEqual(result["cutoff_ratio"], rc)
