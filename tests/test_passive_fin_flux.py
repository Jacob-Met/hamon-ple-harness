"""Cold passive-fin boundary checks through the released PDE router."""
from decimal import Decimal, localcontext
import math
import unittest

import numpy as np
from scipy.linalg import solve_banded

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(length, base, tip_bc, tip_ratio, query, **claim):
    # kA = hP = 1, so the fin constant m and the base admittance are both 1.
    spec = dict(mode="convective_fin_1d", tip_bc=tip_bc, k_w_mk=2., area_m2=.5,
                perimeter_m=.5, h_w_m2k=2., h_tip_w_m2k=2*tip_ratio,
                length_m=length, t_base_c=base, t_inf_c=0., x_eval_m=query, **claim)
    return run_router_conservation_cad_verifier({"mode": "pde", "spec": spec})


def boundary_reference(length, base, ratio, query):
    # Direct high-precision hyperbolic solution to T''=T, -T'(L)=ratio*T(L).
    with localcontext() as ctx:
        ctx.prec = 65
        z, temp, beta, x = (Decimal(str(v)) for v in (length, base, ratio, query))
        def hyperbolic(v):
            pos, neg = v.exp(), (-v).exp()
            return (pos+neg)/2, (pos-neg)/2
        c, s = hyperbolic(z)
        cx, sx = hyperbolic(z-x)
        norm = c + beta*s
        return tuple(float(temp*v/norm) for v in (s + beta*c, cx + beta*sx, 1))


class PassiveFinTests(unittest.TestCase):
    def test_long_short_and_mixed_tip_boundaries_use_their_own_heat_balance(self):
        for length in (0.003, 1., 40., 120., 1000., 10000.):
            for boundary, beta in (("adiabatic", 0.), ("convective", 0.),
                                   ("convective", .037), ("convective", 200.),
                                   ("infinite", 1.)):
                for base in (-24., 80.):
                    with self.subTest(length=length, tip=boundary, beta=beta, base=base):
                        x = min(1.75, length*.75)
                        q, tx, tip = boundary_reference(length, base, beta, x)
                        r = solve(length, base, boundary, beta, x, claimed_field="q_fin_w",
                                  claimed_value=q)
                        self.assertTrue(r["passed"], r)
                        for field, expected in (("q_fin_w", q), ("t_eval_c", tx), ("t_tip_c", tip)):
                            self.assertTrue(math.isclose(r[field], expected, rel_tol=2e-13,
                                                         abs_tol=1e-300), (field, expected, r[field]))
                        self.assertLess(r["fvm_rel_error"], 1e-3)
                        self.assertFalse(solve(length, base, boundary, beta, x,
                                               claimed_field="q_fin_w", claimed_value=-q)["passed"])

    def test_condensed_cell_flux_matches_separately_assembled_finite_volume_grid(self):
        # Assemble every nodal balance, including both half-cell sink terms.
        length = 6.
        n = 512
        dx = length / n
        for beta in (0., .3, 1., 40.):
            matrix = np.zeros((3, n+1))
            matrix[1, 0] = 1.
            matrix[1, 1:n] = 2/dx+dx
            matrix[1, -1] = 1/dx+dx/2+beta
            matrix[0, 2:] = -1/dx
            matrix[2, :-1] = -1/dx
            rhs = np.zeros(n+1)
            rhs[0] = 70.
            state = solve_banded((1, 1), matrix, rhs)
            q = dx*(state[0]/2 + state[1:-1].sum() + state[-1]/2) + beta*state[-1]
            result = solve(length, 70., "convective", beta, length)
            self.assertTrue(result["passed"], result)
            self.assertAlmostEqual(result["q_fvm_w"], q, delta=2e-9)
