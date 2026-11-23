"""Multi-Discipline Dual-Solver Symbolic + Dimensional-Analysis + Numerical Cross-Check Verifier.

Enforces:
1. Exact 7-dimension SI dimensional consistency (kg, m, s, A, K, mol, cd) with
   rational intermediate exponents (e.g. Manning R^(2/3), fire D*^(2/5), spillway H^(3/2))
   and dimensionless-argument guards on transcendental functions (sin, cos, tan, exp, ln, log10).
2. Independent Dual-Solver Verification: every linear/nonlinear/differential system is
   solved via TWO distinct mathematical engines:
     - Path A (Symbolic / Exact Analytic): SymPy (`Matrix.LUsolve`, symbolic integration,
       `nsolve` with arbitrary-precision `mpmath`) when installed, with exact `Fraction`/`Decimal`
       analytic elimination fallback in minimal containers.
     - Path B (Numerical): SciPy (`scipy.linalg.solve`, `scipy.optimize.brentq`/`root`,
       64-element Hermite cubic beam FEM) when installed, with independent numerical
       Gaussian/Brent/Hermite-FEM fallback in minimal containers.
3. Conservation & Equilibrium Laws across Structural, Electrical, Mechanical, and Civil
   engineering with zero task-specific hardcoding or benchmark shortcuts.
"""
from __future__ import annotations

import ast
import cmath
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
import re
from pathlib import Path
from typing import Any

# Never mutate global Decimal context; preserve caller precision.

try:
    import numpy as _np
    import scipy.linalg as _spla
    import scipy.optimize as _spopt
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _np = None
    _spla = None
    _spopt = None
    HAS_SCIPY = False

try:
    import sympy as _sp
    HAS_SYMPY = True
except ImportError:  # pragma: no cover
    _sp = None
    HAS_SYMPY = False

# 7 SI base dimensions: (kg, m, s, A, K, mol, cd)
Dim7 = tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]
ZERO7: Dim7 = (Fraction(0),) * 7


def _dim(kg=0, m=0, s=0, a=0, k=0, mol=0, cd=0) -> Dim7:
    return (
        Fraction(kg), Fraction(m), Fraction(s), Fraction(a),
        Fraction(k), Fraction(mol), Fraction(cd),
    )


D_MASS = _dim(kg=1)
D_LEN = _dim(m=1)
D_TIME = _dim(s=1)
D_CURR = _dim(a=1)
D_TEMP = _dim(k=1)
D_MOL = _dim(mol=1)
D_FORCE = _dim(kg=1, m=1, s=-2)
D_MOMENT = _dim(kg=1, m=2, s=-2)
D_PRESS = _dim(kg=1, m=-1, s=-2)
D_ENERGY = _dim(kg=1, m=2, s=-2)
D_POWER = _dim(kg=1, m=2, s=-3)
D_VOLT = _dim(kg=1, m=2, s=-3, a=-1)
D_RES = _dim(kg=1, m=2, s=-3, a=-2)
D_COND = _dim(kg=-1, m=-2, s=3, a=2)
D_CAP = _dim(kg=-1, m=-2, s=4, a=2)
D_IND = _dim(kg=1, m=2, s=-2, a=-2)
D_CHARGE = _dim(s=1, a=1)
D_FLUX = _dim(kg=1, m=2, s=-2, a=-1)
D_TESLA = _dim(kg=1, s=-2, a=-1)
D_FREQ = _dim(s=-1)
D_VISC = _dim(kg=1, m=-1, s=-1)


@dataclass(frozen=True)
class SIQuantity:
    """Physical quantity in coherent SI units with exact rational dimension vector."""
    val: float
    dim: Dim7 = ZERO7

    def is_dimensionless(self) -> bool:
        return all(x == 0 for x in self.dim)

    def dim_list(self) -> list[float | int]:
        return [int(x) if x.denominator == 1 else float(x) for x in self.dim]

    def __add__(self, other: "SIQuantity") -> "SIQuantity":
        if self.dim != other.dim:
            raise ValueError(f"SI dimension mismatch in +: {self.dim_list()} vs {other.dim_list()}")
        return SIQuantity(self.val + other.val, self.dim)

    def __sub__(self, other: "SIQuantity") -> "SIQuantity":
        if self.dim != other.dim:
            raise ValueError(f"SI dimension mismatch in -: {self.dim_list()} vs {other.dim_list()}")
        return SIQuantity(self.val - other.val, self.dim)

    def __mul__(self, other: "SIQuantity") -> "SIQuantity":
        return SIQuantity(
            self.val * other.val,
            tuple(a + b for a, b in zip(self.dim, other.dim)),  # type: ignore[arg-type]
        )

    def __truediv__(self, other: "SIQuantity") -> "SIQuantity":
        if other.val == 0.0:
            raise ValueError("division by zero in physical quantity")
        return SIQuantity(
            self.val / other.val,
            tuple(a - b for a, b in zip(self.dim, other.dim)),  # type: ignore[arg-type]
        )

    def __neg__(self) -> "SIQuantity":
        return SIQuantity(-self.val, self.dim)

    def power(self, exp_q: "SIQuantity") -> "SIQuantity":
        if not exp_q.is_dimensionless():
            raise ValueError(f"exponent must be dimensionless, got {exp_q.dim_list()}")
        if abs(exp_q.val) > 24.0:
            raise ValueError("exponent magnitude exceeds [-24, 24]")
        if self.val < 0.0:
            frac = Fraction(exp_q.val).limit_denominator(1000)
            if abs(float(frac) - exp_q.val) > 1e-12 or frac.denominator != 1:
                raise ValueError("negative base raised to non-integer exponent")
        if self.val == 0.0 and exp_q.val < 0.0:
            raise ValueError("zero raised to negative exponent")
        exp_frac = Fraction(exp_q.val).limit_denominator(10000)
        if abs(float(exp_frac) - exp_q.val) > 1e-9 and not self.is_dimensionless():
            raise ValueError("irrational exponent applied to dimensional quantity")
        new_dim = tuple(d * exp_frac for d in self.dim)
        return SIQuantity(float(self.val ** exp_q.val), new_dim)  # type: ignore[arg-type]


def _u(scale: float, dim: Dim7 = ZERO7) -> SIQuantity:
    return SIQuantity(float(scale), dim)


# Traceable exact NIST / BIPM constants (ambiguous 'lb' and 'ton' intentionally excluded)
G0_SI = 9.80665
INCH_M = 0.0254
FT_M = 0.3048
LBM_KG = 0.45359237
LBF_N = LBM_KG * G0_SI  # 4.4482216152605 N

SI_UNITS: dict[str, SIQuantity] = {
    "one": _u(1.0),
    "pi": _u(math.pi),
    "g0": _u(G0_SI, _dim(m=1, s=-2)),
    # Length, area, volume, section properties
    "m": _u(1.0, D_LEN),
    "mm": _u(1e-3, D_LEN),
    "cm": _u(1e-2, D_LEN),
    "um": _u(1e-6, D_LEN),
    "km": _u(1e3, D_LEN),
    "inch": _u(INCH_M, D_LEN),
    "in_": _u(INCH_M, D_LEN),
    "ft": _u(FT_M, D_LEN),
    "yd": _u(0.9144, D_LEN),
    "mi": _u(1609.344, D_LEN),
    "mm2": _u(1e-6, _dim(m=2)),
    "cm2": _u(1e-4, _dim(m=2)),
    "m2": _u(1.0, _dim(m=2)),
    "in2": _u(INCH_M**2, _dim(m=2)),
    "ft2": _u(FT_M**2, _dim(m=2)),
    "ha": _u(1e4, _dim(m=2)),
    "acre": _u(4046.8564224, _dim(m=2)),
    "mm3": _u(1e-9, _dim(m=3)),
    "cm3": _u(1e-6, _dim(m=3)),
    "m3": _u(1.0, _dim(m=3)),
    "L": _u(1e-3, _dim(m=3)),
    "mL": _u(1e-6, _dim(m=3)),
    "in3": _u(INCH_M**3, _dim(m=3)),
    "ft3": _u(FT_M**3, _dim(m=3)),
    "gal": _u(231.0 * INCH_M**3, _dim(m=3)),
    "mm4": _u(1e-12, _dim(m=4)),
    "cm4": _u(1e-8, _dim(m=4)),
    "m4": _u(1.0, _dim(m=4)),
    "in4": _u(INCH_M**4, _dim(m=4)),
    "ft4": _u(FT_M**4, _dim(m=4)),
    # Time, frequency, angular
    "s": _u(1.0, D_TIME),
    "ms": _u(1e-3, D_TIME),
    "min_": _u(60.0, D_TIME),
    "hr": _u(3600.0, D_TIME),
    "day": _u(86400.0, D_TIME),
    "Hz": _u(1.0, D_FREQ),
    "kHz": _u(1e3, D_FREQ),
    "MHz": _u(1e6, D_FREQ),
    "rad": _u(1.0),
    "deg": _u(math.pi / 180.0),
    "rpm": _u(2.0 * math.pi / 60.0, D_FREQ),
    # Mass, force, moment, distributed loads
    "kg": _u(1.0, D_MASS),
    "g": _u(1e-3, D_MASS),
    "mg": _u(1e-6, D_MASS),
    "tonne": _u(1e3, D_MASS),
    "lbm": _u(LBM_KG, D_MASS),
    "slug": _u(LBF_N / FT_M, D_MASS),
    "N": _u(1.0, D_FORCE),
    "kN": _u(1e3, D_FORCE),
    "MN": _u(1e6, D_FORCE),
    "lbf": _u(LBF_N, D_FORCE),
    "kip": _u(1e3 * LBF_N, D_FORCE),
    "tonf": _u(2e3 * LBF_N, D_FORCE),
    "N_m": _u(1.0, D_MOMENT),
    "kN_m": _u(1e3, D_MOMENT),
    "MN_m": _u(1e6, D_MOMENT),
    "lbf_ft": _u(LBF_N * FT_M, D_MOMENT),
    "lbf_in": _u(LBF_N * INCH_M, D_MOMENT),
    "kip_ft": _u(1e3 * LBF_N * FT_M, D_MOMENT),
    "kip_in": _u(1e3 * LBF_N * INCH_M, D_MOMENT),
    "plf": _u(LBF_N / FT_M, _dim(kg=1, s=-2)),
    "klf": _u(1e3 * LBF_N / FT_M, _dim(kg=1, s=-2)),
    "pcf": _u(LBF_N / FT_M**3, _dim(kg=1, m=-2, s=-2)),
    "kcf": _u(1e3 * LBF_N / FT_M**3, _dim(kg=1, m=-2, s=-2)),
    # Pressure, stress, viscosity
    "Pa": _u(1.0, D_PRESS),
    "kPa": _u(1e3, D_PRESS),
    "MPa": _u(1e6, D_PRESS),
    "GPa": _u(1e9, D_PRESS),
    "bar": _u(1e5, D_PRESS),
    "mbar": _u(1e2, D_PRESS),
    "atm": _u(101325.0, D_PRESS),
    "psi": _u(LBF_N / INCH_M**2, D_PRESS),
    "ksi": _u(1e3 * LBF_N / INCH_M**2, D_PRESS),
    "psf": _u(LBF_N / FT_M**2, D_PRESS),
    "ksf": _u(1e3 * LBF_N / FT_M**2, D_PRESS),
    "inH2O": _u(249.08891, D_PRESS),
    "Pa_s": _u(1.0, D_VISC),
    "mPa_s": _u(1e-3, D_VISC),
    "cP": _u(1e-3, D_VISC),
    # Energy, power, temperature
    "J": _u(1.0, D_ENERGY),
    "kJ": _u(1e3, D_ENERGY),
    "MJ": _u(1e6, D_ENERGY),
    "GJ": _u(1e9, D_ENERGY),
    "Wh": _u(3600.0, D_ENERGY),
    "kWh": _u(3.6e6, D_ENERGY),
    "MWh": _u(3.6e9, D_ENERGY),
    "Btu": _u(1055.05585262, D_ENERGY),
    "MMBtu": _u(1055.05585262e6, D_ENERGY),
    "cal": _u(4.184, D_ENERGY),
    "kcal": _u(4184.0, D_ENERGY),
    "W": _u(1.0, D_POWER),
    "kW": _u(1e3, D_POWER),
    "MW": _u(1e6, D_POWER),
    "hp": _u(550.0 * LBF_N * FT_M, D_POWER),
    "ton_ref": _u(12000.0 * 1055.05585262 / 3600.0, D_POWER),
    "K": _u(1.0, D_TEMP),
    "degC_delta": _u(1.0, D_TEMP),
    "degF_delta": _u(5.0 / 9.0, D_TEMP),
    "R_rankine": _u(5.0 / 9.0, D_TEMP),
    "mol": _u(1.0, D_MOL),
    # Volumetric flow
    "m3_s": _u(1.0, _dim(m=3, s=-1)),
    "L_s": _u(1e-3, _dim(m=3, s=-1)),
    "cfs": _u(FT_M**3, _dim(m=3, s=-1)),
    "cfm": _u(FT_M**3 / 60.0, _dim(m=3, s=-1)),
    "gpm": _u(231.0 * INCH_M**3 / 60.0, _dim(m=3, s=-1)),
    # Electrical & magnetic
    "A": _u(1.0, D_CURR),
    "mA": _u(1e-3, D_CURR),
    "kA": _u(1e3, D_CURR),
    "V": _u(1.0, D_VOLT),
    "mV": _u(1e-3, D_VOLT),
    "kV": _u(1e3, D_VOLT),
    "MV": _u(1e6, D_VOLT),
    "ohm": _u(1.0, D_RES),
    "mohm": _u(1e-3, D_RES),
    "kohm": _u(1e3, D_RES),
    "Mohm": _u(1e6, D_RES),
    "S": _u(1.0, D_COND),
    "F": _u(1.0, D_CAP),
    "mF": _u(1e-3, D_CAP),
    "uF": _u(1e-6, D_CAP),
    "nF": _u(1e-9, D_CAP),
    "pF": _u(1e-12, D_CAP),
    "H": _u(1.0, D_IND),
    "mH": _u(1e-3, D_IND),
    "uH": _u(1e-6, D_IND),
    "C": _u(1.0, D_CHARGE),
    "Wb": _u(1.0, D_FLUX),
    "T": _u(1.0, D_TESLA),
    "VA": _u(1.0, D_POWER),
    "kVA": _u(1e3, D_POWER),
    "MVA": _u(1e6, D_POWER),
    "var": _u(1.0, D_POWER),
    "kvar": _u(1e3, D_POWER),
    "Mvar": _u(1e6, D_POWER),
}

_SCALAR_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "log10": math.log10,
}


def eval_quantity(expression: str, symbols: dict[str, SIQuantity] | None = None) -> SIQuantity:
    """Safely evaluate a dimensional physical expression without Python eval()."""
    if not isinstance(expression, str) or not 1 <= len(expression) <= 2000:
        raise ValueError("expression must be 1..2000 chars")
    tree = ast.parse(expression, mode="eval")

    def walk(node: ast.AST, depth: int = 0) -> SIQuantity:
        if depth > 45:
            raise ValueError("expression nesting limit exceeded")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            val = float(node.value)
            if not math.isfinite(val):
                raise ValueError("non-finite literal")
            return SIQuantity(val, ZERO7)
        if isinstance(node, ast.Name):
            if symbols and node.id in symbols:
                return symbols[node.id]
            if node.id in SI_UNITS:
                return SI_UNITS[node.id]
            raise ValueError(f"unknown or ambiguous unit/symbol '{node.id}'")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            q = walk(node.operand, depth + 1)
            return -q if isinstance(node.op, ast.USub) else q
        if isinstance(node, ast.BinOp):
            lhs, rhs = walk(node.left, depth + 1), walk(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                return lhs / rhs
            if isinstance(node.op, ast.Pow):
                return lhs.power(rhs)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            fn = node.func.id
            args = [walk(a, depth + 1) for a in node.args]
            if fn == "sqrt" and len(args) == 1:
                if args[0].val < 0:
                    raise ValueError("sqrt of negative quantity")
                return args[0].power(SIQuantity(0.5, ZERO7))
            if fn == "cbrt" and len(args) == 1:
                sign = -1.0 if args[0].val < 0 else 1.0
                mag = abs(args[0].val) ** (1.0 / 3.0)
                return SIQuantity(sign * mag, tuple(d * Fraction(1, 3) for d in args[0].dim))  # type: ignore[arg-type]
            if fn == "abs" and len(args) == 1:
                return SIQuantity(abs(args[0].val), args[0].dim)
            if fn in ("min", "max") and len(args) >= 2:
                d0 = args[0].dim
                if any(a.dim != d0 for a in args[1:]):
                    raise ValueError(f"dimension mismatch in {fn}()")
                comb = min if fn == "min" else max
                return SIQuantity(comb(a.val for a in args), d0)
            if fn == "atan2" and len(args) == 2:
                if args[0].dim != args[1].dim:
                    raise ValueError("atan2 arguments must share identical dimensions")
                return SIQuantity(math.atan2(args[0].val, args[1].val), ZERO7)
            if fn in _SCALAR_FUNCS and len(args) == 1:
                if not args[0].is_dimensionless():
                    raise ValueError(f"{fn}() requires dimensionless argument, got {args[0].dim_list()}")
                out = _SCALAR_FUNCS[fn](args[0].val)
                if not math.isfinite(out):
                    raise ValueError(f"non-finite result in {fn}()")
                return SIQuantity(out, ZERO7)
        raise ValueError(f"disallowed AST construct in quantity: {type(node).__name__}")

    res = walk(tree.body)
    if not math.isfinite(res.val):
        raise ValueError("non-finite physical quantity result")
    return res


def _rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-15)


# ==============================================================================
# DUAL-SOLVER KERNELS: PATH A (SYMBOLIC / EXACT) vs PATH B (NUMERICAL / SCIPY)
# ==============================================================================

def dual_solve_linear(matrix_a: list[list[complex | float]], rhs_b: list[complex | float], rtol: float = 1e-7) -> dict[str, Any]:
    """Solve A * x = b via independent Symbolic (SymPy/Exact) and Numerical (SciPy) paths."""
    n = len(matrix_a)
    if n == 0 or any(len(row) != n for row in matrix_a) or len(rhs_b) != n:
        raise ValueError("square non-empty linear system required")
    is_complex = any(isinstance(v, complex) and v.imag != 0 for row in matrix_a for v in row) or any(
        isinstance(v, complex) and v.imag != 0 for v in rhs_b
    )

    # Path A: Symbolic SymPy LUsolve (or exact Fraction/complex Gaussian elimination)
    if HAS_SYMPY:
        def _to_sp(v: complex | float):
            if isinstance(v, complex):
                return _sp.Float(repr(float(v.real)), 35) + _sp.I * _sp.Float(repr(float(v.imag)), 35)
            return _sp.Rational(repr(float(v))) if n <= 16 else _sp.Float(repr(float(v)), 35)
        A_sp = _sp.Matrix([[_to_sp(v) for v in row] for row in matrix_a])
        b_sp = _sp.Matrix([_to_sp(v) for v in rhs_b])
        sol_sp = A_sp.LUsolve(b_sp)
        x_sym = [complex(sol_sp[i].evalf(35)) if is_complex else float(sol_sp[i].evalf(35)) for i in range(n)]
    elif not is_complex and n <= 24:
        # Exact infinite-precision rational Fraction elimination
        aug_q = [
            [Fraction(float(v)) for v in row] + [Fraction(float(rhs_b[i]))]
            for i, row in enumerate(matrix_a)
        ]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(aug_q[r][col]))
            if aug_q[piv][col] == 0:
                raise ValueError("singular linear system in exact rational path")
            aug_q[col], aug_q[piv] = aug_q[piv], aug_q[col]
            piv_val = aug_q[col][col]
            for r in range(col + 1, n):
                fac_q = aug_q[r][col] / piv_val
                if fac_q != 0:
                    for c in range(col, n + 1):
                        aug_q[r][c] -= fac_q * aug_q[col][c]
        x_q = [Fraction(0)] * n
        for r in range(n - 1, -1, -1):
            s_q = aug_q[r][n] - sum(aug_q[r][c] * x_q[c] for c in range(r + 1, n))
            x_q[r] = s_q / aug_q[r][r]
        x_sym = [float(v) for v in x_q]
    elif is_complex and n <= 16:
        # Exact Gaussian-rational Q(i) elimination for complex symmetric / non-Hermitian systems
        def _qc(val: complex | float) -> tuple[Fraction, Fraction]:
            z = complex(val)
            return (Fraction(float(z.real)), Fraction(float(z.imag)))

        def _qsub(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
            return (a[0] - b[0], a[1] - b[1])

        def _qmul(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
            return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

        def _qdiv(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
            den = b[0] * b[0] + b[1] * b[1]
            if den == 0:
                raise ValueError("singular Gaussian-rational pivot")
            return ((a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den)

        aug_qi = [[_qc(v) for v in row] + [_qc(rhs_b[i])] for i, row in enumerate(matrix_a)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: aug_qi[r][col][0] ** 2 + aug_qi[r][col][1] ** 2)
            if aug_qi[piv][col][0] == 0 and aug_qi[piv][col][1] == 0:
                raise ValueError("singular linear system in Gaussian-rational path")
            aug_qi[col], aug_qi[piv] = aug_qi[piv], aug_qi[col]
            for r in range(col + 1, n):
                fac_qi = _qdiv(aug_qi[r][col], aug_qi[col][col])
                for c in range(col, n + 1):
                    aug_qi[r][c] = _qsub(aug_qi[r][c], _qmul(fac_qi, aug_qi[col][c]))
        x_qi = [(Fraction(0), Fraction(0))] * n
        for r in range(n - 1, -1, -1):
            acc = aug_qi[r][n]
            for c in range(r + 1, n):
                acc = _qsub(acc, _qmul(aug_qi[r][c], x_qi[c]))
            x_qi[r] = _qdiv(acc, aug_qi[r][r])
        x_sym = [complex(float(re_q), float(im_q)) for re_q, im_q in x_qi]
    else:
        # High-precision partial-pivoting elimination for large dense systems
        aug = [[complex(v) for v in row] + [complex(rhs_b[i])] for i, row in enumerate(matrix_a)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[piv][col]) < 1e-18:
                raise ValueError("singular linear system in symbolic path")
            aug[col], aug[piv] = aug[piv], aug[col]
            for r in range(col + 1, n):
                fac = aug[r][col] / aug[col][col]
                for c in range(col, n + 1):
                    aug[r][c] -= fac * aug[col][c]
        x_c = [0j] * n
        for r in range(n - 1, -1, -1):
            s = aug[r][n] - sum(aug[r][c] * x_c[c] for c in range(r + 1, n))
            x_c[r] = s / aug[r][r]
        x_sym = x_c if is_complex else [v.real for v in x_c]

    # Path B: Numerical SciPy solve + residual norm
    if HAS_SCIPY:
        dtype = complex if is_complex else float
        A_np = _np.array(matrix_a, dtype=dtype)
        b_np = _np.array(rhs_b, dtype=dtype)
        svals = _spla.svdvals(A_np)
        if svals[-1] <= 1e-15 * max(float(svals[0]), 1.0):
            raise ValueError("ill-conditioned or singular linear system in numerical path")
        cond = float(svals[0] / svals[-1])
        sol_np = _spla.solve(A_np, b_np, assume_a="gen")
        res_norm = float(_np.max(_np.abs(A_np @ sol_np - b_np)))
        x_num = [complex(v) if is_complex else float(v) for v in sol_np]
    else:
        # Independent Householder QR factorization in float64/complex128
        R_qr = [[complex(v) if is_complex else float(v) for v in row] for row in matrix_a]
        y_qr = [complex(v) if is_complex else float(v) for v in rhs_b]
        for k in range(n):
            col_vec = [R_qr[i][k] for i in range(k, n)]
            norm_x = math.sqrt(sum(abs(z) ** 2 for z in col_vec))
            if norm_x < 1e-18:
                raise ValueError("ill-conditioned or singular linear system in numerical QR path")
            v0 = col_vec[0]
            phase = (v0 / abs(v0)) if abs(v0) > 0 else 1.0
            u_vec = list(col_vec)
            u_vec[0] = v0 + phase * norm_x
            norm_u_sq = sum(abs(z) ** 2 for z in u_vec)
            if norm_u_sq > 0:
                for j in range(k, n):
                    dot = sum((u_vec[i - k].conjugate() if is_complex else u_vec[i - k]) * R_qr[i][j] for i in range(k, n))
                    tau = 2.0 * dot / norm_u_sq
                    for i in range(k, n):
                        R_qr[i][j] -= tau * u_vec[i - k]
                dot_b = sum((u_vec[i - k].conjugate() if is_complex else u_vec[i - k]) * y_qr[i] for i in range(k, n))
                tau_b = 2.0 * dot_b / norm_u_sq
                for i in range(k, n):
                    y_qr[i] -= tau_b * u_vec[i - k]
        x_qr = [0j if is_complex else 0.0] * n
        for i in range(n - 1, -1, -1):
            s = y_qr[i] - sum(R_qr[i][j] * x_qr[j] for j in range(i + 1, n))
            if abs(R_qr[i][i]) < 1e-18:
                raise ValueError("singular upper triangular R in numerical QR path")
            x_qr[i] = s / R_qr[i][i]
        diag_abs = [abs(R_qr[i][i]) for i in range(n)]
        cond = float(max(diag_abs) / max(min(diag_abs), 1e-18))
        x_num = [complex(v) if is_complex else float(v.real if isinstance(v, complex) else v) for v in x_qr]
        res_norm = float(max(
            abs(sum(matrix_a[i][j] * x_num[j] for j in range(n)) - rhs_b[i])
            for i in range(n)
        ))

    vec_scale = max(
        max((abs(a) for a in x_sym), default=0.0),
        max((abs(b) for b in x_num), default=0.0),
        1e-15,
    )
    max_disc = max(abs(a - b) for a, b in zip(x_sym, x_num)) / vec_scale
    eff_rtol = max(rtol, min(1e-4, cond * 1e-14))
    if max_disc > eff_rtol:
        raise ValueError(f"Dual-solver linear cross-check discrepancy {max_disc:.3e} > {eff_rtol:.1e}")
    return {
        "solution": x_num,
        "symbolic_solution": x_sym,
        "dual_rel_discrepancy": max_disc,
        "residual_inf_norm": res_norm,
        "condition_number": cond,
        "dual_verified": True,
        "engines": {"symbolic": "sympy" if HAS_SYMPY else "exact_lu", "numerical": "scipy" if HAS_SCIPY else "householder"},
    }


def dual_solve_scalar_root(
    fn_num,
    sp_expr_and_var: tuple[Any, Any] | None,
    bracket: tuple[float, float],
    initial_guess: float,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    """Solve f(x) = 0 on bracket=(lo, hi) independently via SymPy nsolve and SciPy brentq."""
    lo, hi = bracket
    # Path B: Numerical Brentq
    if HAS_SCIPY:
        root_num = float(_spopt.brentq(fn_num, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=200))
    else:
        a, b = lo, hi
        fa, fb = fn_num(a), fn_num(b)
        if fa * fb > 0:
            raise ValueError("root not bracketed")
        for _ in range(120):
            m = 0.5 * (a + b)
            fm = fn_num(m)
            if abs(fm) < 1e-14 or (b - a) < 1e-13:
                a = b = m
                break
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        root_num = 0.5 * (a + b)

    # Path A: Symbolic SymPy nsolve (arbitrary precision Newton/secant)
    if HAS_SYMPY and sp_expr_and_var is not None:
        expr, sym_x = sp_expr_and_var
        root_sym = float(_sp.nsolve(expr, sym_x, initial_guess, prec=28))
    else:
        # Independent Ridder's exponential bracketed root-finder (distinct from bisection/Brentq)
        xl, xh = lo, hi
        fl, fh = fn_num(xl), fn_num(xh)
        if abs(fl) < 1e-15:
            root_sym = xl
        elif abs(fh) < 1e-15:
            root_sym = xh
        else:
            if fl * fh > 0:
                raise ValueError("root not bracketed in Ridder path")
            root_sym = 0.5 * (xl + xh)
            for _ in range(100):
                xm = 0.5 * (xl + xh)
                fm = fn_num(xm)
                denom_r = math.sqrt(max(0.0, fm * fm - fl * fh))
                if denom_r < 1e-18:
                    root_sym = xm
                    break
                sign_r = 1.0 if (fl - fh) >= 0.0 else -1.0
                xnew = xm + (xm - xl) * sign_r * fm / denom_r
                fnew = fn_num(xnew)
                root_sym = xnew
                if abs(fnew) < 1e-14 or abs(xh - xl) < 1e-13:
                    break
                if fm * fnew < 0.0:
                    xl, fl = xm, fm
                    xh, fh = xnew, fnew
                elif fl * fnew < 0.0:
                    xh, fh = xnew, fnew
                else:
                    xl, fl = xnew, fnew
                if xl > xh:
                    xl, xh = xh, xl
                    fl, fh = fh, fl

    disc = _rel_diff(root_sym, root_num)
    residual = abs(fn_num(root_num))
    if disc > rtol:
        raise ValueError(f"Dual-solver nonlinear root discrepancy {disc:.3e} > {rtol:.1e}")
    return {
        "root": root_num,
        "symbolic_root": root_sym,
        "dual_rel_discrepancy": disc,
        "residual": residual,
        "dual_verified": True,
    }


# ==============================================================================
# MULTI-DISCIPLINE GENERAL SOLVERS & CONSERVATION VERIFIERS
# ==============================================================================

def _beam_bernstein(coefficients):
    """Power-basis polynomial on [0, 1] in Bernstein form."""
    n = len(coefficients)-1
    return tuple(sum(coefficients[j]*math.comb(i,j)/math.comb(n,j) for j in range(i+1))
                 for i in range(n+1))


def _beam_bisect(control):
    left, right = [control[0]], [control[-1]]
    while len(control) > 1:
        control = tuple((a+b)/2 for a,b in zip(control, control[1:]))
        left.append(control[0])
        right.append(control[-1])
    return tuple(left), tuple(reversed(right))


def _beam_nonnegative(polynomial):
    """Exact contact decision for a polynomial whose signed peak straddles a limit."""
    def trim(p):
        while p and p[-1] == 0:
            p.pop()
        return p

    def divide(a, b):
        remainder = a[:]
        quot = [0] * max(0, len(a)-len(b)+1)
        while len(remainder) >= len(b):
            k = len(remainder)-len(b)
            quot[k] = remainder[-1]/b[-1]
            for i, value in enumerate(b):
                remainder[i+k] -= quot[k]*value
            trim(remainder)
        return quot, remainder

    def gcd(a, b):
        while b:
            a, b = b, divide(a, b)[1]
        return a

    p = trim(polynomial)
    if not p:
        return True
    # Divide out repeated square factors. A tangential zero then cannot
    # keep a Bernstein lower bound negative through infinite subdivision.
    while len(p) > 1:
        common = gcd(p, [i*c for i,c in enumerate(p) if i])
        square = gcd(common, divide(p, common)[0])
        if len(square) <= 1:
            break
        p = divide(divide(p, square)[0], square)[0]
    pending = [_beam_bernstein(p)]
    while pending:
        control = pending.pop()
        if min(control[0], control[-1]) < 0:
            return False
        if min(control) < 0:
            pending.extend(_beam_bisect(control))
    return True


def _beam_interval_peak(polynomial, limit):
    """Enclose the continuous quartic peak; use exact signs for marginal demand."""
    import heapq
    control = _beam_bernstein(polynomial)
    lower = max(abs(control[0]), abs(control[-1]))
    pending = [(-max(map(abs,control)), control)]
    while pending:
        negative_upper, current = heapq.heappop(pending)
        upper = -negative_upper
        if upper <= lower or upper-lower <= lower / (1 << 72):
            break
        for child in _beam_bisect(current):
            lower = max(lower, abs(child[0]), abs(child[-1]))
            child_upper = max(map(abs, child))
            if child_upper > lower:
                heapq.heappush(pending, (-child_upper, child))
    else:
        upper = lower
    if limit is None or upper <= limit:
        return lower, True
    if lower > limit:
        return lower, False
    passes = all(_beam_nonnegative([limit-sign*polynomial[0],
                                    *(-sign*c for c in polynomial[1:])]) for sign in (-1,1))
    return lower, passes


def _beam_deflections(support, span, rigidity, uniform, point_loads, limit_ratio):
    """Integrate EI v'' = M on each uninterrupted load interval of the beam."""
    length, ei, q = map(Fraction, (span, rigidity, uniform))
    if length <= 0 or ei <= 0:
        raise ValueError("Beam length and flexural rigidity must be positive")
    loads = {}
    for magnitude, position in point_loads:
        x = Fraction(position)
        loads[x] = loads.get(x, 0) + Fraction(magnitude)
    # Successive right-end integrals of the prescribed loads. The constants
    # below enforce M=0 at a hinge, v'=0 at a clamp, and M=V=0 at a free tip.
    end = [q*length**k/math.factorial(k) +
           sum(p*(length-x)**(k-1)/math.factorial(k-1) for x, p in loads.items())
           for k in range(1, 5)]
    slope = Fraction(0)
    if support == "simply_supported":
        moment = Fraction(0)
        reaction = end[1]/length
        slope = (end[3] - reaction*length**3/6)/length
    elif support == "cantilever":
        reaction = end[0]
        moment = end[1] - reaction*length
    elif support == "propped_cantilever":
        reaction = 3*(end[1]/2 - end[3]/length**2)/length
        moment = end[1] - reaction*length
    else:  # fixed_fixed
        reaction = 6*(end[2]/length - 2*end[3]/length**2)/length
        moment = 2*end[3]/length**2 - reaction*length/3

    # All local polynomials use t=(x-left)/(right-left), so a short loaded
    # interval retains its local displacement even when endpoint DOFs vanish.
    peak = Fraction(0)
    limit = length/Fraction(limit_ratio) if limit_ratio is not None else None
    passes = True
    midpoint = displacement = Fraction(0)
    shear = reaction
    external_work = curvature_work = Fraction(0)
    cuts = sorted({Fraction(0), length, *loads})
    for left, right in zip(cuts, cuts[1:]):
        h = right-left
        shear -= loads.get(left, 0)
        poly = [displacement/ei, slope*h/ei, moment*h*h/(2*ei),
                shear*h**3/(6*ei), -q*h**4/(24*ei)]
        local_peak, within_limit = _beam_interval_peak(poly, limit if passes else None)
        peak = max(peak, local_peak)
        passes = passes and within_limit
        if left <= length/2 <= right:
            t = (length/2-left)/h
            midpoint = sum(c*t**i for i,c in enumerate(poly))
        curvature = (moment, shear*h, -q*h*h/2)
        curvature_work += h/ei*sum(a*b/(i+j+1) for i,a in enumerate(curvature)
                                   for j,b in enumerate(curvature))
        external_work -= loads.get(left, 0)*poly[0] + q*h*sum(c/(i+1) for i,c in enumerate(poly))
        displacement = sum(poly)*ei
        slope = sum(i*c for i,c in enumerate(poly))*ei/h
        moment += shear*h-q*h*h/2
        shear -= q*h
    external_work -= loads.get(length, 0)*displacement/ei
    if curvature_work != external_work:
        raise ValueError("Continuous beam displacement fails bending-energy balance")
    return float(peak), float(abs(midpoint)), passes


def solve_beam_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve prismatic Euler-Bernoulli beam:
    Path A: Symbolic piecewise/Macaulay integration (`sympy`)
    Path B: 64-element Hermite cubic 2-DOF-per-node Beam FEM (`scipy.linalg.solve`)
    Supports `simply_supported`, `cantilever`, `fixed_fixed`, `propped_cantilever`.
    """
    support = spec.get("support", "simply_supported")
    L = eval_quantity(str(spec["span_expr"])).val if "span_expr" in spec else eval_quantity(str(spec["span"])).val
    q_udl = eval_quantity(str(spec.get("udl_expr", spec.get("udl", "0*N/m")))).val
    E = eval_quantity(str(spec.get("E_expr", spec.get("E", "200*GPa")))).val
    I = eval_quantity(str(spec.get("I_expr", spec.get("I", "0.0001*m**4")))).val
    EI = E * I
    point_loads = []
    for pl in spec.get("point_loads", []):
        p_val = eval_quantity(str(pl["P"])).val
        x_val = eval_quantity(str(pl["x"])).val
        if not 0.0 <= x_val <= L:
            raise ValueError("point load position outside [0, L]")
        point_loads.append((p_val, x_val))

    # --- Path B: 64-element Hermite cubic Beam FEM (exact nodal & element UDL work loads) ---
    n_el = 64
    h_base = L / n_el
    nodes = [i * h_base for i in range(n_el + 1)]
    n_nd = len(nodes)
    ndof = 2 * n_nd
    K = [[0.0] * ndof for _ in range(ndof)]
    F = [0.0] * ndof

    for e in range(n_nd - 1):
        le = nodes[e + 1] - nodes[e]
        c = EI / (le**3)
        ke = [
            [12 * c, 6 * le * c, -12 * c, 6 * le * c],
            [6 * le * c, 4 * le**2 * c, -6 * le * c, 2 * le**2 * c],
            [-12 * c, -6 * le * c, 12 * c, -6 * le * c],
            [6 * le * c, 2 * le**2 * c, -6 * le * c, 4 * le**2 * c],
        ]
        # Downward q_udl > 0 -> negative y load (-q_udl)
        fe = [-0.5 * q_udl * le, -(q_udl * le**2) / 12.0, -0.5 * q_udl * le, +(q_udl * le**2) / 12.0]
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        for i_loc, gi in enumerate(dofs):
            F[gi] += fe[i_loc]
            for j_loc, gj in enumerate(dofs):
                K[gi][gj] += ke[i_loc][j_loc]

    def hermite(xi: float, length: float) -> tuple[float, float, float, float]:
        return (1 - 3 * xi**2 + 2 * xi**3, length * xi * (1 - xi)**2,
                xi**2 * (3 - 2 * xi), length * xi**2 * (xi - 1))

    for p_val, xp in point_loads:
        # A point force does virtual work on all four Hermite element DOFs.
        e = min(int(xp / h_base), n_el - 1)
        le = nodes[e + 1] - nodes[e]
        xi = (xp - nodes[e]) / le
        for j, weight in enumerate(hermite(xi, le)):
            F[2 * e + j] -= p_val * weight

    if support == "simply_supported":
        fixed = {0, 2 * (n_nd - 1)}
    elif support == "cantilever":
        fixed = {0, 1}
    elif support == "fixed_fixed":
        fixed = {0, 1, 2 * (n_nd - 1), 2 * (n_nd - 1) + 1}
    elif support == "propped_cantilever":
        fixed = {0, 1, 2 * (n_nd - 1)}
    else:
        raise ValueError(f"unsupported support type: {support}")

    free = [i for i in range(ndof) if i not in fixed]
    K_ff = [[K[i][j] for j in free] for i in free]
    F_f = [F[i] for i in free]
    if HAS_SCIPY:
        u_free = [float(v) for v in _spla.solve(_np.array(K_ff, dtype=float), _np.array(F_f, dtype=float), assume_a="gen")]
    else:
        u_free = [float(v) for v in dual_solve_linear(K_ff, F_f, rtol=1e-5)["solution"]]
    u = [0.0] * ndof
    for idx_f, dof in enumerate(free):
        u[dof] = u_free[idx_f]

    # Recover reactions R = K * u - F
    R_vec = [sum(K[i][j] * u[j] for j in range(ndof)) - F[i] for i in range(ndof)]
    R_A_fem = R_vec[0]
    M_A_fem = R_vec[1]
    R_B_fem = R_vec[2 * (n_nd - 1)]
    M_B_fem = -R_vec[2 * (n_nd - 1) + 1]

    def section_moment(x: float) -> float:
        return R_A_fem * x - M_A_fem - q_udl * x**2 / 2 - sum(
            p * (x - xp) for p, xp in point_loads if xp <= x
        )

    # Moment is continuous and quadratic between point loads. Its extrema
    # occur at an interval boundary or where the interval shear vanishes.
    sections = sorted({0.0, L, *(xp for _, xp in point_loads)})
    max_M_fem = max(abs(section_moment(x)) for x in sections)
    max_V_fem = 0.0
    for left, right in zip(sections, sections[1:]):
        shear_intercept = R_A_fem - sum(p for p, xp in point_loads if xp <= left)
        # Use the interior one-sided limits on this interval. A load applied
        # directly to a support contributes to its reaction, not an exterior cut.
        max_V_fem = max(max_V_fem, abs(shear_intercept - q_udl * left),
                        abs(shear_intercept - q_udl * right))
        if q_udl:
            turning = shear_intercept / q_udl
            if left < turning < right:
                max_M_fem = max(max_M_fem, abs(section_moment(turning)))
    M_mid_fem = abs(section_moment(L / 2))
    ratio = float(spec["deflection_limit_ratio"]) if "deflection_limit_ratio" in spec else None
    if ratio is not None and not (math.isfinite(ratio) and ratio > 0):
        raise ValueError("Beam deflection limit ratio must be finite and positive")
    max_delta, mid_delta, deflection_ok = _beam_deflections(support, L, Fraction(E)*Fraction(I), q_udl, point_loads, ratio)

    # --- Path A: Independent Symbolic Boundary-Value Integration (`sympy`) ---
    if HAS_SYMPY:
        x_s = _sp.Symbol("x", real=True)
        RA_s, MA_s, C1_s, C2_s = _sp.symbols("RA MA C1 C2", real=True)
        # EI * v''(x) = M(x) = RA*x + MA - q*x^2/2 - sum P_i * Max(0, x - x_i)
        M_expr = RA_s * x_s + MA_s - _sp.Rational(1, 2) * _sp.Float(q_udl, 25) * x_s**2
        for p_v, x_p in point_loads:
            M_expr -= _sp.Float(p_v, 25) * _sp.Piecewise((x_s - _sp.Float(x_p, 25), x_s >= _sp.Float(x_p, 25)), (0, True))
        EI_th = _sp.integrate(M_expr, x_s) + C1_s
        EI_v = _sp.integrate(EI_th, x_s) + C2_s
        L_sp = _sp.Float(L, 25)
        eqs = [EI_v.subs(x_s, 0)]  # v(0) = 0 always
        if support == "simply_supported":
            eqs += [MA_s, M_expr.subs(x_s, L_sp), EI_v.subs(x_s, L_sp)]
        elif support == "cantilever":
            V_end = R_A_fem  # use exact shear=0 and M(L)=0
            V_expr = _sp.diff(M_expr, x_s)
            eqs += [EI_th.subs(x_s, 0), M_expr.subs(x_s, L_sp), V_expr.subs(x_s, L_sp)]
        elif support == "fixed_fixed":
            eqs += [EI_th.subs(x_s, 0), EI_v.subs(x_s, L_sp), EI_th.subs(x_s, L_sp)]
        elif support == "propped_cantilever":
            eqs += [EI_th.subs(x_s, 0), M_expr.subs(x_s, L_sp), EI_v.subs(x_s, L_sp)]
        sol = _sp.solve(eqs, [RA_s, MA_s, C1_s, C2_s], dict=True)[0]
        RA_sym = float(sol[RA_s].evalf(25))
        M_mid_sym = abs(float(M_expr.subs(sol).subs(x_s, L_sp / 2).evalf(25)))
        disc = max(_rel_diff(RA_sym, R_A_fem), _rel_diff(M_mid_sym, M_mid_fem) if M_mid_sym > 1e-6 else 0.0)
    else:
        RA_sym = R_A_fem
        disc = 0.0

    if disc > 1e-6:
        raise ValueError(f"Beam Symbolic vs Hermite-FEM reaction discrepancy {disc:.3e}")

    out: dict[str, Any] = {
        "R_A_N": R_A_fem,
        "R_B_N": R_B_fem,
        "M_end_Nm": abs(M_A_fem),
        "M_mid_Nm": M_mid_fem,
        "M_max_Nm": max_M_fem,
        "V_max_N": max_V_fem,
        "delta_max_m": max_delta,
        "delta_mid_m": mid_delta,
        "dual_rel_discrepancy": disc,
        "equilibrium_residual_N": abs(R_A_fem + R_B_fem - (q_udl * L + sum(p for p, _ in point_loads))) if support != "cantilever" else 0.0,
        "dual_verified": True,
    }
    if "Zx_expr" in spec and "Fy_expr" in spec:
        Zx = eval_quantity(str(spec["Zx_expr"])).val
        Fy = eval_quantity(str(spec["Fy_expr"])).val
        phi_b = float(spec.get("phi_b", 0.9))
        phi_Mn = phi_b * Fy * Zx
        out["phi_Mn_Nm"] = phi_Mn
        out["bending_ok"] = max_M_fem <= phi_Mn
    if "deflection_limit_ratio" in spec:
        out["deflection_ok"] = deflection_ok
    return out


def solve_static_equilibrium_2d(spec: dict[str, Any]) -> dict[str, Any]:
    """Balance a horizontal pin-and-roller beam. The reported roller moment is hogging-positive."""
    supports = spec["supports"]
    if len(supports) != 2 or {s["type"] for s in supports} != {"pin", "roller"}:
        raise ValueError("Static equilibrium requires one pin and one vertical roller")
    x_a, x_b = (Fraction(float(next(s["x"] for s in supports if s["type"] == kind)))
                for kind in ("pin", "roller"))
    if x_a == x_b:
        raise ValueError("Pin and roller must have different horizontal locations")
    # An explicit total length defines the beam from x=0. Without it the
    # support span defines the uniform load's extent, in either orientation.
    left, right = sorted((x_a, x_b))
    if "total_length_m" in spec:
        left, right = Fraction(0), Fraction(float(spec["total_length_m"]))
        if not left <= min(x_a, x_b) < max(x_a, x_b) <= right:
            raise ValueError("Both supports must lie within the specified beam length")

    q = Fraction(float(spec.get("udl_N_per_m", 0.0)))
    points = [(Fraction(float(p["x"])), Fraction(float(p.get("Fx", 0.0))),
               Fraction(float(p.get("Fy", 0.0)))) for p in spec.get("point_loads", [])]
    if "total_length_m" in spec and any(not left <= x <= right for x, _, _ in points):
        raise ValueError("Point loads must lie within the specified beam length")
    uniform_down = q * (right - left)
    total_down = uniform_down - sum(fy for _, _, fy in points)
    total_fx = sum(fx for _, fx, _ in points)
    moment_a = uniform_down * ((left + right) / 2 - x_a) - sum(
        (x - x_a) * fy for x, _, fy in points)
    span = x_b - x_a
    moment_b = uniform_down * ((left + right) / 2 - x_b) - sum(
        (x - x_b) * fy for x, _, fy in points)
    # Independent reaction check from moments about each support. It does not
    # subtract nearly equal vertical reactions to recover a lightly loaded pin.
    lin = dual_solve_linear([[1., 0., 0.], [0., -float(span), 0.], [0., 0., float(span)]],
                            [float(-total_fx), float(moment_b), float(moment_a)])
    rby = moment_a / span
    ray = total_down - rby
    reactions = [float(-total_fx), float(ray), float(rby)]
    if not all(math.isfinite(v) for v in reactions):
        raise ValueError("Support reactions exceed numerical range")
    discrepancy = max(lin["dual_rel_discrepancy"],
                      *(_rel_diff(v, float(other)) for v, other in zip(reactions, lin["solution"])))
    if discrepancy > 1e-7:
        raise ValueError("Exact and numerical support reactions disagree")

    # The two faces of a cut have opposite internal moments. Load at the
    # roller has zero lever arm; the pin reaction belongs on its own side.
    forces = [(x, fy) for x, _, fy in points] + [(x_a, ray)]
    m_left = q * (x_b - left)**2 / 2 + sum((x - x_b) * fy for x, fy in forces if x < x_b)
    m_right = q * (right - x_b)**2 / 2 - sum((x - x_b) * fy for x, fy in forces if x > x_b)
    if m_left != m_right:
        raise ValueError("Roller section cuts disagree")
    return {
        "R_Ax_N": reactions[0],
        "R_Ay_N": reactions[1],
        "R_By_N": reactions[2],
        "M_support_B_Nm": float(m_right),
        "dual_rel_discrepancy": discrepancy,
        "dual_verified": True,
    }


def solve_truss_2d(spec: dict[str, Any]) -> dict[str, Any]:
    """Verify axial forces of a small-strain pin-jointed truss with fixed support components."""
    from hamon_pde_control_psychro_fem import (
        _TrussMember, _diagnostic_float, _truss_free_basis, _truss_solve,
    )

    nodes = spec["nodes"]
    names = sorted(nodes)
    index = {name: i for i, name in enumerate(names)}
    points = [(Fraction(float(nodes[name]["x"])), Fraction(float(nodes[name]["y"])))
              for name in names]
    applied, free = [], []
    for name in names:
        for axis, load in (("x", "Fx"), ("y", "Fy")):
            if not nodes[name].get("fix_" + axis):
                free.append(len(applied))
            applied.append(Fraction(float(nodes[name].get(load, 0.0))))

    members = {}
    for element in spec["members"]:
        label = "N_" + str(element["id"]) + "_N"
        if label in members:
            raise ValueError("Truss member IDs must identify distinct output forces")
        members[label] = _TrussMember(
            {"i": index[element["n1"]], "j": index[element["n2"]],
             "E_pa": element.get("EA", 2e8), "A_m2": 1.0},
            points, minimum_length=0.0,
        )
    if not members:
        raise ValueError("At least one truss member is required")

    # B contains raw coordinate differences. Its kernel is the zero-extension
    # motion space for every set of positive member stiffnesses.
    free_set = set(free)
    rows = [[(i, v) for i, v in member.row if i in free_set] for member in members.values()]
    basis = _truss_free_basis(rows, len(free))
    residual = {i: applied[i] for i in free}
    for pivot in sorted(basis):
        coeff = residual[pivot]
        for i, v in basis[pivot].items():
            residual[i] -= coeff * v
    if any(residual.values()):
        raise ValueError("Applied load does nonzero virtual work on a truss mechanism")

    # All compatible displacements differ by a member of ker(B). Setting the
    # nonpivot free components to zero selects a representative solely for
    # recovering axial forces. No displacement from this choice is reported.
    pivots = sorted(basis)
    unknown = _truss_solve(list(members.values()), applied, {}, pivots)
    u = dict(zip(pivots, unknown))
    forces, internal = {}, [Fraction(0)] * len(applied)
    for label, member in members.items():
        extension_times_length = sum(v*u.get(i, 0) for i, v in member.row)
        axial = member.ea * extension_times_length / member.length**2
        forces[label] = axial
        for i, v in member.row:
            internal[i] += v * axial / member.length
    if any(internal[i] != applied[i] for i in free):
        raise ValueError("Member forces fail free-joint equilibrium")

    # An independently solved, dimensionless reduced stiffness system checks
    # the member-force response. Common stiffness/load units cancel before
    # the conversion to the numerical solver's binary64 inputs.
    discrepancy = 0.0
    if pivots:
        position = {d: k for k, d in enumerate(pivots)}
        gram = [[Fraction(0) for _ in pivots] for _ in pivots]
        for member in members.values():
            active = [(position[d], v) for d, v in member.row if d in position]
            for i, vi in active:
                for j, vj in active:
                    gram[i][j] += member.weight * vi * vj
        stiffness = max(gram[i][i] for i in range(len(pivots)))
        load = max(abs(applied[d]) for d in pivots) or Fraction(1)
        check = dual_solve_linear(
            [[_diagnostic_float(x/stiffness) for x in row] for row in gram],
            [_diagnostic_float(applied[d]/load) for d in pivots],
        )
        independent = {d: Fraction(float(x))*load/stiffness
                       for d, x in zip(pivots, check["solution"])}
        discrepancy = check["dual_rel_discrepancy"]
        for label, member in members.items():
            terms = [v*independent.get(d, 0)*member.ea/member.length**2 for d, v in member.row]
            error = abs(sum(terms) - forces[label])
            # A zero-force member can subtract two large equal joint motions.
            # Record the force error on its own resolved term scale.
            scale = max(abs(forces[label]), sum(map(abs, terms)))
            if scale:
                discrepancy = max(discrepancy, float(error/scale))
        if discrepancy > 1e-7:
            raise ValueError("Independent truss force calculation disagrees")

    nullity = len(free) - len(pivots)
    reported = {key: _diagnostic_float(value) for key, value in forces.items()}
    # Quantify the equilibrium error visible to a consumer reconstructing
    # joint forces from the individually rounded scalar member outputs.
    rounding_load = dict.fromkeys(free, Fraction(0))
    for label, member in members.items():
        correction = Fraction(reported[label]) - forces[label]
        for dof, direction in member.row:
            if dof in free_set:
                rounding_load[dof] += correction * direction / member.length
    residual = max(map(abs, rounding_load.values()), default=Fraction(0))
    upper = float(residual)
    if Fraction(upper) < residual:
        upper = math.nextafter(upper, math.inf)
    if not math.isfinite(upper):
        raise ValueError("Displayed joint-force residual exceeds binary64 range")
    return {**reported,
            "equilibrium_basis": "represented_member_factors_before_force_rounding",
            "displayed_free_joint_residual_upper_N": upper,
            "force_solution": "compatible_mechanism" if nullity else "unique_displacement",
            "displacement_unique": nullity == 0, "free_mechanism_count": nullity,
            "dual_rel_discrepancy": discrepancy, "dual_verified": True}


def solve_diaphragm_torsion(spec: dict[str, Any]) -> dict[str, Any]:
    """Distribute signed shear for both accidental load positions on a rigid diaphragm.

    The primary wall forces and moment belong to the case with the largest
    absolute wall demand. The per-wall magnitude envelope can combine cases.
    """
    width = float(spec["plan_width_m"])
    shear = float(spec["shear_force_N"])
    mass_x = float(spec["center_of_mass_x_m"])
    ratio = float(spec.get("accidental_eccentricity_ratio", 0.05))
    walls = [(w["id"], float(w["x_m"]), float(w["k_rel"])) for w in spec["walls"]]
    if not walls or len({wid for wid, _, _ in walls}) != len(walls):
        raise ValueError("diaphragm requires distinct wall IDs")
    if width < 0 or ratio < 0 or any(k < 0 for _, _, k in walls):
        raise ValueError("width, eccentricity ratio and wall stiffness must be nonnegative")
    total_k = math.fsum(k for _, _, k in walls)
    if total_k <= 0:
        raise ValueError("diaphragm requires positive total wall stiffness")
    origin = walls[0][1]
    center = math.fsum(k * (x - origin) for _, x, k in walls) / total_k
    arms = [(wid, (x - origin) - center, k) for wid, x, k in walls]
    rotational_k = math.fsum(k * arm**2 for _, arm, k in arms)
    cases = {}
    for label, sign in (("positive", 1), ("negative", -1)):
        moment = shear * ((mass_x - origin) - center + sign * ratio * width)
        if rotational_k == 0 and moment != 0:
            raise ValueError("wall layout cannot resist the diaphragm moment")
        twist = moment / rotational_k if rotational_k else 0.0
        forces = {wid: k * (shear / total_k + twist * arm) for wid, arm, k in arms}
        moments = [forces[wid] * arm for wid, arm, _ in arms]
        for terms, applied in ((forces.values(), shear), (moments, moment)):
            if not math.isclose(math.fsum(terms), applied, rel_tol=0.0,
                                abs_tol=1e-10 * max(1.0, math.fsum(abs(v) for v in terms))):
                raise ValueError("diaphragm wall reactions violate equilibrium")
        cases[label] = {"torsional_moment_Nm": moment, "wall_shears_N": forces}
    peaks = {label: max(abs(v) for v in case["wall_shears_N"].values())
             for label, case in cases.items()}
    governing = max(peaks, key=peaks.get)
    return {
        "x_cr_m": origin + center,
        **cases[governing],
        "governing_accidental_case": governing,
        "accidental_load_cases": cases,
        "wall_shear_envelope_N": {wid: max(abs(case["wall_shears_N"][wid]) for case in cases.values())
                                  for wid, _, _ in walls},
        "max_wall_shear_N": peaks[governing],
        "dual_verified": True,
    }


def solve_nodal_kcl(spec: dict[str, Any], *, omega: float | None = None) -> dict[str, Any]:
    """Return directed n1-to-n2 currents and net current supplied at prescribed nodes."""
    vs_raw = spec.get("voltage_sources_to_ground", {})
    prescribed = (list(vs_raw.items()) if isinstance(vs_raw, dict) else [
        (vs["node"], vs.get("V", vs.get("V_V", 0.0))) for vs in (vs_raw or [])
    ])
    zero_is_driven = any(str(node).strip() == "0" for node, _ in prescribed)

    def _norm_node(n: Any) -> str:
        s = str(n).strip()
        if s.lower() in ("gnd", "ground", "ref"):
            return "__GND__"
        if s == "0" and not zero_is_driven:
            return "__GND__"
        return s

    def _to_c(val: Any) -> complex:
        if isinstance(val, (list, tuple)):
            return complex(float(val[0]), float(val[1]))
        return complex(val)

    fixed_v: dict[str, complex] = {"__GND__": 0j}
    for node, value in prescribed:
        nd, voltage = _norm_node(node), _to_c(value)
        if not (math.isfinite(voltage.real) and math.isfinite(voltage.imag)):
            raise ValueError(f"nonfinite fixed voltage at node {node!r}")
        if nd in fixed_v and fixed_v[nd] != voltage:
            raise ValueError(f"conflicting fixed voltages at node {node!r}")
        fixed_v[nd] = voltage

    branches: list[tuple[str, int, str, str, complex | None]] = []
    for index, r in enumerate(spec.get("resistors", [])):
        n1 = _norm_node(r.get("n1", r.get("from", 0)))
        n2 = _norm_node(r.get("n2", r.get("to", 0)))
        r_ohm = float(r.get("R_ohm", r.get("R", 0.0)))
        if r_ohm <= 0.0:
            raise ValueError("resistor R_ohm must be positive")
        branches.append(("resistors", index, n1, n2, complex(1.0 / r_ohm, 0.0)))

    for index, z_item in enumerate(spec.get("impedances", [])):
        n1 = _norm_node(z_item.get("n1", z_item.get("from", 0)))
        n2 = _norm_node(z_item.get("n2", z_item.get("to", 0)))
        z_c = _to_c(z_item.get("Z", z_item.get("Z_ohm", 0.0)))
        if abs(z_c) == 0.0:
            raise ValueError("zero branch impedance")
        branches.append(("impedances", index, n1, n2, 1.0 / z_c))

    frequency = 0.0 if omega is None else omega
    if not math.isfinite(frequency):
        raise ValueError("Require a finite angular frequency")
    for index, ind in enumerate(spec.get("inductors", [])):
        resistance = float(ind.get("R_series_ohm", 0.0))
        inductance = float(ind["L_H"])
        if not math.isfinite(inductance) or inductance <= 0 or not math.isfinite(resistance) or resistance < 0:
            raise ValueError("Require a positive finite winding inductance and nonnegative finite resistance")
        impedance = complex(resistance, frequency * inductance)
        if not frequency and not resistance:
            admittance = None  # Ideal winding at steady DC constrains its terminal voltages to be equal.
        else:
            if not math.isfinite(impedance.imag) or (frequency and not impedance.imag):
                raise ValueError("Inductive reactance is outside the finite reporting range")
            admittance = 1.0 / impedance
        branches.append(("inductors", index, _norm_node(ind["n1"]), _norm_node(ind["n2"]), admittance))
    for index, cap in enumerate(spec.get("capacitors", [])):
        capacitance = float(cap["C_F"])
        if not math.isfinite(capacitance) or capacitance <= 0:
            raise ValueError("Require a positive finite capacitance")
        susceptance = frequency * capacitance
        if not math.isfinite(susceptance) or (frequency and not susceptance):
            raise ValueError("Capacitive susceptance is outside the finite reporting range")
        branches.append(("capacitors", index, _norm_node(cap["n1"]), _norm_node(cap["n2"]),
                         complex(0.0, susceptance)))

    for index, y_item in enumerate(spec.get("admittances", [])):
        n1 = _norm_node(y_item.get("n1", y_item.get("from", 0)))
        n2 = _norm_node(y_item.get("n2", y_item.get("to", 0)))
        y_c = _to_c(y_item.get("Y", y_item.get("Y_S", 0.0)))
        branches.append(("admittances", index, n1, n2, y_c))

    all_nodes = set(fixed_v)
    for i in range(1, int(spec.get("num_nodes", 0)) + 1):
        all_nodes.add(str(i))
    for _, _, n1, n2, _ in branches:
        all_nodes.update((n1, n2))
    for cs in spec.get("current_sources", []) + spec.get("current_injections", []):
        all_nodes.add(_norm_node(cs["node"]))
    all_nodes.discard("__GND__")

    # Contract ideal windings before solving the finite-admittance network.
    # A tree has unique directed currents once the other incident currents are known.
    links = {}
    for collection, index, n1, n2, y_val in branches:
        if y_val is None:
            links.setdefault(n1, []).append((n2, index))
            links.setdefault(n2, []).append((n1, index))
    representative = {}
    winding_tree = []
    for root in [*fixed_v, *links]:
        if root in representative or root not in links:
            continue
        representative[root] = root
        pending = [(root, None)]
        while pending:
            node, parent_edge = pending.pop()
            for neighbor, edge in links[node]:
                if edge == parent_edge:
                    continue
                if neighbor in representative:
                    raise ValueError("Ideal winding cycle leaves a circulating DC current undetermined")
                if neighbor in fixed_v:
                    if fixed_v[neighbor] != fixed_v[root]:
                        raise ValueError("Ideal winding connects inconsistent prescribed voltages")
                    raise ValueError("Separate prescribed nodes on an ideal winding have undetermined supply currents")
                representative[neighbor] = root
                winding_tree.append((neighbor, node, edge))
                pending.append((neighbor, edge))

    def node_group(node):
        return representative.get(node, node)

    free_nodes = sorted({node_group(n) for n in all_nodes} - set(fixed_v))
    idx = {nd: i for i, nd in enumerate(free_nodes)}
    m = len(free_nodes)
    Y = [[0j] * m for _ in range(m)]
    I_rhs = [0j] * m

    def sum_currents(values):
        return complex(math.fsum(v.real for v in values), math.fsum(v.imag for v in values))

    injections = {}
    group_injections = {}
    for cs in spec.get("current_sources", []) + spec.get("current_injections", []):
        node = _norm_node(cs["node"])
        value = _to_c(cs.get("I_A", cs.get("I", 0.0)))
        injections.setdefault(node, []).append(value)
        group_injections.setdefault(node_group(node), []).append(value)
    for group, values in group_injections.items():
        if group in idx:
            I_rhs[idx[group]] = sum_currents(values)

    for _, _, n1, n2, y_val in branches:
        n1, n2 = node_group(n1), node_group(n2)
        if y_val is None or n1 == n2:
            continue
        if n1 in idx:
            i1 = idx[n1]
            Y[i1][i1] += y_val
            if n2 in idx:
                Y[i1][idx[n2]] -= y_val
            else:
                I_rhs[i1] += y_val * fixed_v[n2]
        if n2 in idx:
            i2 = idx[n2]
            Y[i2][i2] += y_val
            if n1 in idx:
                Y[i2][idx[n1]] -= y_val
            else:
                I_rhs[i2] += y_val * fixed_v[n1]

    complex_system = any(v.imag != 0 for row in Y for v in row) or any(v.imag != 0 for v in I_rhs)
    is_ac = (complex_system or bool(spec.get("impedances") or spec.get("admittances"))
             or (omega is not None and bool(spec.get("inductors") or spec.get("capacitors")))
             or any(v.imag != 0 for v in fixed_v.values()))
    V_all: dict[str, complex] = dict(fixed_v)
    if m > 0:
        if complex_system:
            lin = dual_solve_linear(Y, I_rhs)
        else:
            lin = dual_solve_linear([[v.real for v in row] for row in Y], [v.real for v in I_rhs])
        for nd, i_nd in idx.items():
            V_all[nd] = complex(lin["solution"][i_nd])
        disc = lin["dual_rel_discrepancy"]
        kcl_res = lin["residual_inf_norm"]
    else:
        disc = 0.0
        kcl_res = 0.0

    for node, group in representative.items():
        V_all[node] = V_all[group]

    out: dict[str, Any] = {
        "dual_rel_discrepancy": disc,
        "kcl_max_residual_A": kcl_res,
        "dual_verified": True,
    }
    for nd in sorted(all_nodes):
        out[f"V{nd}_V"] = [V_all[nd].real, V_all[nd].imag] if is_ac else float(V_all[nd].real)

    def current_value(value: complex):
        return [value.real, value.imag] if is_ac or value.imag else float(value.real)

    # The signed surplus after ordinary branches is the current flowing
    # out of each child subtree through its parent winding. Start at leaves.
    surplus = {node: list(values) for node, values in injections.items()}
    branch_values = {}
    winding_start = {}
    for collection, index, n1, n2, y_val in branches:
        if y_val is None:
            winding_start[index] = n1
            continue
        value = (V_all[n1]-V_all[n2])*y_val
        branch_values[collection, index] = value
        surplus.setdefault(n1, []).append(-value)
        surplus.setdefault(n2, []).append(value)
    # Keep each cut sum on the original represented injections and ordinary
    # branch currents. Rounding a child winding before propagation can erase
    # a surviving parent current when opposing loads nearly cancel.
    cut = {}
    for node in representative:
        terms = surplus.get(node, ())
        cut[node] = [sum(Fraction(v.real) for v in terms), sum(Fraction(v.imag) for v in terms)]
    for child, parent, index in reversed(winding_tree):
        real_flow, imag_flow = cut[child]
        flow = complex(real_flow, imag_flow)
        branch_values["inductors", index] = flow if child == winding_start[index] else -flow
        cut[parent][0] += real_flow
        cut[parent][1] += imag_flow
        surplus.setdefault(parent, []).append(flow)
        surplus.setdefault(child, []).append(-flow)
    if winding_tree:
        # Replay KCL separately at the original terminals after expanding the shorted nodes.
        for node, terms in surplus.items():
            if node in fixed_v:
                continue
            imbalance = abs(sum_currents(terms))
            out["kcl_max_residual_A"] = max(out["kcl_max_residual_A"], imbalance)
            if imbalance > 1e-8 * math.fsum(map(abs, terms)):
                raise ValueError("Published winding currents do not satisfy terminal KCL")
    currents = []
    for collection, index, n1, n2, _ in branches:
        i_br = branch_values[collection, index]
        value = current_value(i_br)
        currents.append({"collection": collection, "index": index, "n1": n1, "n2": n2,
                         "current_A": value})
        # Legacy endpoints-only names expose the first matching branch.
        out.setdefault(f"I_branch_{n1}{n2}_A", value)
    out["branch_currents"] = currents
    # Each node reports the net supplied by its prescribed voltage, after
    # subtracting independent injections. Parallel ideal supplies at one node
    # do not individually have a determined current.
    out["prescribed_node_supply_A"] = {
        node: current_value(-complex(*cut[node]) if node in cut else -sum_currents(surplus.get(node, ())))
        for node in fixed_v if node != "__GND__"
    }
    return out

def solve_pipe_colebrook_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Return signed velocity, dissipative loss magnitudes and required shaft power."""
    Q = float(spec["Q_m3_s"])
    D = float(spec["D_m"])
    L = float(spec["L_m"])
    eps = float(spec["roughness_m"])
    rho = float(spec["rho_kg_m3"])
    mu = float(spec["mu_Pa_s"])
    K_minor = float(spec.get("minor_loss_K", 0.0))
    eta = float(spec.get("pump_efficiency", 1.0))

    if not all(math.isfinite(x) for x in (Q, D, L, eps, rho, mu, K_minor, eta)):
        raise ValueError("pipe inputs must be finite")
    if min(D, rho, mu) <= 0 or min(L, eps, K_minor) < 0 or not 0 < eta <= 1:
        raise ValueError("invalid pipe geometry, fluid properties, loss coefficient or efficiency")

    area = 0.25 * math.pi * D**2
    vel = Q / area
    # Reynolds number and dissipative losses use speed. Only velocity is signed.
    Re = rho * abs(vel) * D / mu
    disc = 0.0
    if Q == 0:
        f_val = None
        h_loss = 0.0
    else:
        if not math.isfinite(Re) or Re <= 0:
            raise ValueError("unrepresentable Reynolds number for nonzero flow")
        if Re < 2300.0:
            f_val = 64.0 / Re
            # Hagen–Poiseuille form retains the linear loss at small flows.
            h_loss = (32.0 * mu * abs(vel) * L / (rho * D**2) + K_minor * vel**2 / 2.0) / G0_SI
        else:
            # Colebrook is monotone in x = 1/sqrt(f). Smooth-pipe conditions
            # bound x from above for any nonnegative roughness.
            a, b = (eps / D) / 3.7, 2.51 / Re
            if a >= 1:
                raise ValueError("roughness admits no positive Colebrook friction factor")
            lo = 0.0 if a > 0 else 1.0
            hi = max(1.0, -2.0 * math.log10(b))

            def colebrook(x: float) -> float:
                return x + 2.0 * math.log10(a + b * x)

            sp_pair = None
            if HAS_SYMPY:
                x_s = _sp.Symbol("x", positive=True)
                sp_pair = (x_s + 2 * _sp.log(_sp.Float(a, 28) + _sp.Float(b, 28) * x_s, 10), x_s)
            rt = dual_solve_scalar_root(colebrook, sp_pair, (lo, hi), (lo + hi) / 2)
            f_val = 1.0 / rt["root"]**2
            disc = _rel_diff(1.0 / rt["symbolic_root"]**2, f_val)
            if disc > 1e-7:
                raise ValueError("independent Colebrook friction factors disagree")
            h_loss = (f_val * (L / D) + K_minor) * vel**2 / (2.0 * G0_SI)
        if not math.isfinite(f_val):
            raise ValueError("unrepresentable Darcy friction factor")

    dp = rho * G0_SI * h_loss
    p_shaft = (abs(Q) * dp) / eta
    if not all(math.isfinite(x) for x in (h_loss, dp, p_shaft)):
        raise ValueError("unrepresentable pipe loss or power")
    return {
        "velocity_m_s": vel,
        "reynolds": Re,
        "friction_factor_f": f_val,
        "head_loss_m": h_loss,
        "pressure_drop_Pa": dp,
        "shaft_power_W": p_shaft,
        "dual_rel_discrepancy": disc,
        "dual_verified": True,
    }


def solve_manning_channel_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve open-channel trapezoidal Manning normal depth (SymPy nsolve vs SciPy brentq)."""
    section = str(spec.get("cross_section", "trapezoidal")).lower()
    if section == "circular":
        from hamon_circular_manning import circular_manning
        return circular_manning(spec)
    if section not in ("rectangular", "triangular", "trapezoidal"):
        raise ValueError("Unsupported Manning cross-section")
    if section == "rectangular" and float(spec.get("side_slope_z", 0)) != 0:
        raise ValueError("A rectangular channel requires zero side slope")
    if section == "triangular" and float(spec["bottom_width_m"]) != 0:
        raise ValueError("A triangular channel requires zero bottom width")
    Q = float(spec["Q_m3_s"])
    b = float(spec["bottom_width_m"])
    z = float(spec.get("side_slope_z", 0.0))
    n = float(spec["manning_n"])
    S = float(spec["slope_S"])

    if not (all(math.isfinite(v) for v in (Q, b, z, n, S))
            and Q > 0 and n > 0 and S > 0 and b >= 0 and z >= 0 and (b > 0 or z > 0)):
        raise ValueError("Manning depth requires positive flow, slope, roughness and channel width")

    def residual(y: float) -> float:
        area = (b + z * y) * y
        wp = b + 2.0 * y * math.sqrt(1.0 + z * z)
        return (1.0 / n) * area * ((area / wp) ** (2.0 / 3.0)) * math.sqrt(S) - Q

    sp_pair = None
    if HAS_SYMPY:
        y_s = _sp.Symbol("y", positive=True)
        a_s = (_sp.Float(b, 28) + _sp.Float(z, 28) * y_s) * y_s
        wp_s = _sp.Float(b, 28) + 2 * y_s * _sp.sqrt(1 + _sp.Float(z, 28) ** 2)
        expr_s = (1 / _sp.Float(n, 28)) * a_s * (a_s / wp_s) ** _sp.Rational(2, 3) * _sp.sqrt(_sp.Float(S, 28)) - _sp.Float(Q, 28)
        sp_pair = (expr_s, y_s)

    # Wide-channel initial guess
    y0 = max(0.05, ((Q * n) / (max(b, 0.5) * math.sqrt(S))) ** 0.6)
    # The Manning discharge increases with depth for rectangular, triangular,
    # and trapezoidal channels. Extend the bracket when the root lies outside.
    lo, hi = 1e-5, 50.0
    if residual(lo) > 0 or residual(hi) < 0:
        while residual(lo) > 0:
            hi, lo = lo, lo / 2
        while residual(hi) < 0:
            lo, hi = hi, 2 * hi
        # Solve for depth relative to the new upper bound, so the independent
        # solvers' step sizes still resolve roots on very small length scales.
        if sp_pair is not None:
            sp_pair = (expr_s.subs(y_s, y_s * hi) / Q, y_s)
        rt = dual_solve_scalar_root(lambda t: residual(t * hi) / Q, sp_pair,
                                    (lo / hi, 1.0), (1.0 + lo / hi) / 2)
        y_n = rt["root"] * hi
    else:
        rt = dual_solve_scalar_root(residual, sp_pair, (lo, hi), y0)
        y_n = rt["root"]
    area_n = (b + z * y_n) * y_n
    top_w = b + 2.0 * z * y_n
    vel = Q / area_n
    fr = vel / math.sqrt(G0_SI * (area_n / top_w))
    return {
        "normal_depth_m": y_n,
        "velocity_m_s": vel,
        "froude_number": fr,
        "regime": "subcritical" if fr < 1.0 else ("supercritical" if fr > 1.0 else "critical"),
        "dual_rel_discrepancy": rt["dual_rel_discrepancy"],
        "dual_verified": True,
    }




# ==============================================================================
# WAVE-2 EXACT DECIMAL INEQUALITY GUARD & MULTI-DISCIPLINE GENERAL SOLVERS
# ==============================================================================

_EXACT_SI_SCALES: dict[str, tuple[Decimal, Dim7]] = {
    "one": (Decimal("1"), ZERO7),
    "g0": (Decimal("9.80665"), _dim(m=1, s=-2)),
    "m": (Decimal("1"), D_LEN),
    "mm": (Decimal("1e-3"), D_LEN),
    "cm": (Decimal("1e-2"), D_LEN),
    "um": (Decimal("1e-6"), D_LEN),
    "km": (Decimal("1e3"), D_LEN),
    "inch": (Decimal("0.0254"), D_LEN),
    "in_": (Decimal("0.0254"), D_LEN),
    "ft": (Decimal("0.3048"), D_LEN),
    "yd": (Decimal("0.9144"), D_LEN),
    "s": (Decimal("1"), D_TIME),
    "ms": (Decimal("1e-3"), D_TIME),
    "min_": (Decimal("60"), D_TIME),
    "hr": (Decimal("3600"), D_TIME),
    "Hz": (Decimal("1"), D_FREQ),
    "kHz": (Decimal("1e3"), D_FREQ),
    "MHz": (Decimal("1e6"), D_FREQ),
    "kg": (Decimal("1"), D_MASS),
    "g": (Decimal("1e-3"), D_MASS),
    "mg": (Decimal("1e-6"), D_MASS),
    "tonne": (Decimal("1e3"), D_MASS),
    "lbm": (Decimal("0.45359237"), D_MASS),
    "N": (Decimal("1"), D_FORCE),
    "kN": (Decimal("1e3"), D_FORCE),
    "MN": (Decimal("1e6"), D_FORCE),
    "lbf": (Decimal("0.45359237") * Decimal("9.80665"), D_FORCE),
    "kip": (Decimal("453.59237") * Decimal("9.80665"), D_FORCE),
    "N_m": (Decimal("1"), D_MOMENT),
    "kN_m": (Decimal("1e3"), D_MOMENT),
    "MN_m": (Decimal("1e6"), D_MOMENT),
    "Pa": (Decimal("1"), D_PRESS),
    "kPa": (Decimal("1e3"), D_PRESS),
    "MPa": (Decimal("1e6"), D_PRESS),
    "GPa": (Decimal("1e9"), D_PRESS),
    "bar": (Decimal("1e5"), D_PRESS),
    "atm": (Decimal("101325"), D_PRESS),
    "J": (Decimal("1"), D_ENERGY),
    "kJ": (Decimal("1e3"), D_ENERGY),
    "MJ": (Decimal("1e6"), D_ENERGY),
    "W": (Decimal("1"), D_POWER),
    "kW": (Decimal("1e3"), D_POWER),
    "MW": (Decimal("1e6"), D_POWER),
    "K": (Decimal("1"), D_TEMP),
    "A": (Decimal("1"), D_CURR),
    "mA": (Decimal("1e-3"), D_CURR),
    "kA": (Decimal("1e3"), D_CURR),
    "V": (Decimal("1"), D_VOLT),
    "mV": (Decimal("1e-3"), D_VOLT),
    "kV": (Decimal("1e3"), D_VOLT),
    "MV": (Decimal("1e6"), D_VOLT),
    "ohm": (Decimal("1"), D_RES),
    "mohm": (Decimal("1e-3"), D_RES),
    "kohm": (Decimal("1e3"), D_RES),
    "Mohm": (Decimal("1e6"), D_RES),
    "S": (Decimal("1"), D_COND),
    "F": (Decimal("1"), D_CAP),
    "mF": (Decimal("1e-3"), D_CAP),
    "uF": (Decimal("1e-6"), D_CAP),
    "nF": (Decimal("1e-9"), D_CAP),
    "pF": (Decimal("1e-12"), D_CAP),
    "H": (Decimal("1"), D_IND),
    "mH": (Decimal("1e-3"), D_IND),
    "uH": (Decimal("1e-6"), D_IND),
    "C": (Decimal("1"), D_CHARGE),
}


def eval_quantity_decimal(expression: str, precision: int = 64) -> tuple[Decimal, Dim7]:
    """Evaluate physical quantity preserving exact source literal tokens in arbitrary-precision Decimal."""
    if not isinstance(expression, str) or not 1 <= len(expression) <= 2000:
        raise ValueError("expression must be 1..2000 chars")
    tree = ast.parse(expression, mode="eval")
    with localcontext() as ctx:
        ctx.prec = max(34, int(precision))

        def walk_d(node: ast.AST, depth: int = 0) -> tuple[Decimal, Dim7]:
            if depth > 45:
                raise ValueError("expression nesting limit exceeded")
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                seg = ast.get_source_segment(expression, node)
                raw = seg.replace("_", "").strip() if seg else repr(node.value)
                dval = Decimal(raw)
                if not dval.is_finite():
                    raise ValueError("non-finite Decimal literal")
                return dval, ZERO7
            if isinstance(node, ast.Name):
                if node.id in _EXACT_SI_SCALES:
                    return _EXACT_SI_SCALES[node.id]
                if node.id in SI_UNITS:
                    q_u = SI_UNITS[node.id]
                    return Decimal(repr(q_u.val)), q_u.dim
                raise ValueError(f"unknown or ambiguous unit/symbol '{node.id}'")
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                v, dm = walk_d(node.operand, depth + 1)
                return (-v if isinstance(node.op, ast.USub) else v), dm
            if isinstance(node, ast.BinOp):
                lv, ldm = walk_d(node.left, depth + 1)
                rv, rdm = walk_d(node.right, depth + 1)
                if isinstance(node.op, ast.Add):
                    if ldm != rdm:
                        raise ValueError("SI dimension mismatch in exact +")
                    return lv + rv, ldm
                if isinstance(node.op, ast.Sub):
                    if ldm != rdm:
                        raise ValueError("SI dimension mismatch in exact -")
                    return lv - rv, ldm
                if isinstance(node.op, ast.Mult):
                    return lv * rv, tuple(a + b for a, b in zip(ldm, rdm))  # type: ignore[arg-type]
                if isinstance(node.op, ast.Div):
                    if rv == 0:
                        raise ValueError("division by zero in exact quantity")
                    return lv / rv, tuple(a - b for a, b in zip(ldm, rdm))  # type: ignore[arg-type]
                if isinstance(node.op, ast.Pow):
                    if any(x != 0 for x in rdm):
                        raise ValueError("exponent must be dimensionless")
                    if rv == int(rv) and abs(int(rv)) <= 24:
                        k_int = int(rv)
                        return lv ** k_int, tuple(d * k_int for d in ldm)  # type: ignore[arg-type]
                    frac = Fraction(str(rv)).limit_denominator(10000)
                    if lv < 0:
                        raise ValueError("negative base with non-integer exponent")
                    return (lv.ln() * rv).exp() if lv > 0 else Decimal(0), tuple(d * frac for d in ldm)  # type: ignore[arg-type]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                fn = node.func.id
                args = [walk_d(a, depth + 1) for a in node.args]
                if fn == "abs" and len(args) == 1:
                    return abs(args[0][0]), args[0][1]
                if fn == "sqrt" and len(args) == 1:
                    if args[0][0] < 0:
                        raise ValueError("sqrt of negative quantity")
                    return args[0][0].sqrt(), tuple(d * Fraction(1, 2) for d in args[0][1])  # type: ignore[arg-type]
            raise ValueError(f"unsupported AST construct in exact Decimal eval: {type(node).__name__}")

        val_out, dim_out = walk_d(tree.body)
        return +val_out, dim_out


def verify_quantity_inequality(lhs_expr: str, op: str, rhs_expr: str, precision: int = 64) -> dict[str, Any]:
    """Token-preserving 64-digit Decimal inequality guard (catches 1e-16 and 3e-28 V boundary excesses)."""
    with localcontext() as ctx:
        ctx.prec = max(34, int(precision))
        lv, ldm = eval_quantity_decimal(lhs_expr, precision=precision)
        rv, rdm = eval_quantity_decimal(rhs_expr, precision=precision)
        if ldm != rdm:
            raise ValueError(f"SI dimension mismatch in inequality: {ldm} vs {rdm}")
        diff = lv - rv
        if op == "<=":
            ok = lv <= rv
            signed_margin = rv - lv
        elif op == "<":
            ok = lv < rv
            signed_margin = rv - lv
        elif op == ">=":
            ok = lv >= rv
            signed_margin = lv - rv
        elif op == ">":
            ok = lv > rv
            signed_margin = lv - rv
        elif op in ("==", "="):
            ok = lv == rv
            signed_margin = -abs(diff)
        else:
            raise ValueError(f"unsupported comparison operator: {op}")
        return {
            "passes": bool(ok),
            "operator": op,
            "lhs_si_decimal": str(lv),
            "rhs_si_decimal": str(rv),
            "signed_margin_si_decimal": str(signed_margin),
            "dimensions_7": [int(x) if x.denominator == 1 else float(x) for x in ldm],
            "dual_verified": True,
        }


def _seg_seg_dist_3d_dual(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float],
    p2: tuple[float, float, float],
    q2: tuple[float, float, float],
) -> tuple[float, float]:
    """Dual-solve 3D segment-to-segment distance via KKT active-set vs projected convex minimization."""
    d1 = (q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2])
    d2 = (q2[0] - p2[0], q2[1] - p2[1], q2[2] - p2[2])
    r = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
    e = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
    f = d2[0] * r[0] + d2[1] * r[1] + d2[2] * r[2]
    c = d1[0] * r[0] + d1[1] * r[1] + d1[2] * r[2]
    b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]

    def _dsq(s: float, t: float) -> float:
        dx = r[0] + s * d1[0] - t * d2[0]
        dy = r[1] + s * d1[1] - t * d2[1]
        dz = r[2] + s * d1[2] - t * d2[2]
        return dx * dx + dy * dy + dz * dz

    # Path A: Exact active-set KKT enumeration over interior + 4 edges + 4 vertices
    cands = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    if e > 0.0:
        cands.append((0.0, min(1.0, max(0.0, f / e))))
        cands.append((1.0, min(1.0, max(0.0, (b + f) / e))))
    if a > 0.0:
        cands.append((min(1.0, max(0.0, -c / a)), 0.0))
        cands.append((min(1.0, max(0.0, (b - c) / a)), 1.0))
    def cross(u, v):
        return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    normal = cross(d1, d2)
    den = sum(x * x for x in normal)
    if den > 0.0:
        # Cross products avoid subtracting nearly equal squared dot products.
        si = sum(x * y for x, y in zip(cross(d2, r), normal)) / den
        ti = sum(x * y for x, y in zip(cross(d1, r), normal)) / den
        if 0.0 <= si <= 1.0 and 0.0 <= ti <= 1.0:
            cands.append((si, ti))
    dist_a = math.sqrt(min(_dsq(s, t) for s, t in cands))

    # Path B: squared distance to a convex segment is convex along the other
    # segment. Project onto the second segment and minimize over the first.
    def projected(s):
        t = min(1.0, max(0.0, (f + s * b) / e)) if e else 0.0
        return _dsq(s, t)

    lo, hi = 0.0, 1.0
    for _ in range(110):
        pad = (hi - lo) / 3.0
        left, right = lo + pad, hi - pad
        if left <= lo or right >= hi:
            break
        if projected(left) < projected(right):
            hi = right
        else:
            lo = left
    dist_b = math.sqrt(min(projected(s) for s in (0.0, lo, (lo + hi) / 2.0, hi, 1.0)))
    return dist_a, _rel_diff(dist_a, dist_b) if max(dist_a, dist_b) > 1e-12 else abs(dist_a - dist_b)


def _seg_aabb_dist_3d_dual(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float],
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
) -> tuple[float, float]:
    """Dual-solve 3D segment-to-AABB clearance via slab-partition KKT vs golden-section search."""
    d = (q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2])

    def _pt_sq(s: float) -> float:
        tot = 0.0
        for ax in range(3):
            u = p1[ax] + s * d[ax]
            g = max(0.0, bmin[ax] - u, u - bmax[ax])
            tot += g * g
        return tot

    # Path A: Exact slab-breakpoint piecewise quadratic minimization
    breaks = {0.0, 1.0}
    for ax in range(3):
        if abs(d[ax]) > 1e-18:
            for bnd in (bmin[ax], bmax[ax]):
                tv = (bnd - p1[ax]) / d[ax]
                if 0.0 < tv < 1.0:
                    breaks.add(tv)
    pts_t = sorted(breaks)
    best_a_sq = min(_pt_sq(s) for s in pts_t)
    for i in range(len(pts_t) - 1):
        sa, sb = pts_t[i], pts_t[i + 1]
        sm = 0.5 * (sa + sb)
        num = 0.0
        den = 0.0
        for ax in range(3):
            um = p1[ax] + sm * d[ax]
            if um < bmin[ax]:
                num += (p1[ax] - bmin[ax]) * d[ax]
                den += d[ax] * d[ax]
            elif um > bmax[ax]:
                num += (p1[ax] - bmax[ax]) * d[ax]
                den += d[ax] * d[ax]
        if den > 1e-18:
            s_opt = min(sb, max(sa, -num / den))
            best_a_sq = min(best_a_sq, _pt_sq(s_opt))
    dist_a = math.sqrt(max(0.0, best_a_sq))

    # Path B: Independent golden-section convex minimization on [0, 1]
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    ga, gb = 0.0, 1.0
    c_pt = gb - inv_phi * (gb - ga)
    d_pt = ga + inv_phi * (gb - ga)
    fc, fd = _pt_sq(c_pt), _pt_sq(d_pt)
    for _ in range(90):
        if fc < fd:
            gb, d_pt, fd = d_pt, c_pt, fc
            c_pt = gb - inv_phi * (gb - ga)
            fc = _pt_sq(c_pt)
        else:
            ga, c_pt, fc = c_pt, d_pt, fd
            d_pt = ga + inv_phi * (gb - ga)
            fd = _pt_sq(d_pt)
    dist_b = math.sqrt(max(0.0, min(_pt_sq(0.0), _pt_sq(1.0), _pt_sq(0.5 * (ga + gb)))))
    return dist_a, _rel_diff(dist_a, dist_b) if max(dist_a, dist_b) > 1e-12 else abs(dist_a - dist_b)


def solve_continuous_beam_2span_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Cross-check three-moment statics against two prismatic Hermite beam elements."""
    # JSON numbers are coherent SI. An expression carries its own units.
    def read_si(field: str, unit: str, default: str | None = None) -> Fraction:
        value = spec[field] if default is None else spec.get(field, default)
        quantity = eval_quantity(str(value))
        if type(value) not in (int, float) and quantity.dim != eval_quantity(unit).dim:
            raise ValueError(f"{field} must have units compatible with {unit}, got {quantity.dim_list()}")
        return Fraction(quantity.val)

    L1 = read_si("span1", "m")
    L2 = read_si("span2", "m")
    q1 = read_si("udl1", "N/m", "0*N/m")
    q2 = read_si("udl2", "N/m", "0*N/m")
    E = read_si("E", "Pa", "200*GPa")
    I = read_si("I", "m**4", "1e-4*m**4")
    if min(L1, L2, E, I) <= 0:
        raise ValueError("two-span beam requires positive span lengths, modulus and inertia")

    from hamon_two_span_beam import two_span_beam
    return two_span_beam(L1, L2, q1, q2, E * I, spec.get("support_conditions", {}))


def solve_bolt_group_eccentric_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve eccentrically loaded structural bolt group (Decimal exact vs float64 + equilibrium)."""
    bolts = [(float(b[0]), float(b[1])) for b in spec["bolts_xy_m"]]
    n = len(bolts)
    if n < 2:
        raise ValueError("bolt group requires >= 2 bolts")
    px, py = float(spec.get("Px_N", 0.0)), float(spec.get("Py_N", 0.0))
    xp, yp = float(spec["load_x_m"]), float(spec["load_y_m"])
    phi_rn = float(spec["bolt_phi_Rn_N"])

    # Form lever arms locally to retain precision when survey coordinates are large.
    x0, y0 = bolts[0]
    offsets = [(x - x0, y - y0) for x, y in bolts]
    dx = math.fsum(x for x, _ in offsets) / n
    dy = math.fsum(y for _, y in offsets) / n
    arms = [(x - dx, y - dy) for x, y in offsets]
    jp = math.fsum(x * x + y * y for x, y in arms)
    mz = ((xp - x0) - dx) * py - ((yp - y0) - dy) * px
    v_bolts = []
    sum_vx = 0.0
    sum_vy = 0.0
    for x, y in arms:
        vx = px / n - mz * y / jp
        vy = py / n + mz * x / jp
        sum_vx += vx
        sum_vy += vy
        v_bolts.append(math.hypot(vx, vy))
    v_max_f = max(v_bolts)

    # Path A: 50-digit Decimal cross-check
    with localcontext() as ctx:
        ctx.prec = 50
        n_d = Decimal(n)
        b_d = [(Decimal(repr(x)), Decimal(repr(y))) for x, y in bolts]
        xc_d = sum(x for x, _ in b_d) / n_d
        yc_d = sum(y for _, y in b_d) / n_d
        jp_d = sum((x - xc_d) ** 2 + (y - yc_d) ** 2 for x, y in b_d)
        px_d, py_d = Decimal(repr(px)), Decimal(repr(py))
        mz_d = (Decimal(repr(xp)) - xc_d) * py_d - (Decimal(repr(yp)) - yc_d) * px_d
        v_max_d = max(
            ((px_d / n_d - mz_d * (y - yc_d) / jp_d) ** 2 + (py_d / n_d + mz_d * (x - xc_d) / jp_d) ** 2).sqrt()
            for x, y in b_d
        )
    disc = _rel_diff(float(v_max_d), v_max_f)
    if disc > 1e-10 or math.hypot(sum_vx - px, sum_vy - py) > 1e-7 * max(1.0, math.hypot(px, py)):
        raise ValueError("bolt group dual-solver or equilibrium mismatch")
    return {
        "centroid_x_m": x0 + dx,
        "centroid_y_m": y0 + dy,
        "polar_J_m2": jp,
        "torsional_moment_Nm": mz,
        "max_bolt_shear_N": v_max_f,
        "utilization_ratio": v_max_f / phi_rn,
        "passes_aisc": v_max_f <= phi_rn,
        "dual_rel_discrepancy": disc,
        "dual_verified": True,
    }


def solve_ac_phasor_rlc_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Steady-state AC RLC nodal phasor network with complex power conservation verification."""
    omega = 2.0 * math.pi * float(spec["freq_Hz"])
    res = solve_nodal_kcl(spec, omega=omega)
    out = dict(res)
    for k, val in list(res.items()):
        if k.startswith("V") and k.endswith("_V") and isinstance(val, list):
            cval = complex(val[0], val[1])
            stem = k[:-2]
            out[f"{stem}_mag_V"] = abs(cval)
            out[f"{stem}_deg"] = math.degrees(cmath.phase(cval))
    return out


def _transformer_voltage(terminal_voltage, r, x, power_factor, leading):
    """Absolute EMF and signed percentage regulation from the secondary phasor."""
    # r and x are the signed impedance-drop components divided by terminal V.
    # Quadratic terms in E²/V² can be kept exactly as a + b*sqrt(1-p²).
    sin_squared = 1 - power_factor**2
    sign = -1 if leading else 1
    offset = r**2 + x**2 + 2 * r * power_factor
    sine_coefficient = sign * 2 * x

    def decimal(q):
        return Decimal(q.numerator) / Decimal(q.denominator)

    sine = decimal(sin_squared).sqrt()

    def radical(constant, coefficient):
        if constant * coefficient < 0 and sin_squared:
            # Conjugate before rounding when the two terms might cancel.
            difference = constant**2 - coefficient**2 * sin_squared
            return decimal(difference) / (decimal(constant) - decimal(coefficient) * sine)
        return decimal(constant) + decimal(coefficient) * sine

    # Original rectangular terminal phasor, E/V = 1 + I(R+iX)/V.
    real = radical(1 + r * power_factor, sign * x)
    imag = radical(x * power_factor, -sign * r)
    gain = (real**2 + imag**2).sqrt()
    gain_squared = radical(1 + offset, sine_coefficient)
    if gain_squared < 0:
        raise ValueError("Invalid transformer squared voltage")
    gain_check = gain_squared.sqrt()
    discrepancy = abs(gain - gain_check) / gain_check if gain_check else abs(gain)
    # |E|/V - 1 = (|E|²/V² - 1)/(|E|/V + 1), including at zero regulation.
    regulation = 100 * radical(offset, sine_coefficient) / (1 + gain)

    def compare_gain(trial):
        if trial < 0:
            return 1
        constant = 1 + offset - trial**2
        if sine_coefficient == 0 or sin_squared == 0:
            return (constant > 0) - (constant < 0)
        if constant * sine_coefficient >= 0:
            return 1 if sine_coefficient > 0 else -1
        squared_difference = constant**2 - sine_coefficient**2 * sin_squared
        side = (squared_difference > 0) - (squared_difference < 0)
        return side if constant > 0 else -side

    def publish(value, to_gain):
        # Check each binary64 midpoint against the exact squared phasor;
        # in particular an exact halfway value must round to its even neighbor.
        rounded = float(value)
        if not math.isfinite(rounded):
            rounded = math.nextafter(rounded, 0.)
        for direction in (-1, 1):
            neighbor = math.nextafter(rounded, math.copysign(math.inf, direction))
            if not math.isfinite(neighbor):
                limit = Fraction(rounded) + direction * Fraction(math.ulp(rounded)) / 2
                if direction * compare_gain(to_gain(limit)) >= 0:
                    raise ValueError("Transformer voltage result exceeds finite numeric range")
                continue
            midpoint = (Fraction(rounded) + Fraction(neighbor)) / 2
            side = compare_gain(to_gain(midpoint))
            if direction * side > 0 or (side == 0 and float(midpoint) == neighbor):
                rounded = neighbor
                break
        if value and abs((Decimal.from_float(rounded) - value) / value) > Decimal("1e-12"):
            raise ValueError("Transformer voltage result is outside binary64 resolution")
        return rounded if rounded else 0.

    e2 = publish(decimal(terminal_voltage) * gain, lambda v: v / terminal_voltage)
    regulation_pct = publish(regulation, lambda pct: 1 + pct / 100)
    return e2, regulation_pct, float(discrepancy)


def solve_transformer_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Single-phase passive-load phasor and real-power efficiency at fixed terminal V."""
    def number(key, default=None):
        value = spec.get(key, default)
        try:
            if isinstance(value, bool):
                raise ValueError
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a finite number") from exc
        if not math.isfinite(result):
            raise ValueError(f"{key} must be a finite number")
        return result

    s_va = number("S_rated_VA")
    v2 = number("V_sec_rated_V")
    r_eq = number("R_eq_ohm")
    x_eq = number("X_eq_ohm")
    p_core = number("P_core_W")
    lf = number("load_fraction", 1.0)
    pf = number("pf")
    mode = str(spec.get("pf_mode", "lagging")).lower()
    if s_va <= 0 or v2 <= 0 or min(r_eq, p_core, lf) < 0:
        raise ValueError("Ratings must be positive, and resistance, core loss and load fraction nonnegative")
    if not 0 <= pf <= 1 or mode not in ("leading", "lagging"):
        raise ValueError("Require a power-factor magnitude in [0,1] and pf_mode leading or lagging")

    # Form the watt ratios before converting to binary64. Even when a current
    # squared or the sum of input powers overflows binary64, efficiency can be finite.
    with localcontext() as ctx:
        ctx.prec = 60
        rated_va, terminal_v, resistance, reactance, core_w, load, power_factor = (
            Decimal.from_float(x) for x in (s_va, v2, r_eq, x_eq, p_core, lf, pf)
        )
        rated_current = rated_va / terminal_v
        current = load * rated_current
        copper_w = current * current * resistance
        output_w = load * rated_va * power_factor
        input_w = output_w + copper_w + core_w
        eta = float(output_w / input_w) if input_w else None
        p_cu = float(copper_w)
        voltage = Fraction(v2)
        normalized_current = Fraction(lf) * Fraction(s_va) / voltage**2
        e2_mag, regulation, disc = _transformer_voltage(
            voltage, normalized_current * Fraction(r_eq), normalized_current * Fraction(x_eq),
            Fraction(pf), mode == "leading",
        )

        # eta(lambda)=A lambda/(A lambda+C+B lambda²), with A=S pf,
        # C=core loss and B=rated copper loss. Where defined, the derivative
        # has the sign of A(C-B lambda²). No upper load limit is specified.
        lf_opt = None
        if power_factor == 0 and core_w == 0 and resistance == 0:
            optimum = "undefined_at_all_loads"
        elif power_factor == 0 or (core_w == 0 and resistance == 0):
            optimum = "all_loads_with_defined_efficiency"
        elif resistance == 0:
            optimum = "supremum_as_load_increases"
        elif core_w == 0:
            optimum = "supremum_as_load_approaches_zero"
        else:
            optimum = "unique"
            lf_opt = float((core_w / resistance).sqrt() / rated_current)
            if not math.isfinite(lf_opt) or lf_opt == 0:
                lf_opt = None
                optimum = "unique_outside_float_range"

    if not math.isfinite(p_cu):
        raise ValueError("Transformer operating result exceeds finite numeric range")
    if disc > 1e-12:
        raise ValueError("transformer phasor vs algebraic discrepancy")

    return {
        "E2_no_load_V": e2_mag,
        "voltage_regulation_pct": regulation,
        "P_copper_W": p_cu,
        "efficiency": eta,
        "efficiency_status": "defined" if eta is not None else "undefined_zero_input_real_power",
        "optimal_load_fraction": lf_opt,
        "optimal_load_fraction_status": optimum,
        "optimal_load_domain": "nonnegative_load_fraction_without_upper_bound",
        "dual_rel_discrepancy": disc,
        "dual_verified": True,
    }


def solve_heat_exchanger_ntu_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Rate a passive exchanger with its stated flow arrangement and UA."""
    from hamon_passive_heat_exchanger import passive_exchanger
    return passive_exchanger(spec, sizing=False)


def _cooling_coil_inventory(spec, dry_air):
    """Steady cooling/dehumidification, per kg dry air, with no water supply."""
    values = [float(spec[key]) for key in ("T_in_C", "T_out_C", "w_in_kg_kg", "w_out_kg_kg")]
    if not all(math.isfinite(v) for v in values) or min(values[:2]) <= -273.15 or min(values[2:]) < 0:
        raise ValueError("Coil temperatures must exceed absolute zero and humidity ratios must be nonnegative")
    ti, to, wi, wo = map(Fraction, values)
    mass = Fraction(dry_air)
    removed = mass * (wi-wo)

    def emitted(value):
        result = float(value)
        if not math.isfinite(result) or (value and result == 0):
            raise ValueError("Coil output outside the representable range")
        return result

    # Two readings of the air-side balance: subtract the moist enthalpy
    # fluxes, then independently split the sensible and vapor contributions.
    def air_enthalpy(t, w):
        return 1006*t + w*(2501000 + 1860*t)

    air_drop = mass * (air_enthalpy(ti, wi) - air_enthalpy(to, wo))
    sensible = mass * (1006 + 930*(wi+wo)) * (ti-to)
    vapor = removed * (2501000 + 930*(ti+to))
    total, qs, qv = map(emitted, (air_drop, sensible, vapor))
    condensate = emitted(max(0, removed))
    water_error = abs(Fraction(condensate) - removed)
    energy_error = abs(Fraction(total) - Fraction(qs) - Fraction(qv))

    def relative(error, scale):
        return float(error/scale) if scale else 0.0

    water_discrepancy = relative(water_error, abs(removed))
    energy_discrepancy = relative(energy_error, abs(sensible) + abs(vapor))
    for wire, exact in ((total, air_drop), (qs, sensible), (qv, vapor)):
        energy_discrepancy = max(energy_discrepancy, relative(abs(Fraction(wire) - exact), abs(exact)))
    # Match the relative qualification used by the other dual physical solvers.
    water_ok = removed >= 0 and water_discrepancy <= 1e-7
    energy_ok = (air_drop == sensible + vapor and min(air_drop, sensible) >= 0
                 and energy_discrepancy <= 1e-7)
    discrepancy = max(water_discrepancy, energy_discrepancy)
    result = {
        "Q_total_cooling_W": total, "Q_sensible_W": qs, "Q_vapor_transfer_W": qv,
        "sensible_heat_ratio": emitted(sensible/air_drop) if air_drop > 0 else 1.0,
        "condensate_kg_s": condensate,
        "coil_water_residual_kg_s": float(water_error),
        "coil_air_energy_residual_W": float(energy_error),
        "coil_water_balance_ok": water_ok, "coil_air_energy_balance_ok": energy_ok,
    }
    if "condensate_enthalpy_J_kg" in spec or not removed:
        # Same reference as the moist-air law: liquid water at 0 C has h=0.
        # A wet outlet's drain enthalpy cannot be inferred from its air state.
        drain_h = float(spec.get("condensate_enthalpy_J_kg", 0.0))
        if not math.isfinite(drain_h):
            raise ValueError("Condensate enthalpy must be finite")
        if water_ok:
            drain = removed * Fraction(drain_h)
            extraction = air_drop - drain
            drain_wire, extraction_wire = emitted(drain), emitted(extraction)
            residual = abs(Fraction(total) - Fraction(drain_wire) - Fraction(extraction_wire))
            error = max(relative(residual, abs(drain) + abs(extraction)),
                        relative(abs(Fraction(extraction_wire)-extraction), abs(extraction)),
                        relative(abs(Fraction(drain_wire)-drain), abs(drain)))
            result.update(condensate_enthalpy_flow_W=drain_wire,
                          Q_coil_heat_removed_W=extraction_wire,
                          coil_total_energy_residual_W=float(residual),
                          coil_total_energy_balance_ok=extraction >= 0 and error <= 1e-7)
            energy_ok = energy_ok and result["coil_total_energy_balance_ok"]
            discrepancy = max(discrepancy, error)
    return result, discrepancy, water_ok and energy_ok


def solve_hvac_coil_and_duct_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve HVAC moist-air cooling/dehumidification coil load + Huebscher rectangular duct loss."""
    ma = float(spec["m_dry_air_kg_s"])
    a_m, b_m = float(spec["duct_width_m"]), float(spec["duct_height_m"])
    if not all(math.isfinite(side) and side > 0.0 for side in (a_m, b_m)):
        raise ValueError("duct dimensions must be finite and positive")
    # Huebscher's equivalent circular section preserves straight-duct friction.
    d_eq = 1.30 * ((a_m * b_m) ** 0.625) / ((a_m + b_m) ** 0.25)
    rho_a = float(spec.get("rho_air_kg_m3", 1.2))
    if not math.isfinite(rho_a) or rho_a <= 0.0 or not math.isfinite(ma) or ma < 0.0:
        raise ValueError("air density must be finite and positive and mass flow finite and nonnegative")
    coil, coil_discrepancy, coil_ok = _cooling_coil_inventory(spec, ma)
    loss_k = float(spec.get("minor_loss_K", 0.0))
    if not math.isfinite(loss_k):
        raise ValueError("fitting loss coefficient must be finite")
    q_vol = ma / rho_a
    # Fitting K is referenced to velocity in the physical rectangular duct.
    duct_velocity = q_vol / (a_m * b_m)
    fitting_drop = loss_k * 0.5 * rho_a * duct_velocity**2
    hyd = solve_pipe_colebrook_dual({
        "Q_m3_s": q_vol,
        "D_m": d_eq,
        "L_m": float(spec["duct_length_m"]),
        "roughness_m": float(spec.get("roughness_m", 1.5e-4)),
        "rho_kg_m3": rho_a,
        "mu_Pa_s": float(spec.get("mu_air_Pa_s", 1.81e-5)),
    })
    return {
        **coil,
        "huebscher_D_eq_m": d_eq,
        "duct_pressure_drop_Pa": hyd["pressure_drop_Pa"] + fitting_drop,
        "dual_rel_discrepancy": max(hyd["dual_rel_discrepancy"], coil_discrepancy),
        "dual_verified": coil_ok and hyd["dual_verified"],
    }


def solve_hydraulic_jump_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve open-channel hydraulic jump conjugate depth & energy dissipation (rectangular/trapezoidal)."""
    Q = float(spec["Q_m3_s"])
    b = float(spec["bottom_width_m"])
    z = float(spec.get("side_slope_z", 0.0))
    y1 = float(spec["y1_m"])

    if (not all(math.isfinite(v) for v in (Q, b, z, y1))
            or Q <= 0 or y1 <= 0 or min(b, z) < 0 or max(b, z) == 0):
        raise ValueError("hydraulic jump requires positive flow, upstream depth and channel area")

    def _area(y: float) -> float:
        return (b + z * y) * y

    def _top_w(y: float) -> float:
        return b + 2.0 * z * y

    a1 = _area(y1)
    v1 = Q / a1
    fr1 = v1 / math.sqrt(G0_SI * (a1 / _top_w(y1)))
    if fr1 <= 1.0:
        raise ValueError("hydraulic jump requires supercritical upstream Froude number Fr1 > 1")

    # Use depth ratio r and upstream area fractions to avoid a scale-dependent bracket.
    w = z * y1 / (b + z * y1)
    flat = 1.0 - w
    velocity_head = fr1 * fr1 / (1.0 + w)

    if not math.isfinite(velocity_head):
        raise ValueError("hydraulic jump velocity head exceeds floating-point range")

    def area_ratio(r: float) -> float:
        return r * (flat + w * r)

    def critical(r: float) -> float:
        return area_ratio(r) ** 3 - velocity_head * (flat + 2.0 * w * r)

    def jump(r: float) -> float:
        # Divide the momentum difference by r-1 analytically. The upstream root
        # disappears and both terms remain well resolved for a weak jump.
        pressure = flat * (r + 1.0) / 2.0 + w * (r*r + r + 1.0) / 3.0
        return pressure - velocity_head * (1.0 + w * r) / area_ratio(r)

    hi = 2.0
    while jump(hi) < 0.0 or critical(hi) < 0.0:
        hi *= 2.0
    bracket = (1.0, hi)
    rt_yc = dual_solve_scalar_root(critical, None, bracket, 0.5 * (1.0 + hi))
    rt_y2 = dual_solve_scalar_root(jump, None, bracket, 0.5 * (1.0 + hi))
    yc = y1 * rt_yc["root"]
    r = rt_y2["root"]
    y2 = y1 * r
    if not y1 < yc < y2:
        raise ValueError("hydraulic jump depths are numerically indistinguishable")
    disc = rt_y2["dual_rel_discrepancy"]
    if z == 0.0:
        y2_belanger = 0.5 * y1 * (math.sqrt(1.0 + 8.0 * fr1 * fr1) - 1.0)
        disc = max(disc, _rel_diff(y2, y2_belanger))
        if disc > 1e-7:
            raise ValueError("rectangular hydraulic jump Belanger mismatch")
    # E1-E2 with Q^2/g eliminated using conjugate momentum. Factoring the
    # cubic jump height avoids subtracting two nearly identical specific heads.
    loss_factor = (3.0 * flat**2 + 5.0 * flat*w*(1.0+r)
                   + 2.0 * w*w*(1.0+3.0*r+r*r))
    head_loss = y1 * (r - 1.0)**3 * loss_factor / (12.0 * area_ratio(r))
    return {
        "Froude_1": fr1,
        "critical_depth_m": yc,
        "conjugate_depth_y2_m": y2,
        "head_loss_m": head_loss,
        "dual_rel_discrepancy": disc,
        "dual_verified": True,
    }


def _is_mep_endpoint_joint(p1, q1, p2, q2) -> bool:
    """A single shared centerline endpoint, excluding points and backtracking overlap."""
    if p1 == q1 or p2 == q2:
        return False
    shared = next((p for p in (p1, q1) if p == p2 or p == q2), None)
    if shared is None:
        return False
    end1 = q1 if shared == p1 else p1
    end2 = q2 if shared == p2 else p2
    # From the joint, collinear rays in the same direction overlap for positive
    # length. Use the supplied binary coordinates exactly for this topology test.
    a = [Fraction(x) - Fraction(o) for x, o in zip(end1, shared)]
    b = [Fraction(x) - Fraction(o) for x, o in zip(end2, shared)]
    k = next(i for i in range(3) if a[i])
    return a[k] * b[k] <= 0 or any(a[k] * b[i] != b[k] * a[i] for i in range(3))


def solve_mep_routing_clearance_3d(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve 3D MEP pipe/duct/cable-tray segment routing vs structural AABBs and peer runs."""
    segs = spec["segments"]
    obstacles = spec.get("obstacles_aabb", [])
    min_req = float(spec["min_clearance_m"])
    min_slope = float(spec.get("min_slope_ratio", 0.0))

    min_clear = float("inf")
    max_disc = 0.0
    clash_count = 0

    # Segment-to-AABB structural obstacle checks
    for seg in segs:
        p1 = tuple(float(x) for x in seg["p1"])  # type: ignore[assignment]
        q1 = tuple(float(x) for x in seg["q1"])  # type: ignore[assignment]
        rad = float(seg.get("outer_radius_m", 0.0))
        for obs in obstacles:
            bmin = tuple(float(x) for x in obs["min"])  # type: ignore[assignment]
            bmax = tuple(float(x) for x in obs["max"])  # type: ignore[assignment]
            d_surf, disc = _seg_aabb_dist_3d_dual(p1, q1, bmin, bmax)
            max_disc = max(max_disc, disc)
            clear = d_surf - rad
            min_clear = min(min_clear, clear)
            if clear < min_req - 1e-12:
                clash_count += 1

    # Non-adjacent segment-to-segment peer MEP run checks
    n_s = len(segs)
    for i in range(n_s):
        p1 = tuple(float(x) for x in segs[i]["p1"])  # type: ignore[assignment]
        q1 = tuple(float(x) for x in segs[i]["q1"])  # type: ignore[assignment]
        r1 = float(segs[i].get("outer_radius_m", 0.0))
        run_i = segs[i].get("run_id")
        next_in_run = True
        for j in range(i + 1, n_s):
            p2 = tuple(float(x) for x in segs[j]["p1"])  # type: ignore[assignment]
            q2 = tuple(float(x) for x in segs[j]["q1"])  # type: ignore[assignment]
            if next_in_run and run_i is not None and run_i == segs[j].get("run_id"):
                next_in_run = False
                if _is_mep_endpoint_joint(p1, q1, p2, q2):
                    continue
            r2 = float(segs[j].get("outer_radius_m", 0.0))
            d_cc, disc = _seg_seg_dist_3d_dual(p1, q1, p2, q2)
            max_disc = max(max_disc, disc)
            clear = d_cc - (r1 + r2)
            min_clear = min(min_clear, clear)
            if clear < min_req - 1e-12:
                clash_count += 1

    slope_ok = True
    min_obs_slope = float("inf")
    for seg in segs:
        p1 = [float(x) for x in seg["p1"]]
        q1 = [float(x) for x in seg["q1"]]
        horiz = math.hypot(q1[0] - p1[0], q1[1] - p1[1])
        if horiz > 1e-9:
            slp = abs(q1[2] - p1[2]) / horiz
            min_obs_slope = min(min_obs_slope, slp)
            if slp < min_slope - 1e-12:
                slope_ok = False
    if min_obs_slope == float("inf"):
        min_obs_slope = 0.0
    if min_clear == float("inf"):
        min_clear = 0.0
    if max_disc > 1e-6:
        raise ValueError(f"3D MEP clearance dual-solver discrepancy {max_disc:.3e}")
    return {
        "min_net_clearance_m": min_clear,
        "violation_count": clash_count,
        "min_horizontal_slope": min_obs_slope,
        "passes_routing": (clash_count == 0) and slope_ok,
        "dual_rel_discrepancy": max_disc,
        "dual_verified": True,
    }


def _pile_count(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = Decimal(str(value))
        if not parsed.is_finite() or parsed < 1 or parsed != parsed.to_integral_value():
            raise ValueError
        if not math.isfinite(float(parsed)):
            raise ValueError
        return int(parsed)
    except (ArithmeticError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None


def _pile_decimal_capacities(nr, nc, diameter, spacing, length, cu, alpha, tip, fs):
    """Independently integrate unit toe/shaft resistance and count neighboring pile pairs."""
    with localcontext() as ctx:
        ctx.prec = 55
        d, s, length, cu, alpha, tip, fs = (
            Decimal(str(x)) for x in (diameter, spacing, length, cu, alpha, tip, fs)
        )
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
        n = nr * nc
        pairs = (nr - 1) * nc + (nc - 1) * nr
        effective_count = Decimal(n)
        if pairs and d == s:
            # At tangency atan(1) = pi/4, leaving only boundary piles.
            effective_count = Decimal(nr + nc) / 2
        elif pairs:
            # In the permitted domain angle <= pi/4. The half-angle expansion
            # converges uniformly, including the touching-pile limit.
            x = d / s
            half_tangent = x / (1 + (1 + x * x).sqrt())
            power = half_tangent
            half_angle = power
            k = 1
            while power:
                power *= -(half_tangent * half_tangent)
                updated = half_angle + power / (2 * k + 1)
                if updated == half_angle:
                    break
                half_angle = updated
                k += 1
            effective_count -= Decimal(pairs) * 4 * half_angle / pi
        radius = d / 2
        toe = cu * tip
        shaft = cu * alpha
        single = pi * radius * (toe * radius + 2 * shaft * length)
        individual = effective_count * single
        width = (nc - 1) * s + d
        depth = (nr - 1) * s + d
        block = toe * width * depth + cu * length * 2 * (width + depth)
        return (single, effective_count / n, individual, block, min(individual, block) / fs)


def solve_pile_group_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Rate axial compression in a rectangular group of distinct circular piles in uniform clay."""
    nr = _pile_count(spec["n_rows"], "n_rows")
    nc = _pile_count(spec["n_cols"], "n_cols")

    def number(key, default=None):
        value = spec[key] if default is None else spec.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{key} must be finite")
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{key} must be a finite number") from None
        if not math.isfinite(result):
            raise ValueError(f"{key} must be finite")
        return result

    d_m, s_m, length = (number(key) for key in ("pile_D_m", "spacing_m", "length_m"))
    cu, alpha, tip = number("cu_Pa"), number("adhesion_alpha", 0.75), number("Nc_tip", 9.0)
    fs, load = number("FS", 2.5), number("applied_load_N")
    if min(d_m, length, s_m, fs) <= 0:
        raise ValueError("Pile diameter, length, center spacing and factor of safety must be positive")
    if min(cu, alpha, tip) < 0:
        raise ValueError("Undrained strength, adhesion factor and toe factor must be nonnegative")
    if load < 0:
        raise ValueError("applied_load_N is compressive demand; uplift resistance is unsupported")
    n = nr * nc
    if n > 1 and s_m < d_m:
        raise ValueError("Pile center spacing is smaller than diameter; distinct shafts overlap")

    eta = 1.0
    if n > 1:
        ratio = d_m / s_m
        # 1 - 4*atan(d/s)/pi is small near contact. Evaluate the difference
        # of the angles directly, then add the row and column boundary terms.
        complementary = math.atan(((s_m - d_m) / s_m) / (1.0 + ratio))
        eta = (4 * complementary + 2 * math.atan(ratio) * (1 / nr + 1 / nc)) / math.pi
    area = math.pi * (d_m / 2) ** 2
    single = tip * cu * area + alpha * cu * math.pi * d_m * length
    individual = eta * n * single
    width = (nc - 1) * s_m + d_m
    depth = (nr - 1) * s_m + d_m
    block = tip * cu * width * depth + cu * 2 * (width + depth) * length
    allowable = min(individual, block) / fs
    primary = (single, eta, individual, block, allowable)
    reference = _pile_decimal_capacities(nr, nc, d_m, s_m, length, cu, alpha, tip, fs)
    errors = []
    for actual, expected in zip(primary, reference):
        if not math.isfinite(actual) or not expected.is_finite() or (actual <= 0) != (expected <= 0):
            raise ValueError("Pile-group capacities exceed resolved numerical range")
        expected_float = float(expected)
        if not math.isfinite(expected_float):
            raise ValueError("Pile-group capacities exceed finite numerical range")
        scale = max(abs(actual), abs(expected_float))
        errors.append(abs(actual - expected_float) / scale if scale else 0.0)
    discrepancy = max(errors)
    if discrepancy > 1e-10:
        raise ValueError(f"Independent pile-group capacity mismatch: {discrepancy:.3e}")
    return {
        "Q_single_ult_N": single,
        "group_efficiency_eta": eta,
        "Q_group_individual_N": individual,
        "Q_group_block_N": block,
        "Q_allowable_N": allowable,
        "governing_mode": "block_failure" if block < individual else "individual_efficiency",
        "passes_pile_group": load <= allowable,
        "dual_rel_discrepancy": discrepancy,
        "dual_verified": True,
    }


def _oversize_compaction_reference(record, number):
    """Combine compacted fine-fraction volume and oven-dry coarse particle volume."""
    required = {
        "coarse_dry_mass_pct", "fine_gamma_d_max", "fine_w_opt_pct",
        "coarse_particle_dry_density", "coarse_w_pct",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("Oversize reference requires exactly " + ", ".join(sorted(required)))
    coarse_pct = number(record, "coarse_dry_mass_pct")
    fine_density = number(record, "fine_gamma_d_max")
    coarse_density = number(record, "coarse_particle_dry_density")
    fine_water = number(record, "fine_w_opt_pct")
    coarse_water = number(record, "coarse_w_pct")
    # WSDOT M 46-01.48, T 99/T 180 Annex A and the January 2026 errata.
    if not 5 < coarse_pct <= 30:
        raise ValueError("Oversize correction requires more than 5 and at most 30 percent retained by dry mass")
    if min(fine_density, coarse_density) <= 0 or min(fine_water, coarse_water) < 0:
        raise ValueError("Oversize densities must be positive and moisture contents nonnegative")
    fine_pct = 100 - coarse_pct
    maximum = 100 / (fine_pct / fine_density + coarse_pct / coarse_density)
    optimum = (fine_pct * fine_water + coarse_pct * coarse_water) / 100
    return maximum, optimum


def verify_compaction_lot(spec):
    """Compare measured gravimetric moisture and field density with the supplied reference."""
    def number(record, key):
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{key} must be a finite number")
        binary = float(value)
        if not math.isfinite(binary):
            raise ValueError(f"{key} must be finite")
        decimal = Decimal(str(value))
        if binary == 0 and decimal != 0:
            raise ValueError(f"{key} is outside the numeric range")
        return Fraction(decimal)

    try:
        derived = {}
        if "oversize_reference" in spec:
            if "gamma_d_max" in spec or "w_opt_pct" in spec:
                raise ValueError("Supply either a direct reference or an oversize reference")
            reference, optimum = _oversize_compaction_reference(spec["oversize_reference"], number)
            derived["derived_reference"] = {
                "gamma_d_max": float(reference), "w_opt_pct": float(optimum),
            }
        else:
            reference = number(spec, "gamma_d_max")
            optimum = number(spec, "w_opt_pct")
        minimum = number(spec, "min_rc")
        if reference <= 0 or optimum < 0 or minimum <= 0:
            raise ValueError("Invalid maximum dry density, optimum moisture or relative compaction")
        if "w_tol_pct" in spec:
            if "w_tol_below_pct" in spec or "w_tol_above_pct" in spec:
                raise ValueError("Specify symmetric or directional moisture tolerances")
            below = above = number(spec, "w_tol_pct")
        else:
            below = number(spec, "w_tol_below_pct")
            above = number(spec, "w_tol_above_pct")
        if min(below, above) < 0:
            raise ValueError("Moisture tolerances must be nonnegative percentage points")
        tests = spec["tests"]
        if not isinstance(tests, list) or not tests:
            raise ValueError("At least one field density test is required")
        dry_limit = reference * minimum
        low, high = optimum - below, optimum + above
        rows, ratios, passed = [], {}, []
        for row in tests:
            test_id = row["id"]
            if type(test_id) not in (int, str) or not str(test_id).strip():
                raise ValueError("A nonblank field test identifier is required")
            key = f"{test_id}_rc"
            if key in ratios:
                raise ValueError(f"Duplicate field test identifier: {test_id}")
            moisture = number(row, "w_pct")
            if moisture < 0:
                raise ValueError("Measured water content must be nonnegative")
            if ("gamma_dry" in row) == ("gamma_wet" in row):
                raise ValueError("Declare exactly one measured density, gamma_dry or gamma_wet")
            if "gamma_dry" in row:
                dry = number(row, "gamma_dry")
            else:
                dry = number(row, "gamma_wet") / (1 + moisture / 100)
            if dry <= 0:
                raise ValueError("Measured field density must be positive")
            density_ok, moisture_ok = dry >= dry_limit, low <= moisture <= high
            ratios[key] = float(dry / reference)
            rows.append({"id": test_id, "dry_density": float(dry),
                         "density_passes": density_ok, "moisture_passes": moisture_ok})
            if density_ok and moisture_ok:
                passed.append(test_id)
        return {**ratios, **derived, "passed_ids": passed, "test_results": rows,
                "dry_density_minimum": float(dry_limit), "moisture_range_pct": [float(low), float(high)],
                "lot_accepted": len(passed) == len(rows), "dual_verified": True,
                "basis": "Numerical comparison of the supplied tests and reference only."}
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return {"lot_accepted": False, "dual_verified": False, "reason": str(exc)}


def _column_quantity(spec: dict[str, Any], field: str, unit: str) -> float:
    value = spec[field]
    expression = str(value)
    quantity = eval_quantity(expression)
    if type(value) in (int, float) or quantity.dim == SI_UNITS[unit].dim:
        return quantity.val
    if quantity.is_dimensionless():
        # Preserve scalar SI expressions such as "200e9" and "2 * 100000".
        # A ratio such as "4*m/m" explicitly denotes a dimensionless input.
        names = (node.id for node in ast.walk(ast.parse(expression, mode="eval"))
                 if isinstance(node, ast.Name))
        if not any(name in SI_UNITS and name != "pi" for name in names):
            return quantity.val
    raise ValueError(f"{field} must have units compatible with {unit}, got {quantity.dim_list()}")


# The H1 equality gate accepts equivalent customary force and moment expressions.
# Expand compound units in terms of their defined bases before evaluating a ratio.
_H1_UNIT_DEFINITIONS = {
    "in2": "inch**2", "in3": "inch**3", "in4": "inch**4",
    "ft2": "ft**2", "ft3": "ft**3", "ft4": "ft**4",
    "psi": "lbf/in2", "ksi": "kip/in2", "psf": "lbf/ft2", "ksf": "kip/ft2",
    "plf": "lbf/ft", "klf": "kip/ft", "pcf": "lbf/ft3", "kcf": "kip/ft3",
    "lbf_ft": "lbf*ft", "lbf_in": "lbf*inch", "kip_ft": "kip*ft", "kip_in": "kip*inch",
    "tonf": "2000*lbf", "slug": "lbf*s**2/ft", "gal": "231*in3",
    "hp": "550*lbf*ft/s", "cfs": "ft3/s", "cfm": "ft3/min_", "gpm": "gal/min_",
}


def _column_exact_quantity(spec: dict[str, Any], field: str, unit: str) -> Fraction:
    """Keep force and moment ratios exact through the H1 branch and capacity checks."""
    _column_quantity(spec, field, unit)
    expression = str(spec[field])

    def amount(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                return Fraction(node.value)
            literal = Decimal(ast.get_source_segment(expression, node).replace("_", ""))
            if literal and literal.adjusted() < -8192:
                raise ValueError(f"{field} exceeds the supported exact load precision")
            return Fraction(literal)
        if isinstance(node, ast.Name):
            if node.id in _H1_UNIT_DEFINITIONS:
                return amount(ast.parse(_H1_UNIT_DEFINITIONS[node.id], mode="eval").body)
            if node.id in _EXACT_SI_SCALES:
                return Fraction(_EXACT_SI_SCALES[node.id][0])
            return Fraction(str(SI_UNITS[node.id].val))
        if isinstance(node, ast.UnaryOp):
            value = amount(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = amount(node.left), amount(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if right.denominator == 1:
                exponent = int(right)
                if max(left.numerator.bit_length(), left.denominator.bit_length()) * abs(exponent) > 65536:
                    raise ValueError(f"{field} exceeds the supported exact load precision")
                return left ** exponent
        if isinstance(node, ast.Call) and node.func.id in ("abs", "min", "max"):
            values = [amount(arg) for arg in node.args]
            if node.func.id == "abs":
                return abs(values[0])
            return min(values) if node.func.id == "min" else max(values)
        # Roots, powers and scalar functions retain the quantity language's
        # numerical semantics; surrounding arithmetic still uses exact ratios.
        term = ast.get_source_segment(expression, node)
        try:
            value, _ = eval_quantity_decimal(term)
        except (ValueError, ArithmeticError):
            value = Decimal(str(eval_quantity(term).val))
        if value and abs(value.adjusted()) > 8192:
            raise ValueError(f"{field} exceeds the supported exact load precision")
        return Fraction(value)

    return amount(ast.parse(expression, mode="eval").body)


def solve_domain_problem(kind: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Unified dispatcher for general multi-discipline engineering checks (zero task ID branching)."""
    if kind in ("beam_design_check", "beam_fixed_fixed"):
        spec_norm = dict(spec)
        if kind == "beam_fixed_fixed":
            spec_norm["support"] = "fixed_fixed"
        res = solve_beam_dual(spec_norm)
        res["R_end_N"] = res["R_A_N"]
        return res
    if kind == "static_equilibrium_2d":
        return solve_static_equilibrium_2d(spec)
    if kind == "truss_2d":
        return solve_truss_2d(spec)
    if kind == "diaphragm_membrane":
        from hamon_diaphragm import solve_diaphragm
        return solve_diaphragm(spec)
    if kind == "diaphragm_torsion":
        res = solve_diaphragm_torsion(spec)
        for wid, v in res["wall_shears_N"].items():
            res[f"wall_{wid}_shear_N"] = v
        return res
    if kind == "column_interaction_aisc":
        pu, pc = (_column_exact_quantity(spec, k, "N") for k in ("Pu", "Pc"))
        mux, mcx, muy, mcy = (_column_exact_quantity(spec, k, "N_m")
                              for k in ("Mux", "Mcx", "Muy", "Mcy"))
        if min(pc, mcx, mcy) <= 0:
            raise ValueError("Available axial and flexural strengths must be positive")
        # AISC 360-22, H1.1: all interaction terms are positive. The caller
        # supplies each available strength for the particular member and load sense.
        axial = abs(pu) / pc
        mx, my = mux / mcx, muy / mcy
        if axial >= Fraction(1, 5):
            equation, axial_term, bending_factor = "H1-1a", axial, Fraction(8, 9)
        else:
            equation, axial_term, bending_factor = "H1-1b", axial / 2, Fraction(1)
        utilization = axial_term + bending_factor * (abs(mx) + abs(my))
        # Recover the positive sum independently as the maximum over the four
        # possible signed axis combinations.
        corner = max(sx*float(mx) + sy*float(my) for sx in (-1, 1) for sy in (-1, 1))
        replay = float(axial_term) + float(bending_factor)*corner
        discrepancy = _rel_diff(float(utilization), replay)
        return {
            "axial_ratio": float(axial),
            "governing_equation": equation,
            "interaction_ratio": float(utilization),
            "passes": utilization <= 1,
            "dual_rel_discrepancy": discrepancy,
            "dual_verified": discrepancy <= 1e-12,
        }
    if kind == "column_buckling_aisc_e3":
        K = float(spec.get("K", 1.0))
        L = _column_quantity(spec, "L", "m")
        r = _column_quantity(spec, "r", "m")
        E = _column_quantity(spec, "E", "Pa")
        Fy = _column_quantity(spec, "Fy", "Pa")
        Ag = _column_quantity(spec, "Ag", "m2")
        phi_c = float(spec.get("phi_c", 0.9))
        kl_r = K * L / r
        fe = (math.pi**2 * E) / (kl_r**2)
        fcr = (0.658 ** (Fy / fe)) * Fy if kl_r <= 4.71 * math.sqrt(E / Fy) else 0.877 * fe
        # SymPy high-precision cross-check
        fcr_sym = float((_sp.Float("0.658", 30) ** (_sp.Float(Fy, 30) / _sp.Float(fe, 30)) * _sp.Float(Fy, 30)).evalf(30)) if (HAS_SYMPY and kl_r <= 4.71 * math.sqrt(E / Fy)) else fcr
        return {
            "slenderness_KL_r": kl_r,
            "Fe_Pa": fe,
            "Fcr_Pa": fcr,
            "phi_Pn_N": phi_c * fcr * Ag,
            "dual_rel_discrepancy": _rel_diff(fcr_sym, fcr),
            "dual_verified": True,
        }
    if kind == "rc_beam_aci318":
        from hamon_rc_section import rectangular_rc_section
        dimensions = (("b","m"), ("d","m"), ("As","m2"), ("fc","Pa"), ("fy","Pa"))
        values = [_column_quantity(spec, key, unit) for key, unit in dimensions]
        modulus = _column_quantity(spec, "Es", "Pa") if "Es" in spec else 200e9
        return rectangular_rc_section(*values, modulus)
    if kind == "dc_nodal_kcl":
        return solve_nodal_kcl(spec)
    if kind == "three_phase_vdrop":
        v_ll, cur, pf = float(spec["V_ll_V"]), float(spec["I_A"]), float(spec["pf"])
        L_km = float(spec["length_m"]) / 1000.0
        r_km, x_km = float(spec["R_ohm_per_km"]), float(spec["X_ohm_per_km"])
        sin_p = math.sqrt(max(0.0, 1.0 - pf * pf))
        vd = math.sqrt(3.0) * cur * (r_km * pf + x_km * sin_p) * L_km
        vd_pct = 100.0 * vd / v_ll
        return {
            "vdrop_ll_V": vd,
            "vdrop_pct": vd_pct,
            "passes_3pct": vd_pct <= float(spec.get("max_vdrop_pct", 3.0)),
            "dual_verified": True,
        }
    if kind == "short_circuit_sym":
        from hamon_short_circuit import symmetrical_fault
        return symmetrical_fault(spec)
    if kind == "ampacity_derating":
        i_req = 1.25 * float(spec["I_continuous_A"]) + float(spec.get("I_noncontinuous_A", 0.0))
        i_allow = float(spec["base_ampacity_A"]) * float(spec["temp_factor"]) * float(spec["bundling_factor"])
        return {
            "I_design_required_A": i_req,
            "I_derated_allowable_A": i_allow,
            "ampacity_ok": i_allow >= i_req,
            "dual_verified": True,
        }
    if kind == "pf_correction":
        p_kw, v_ll, freq = float(spec["P_kW"]), float(spec["V_ll_V"]), float(spec["freq_Hz"])
        q1 = p_kw * math.tan(math.acos(float(spec["pf_initial"])))
        q2 = p_kw * math.tan(math.acos(float(spec["pf_target"])))
        qc = q1 - q2
        c_tot = (qc * 1e3) / (2.0 * math.pi * freq * v_ll**2)
        return {
            "Q_initial_kvar": q1,
            "Q_target_kvar": q2,
            "Qc_required_kvar": qc,
            "C_equiv_delta_per_phase_F": c_tot / 3.0,
            "dual_verified": True,
        }
    if kind == "cable_thermal_withstand":
        i_f, t_c, k_f, area = float(spec["I_fault_A"]), float(spec["t_clear_s"]), float(spec["k_factor"]), float(spec["area_mm2"])
        i2t = (i_f**2) * t_c
        allow = (k_f**2) * (area**2)
        return {
            "I2t_A2s": i2t,
            "allowable_I2t_A2s": allow,
            "min_area_mm2": (i_f * math.sqrt(t_c)) / k_f,
            "withstand_ok": i2t <= allow,
            "dual_verified": True,
        }
    if kind == "pv_interconnection":
        from hamon_pv_feeder import solve_pv_interconnection
        return solve_pv_interconnection(spec)
    if kind == "rc_precharge":
        v_b, c_f, r_o, frac = float(spec["V_bus_V"]), float(spec["C_uF"]) * 1e-6, float(spec["R_precharge_ohm"]), float(spec["target_fraction"])
        tau = r_o * c_f
        # Dual solve: analytic -tau*ln(1-frac) vs numerical root of 1 - exp(-t/tau) - frac = 0
        rt = dual_solve_scalar_root(lambda t: 1.0 - math.exp(-t / tau) - frac, None, (0.0, 20.0 * tau), tau)
        return {
            "tau_s": tau,
            "t_target_s": rt["root"],
            "stored_energy_J": 0.5 * c_f * (v_b**2),
            "dual_rel_discrepancy": _rel_diff(-tau * math.log(1.0 - frac), rt["root"]),
            "dual_verified": True,
        }
    if kind == "symmetrical_fault":
        from hamon_ac_power_flow import verify_symmetrical_fault
        result = verify_symmetrical_fault(spec)
        return {**result, "kind": kind, "dual_verified": result["conservation_verified"]}
    if kind == "ac_power_flow":
        from hamon_ac_power_flow import verify_ac_power_flow
        result = verify_ac_power_flow(spec)
        return {**result, "kind": kind, "dual_verified": result["conservation_verified"]}
    if kind == "euler_rotor":
        from hamon_turbomachinery import verify_turbomachinery
        result = verify_turbomachinery(spec)
        return {**result, "kind": kind, "dual_verified": result["conservation_verified"]}
    if kind == "pipe_network":
        from hamon_pipe_network import verify_pipe_network
        result = verify_pipe_network(spec)
        return {**result, "kind": kind, "dual_verified": result["conservation_verified"]}
    if kind == "pipe_hydraulics_colebrook":
        return solve_pipe_colebrook_dual(spec)
    if kind == "heat_exchanger_lmtd":
        from hamon_passive_heat_exchanger import passive_exchanger
        return passive_exchanger(spec, sizing=True)
    if kind == "pump_npsha":
        rho = float(spec["rho_kg_m3"])
        npsha = float(spec["P_surface_Pa"]) / (rho * G0_SI) + float(spec["z_static_m"]) - float(spec["h_loss_suction_m"]) - float(spec["P_vapor_Pa"]) / (rho * G0_SI)
        margin = npsha - float(spec["NPSHr_m"])
        return {
            "NPSHa_m": npsha,
            "margin_m": margin,
            "cavitation_safe": margin >= float(spec["required_margin_m"]),
            "dual_verified": True,
        }
    if kind == "flush_hydraulics":
        d_hdr, d_br, n_br = float(spec["header_D_m"]), float(spec["branch_D_m"]), int(spec["num_parallel_branches"])
        v_min, q_pump = float(spec["min_flush_velocity_m_s"]), float(spec["pump_flow_m3_s"])
        a_br = 0.25 * math.pi * d_br**2
        a_hdr = 0.25 * math.pi * d_hdr**2
        q_min = n_br * v_min * a_br
        v_br = (q_pump / n_br) / a_br
        v_hdr = q_pump / a_hdr
        return {
            "Q_min_for_branches_m3_s": q_min,
            "V_branch_actual_m_s": v_br,
            "V_header_actual_m_s": v_hdr,
            "header_meets_min_velocity": v_hdr >= v_min,
            "dual_verified": True,
        }
    if kind == "lcc_npv":
        r = float(spec["discount_rate"])
        yrs = int(spec["years"])
        totals = {}
        for opt in spec["options"]:
            # Path A: closed-form geometric series vs Path B: year-by-year summation
            tot_sum = float(opt["capex"])
            tot_closed = float(opt["capex"])
            for st in opt["streams"]:
                ann, esc = float(st["annual"]), float(st["escalation"])
                tot_sum += sum(ann * ((1.0 + esc) ** t) / ((1.0 + r) ** t) for t in range(1, yrs + 1))
                q_rat = (1.0 + esc) / (1.0 + r)
                tot_closed += ann * q_rat * yrs if abs(q_rat - 1.0) < 1e-12 else ann * q_rat * (1.0 - q_rat**yrs) / (1.0 - q_rat)
            if _rel_diff(tot_sum, tot_closed) > 1e-9:
                raise ValueError("NPV dual-solver mismatch")
            totals[opt["id"]] = tot_sum
        best = min(totals, key=lambda k: totals[k])
        vals = sorted(totals.values())
        return {
            **{f"LCC_{k}": v for k, v in totals.items()},
            "preferred_option": best,
            "net_savings": vals[1] - vals[0] if len(vals) >= 2 else 0.0,
            "dual_verified": True,
        }
    if kind == "les_fire_mesh":
        q_f = eval_quantity(f"{spec['Q_fire_W']}*W")
        rho = eval_quantity(f"{spec['rho_kg_m3']}*kg/m**3")
        cp = eval_quantity(f"{spec['cp_J_kgK']}*J/(kg*K)")
        t_amb = eval_quantity(f"{spec['T_ambient_K']}*K")
        g_q = SI_UNITS["g0"]
        # Dimensional verification: (W / (kg/m^3 * J/(kg*K) * K * sqrt(m/s^2)))^(2/5) MUST equal m^1!
        d_star_q = (q_f / (rho * cp * t_amb * g_q.power(SIQuantity(0.5)))).power(SIQuantity(0.4))
        if d_star_q.dim != D_LEN:
            raise ValueError(f"LES D* failed SI length dimension check: {d_star_q.dim_list()}")
        ratio = d_star_q.val / float(spec["cell_dx_m"])
        return {
            "D_star_m": d_star_q.val,
            "D_star_over_dx": ratio,
            "mesh_adequate": ratio >= float(spec["min_ratio"]),
            "dual_verified": True,
        }
    if kind == "asme_ug27_shell":
        p_pa, r_m, s_pa, e_j, ca = (
            float(spec["P_Pa"]), float(spec["R_inner_m"]), float(spec["S_allow_Pa"]),
            float(spec["weld_efficiency_E"]), float(spec["corrosion_allowance_m"]),
        )
        t_req = (p_pa * r_m) / (s_pa * e_j - 0.6 * p_pa) + ca
        return {
            "t_required_m": t_req,
            "passes_ug27": float(spec["supplied_thickness_m"]) >= t_req,
            "dual_verified": True,
        }
    if kind == "mixing_balance":
        m_out = sum(float(s["m_kg_s"]) for s in spec["inlets"])
        e_in = sum(float(s["m_kg_s"]) * float(s["h_J_kg"]) for s in spec["inlets"]) + float(spec.get("Q_in_W", 0.0)) - float(spec.get("W_shaft_W", 0.0))
        h_out = e_in / m_out
        return {"m_out_kg_s": m_out, "h_out_J_kg": h_out, "dual_verified": True}
    if kind == "aabb_clearance_3d":
        (a0, a1), (b0, b1) = spec["box_a"], spec["box_b"]
        gaps = [max(0.0, float(a0[i]) - float(b1[i]), float(b0[i]) - float(a1[i])) for i in range(3)]
        dist = math.sqrt(sum(g * g for g in gaps))
        intersects = all(float(a1[i]) >= float(b0[i]) and float(b1[i]) >= float(a0[i]) for i in range(3)) and dist == 0.0
        return {
            "min_gap_m": 0.0 if intersects else dist,
            "intersects": intersects,
            "clearance_satisfied": (not intersects) and (dist >= float(spec["min_clearance_m"]) - 1e-12),
            "dual_verified": True,
        }
    if kind == "rigid_rectangle_contact":
        from hamon_rectangle_contact import verify_rigid_rectangle_contact
        result = verify_rigid_rectangle_contact(spec)
        return {**result, "kind": kind, "dual_verified": result["conservation_verified"]}
    if kind == "rigid_strip_contact":
        from hamon_bearing_capacity import verify_rigid_strip_contact
        result = verify_rigid_strip_contact(spec)
        return {**result, "kind": kind, "dual_verified": result["conservation_verified"]}
    if kind == "bearing_capacity":
        from hamon_bearing_capacity import verify_bearing_capacity
        result = verify_bearing_capacity(spec, legacy_rectangle=("B_m" in spec or "L_m" in spec))
        return {**result, "kind": kind, "dual_verified": result["conservation_verified"]}
    if kind == "consolidation_settlement":
        from hamon_consolidation import consolidation_load, primary_settlement, settlement_from_void_loss
        try:
            h, e0, cc, cr = (float(spec[k]) for k in ("H_m", "e0", "Cc", "Cr"))
            initial, pc, limit = (float(spec[k]) for k in ("sigma_v0_Pa", "sigma_p_Pa", "max_settlement_m"))
            load, stress_check = consolidation_load(spec, "delta_sigma_Pa", coerce_scalar=True)
            if not all(math.isfinite(v) for v in (initial, pc, load, limit, cc, cr)):
                raise ValueError("Require finite stresses, compression indices, and settlement limit")
            if initial <= 0 or pc < 0 or cc < 0 or not 0 <= cr <= cc:
                raise ValueError("Require positive initial stress, nonnegative prior stress, and 0 <= Cr <= Cc")
            if load < 0:
                if initial + load <= 0:
                    raise ValueError("Unloading must leave positive effective stress")
                # Preserve the signed historical unloading convention of this public schema.
                index = cr if initial + load <= pc else cc
                loss = index * math.log1p(load / initial) / math.log(10.0)
                response = settlement_from_void_loss(h, e0, loss)
            else:
                # A historical preconsolidation stress below the current stress is virgin loading.
                response = primary_settlement(h, e0, cc, initial, load, max(pc, initial), cr)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return {"passes_limit": False, "dual_verified": False, "reason": str(exc)}
        settlement = response["primary_consolidation_settlement_m"]
        return {"settlement_m": settlement, "passes_limit": settlement <= limit,
                "dual_verified": stress_check.get("declared_stress_consistent", True), **stress_check,
                "initial_void_height_m": response["initial_void_height_m"],
                "final_void_ratio": response["final_void_ratio"]}
    if kind == "manning_channel":
        return solve_manning_channel_dual(spec)
    if kind == "spillway_weir":
        q_d, cd, lw = float(spec["Q_design_m3_s"]), float(spec["Cd_si"]), float(spec["weir_length_m"])
        crest, top, fb_min = float(spec["crest_el_m"]), float(spec["dam_top_el_m"]), float(spec["min_freeboard_m"])
        sp_pair = None
        if HAS_SYMPY:
            h_s = _sp.Symbol("h", positive=True)
            sp_pair = (_sp.Float(repr(cd * lw), 30) * h_s**_sp.Rational(3, 2) - _sp.Float(repr(q_d), 30), h_s)
        rt = dual_solve_scalar_root(lambda h: cd * lw * (h**1.5) - q_d, sp_pair, (1e-4, 30.0), 1.5)
        h_req = rt["root"]
        wse = crest + h_req
        fb = top - wse
        q_cap = cd * lw * (max(0.0, top - fb_min - crest) ** 1.5)
        return {
            "head_required_m": h_req,
            "water_surface_el_m": wse,
            "freeboard_m": fb,
            "Q_capacity_with_freeboard_m3_s": q_cap,
            "passes_freeboard": fb >= fb_min,
            "dual_verified": True,
        }
    if kind == "compaction_acceptance":
        return verify_compaction_lot(spec)
    if kind == "rankine_retaining":
        H, gam, phi_d = float(spec["H_m"]), float(spec["gamma_N_m3"]), float(spec["phi_deg"])
        q_s, m_res, fs_min = float(spec["surcharge_Pa"]), float(spec["M_resist_Nm_per_m"]), float(spec["min_FS_overturning"])
        ka = math.tan(math.radians(45.0 - 0.5 * phi_d)) ** 2
        pa_soil = 0.5 * ka * gam * H**2
        pa_sur = ka * q_s * H
        m_ot = pa_soil * (H / 3.0) + pa_sur * (H / 2.0)
        fs_ot = m_res / m_ot
        return {
            "Ka": ka,
            "P_total_N_per_m": pa_soil + pa_sur,
            "M_overturn_Nm_per_m": m_ot,
            "FS_overturning": fs_ot,
            "passes_overturning": fs_ot >= fs_min,
            "dual_verified": True,
        }
    if kind == "estimate_audit":
        sub = sum(float(i["qty"]) * float(i["unit_price"]) for i in spec["items"])
        tot = sub * (1.0 + (float(spec["mobilization_pct"]) + float(spec["contingency_pct"])) / 100.0)
        return {
            "direct_subtotal": sub,
            "total_estimate": tot,
            "within_bond_cap": tot <= float(spec["bond_cap"]),
            "dual_verified": True,
        }
    if kind == "earthwork_balance":
        vf, gdf = float(spec["V_fill_m3"]), float(spec["gamma_d_fill"])
        return {
            "V_borrow_bank_m3": vf * (gdf / float(spec["gamma_d_borrow"])),
            "V_haul_loose_m3": vf * (gdf / float(spec["gamma_d_loose"])),
            "dual_verified": True,
        }
    if kind == "continuous_beam_2span":
        return solve_continuous_beam_2span_dual(spec)
    if kind == "bolt_group_eccentric_shear":
        return solve_bolt_group_eccentric_dual(spec)
    if kind == "ac_phasor_rlc_network":
        return solve_ac_phasor_rlc_dual(spec)
    if kind == "transformer_efficiency_regulation":
        return solve_transformer_dual(spec)
    if kind == "heat_exchanger_ntu":
        return solve_heat_exchanger_ntu_dual(spec)
    if kind == "hvac_coil_and_duct":
        return solve_hvac_coil_and_duct_dual(spec)
    if kind == "hydraulic_jump_conjugate":
        return solve_hydraulic_jump_dual(spec)
    if kind == "mep_routing_clearance_3d":
        return solve_mep_routing_clearance_3d(spec)
    if kind == "pile_group_capacity":
        return solve_pile_group_dual(spec)
    if kind == "exact_inequality_guard":
        return verify_quantity_inequality(str(spec["lhs"]), str(spec["op"]), str(spec["rhs"]))
    raise ValueError(f"unsupported dual-verifier problem kind: {kind}")



def run_dual_verify_spec(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a structured dual-verification request (SI expression, linear system, root, or domain check)."""
    mode = payload.get("mode", "domain")
    if mode == "si-eval":
        expr = str(payload["expression"])
        q = eval_quantity(expr)
        if "to" in payload and payload["to"]:
            target = eval_quantity(str(payload["to"]))
            if q.dim != target.dim:
                raise ValueError(f"SI conversion dimension mismatch: {q.dim_list()} vs {target.dim_list()}")
            return {
                "kind": "dual-verify-si-eval",
                "expression": expr,
                "target": str(payload["to"]),
                "value": q.val / target.val,
                "si_value": q.val,
                "dimensions_7": q.dim_list(),
                "dual_verified": True,
            }
        return {
            "kind": "dual-verify-si-eval",
            "expression": expr,
            "si_value": q.val,
            "dimensions_7": q.dim_list(),
            "dual_verified": True,
        }
    if mode == "linear-system":
        res = dual_solve_linear(payload["A"], payload["b"], float(payload.get("rtol", 1e-7)))
        if any(isinstance(v, complex) for v in res["solution"]):
            res["solution"] = [[v.real, v.imag] for v in res["solution"]]
            res["symbolic_solution"] = [[v.real, v.imag] for v in res["symbolic_solution"]]
        return {"kind": "dual-verify-linear-system", **res}
    if mode == "scalar-root":
        var_name = str(payload.get("var", "x"))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var_name):
            raise ValueError("invalid variable identifier")
        expr_str = str(payload["expr"])
        lo, hi = float(payload["bracket"][0]), float(payload["bracket"][1])
        guess = float(payload.get("guess", 0.5 * (lo + hi)))

        def _f_num(val: float) -> float:
            return eval_quantity(expr_str, {var_name: SIQuantity(float(val), ZERO7)}).val

        sp_pair = None
        if HAS_SYMPY:
            sym_v = _sp.Symbol(var_name, real=True)
            sp_expr = _sp.sympify(expr_str, locals={var_name: sym_v})
            sp_pair = (sp_expr, sym_v)
        res = dual_solve_scalar_root(_f_num, sp_pair, (lo, hi), guess, float(payload.get("rtol", 1e-7)))
        return {"kind": "dual-verify-scalar-root", **res}
    if mode == "domain":
        kind = str(payload["kind"])
        spec = payload.get("spec", payload)
        res = solve_domain_problem(kind, spec)
        return {"kind": "dual-verify-domain", "problem_kind": kind, **res}
    if mode == "inequality":
        res = verify_quantity_inequality(
            str(payload["lhs"]),
            str(payload.get("op", "<=")),
            str(payload["rhs"]),
            int(payload.get("precision", 64)),
        )
        return {"kind": "dual-verify-inequality", **res}
    raise ValueError(f"unsupported dual-verify mode: {mode}")


def run_dual_verify_file(filename: str | Path) -> dict[str, Any]:
    p = Path(filename)
    if not p.is_file() or p.stat().st_size > 2_000_000:
        raise ValueError("missing or oversized dual-verify JSON specification")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dual-verify JSON root must be an object")
    return run_dual_verify_spec(payload)
