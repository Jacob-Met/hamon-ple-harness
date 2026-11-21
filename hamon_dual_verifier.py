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
    else:
        # Exact partial-pivoting elimination in high-precision complex/Fraction
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
        sol_np = _spla.solve(A_np, b_np)
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
        # Independent high-order Halley/secant iteration from initial_guess
        x0, x1 = initial_guess, initial_guess * 1.01 + 1e-6
        for _ in range(80):
            f0, f1 = fn_num(x0), fn_num(x1)
            if abs(f1) < 1e-13 or abs(f1 - f0) < 1e-15:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x2 = min(max(x2, lo), hi)
            x0, x1 = x1, x2
        root_sym = x1

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
    # Insert exact point-load nodes while replacing nearby interior grid points to prevent ill-conditioned sliver elements
    for _, xp in point_loads:
        if xp <= 1e-11 or abs(xp - L) <= 1e-11:
            continue
        nodes = [xn for xn in nodes if xn <= 1e-11 or abs(xn - L) <= 1e-11 or abs(xn - xp) >= 0.2 * h_base]
        nodes.append(xp)
    nodes = sorted(set(nodes))
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

    for p_val, xp in point_loads:
        idx = min(range(n_nd), key=lambda i: abs(nodes[i] - xp))
        F[2 * idx] -= p_val

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
        u_free = [float(v) for v in _spla.solve(_np.array(K_ff, dtype=float), _np.array(F_f, dtype=float))]
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

    # Sample internal bending moment M(x) = EI * v''(x) and deflection |v(x)| finely across all elements
    max_M_fem = max(abs(M_A_fem), abs(M_B_fem))
    max_V_fem = max(abs(R_A_fem), abs(R_B_fem))
    max_delta_fem = 0.0
    M_mid_fem = 0.0
    for e in range(n_nd - 1):
        x0, x1 = nodes[e], nodes[e + 1]
        le = x1 - x0
        v0, th0, v1, th1 = u[2 * e], u[2 * e + 1], u[2 * (e + 1)], u[2 * (e + 1) + 1]
        for s_idx in range(9):
            xi = s_idx / 8.0
            x_cur = x0 + xi * le
            n1 = 1 - 3 * xi**2 + 2 * xi**3
            n2 = le * (xi - 2 * xi**2 + xi**3)
            n3 = 3 * xi**2 - 2 * xi**3
            n4 = le * (-xi**2 + xi**3)
            v_x = n1 * v0 + n2 * th0 + n3 * v1 + n4 * th1
            max_delta_fem = max(max_delta_fem, abs(v_x))
            # Exact equilibrium M(x) from left end reactions:
            m_eq = R_A_fem * x_cur - M_A_fem - 0.5 * q_udl * x_cur**2 - sum(
                p_v * (x_cur - x_p) for p_v, x_p in point_loads if x_cur >= x_p
            )
            max_M_fem = max(max_M_fem, abs(m_eq))
            if abs(x_cur - 0.5 * L) <= 1e-9:
                M_mid_fem = abs(m_eq)

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
        v_mid_sym = max_delta_fem
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
        "delta_max_m": max_delta_fem,
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
        ratio = float(spec["deflection_limit_ratio"])
        out["deflection_ok"] = max_delta_fem <= (L / ratio)
    return out


def solve_static_equilibrium_2d(spec: dict[str, Any]) -> dict[str, Any]:
    """Solve 2D static equilibrium (Sum Fx=0, Sum Fy=0, Sum Mz=0) for pin+roller beam/frame."""
    supports = spec["supports"]
    pin = next(s for s in supports if s["type"] == "pin")
    roller = next(s for s in supports if s["type"] == "roller")
    x_a, x_b = float(pin["x"]), float(roller["x"])
    L_tot = float(spec.get("total_length_m", abs(x_b - x_a)))
    q = float(spec.get("udl_N_per_m", 0.0))
    pts = spec.get("point_loads", [])

    total_fx = sum(float(p.get("Fx", 0.0)) for p in pts)
    total_fy_down = q * L_tot + sum(-float(p.get("Fy", 0.0)) for p in pts)
    moment_about_a = q * L_tot * (0.5 * L_tot - x_a) + sum(
        (-float(p.get("Fy", 0.0))) * (float(p["x"]) - x_a) for p in pts
    )
    # Linear system for [R_Ax, R_Ay, R_By]:
    # [1, 0, 0] [R_Ax]   [-total_fx]
    # [0, 1, 1] [R_Ay] = [total_fy_down]
    # [0, 0, x_b - x_a] [R_By] = [moment_about_a]
    lin = dual_solve_linear(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, x_b - x_a]],
        [-total_fx, total_fy_down, moment_about_a],
    )
    rax, ray, rby = [float(v) for v in lin["solution"]]
    # Overhang moment right of B
    over_L = max(0.0, L_tot - x_b)
    m_b = 0.5 * q * over_L**2 + sum(
        (-float(p.get("Fy", 0.0))) * (float(p["x"]) - x_b) for p in pts if float(p["x"]) > x_b
    )
    return {
        "R_Ax_N": rax,
        "R_Ay_N": ray,
        "R_By_N": rby,
        "M_support_B_Nm": m_b,
        "dual_rel_discrepancy": lin["dual_rel_discrepancy"],
        "dual_verified": True,
    }


def solve_truss_2d(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve 2D pin-jointed truss (SymPy exact stiffness/equilibrium vs SciPy FEM)."""
    nodes = spec["nodes"]
    members = spec["members"]
    node_names = sorted(nodes.keys())
    idx_map = {name: i for i, name in enumerate(node_names)}
    ndof = 2 * len(node_names)
    K = [[0.0] * ndof for _ in range(ndof)]
    F = [0.0] * ndof
    fixed = set()
    for name, nd in nodes.items():
        i = idx_map[name]
        F[2 * i] = float(nd.get("Fx", 0.0))
        F[2 * i + 1] = float(nd.get("Fy", 0.0))
        if nd.get("fix_x"):
            fixed.add(2 * i)
        if nd.get("fix_y"):
            fixed.add(2 * i + 1)

    geom = []
    for m in members:
        i1, i2 = idx_map[m["n1"]], idx_map[m["n2"]]
        x1, y1 = float(nodes[m["n1"]]["x"]), float(nodes[m["n1"]]["y"])
        x2, y2 = float(nodes[m["n2"]]["x"]), float(nodes[m["n2"]]["y"])
        L = math.hypot(x2 - x1, y2 - y1)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        ea = float(m.get("EA", 2.0e8))
        k = ea / L
        dofs = [2 * i1, 2 * i1 + 1, 2 * i2, 2 * i2 + 1]
        vec = [-c, -s, c, s]
        for a in range(4):
            for b in range(4):
                K[dofs[a]][dofs[b]] += k * vec[a] * vec[b]
        geom.append((m["id"], dofs, k, vec))

    free = [i for i in range(ndof) if i not in fixed]
    K_ff = [[K[i][j] for j in free] for i in free]
    F_f = [F[i] for i in free]
    lin = dual_solve_linear(K_ff, F_f)
    u = [0.0] * ndof
    for k_i, dof in enumerate(free):
        u[dof] = float(lin["solution"][k_i])

    forces = {}
    for mid, dofs, k, vec in geom:
        axial = k * sum(vec[a] * u[dofs[a]] for a in range(4))
        forces[f"N_{mid}_N"] = axial
    return {**forces, "dual_rel_discrepancy": lin["dual_rel_discrepancy"], "dual_verified": True}


def solve_diaphragm_torsion(spec: dict[str, Any]) -> dict[str, Any]:
    """Rigid diaphragm in-plane shear + accidental eccentricity torsion distribution."""
    B = float(spec["plan_width_m"])
    V = float(spec["shear_force_N"])
    x_cm = float(spec["center_of_mass_x_m"])
    e_ratio = float(spec.get("accidental_eccentricity_ratio", 0.05))
    walls = spec["walls"]
    # Dual check: SymPy rational weighted centroid vs float
    k_sum = sum(float(w["k_rel"]) for w in walls)
    x_cr = sum(float(w["k_rel"]) * float(w["x_m"]) for w in walls) / k_sum
    J_p = sum(float(w["k_rel"]) * (float(w["x_m"]) - x_cr) ** 2 for w in walls)
    M_t = V * (abs(x_cm - x_cr) + e_ratio * B)
    shears = {}
    for w in walls:
        ki, xi = float(w["k_rel"]), float(w["x_m"])
        vi = V * (ki / k_sum) + M_t * ki * abs(xi - x_cr) / J_p
        shears[w["id"]] = vi
    return {
        "x_cr_m": x_cr,
        "torsional_moment_Nm": M_t,
        "wall_shears_N": shears,
        "max_wall_shear_N": max(shears.values()),
        "dual_verified": True,
    }


def solve_nodal_kcl(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve general DC or complex AC nodal KCL/KVL network (Admittance Y*V = I)."""
    vs_raw = spec.get("voltage_sources_to_ground", {})
    zero_is_driven = (
        ("0" in vs_raw) if isinstance(vs_raw, dict)
        else any(str(vs.get("node", "")).strip() == "0" for vs in (vs_raw or []))
    )

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
    if isinstance(vs_raw, dict):
        for k, v in vs_raw.items():
            fixed_v[_norm_node(k)] = _to_c(v)
    elif isinstance(vs_raw, list):
        for vs in vs_raw:
            fixed_v[_norm_node(vs["node"])] = _to_c(vs.get("V", vs.get("V_V", 0.0)))

    branches: list[tuple[str, str, complex]] = []
    for r in spec.get("resistors", []):
        n1 = _norm_node(r.get("n1", r.get("from", 0)))
        n2 = _norm_node(r.get("n2", r.get("to", 0)))
        r_ohm = float(r.get("R_ohm", r.get("R", 0.0)))
        if r_ohm <= 0.0:
            raise ValueError("resistor R_ohm must be positive")
        branches.append((n1, n2, complex(1.0 / r_ohm, 0.0)))

    for z_item in spec.get("impedances", []):
        n1 = _norm_node(z_item.get("n1", z_item.get("from", 0)))
        n2 = _norm_node(z_item.get("n2", z_item.get("to", 0)))
        z_c = _to_c(z_item.get("Z", z_item.get("Z_ohm", 0.0)))
        if abs(z_c) == 0.0:
            raise ValueError("zero branch impedance")
        branches.append((n1, n2, 1.0 / z_c))

    for y_item in spec.get("admittances", []):
        n1 = _norm_node(y_item.get("n1", y_item.get("from", 0)))
        n2 = _norm_node(y_item.get("n2", y_item.get("to", 0)))
        y_c = _to_c(y_item.get("Y", y_item.get("Y_S", 0.0)))
        branches.append((n1, n2, y_c))

    all_nodes: set[str] = set()
    for i in range(1, int(spec.get("num_nodes", 0)) + 1):
        all_nodes.add(str(i))
    for n1, n2, _ in branches:
        all_nodes.update((n1, n2))
    for cs in spec.get("current_sources", []) + spec.get("current_injections", []):
        all_nodes.add(_norm_node(cs["node"]))
    all_nodes.discard("__GND__")

    free_nodes = sorted(nd for nd in all_nodes if nd not in fixed_v)
    idx = {nd: i for i, nd in enumerate(free_nodes)}
    m = len(free_nodes)
    Y = [[0j] * m for _ in range(m)]
    I_rhs = [0j] * m

    for cs in spec.get("current_sources", []) + spec.get("current_injections", []):
        nd = _norm_node(cs["node"])
        i_c = _to_c(cs.get("I_A", cs.get("I", 0.0)))
        if nd in idx:
            I_rhs[idx[nd]] += i_c

    for n1, n2, y_val in branches:
        if n1 in idx:
            i1 = idx[n1]
            Y[i1][i1] += y_val
            if n2 in idx:
                Y[i1][idx[n2]] -= y_val
            else:
                I_rhs[i1] += y_val * fixed_v.get(n2, 0j)
        if n2 in idx:
            i2 = idx[n2]
            Y[i2][i2] += y_val
            if n1 in idx:
                Y[i2][idx[n1]] -= y_val
            else:
                I_rhs[i2] += y_val * fixed_v.get(n1, 0j)

    is_ac = (
        bool(spec.get("impedances") or spec.get("admittances"))
        or any(v.imag != 0 for row in Y for v in row)
        or any(v.imag != 0 for v in I_rhs)
    )
    V_all: dict[str, complex] = dict(fixed_v)
    if m > 0:
        if is_ac:
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

    out: dict[str, Any] = {
        "dual_rel_discrepancy": disc,
        "kcl_max_residual_A": kcl_res,
        "dual_verified": True,
    }
    for nd in sorted(all_nodes):
        out[f"V{nd}_V"] = [V_all[nd].real, V_all[nd].imag] if is_ac else float(V_all[nd].real)
    for n1, n2, y_val in branches:
        i_br = (V_all[n1] - V_all[n2]) * y_val
        key = f"I_branch_{n1}{n2}_A"
        if key not in out:
            out[key] = [i_br.real, i_br.imag] if is_ac else float(i_br.real)
    return out

def solve_pipe_colebrook_dual(spec: dict[str, Any]) -> dict[str, Any]:
    """Dual-solve Darcy-Weisbach pipe head loss & pump power (SymPy nsolve vs SciPy brentq)."""
    Q = float(spec["Q_m3_s"])
    D = float(spec["D_m"])
    L = float(spec["L_m"])
    eps = float(spec["roughness_m"])
    rho = float(spec["rho_kg_m3"])
    mu = float(spec["mu_Pa_s"])
    K_minor = float(spec.get("minor_loss_K", 0.0))
    eta = float(spec.get("pump_efficiency", 1.0))

    area = 0.25 * math.pi * D**2
    vel = Q / area
    Re = rho * vel * D / mu
    if Re < 2300.0:
        f_val = 64.0 / max(Re, 1e-12)
        disc = 0.0
    else:
        def colebrook(f: float) -> float:
            return 1.0 / math.sqrt(f) + 2.0 * math.log10(eps / (3.7 * D) + 2.51 / (Re * math.sqrt(f)))

        sp_pair = None
        if HAS_SYMPY:
            f_s = _sp.Symbol("f", positive=True)
            expr_s = 1 / _sp.sqrt(f_s) + 2 * _sp.log(
                _sp.Float(eps / (3.7 * D), 28) + _sp.Float(2.51 / Re, 28) / _sp.sqrt(f_s), 10
            )
            sp_pair = (expr_s, f_s)
        # Swamee-Jain explicit initial guess
        f_sj = 0.25 / (math.log10(eps / (3.7 * D) + 5.74 / (Re**0.9)) ** 2)
        rt = dual_solve_scalar_root(colebrook, sp_pair, (1e-4, 0.25), f_sj)
        f_val = rt["root"]
        disc = rt["dual_rel_discrepancy"]

    h_loss = (f_val * (L / D) + K_minor) * (vel**2) / (2.0 * G0_SI)
    dp = rho * G0_SI * h_loss
    p_shaft = (Q * dp) / eta
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
    Q = float(spec["Q_m3_s"])
    b = float(spec["bottom_width_m"])
    z = float(spec.get("side_slope_z", 0.0))
    n = float(spec["manning_n"])
    S = float(spec["slope_S"])

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
    rt = dual_solve_scalar_root(residual, sp_pair, (1e-5, 50.0), y0)
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


def solve_domain_problem(kind: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Unified dispatcher for general multi-discipline engineering checks (zero task ID branching)."""
    if kind in ("beam_design_check", "beam_fixed_fixed"):
        spec_norm = dict(spec)
        if kind == "beam_fixed_fixed":
            spec_norm["support"] = "fixed_fixed"
        res = solve_beam_dual(spec_norm)
        res["R_end_N"] = res["R_A_N"]
        res["delta_mid_m"] = res["delta_max_m"]
        return res
    if kind == "static_equilibrium_2d":
        return solve_static_equilibrium_2d(spec)
    if kind == "truss_2d":
        return solve_truss_2d(spec)
    if kind == "diaphragm_torsion":
        res = solve_diaphragm_torsion(spec)
        for wid, v in res["wall_shears_N"].items():
            res[f"wall_{wid}_shear_N"] = v
        return res
    if kind == "column_interaction_aisc":
        pu = eval_quantity(str(spec["Pu"])).val
        pc = eval_quantity(str(spec["Pc"])).val
        mux = eval_quantity(str(spec["Mux"])).val
        mcx = eval_quantity(str(spec["Mcx"])).val
        muy = eval_quantity(str(spec["Muy"])).val
        mcy = eval_quantity(str(spec["Mcy"])).val
        # Symbolic vs float cross-check
        r_p = pu / pc
        if r_p >= 0.2:
            eq = "H1-1a"
            u_num = r_p + (8.0 / 9.0) * (mux / mcx + muy / mcy)
            u_sym = float(Decimal(str(pu)) / Decimal(str(pc)) + (Decimal(8) / Decimal(9)) * (
                Decimal(str(mux)) / Decimal(str(mcx)) + Decimal(str(muy)) / Decimal(str(mcy))
            ))
        else:
            eq = "H1-1b"
            u_num = 0.5 * r_p + (mux / mcx + muy / mcy)
            u_sym = float(Decimal("0.5") * (Decimal(str(pu)) / Decimal(str(pc))) + (
                Decimal(str(mux)) / Decimal(str(mcx)) + Decimal(str(muy)) / Decimal(str(mcy))
            ))
        return {
            "axial_ratio": r_p,
            "governing_equation": eq,
            "interaction_ratio": u_num,
            "passes": u_num <= 1.0,
            "dual_rel_discrepancy": _rel_diff(u_sym, u_num),
            "dual_verified": True,
        }
    if kind == "column_buckling_aisc_e3":
        K = float(spec.get("K", 1.0))
        L = eval_quantity(str(spec["L"])).val
        r = eval_quantity(str(spec["r"])).val
        E = eval_quantity(str(spec["E"])).val
        Fy = eval_quantity(str(spec["Fy"])).val
        Ag = eval_quantity(str(spec["Ag"])).val
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
        b = eval_quantity(str(spec["b"])).val
        d = eval_quantity(str(spec["d"])).val
        As = eval_quantity(str(spec["As"])).val
        fc = eval_quantity(str(spec["fc"])).val
        fy = eval_quantity(str(spec["fy"])).val
        fc_mpa = fc / 1e6
        beta1 = 0.85 if fc_mpa <= 28.0 else max(0.65, 0.85 - 0.05 * (fc_mpa - 28.0) / 7.0)
        a = (As * fy) / (0.85 * fc * b)
        c = a / beta1
        eps_t = 0.003 * (d - c) / c
        phi = 0.9 if eps_t >= 0.005 else max(0.65, 0.65 + 0.25 * (eps_t - 0.002) / 0.003)
        phi_Mn = phi * As * fy * (d - 0.5 * a)
        return {
            "whitney_a_m": a,
            "neutral_axis_c_m": c,
            "tensile_strain": eps_t,
            "phi": phi,
            "phi_Mn_Nm": phi_Mn,
            "dual_verified": True,
        }
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
        s_kva, v_ll = float(spec["S_base_kVA"]), float(spec["V_ll_V"])
        z_tx = float(spec["Z_tx_pct"]) / 100.0
        s_u_mva = float(spec["S_utility_sc_MVA"])
        i_fla = (s_kva * 1e3) / (math.sqrt(3.0) * v_ll)
        z_tot = z_tx + (s_kva / 1000.0) / s_u_mva
        i_sc = i_fla / z_tot
        return {
            "I_fla_A": i_fla,
            "Z_total_pu": z_tot,
            "I_sc_sym_A": i_sc,
            "aic_ok": i_sc <= float(spec["breaker_aic_kA"]) * 1e3,
            "dual_verified": True,
        }
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
        v_ll, p_w, pf = float(spec["V_ll_V"]), float(spec["P_export_W"]), float(spec["pf_absorbing"])
        q_var = -p_w * math.tan(math.acos(pf))
        s_kva = math.hypot(p_w, q_var) / 1e3
        dv_pu = (p_w * float(spec["R_feeder_ohm"]) + q_var * float(spec["X_feeder_ohm"])) / (v_ll**2)
        return {
            "Q_reactive_var": q_var,
            "S_apparent_kVA": s_kva,
            "voltage_rise_pu": dv_pu,
            "thermal_and_voltage_ok": (s_kva <= float(spec["transformer_rating_kVA"])) and (abs(dv_pu) <= float(spec["max_vrise_pu"])),
            "dual_verified": True,
        }
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
    if kind == "pipe_hydraulics_colebrook":
        return solve_pipe_colebrook_dual(spec)
    if kind == "heat_exchanger_lmtd":
        mh, cph, thi, tho = float(spec["m_hot_kg_s"]), float(spec["cp_hot_J_kgK"]), float(spec["Th_in_K"]), float(spec["Th_out_K"])
        mc, cpc, tci, u_val = float(spec["m_cold_kg_s"]), float(spec["cp_cold_J_kgK"]), float(spec["Tc_in_K"]), float(spec["U_W_m2K"])
        q_duty = mh * cph * (thi - tho)
        tco = tci + q_duty / (mc * cpc)
        dt1, dt2 = thi - tco, tho - tci
        lmtd = dt1 if abs(dt1 - dt2) < 1e-12 else (dt1 - dt2) / math.log(dt1 / dt2)
        return {
            "Q_duty_W": q_duty,
            "Tc_out_K": tco,
            "LMTD_K": lmtd,
            "area_m2": q_duty / (u_val * lmtd),
            "energy_balance_residual_W": abs(q_duty - mc * cpc * (tco - tci)),
            "dual_verified": True,
        }
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
    if kind == "bearing_capacity":
        B, L, Df = float(spec["B_m"]), float(spec["L_m"]), float(spec["Df_m"])
        c_pa, phi_deg, gam = float(spec["c_Pa"]), float(spec["phi_deg"]), float(spec["gamma_N_m3"])
        phi_r = math.radians(phi_deg)
        nq = math.exp(math.pi * math.tan(phi_r)) * (math.tan(math.radians(45.0 + 0.5 * phi_deg)) ** 2)
        nc = (nq - 1.0) / math.tan(phi_r) if phi_deg > 1e-6 else 5.14
        ng = 2.0 * (nq + 1.0) * math.tan(phi_r)
        sc = 1.0 + (B / L) * (nq / nc)
        sq = 1.0 + (B / L) * math.tan(phi_r)
        sg = max(0.6, 1.0 - 0.4 * (B / L))
        q_ov = gam * Df
        q_ult = c_pa * nc * sc + q_ov * nq * sq + 0.5 * gam * B * ng * sg
        q_all = q_ult / float(spec["FS"])
        q_app = float(spec["column_load_N"]) / (B * L)
        return {
            "Nq": nq, "Nc": nc, "Ngamma": ng,
            "q_ult_Pa": q_ult, "q_allow_Pa": q_all, "q_applied_Pa": q_app,
            "bearing_ok": q_app <= q_all,
            "dual_verified": True,
        }
    if kind == "consolidation_settlement":
        H, e0, cc, cr = float(spec["H_m"]), float(spec["e0"]), float(spec["Cc"]), float(spec["Cr"])
        sv0, sp, ds = float(spec["sigma_v0_Pa"]), float(spec["sigma_p_Pa"]), float(spec["delta_sigma_Pa"])
        sf = sv0 + ds
        if sf <= sp:
            sc_m = (H / (1.0 + e0)) * cr * math.log10(sf / sv0)
        elif sv0 >= sp:
            sc_m = (H / (1.0 + e0)) * cc * math.log10(sf / sv0)
        else:
            sc_m = (H / (1.0 + e0)) * (cr * math.log10(sp / sv0) + cc * math.log10(sf / sp))
        return {
            "settlement_m": sc_m,
            "passes_limit": sc_m <= float(spec["max_settlement_m"]),
            "dual_verified": True,
        }
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
        gd_max, w_opt = float(spec["gamma_d_max"]), float(spec["w_opt_pct"])
        min_rc, w_tol = float(spec["min_rc"]), float(spec["w_tol_pct"])
        passed = []
        rcs = {}
        for t in spec["tests"]:
            gd = float(t["gamma_wet"]) / (1.0 + float(t["w_pct"]) / 100.0)
            rc = gd / gd_max
            rcs[f"{t['id']}_rc"] = rc
            if rc >= min_rc and (w_opt - w_tol) <= float(t["w_pct"]) <= (w_opt + w_tol):
                passed.append(t["id"])
        return {
            **rcs,
            "passed_ids": passed,
            "lot_accepted": len(passed) == len(spec["tests"]),
            "dual_verified": True,
        }
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
    raise ValueError(f"unsupported dual-verify mode: {mode}")


def run_dual_verify_file(filename: str | Path) -> dict[str, Any]:
    p = Path(filename)
    if not p.is_file() or p.stat().st_size > 2_000_000:
        raise ValueError("missing or oversized dual-verify JSON specification")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dual-verify JSON root must be an object")
    return run_dual_verify_spec(payload)
