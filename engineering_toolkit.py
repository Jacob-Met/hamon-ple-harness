"""Deterministic, local-only engineering checks for PLE task containers.

A helper, not a design authority. IFC bounding boxes are BROAD PHASE ONLY,
PDF coordinates are page-space (not scale verified), and formula cache presence
is not calculation correctness. None of these commands modify task files.
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
from fractions import Fraction
import io
import json
import math
import operator
import pathlib
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

from office_package import (
    _DOCUMENT_RELS, _SPREADSHEET_NS, _xlsx_relationships, _xlsx_target, _xlsx_xml, workbook_parts as _xlsx_parts,
    check_office_package,
)

getcontext().prec = 28
ZERO = (0, 0, 0, 0, 0)  # kg, m, s, A, K
MASS, LENGTH, TIME, CURRENT, TEMP = (
    (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0),
    (0, 0, 0, 1, 0), (0, 0, 0, 0, 1))
FORCE = (1, 1, -2, 0, 0)
PRESSURE = (1, -1, -2, 0, 0)
ENERGY = (1, 2, -2, 0, 0)
POWER = (1, 2, -3, 0, 0)
VOLTAGE = (1, 2, -3, -1, 0)
RESISTANCE = (1, 2, -3, -2, 0)

def D(value):
    return Decimal(str(value))

def scaled(a, b):
    return tuple(x + y for x, y in zip(a, b))

def reduced(a, b):
    return tuple(x - y for x, y in zip(a, b))

@dataclass(frozen=True)
class Q:
    val: Decimal
    dim: tuple[int, int, int, int, int] = ZERO

    def __add__(self, other):
        if self.dim != other.dim:
            raise ValueError(f"dimension mismatch in addition: {self.dim} vs {other.dim}")
        return Q(self.val + other.val, self.dim)

    def __sub__(self, other):
        if self.dim != other.dim:
            raise ValueError(f"dimension mismatch in subtraction: {self.dim} vs {other.dim}")
        return Q(self.val - other.val, self.dim)

    def __mul__(self, other):
        return Q(self.val * other.val, scaled(self.dim, other.dim))

    def __truediv__(self, other):
        if other.val == 0:
            raise ValueError("division by zero")
        return Q(self.val / other.val, reduced(self.dim, other.dim))

    def __neg__(self):
        return Q(-self.val, self.dim)

    def power(self, exponent):
        if exponent.dim != ZERO or exponent.val != int(exponent.val):
            raise ValueError("integer dimensionless exponent required")
        n = int(exponent.val)
        if n < -12 or n > 12:
            raise ValueError("exponent outside [-12,12]")
        if self.val == 0 and n < 0:
            raise ValueError("zero to negative power")
        return Q(self.val ** n, tuple(x*n for x in self.dim))

    def sqrt(self):
        if self.val < 0 or any(x % 2 for x in self.dim):
            raise ValueError("sqrt requires nonnegative value and even dimensions")
        return Q(self.val.sqrt(), tuple(x//2 for x in self.dim))


def unit(value, dimension):
    return Q(D(value), dimension)

# Traceable exact constants: international inch = 0.0254 m, standard g = 9.80665 m/s^2,
# international avoirdupois lb = 0.45359237 kg. Ambiguous "lb" is deliberately absent.
UNITS = {
    "one": unit(1,ZERO),
    "m": unit(1,LENGTH), "mm": unit("0.001",LENGTH),
    "cm": unit("0.01",LENGTH), "km": unit(1000,LENGTH),
    "inch": unit("0.0254",LENGTH), "in_": unit("0.0254",LENGTH),
    "ft": unit("0.3048",LENGTH), "yd": unit("0.9144",LENGTH),
    "s": unit(1,TIME), "min_": unit(60,TIME), "hr": unit(3600,TIME),
    "kg": unit(1,MASS), "g": unit("0.001",MASS),
    "lbm": unit("0.45359237",MASS),
    "K": unit(1,TEMP), "A": unit(1,CURRENT),
    "N": unit(1,FORCE), "kN": unit(1000,FORCE),
    "MN": unit(1000000,FORCE),
    "lbf": unit("4.4482216152605",FORCE),
    "kip": unit("4448.2216152605",FORCE),
    "Pa": unit(1,PRESSURE), "kPa": unit(1000,PRESSURE),
    "MPa": unit(1000000,PRESSURE), "GPa": unit(1000000000,PRESSURE),

    "J": unit(1,ENERGY), "kJ": unit(1000,ENERGY),
    "W": unit(1,POWER), "kW": unit(1000,POWER),
    "V": unit(1,VOLTAGE), "ohm": unit(1,RESISTANCE),
}

# Derive every imperial pressure scale from the same exactly defined pound-
# force and international length units. Independently rounded constants would
# make an otherwise exact 1000 psi == 1 ksi check fail.
UNITS.update({
    "psi": UNITS["lbf"]/(UNITS["inch"]*UNITS["inch"]),
    "ksi": Q(D(1000)) * UNITS["lbf"]/(UNITS["inch"]*UNITS["inch"]),
    "psf": UNITS["lbf"]/(UNITS["ft"]*UNITS["ft"]),
    "ksf": Q(D(1000)) * UNITS["lbf"]/(UNITS["ft"]*UNITS["ft"]),
})

def quantity(expression: str, symbols: dict[str,Q] | None = None) -> Q:
    """Evaluate an arithmetic unit expression, never Python eval or arbitrary calls."""
    if len(expression) > 1000:
        raise ValueError("quantity expression exceeds 1000 characters")
    tree = ast.parse(expression, mode="eval")

    def walk(node, depth=0):
        if depth > 35:
            raise ValueError("expression nesting limit")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node) if type(node.value) is float else node.value
            return Q(D(literal))
        if isinstance(node, ast.Name):
            if symbols is not None and node.id in symbols:
                return symbols[node.id]
            if node.id not in UNITS:
                raise ValueError(f"unknown or ambiguous unit '{node.id}'")
            return UNITS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            q = walk(node.operand,depth+1)
            return -q if isinstance(node.op, ast.USub) else q
        if isinstance(node, ast.BinOp):
            a,b=walk(node.left,depth+1),walk(node.right,depth+1)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div): return a/b
            if isinstance(node.op, ast.Pow): return a.power(b)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "sqrt" and len(node.args) == 1
                and not node.keywords):
            return walk(node.args[0],depth+1).sqrt()
        raise ValueError(f"disallowed expression: {type(node).__name__}")
    q = walk(tree.body)
    if not q.val.is_finite():
        raise ValueError("non-finite result")
    return q

def conversion(expr: str, target: str):
    actual,target_quantity=quantity(expr),quantity(target)
    if actual.dim != target_quantity.dim:
        raise ValueError(f"unit conversion mismatch {actual.dim} vs {target_quantity.dim}")
    if target_quantity.val <= 0:
        raise ValueError("target unit must have positive scale")
    return {"expression":expr,"target":target,
            "value":str(actual.val/target_quantity.val),
            "si_value":str(actual.val),"dimensions":list(actual.dim)}


def equal_dimensions(lhs: str, rhs: str, rtol: str):
    a,b=quantity(lhs),quantity(rhs)
    if a.dim != b.dim:
        raise ValueError(f"assertion compares unlike dimensions: {a.dim} vs {b.dim}")
    tolerance=D(rtol)
    if tolerance < 0 or tolerance >= 1:
        raise ValueError("rtol must be in [0,1)")
    error=abs(a.val-b.val)
    limit=tolerance*max(abs(a.val),abs(b.val),D("1e-30"))
    return {"lhs_si":str(a.val),"rhs_si":str(b.val),
            "same_dimension":True,"passed":error <= limit,
            "absolute_difference_si":str(error),"relative_tolerance":str(tolerance),
            "dimensions":list(a.dim)}



_CELL_REF = r"\$?[A-Z]{1,3}\$?[1-9][0-9]{0,6}"
_EXACT_LOOKUP = re.compile(
    rf"(?P<kind>[VH])LOOKUP\(\s*(?P<key>{_CELL_REF})\s*,\s*"
    rf"(?:(?P<sheet>'(?:[^']|'')+'|[A-Z_][A-Z0-9_.]*)!)?"
    rf"(?P<start>{_CELL_REF}):(?P<end>{_CELL_REF})\s*,\s*"
    r"(?P<index>[1-9][0-9]{0,4})\s*,\s*(?:FALSE|0)\s*\)", re.I | re.ASCII)


# Excel CONVERT unit tokens are case-sensitive. Only multiplicative units with
# a matching toolkit definition qualify; Excel and toolkit aliases differ.
_CONVERT_UNITS = {
    key: UNITS[name] for key, name in {
        "m": "m", "mm": "mm", "cm": "cm", "km": "km", "in": "inch", "ft": "ft",
        "yd": "yd", "g": "g", "kg": "kg", "lbm": "lbm", "sec": "s", "hr": "hr",
        "N": "N", "kN": "kN", "MN": "MN",
        "Pa": "Pa", "kPa": "kPa", "MPa": "MPa", "GPa": "GPa",
    }.items()
}
# Excel permits both m2 and m^2. Prefixes apply to the length before squaring.
for _length_token in ("m", "mm", "cm", "km", "in", "ft", "yd"):
    _length_unit = _CONVERT_UNITS[_length_token]
    for _suffix in ("2", "^2"):
        _CONVERT_UNITS[_length_token + _suffix] = _length_unit * _length_unit


def _grid_position(address):
    match = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]{0,6})", address.replace("$", "").upper())
    if match is None:
        raise ValueError("invalid worksheet cell reference")
    col, row = match.groups()
    x = 0
    for char in col:
        x = 26 * x + ord(char) - 64
    if x > 16384 or int(row) > 1048576:
        raise ValueError("cell reference outside worksheet")
    return x, int(row)


def _address(column, row):
    letters = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters + str(row)


def _shared_lookup_masters(cells, ns):
    """Find sheet-scoped master definitions before a follower is inspected."""
    masters = {}
    for cell in cells:
        formula = cell.find(ns + "f")
        if formula is None or formula.get("t") != "shared":
            continue
        if not (formula.text or "").strip() and "ref" not in formula.attrib:
            continue
        group = formula.get("si", "")
        if not re.fullmatch(r"[0-9]{1,10}", group) or int(group) > 4294967295:
            continue
        group = int(group)
        if group in masters:
            masters[group] = None
        elif len(masters) >= 1024:
            return {}
        else:
            masters[group] = (cell.get("r", ""), formula.get("ref", ""), (formula.text or "").strip())
    return masters


def _shared_lookup_at(master, address):
    """Translate only captured A1 references, leaving quoted sheet names intact."""
    origin, area, expression = master
    match = _EXACT_LOOKUP.fullmatch(expression)
    bounds = re.fullmatch(rf"({_CELL_REF})(?::({_CELL_REF}))?", area, re.I | re.ASCII)
    if match is None or bounds is None:
        return ""
    try:
        x, y = _grid_position(origin)
        to_x, to_y = _grid_position(address)
        x1, y1 = _grid_position(bounds[1])
        x2, y2 = _grid_position(bounds[2] or bounds[1])
        if not (x1 <= x <= x2 and y1 <= y <= y2 and x1 <= to_x <= x2 and y1 <= to_y <= y2):
            return ""
        for field in ("end", "start", "key"):
            ref = match[field]
            col, row = _grid_position(ref)
            if not ref.startswith("$"):
                col += to_x - x
            if "$" not in ref[1:]:
                row += to_y - y
            if not (1 <= col <= 16384 and 1 <= row <= 1048576):
                return ""
            lo, hi = match.span(field)
            expression = expression[:lo] + _address(col, row) + expression[hi:]
    except ValueError:
        return ""
    return expression


def _numeric_cell(cell, value):
    if cell.get("t", "n") != "n" or value is None or not value.text:
        return None
    try:
        number = float(value.text)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None



def _typed_numeric_cache(cell, value, ns):
    typ = cell.get("t", "n")
    if value is not None and (typ == "str" or (typ in ("s", "b", "e") and value.text is not None)):
        return typ
    if typ == "inlineStr" and cell.find(ns + "is") is not None:
        return typ
    return None


_SHEET_TOKEN = r"'(?:[^'\[\]:*!?/\\]|'')+'|[A-Z_][A-Z0-9_.]*"


def _local_reference(prefix, vector=False):
    sheet = rf"(?:(?P<{prefix}_sheet>{_SHEET_TOKEN})!)?"
    start = rf"(?P<{prefix}_start>{_CELL_REF})"
    end = rf":(?P<{prefix}_end>{_CELL_REF})" if vector else ""
    return sheet + start + end


_INDEX_MATCH = re.compile(
    rf"INDEX\(\s*{_local_reference('values', True)}\s*,\s*"
    rf"MATCH\(\s*{_local_reference('needle')}\s*,\s*{_local_reference('keys', True)}"
    r"\s*,\s*0\s*\)\s*\)", re.I | re.ASCII)


def _verify_index_match(sheet, formula, cached, literals):
    match = _INDEX_MATCH.fullmatch(formula)
    if match is None:
        return {"status": "unverified", "reason": "requires_vertical_INDEX_and_exact_MATCH_0"}
    if cached is None:
        return {"status": "unverified", "reason": "missing_numeric_cache"}
    resolved = {}
    for field in ("needle", "keys", "values"):
        name = match[field + "_sheet"] or sheet
        if name.startswith("'"):
            name = name[1:-1].replace("''", "'")
        candidates = [s for s in literals if s is not None and s.casefold() == name.casefold()]
        if len(candidates) != 1 or (candidates[0] != name and not (name.isascii() and candidates[0].isascii())):
            return {"status": "unverified", "reason": "missing_or_ambiguous_sheet"}
        resolved[field] = candidates[0]
    try:
        key_x, key_y = _grid_position(match["keys_start"])
        key_end_x, key_end_y = _grid_position(match["keys_end"])
        val_x, val_y = _grid_position(match["values_start"])
        val_end_x, val_end_y = _grid_position(match["values_end"])
        needle_cell = _address(*_grid_position(match["needle_start"]))
    except ValueError:
        return {"status": "unverified", "reason": "invalid_cell_reference"}
    length = key_end_y - key_y + 1
    if (key_x != key_end_x or val_x != val_end_x or length != val_end_y - val_y + 1
            or not 1 <= length <= 2048):
        return {"status": "unverified", "reason": "requires_equal_bounded_vertical_vectors"}
    needle = literals[resolved["needle"]].get(needle_cell)
    if needle is None:
        return {"status": "unverified", "reason": "requires_literal_numeric_lookup_key"}
    for offset in range(length):
        key = literals[resolved["keys"]].get(_address(key_x, key_y + offset))
        if key is None:
            return {"status": "unverified", "reason": "unresolved_preceding_lookup_key"}
        if key == needle:
            address = _address(val_x, val_y + offset)
            expected = literals[resolved["values"]].get(address)
            if expected is None:
                return {"status": "unverified", "reason": "requires_literal_numeric_return_cell"}
            return {"status": "matched" if cached == expected else "mismatch",
                    "source_sheet": resolved["values"], "source_cell": address,
                    "expected": expected, "cached": cached}
    return {"status": "unverified", "reason": "no_exact_numeric_match"}


def _verify_lookup(sheet, formula, cached, literals):
    if formula.upper().startswith("INDEX("):
        return _verify_index_match(sheet, formula, cached, literals)
    match = _EXACT_LOOKUP.fullmatch(formula)
    if match is None or cached is None:
        return {"status": "unverified"}
    table = match["sheet"] or sheet
    if table.startswith("'"):
        table = table[1:-1].replace("''", "'")
    # Require the exact declared sheet name. No external, named, or implicit ranges.
    if table not in literals or sheet not in literals:
        return {"status": "unverified"}
    needle = literals[sheet].get(match["key"].replace("$", "").upper())
    try:
        _grid_position(match["key"])
        x1, y1 = _grid_position(match["start"])
        x2, y2 = _grid_position(match["end"])
    except ValueError:
        return {"status": "unverified"}
    index = int(match["index"])
    vertical = match["kind"].upper() == "V"
    length, width = (y2-y1+1, x2-x1+1) if vertical else (x2-x1+1, y2-y1+1)
    if needle is None or not 1 <= length <= 2048 or not 1 <= index <= width:
        return {"status": "unverified"}
    cells = literals[table]
    for step in range(length):
        x, y = (x1, y1+step) if vertical else (x1+step, y1)
        key = cells.get(_address(x, y))
        if key is None:
            # A blank, nonnumeric or derived key could affect first-match semantics.
            break
        if key == needle:
            address = _address(x+index-1, y) if vertical else _address(x, y+index-1)
            expected = cells.get(address)
            if expected is not None:
                return {"status": "matched" if cached == expected else "mismatch",
                        "source_sheet": table, "source_cell": address,
                        "expected": expected, "cached": cached}
            break
    return {"status": "unverified"}



_ARITHMETIC_TOKEN = re.compile(
    rf"(?P<convert>CONVERT)(?=\s*\()|{_local_reference('ref')}|"
    r"(?P<number>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:E[+-]?[0-9]+)?)"
    r'|(?P<unit>"[A-Z]+(?:\^?2)?")|(?P<operator>[(),+*/^-])|(?P<space>\s+)',
    re.I | re.ASCII)

_COEFFICIENT = object()

# LibreOffice 24.2 retains this historical pound conversion constant. Keep it
# separate from the exact SI pound in _CONVERT_UNITS. A cache matching only
# this engine convention is unverified, rather than a proven stale value.
# https://github.com/LibreOffice/core/blob/libreoffice-24.2.7.2/scaddins/source/analysis/analysishelper.cxx
_CALC_LBM_PER_KG = D("2.20462291469134")


@dataclass(frozen=True)
class _FormulaUnit:
    scale: Fraction
    dimension: tuple


@dataclass(frozen=True)
class _FormulaNumber:
    value: float
    unit: Any = None
    roots: tuple = ()
    conversion_depth: int = 0
    arithmetic_depth: int = 0
    evidence: Any = None
    constraint: Any = None

    @property
    def required_unit(self):
        return self.unit if self.constraint is None else self.constraint


class _FormulaConflict(ValueError):
    def __init__(self, key, status, reason, **evidence):
        super().__init__(reason)
        self.key, self.status, self.evidence = key, status, evidence


def _bounded_frame(unit):
    if (max(unit.scale.numerator.bit_length(), unit.scale.denominator.bit_length()) > 4096
            or max(map(abs, unit.dimension)) > 1024):
        raise ValueError("unsupported_unit_range")
    return unit


def _formula_unit(key, operation, left, right, *, conditional=False):
    if conditional and isinstance(operation, (ast.Add, ast.Sub)):
        # A valid sum must share the known operand's frame even when another
        # term has no declared unit. This is a necessary condition, not a unit
        # annotation for that unlabelled term.
        if isinstance(left, _FormulaUnit) and (right is None or right is _COEFFICIENT):
            return left
        if isinstance(right, _FormulaUnit) and (left is None or left is _COEFFICIENT):
            return right
    if left is None or right is None:
        return None
    if left is _COEFFICIENT and right is _COEFFICIENT:
        return _COEFFICIENT
    if isinstance(operation, (ast.Add, ast.Sub)):
        if left is _COEFFICIENT or right is _COEFFICIENT:
            return None
        if left.dimension != right.dimension:
            raise _FormulaConflict(key, "dimension_mismatch", "incompatible_additive_dimensions",
                                   left_dimension=list(left.dimension), right_dimension=list(right.dimension))
        if left.scale != right.scale:
            raise _FormulaConflict(key, "unit_scale_mismatch", "incompatible_additive_unit_scales",
                                   left_scale_si=str(left.scale), right_scale_si=str(right.scale),
                                   si_dimension=list(left.dimension))
        return left
    left = _FormulaUnit(Fraction(1), ZERO) if left is _COEFFICIENT else left
    right = _FormulaUnit(Fraction(1), ZERO) if right is _COEFFICIENT else right
    if isinstance(operation, ast.Mult):
        unit = _FormulaUnit(left.scale*right.scale, scaled(left.dimension, right.dimension))
    else:
        unit = _FormulaUnit(left.scale/right.scale, reduced(left.dimension, right.dimension))
    return _bounded_frame(unit)


def _excel_expression(tokens):
    """Parse scalar spreadsheet operators with Excel's precedence and associativity."""
    # Excel evaluates repeated powers left to right, and negation before power.
    binary = {"+": (1, ast.Add), "-": (1, ast.Sub), "*": (2, ast.Mult),
              "/": (2, ast.Div), "^": (3, ast.Pow)}
    cursor = 0

    def pop():
        nonlocal cursor
        if cursor == len(tokens):
            raise ValueError("incomplete_scalar_formula")
        cursor += 1
        return tokens[cursor - 1]

    def expect(token):
        if pop() != token:
            raise ValueError("invalid_scalar_formula")

    def expression(minimum):
        token = pop()
        if token in ("+", "-"):
            op = ast.UAdd() if token == "+" else ast.USub()
            result = ast.UnaryOp(op=op, operand=expression(4))
        elif token == "(":
            result = expression(1)
            expect(")")
        elif token == "_convert":
            expect("(")
            args = [expression(1)]
            for _ in range(2):
                expect(",")
                unit = pop()
                if not unit.startswith('"'):
                    raise ValueError("unsupported_conversion_units")
                args.append(ast.Constant(value=unit[1:-1]))
            expect(")")
            result = ast.Call(func=ast.Name(id=token), args=args, keywords=[])
        elif token.startswith("v"):
            result = ast.Name(id=token)
        else:
            raise ValueError("invalid_scalar_formula")
        while cursor < len(tokens) and tokens[cursor] in binary:
            priority, operation = binary[tokens[cursor]]
            if priority < minimum:
                break
            pop()
            result = ast.BinOp(left=result, op=operation(), right=expression(priority + 1))
        return result

    result = expression(1)
    if cursor != len(tokens):
        raise ValueError("invalid_scalar_formula")
    return result


def _power_frame(frame, power):
    if frame is None and power == 0:
        # Every nonzero measured quantity loses its unit in the zeroth power.
        return _FormulaUnit(Fraction(1), ZERO)
    if not isinstance(frame, _FormulaUnit):
        return frame
    unit = _FormulaUnit(frame.scale ** power, tuple(e * power for e in frame.dimension))
    return _bounded_frame(unit)


def _parse_scalar_formula(key, expression, literals):
    tokens, symbols, references = [], {}, {}
    pos = 0
    while pos < len(expression):
        token = _ARITHMETIC_TOKEN.match(expression, pos)
        if token is None or len(tokens) >= 96:
            raise ValueError("unsupported_or_oversize_formula")
        pos = token.end()
        if token["space"]:
            continue
        if token["convert"]:
            tokens.append("_convert")
            continue
        if token["operator"]:
            tokens.append(token["operator"])
            continue
        if token["unit"]:
            tokens.append(token["unit"])
            continue
        operand = f"v{len(symbols)+len(references)}"
        try:
            if token["number"]:
                number = float(token["number"])
                if not math.isfinite(number) or 0 < abs(number) < sys.float_info.min:
                    raise ValueError("unsupported_operand_range")
                symbols[operand] = _FormulaNumber(number, _COEFFICIENT)
            else:
                name = token["ref_sheet"] or key[0]
                if name.startswith("'"):
                    name = name[1:-1].replace("''", "'")
                sheets = [s for s in literals if s == name or (
                    s.isascii() and name.isascii() and s.lower() == name.lower())]
                if len(sheets) != 1:
                    raise ValueError("unknown_or_ambiguous_worksheet")
                references[operand] = (sheets[0], _address(*_grid_position(token["ref_start"])))
        except (ValueError, ArithmeticError) as exc:
            # Keep valid expression syntax available to the structural
            # pass; numeric/reference availability is an evaluation error.
            symbols[operand] = exc
        tokens.append(operand)
    tree = _excel_expression(tokens)
    # An explicit dimensional contradiction does not depend on a cached
    # number, a present input, or successful evaluation of a dependency.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_convert" and len(node.args) == 3 and not node.keywords
                and all(isinstance(arg, ast.Constant) and type(arg.value) is str
                        for arg in node.args[1:])):
            from_name, to_name = (arg.value for arg in node.args[1:])
            source, target = _CONVERT_UNITS.get(from_name), _CONVERT_UNITS.get(to_name)
            if source is not None and target is not None and source.dim != target.dim:
                raise _FormulaConflict(key, "dimension_mismatch", "incompatible_conversion_dimensions",
                                       from_unit=from_name, to_unit=to_name, dimension_mismatch=True)
    return tree, symbols, references



def _scalar_formula_audit(arithmetic, conversions, literals, legacy_lbm=False):
    """Evaluate one dependency graph for ordinary arithmetic and CONVERT calls.

    Topological evaluation preserves the 256-cell conversion-chain contract
    without growing the Python call stack. All numeric dependencies come from
    literals or reconstructed formulas; cached results are comparison data only.
    """
    producers = {key: (*row, "arithmetic") for key, row in arithmetic.items()}
    for row in conversions:
        sheet, address, expression, cached, *typed = row
        producers[(sheet, address)] = (expression, cached, typed[0] if typed else None, "conversion")
    parsed, values, reverse, waiting = {}, {}, {}, {}
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    for key, row in producers.items():
        try:
            parsed[key] = _parse_scalar_formula(key, row[0], literals)
        except (ValueError, SyntaxError, ArithmeticError) as exc:
            values[key] = exc
    for key in parsed:
        dependencies = {ref for ref in parsed[key][2].values() if ref in producers and ref not in values}
        waiting[key] = len(dependencies)
        for dependency in dependencies:
            reverse.setdefault(dependency, []).append(key)

    def reference(ref):
        if ref in values:
            result = values[ref]
            if isinstance(result, Exception):
                raise result
            return result
        if ref in producers:
            raise ValueError("cyclic_reference")
        value = literals[ref[0]].get(ref[1])
        if value is None:
            raise ValueError("unsupported_or_missing_dependency")
        if not math.isfinite(value) or 0 < abs(value) < sys.float_info.min:
            raise ValueError("unsupported_operand_range")
        return _FormulaNumber(value, roots=(ref,))

    def evaluate(key):
        tree, symbols, references = parsed[key]

        def scalar(node):
            if isinstance(node, ast.Name):
                result = reference(references[node.id]) if node.id in references else symbols[node.id]
                if isinstance(result, Exception):
                    raise result
                return result
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                result = scalar(node.operand)
                return _FormulaNumber(-result.value if isinstance(node.op, ast.USub) else result.value,
                                      result.unit, result.roots, result.conversion_depth, result.arithmetic_depth,
                                      constraint=result.constraint)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_convert" and len(node.args) == 3 and not node.keywords):
                if any(not isinstance(arg, ast.Constant) or type(arg.value) is not str for arg in node.args[1:]):
                    raise ValueError("unsupported_conversion_units")
                from_name, to_name = (arg.value for arg in node.args[1:])
                source, target = _CONVERT_UNITS.get(from_name), _CONVERT_UNITS.get(to_name)
                if source is None or target is None:
                    raise ValueError("unsupported_unit")
                actual = scalar(node.args[0])
                evidence = {"source_value": actual.value, "from_unit": from_name, "to_unit": to_name,
                            "conversion_depth": actual.conversion_depth+1}
                if isinstance(node.args[0], ast.Name) and node.args[0].id in references:
                    ref = references[node.args[0].id]
                    evidence.update(source_sheet=ref[0], source_cell=ref[1])
                if len(actual.roots) == 1:
                    evidence.update(root_sheet=actual.roots[0][0], root_cell=actual.roots[0][1])
                elif actual.roots:
                    evidence["root_cells"] = [list(ref) for ref in actual.roots]
                source_unit = _FormulaUnit(Fraction(source.val), source.dim)
                frame = actual.required_unit
                if isinstance(frame, _FormulaUnit) and frame != source_unit:
                    status = ("dimension_mismatch" if frame.dimension != source_unit.dimension
                              else "unit_scale_mismatch")
                    raise _FormulaConflict(key, status, "upstream_unit_disagrees", **evidence,
                                           required_dimension=list(frame.dimension),
                                           declared_dimension=list(source_unit.dimension),
                                           required_scale_si=str(frame.scale),
                                           declared_scale_si=str(source_unit.scale))
                numeric_source = D(1)/_CALC_LBM_PER_KG if legacy_lbm and from_name == "lbm" else source.val
                numeric_target = D(1)/_CALC_LBM_PER_KG if legacy_lbm and to_name == "lbm" else target.val
                expected = float(D(actual.value)*numeric_source/numeric_target)
                if (not math.isfinite(expected) or (expected == 0 and actual.value != 0)
                        or 0 < abs(expected) < sys.float_info.min):
                    raise ValueError("numeric_range")
                return _FormulaNumber(expected, _FormulaUnit(Fraction(target.val), target.dim),
                                      actual.roots, actual.conversion_depth+1, actual.arithmetic_depth, evidence)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                base, exponent = scalar(node.left), scalar(node.right)
                if exponent.unit is not _COEFFICIENT or not exponent.value.is_integer() or abs(exponent.value) > 12:
                    raise ValueError("power_requires_bounded_integer_coefficient")
                power = int(exponent.value)
                if base.value == 0 and power <= 0:
                    raise ValueError("unsupported_zero_power")
                number = base.value ** power
                if (not math.isfinite(number) or (number == 0 and base.value != 0)
                        or 0 < abs(number) < sys.float_info.min):
                    raise ValueError("unsupported_result_range")
                return _FormulaNumber(number, _power_frame(base.unit, power),
                                      tuple(sorted(set(base.roots + exponent.roots))),
                                      max(base.conversion_depth, exponent.conversion_depth),
                                      max(base.arithmetic_depth, exponent.arithmetic_depth),
                                      constraint=_power_frame(base.required_unit, power))
            if not isinstance(node, ast.BinOp) or type(node.op) not in operations:
                raise ValueError("unsupported_arithmetic_operator")
            left, right = scalar(node.left), scalar(node.right)
            unit = _formula_unit(key, node.op, left.unit, right.unit)
            constraint = _formula_unit(key, node.op, left.required_unit, right.required_unit, conditional=True)
            number = operations[type(node.op)](left.value, right.value)
            if isinstance(node.op, (ast.Add, ast.Sub)) and 0 < max(abs(left.value), abs(right.value)):
                if abs(number) <= 1e-14*max(abs(left.value), abs(right.value)):
                    raise ValueError("cancellation_sensitive_result")
            if not math.isfinite(number) or 0 < abs(number) < sys.float_info.min:
                raise ValueError("unsupported_result_range")
            return _FormulaNumber(number, unit, tuple(sorted(set(left.roots+right.roots))),
                                  max(left.conversion_depth,right.conversion_depth),
                                  max(left.arithmetic_depth,right.arithmetic_depth), constraint=constraint)

        result = scalar(tree)
        # A plain CONVERT chain may contain all 256 indexed conversion cells.
        # Arithmetic dependency depth keeps its independent, smaller bound.
        has_arithmetic = (producers[key][3] == "arithmetic"
                          or any(isinstance(node,(ast.BinOp,ast.UnaryOp)) for node in ast.walk(tree)))
        result = _FormulaNumber(result.value, result.unit, result.roots, result.conversion_depth,
                                result.arithmetic_depth+int(has_arithmetic), result.evidence, result.constraint)
        if result.arithmetic_depth > 40:
            raise ValueError("cyclic_or_deep_dependency")
        return result

    ready = [key for key, count in waiting.items() if count == 0]
    while ready:
        key = ready.pop()
        try:
            values[key] = evaluate(key)
        except (ValueError, SyntaxError, ArithmeticError, RecursionError, KeyError) as exc:
            values[key] = exc
        for child in reverse.get(key, ()):
            waiting[child] -= 1
            if waiting[child] == 0:
                ready.append(child)
    for key in producers:
        if key not in values:
            values[key] = ValueError("cyclic_reference")

    reports = {"arithmetic": [], "conversion": []}
    for key, (expression, cached, incompatible_type, kind) in producers.items():
        record = {"sheet":key[0], "cell":key[1], "cached":cached, "status":"unverified"}
        result = values[key]
        if isinstance(result, _FormulaConflict):
            if result.key == key:
                record.update(reason=str(result), **result.evidence)
                record["status"] = result.status
            else:
                record.update(reason="upstream_"+result.status,
                              source_sheet=result.key[0], source_cell=result.key[1])
        elif isinstance(result, Exception):
            reason = str(result) if isinstance(result,ValueError) else type(result).__name__
            if kind == "arithmetic" and reason == "cyclic_reference":
                reason = "cyclic_or_deep_dependency"
            record["reason"] = reason
        else:
            record.update(result.evidence or {})
            record["expected"] = result.value
            if isinstance(result.unit,_FormulaUnit):
                record.update(si_dimension=list(result.unit.dimension), unit_scale_si=str(result.unit.scale))
            elif isinstance(result.constraint, _FormulaUnit):
                record.update(conditional_si_dimension=list(result.constraint.dimension),
                              conditional_unit_scale_si=str(result.constraint.scale))
            if incompatible_type:
                record.update(status="mismatch", cached_type=incompatible_type, reason="nonnumeric_cache")
            elif cached is None:
                record["reason"] = "missing_numeric_cache"
            else:
                record["status"] = "matched" if math.isclose(cached,result.value,rel_tol=1e-12) else "mismatch"
        reports[kind].append(record)
    if not legacy_lbm and any('"lbm"' in row[0] for row in producers.values()):
        # Reconstruct the whole graph under one alternate numeric convention;
        # never use a saved intermediate to explain a downstream result.
        alternate = _scalar_formula_audit(arithmetic, conversions, literals, legacy_lbm=True)
        for kind, records in reports.items():
            for record, other in zip(records, alternate[kind]):
                if record["status"] == "mismatch" and other["status"] == "matched":
                    record.update(status="unverified", reason="engine_specific_lbm_scale",
                                  expected_si=record.pop("expected"),
                                  expected_legacy_lbm=other["expected"])
    return reports


def _equipment_catalog(filename, edition, sheet_name, title_sheet):
    """Resolve both the selected equipment sheet and its issue through the workbook index."""
    from collections import defaultdict
    def rich_string(node):
        # Phonetic guides and document annotations are not part of a cell's display text.
        return "".join(t.text or "" for t in node.findall("{*}t") + node.findall("{*}r/{*}t"))
    with zipfile.ZipFile(filename) as archive:
        workbook, parts = _xlsx_parts(archive)
        if not workbook or sheet_name == title_sheet:
            raise ValueError("expected distinct named title and catalog worksheets")
        def part(name):
            found = [s for s in parts if s["name"] == name]
            if len(found) != 1 or found[0]["kind"] != "worksheet" or found[0]["state"] != "visible":
                raise ValueError(f"expected one visible worksheet named {name!r}")
            return found[0]["part"]
        root = pathlib.PurePosixPath(workbook)
        rels = _xlsx_relationships(archive, str(root.parent / "_rels" / (root.name + ".rels")))
        shared = []
        tables = [r for r in rels.values() if r.get("Type") in {base + "/sharedStrings" for base in _DOCUMENT_RELS}]
        if tables:
            if len(tables) != 1 or tables[0].get("TargetMode", "Internal") != "Internal":
                raise ValueError("ambiguous shared string table")
            table = _xlsx_xml(archive, _xlsx_target(workbook, tables[0].get("Target", "")), 25000000)
            shared = [rich_string(node) for node in table.findall("{*}si")]
        def read(name):
            rows = defaultdict(dict)
            worksheet = _xlsx_xml(archive, part(name), 25000000)
            if any(r.get("hidden") in ("1", "true") for r in worksheet.findall("{*}sheetData/{*}row")):
                raise ValueError(f"hidden rows in source worksheet {name!r}")
            for c in worksheet.findall("{*}sheetData/{*}row/{*}c"):
                match = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]{0,6})", c.get("r",""))
                if not match:
                    raise ValueError("invalid worksheet cell reference")
                col, number = match.groups()
                if col in rows[int(number)]:
                    raise ValueError("duplicate worksheet cell reference")
                typ = c.get("t", "n")
                value = c.findtext("{*}v", "")
                if c.find("{*}f") is not None or typ == "e":
                    value = None
                elif typ == "s":
                    if not value.isascii() or not value.isdigit() or int(value) >= len(shared):
                        raise ValueError("unresolved shared string")
                    value = shared[int(value)]
                elif typ == "inlineStr":
                    node = c.find("{*}is")
                    value = rich_string(node) if node is not None else ""
                elif typ not in ("str",):
                    value = None if value else ""
                rows[int(number)][col] = value
            return rows
        def norm(s):
            return " ".join((s or "").split()).casefold()
        title = read(title_sheet)
        issues = [(n,row.get("B")) for n,row in title.items() if norm(row.get("A")) in {"doc no.","doc no","document number"}]
        if len(issues)!=1 or issues[0][1]!=edition:
            raise ValueError("requested edition does not match the unique title-block document number")
        rows = read(sheet_name)
        headers = [(n,[col for col,v in row.items() if norm(v)=="manufacturer"],
                       [col for col,v in row.items() if norm(v)=="model"]) for n,row in rows.items()]
        headers = [h for h in headers if h[1] and h[2]]
        if len(headers)!=1 or len(headers[0][1])!=1 or len(headers[0][2])!=1:
            raise ValueError("expected one manufacturer/model header row")
        start, (maker,), (model,) = headers[0]
        catalog = defaultdict(list)
        for n, row in rows.items():
            if n<=start or all(v=="" for v in row.values()): continue
            if not norm(row.get(maker)) or not norm(row.get(model)):
                raise ValueError(f"missing literal manufacturer or model at {sheet_name} row {n}")
            key = " ".join((row[maker]+" "+row[model]).split())
            catalog[key].append({"manufacturer":row[maker],"model":row[model],
                                 "cells":[f"{sheet_name}!{maker}{n}",f"{sheet_name}!{model}{n}"]})
        if not catalog:
            raise ValueError("empty equipment catalog")
        return catalog, f"{title_sheet}!B{issues[0][0]}"


def witness_equipment(pdf, filename, edition, sheet, title_sheet, equipment_type):
    import hashlib
    import hamon_witness_package
    documents = [pathlib.Path(pdf), pathlib.Path(filename)]
    payloads = []
    for path in documents:
        with path.open("rb") as source:
            data = source.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError("equipment source exceeds 32 MiB")
        payloads.append(data)
    catalog, title_cell = _equipment_catalog(io.BytesIO(payloads[1]), edition, sheet, title_sheet)
    converted = subprocess.run(["pdftotext","-layout","-","-"],input=payloads[0],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
    if converted.returncode:
        raise ValueError("unable to read PDF text")
    result = hamon_witness_package.match_installation(converted.stdout.decode("utf-8"), catalog, equipment_type)
    result.update(edition=edition, title_cell=title_cell, catalog_sheet=sheet,
                  sources=[{"file":str(p),"sha256":hashlib.sha256(b).hexdigest()} for p,b in zip(documents,payloads)])
    return result


_ARITHMETIC_GRAPH_LIMIT = 256
_ARITHMETIC_INDEX_LIMIT = 4096


def _audit_scalar_components(arithmetic, conversions, literals):
    """Evaluate independent producer graphs without treating cached formulas as literals."""
    if len(arithmetic) <= _ARITHMETIC_GRAPH_LIMIT:
        return _scalar_formula_audit(arithmetic, conversions, literals)

    conversion_by_cell = {(row[0], row[1]): row for row in conversions}
    parents = {key: key for key in (*arithmetic, *conversion_by_cell)}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    # Only formula-to-formula edges join components. Many independent cost
    # rows can read the same literal markup table without joining each other.
    # Discard each parsed tree here; full numeric trees live one component at a time.
    for key in parents:
        expression = arithmetic[key][0] if key in arithmetic else conversion_by_cell[key][2]
        try:
            _, _, references = _parse_scalar_formula(key, expression, literals)
        except (ValueError, SyntaxError, ArithmeticError):
            continue
        group = root(key)
        for dependency in references.values():
            if dependency in parents:
                parents[root(dependency)] = group

    groups = {}
    for key in parents:
        groups.setdefault(root(key), []).append(key)

    reports = {"arithmetic": [], "conversion": []}
    for keys in groups.values():
        local_arithmetic = {}
        local_conversions = []
        for key in keys:
            if key in conversion_by_cell:
                local_conversions.append(conversion_by_cell[key])
            elif len(local_arithmetic) < _ARITHMETIC_GRAPH_LIMIT:
                local_arithmetic[key] = arithmetic[key]
            else:
                reports["arithmetic"].append({
                    "sheet": key[0], "cell": key[1], "cached": arithmetic[key][1],
                    "status": "unverified", "reason": "arithmetic_component_limit",
                })
        local = _scalar_formula_audit(local_arithmetic, local_conversions, literals)
        for kind in reports:
            reports[kind].extend(local[kind])
    return reports


def xlsx_audit(filename: str):
    """Audit declared worksheet cells and formula caches without evaluating a workbook."""
    manifest = {"file": filename, "kind": "xlsx-formula-audit", "sheets": [], "other_sheets": []}
    literals = {}
    lookups = []
    conversions = []
    arithmetic = {}
    arithmetic_total = arithmetic_index_omitted = 0
    conversion_total = lookup_total = indexed = 0
    complete_index = True
    with zipfile.ZipFile(filename) as archive:
        workbook, parts = _xlsx_parts(archive)
        full_precision = True
        if workbook:
            settings = _xlsx_xml(archive, workbook, 2_000_000).find("{*}calcPr")
            full_precision = settings is None or settings.get("fullPrecision", "1") in ("1", "true")
        manifest["workbook_part"] = workbook
        manifest["discovery"] = "workbook_relationships" if workbook else "standalone_parts"
        if len(parts) > 300:
            raise ValueError("over 300 sheets; run focused audit")
        for sheet in parts:
            if sheet["kind"] != "worksheet":
                manifest["other_sheets"].append(sheet)
                continue
            root = _xlsx_xml(archive, sheet["part"], 25_000_000)
            ns = root.tag.removesuffix("worksheet")
            if ns not in {"{" + v + "}" for v in _SPREADSHEET_NS}:
                raise ValueError(f"Invalid worksheet namespace or root: {sheet['part']}")
            cells = formulas = missing = empty_strings = followers = error_count = 0
            errors = []
            name = sheet["name"]
            if not name or name in literals:
                complete_index = False
            sheet_values = literals.setdefault(name, {})
            derived_ranges = []
            cell_path = f"{ns}sheetData/{ns}row/{ns}c"
            unresolved_shared_followers = 0
            masters = _shared_lookup_masters(root.iterfind(cell_path), ns)
            first_conversion = len(conversions)
            sheet_arithmetic = {}
            for cell in root.iterfind(cell_path):
                cells += 1
                formula = cell.find(ns + "f")
                value = cell.find(ns + "v")
                typ = cell.get("t", "n")
                address = cell.get("r", "")
                if complete_index:
                    indexed += 1
                    if indexed > 100_000 or address in sheet_values:
                        complete_index = False
                    else:
                        sheet_values[address] = _numeric_cell(cell, value) if formula is None else None
                if formula is not None:
                    if formula.get("t") in ("array", "dataTable") and derived_ranges is not None:
                        area = re.fullmatch(rf"({_CELL_REF})(?::({_CELL_REF}))?",
                                            formula.get("ref", ""), re.I | re.ASCII)
                        try:
                            if area is None or len(derived_ranges) >= 128:
                                raise ValueError("unbounded formula dependency range")
                            x1, y1 = _grid_position(area[1])
                            x2, y2 = _grid_position(area[2] or area[1])
                            x, y = _grid_position(address)
                            if not (x1 <= x <= x2 and y1 <= y <= y2):
                                raise ValueError("formula master outside dependency range")
                            derived_ranges.append((x1, y1, x2, y2))
                        except ValueError:
                            derived_ranges = None
                    expression = (formula.text or "").strip()
                    formula_type = formula.get("t", "normal")
                    shared_master = None
                    if formula_type == "shared":
                        group = formula.get("si", "")
                        if re.fullmatch(r"[0-9]{1,10}", group):
                            shared_master = masters.get(int(group))
                        if not expression:
                            if shared_master is None:
                                unresolved_shared_followers += 1
                            else:
                                expression = shared_master[2]
                    if expression.upper().startswith(("VLOOKUP(", "HLOOKUP(", "INDEX(")):
                        lookup_total += 1
                        if len(lookups) < 256 and len(expression) <= 512 and formula_type in ("normal", "shared"):
                            if formula_type == "shared":
                                expression = _shared_lookup_at(shared_master, address) if shared_master else ""
                            lookups.append((name, address, expression, _numeric_cell(cell, value)))
                    if expression.upper().startswith("CONVERT("):
                        conversion_total += 1
                        if len(conversions) < 256 and len(expression) <= 512 and formula_type == "normal":
                            conversions.append((name, address, expression, _numeric_cell(cell, value),
                                                _typed_numeric_cache(cell, value, ns)))
                    if not expression.upper().startswith(("CONVERT(", "VLOOKUP(", "HLOOKUP(", "INDEX(")):
                        arithmetic_total += 1
                        eligible = (complete_index and full_precision and formula_type == "normal"
                                    and len(expression) <= 512)
                        if eligible and len(arithmetic) + len(sheet_arithmetic) >= _ARITHMETIC_INDEX_LIMIT:
                            arithmetic_index_omitted += 1
                        elif eligible:
                            # Typed string and Boolean results are physically distinct
                            # from a missing numeric cache. A date may use another encoding.
                            typed_cache = None
                            if value is not None and (typ == "str" or (
                                    typ in ("s", "b", "e") and value.text is not None)):
                                typed_cache = typ
                            if typ == "inlineStr" and cell.find(ns + "is") is not None:
                                typed_cache = typ
                            sheet_arithmetic[(name, address)] = (expression, _numeric_cell(cell, value), typed_cache)
                    formulas += 1
                    if value is None:
                        missing += 1
                    elif typ == "str":
                        empty_strings += not value.text
                    elif not (value.text or "").strip():
                        missing += 1
                    if (formula.get("t") == "shared" and not (formula.text or "").strip()
                            and formula.get("ref") is None):
                        followers += 1
                if typ == "e":
                    error_count += 1
                    if len(errors) < 50:
                        errors.append({"cell": cell.get("r", ""), "value": value.text if value is not None else ""})
            # Followers may contain only cached <v>; their master can appear later.
            # Rectangles never expand into coordinates, even for a whole worksheet.
            if derived_ranges is None:
                sheet_values.clear()
                del conversions[first_conversion:]
                sheet_arithmetic.clear()
            elif derived_ranges and complete_index:
                def is_derived(address):
                    x, y = _grid_position(address)
                    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in derived_ranges)

                try:
                    for address, numeric in sheet_values.items():
                        if numeric is not None and is_derived(address):
                            sheet_values[address] = None
                    sheet_arithmetic = {key: row for key, row in sheet_arithmetic.items()
                                        if not is_derived(key[1])}
                    conversions[first_conversion:] = [
                        row for row in conversions[first_conversion:] if not is_derived(row[1])
                    ]
                except ValueError:
                    sheet_values.clear()
                    del conversions[first_conversion:]
                    sheet_arithmetic.clear()
            arithmetic.update(sheet_arithmetic)
            manifest["sheets"].append({**sheet, "cells": cells, "formulas": formulas,
                                       "formulas_without_cached_values": missing,
                                       "formulas_with_empty_string_cache": empty_strings,
                                       "shared_formula_followers": followers,
                                       "unresolved_shared_formula_followers": unresolved_shared_followers,
                                       "error_cells": errors,
                                       "error_count": error_count})
    counts = {"matched": 0, "mismatch": 0, "unverified": lookup_total}
    examples = []
    if complete_index:
        for name, address, expression, cached in lookups:
            result = _verify_lookup(name, expression, cached, literals)
            if result["status"] != "unverified":
                counts["unverified"] -= 1
                counts[result["status"]] += 1
            examples.append({"sheet": name, "cell": address, **result})
    manifest["lookup_audit"] = {"counts": counts, "examples": examples}
    reconstructed = (_audit_scalar_components(arithmetic, conversions, literals)
                     if complete_index and full_precision else {"conversion": [], "arithmetic": []})
    for kind, total in (("conversion", conversion_total), ("arithmetic", arithmetic_total)):
        counts = {"matched": 0, "mismatch": 0, "unverified": total}
        if kind == "conversion":
            counts["dimension_mismatch"] = 0
        examples = reconstructed[kind]
        for record in examples:
            if record["status"] != "unverified":
                counts["unverified"] -= 1
                counts[record["status"]] = counts.get(record["status"], 0) + 1
        report = {"counts": counts, "examples": examples}
        if kind == "arithmetic":
            limited = sum(record.get("reason") == "arithmetic_component_limit" for record in examples)
            if limited or arithmetic_index_omitted:
                report["limits"] = {"component_omitted": limited, "index_omitted": arithmetic_index_omitted}
            if len(examples) > _ARITHMETIC_GRAPH_LIMIT:
                # Count every result, but keep the output bounded and show contradictions first.
                priority = {"dimension_mismatch": 0, "unit_scale_mismatch": 0, "mismatch": 0,
                            "unverified": 1, "matched": 2}
                report["examples"] = sorted(examples, key=lambda row: priority[row["status"]])[:_ARITHMETIC_GRAPH_LIMIT]
                report["examples_omitted"] = len(examples) - len(report["examples"])
        manifest[kind + "_audit"] = report
    manifest["note"] = ("Checked numeric formulas are reconstructed from literal inputs. Explicit conversion units "
                        "establish dimensions and exact multiplicative unit scales. Unverified caches can be stale.")
    return manifest


def pdf_words(filename: str, page: int, limit: int):
    """Returns word boxes in PDF page coordinates, without assuming document scale."""
    if page < 1 or page > 10000 or limit < 1 or limit > 2000:
        raise ValueError("page 1..10000 and limit 1..2000 required")
    exe=shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("pdftotext unavailable in task container; use document MCP")
    cp=subprocess.run([exe,"-f",str(page),"-l",str(page),
                       "-bbox-layout",filename,"-"],
                      capture_output=True,timeout=12,check=False)
    if cp.returncode != 0:
        raise ValueError("pdftotext failed; PDF may be encrypted or scanned")
    if len(cp.stdout)>8_000_000:
        raise ValueError("PDF vector layout exceeds 8 MB; crop or select another page")
    root=ET.fromstring(cp.stdout)
    words=[]
    for el in root.iter():
        if el.tag.endswith("word"):
            try:
                box=[float(el.attrib[k]) for k in ("xMin","yMin","xMax","yMax")]
            except (ValueError,KeyError):
                continue
            words.append({"text":(el.text or "")[:140],"box":box})
            if len(words)>=limit: break
    return {"kind":"pdf-vector-words","file":filename,"page":page,
            "words":words,"limit":limit,
            "note":"Coordinates are PDF page-space; drawing scale not verified."}


def overlaps(a, b, clearance=0.0):
    """AABB broad-phase: no semantic collision or precise mesh contact claim."""
    for i in range(3):
        if a[1][i]+clearance < b[0][i] or b[1][i]+clearance < a[0][i]:
            return False
    return True


def ifc_broadphase(filename: str, class_a: str, class_b: str, limit: int):
    """Optional IfcOpenShell world-coordinate broad phase; no false collision verdict."""
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError as exc:
        raise RuntimeError("ifcopenshell absent; use the task CAD MCP or available IFC tool") from exc
    if limit < 1 or limit > 2000:
        raise ValueError("element limit 1..2000 required")
    obj=ifcopenshell.open(filename)
    settings=ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS,True)

    bounds_by_id = {}

    def boxes(class_names):
        found = []
        skipped = 0
        seen = set()
        for class_name in class_names.split(","):
            class_name = class_name.strip()
            if not re.fullmatch(r"Ifc[A-Za-z0-9]{2,50}", class_name):
                raise ValueError(f"invalid IFC class: {class_name}")
            for element in obj.by_type(class_name):
                uid = getattr(element, "GlobalId", None)
                identity = uid if uid else element.id()
                if identity in seen:
                    continue
                seen.add(identity)
                if not uid or len(found) >= limit:
                    skipped += 1
                    continue
                if uid not in bounds_by_id:
                    bounds = None
                    try:
                        shape = ifcopenshell.geom.create_shape(settings, element)
                        coords = list(shape.geometry.verts)
                        if coords and len(coords) % 3 == 0:
                            axes = [coords[i::3] for i in range(3)]
                            bounds = ([min(axis) for axis in axes], [max(axis) for axis in axes])
                    except (AttributeError, RuntimeError, ValueError):
                        pass
                    bounds_by_id[uid] = bounds
                bounds = bounds_by_id[uid]
                if bounds is None:
                    skipped += 1
                else:
                    found.append((uid, class_name, *bounds))
        return found, skipped

    left,skip_left=boxes(class_a)
    right,skip_right=boxes(class_b)
    left_rank = {entry[0]: i for i, entry in enumerate(left)}
    right_ids = {entry[0] for entry in right}
    witnesses=[]; comparisons=0;total=0
    for i, (aid,ac,alo,ahi) in enumerate(left):
        for bid,bc,blo,bhi in right:
            if aid==bid: continue
            # The reverse pair was visited only if both switched roles were eligible.
            if aid in right_ids and left_rank.get(bid, i) < i:
                continue
            comparisons+=1
            if overlaps((alo,ahi),(blo,bhi)):
                total+=1
                if len(witnesses)<100:
                    witnesses.append({"a":aid,"a_class":ac,"b":bid,"b_class":bc,
                                      "overlap_envelope":[max(0.0,min(ahi[i],bhi[i])-max(alo[i],blo[i])) for i in range(3)]})
    return {"kind":"ifc-broad-phase","file":filename,"a_count":len(left),
            "b_count":len(right),"skipped_a":skip_left,"skipped_b":skip_right,
            "comparisons":comparisons,"candidate_pairs":total,
            "witnesses":witnesses,
            "warning":"Envelope overlap only; not a verified mesh clash. Verify units, tolerances, geometry and allowed penetrations before engineering decisions."}



# Offline, dimension-checked calculation DAG: verifies supplied expressions and
# limit values, not the physical assumptions or governing engineering codes.
def calc_pack(filename: str):
    import hashlib
    path = pathlib.Path(filename)
    if path.stat().st_size > 2_000_000:
        raise ValueError("calc-pack input exceeds 2 MB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "hamon.engineering.calcpack.v1":
        raise ValueError("expected hamon.engineering.calcpack.v1")
    definitions, checks = data.get("quantities"), data.get("checks")
    if not isinstance(definitions, dict) or not isinstance(checks, list):
        raise ValueError("quantities object and checks array required")
    if not 1 <= len(definitions) <= 256 or len(checks) > 256:
        raise ValueError("quantity/check count exceeds limits")
    for name, spec in definitions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}",name) or name in UNITS or name == "sqrt":
            raise ValueError(f"invalid or reserved quantity name {str(name)[:64]!r}")
        if not isinstance(spec, dict) or not isinstance(spec.get("expr"),str) or len(spec["expr"]) > 1000:
            raise ValueError(f"{name}: bounded expression string required")
    values, visiting = {}, set()
    def resolve(name):
        if name in values: return values[name]
        if name in visiting: raise ValueError(f"circular quantity dependency at {name}")
        visiting.add(name)
        try:
            expr = definitions[name]["expr"]
            deps = {n.id for n in ast.walk(ast.parse(expr,mode="eval"))
                    if isinstance(n, ast.Name) and n.id in definitions}
            symbols = {n:resolve(n) for n in sorted(deps)}
            values[name] = quantity(expr, symbols)
            return values[name]
        finally:
            visiting.remove(name)
    for name in sorted(definitions): resolve(name)
    quantities, warnings = {}, []
    for name in sorted(definitions):
        spec, q = definitions[name], values[name]
        target = spec.get("to")
        if target is not None and (not isinstance(target,str) or len(target)>100):
            raise ValueError(f"{name}: invalid target unit")
        if target:
            u = quantity(target)
            if q.dim != u.dim or u.val <= 0:
                raise ValueError(f"{name}: incompatible target unit")
            shown = q.val/u.val
        else:
            shown = q.val
        source = spec.get("source")
        if not source: warnings.append(f"{name}: supplied quantity lacks a source")
        quantities[name] = {"expr":spec["expr"],"si_value":str(q.val),
                            "value":str(shown),"target":target,
                            "dimensions":list(q.dim),"source":source}
    out, used = [], set()
    for n, spec in enumerate(checks):
        if not isinstance(spec,dict): raise ValueError(f"check {n}: object required")
        key, op = spec.get("id"), spec.get("relation")
        if not isinstance(key,str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}",key) or key in used:
            raise ValueError(f"check {n}: invalid or duplicate id")
        used.add(key)
        if op not in ("<","<=",">",">=","==","~="):
            raise ValueError(f"{key}: unsupported relation")
        lhs, rhs = spec.get("lhs"), spec.get("rhs")
        if not isinstance(lhs,str) or not isinstance(rhs,str):
            raise ValueError(f"{key}: two expressions required")
        a,b = quantity(lhs,values),quantity(rhs,values)
        if a.dim != b.dim: raise ValueError(f"{key}: dimension mismatch {a.dim} vs {b.dim}")
        tolerance = D(spec.get("relative_tolerance","0"))
        if not tolerance.is_finite() or tolerance < 0 or tolerance >= 1:
            raise ValueError(f"{key}: invalid relative tolerance")
        diff = a.val-b.val
        if op == "<": passed=diff<0
        elif op == "<=": passed=diff<=0
        elif op == ">": passed=diff>0
        elif op == ">=": passed=diff>=0
        elif op == "==": passed=diff==0
        else: passed=abs(diff)<=tolerance*max(abs(a.val),abs(b.val),D("1e-30"))
        if not spec.get("source"): warnings.append(f"{key}: limit state lacks a source")
        out.append({"id":key,"lhs":lhs,"relation":op,"rhs":rhs,
                    "passed":passed,"lhs_si":str(a.val),"rhs_si":str(b.val),
                    "signed_residual_si":str(diff),"dimensions":list(a.dim),
                    "utilization":str(a.val/b.val) if b.val>0 else None,
                    "source":spec.get("source")})
    assumptions = data.get("assumptions",[])
    if not isinstance(assumptions,list) or len(assumptions)>128 or any(
        not isinstance(x,str) or len(x)>1000 for x in assumptions):
        raise ValueError("assumptions must be a bounded string array")
    if not checks: warnings.append("no design checks supplied")
    digest=hashlib.sha256(json.dumps(data,sort_keys=True,
                          separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    return {"kind":"engineering-calculation-pack","input_sha256":digest,
            "all_checks_pass":all(c["passed"] for c in out),
            "quantities":quantities,"checks":out,"warnings":warnings,
            "assumptions":assumptions,"code_compliance_checked":False,
            "note":"Algebraic verification only. Independently verify the physics, source documents, load cases, assumptions and governing design code."}


_STUB_MARKERS = frozenset({
    "[tool call]",
    "<tool_call>",
    "(pi stopped without a final message)",
})


def audit_local_deliverable(path: str | pathlib.Path) -> dict[str, Any]:
    """Deterministic format, syntax, STEP/IFC & stub hygiene guard on a deliverable file."""
    p = pathlib.Path(path)
    if not p.is_file():
        return {"ok": False, "path": str(p), "reason": "missing"}
    sz = p.stat().st_size
    if sz == 0:
        return {"ok": False, "path": str(p), "reason": "empty"}
    ext = p.suffix.lower()
    if ext == ".json":
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"ok": False, "path": str(p), "reason": f"invalid_json:{type(exc).__name__}"}
    elif ext == ".csv":
        try:
            rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8", errors="strict"))))
            if not rows or any(len(r) != len(rows[0]) for r in rows):
                return {"ok": False, "path": str(p), "reason": "ragged_or_empty_csv"}
        except Exception as exc:
            return {"ok": False, "path": str(p), "reason": f"invalid_csv:{type(exc).__name__}"}
    elif ext in (".md", ".txt"):
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        if not txt or txt.lower() in _STUB_MARKERS:
            return {"ok": False, "path": str(p), "reason": "placeholder_or_stub_text"}
    elif ext == ".py":
        try:
            src = p.read_text(encoding="utf-8", errors="strict")
            tree = ast.parse(src)
            if not tree.body:
                return {"ok": False, "path": str(p), "reason": "empty_python_ast"}
        except Exception as exc:
            return {"ok": False, "path": str(p), "reason": f"invalid_python:{type(exc).__name__}"}
    elif ext in (".ifc", ".step", ".stp"):
        head_tail = p.read_text(encoding="utf-8", errors="replace")
        if "ISO-10303-21;" not in head_tail[:512] or "DATA;" not in head_tail or "END-ISO-10303-21;" not in head_tail[-512:]:
            return {"ok": False, "path": str(p), "reason": "malformed_ifc_step_structure"}
    elif ext in (".xlsx", ".xlsm"):
        audit = xlsx_audit(str(p))
        err_cells = sum(s.get("error_count", 0) for s in audit.get("sheets", []))
        mismatches = audit.get("lookup_audit", {}).get("counts", {}).get("mismatch", 0)
        conversion = audit["conversion_audit"]["counts"]
        conversion_errors = (conversion["mismatch"] + conversion["dimension_mismatch"]
                             + conversion.get("unit_scale_mismatch", 0))
        arithmetic = audit["arithmetic_audit"]["counts"]
        dimension_errors = arithmetic.get("dimension_mismatch", 0)
        scale_errors = arithmetic.get("unit_scale_mismatch", 0)
        if err_cells > 0 or mismatches > 0 or conversion_errors or arithmetic["mismatch"] or dimension_errors or scale_errors:
            detail = f"error_cells={err_cells},lookup_mismatches={mismatches}"
            if arithmetic["mismatch"]:
                detail += f",arithmetic_mismatches={arithmetic['mismatch']}"
            if dimension_errors:
                detail += f",arithmetic_dimension_mismatches={dimension_errors}"
            if scale_errors:
                detail += f",arithmetic_unit_scale_mismatches={scale_errors}"
            if conversion_errors:
                detail += f",conversion_errors={conversion_errors}"
            return {
                "ok": False,
                "path": str(p),
                "reason": f"xlsx_errors({detail})",
            }
        unresolved = {"arithmetic": arithmetic["unverified"], "conversion": conversion["unverified"],
                      "lookup": audit["lookup_audit"]["counts"]["unverified"]}
        if any(unresolved.values()):
            return {"ok": True, "path": str(p), "bytes": sz, "unverified_formula_caches": unresolved}
    elif ext in (".zip", ".docx"):
        try:
            with zipfile.ZipFile(p, "r") as zf:
                if not zf.namelist() or zf.testzip() is not None:
                    return {"ok": False, "path": str(p), "reason": "corrupt_or_empty_zip"}
                if ext == ".docx":
                    try:
                        check_office_package(zf, ext)
                    except (ValueError, ET.ParseError) as exc:
                        return {"ok": False, "path": str(p), "reason": f"invalid_docx:{exc}"}
        except Exception:
            return {"ok": False, "path": str(p), "reason": "bad_zip"}
    return {"ok": True, "path": str(p), "bytes": sz}


def run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    we=sub.add_parser("witness-equipment", help="Join explicit as-installed PDF evidence to an exact model row in a named workbook edition")
    we.add_argument("--equipment-type", choices=("inverter","pcs"), default="inverter")
    we.add_argument("pdf"); we.add_argument("catalog"); we.add_argument("--edition", required=True)
    we.add_argument("--sheet", required=True); we.add_argument("--title-sheet", required=True)
    wp=sub.add_parser("witness-package", help="Reconcile dated scope, release evidence and standing case items before writing a Word package")
    wp.add_argument("file", nargs="?")
    wp.add_argument("--example", action="store_true", help="print a constructed input template")
    wp.add_argument("--output", help="new DOCX path, written only on an exact schedule match")
    draft=sub.add_parser("materialize-draft",help="Export a fresh participant-authored Word or archive draft")
    for name in ("source", "target", "fingerprint", "members"):
        draft.add_argument(name)
    q=sub.add_parser("quantity",help="Convert a dimension-safe arithmetic expression")
    q.add_argument("expression");q.add_argument("--to",required=True)
    e=sub.add_parser("check",help="Assert two physical quantities match under relative tolerance")
    e.add_argument("lhs");e.add_argument("rhs");e.add_argument("--rtol",default="0.000001")
    x=sub.add_parser("xlsx-audit",help="Inspect workbook formula caches and error cells")
    x.add_argument("file")
    f=sub.add_parser("pdf-words",help="Extract PDF word boxes on one page (not drawing scale)")
    f.add_argument("file");f.add_argument("--page",type=int,default=1);f.add_argument("--limit",type=int,default=250)
    links=sub.add_parser("pdf-detail-links",help="Resolve printed detail references across drawing sheets")
    links.add_argument("file")
    links.add_argument("--target-sheet", help="Return references to this printed sheet ID after indexing the entire document")
    bom=sub.add_parser("pdf-bom-sheet",help="Find a schedule part on a uniquely titled drawing sheet")
    bom.add_argument("file")
    bom.add_argument("--equipment", required=True)
    bom.add_argument("--part-id", required=True)
    bom.add_argument("--title", required=True)
    native=sub.add_parser("eagle-netlist",help="Read native EAGLE XML schematic pins and physical pad nets")
    native.add_argument("file")
    native.add_argument("--board",help="Compare predicted pad nets with an EAGLE XML PCB")
    native.add_argument("--contact-nets",nargs=2,metavar=("FIRST","SECOND"),
                        help="Check two board signals for explicit same-layer wire overlap")
    native.add_argument("--routed-net",help="Check a named signal for copper paths between schematic terminals")
    native.add_argument("--routed-pads",nargs=4,metavar=("PART1","PAD1","PART2","PAD2"),
                        help="Check only this pair of schematic pads on the selected net")
    eagle=sub.add_parser("pdf-eagle-connectivity",help="Read wire connectivity evidence from a two-color EAGLE schematic")
    eagle.add_argument("file")
    scope=eagle.add_mutually_exclusive_group()
    scope.add_argument("--page",type=int,default=1)
    scope.add_argument("--flat-document",action="store_true",help="Link matching printed net anchors across sheets of one flat EAGLE design")
    dv=sub.add_parser("pdf-vector",help="Multi-sheet native PDF vector and printed-scale inventory")
    dv.add_argument("file");dv.add_argument("--from-page",type=int,default=1)
    dv.add_argument("--to-page",type=int,default=None)
    dv.add_argument("--limit-paths",type=int,default=250)
    dv.add_argument("--limit-words",type=int,default=500)
    dv.add_argument("--region",type=float,nargs=4,default=None,
                    metavar=("XMIN","YMIN","XMAX","YMAX"))
    dw=sub.add_parser("pdf-window",help="Render a bounded PNG in the same SVG coordinate frame, with source provenance")
    dw.add_argument("file");dw.add_argument("--output",required=True)
    dw.add_argument("--page",type=int,default=1)
    dw.add_argument("--dpi",type=float,default=120)
    dw.add_argument("--limit-paths",type=int,default=128)
    dw.add_argument("--region",type=float,nargs=4,default=None,
                    metavar=("XMIN","YMIN","XMAX","YMAX"))
    dx=sub.add_parser("dxf-dimensions",help="Read native ASCII DXF dimensions and qualify linear source geometry")
    dx.add_argument("file");dx.add_argument("--limit",type=int,default=1000)
    dm=sub.add_parser("pdf-measure",help="Source-anchored metric length estimates from real PDF SVG vertices")
    dm.add_argument("file");dm.add_argument("spec")
    dl=sub.add_parser("pdf-dimension-links",help="Ground printed dimensions to actual PDF vector lines with two witnessed ends")
    dl.add_argument("file");dl.add_argument("--page",type=int,default=1)
    dl.add_argument("--dxf",help="Register outlined PDF marker axes to this explicit ASCII DXF sidecar")
    da=sub.add_parser("pdf-auto-measure",help="Conditionally calibrate and measure from machine-witnessed source dimensions")
    da.add_argument("file");da.add_argument("spec")
    c=sub.add_parser("ifc-broadphase",help="Optional IfcOpenShell world-coordinate envelope overlaps")
    c.add_argument("file");c.add_argument("--a",default="IfcDuctSegment,IfcPipeSegment")
    c.add_argument("--b",default="IfcBeam,IfcColumn");c.add_argument("--limit",type=int,default=400)
    pack=sub.add_parser("calc-pack",help="Audit a source-qualified unit-safe calculation graph")
    pack.add_argument("file")
    n=sub.add_parser("ifc-narrowphase",help="Qualified IFC mesh contact/containment screen")
    n.add_argument("file")
    n.add_argument("--a",default="IfcDuctSegment,IfcPipeSegment")
    n.add_argument("--b",default="IfcBeam,IfcColumn")
    n.add_argument("--limit",type=int,default=400)
    n.add_argument("--triangle-budget",type=int,default=250000)
    n.add_argument("--min-clearance-mm",type=float,default=None)
    dvf=sub.add_parser("dual-verify",help="Symbolic + 7-dim SI + numerical dual-solver engineering verifier")
    dvf.add_argument("file")
    dxt=sub.add_parser("dxf-takeoff",help="SI-calibrated DXF LINE/LWPOLYLINE/CIRCLE/ARC/INSERT spatial takeoff")
    dxt.add_argument("file")
    dxt.add_argument("--existing-layer",help="Existing triangular terrain surface layer; requires --proposed-layer")
    dxt.add_argument("--proposed-layer",help="Proposed triangular terrain surface layer; requires --existing-layer")
    fmg=sub.add_parser("format-guard",help="Audit local deliverable JSON/CSV/XLSX/ZIP format and formula caches")
    fmg.add_argument("file")
    args=p.parse_args(argv)
    try:
        if args.cmd=="witness-equipment":
            out=witness_equipment(args.pdf,args.catalog,args.edition,args.sheet,args.title_sheet,args.equipment_type)
        elif args.cmd=="materialize-draft":
            from hamon_delivery_draft import render
            out=render(args.source,args.target,json.loads(args.members),args.fingerprint)
        elif args.cmd=="witness-package":
            import hamon_witness_package
            if args.example and args.file is None and args.output is None:
                print(json.dumps(hamon_witness_package.example(), indent=2))
                return 0
            if not args.file or args.example:
                raise ValueError("supply a spec file or use --example by itself")
            out=hamon_witness_package.witness_package(args.file,args.output)
            if not out["schedule_matches"]:
                print(json.dumps({"ok":False,**out},ensure_ascii=True))
                return 3
        elif args.cmd=="calc-pack": out=calc_pack(args.file)
        elif args.cmd=="quantity": out=conversion(args.expression,args.to)
        elif args.cmd=="check": out=equal_dimensions(args.lhs,args.rhs,args.rtol)
        elif args.cmd=="xlsx-audit": out=xlsx_audit(args.file)
        elif args.cmd=="pdf-words": out=pdf_words(args.file,args.page,args.limit)
        elif args.cmd=="pdf-window":
            import hamon_drawing
            out=hamon_drawing.render_window(args.file,args.output,args.page,args.region,
                                           args.dpi,args.limit_paths)
        elif args.cmd=="pdf-bom-sheet":
            import hamon_drawing
            out=hamon_drawing.bom_sheet(args.file,args.equipment,args.part_id,args.title)
        elif args.cmd=="pdf-detail-links":
            import hamon_drawing
            out=hamon_drawing.detail_links(args.file, args.target_sheet)
        elif args.cmd=="eagle-netlist":
            from hamon_eagle_netlist import schematic_netlist, compare_board
            out=schematic_netlist(args.file)
            if args.board:
                out["board_comparison"]=compare_board(out,args.board)
            if args.contact_nets is not None:
                if not args.board:
                    raise ValueError("--contact-nets requires --board")
                from hamon_eagle_copper import inter_net_contact
                out["inter_net_contact"]=inter_net_contact(args.board,args.contact_nets)
            if args.routed_pads is not None and args.routed_net is None:
                raise ValueError("--routed-pads requires --routed-net")
            if args.routed_net is not None:
                if not args.board:
                    raise ValueError("--routed-net requires --board")
                from hamon_eagle_copper import routed_continuity
                out["routed_continuity"]=routed_continuity(out,args.board,args.routed_net,pad_pair=args.routed_pads)
        elif args.cmd=="pdf-eagle-connectivity":
            import hamon_eagle_schematic
            if args.flat_document:
                out=hamon_eagle_schematic.document_connectivity(args.file)
            else:
                out=hamon_eagle_schematic.page_connectivity(args.file,args.page)
        elif args.cmd=="pdf-vector":
            import hamon_drawing
            out=hamon_drawing.scan_pdf(
                args.file,args.from_page,args.to_page,args.limit_paths,
                args.limit_words,args.region)
        elif args.cmd=="dxf-dimensions":
            import hamon_dxf
            out=hamon_dxf.scan_dxf(args.file,args.limit)
        elif args.cmd=="pdf-measure":
            import hamon_scale
            out=hamon_scale.measure_pdf(args.file,args.spec,quantity,UNITS["m"])
        elif args.cmd=="pdf-dimension-links":
            import hamon_callout
            out=(hamon_callout.associate_sidecar(args.file,args.dxf,args.page) if args.dxf
                 else hamon_callout.associate_pdf(args.file,args.page))
        elif args.cmd=="pdf-auto-measure":
            import hamon_callout
            out=hamon_callout.auto_measure_pdf(args.file,args.spec,quantity,UNITS["m"])
        elif args.cmd=="ifc-broadphase": out=ifc_broadphase(args.file,args.a,args.b,args.limit)
        elif args.cmd=="ifc-narrowphase":
            import hamon_geometry
            out=hamon_geometry.ifc_narrowphase(
                args.file,args.a,args.b,args.limit,args.triangle_budget,args.min_clearance_mm)
        elif args.cmd=="dual-verify":
            import hamon_dual_verifier
            out=hamon_dual_verifier.run_dual_verify_file(args.file)
        elif args.cmd=="dxf-takeoff":
            if (args.existing_layer is None) != (args.proposed_layer is None):
                raise ValueError("terrain takeoff requires both --existing-layer and --proposed-layer")
            if args.existing_layer is not None:
                import hamon_terrain_volume
                out=hamon_terrain_volume.scan_dxf_terrain_volume(
                    args.file,args.existing_layer,args.proposed_layer)
            else:
                import hamon_cad_takeoff
                out=hamon_cad_takeoff.scan_dxf_takeoff(args.file)
        elif args.cmd=="format-guard":
            out = audit_local_deliverable(args.file)
            if not out.get("ok", False):
                print(json.dumps(out,ensure_ascii=True,default=str))
                return 3
        else: raise ValueError("unsupported command")
        print(json.dumps({"ok":True,**out},ensure_ascii=True,default=str))
        if args.cmd=="dxf-takeoff" and args.existing_layer is not None:
            return 0 if out.get("qualified") is True else 3
        return 3 if args.cmd=="calc-pack" and not out["all_checks_pass"] else 0
    except (OSError,ValueError,InvalidOperation,RuntimeError,ImportError,ET.ParseError,zipfile.BadZipFile,
            subprocess.TimeoutExpired,ZeroDivisionError,OverflowError,SyntaxError) as exc:
        print(json.dumps({"ok":False,"error":type(exc).__name__,
                          "message":str(exc)[:300],"command":args.cmd}))
        return 2

if __name__=="__main__":
    sys.exit(run())

