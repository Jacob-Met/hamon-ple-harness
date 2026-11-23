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
import io
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

from office_package import (
    _DOCUMENT_RELS, _SPREADSHEET_NS, _xlsx_relationships, _xlsx_target, _xlsx_xml,
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


def _xlsx_parts(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate OOXML package member")
    if "_rels/.rels" not in names and "xl/workbook.xml" not in names:
        # Earlier callers also audit standalone worksheet fragments.
        parts = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        if not parts:
            raise ValueError("No workbook or standalone worksheet parts")
        return None, [{"part": n, "kind": "worksheet", "name": None} for n in parts]
    if "_rels/.rels" not in names:
        workbook = "xl/workbook.xml"
    else:
        office = [r for r in _xlsx_relationships(archive, "_rels/.rels").values()
                  if r.get("Type") in {base + "/officeDocument" for base in _DOCUMENT_RELS}]
        if len(office) != 1 or office[0].get("TargetMode", "Internal") != "Internal":
            raise ValueError("Expected exactly one internal workbook relationship")
        workbook = _xlsx_target("", office[0].get("Target", ""))
    root = _xlsx_xml(archive, workbook, 2_000_000)
    ns = root.tag.removesuffix("workbook")
    if ns not in {"{" + v + "}" for v in _SPREADSHEET_NS}:
        raise ValueError(f"Invalid spreadsheet workbook: {workbook}")
    sheets = root.find(ns + "sheets")
    if sheets is None:
        raise ValueError("Workbook is missing its sheets collection")
    source = pathlib.PurePosixPath(workbook)
    rels = _xlsx_relationships(archive, str(source.parent / "_rels" / (source.name + ".rels")))
    result = []
    for sheet in sheets.findall(ns + "sheet"):
        ids = [sheet.get("{" + prefix + "}id") for prefix in _DOCUMENT_RELS]
        ids = [rid for rid in ids if rid is not None]
        if len(ids) != 1 or ids[0] not in rels:
            raise ValueError(f"Unresolved relationship for sheet {sheet.get('name')!r}")
        rel = rels[ids[0]]
        if rel.get("TargetMode", "Internal") != "Internal":
            raise ValueError(f"Workbook sheet {sheet.get('name')!r} has an external target")
        base, _, kind = rel.get("Type", "").rpartition("/")
        if base not in _DOCUMENT_RELS:
            kind = rel.get("Type", "unknown")
        result.append({"name": sheet.get("name"), "sheet_id": sheet.get("sheetId"),
                       "state": sheet.get("state", "visible"), "kind": kind,
                       "part": _xlsx_target(workbook, rel.get("Target", ""))})
    return workbook, result


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
    }.items()
}
_DIRECT_CONVERT = re.compile(
    rf"CONVERT\(\s*(?:(?P<sheet>'(?:[^']|'')+'|[A-Z_][A-Z0-9_.]*)!)?"
    rf'(?P<source>{_CELL_REF})\s*,\s*"(?P<from_unit>[A-Z]+)"\s*,\s*'
    r'"(?P<to_unit>[A-Z]+)"\s*\)', re.I | re.ASCII)


def _conversion_audit(formulas, literals):
    """Reconstruct normal CONVERT chains independently of all saved formula caches."""
    producers = {(sheet, addr): expression for sheet, addr, expression, _ in formulas}
    evaluated = {}
    visiting = set()

    def evaluate(key):
        if key in evaluated:
            return evaluated[key]
        if key in visiting:
            return {"reason": "cyclic_reference"}
        visiting.add(key)
        result = calculate(key)
        visiting.remove(key)
        evaluated[key] = result
        return result

    def calculate(key):
        sheet, _ = key
        match = _DIRECT_CONVERT.fullmatch(producers[key])
        if match is None:
            return {"reason": "unsupported_conversion"}
        ref_sheet = match["sheet"] or sheet
        if ref_sheet.startswith("'"):
            ref_sheet = ref_sheet[1:-1].replace("''", "'")
        if any(c in ref_sheet for c in "[]:"):
            return {"reason": "external_or_multisheet_reference"}
        names = [n for n in literals if n == ref_sheet or (
            n.isascii() and ref_sheet.isascii() and n.lower() == ref_sheet.lower())]
        if len(names) != 1:
            return {"reason": "unknown_or_ambiguous_worksheet"}
        ref = (names[0], match["source"].replace("$", "").upper())
        try:
            _grid_position(ref[1])
        except ValueError:
            return {"reason": "invalid_cell_reference"}
        source = _CONVERT_UNITS.get(match["from_unit"])
        target = _CONVERT_UNITS.get(match["to_unit"])
        if source is None or target is None:
            return {"reason": "unsupported_unit"}
        value = literals[ref[0]].get(ref[1])
        root_sheet, root_cell = ref
        depth = 1
        if value is None:
            if ref not in producers:
                return {"reason": "unsupported_or_missing_dependency"}
            prior = evaluate(ref)
            if "expected" not in prior:
                return {"reason": prior.get("reason", "invalid_upstream_conversion")}
            if prior["to_unit"] != match["from_unit"]:
                return {"reason": "upstream_unit_disagrees", "upstream_unit": prior["to_unit"],
                        "from_unit": match["from_unit"]}
            value = prior["expected"]
            root_sheet, root_cell = prior["root_sheet"], prior["root_cell"]
            depth = prior["conversion_depth"] + 1
        evidence = {"source_sheet": ref[0], "source_cell": ref[1], "source_value": value,
                    "root_sheet": root_sheet, "root_cell": root_cell, "conversion_depth": depth,
                    "from_unit": match["from_unit"], "to_unit": match["to_unit"]}
        if source.dim != target.dim:
            return {**evidence, "dimension_mismatch": True}
        expected = float(D(value) * source.val / target.val)
        if not math.isfinite(expected) or (expected == 0 and value != 0):
            return {"reason": "numeric_range"}
        return {**evidence, "expected": expected}

    for sheet, address, _, cached in formulas:
        evidence = evaluate((sheet, address))
        status = "unverified"
        if cached is not None:
            if evidence.get("dimension_mismatch"):
                status = "dimension_mismatch"
            elif "expected" in evidence:
                agrees = math.isclose(cached, evidence["expected"], rel_tol=1e-12, abs_tol=0.0)
                status = "matched" if agrees else "mismatch"
        yield {"sheet": sheet, "cell": address, **evidence, "cached": cached, "status": status}


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


def xlsx_audit(filename: str):
    """Audit declared worksheet cells and formula caches without evaluating a workbook."""
    manifest = {"file": filename, "kind": "xlsx-formula-audit", "sheets": [], "other_sheets": []}
    literals = {}
    lookups = []
    conversions = []
    conversion_total = lookup_total = indexed = 0
    complete_index = True
    with zipfile.ZipFile(filename) as archive:
        workbook, parts = _xlsx_parts(archive)
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
                            conversions.append((name, address, expression, _numeric_cell(cell, value)))
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
            elif derived_ranges and complete_index:
                def is_derived(address):
                    x, y = _grid_position(address)
                    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in derived_ranges)

                try:
                    for address, numeric in sheet_values.items():
                        if numeric is not None and is_derived(address):
                            sheet_values[address] = None
                    conversions[first_conversion:] = [
                        row for row in conversions[first_conversion:] if not is_derived(row[1])
                    ]
                except ValueError:
                    sheet_values.clear()
                    del conversions[first_conversion:]
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
    counts = {"matched": 0, "mismatch": 0, "dimension_mismatch": 0, "unverified": conversion_total}
    examples = []
    if complete_index:
        for result in _conversion_audit(conversions, literals):
            if result["status"] != "unverified":
                counts["unverified"] -= 1
                counts[result["status"]] += 1
            examples.append(result)
    manifest["conversion_audit"] = {"counts": counts, "examples": examples}
    manifest["note"] = "Checked lookups use literal cells. Checked conversion chains start at a numeric literal. Other caches can be stale."
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
        conversion_errors = conversion["mismatch"] + conversion["dimension_mismatch"]
        if err_cells > 0 or mismatches > 0 or conversion_errors:
            detail = f"error_cells={err_cells},lookup_mismatches={mismatches}"
            if conversion_errors:
                detail += f",conversion_errors={conversion_errors}"
            return {
                "ok": False,
                "path": str(p),
                "reason": f"xlsx_errors({detail})",
            }
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
    wp=sub.add_parser("witness-package", help="Reconcile dated scope, release evidence and standing case items before writing a Word package")
    wp.add_argument("file", nargs="?")
    wp.add_argument("--example", action="store_true", help="print a constructed input template")
    wp.add_argument("--output", help="new DOCX path, written only on an exact schedule match")
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
        if args.cmd=="witness-package":
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
