"""Deterministic, local-only engineering checks for PLE task containers.

A helper, not a design authority. IFC bounding boxes are BROAD PHASE ONLY,
PDF coordinates are page-space (not scale verified), and formula cache presence
is not calculation correctness. None of these commands modify task files.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

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
    "psi": unit("6894.757293168361",PRESSURE),
    "ksi": unit("6894757.293168361",PRESSURE),
    "psf": unit("47.88025898033584",PRESSURE),
    "ksf": unit("47880.25898033584",PRESSURE),
    "J": unit(1,ENERGY), "kJ": unit(1000,ENERGY),
    "W": unit(1,POWER), "kW": unit(1000,POWER),
    "V": unit(1,VOLTAGE), "ohm": unit(1,RESISTANCE),
}

def quantity(expression: str) -> Q:
    """Evaluate an arithmetic unit expression, never Python eval or arbitrary calls."""
    if len(expression) > 1000:
        raise ValueError("quantity expression exceeds 1000 characters")
    tree = ast.parse(expression, mode="eval")

    def walk(node, depth=0):
        if depth > 35:
            raise ValueError("expression nesting limit")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Q(D(node.value))
        if isinstance(node, ast.Name):
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


def xlsx_audit(filename: str):
    """Read OOXML XML without computing formulas or trusting cached result values."""
    manifest={"file":filename,"kind":"xlsx-formula-audit","sheets":[]}
    with zipfile.ZipFile(filename) as z:
        names=[s for s in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml",s)]
        if len(names)>300:
            raise ValueError("over 300 worksheets; run focused audit")
        for name in sorted(names):
            if z.getinfo(name).file_size > 25_000_000:
                raise ValueError(f"{name}: uncompressed worksheet exceeds 25 MB")
            root=ET.fromstring(z.read(name))
            ns="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
            formulas=0;missing_cache=0;errors=[];cells=0
            for cell in root.iter(ns+"c"):
                cells+=1
                f=cell.find(ns+"f")
                if f is not None:
                    formulas+=1
                    if cell.find(ns+"v") is None:
                        missing_cache+=1
                if cell.attrib.get("t") == "e":
                    v=cell.find(ns+"v")
                    errors.append({"cell":cell.attrib.get("r",""),"value":v.text if v is not None else ""})
            manifest["sheets"].append({"part":name,"cells":cells,"formulas":formulas,
                  "formulas_without_cached_values":missing_cache,"error_cells":errors[:50],
                  "error_count":len(errors)})
    manifest["note"]="Formula results are not evaluated; cached results can be stale."
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

    def boxes(class_names):
        found=[]; skipped=0
        for class_name in class_names.split(","):
            class_name=class_name.strip()
            if not re.fullmatch(r"Ifc[A-Za-z0-9]{2,50}",class_name):
                raise ValueError(f"invalid IFC class: {class_name}")
            for el in obj.by_type(class_name):
                if len(found)>=limit:
                    skipped+=1
                    continue
                try:
                    geom=ifcopenshell.geom.create_shape(settings,el).geometry
                    coords=list(geom.verts)
                    if not coords or len(coords)%3:
                        skipped+=1;continue
                    axes=[coords[i::3] for i in range(3)]
                    lo=[min(axes[i]) for i in range(3)]
                    hi=[max(axes[i]) for i in range(3)]
                    found.append((str(el.GlobalId),class_name,lo,hi))
                except (AttributeError,RuntimeError,ValueError):
                    skipped+=1
        return found,skipped
    left,skip_left=boxes(class_a)
    right,skip_right=boxes(class_b)
    witnesses=[]; comparisons=0;total=0
    for aid,ac,alo,ahi in left:
        for bid,bc,blo,bhi in right:
            if aid==bid: continue
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


def run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("quantity",help="Convert a dimension-safe arithmetic expression")
    q.add_argument("expression");q.add_argument("--to",required=True)
    e=sub.add_parser("check",help="Assert two physical quantities match under relative tolerance")
    e.add_argument("lhs");e.add_argument("rhs");e.add_argument("--rtol",default="0.000001")
    x=sub.add_parser("xlsx-audit",help="Inspect workbook formula caches and error cells")
    x.add_argument("file")
    f=sub.add_parser("pdf-words",help="Extract PDF word boxes on one page (not drawing scale)")
    f.add_argument("file");f.add_argument("--page",type=int,default=1);f.add_argument("--limit",type=int,default=250)
    c=sub.add_parser("ifc-broadphase",help="Optional IfcOpenShell world-coordinate envelope overlaps")
    c.add_argument("file");c.add_argument("--a",default="IfcDuctSegment,IfcPipeSegment")
    c.add_argument("--b",default="IfcBeam,IfcColumn");c.add_argument("--limit",type=int,default=400)
    args=p.parse_args(argv)
    try:
        if args.cmd=="quantity": out=conversion(args.expression,args.to)
        elif args.cmd=="check": out=equal_dimensions(args.lhs,args.rhs,args.rtol)
        elif args.cmd=="xlsx-audit": out=xlsx_audit(args.file)
        elif args.cmd=="pdf-words": out=pdf_words(args.file,args.page,args.limit)
        elif args.cmd=="ifc-broadphase": out=ifc_broadphase(args.file,args.a,args.b,args.limit)
        else: raise ValueError("unsupported command")
        print(json.dumps({"ok":True,**out},ensure_ascii=True,default=str))
        return 0
    except (OSError,ValueError,InvalidOperation,RuntimeError,ET.ParseError,zipfile.BadZipFile,
            subprocess.TimeoutExpired,ZeroDivisionError,OverflowError,SyntaxError) as exc:
        print(json.dumps({"ok":False,"error":type(exc).__name__,
                          "message":str(exc)[:300],"command":args.cmd}))
        return 2

if __name__=="__main__":
    sys.exit(run())
