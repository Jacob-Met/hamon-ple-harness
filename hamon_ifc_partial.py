"""Partial occurrence-level takeoff and pair coverage for the ordinary IFC reader."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
import math
import re

import numpy as np

import hamon_ifc_spatial_clash as reader


def _ref(value):
    return value[1] if isinstance(value, tuple) and value[0] == "ref" else None


def _source_sections(text):
    """Locate the single STEP data section without interpreting delimiters inside strings."""
    lexical = re.compile(r"'(?:[^']|'')*'|/\*.*?\*/|/\*|'", re.S)
    masked, cleaned = [], []
    start = 0
    for token in lexical.finditer(text):
        masked.append(text[start:token.start()])
        cleaned.append(text[start:token.start()])
        value = token[0]
        if value in ("'", "/*"):
            raise ValueError("Unterminated STEP string or comment")
        masked.append(" " * len(value))
        cleaned.append(value if value.startswith("'") else " " * len(value))
        start = token.end()
    masked.append(text[start:])
    cleaned.append(text[start:])
    masked, cleaned = "".join(masked), "".join(cleaned)
    sections = re.fullmatch(
        r"\s*ISO-10303-21\s*;\s*HEADER\s*;(?P<header>.*?)\bENDSEC\s*;"
        r"\s*DATA\s*;(?P<data>.*?)\bENDSEC\s*;\s*END-ISO-10303-21\s*;\s*",
        masked, re.I | re.S)
    if sections is None:
        raise ValueError("Expected a complete STEP exchange structure")
    start, end = sections.span("header")
    records = list(re.finditer(r"(?:^|;)\s*FILE_SCHEMA\s*\([^;]*\)\s*(?=;)",
                               masked[start:end], re.I))
    if len(records) != 1:
        raise ValueError("Expected one IFC schema declaration")
    record = records[0]
    schema = re.fullmatch(r";?\s*FILE_SCHEMA\s*\(\s*\(\s*'([A-Za-z0-9_]+)'\s*\)\s*\)\s*",
                          cleaned[start+record.start():start+record.end()], re.I)
    if schema is None:
        raise ValueError("Expected one name in FILE_SCHEMA")
    start, end = sections.span("data")
    return schema[1].upper(), cleaned[start:end]


_TOKEN = re.compile(r"""\s*('(?:[^']|'')*'|"[^"]*"|#\d+|\.[A-Za-z][\w]*\.|[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?|[A-Za-z][A-Za-z0-9_]*|[(),$*;])""")
_RECORD = re.compile(r"\s*#([1-9]\d*)\s*=\s*([A-Za-z][A-Za-z0-9_]*)\s*\(")


def _entities(data):
    """Read one complete STEP table, rejecting duplicate and malformed source records."""
    def token(pos):
        match = _TOKEN.match(data, pos)
        if match is None:
            raise ValueError("Malformed STEP attribute")
        return match[1], match.end()

    def values(pos, depth):
        if depth > 128:
            raise ValueError("Excessive STEP attribute nesting")
        val, pos = token(pos)
        out = []
        if val == ")":
            return out, pos
        while True:
            if val == "(":
                value, pos = values(pos, depth + 1)
            elif val[0].isalpha():
                opening, pos = token(pos)
                if opening != "(":
                    raise ValueError("Malformed typed STEP attribute")
                args, pos = values(pos, depth + 1)
                value = (val.upper(), args)
            elif val in (")", ",", ";"):
                raise ValueError("Missing STEP attribute")
            else:
                value = reader._coerce_step_atom(val)
            out.append(value)
            delimiter, pos = token(pos)
            if delimiter == ")":
                return out, pos
            if delimiter != ",":
                raise ValueError("Malformed STEP attribute delimiter")
            val, pos = token(pos)

    table, pos = {}, 0
    while True:
        record = _RECORD.match(data, pos)
        if record is None:
            if data[pos:].strip():
                raise ValueError("Malformed STEP entity record")
            return table
        step = int(record[1])
        if step in table:
            raise ValueError(f"Duplicate STEP entity #{step}")
        attrs, pos = values(record.end(), 0)
        terminator, pos = token(pos)
        if terminator != ";":
            raise ValueError(f"Missing terminator after STEP entity #{step}")
        table[step] = record[2].upper(), attrs


@lru_cache(maxsize=8)
def _schema_roles(name):
    from ifcopenshell import ifcopenshell_wrapper
    schema = ifcopenshell_wrapper.schema_by_name(name)

    def descendants(type_name):
        stack = [schema.declaration_by_name(type_name)]
        found = set()
        while stack:
            node = stack.pop()
            found.add(node.name().upper())
            stack.extend(node.subtypes())
        return found

    features = descendants("IfcFeatureElement")
    roles = dict.fromkeys(descendants("IfcProduct"), "other")
    physical = descendants("IfcElement") - descendants("IfcVirtualElement") - features
    roles.update(dict.fromkeys(physical, "physical"))
    roles.update(dict.fromkeys(features, "feature"))
    roles.update(dict.fromkeys(descendants("IfcOpeningElement"), "opening"))
    return roles, {node.name().upper() for node in schema.declarations()}


def _model_context(entities):
    projects = [a for k, a in entities.values() if k == "IFCPROJECT"]
    assignments = [("ref", step) for step, (k, _) in entities.items() if k == "IFCUNITASSIGNMENT"]
    contexts = None
    if projects:
        if len(projects) != 1 or len(projects[0]) < 9 or not isinstance(projects[0][7], list):
            raise ValueError("Expected one project with declared representation contexts")
        contexts = {_ref(ref) for ref in projects[0][7]}
        if not contexts or any(entities.get(step, ("",))[0] != "IFCGEOMETRICREPRESENTATIONCONTEXT"
                               for step in contexts):
            raise ValueError("Unresolved project representation context")
        assignment = projects[0][8]
    elif len(assignments) == 1:
        assignment = assignments[0]
    else:
        raise ValueError("No unambiguous project length-unit assignment")

    units, = reader._required_ifc_entity(entities, assignment, "IFCUNITASSIGNMENT")
    for ref in units:
        kind, attrs = entities.get(_ref(ref), ("", []))
        if kind == "IFCCONVERSIONBASEDUNIT" and len(attrs) > 1 and attrs[1] == "LENGTHUNIT":
            dims = reader._required_ifc_entity(entities, attrs[0], "IFCDIMENSIONALEXPONENTS")
            if dims != [1, 0, 0, 0, 0, 0, 0] or any(type(value) is not int for value in dims):
                raise ValueError("Invalid dimensions for the project length conversion")
    return reader._detect_length_unit_scale(entities), contexts


def _refs(value):
    if _ref(value) is not None:
        yield _ref(value)
    elif isinstance(value, list):
        for entry in value:
            yield from _refs(entry)
    elif isinstance(value, tuple):
        yield from _refs(value[1])


def _context(entities, ref, projects):
    step = _ref(ref)
    kind, attrs = entities.get(step, ("", []))
    if kind == "IFCGEOMETRICREPRESENTATIONSUBCONTEXT" and len(attrs) > 6:
        step = _ref(attrs[6])
        kind, attrs = entities.get(step, ("", []))
    if (kind != "IFCGEOMETRICREPRESENTATIONCONTEXT" or len(attrs) < 5
            or type(attrs[2]) is not int or attrs[2] != 3 or (projects is not None and step not in projects)):
        raise ValueError("Body needs a declared 3D project representation context")
    if not np.array_equal(reader._rigid_ifc_frame(entities, attrs[4], 1.), np.eye(4)):
        raise ValueError("Unsupported nonidentity project coordinate system")


def _checked_triangulation(entities, step, attrs):
    if len(attrs) != 5 or attrs[4] is not None or (attrs[2] is not None and attrs[2] is not True):
        raise ValueError(f"Unqualified closure or triangulation indexing on #{step}")
    fields = reader._required_ifc_entity(entities, attrs[0], "IFCCARTESIANPOINTLIST3D")
    points = fields[0]
    if not isinstance(points, list) or not points or not isinstance(attrs[3], list) or not attrs[3]:
        raise ValueError(f"Missing triangles or coordinates on #{step}")
    for point in points:
        if not isinstance(point, list) or len(point) != 3 or any(not math.isfinite(reader._ifc_number(v)) for v in point):
            raise ValueError(f"Invalid triangulation coordinates on #{step}")
    for tri in attrs[3]:
        if not isinstance(tri, list) or len(tri) != 3 or any(type(v) is not int or not 1 <= v <= len(points) for v in tri):
            raise ValueError(f"Invalid triangulation indices on #{step}")
    faces = np.asarray(attrs[3], dtype=np.int64) - 1
    _, closed, coherent = reader._closed_surface_components(np.asarray(points), faces)
    if len(closed) != len(faces) or not coherent:
        raise ValueError(f"Open or inconsistently oriented triangulation on #{step}")


def _validate_body(entities, step, project_contexts, visited, active):
    if step in active or len(active) > 128:
        raise ValueError(f"Cyclic or excessively nested geometry at #{step}")
    if step in visited:
        return
    if step not in entities:
        raise ValueError(f"Missing source geometry #{step}")
    kind, attrs = entities[step]
    if kind == "IFCSHAPEREPRESENTATION":
        if len(attrs) != 4 or str(attrs[1]).upper() != "BODY" or not isinstance(attrs[3], list) or not attrs[3]:
            raise ValueError(f"Incomplete source Body #{step}")
        _context(entities, attrs[0], project_contexts)
    elif kind == "IFCREPRESENTATIONMAP":
        if len(attrs) != 2 or entities.get(_ref(attrs[1]), ("",))[0] != "IFCSHAPEREPRESENTATION":
            raise ValueError(f"Invalid representation map #{step}")
        reader._rigid_ifc_frame(entities, attrs[0], 1.)
    elif kind == "IFCMAPPEDITEM":
        if len(attrs) != 2 or entities.get(_ref(attrs[0]), ("",))[0] != "IFCREPRESENTATIONMAP":
            raise ValueError(f"Invalid mapped Body item #{step}")
    elif kind in ("IFCCARTESIANTRANSFORMATIONOPERATOR3D", "IFCCARTESIANTRANSFORMATIONOPERATOR3DNONUNIFORM"):
        if len(attrs) != (5 if kind.endswith("3D") else 7):
            raise ValueError(f"Malformed transform #{step}")
        for ref in (attrs[0], attrs[1], attrs[4]):
            if ref is not None:
                coords, = reader._required_ifc_entity(entities, ref, "IFCDIRECTION")
                if len(coords) != 3 or not any(reader._ifc_number(v) for v in coords):
                    raise ValueError(f"Invalid transform axis on #{step}")
        coords, = reader._required_ifc_entity(entities, attrs[2], "IFCCARTESIANPOINT")
        if len(coords) != 3 or any(v is not None and reader._ifc_number(v) <= 0 for v in [attrs[3], *attrs[5:]]):
            raise ValueError(f"Invalid mapped origin or scale on #{step}")
    elif kind == "IFCBLOCK":
        if len(attrs) != 4 or any(reader._ifc_number(v) <= 0 for v in attrs[1:]):
            raise ValueError(f"Invalid block dimensions #{step}")
        reader._rigid_ifc_frame(entities, attrs[0], 1.)
    elif kind == "IFCTRIANGULATEDFACESET":
        _checked_triangulation(entities, step, attrs)
    elif kind in ("IFCCARTESIANPOINT", "IFCDIRECTION"):
        if len(attrs) != 1 or not isinstance(attrs[0], list) or not 1 <= len(attrs[0]) <= 3:
            raise ValueError(f"Invalid coordinate or direction #{step}")
        values = [reader._ifc_number(v) for v in attrs[0]]
        if kind == "IFCDIRECTION" and not any(values):
            raise ValueError(f"Invalid zero direction #{step}")
    elif kind in ("IFCADVANCEDBREP", "IFCREVOLVEDAREASOLID", "IFCINDEXEDPOLYCURVE",
                  "IFCCIRCLEPROFILEDEF", "IFCCIRCLEHOLLOWPROFILEDEF"):
        raise ValueError(f"Partial physical-solid certification does not cover {kind} #{step}")
    active.add(step)
    for dependency in _refs(attrs):
        _validate_body(entities, dependency, project_contexts, visited, active)
    active.remove(step)
    visited.add(step)
