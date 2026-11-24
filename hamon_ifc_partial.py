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
    elif kind in ("IFCCARTESIANPOINTLIST2D", "IFCCARTESIANPOINTLIST3D"):
        dim = 2 if kind.endswith("2D") else 3
        if not attrs or not isinstance(attrs[0], list) or not attrs[0]:
            raise ValueError(f"Missing point-list coordinates on #{step}")
        for point in attrs[0]:
            if (not isinstance(point, list) or len(point) != dim
                    or any(not math.isfinite(reader._ifc_number(v)) for v in point)):
                raise ValueError(f"Invalid {dim}D point-list coordinates on #{step}")
    elif kind == "IFCPOLYGONALFACESET":
        if len(attrs) != 4 or (attrs[1] is not None and attrs[1] is not True):
            raise ValueError(f"Unqualified polygonal face-set closure on #{step}")
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


def _products(entities, roles, defined):
    rows = {}
    for step, (kind, attrs) in sorted(entities.items()):
        if kind not in defined:
            raise ValueError(f"Unrecognized IFC entity {kind} on #{step}")
        if kind not in roles:
            continue
        shape = entities.get(_ref(attrs[6]) if len(attrs) > 6 else None)
        reps = shape[1][2] if shape and len(shape[1]) > 2 and isinstance(shape[1][2], list) else []
        labels = []
        for ref in reps:
            fields = entities.get(_ref(ref), ("", []))[1]
            labels.append(fields[1] if len(fields) > 1 and isinstance(fields[1], str) else None)
        name = attrs[2] if len(attrs) > 2 and isinstance(attrs[2], str) else ""
        rows[step] = dict(step_id=step, guid=attrs[0] if attrs and isinstance(attrs[0], str) else None,
                          ifc_type=kind, role=roles[kind], representation_roles=labels,
                          discipline=reader._infer_discipline(kind, name), void_relation_ids=[])
    return rows


def _dependencies(entities, rows):
    voids, fills = [], []
    cutters, issues = defaultdict(list), {}
    for step, (kind, attrs) in sorted(entities.items()):
        if kind not in ("IFCRELVOIDSELEMENT", "IFCRELFILLSELEMENT", "IFCRELAGGREGATES",
                        "IFCRELNESTS", "IFCRELPROJECTSELEMENT"):
            continue
        parent = _ref(attrs[4]) if len(attrs) > 4 else None
        end = _ref(attrs[5]) if len(attrs) > 5 else None
        if kind == "IFCRELVOIDSELEMENT":
            if parent not in rows or rows[parent]["role"] != "physical":
                raise ValueError(f"Unassignable physical host on void relation #{step}")
            voids.append(dict(step_id=step, host_step_id=parent, cutter_step_id=end))
            rows[parent]["void_relation_ids"].append(step)
            cutters[parent].append(end)
            if len(attrs) != 6 or end not in rows or rows[end]["role"] != "opening":
                issues[parent] = f"Unresolved cutter #{end} on void relation #{step}"
        elif kind == "IFCRELFILLSELEMENT":
            if (len(attrs) != 6 or parent not in rows or end not in rows
                    or rows[parent]["role"] != "opening" or rows[end]["role"] != "physical"):
                raise ValueError(f"Unassignable filling relation #{step}")
            fills.append(dict(step_id=step, opening_step_id=parent, filling_step_id=end))
        elif kind == "IFCRELPROJECTSELEMENT":
            if parent not in rows:
                raise ValueError(f"Unassignable projected element on #{step}")
            issues[parent] = f"Unqualified projected feature relation #{step}"
        elif parent in rows and rows[parent]["role"] == "physical":
            issues[parent] = f"Unqualified physical assembly relation #{step}"
    owners = Counter(c for children in cutters.values() for c in children)
    for host, children in cutters.items():
        if any(owners[c] > 1 for c in children):
            issues[host] = "Opening belongs to multiple void relationships"
    return voids, fills, cutters, issues


_GEOMETRY_ERRORS = (ValueError, ArithmeticError, TypeError, IndexError, KeyError, RecursionError)


def _valid_mesh(mesh):
    v, f = mesh.vertices, mesh.faces
    if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or f.ndim != 2
            or f.shape[1] != 3 or not len(f) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError("Invalid source mesh")
    triangles = v[f]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if not np.isfinite(normals).all() or np.any(np.all(normals == 0, axis=1)):
        raise ValueError("Unrepresentable or degenerate world-space triangle")


def _inventory(text):
    schema, data = _source_sections(text)
    roles, defined = _schema_roles(schema)
    entities = _entities(data)
    scale, contexts = _model_context(entities)
    rows = _products(entities, roles, defined)
    voids, fills, cutters, issues = _dependencies(entities, rows)
    participants = set(cutters).union(*(set(children) for children in cutters.values()))
    gross, empty, visited = {}, set(), set()
    for step, row in rows.items():
        if row["role"] not in ("physical", "opening"):
            continue
        row.update(geometry_status="unresolved", volume_status="unresolved", volume_m3=None)
        kind, attrs = entities[step]
        try:
            if len(attrs) < 7:
                raise ValueError("Incomplete product attributes")
            shape = reader._required_ifc_entity(entities, attrs[6], "IFCPRODUCTDEFINITIONSHAPE")
            if len(shape) != 3 or not isinstance(shape[2], list):
                raise ValueError("Incomplete product definition shape")
            items = reader._opening_body_items(entities, step, mass_takeoff=True)
            row["body_item_ids"] = [_ref(ref) for ref in items]
            for ref in shape[2]:
                fields = entities[_ref(ref)][1]
                if str(fields[1]).upper() == "BODY":
                    _validate_body(entities, _ref(ref), contexts, visited, set())
            bodies, empties = reader._gross_ifc_solids(
                entities, scale, participants, [(step, kind, attrs, shape)], strict_geometry=True)
            if step in empties:
                empty.add(step)
                if row["role"] == "opening":
                    row.update(geometry_status="empty", volume_status="excluded")
                continue
            body, = bodies
            _valid_mesh(body)
            gross[step] = body
            if row["role"] == "opening":
                row.update(geometry_status="known", volume_status="excluded")
        except _GEOMETRY_ERRORS as exc:
            row["reason"] = str(exc)

    meshes = {}
    for step, row in rows.items():
        if row["role"] != "physical" or (step not in gross and step not in empty):
            continue
        try:
            if step in issues:
                raise ValueError(issues[step])
            bodies = [gross[step]] if step in gross else []
            if step in cutters:
                missing = [c for c in cutters[step] if c not in gross and c not in empty]
                if missing:
                    raise ValueError(f"Unresolved required cutters {missing}")
                bodies = reader._physical_occurrences(
                    [*bodies, *(gross[c] for c in cutters[step] if c in gross)],
                    entities, scale, {step: set(cutters[step])}, empty)
            if not bodies:
                row.update(geometry_status="empty", volume_status="known", volume_m3=0.)
                continue
            body, = bodies
            _valid_mesh(body)
            meshes[step] = replace(body, element_id=f"#{step}")
            row["geometry_status"] = "known"
            volume = body.volume_m3()
            if not math.isfinite(volume) or volume <= 0:
                raise ValueError("Unrepresentable material quantity")
            row.update(volume_status="known", volume_m3=volume)
        except _GEOMETRY_ERRORS as exc:
            row["reason"] = str(exc)
    return rows, meshes, voids, fills


def _known_pairs(meshes, clearance, cross_only):
    """Preserve broadphase in the normal case and isolate exceptional numeric pairs."""
    failed = []
    try:
        report = reader._clashes_for_solids(meshes, clearance, cross_only, include_quantity=False)
        pairs = report["all_pairs"]
    except _GEOMETRY_ERRORS:
        pairs = []
        for i, a in enumerate(meshes):
            for b in meshes[i + 1:]:
                if cross_only and a.discipline == b.discipline:
                    continue
                try:
                    pairs.append(vars(reader.evaluate_pair_clash(a, b, clearance)))
                except _GEOMETRY_ERRORS as exc:
                    failed.append(dict(element_a=a.element_id, element_b=b.element_id, reason=str(exc)))
    valid = []
    for pair in pairs:
        distance, bound = pair["min_distance_m"], pair.get("distance_lower_bound_m")
        if not ((distance is not None and math.isfinite(distance) and distance >= 0)
                or (distance is None and pair["status"] == "clear" and bound is not None
                    and math.isfinite(bound) and bound > clearance)):
            failed.append(dict(element_a=pair["element_a"], element_b=pair["element_b"],
                               reason="Unresolved pair distance or clearance bound"))
        else:
            valid.append(pair)
    return valid, failed


def partial_clash_report(text, required_clearance_m, cross_discipline_only):
    out = dict(tool="hamon_ifc_spatial_clash", ok=False, total_solid_volume_m3=None,
               scope="physical IfcElement occurrences excluding virtual elements and feature elements",
               hard_clash_count=None, clearance_violation_count=None, min_clearance_m=None)
    try:
        if not math.isfinite(required_clearance_m) or required_clearance_m < 0:
            raise ValueError("Expected finite nonnegative clearance")
        rows, meshes, voids, fills = _inventory(text)
    except (ValueError, ArithmeticError, TypeError, IndexError, KeyError, ImportError, RuntimeError) as exc:
        return {**out, "status": "unresolved_model", "reason": str(exc)}

    physical = [r for r in rows.values() if r["role"] == "physical"]
    quantities = [r for r in physical if r["volume_status"] == "known"]
    missing = [r["step_id"] for r in physical if r["geometry_status"] == "unresolved"]
    pairs, failed = _known_pairs(list(meshes.values()), required_clearance_m, cross_discipline_only)
    missing_pairs = empty_pairs = 0
    for i, a in enumerate(physical):
        for b in physical[i + 1:]:
            if cross_discipline_only and a["discipline"] == b["discipline"]:
                continue
            states = a["geometry_status"], b["geometry_status"]
            if "empty" in states:
                empty_pairs += 1
            elif "unresolved" in states:
                missing_pairs += 1
    try:
        volume = math.fsum(r["volume_m3"] for r in quantities)
    except OverflowError:
        volume = None
    complete = len(quantities) == len(physical) and not failed and volume is not None
    measured = [p["min_distance_m"] for p in pairs if p["min_distance_m"] is not None]
    minimum = min(measured, default=None)
    hard = [p for p in pairs if p["status"] == "hard_clash"]
    violations = [p for p in pairs if p["status"] == "clearance_violation"]
    return dict(out, ok=complete, status="complete" if complete else "partial",
                cross_discipline_only=cross_discipline_only, products=list(rows.values()),
                role_counts=dict(Counter(r["role"] for r in rows.values())),
                void_relationships=voids, filling_relationships=fills,
                physical_element_count=len(physical), known_mesh_count=len(meshes),
                unresolved_geometry_step_ids=missing, known_volume_step_ids=[r["step_id"] for r in quantities],
                known_occurrence_volume_sum_m3=volume, total_solid_volume_m3=volume if complete else None,
                requested_pair_count=len(pairs) + len(failed) + empty_pairs + missing_pairs,
                evaluated_pair_count=len(pairs), certified_empty_pair_count=empty_pairs,
                missing_geometry_pair_count=missing_pairs, unresolved_pair_count=len(failed),
                unresolved_pairs=failed, measured_pair_count=len(measured), bound_only_pair_count=len(pairs)-len(measured),
                known_hard_clash_count=len(hard), known_clearance_violation_count=len(violations),
                known_min_clearance_m=minimum, hard_clash_count=len(hard) if complete else None,
                clearance_violation_count=len(violations) if complete else None,
                min_clearance_m=minimum if minimum == 0 or (not failed and not missing_pairs) else None,
                clashes=hard+violations, all_pairs=pairs)
