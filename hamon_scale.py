"""Source-anchored, fail-closed metric measurements on engineering PDF vectors.

This module never infers a drawing scale from a page-size, scale caption or
pixel DPI. Explicit, independent dimension callouts and real SVG vertices must
agree before a dimension estimate is returned. A source text match is evidence
that the callout exists, NOT proof it refers to the chosen dimension line.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re

import hamon_drawing


def _finite_point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected a two-component SVG coordinate")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) for x in value):
        raise ValueError("SVG coordinates must be finite numbers")
    return (float(value[0]), float(value[1]))


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _resolve_point(endpoint, vectors):
    if not isinstance(endpoint, dict) or set(endpoint) != {"vector", "vertex"}:
        raise ValueError("endpoint must select {vector,vertex} from extracted SVG")
    key, index = endpoint["vector"], endpoint["vertex"]
    if not isinstance(key, str) or not re.fullmatch(r"v[1-9][0-9]*", key):
        raise ValueError("invalid source vector ID")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("source vertex must be a nonnegative integer")
    vector = vectors.get(key)
    if vector is None:
        raise ValueError(f"source vector {key} not reported on this page")
    points = vector.get("svg_points")
    if not isinstance(points, list):
        raise ValueError(f"source vector {key} has no exact, bounded vertex list")
    if index >= len(points):
        raise ValueError(f"source vector {key} has no vertex {index}")
    return _finite_point(points[index])


def _segment(row, vectors):
    if not isinstance(row, dict):
        raise ValueError("segment must be an object")
    return (_resolve_point(row.get("from"), vectors),
            _resolve_point(row.get("to"), vectors))


def _source_text(words, label):
    if not isinstance(label, str) or not 2 <= len(label) <= 100:
        raise ValueError("source_label must be 2..100 characters")
    raw = " ".join(w.get("text", "") for w in words)
    norm = lambda x: re.sub(r"\s+", "", x).casefold()
    if norm(label) not in norm(raw):
        raise ValueError(f"dimension source_label {label!r} not found in PDF page words")
    return label


def calibrate_inventory(page, spec, quantity, metre):
    """Calibrate one extracted PDF page, with a dimensional-unit evaluator.

    spec:
      references: >=2 independent, nonparallel SVG vertex-to-vertex segments,
          each with "from", "to", "length" (e.g. "2400*mm"), and
          "source_label" (text visibly extracted from this same PDF page).
      targets: vertex-to-vertex segments with a "name" and optional "to" unit.
      max_relative_residual: 0..0.1; default 0.01.

    Each endpoint uses {"vector":"v5","vertex":0} and selects an extracted,
    transformed, exact vector vertex; model-invented coordinates are rejected.
    Dimensional evidence is user-selected, and must still be visually verified.
    """
    if not isinstance(spec, dict):
        raise ValueError("measurement specification must be an object")
    if not isinstance(page, dict) or not isinstance(page.get("words"), list):
        raise ValueError("PDF page word/vector inventory required")
    references, targets = spec.get("references"), spec.get("targets")
    if not isinstance(references, list) or not 2 <= len(references) <= 64:
        raise ValueError("2..64 independently grounded references required")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 256:
        raise ValueError("1..256 target segments required")
    tolerance = spec.get("max_relative_residual", 0.01)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) \
            or not math.isfinite(tolerance) or not 0 <= tolerance <= 0.1:
        raise ValueError("max_relative_residual must be finite within [0,0.1]")
    reported = page.get("reported_vectors", [])
    if not isinstance(reported, list):
        raise ValueError("reported vector list required")
    vectors = {v["id"]: v for v in reported if isinstance(v, dict)
               and isinstance(v.get("id"), str)}
    if len(vectors) != len(reported):
        raise ValueError("source vector identifiers must be unique and complete")
    if page.get("vector_limit_hit"):
        raise ValueError("source vector listing truncated; rescan a bounded region")
    if page.get("unlocalized_vectors"):
        # Other curves can coexist, but calibration uses only exact SVG vertices.
        pass

    metre_value = metre.val
    direction_vectors = []
    observations = []
    source_labels = set()
    for i, reference in enumerate(references):
        a, b = _segment(reference, vectors)
        width = _distance(a, b)
        if width <= 1e-9:
            raise ValueError(f"reference {i}: coincident SVG endpoints")
        expr = reference.get("length")
        if not isinstance(expr, str) or not 1 <= len(expr) <= 200:
            raise ValueError(f"reference {i}: length expression required")
        physical = quantity(expr)
        if physical.dim != metre.dim or not physical.val.is_finite() \
                or physical.val <= 0:
            raise ValueError(f"reference {i}: positive physical length required")
        label = _source_text(page["words"], reference.get("source_label"))
        if label in source_labels:
            raise ValueError("independent source_label required per reference")
        source_labels.add(label)
        length_m = float(physical.val / metre_value)
        direction_vectors.append(((b[0] - a[0]) / width,
                                  (b[1] - a[1]) / width))
        observations.append({"source_label": label, "svg_units": width,
                             "physical_m": length_m, "source_vector_ids":
                             [reference["from"]["vector"], reference["to"]["vector"]]})

    # Two independent and materially nonparallel dimension baselines are the
    # minimum check against misleading anisotropic PDF exports. Parallel
    # references cannot prove a uniform scale in the other direction.
    largest_cross = max(
        abs(a[0] * b[1] - a[1] * b[0])
        for i, a in enumerate(direction_vectors)
        for b in direction_vectors[i+1:]
    )
    if largest_cross < 0.25:
        raise ValueError("two nonparallel reference dimensions required")
    # Origin-constrained least squares: physical_m ~= metres_per_svg * svg_len.
    num = sum(r["svg_units"] * r["physical_m"] for r in observations)
    den = sum(r["svg_units"] ** 2 for r in observations)
    factor = num / den
    residuals = [abs(factor * r["svg_units"] - r["physical_m"]) / r["physical_m"]
                 for r in observations]
    if not math.isfinite(factor) or factor <= 0 \
            or max(residuals) > tolerance + 1e-12:
        raise ValueError("inconsistent dimension anchors; scale not qualified")

    result = []
    names = set()
    for i, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValueError(f"target {i}: object required")
        name = target.get("name")
        if not isinstance(name, str) or not 1 <= len(name) <= 90 or name in names:
            raise ValueError(f"target {i}: unique name 1..90 characters required")
        names.add(name)
        a, b = _segment(target, vectors)
        span = _distance(a, b)
        if span <= 1e-9:
            raise ValueError(f"target {name}: coincident endpoints")
        to = target.get("unit", "m")
        if not isinstance(to, str) or not 1 <= len(to) <= 40:
            raise ValueError(f"target {name}: target unit expression invalid")
        unit = quantity(to)
        if unit.dim != metre.dim or unit.val <= 0:
            raise ValueError(f"target {name}: conversion unit must have length dimension")
        distance_m = factor * span
        result.append({"name": name, "svg_length": span,
                       "estimate_m": distance_m,
                       "estimate": distance_m * float(metre_value / unit.val),
                       "unit": to,
                       "from": target["from"], "to_vertex": target["to"],
                       "status": "conditional_source_anchored_length"})

    return {"kind": "pdf-source-anchored-measurement",
            "page": page.get("page"), "calibration_method": "origin_least_squares",
            "physical_scale_verified": False,
            "conditional_scale_m_per_svg_unit": factor,
            "reference_count": len(observations),
            "reference_max_relative_residual": max(residuals),
            "reference_pair_max_sine": largest_cross,
            "max_allowed_relative_residual": tolerance,
            "references": [{**r, "relative_residual": e}
                           for r, e in zip(observations, residuals)],
            "targets": result,
            "warning": "Conditional measurement ONLY: text presence and vector vertices do not prove a callout labels those endpoints. Independently check leader lines, projection, viewport, plotted scale, source geometry, units and engineering code; never use as autonomous approval."}


def measure_pdf(pdf_file, spec_file, quantity, metre):
    source = pathlib.Path(spec_file)
    if source.stat().st_size > 500_000:
        raise ValueError("measurement spec exceeds 500 KB")
    spec = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema") != "hamon.pdf.measure.v1":
        raise ValueError("measurement schema must be hamon.pdf.measure.v1")
    page_number = spec.get("page")
    if isinstance(page_number, bool) or not isinstance(page_number, int) \
            or not 1 <= page_number <= 10000:
        raise ValueError("positive PDF page required")
    inventory = hamon_drawing.scan_pdf(
        pdf_file, first=page_number, last=page_number,
        limit_paths=5000, limit_words=5000)
    page = inventory["pages"][0]
    if not page.get("word_boxes_reported"):
        raise ValueError("no extracted source labels; cannot qualify dimensions")
    out = calibrate_inventory(page, spec, quantity, metre)
    out["pdf_sha256"] = hashlib.sha256(pathlib.Path(pdf_file).read_bytes()).hexdigest()
    out["spec_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    out["pdf_vector_primitives"] = page.get("vector_primitives")
    out["pdf_words_scanned"] = page.get("word_boxes_reported")
    return out
