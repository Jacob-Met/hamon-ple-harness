# Source-anchored drawing measurements (PLE)

The default task-native harness exposes `pdf-vector` and `pdf-measure`. This is
a deterministic, source-backed dimensional tool, **not** a drawing-scale guesser
or engineering signoff. It lets an agent inspect unfamiliar multi-sheet PDFs
without treating 1:50 captions, page DPI, or pixels as physical calibration.

## In a task container

1. Inspect one PDF sheet:

   `python3 /tmp/hamon-engineering-toolkit.py pdf-vector sheets.pdf --from-page 2 --to-page 2 --limit-paths 5000`

2. Choose two independently dimensioned, **nonparallel** lines. Confirm their
   source text labels truly refer to those endpoints. Extract their returned
   `id` and `svg_points`. Only exact transformed SVG vertices are eligible.
3. Write `anchors.json`:

   `{"schema":"hamon.pdf.measure.v1","page":2,"max_relative_residual":0.01,
   "references":[
     {"from":{"vector":"v12","vertex":0},"to":{"vector":"v12","vertex":1},
      "length":"2400*mm","source_label":"2400mm"},
     {"from":{"vector":"v21","vertex":0},"to":{"vector":"v21","vertex":1},
      "length":"1.5*m","source_label":"1.5m"}],
   "targets":[{"name":"opening-width",
               "from":{"vector":"v65","vertex":0},
               "to":{"vector":"v65","vertex":1},"unit":"mm"}]}`

   The IDs are **illustrative**. Replace IDs, labels and lengths with real
   evidence from the task's source sheet. Do not synthesize x/y coordinates.
4. Run:

   `python3 /tmp/hamon-engineering-toolkit.py pdf-measure sheets.pdf anchors.json`

The command returns a source PDF SHA-256, unit-safe target estimates, independent
anchor residuals, and explicit failures. The default maximum relative residual
is 1%; the caller can tighten but not bypass evidence gates.

## Grounding and limits

- Parallel-only references, missing independent labels, fabricated vector IDs,
  unsupported curved endpoints, incompatible units, and inconsistent scale
  **fail closed**.
- Dense drawings can select real vector IDs beyond the normal `pdf-vector`
  report limit. The source ID is a full-page primitive ID, not an array index
  in a cropped listing.
- Matched text proves only that the label occurs on that sheet; it does **not**
  prove which line it annotates. Independently check leaders, viewport,
  projection, plotted scale and local datum.
- `NTS` captions, paper page size, raster pixels and document DPI never
  establish metric scale. Valid independent reference dimensions can still
  ground a local conditional estimate on a sheet marked NTS.
- The result is a **conditional 2D on-sheet measurement**, not an IFC/3D
  clearance, authoritative BRep dimension, or engineering-code approval.

Implementation: `hamon_scale.py`, `hamon_drawing.py`,
`engineering_toolkit.py`, and task-container provisioning via
`engineering_intake.py`. Regression controls exercise real Poppler-native
vectors and negative dimension checks. These are local engineering tool checks,
not a hosted PLE leaderboard score.

## Affine/sheared drawing exports: do not trust two-axis scale

A global similarity scale can fit horizontal and vertical reference dimensions
**exactly** and still be wrong on a slanted edge. An unknown affine distortion
requires a 2-D symmetric positive-definite metric G, so physical length is the
square root of v-transpose-G-v for each SVG source displacement v. The richer
model requires **four or more** distinct independently source-labeled
directions, rather than interpolating three dimensions with three unknowns.

In the same source-anchored JSON, set "calibration_model":"affine_spd" and
supply four independent references with true source vertices and distinct
printed labels. Typical directions include horizontal, vertical, positive
diagonal, and negative diagonal. The CLI solves the tensor from all dimensions,
independently refits with EACH reference omitted, and refuses output if the
worst in-fit or held-out length residual exceeds the specified tolerance.
Near-dependent directions, a non-positive-definite tensor, and principal-axis
distortion exceeding 10x also fail closed. A full-rank fit with only three
references is *not* accepted, even when it interpolates perfectly.

The result supplies "affine_metric.metric_tensor_m2_per_svg_unit2",
"affine_metric.distortion_principal_scale_ratio",
"reference_max_heldout_relative_residual", conditional length estimates, and
source provenance. The "conditional_scale_m_per_svg_unit" is null because an
affine physical metric has no single scalar scale.

**Counterexample:** native source vectors (100,0) and (0,100) each label 1m,
so the similarity scale is 0.01m/SVG unit. With physical shear metric
G = [[0.0001,0.00004],[0.00004,0.0001]], a (80,40) target is
sqrt(1.056)m rather than the similarity estimate sqrt(0.8)m: a ~13% miss,
despite both original reference dimensions having zero residual. Source
dimensions along (100,100) and (100,-100) are sqrt(2.8)m and sqrt(1.2)m;
the four-direction held-out fit recovers G and the correct oblique length.

This is a **2-D distance metric**, not proof that the drawing was affinely
distorted or that any annotation belongs to a particular line. Drawing
semantics, projection, actual site conditions and code approval remain external.

## Automatic source-geometry dimension association

The older pdf-measure command expects manually linked printed callouts and
source endpoints. Existence of a label somewhere on a sheet does NOT prove
that the label belongs to the selected geometry.

The new pdf-dimension-links builds a bounded source-stroke/word-box association
graph from Poppler with no OCR, invented coordinates or page-DPI scale:

    python3 /tmp/hamon-engineering-toolkit.py pdf-dimension-links plan.pdf --page 2

A possible dimension line must have two independent short perpendicular
source strokes crossing its opposite original SVG endpoints and a near-axis
PDF word box containing a numeric value with explicit physical units. The
matching rejects incomplete inventory, mismatched PDF/SVG coordinate
frames, unsupported curved geometry, missing endpoint witnesses, candidate
ambiguity and different source labels claiming the same line. Original
svg_segment_vertex_pairs do not bridge disjoint M-subpaths in compound SVG
paths. A matching record includes original word IDs/bounds, exact SVG
vector and endpoint indices, tick witness geometry, and association margins;
claim_of_engineering_approval is always false.

To measure an unlabelled target with reference callouts automatically
recovered from the same source sheet:

    python3 /tmp/hamon-engineering-toolkit.py pdf-auto-measure plan.pdf targets.json

For example, targets.json (replace illustrative IDs with actual source vectors):

    {
      "schema": "hamon.pdf.auto-measure.v1",
      "page": 2,
      "calibration_model": "affine_spd",
      "max_relative_residual": 0.01,
      "targets": [
        {"name": "duct-offset",
         "from": {"vector": "v65", "vertex": 0},
         "to": {"vector": "v65", "vertex": 1},
         "unit": "mm"}
      ]
    }

The affine_spd setting requires at least four independently witnessed
directional dimensions and held-out leave-one-anchor-out metric consistency.
A changed or contradictory dimension label changes the physical estimate
or fails the fit: it cannot silently preserve stale measurements. A
similarity model may use two nonparallel references only when isotropy
is justified; it cannot detect arbitrary PDF shear.

This is NOT a universal drawing parser. Continuous dimension chains,
rotated text, multi-panel sheets, jogged leaders, large arrowheads,
unsupported curves, missing source ticks, raster/OCR plans, duplicate labels,
and elevation projection may be unknown or rejected. Geometric adjacency
alone cannot prove architectural author intent. Inspect flagged rejections
and use a professional drawing review when safety matters. Returned
physical lengths are conditional source-backed estimates, not proof of
site dimensions, physical conditions or engineering-code approval.

Source is hamon_callout.py, hamon_drawing.py and hamon_scale.py; the
test_drawing_callout_graph.py cold suite builds a genuine PDF and runs
both CLI commands through native Poppler, plus adversarial swapped
labels, missing ticks, decoys, ambiguous lines, coordinate-frame mismatches
and compound SVG disjoint-subpath tests. These local checks are not a
hosted Physical Engineering's Last Exam score.
