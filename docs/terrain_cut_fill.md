# Cut and fill from paired terrain layers

Use the ordinary task-side CLI with an existing and a proposed terrain layer:

```bash
python3 /tmp/hamon-engineering-toolkit.py dxf-takeoff site.dxf \
  --existing-layer EXISTING --proposed-layer PROPOSED
```

The two flags must occur together and identify distinct layers. Names are matched without case. The unflagged command retains its existing surface/length takeoff behavior. The terrain operation reads the drawing without changing it and returns its SHA-256 with the result.

## Supported source geometry

Each selected layer must contain explicit model-space triangular `3DFACE` entities and declare a supported `$INSUNITS` value. All four XYZ corners must be present; the fourth must repeat the third. The [Autodesk DXF definition](https://help.autodesk.com/cloudhelp/2020/ENU/AutoCAD-DXF/files/GUID-747865D5-51F0-45F2-BEFE-9572DBC5B151.htm) specifies world-coordinate corners and this triangular representation. A consistent paper-space marker excludes an entity. The operation does not transform viewports, expand block instances or tessellate curved surfaces into terrain.

The selected triangles must form single-valued piecewise-linear height fields in XY. Vertical or degenerate triangles, quadrilaterals, ambiguous layouts, unsupported selected entities, positive-area overlaps within a layer and conflicting heights at shared boundaries prevent qualification. Triangles may use different diagonals in the two surfaces. Gaps are allowed only when both declared footprints agree; areas outside the supplied surfaces are not inferred as part of a site.

## Quantity definition

For each overlap polygon of an existing and a proposed triangle, the algorithm computes the affine height difference `proposed - existing`. It splits the polygon where that difference is zero. Positive differences contribute fill; negative differences contribute cut. Each affine function is integrated by triangle area times mean vertex height. Cut and fill are separate nonnegative quantities even when their net is zero.

Coordinate decimals, polygon intersections, sign splits, coverage checks and integration use exact rational arithmetic. Unit conversion includes the exact U.S. survey-foot ratio `1200/3937` metres. JSON area and volume numbers are rounded to finite binary64 values; a quantity outside that representation causes an unqualified result.

| Field | Meaning |
| --- | --- |
| `qualified` | Both declared footprints are valid and fully covered by one another, with quantities representable within the stated bounds. |
| `cut_m3`, `fill_m3` | Geometric volumes removed and added over the declared footprints. |
| `net_fill_m3` | Fill minus cut. |
| `existing_area_m2`, `proposed_area_m2` | XY plan areas of the respective selected surfaces. |
| `common_area_m2` | Their common XY plan area. |
| `uncovered_existing_area_m2`, `uncovered_proposed_area_m2` | Each surface's area outside the other footprint. |
| `source_sha256`, `declared_insunits`, `scale_to_m` | Input byte identity and declared length conversion. |
| `reasons` | Why a readable drawing did not establish qualified quantities. |

Partial coverage preserves the area diagnostics and returns all three volume fields as null. The CLI exits 3 for an unqualified result and 2 for a request or parse exception; qualified terrain results exit 0. Treat the declared drawing geometry as the calculation input. Geological assumptions, survey accuracy, a larger unmodeled site boundary, material swell/shrinkage, compaction and haul conversions require separate evidence and calculations.

## Bounded work

The implementation accepts at most 8,000,000 bytes and 2,048 selected faces per layer. It bounds candidate pairs whose X intervals overlap at 200,000 and polygon intersections at 32,000. Decimal coordinates have text/digit/exponent limits, and exact accumulators have an 8,192-bit limit. Exceeding a bound withholds quantities; it never samples a subset and labels it a complete site.

## Verification scope

The committed synthetic course includes crossing planes, identical corner heights with opposite triangulations, rotations/translations, millimetre conversion, winding changes, partial coverage and invalid selected surfaces. Separate tests exercise the provisioned CLI and the registered Pi bash tool through Relay and Session. These checks cover source and integration behavior. They do not constitute a PLE development-set run, repair inherited sample artifacts or establish a competition score. The official evaluation must generate fresh artifacts from the exact harness through its authorized kit/model route.
