# Joint pressure on a rigid rectangular base

`rigid_rectangle_contact` solves one compression-only affine pressure field
on an axis-aligned rectangle. It complements the existing strip calculation.
It does not infer bearing capacity, settlement, friction or a soil law.

## Inputs and coordinates

Use `hamon_conservation_cad_verifier` with `mode: civil_geotech` and this spec:

```json
{
  "calculation": "rigid_rectangle_contact",
  "base_width_m": 2,
  "base_length_m": 2,
  "vertical_resultant_N": 4,
  "moment_x_Nm": 2,
  "moment_y_Nm": 2
}
```

Numeric inputs use SI. Dimensioned strings such as `"2000*mm"` are accepted;
force per length cannot substitute for total force. Both moments are required.
They mean `Mx = integral(x*p*dA)` and `My = integral(y*p*dA)`, signed pressure
first moments about the global origin, not right-hand torque components.
Optional `base_center_m: [cx, cy]` translates the base; the default is `[0,0]`.
`eccentricity_m` is relative to that center.

The example is exactly `p = 3*max(0, x+y)`. Its triangular contact has area 2,
peak pressure 6 at `(1,1)`, and zero pressure on `x+y=0`. Force and both
first moments are exactly 4, 2 and 2.

Two independent strips can reproduce those resultants with a different
field: their product has contact area 9/4 and peak 64/9. Matching the three
resultants does not qualify the joint footprint or peak. This verifier
rejects those proposed values and the product's rectangular footprint.

## Inversion

Reflect the load toward the positive corner and normalize the rectangle to
`0 <= d,e <= 2`, measured from that corner. The field becomes
`p_peak * max(0, 1-rx*d-ry*e)`. The prescribed centroids determine the two
slopes, then force determines the peak.

There are four regimes: full rectangle, corner triangle, edge trapezoid,
and clipped pentagon. The first three use closed forms. The pentagon uses
damped Newton iteration with a Jacobian from polygon area moments.
Nonconvergence returns an unverified result. A centroid on or beyond an
edge cannot be balanced by finite nonnegative area pressure and is refused.

## Independent checks and numeric limits

The solver uses degree-two triangle integration. It separately integrates
the returned polygon and vertex pressures with exact rational arithmetic
on their binary64 values. Force, both moments, area and the returned affine
field must agree within 1e-9 relative physical scales before
`output_geometry_verified` is true. Moment residuals scale with force times
base dimension, not distance to the global origin. Rounded coordinates
that collapse an actual contact patch cannot pass.

`ship/verify_rectangular_contact.py` supplies an independent forward oracle:
rational vertical cross-sections integrated by Simpson 3/8, exact for the
piecewise cubic integrands. It does not import the solver's clipping or
triangle formulas. Its 675 fields cover all regimes, reflections and three
base shapes, including a large aspect ratio. They are authored mathematical
controls, not judged PLE examples.

Optional `proposed_` fields check `contact_area_m2`, `q_min_Pa`, `q_max_Pa`,
`eccentricity_m`, `corner_pressure_Pa`, `contact_regime`, `contact_polygon_m`
and `contact_vertex_pressure_Pa`. Vertex pressures require their polygon.
Cyclic order and reversed winding are accepted; the proposed field is
integrated again against the loads. Coordinate tolerances scale with contact
extent, so a thin patch cannot be replaced by a collapsed or shifted polygon
near a large absolute coordinate.

The pressure plane uses a reference point, pressure at that point, and two
gradients, avoiding a large global intercept. `corner_pressure_Pa` uses the
base-corner order `(-,-), (+,-), (+,+), (-,+)` about the base center. The
contact polygon is counterclockwise and starts at the loaded corner.

Inspect `conservation_verified`, which combines numeric qualification and
every supplied claim. `ok` alone is insufficient. Zero load with zero moments
has no determined footprint: area, polygon and eccentricity are null.
Overflow, underflow, unresolved geometry and unsupported actions are not
certified. Arbitrarily non-affine pressure laws are outside this model.

## Reproduction

```sh
python ship/verify_rectangular_contact.py --out /path/to/controls.json
python -m unittest tests.test_rigid_rectangle_contact -v
```

Tests include translations, thin patches, boundary neighbors, units, invalid
claims, ordinary and domain routes, source-only execution away from the
checkout, and the actual registered Pi callback when its runtime is installed.
These checks make no scored submission.
