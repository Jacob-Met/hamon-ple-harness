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
`output_geometry_verified` is true. The acceptance check integrates first moments about the base center. It uses
the displayed force to center the displayed moments and the requested force
to center the requested moments. This keeps the verdict invariant when the
same representable local geometry and pressures move to a different global
origin. The reported `moment_residual_Nm` retains its global-origin convention.
Rounded coordinates that collapse an actual contact patch cannot pass.

For full rectangular contact, proposed corner and polygon pressure fields
also have a separate tolerance for each centered first moment. The allowance
sums the displayed reference field's absolute component residual, one part
in a billion of the requested component magnitude, and a pressure-rounding
allowance. The last term is the displayed area times the largest distance
along that axis from the base center times the largest binary64 pressure ULP
among the vertices. `full_contact_moment_resolution` reports, separately
for corner and polygon fields, whether each nonzero requested component
exceeds that allowance. A zero demand is recorded as null. A flat or
reversed component still fails the direction check for a nonzero demand.
This is a numerical acceptance bound for the reported field.

For partial contact, each proposed pressure field is integrated only over
the loaded polygon. A corner-only claim supplies pressure at loaded base
corners. The reference release intersections retain zero pressure. A polygon
claim supplies its own coordinates and vertex pressures. Both use the same
per-component reference residual and pressure allowance described above.
For the polygon, moving a displayed coordinate by one binary64 ULP also
perturbs triangle areas and lever arms. The bound sums those changes over
the contact triangles, including products of coordinate errors and pressure
rounding. The corner-only calculation uses fixed reference geometry.
`clipped_contact_moment_resolution` reports resolution for each axis and
claim type. The resolution flags describe numeric scale; the certificate
also requires the existing force, area, component-direction and field checks.

On a 2 m square, the field `1.5 * max(0, x + e*y)` with `e=2**-42`
has weak first moment `e`. Replacing the tilt by `1024*e` retains the
contact regime and the sign but gives a weak moment of `1024*e`.
Changing only the two loaded vertex pressures on the original footprint
gives `512.5*e` for the submitted triangular interpolant. Both exceed the
component allowance.

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
