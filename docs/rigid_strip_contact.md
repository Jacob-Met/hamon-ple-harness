# Compression-only rigid strip contact

Use `hamon_conservation_cad_verifier` with `mode: civil_geotech` and
`spec.calculation: rigid_strip_contact`. The automatic router also recognizes
that calculation. The provisioned CLI supports
`dual-verify` with `mode: domain`, `kind: rigid_strip_contact`, and the same
`spec`.

## Model and signs

A rigid horizontal strip has width B and coordinate x from -B/2 to +B/2.
The normal resultant N is positive downward, per unit out-of-plane length.
The signed load moment M is defined so that the balancing pressure satisfies

    integral p(x) dx = N
    integral x p(x) dx = M

A positive M places the compression resultant toward positive x. A balanced
pair of horizontal forces at different elevations contributes a signed couple
even when its net thrust is zero.

This model requires **p(x) = max(0, a + b*x)** over the whole base. It is the
standard linear pressure, compression-only contact idealization. An arbitrary
affine patch with only the two equilibrium constraints is not enough: for
N=100000 N/m and M=144000 N m/m, uniform pressure 1000000 Pa over
[1.39, 1.49] m also satisfies those two integrals but violates this contact
compatibility. The compatibility condition selects the edge-attached triangle.

The solver supplies contact equilibrium, not bearing resistance, sliding
resistance, soil settlement, or structural design approval. Existing
`bearing_capacity` and its `meyerhof_effective_area` option remain separate:
an equivalent uniform effective-area pressure is not the contact diagram here.

## Inputs and outputs

Supply the strip width and either the two aggregate loads below or an
`applied_loads` list.

| Input | Numeric SI unit | Meaning |
|---|---|---|
| base_width_m | m | Positive strip width |
| vertical_resultant_N_m | N/m | Downward normal load per unit length |
| moment_Nm_m | N m/m | Signed load moment per unit length about the center |

In `applied_loads`, a force has `horizontal_N_m` (right positive) and its
`height_m` (up positive), `vertical_N_m` (down positive) and its `x_m`
(right positive), or both pairs for a force with two components. A free
couple has the single field `couple_Nm_m` (clockwise positive). The optional
`base_center` object supplies both `x_m` and `height_m` of the base center
in the same frame as the application points. Its default is (0, 0).
For each action the solver sums H, V, and
`H*(height-center_height) + V*(x-center_x) + couple`. The reported contact
coordinates always refer to the base center. Action input also reports
`horizontal_resultant_N_m`. A nonzero value prevents an equilibrium
certificate because the model supplies only normal pressure.

Numbers use those SI units. Strings must carry compatible dimensions, such as
`"3000*mm"`, `"100*kN/m"`, and `"144*kN*m/m"`. Booleans, nonfinite
values, wrong dimensions, unsupported inputs and alternate pressure models
are rejected. The optional `pressure_model` must be `positive_part_affine`.
The equilibrium calculation uses the exact rational values of the converted
binary64 SI inputs and rounds its reported numbers once.

For N>0, let e=M/N:

- |e| <= B/6: full contact, including zero pressure at one endpoint at equality.
  The edge pressures are N/B - 6M/B^2 and N/B + 6M/B^2.
- B/6 < |e| < B/2: partial contact, with width 3(B/2-|e|), zero at the inner
  edge and maximum pressure 2N/width at the loaded outer edge.
- |e| >= B/2: no finite pressure equilibrium. The exact edge limit would
  require a concentrated point reaction, which this pressure model excludes.
- N=M=0: zero pressure, with no determined contact interval, width, or
  eccentricity. Those three fields are null.
- Negative N, or N=0 with nonzero M, has no nonnegative pressure solution.

`contact_interval_m` is ordered from negative to positive x.
`contact_edge_pressure_Pa` gives pressure at its two endpoints, linearly
interpolated within that interval and zero outside.
`q_left_Pa` and `q_right_Pa` are pressures at the base edges.
`q_min_Pa` and `q_max_Pa` apply to the whole base.
The analytic segment integrals are reported separately; displayed floating
coordinates and pressures are rounded. Inputs whose required results cannot
be represented finitely are refused instead of receiving an equilibrium
certificate. The reported rounded segment is separately integrated;
output_geometry_verified is false if its force error exceeds 1e-9 * N or
its moment error exceeds 1e-9 * N * B (N and B denote the supplied force and
width). This also prevents conservation_verified from certifying a contact
interval whose narrow width is lost in rounded coordinates.

To check a supplied answer, add `proposed_contact_width_m`,
`proposed_eccentricity_m`, `proposed_q_min_Pa`, `proposed_q_max_Pa`,
`proposed_q_left_Pa`, `proposed_q_right_Pa`,
`proposed_contact_interval_m`, or `proposed_contact_regime`.
Numeric claims allow relative error 1e-9 plus eight output ulps.
An undetermined field matches only null. Inspect `conservation_verified`
and `claim_checks`; a completed calculation can reject an answer.

## Example

```json
{
  "mode": "civil_geotech",
  "spec": {
    "calculation": "rigid_strip_contact",
    "base_width_m": 3,
    "vertical_resultant_N_m": 100000,
    "moment_Nm_m": 144000
  }
}
```

This gives partial contact over [1.32, 1.50] m, contact width 0.18 m, and
maximum pressure 1111111.111111111 Pa. Reversing M reflects the interval to
[-1.50, -1.32] m with the same width and maximum pressure. The ordinary
full-base extrapolation would instead demand -62666.6667 Pa at one edge.

These individual actions give the same pressure diagram:

```json
{
  "calculation": "rigid_strip_contact",
  "base_width_m": 3,
  "applied_loads": [
    {"vertical_N_m": 100000, "x_m": 0},
    {"horizontal_N_m": 48000, "height_m": 3},
    {"horizontal_N_m": -48000, "height_m": 0}
  ]
}
```

## Basis

The relationship between resultant location, loss of contact and triangular
base pressure is illustrated in USACE
[EM 1110-2-2200, Figure 4-2](https://www.publications.usace.army.mil/portals/76/publications/engineermanuals/em_1110-2-2200.pdf).
USACE [EM 1110-2-2100, section 2-4](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2100.pdf)
states the no-tension interface assumption when the resultant leaves the
middle third. This implementation derives its contact regime and pressure
from the stated compatibility and two integrals. It does not implement
the manuals' load combinations or acceptance criteria.

A proposed contact interval must have positive ordered span, stay inside the base, and retain the loaded base edge (both edges in full contact). Its span is checked on the contact-width scale, and the proposed coordinates are integrated with the supplied endpoint or peak pressures when present. A coordinate tolerance cannot certify collapsed, reversed, shifted-off-edge or materially shortened contact. Correct representable near-edge intervals remain eligible.

## Why two strip checks do not verify a rectangular contact patch

The strip inputs are normal force and one bending moment **per unit length**. For a rectangular base a positive-affine area pressure has the form `max(0, a + b*x + c*y)`. Its loaded region can have an oblique edge. Its total load and both first moments have units of force and force times length.

On the square `[-1,1] × [-1,1]`, consider the unit-scaled field `p = max(0,x+y)`. Integrating in y gives the line load `(1+x)^2/2`. The total force is 4/3, and symmetry gives both first moments as 2/3. The contact footprint is a triangle of area 2 and the peak pressure is 2.

Two separately equilibrated strip diagrams can give a different area field with exactly those same resultants. Each triangular marginal with centroid 1/2 extends from −1/2 to 1. Their normalized product is `(256/243)*max(0,x+1/2)*max(0,y+1/2)`. Its contact area is 9/4 and its peak pressure is 64/27, 18.52% above the positive-affine solution. At the base center it predicts 64/243 where the prescribed area law gives zero. Reflections of either coordinate preserve the discrepancy for all four moment-sign pairs.

An area-pressure verifier must check the joint contact footprint and integrate the specified area law against 1, x and y. A successful strip certificate establishes only the corresponding one-dimensional equilibrium. This example is a mathematical model qualification.
