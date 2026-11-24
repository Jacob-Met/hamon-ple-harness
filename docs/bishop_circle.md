# Simplified Bishop equilibrium on a specified slip circle

Use `hamon_conservation_cad_verifier` with `mode=civil_geotech` and `spec.calculation=bishop_slip_circle`. The `dual-verify` CLI accepts the same specification under `kind=bishop_slip_circle`. The example below specifies one 2 m wide slice carrying 1000 N per metre of out-of-plane width.

```json
{
  "calculation": "bishop_slip_circle",
  "circle": {"center_x_m": 0, "center_y_m": 13, "radius_m": 10},
  "slide_direction": "right",
  "slices": [
    {"left_x_m": -6, "right_x_m": -4, "weight_N_per_m": 1000,
     "cohesion_Pa": 120, "friction_angle_deg": 30, "pore_pressure_Pa": 95}
  ]
}
```

Coordinates increase to the right and upward. Slices cover contiguous horizontal intervals from left to right. Each supplied total weight acts vertically through its horizontal midpoint. The solver uses the lower circle, its tangent at that midpoint and the corresponding planar base length. The base angle is positive downhill in the requested sliding direction. Basal pore pressure acts normally over that length. The model assumes horizontal interslice forces, drained Mohr–Coulomb strength and gravity loading. The reported factor applies to the supplied discretization and trial circle. A critical-surface search, Janbu analysis, buoyancy from water above the slope, seismic loading, and design or site acceptance require separate analyses.

For horizontal slice width `b`, base angle `a`, total weight `W`, effective strengths `c, phi` and pore pressure `u`, the solver balances the signed sum of `W sin(a)` against the sum of `[c b + (W - u b) tan(phi)] / [F cos(a) + sin(a) tan(phi)]`. It brackets the positive root above every denominator pole. Negative available shear strength, tensile effective basal contact, unresolved local denominators or nonpositive driving moment withhold verification. Opposing toe contributions retain their sign.

The result includes individual effective normal forces, mobilized resisting shears, signed driving moment, resisting moment and a residual. Vertical force equilibrium on every slice and global moment equilibrium must agree to better than one part in 100 million. An optional `proposed_factor_of_safety` must separately match within one part per million. In the one-slice case the independent normal projection gives `N' = W cos(a) - u b / cos(a)` and the mobilized shear equals `W sin(a)`.

Method reference: U.S. Army Corps of Engineers, *Slope Stability*, EM 1110-2-1902, Appendix C, section C-3. [Official manual](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1902.pdf)

## Numerical range and independent checks

The geometry adapter uses one shared numerical core. It scales the force terms
by an exactly representable binary power before the monotone root solve,
retains the admissible side of every pole, and
reconstructs both vertical force balance and the Mohr–Coulomb law on each slice.
Nonzero loads that disappear in a width product and any loss of input-force
information during normalization are refused. Distinct adjacent loads cannot
round to the same normalized force. Effective normal forces too close to zero
to establish compressive contact at the working precision are also refused.
Moments combine their binary exponents before conversion back to dimensional
units, so an intermediate aggregate force cannot overflow when the final moment
is representable. An unresolved near-pole denominator still withholds verification.

The result distinguishes equilibrium, compressive contact, the supplied circle
geometry, horizontal equilibrium, and critical-surface search. Geometry
verification covers the supplied bounds and midpoint tangents. The caller still
supplies weights and material/pore-pressure data. Horizontal equilibrium and a
search for the globally critical surface are not performed. A proposed factor
has to match separately; disagreement does not erase the computed equilibrium.

Tests compare twelve wet/dry material tables against the unchanged independent
Bishop implementation from [XSLOPE at 28f16877](https://github.com/njones61/xslope/blob/28f16877a9c204ed421d3e2d6e353609ec8d0c8b/test/indep_bishop_check.py).
The upstream source, Apache 2.0 license and NOTICE are included under
`tests/oracles`. The adapter only supplies its column interface.

A physical geometry check derives weights by quadrature from
[the same pinned simple1 workbook](https://github.com/njones61/xslope/blob/28f16877a9c204ed421d3e2d6e353609ec8d0c8b/docs/inputs/slope/xslope_simple1.xlsx):
a 20-foot-high 1:1 slope, circle center (10,40) feet, radius 40 feet,
and unit weight 125 lb/ft³. The accepted frictional control explicitly changes
the material to c=0 and phi=35°. At 20 and 80 slices, the computed factors
2.116653407297558 and 2.116140785963024 agree with the independent function
to eleven decimal places on the same supplied tables.

Retaining the workbook's original c=500 lb/ft² and phi=0 produces a positive
scalar equilibrium factor, but the uppermost tangent slices require tension.
The verifier refuses that contact state. This test uses integrated weights
and midpoint tangent bases; it does not claim to reproduce XSLOPE's different
arc/chord discretization or its rounded README factor.

An extreme-range regression uses four individually representable loads whose
sum overflows binary64, while the requested moment remains finite. The former
unscaled solver refused with intermediate overflow. The shared core returns
F=0.7187467766441823 with residual below 5e-16 and the same factor as its
ordinary-scale counterpart. This is numerical range coverage, not a field
benchmark or a hosted PLE score. A separate adversarial case verifies that a
tiny tensile-contact load cannot vanish beside a much larger slice load.

The analytical reference is USACE EM 1110-2-1902, Appendix F, equations F-1,
F-4 and F-6, specialized to gravity and basal pore pressure.
