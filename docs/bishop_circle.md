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
