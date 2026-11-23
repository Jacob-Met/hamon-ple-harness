# Transient consolidation across unlike soil layers

`layered_consolidation_1d` computes settlement and excess pore pressure for one
to eight saturated, horizontal layers with constant positive `mv_per_Pa` and
`cv_m2_s`. Supply layers in top-to-bottom order and a spatially uniform,
nonnegative surcharge history. Repeated time ordinates describe a jump;
distinct ordinates describe a linear ramp. `single` drains the top and seals
the base; `double` drains both exterior faces. Internal interfaces have
perfect hydraulic contact and are not extra drains.

```json
{
  "mode": "civil_geotech",
  "spec": {
    "calculation": "layered_consolidation_1d",
    "consolidation_layers": [
      {"thickness_m": 1, "cv_m2_s": 0.000001, "mv_per_Pa": 0.0000001},
      {"thickness_m": 1, "cv_m2_s": 0.00000001, "mv_per_Pa": 0.000001}
    ],
    "drainage": "single",
    "time_s": 10000000,
    "load_history": [{"time_s": 0, "surcharge_Pa": 10000}]
  }
}
```

## Physical model and checks

Within each layer, `mv * du/dt = d/dz(cv * mv * du/dz) + mv * dq/dt`.
Here `u` is excess pore pressure and `q` is surcharge. Both pressure and
`cv * mv * du/dz` are continuous at interfaces. Settlement integrates
`mv * (q - u)` over depth. Since `cv = k / (gamma_water * mv)`, weighting
interface fluxes by diffusivity alone is incorrect. See the classical
[layered-model weak form](https://www.mdpi.com/2075-5309/16/21/4360) and
[Army foundation manual, equation 3-10](https://www.wbdg.org/FFC/DOD/UFC/ARCHIVES/ufc_3_220_03fa_2004.pdf).

The primary method solves each layer analytically in the Laplace domain,
joins interfaces through a dynamic-stiffness system, and numerically inverts
at two orders. An independent finite volume calculation uses half-cell
hydraulic resistances, exact modal time evolution, and 64/128-cell Richardson
refinement per layer.

Both effective stress and pore pressure are compared in every layer. The
largest method disagreement must be strictly below `1e-6` of the largest
applied surcharge up to the observation time. Each component must also agree
to `1e-6` relative to its own magnitude; a tiny response cannot disappear
under the peak-load tolerance. Inversion and load-balance
disagreements must each be below `1e-9` on that scale. Unresolved inputs are
refused. These are numerical agreement checks, normalized by applied load,
while the additional component gate resolves small residual responses.
Collapsed pulse intervals and underflowed normalized components are refused.
Dimensional products and sums are accumulated exactly before final rounding,
so a subnormal compliance cannot corrupt a representable settlement.

`mean_excess_pore_pressure_Pa` is the depth average. The separate
`storage_weighted_excess_pore_pressure_Pa` weights by `mv * thickness`; only
the latter combines with total compliance to recover settlement. Layer means
and settlements are returned. Unloading can produce negative excess pressure
while effective stress and settlement remain positive.

## Scope and validation

The model excludes nonlinear compression indices, plasticity, secondary
compression, finite footing stress fields, imperfect contact, internal drains
and varying properties. Such inputs are rejected. Applied strains must remain
in the stated linear model's physical range.

Tests change geometry and material constants, both drainage conditions,
ramps, jumps, rebound, homogeneous subdivision and length/time scaling.
An independent Fourier oracle uses `x = integral(dz / sqrt(cv))` when
`mv * sqrt(cv)` is constant. Unequal-impedance reversal and an invented
internal drain expose averaging and independent-layer shortcuts. A corrupted
secondary solver must prevent verification.

The existing inline budget route accepts a complete bounded JSON request
only with an observed empty file inventory, no MCP servers, and no requested
deliverable files. All residuals must pass. General engineering tasks retain
their normal model and deliverable workflow.

These checks are not a new score on the 29 public PLE tasks. A mock SSE
provider checks protocol behavior. Genuine dev artifacts must come from the
organizer's `eval/run.py`; this change does not edit historical `artifacts/`
or submit a portal entry.
