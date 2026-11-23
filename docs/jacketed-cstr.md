# Jacketed CSTR: coupled steady states and local dynamic stability

The reactor verifier previously ignored a supplied heat-transfer object and
solved the adiabatic model. A standard exothermic CSTR therefore returned one
`verified` state at 559.1868751226525 K even though its specified jacket confines
every physical steady state to 316.17050067659–383.82949932341 K.

The `reactor_kinetics` family now recognizes the explicit `heat_transfer` object
for one fixed-volume, constant-density liquid CSTR. It solves both original
balances, enumerates all conversion branches, and classifies the full two-state
linearization. A steady balance certificate does not imply dynamic stability.

## Input and scope

```json
{
  "reactor_type": "cstr",
  "order": 1,
  "C_A0": 1000,
  "v0": 0.0016666666666666668,
  "volume": 0.1,
  "A_preexp": 1200000000,
  "E_a": 72750,
  "T0": 350,
  "delta_T_adiabatic": 209.20502092050208,
  "steady_state_branch": "all",
  "heat_transfer": {
    "UA_W_K": 833.3333333333334,
    "rho_kg_m3": 1000,
    "Cp_J_kg_K": 239,
    "coolant_T_K": 300
  }
}
```

Concentration is mol/m³, volume is m³, volumetric flow is m³/s, and temperature
and temperature rise are K. Activation energy is J/mol. The rate constant has
units `(mol/m³)^(1-order)/s`. `UA_W_K` is the total conductance, not a film
coefficient or an area alone. The fixed coolant temperature has no jacket
dynamic state. Density and heat capacity are constant and positive; conductance
is nonnegative. A negative adiabatic temperature rise represents an endothermic
reaction. The nonnegative power-law order defaults to one.

Supply exactly one kinetic law: `k`; `A_preexp` and `E_a`; or `k_ref`, `T_ref`
and `E_a`. The heat-transfer object requires exactly the four documented keys.
Gas expansion, PFRs, reactor cascades, a prescribed exit temperature, conversion
as an independent input, and inconsistent redundant molar feed are refused.
Every point on the conversion interval must have positive temperature. The
model does not cover multiple reactions, variable properties, boiling, spatial
gradients, a dynamic coolant jacket, or nonlinear stability beyond the local
linearization. Numeric strings, booleans, nonfinite values and unrepresentable
output coordinates are refused.

The existing top-level `jacket_to_feed_heat_capacity_ratio` and `coolant_T_K`
interface remains available, including its inverse-design behavior. Do not mix
that interface with `heat_transfer`. The full dynamic classification documented
here accompanies the physical-SI `heat_transfer` model.

With several steady states, select `"all"` or a zero-based index in ascending
conversion order. Omitting the choice is an error. A unique state is selected
automatically. Each row reports both conversion and remaining reactant to
avoid losing a small complementary fraction.

## Equations and exact branch isolation

Let `q` denote flow, `V` volume, `C0` feed concentration, `r=k(T) C^n`, and
`β=delta_T_adiabatic/C0`. The original dynamic balances are

```
dC/dt = (q/V)(C0-C) - r
dT/dt = (q/V)(T0-T) + β r + UA/(ρ Cp V)(Tc-T).
```

Eliminating reaction rate only at steady state gives

```
λ = UA/(ρ Cp q)
T(X) = (T0 + λ Tc + delta_T_adiabatic X)/(1+λ).
```

The coefficients remain exact rational values of the supplied JSON numbers.
The shared CSTR isolator partitions conversion at every stationary point of
the logarithmic mass balance using exact polynomial isolation, then refines
each branch. This avoids the branch loss of a single bracket or sampled grid.
Its internal `exact_inputs` path also retains nonzero slopes below the legacy
absolute threshold and integer kinetic values above binary64's consecutive
integer range. Existing callers retain their established parsing behavior.

The returned coordinates are independently substituted into the original
material and heat balances. Relative residuals use the actual magnitudes of
the balance terms without an absolute floor. `verified` requires both
residuals below 1e-9; it is not a claim that the reactor will stay at that state.

## Stability calculation

At equilibrium, `a=q/V`, `h=UA/(ρ Cp V)`, `r_C=n a X/(1-X)`, and
`r_T=C0 a X (Ea/R)/T²`. Constant-rate kinetics use `Ea=0`.

```
J = [[-a-r_C,   -r_T],
     [ β r_C, -a-h+β r_T]].
```

Exact rational bounds over the isolated conversion interval enclose the trace
and determinant. The only possible interior extremum of `X/T(X)^2` is included.
Negative determinant establishes a saddle; positive trace establishes
instability; positive determinant and negative trace establish asymptotic
linear stability. Exact zero determinant or a zero trace with positive
determinant is nonhyperbolic. An enclosure that cannot establish a sign returns
`unresolved_near_boundary`. Serialized interval endpoints round outward.
Complex eigenvalues and the Jacobian are additional numerical diagnostics;
classification does not subtract rounded eigenvalues.

## Independent controls and observations

The primary reference is the Notre Dame CBE30338
[exothermic CSTR notebook](https://ndcbe.github.io/controls/notebooks/2/Exothermic-CSTR.html),
which specifies the original concentration/energy equations and a two-state
Jacobian. Parameters above are converted from its litre/minute/gram units.
This repository uses `R=8.31446261815324`; the tutorial rounds it to `8.314`.
The numerical oracle uses the repository's constant and identical binary input
values, so the temperatures below are an independently reproduced model, not
quoted notebook outputs. The University of Michigan
[CSTR steady-state discussion](https://websites.umich.edu/~elements/7e/12chap/prof-steady.html)
also describes multiplicity and turning-point conditions.

Before implementation, independent SciPy temperature bracketing of the original
balances found the following states. Tests independently reconstruct both ODEs,
the analytic Jacobian, and NumPy eigenvalues at the reported coordinates.

| Coolant K | Steady-state temperature K | Local linear classification |
| --- | --- | --- |
| 290 | 312.6587596626876 | Asymptotically stable |
| 295 | 317.74658211425844 | Asymptotically stable |
| 300 | 324.48605305000467 | Asymptotically stable |
| 300 | 349.9623318873878 | Unstable saddle |
| 300 | 369.7246520915924 | Unstable oscillatory state |
| 305 | 378.07277063838217 | Unstable oscillatory state |
| 310 | 383.8921610143717 | Asymptotically stable |

Additional controls exercise the zero-conductance adiabatic limit, an
endothermic second-order reactor, explicit/ambiguous branch selection, strict
input refusals, an exact fold, and an exact Hopf boundary. Adjacent binary64
conductances around the fold produce 3/2/1 distinct roots. Scaling every
temperature and activation energy by `2^-60` preserves all three branches.
Integer rate constants `2^53-1`, `2^53`, `2^53+1`, `2^53+2` across a scaled fold
produce 1/2/3/3 roots. These last two controls came from independent peer review.

`test_cstr_thermal_registered.py` exercises the actual Pi tool registration,
TypeScript execution and Python dispatch with the same heat-balance and
boundary cases. The registered verifier runs in the harness image; the separate
task-container CLI exposes a different subset of engineering commands.
These are physics and runtime regression controls, not organizer benchmark
scores or operational process qualifications. No portal submission is involved.
