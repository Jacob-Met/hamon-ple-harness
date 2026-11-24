# Simultaneous heat and friction in a constant-area duct

Call `hamon_conservation_cad_verifier` with `mode=compressible_acoustics`
and `spec.subkind=fanno_rayleigh`. This solves steady one-dimensional flow of
a calorically perfect gas with a constant friction factor and uniform heat
addition or removal per unit length. It follows the supplied inlet branch and
refuses an exit beyond the first sonic point.

```json
{
  "mode": "compressible_acoustics",
  "spec": {
    "subkind": "fanno_rayleigh",
    "M1": 0.5,
    "T01": 315.0,
    "P1": 101325.0,
    "L": 0.5,
    "D_h": 0.1,
    "f": 0.01,
    "friction_factor_convention": "fanning",
    "q_J_kg": 40000.0,
    "gamma": 1.4,
    "R": 287.05
  }
}
```

`T01` is inlet **stagnation** temperature in kelvin; `P1` is inlet **static**
absolute pressure in pascals, defaulting to 101325. `L` and `D_h` are metres.
`q_J_kg` is the signed total heat transferred per kilogram over the requested
length. It is distributed uniformly: its rate per metre is `q_J_kg/L`.
The default friction convention is Fanning; Darcy is four times Fanning.
The specific gas constant `R` is J/(kg K). The optional `cp` uses the existing
shared perfect-gas consistency check. `gamma`, `R`, and `cp` must describe one
gas. An optional `branch` must agree with `M1`.

For this example, the coupled outlet Mach is 0.6069166972327382 and the static
temperature is 330.46840318607326 K. The friction-only answer is 0.5270569917822787
and 298.4204139912909 K. The old Fanno route accepted the heat parameter but
ignored it. Fanno now refuses nonzero heat; Rayleigh refuses nonzero friction.
Their pure single-effect calculations remain available.

## Governing equations and solution

The model uses constant mass flux `G`, velocity `u`, static pressure `p`, static
temperature `T`, stagnation temperature `T0`, and Darcy friction `f_D`:

\[
G=\rho u,\qquad p=GRT/u,
\]
\[
\frac{dp}{dx}+G\frac{du}{dx}=-\frac{f_D}{2D_h}Gu,
\qquad c_pT+\frac{u^2}{2}=c_pT_{01}+\frac{q}{L}x.
\]

These are the simultaneous heat/friction balances given in Caltech's
[compressible-flow guide, section 3.3](https://shepherd.caltech.edu/T5/resources/AE101_Guide.pdf).
NASA's [Sturas, TN D-6566 (1971)](https://ntrs.nasa.gov/citations/19720004565)
develops an integrating-factor solution for this constant-area, constant-flux
model. The implementation below derives its own nondimensional form from the
two balances; no chart values or task-specific answers enter the solver.

Set

\[
t=x/L,\quad w=u/\sqrt{RT_{01}},\quad
e=T_0/T_{01}=1+\epsilon t,\quad \epsilon=q/(c_pT_{01}),
\]
\[
b=R/(2c_p),\quad a=1-b,\quad K=f_DL/(2D_h).
\]

Momentum becomes

\[
\frac{dw}{dt}=\frac{w(Kw^2+\epsilon)}{e-aw^2}.
\]

For nonzero heat and friction, `c=epsilon/K` and
`mu(w)=sqrt(abs(w*w+c))/w` give the implicit solution

\[
e\mu(w)+ac\,\operatorname{sgn}(w_0^2+c)
\left[J(w)-J(w_0)\right]=\mu(w_0).
\]

`J` is `asinh(w/sqrt(c))` for positive `c`; for negative `c` it is
`acosh(w/sqrt(-c))` above the constant-velocity boundary and
`asin(w/sqrt(-c))` below it. The solver selects that boundary using exact
rationals from the supplied numbers. It handles the exact zero-force cooling
trajectory as constant velocity, avoiding a singular neighbouring chart.
Pure Fanno, pure Rayleigh, and unforced identity limits use their corresponding
analytic equations rather than dividing by zero heat or friction.

The root bracket intersects four physical constraints: the inlet Mach branch,
positive static temperature, the same side of the constant-velocity boundary,
and the forward direction of velocity change. At sonic conditions `e=aw*w`.
The sonic equation has derivative `2*a*w*mu(w)>0`, so it has at most one root
on a connected chart. The first reachable sonic point sets the length limit.

## Results and limits

Successful results include `M2`, `P2_Pa`, `T2_K`, `T02_K`, `u2_m_s`, pressure
and temperature ratios, and inlet mass flux. `verified` requires all reported
mass, energy, momentum, and path residuals below `1e-10`. Computation uses a
private 100-digit context and a bounded scalar root solve.

Mass and energy are replayed from the published binary64 state. Momentum uses
the published state and a separate wall-force integral over the physical
trajectory, with 32- and 64-point quadrature agreement required. Its endpoint
is the solved physical exit: inverting a rounded velocity to a different
length is ill-conditioned near constant-velocity cooling. The path residual
checks the high-precision implicit root at the requested distance. States
whose rounded Mach no longer distinguishes their branch are not verified.

`L_choke_m`, when present, extends the **same heat rate per unit length** and
friction factor. `q_choke_at_fixed_heat_rate_J_kg` is the heat accumulated at
that point. It is not an independent maximum heat input for arbitrary length.
For `M1=2` in the example, the separate friction-only and heat-only problems
permit a 0.5 m exit. Their simultaneous flow chokes at 0.42522686591353687 m.
The result therefore has `requested_state_feasible=false`, `choked=true`,
and a `reachable_sonic_state`, without a fabricated `M2` exit.

The model does not cover area variation, a wall-temperature boundary condition,
variable friction, nonuniform heat flux, shocks, mass injection, variable heat
capacity, real-gas effects, or a transonic continuation. A forced sonic inlet
is refused. Excess cooling that removes a finite positive static state is
also refused. Inlet Mach and pressure remain fixed; the solver does not adjust
the mass flow to rescue an infeasible duct. Unknown physical inputs are rejected.

## Independent checks

`tests/test_coupled_duct.py` integrates static pressure and velocity by solving
the original momentum and energy matrix at every DOP853 step. Its 32 fixed-seed
cases cover both Mach branches, heating and cooling, gas properties, pressure,
length, and friction. The initial run's worst outlet relative difference was
8.89e-16. Additional controls cover combined choking, physical partition into
two uniform segments, pure Fanno/Rayleigh limits, Darcy/Fanning equivalence,
pressure scaling, the exactly balanced cooling trajectory and adjacent input
floats, invalid properties, and zero forcing.

`tests/test_coupled_duct_registered.py` exercises actual Pi tool registration,
TypeScript dispatch, and the Python solver, including the reference response.
These numerical and runtime checks do not constitute a new PLE engineering
score, a repaired historical deliverable, or a measured model-token saving.
