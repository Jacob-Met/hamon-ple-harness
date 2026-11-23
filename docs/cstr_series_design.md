# Equal-volume CSTR inverse design

`solve_reactor_kinetics` and `dispatch_wave6_verifier` solve a prescribed final
conversion for a series of equal-volume, steady CSTRs from the coupled material
balances. The fixed-volume forward solver retains its existing all-trajectory
enumeration.

## Why the old inversion failed

The old path chose one forward root at each tank and then searched for a volume
that produced the requested conversion. With an adiabatic Arrhenius rate, that
chosen outlet can jump between steady branches. A scalar root finder can converge
to the jump even though no trajectory at that volume has the requested outlet.
Recomputing the same volume function then reported a zero residual.

For two tanks with `C_A0=v0=k_ref=1`, `T0=300`, `T_ref=350`,
`delta_T_adiabatic=200`, `E_a=80000`, first-order liquid kinetics, eight requested
conversions from 0.01 to 0.99 collapsed to a volume near 0.0642357003515. At a
requested conversion of 0.2, the closest physical forward trajectory missed by
0.1993357527 despite `verified=true`.

## Coupled design equation

Let `h=V_stage/F_A0`. For a specified final outlet, recover each preceding outlet
from `X_previous = X_next - h*r(X_next)`. A physical design is a zero of the
recovered feed conversion with every intermediate outlet in `[0, X_target]`.
The final tank gives the finite search bound `h <= X_target/r(X_target)`.

The implementation encloses both this recurrence and its derivative using an
independent 80-digit interval context. It excludes intervals that cannot contain
a physical zero, proves unique roots using a strict derivative sign and opposite
endpoint signs, and subdivides the remaining intervals. Overlapping subdivisions
keep boundary roots inside an interval. Duplicate root enclosures are merged only
after proving uniqueness on their union. A single tank uses its direct linear
balance. Root refinement uses interval signs; it does not use `iv.findroot`.

Each accepted design is checked again using the rounded volume and independently
reported conversion/unreacted-fraction coordinates. Every tank's material balance
and temperature relation must pass a relative `1e-9` residual. Nonfinite or
unrepresentable volume, residence time or exit rate is refused.

## Branch selection

```python
result = solve_reactor_kinetics({
    "reactor_type": "cstr_series", "n_stages": 2,
    "order": 1, "C_A0": 1, "v0": 1,
    "T0": 300, "delta_T_adiabatic": 200,
    "k_ref": 1, "T_ref": 350, "E_a": 80000,
    "gas_phase": False, "conversion": 0.99,
    "design_branch": "all",
})
```

This example has three physical design volumes, approximately 0.01191703,
0.01536138 and 0.05602327 cubic metres. `design_branch="all"` returns every design
in ascending volume order, with its full stage history. A zero-based integer
selects one of those volumes. A unique design needs no selector. Multiple designs
without a selector raise an error. `steady_state_branch` applies to fixed-volume
outlets and is rejected in the prescribed-conversion mode.

## Verification and limits

Tests exercise the public dispatcher and independently reconstruct each tank's
balances. They replay inverse outputs through the existing all-branch forward
solver and compare the two-tank root set against an independently derived
stationary polynomial. Analytic first- and zero-order controls cover one through
eight tanks; additional cases cover fractional order, gas expansion, kinetic
parameter precedence, ambiguous selections and overflow.

The algorithm refuses unresolved tangencies, inseparable output roots and an
exhausted interval budget (32,768 examined boxes or 1,200 subdivisions). Such a
refusal is not a claim that no physical design exists. It is not a completeness
guarantee for inputs that exceed the supported precision or budget. The underlying
mpmath interval context is experimental; this path uses its elementary arithmetic,
exponential, logarithm and power operations, with independent physical controls.

This is a numerical verifier change. It does not establish PLE task accuracy,
end-to-end cost, kit-check readiness or a scored portal result.

Physical references:
- [University of Michigan, CSTR multiplicity](https://websites.umich.edu/~elements/7e/12chap/prof-steady.html)
- [University of Michigan, adiabatic energy balances](https://public.websites.umich.edu/~elements/7e/11chap/summary.html)
- [mpmath interval contexts](https://mpmath.org/doc/current/contexts.html)
