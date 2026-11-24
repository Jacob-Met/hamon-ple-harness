# Prescribed-time two-body transfers

Select `mode=lagrangian_orbital` and `spec.subkind=lambert_transfer` on the registered engineering tool. The input is a pair of inertial Cartesian position vectors and a positive flight time:

```json
{
  "mode": "lagrangian_orbital",
  "spec": {
    "subkind": "lambert_transfer",
    "mu": 398600441800000,
    "r1_m": [7000000, 0, 0],
    "r2_m": [0, 7000000, 0],
    "tof_s": 1457.1291594215038,
    "transfer_path": "short"
  }
}
```

Use metres, seconds and `mu` in m³/s², in one common inertial frame. `transfer_path` is `short` (default, less than π) or `long` (more than π). It is a geometric choice, independent of any externally chosen north axis. The returned `v1_m_s` and `v2_m_s` are the endpoint velocities along that arc.

For exactly antipodal endpoints, supply a nonzero `plane_normal` perpendicular to the departure position. The short choice follows that normal; long follows its negative. Both sweep π. The perpendicularity check uses the supplied binary64 numbers exactly. Coincident and same-ray radial geometries are refused because they do not identify this transverse orbit. Non-antipodal endpoints already define the plane, so a second normal is refused.

## Computation and evidence

The solver uses the Lancaster–Blanchard time equation and radial/transverse velocity reconstruction described by [Izzo, *Revisiting Lambert’s Problem*](https://arxiv.org/abs/1403.2705), equation 18 and Algorithm 1. It brackets the monotone zero-revolution branch in the elliptic and hyperbolic domains, with the analytic parabolic limit. For 1–32 complete revolutions, it first isolates the time curve's stationary minimum using equation 22. It then brackets the requested root on just one side of that minimum, so a root iteration cannot silently switch solutions. Intermediate arithmetic uses a private 90-decimal-digit context and bounded loops.

After rounding the velocities to their JSON binary64 representation, a separate universal-Kepler initial-value calculation propagates the **published departure velocity** for the supplied flight time. The certificate includes arrival-position error, arrival-velocity error, time residuals, and energy and angular-momentum differences. `endpoint_verified` requires all normalized residuals below 1e-10. The arrival-position error is normalized by the arrival radius, so a large departure radius cannot hide a missed small destination.

Optional `proposed_v1_m_s` and `proposed_v2_m_s` check proposed endpoint velocities. A failed claim makes `verified=false` even if the independently computed solution closes. Reversing both velocities preserves their energy and common angular-momentum magnitude, but fails these directional boundary conditions. Read `verified` and diagnostics before using a result.

The regression course derives elliptic endpoints from eccentric anomaly, hyperbolic endpoints from hyperbolic anomaly and parabolic endpoints from Barker’s equation. It also integrates 24 supplied trajectories with SciPy DOP853 in Cartesian coordinates, independently of either production time equation. A short-arc control uses the exact rounded endpoint and Cartesian Taylor equations: rounding cos(1e-6) introduces a real radial departure velocity of about −4.45e-11 in normalized units.

## Scope

This is an unperturbed point-mass two-body model with zero to 32 complete revolutions. It supports elliptic, parabolic and hyperbolic transfers and explicitly chosen arc directions. It does not establish three-body dynamics, collision clearance, perturbations, thrust, launch constraints, or mission feasibility. `conic_periapsis_m` describes the full fitted conic; that periapsis need not occur during the selected arc.

The default remains `revolutions=0`. For a positive integer count up to 32, `revolution_branch` is mandatory: `lower_x` or `upper_x` selects the smaller or larger Lancaster–Blanchard x root on either side of the minimum-time point. These labels choose solutions, not a minimum-delta-v mission. The short/long geometric arc choice remains separate. `minimum_tof_s` reports the branch-pair threshold rounded to binary64; a time below the true computed threshold is refused, without a permissive rounding tolerance. A rounded threshold may itself lie just below the true value. Supplying a branch for zero revolutions is refused. Additional physical parameters are refused rather than silently discarded. Numerical range, ambiguity or iteration-budget failures return `verified=false`; a small conservation residual alone cannot certify an unresolved endpoint.

The existing `orbital_transfer` subkind retains circular Hohmann sizing. It now refuses Lambert position/time/branch parameters, which it previously ignored. Select `lambert_transfer` to solve those boundary conditions. This implementation and its synthetic tests do not constitute a new scored PLE evaluation.


## Complete-revolution evidence

The independent endpoint replay uses the published departure velocity. Its
specific energy also determines the orbital period, and the elapsed number of
whole periods must equal the requested revolution count. This prevents endpoint
agreement on another revolution branch from being certified. The result exposes
`propagated_revolutions`, `revolution_count_verified`, and
`swept_angle_rad` including complete turns; `transfer_angle_rad` remains the
geometric endpoint angle.

Four published 36,000-second transfers from Der's AMOS 2011 example are checked
against their tabulated velocities, independently transcribed in the
[source-pinned LambertHub reference](https://github.com/jorgepiloto/lamberthub/blob/5010709b000bb6b29831ad628fa4b1ce1f519036/tests/test_multirev_solvers.py).
Both explicit roots, short and long geometric arcs, infeasible flight times,
and the minimum-time bifurcation are covered. Circular trajectories independently
check the revolution count through 32 complete turns. These are physical solver
and protocol checks; they do not measure an organizer score.
