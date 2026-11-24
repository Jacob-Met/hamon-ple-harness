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

The solver uses the Lancaster–Blanchard time equation and radial/transverse velocity reconstruction described by [Izzo, *Revisiting Lambert’s Problem*](https://arxiv.org/abs/1403.2705), equation 18 and Algorithm 1. It brackets the monotone zero-revolution branch in the elliptic and hyperbolic domains, with the analytic parabolic limit. Intermediate arithmetic uses a private 90-decimal-digit context and bounded loops.

After rounding the velocities to their JSON binary64 representation, a separate universal-Kepler initial-value calculation propagates the **published departure velocity** for the supplied flight time. The certificate includes arrival-position error, arrival-velocity error, time residuals, and energy and angular-momentum differences. `endpoint_verified` requires all normalized residuals below 1e-10. The arrival-position error is normalized by the arrival radius, so a large departure radius cannot hide a missed small destination.

Optional `proposed_v1_m_s` and `proposed_v2_m_s` check proposed endpoint velocities. A failed claim makes `verified=false` even if the independently computed solution closes. Reversing both velocities preserves their energy and common angular-momentum magnitude, but fails these directional boundary conditions. Read `verified` and diagnostics before using a result.

The regression course derives elliptic endpoints from eccentric anomaly, hyperbolic endpoints from hyperbolic anomaly and parabolic endpoints from Barker’s equation. It also integrates 24 supplied trajectories with SciPy DOP853 in Cartesian coordinates, independently of either production time equation. A short-arc control uses the exact rounded endpoint and Cartesian Taylor equations: rounding cos(1e-6) introduces a real radial departure velocity of about −4.45e-11 in normalized units.

## Scope

This is an unperturbed point-mass two-body model with zero complete revolutions. It supports elliptic, parabolic and hyperbolic transfers and explicitly chosen arc directions. It does not establish three-body dynamics, collision clearance, perturbations, thrust, launch constraints, or mission feasibility. `conic_periapsis_m` describes the full fitted conic; that periapsis need not occur during the selected arc.

Only `revolutions=0` is accepted. Additional physical parameters are refused rather than silently discarded. Numerical range, ambiguity or iteration-budget failures return `verified=false`; a small conservation residual alone cannot certify an unresolved endpoint.

The existing `orbital_transfer` subkind retains circular Hohmann sizing. It now refuses Lambert position/time/branch parameters, which it previously ignored. Select `lambert_transfer` to solve those boundary conditions. This implementation and its synthetic tests do not constitute a new scored PLE evaluation.

