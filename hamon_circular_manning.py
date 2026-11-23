"""Normal depths for steady uniform free-surface flow in a circular conduit."""

import math

from mpmath import mp


def circular_manning(spec):
    if "bottom_width_m" in spec or "side_slope_z" in spec:
        raise ValueError("Circular Manning flow requires diameter_m rather than trapezoid geometry")
    ctx = mp.clone()
    ctx.dps = 85

    def positive(key, default=None):
        raw = spec.get(key, default)
        if raw is None:
            raise ValueError(f"Circular Manning flow requires {key}")
        value = float(raw)
        if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Circular Manning flow requires positive finite {key}")
        return ctx.mpf(value)

    diameter = positive("diameter_m")
    roughness = positive("manning_n")
    slope = positive("slope_S")
    target = positive("Q_m3_s")
    gravity = positive("gravity_m_s2", 9.80665)
    pi = ctx.pi

    def emit(value):
        result = float(value)
        if not math.isfinite(result) or (value and result == 0):
            raise ValueError("Circular Manning result exceeds the binary64 reporting range")
        return result

    # phi is half the angle subtended by the wetted perimeter. The sign of
    # dQ/dphi is the sign of 5*phi*sin(phi)^2 - phi + sin(phi)*cos(phi).
    # This has exactly one interior zero, in (pi/2, pi).
    peak = ctx.findroot(lambda p: 5*p*ctx.sin(p)**2 - p + ctx.sin(p)*ctx.cos(p), (2.5, 2.7))
    full_area = pi*diameter**2/4
    full_flow = full_area * (diameter/4)**(ctx.mpf(2)/3) * ctx.sqrt(slope)/roughness

    def segment(p):
        if p > ctx.mpf("0.1"):
            return p - ctx.sin(2*p)/2
        # Stable shallow segment area when p and sin(2p)/2 nearly coincide.
        term = 2*p**3/3
        total = term
        for k in range(1, 55):
            term *= -4*p**2 / ((2*k+2)*(2*k+3))
            total += term
            if abs(term) <= abs(total)*ctx.eps:
                break
        return total

    def geometry(p, crown):
        area_fraction = segment(p)/pi
        perimeter_fraction = p/pi
        if crown:
            area_fraction = 1-area_fraction
            perimeter_fraction = 1-perimeter_fraction
        area = full_area*area_fraction
        perimeter = pi*diameter*perimeter_fraction
        flow = full_flow * (area_fraction**5/perimeter_fraction**2)**(ctx.mpf(1)/3)
        return area, perimeter, flow

    max_flow = geometry(peak, False)[2]
    full_wire, peak_wire = emit(full_flow), emit(max_flow)
    if full_wire == peak_wire:
        raise ValueError("Full-section and peak discharges are numerically indistinguishable")
    base = dict(cross_section="circular", full_section_gravity_flow_m3_s=full_wire,
                maximum_gravity_flow_m3_s=peak_wire, depth_at_maximum_flow_m=emit(diameter*ctx.sin(peak/2)**2))
    if target > peak_wire:
        return dict(base, normal_depth_solutions=[], requested_state_feasible=False, dual_verified=False,
                    reason="Requested discharge exceeds maximum uniform gravity flow")

    def root(crown):
        end = pi-peak if crown else peak
        # The shallow discharge scales as phi^(13/3). The excess above
        # full flow scales linearly with the small air-side angle at the crown.
        guess = ((target/full_flow-1)*3*pi/2 if crown
                 else end*(target/max_flow)**(ctx.mpf(3)/13))
        hi = min(end, 2*guess)
        while hi < end and geometry(hi, crown)[2] < target:
            hi = min(end, 2*hi)
        lo = ctx.mpf(0)
        for _ in range(180):
            mid = (lo+hi)/2
            if geometry(mid, crown)[2] < target:
                lo = mid
            else:
                hi = mid
            if lo and hi-lo <= lo*ctx.mpf("1e-34"):
                break
        return (lo+hi)/2

    def solution(p, crown, branch):
        near = diameter*ctx.sin(p/2)**2
        far = diameter*ctx.cos(p/2)**2
        y, clearance = (far, near) if crown else (near, far)
        depth_wire, clearance_wire = emit(y), emit(clearance)
        area, perimeter, _ = geometry(p, crown)

        # Independent replay from the smaller reported vertical distance:
        # integrate horizontal chord widths, not the angle/segment area law.
        from_crown = clearance_wire < depth_wire
        distance = ctx.mpf(clearance_wire if from_crown else depth_wire)
        d = distance/diameter
        area_cap = 4 * distance * ctx.sqrt(diameter*distance) * ctx.quad(
            lambda t: t*t*ctx.sqrt(1-d*t*t), [0, 1])
        arc_cap = 2*diameter*ctx.asin(ctx.sqrt(d))
        replay_area = full_area-area_cap if from_crown else area_cap
        replay_perimeter = pi*diameter-arc_cap if from_crown else arc_cap
        q_replay = replay_area*(replay_area/replay_perimeter)**(ctx.mpf(2)/3)*ctx.sqrt(slope)/roughness
        err = max(abs(q_replay-target)/target, abs(replay_area-area)/area,
                  abs(replay_perimeter-perimeter)/perimeter)
        width = 2*ctx.sqrt(y*clearance)
        speed = target/area
        froude = speed*ctx.sqrt(width/(gravity*area))
        return dict(discharge_branch=branch, normal_depth_m=depth_wire,
                    crown_clearance_m=clearance_wire, free_surface_width_m=emit(width),
                    velocity_m_s=emit(speed), froude_number=emit(froude),
                    regime="subcritical" if froude < 1 else "supercritical" if froude > 1 else "critical",
                    dual_rel_discrepancy=float(err))

    if target == peak_wire:
        solutions = [solution(peak, False, "maximum")]
    else:
        solutions = [solution(root(False), False, "increasing")]
        if target > full_wire:
            solutions.append(solution(root(True), True, "decreasing"))
        elif target == full_wire:
            base["full_section_boundary"] = {"depth_m": float(diameter), "gravity_flow_m3_s": full_wire}
    discrepancy = max(s["dual_rel_discrepancy"] for s in solutions)
    result = dict(base, normal_depth_solutions=solutions, requested_state_feasible=True,
                  dual_rel_discrepancy=discrepancy, dual_verified=discrepancy < 1e-8)
    if len(solutions) == 1:
        result.update({k: solutions[0][k] for k in ("normal_depth_m", "velocity_m_s", "froude_number", "regime")})
    return result
