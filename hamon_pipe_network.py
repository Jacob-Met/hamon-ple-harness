"""Steady, fixed-demand Darcy–Weisbach flow in a connected passive water network."""

import math
from bisect import bisect_left

import numpy as np
from scipy.linalg import null_space


def number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite numeric hydraulic input")
    return float(value)


def identifiers(rows, label):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} data must be a list of objects")
    labels = [r["id"] for r in rows]
    if (
        not rows
        or any(not isinstance(name, str) or not name for name in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(f"{label} IDs must be unique nonempty strings")
    return labels


def pipe_law(length, diameter, roughness, minor, viscosity, gravity=9.80665):
    """Signed loss in m and positive derivative in m per (m³/s)."""
    area = np.pi * diameter**2 / 4
    re_per_flow = diameter / (viscosity * area)
    geometry = length / diameter
    scale = 1 / (2 * gravity * area**2)
    laminar_slope = scale * geometry * 64 / re_per_flow
    relative = roughness / diameter / 3.7
    log_scale = 2 / math.log(10)

    def loss(flow):
        q = np.abs(flow)
        re = re_per_flow * q
        # The turbulent solution is used only in the blend and turbulent regimes.
        rt = np.maximum(re, 2000.0)
        x = -2 * np.log10(relative + 5.74 / rt**0.9)
        for _ in range(6):
            v = relative + 2.51 * x / rt
            x -= (x + 2 * np.log10(v)) / (1 + log_scale * 2.51 / (rt * v))
        v = relative + 2.51 * x / rt
        dx_dre = log_scale * 2.51 * x / (rt * (rt * v + log_scale * 2.51))
        friction = 1 / x**2
        turbulent = scale * geometry * friction * q**2
        turbulent_slope = scale * geometry * q * (2 * friction - 2 * rt * dx_dre / x**3)
        laminar = laminar_slope * q
        fraction = np.clip((re - 2000) / 2000, 0, 1)
        weight = fraction**2 * (3 - 2 * fraction)
        weight_slope = 6 * fraction * (1 - fraction) * re_per_flow / 2000
        magnitude = (1 - weight) * laminar + weight * turbulent + scale * minor * q**2
        slope = (
            (1 - weight) * laminar_slope
            + weight * turbulent_slope
            + weight_slope * (turbulent - laminar)
            + 2 * scale * minor * q
        )
        return np.sign(flow) * magnitude, slope

    low_head = loss(2000 / re_per_flow)[0]
    high_head = loss(4000 / re_per_flow)[0]

    def inverse(head_drop, scale_hint):
        target = np.abs(head_drop)
        # h = Rq + Kq² below transition; use the cancellation-free positive root.
        flow = (
            2
            * target
            / (laminar_slope + np.sqrt(laminar_slope**2 + 4 * scale * minor * target))
        )
        turbulent = (target >= high_head) & (minor == 0)
        if turbulent.any():
            # With no minor loss, u = q sqrt(f) is known from the imposed head.
            # The Colebrook term 1 / (Re sqrt(f)) is then independent of q.
            u = np.sqrt(target[turbulent] / (scale[turbulent] * geometry[turbulent]))
            flow[turbulent] = (
                -2
                * u
                * np.log10(relative[turbulent] + 2.51 / (u * re_per_flow[turbulent]))
            )
        unresolved = (target > low_head) & ~turbulent
        if unresolved.any():
            flow[unresolved] = np.abs(
                invert_loss(head_drop, loss, scale_hint)[unresolved]
            )
        return np.sign(head_drop) * flow

    return loss, inverse, re_per_flow


def invert_loss(head_drop, law, scale):
    """Invert each monotone pipe independently using a bracketed solve."""
    target = np.abs(head_drop)
    lo = np.zeros_like(target)
    hi = np.full_like(target, scale)
    for _ in range(128):
        uncertain = law(hi)[0] < target
        if not uncertain.any():
            break
        hi[uncertain] *= 2
    else:
        raise ValueError("Could not bracket flow from head differences")
    for _ in range(52):
        mid = (lo + hi) / 2
        low = law(mid)[0] < target
        lo = np.where(low, mid, lo)
        hi = np.where(low, hi, mid)
    return np.sign(head_drop) * (lo + hi) / 2


def newton(initial, evaluate, tolerance):
    value = initial.copy()
    for _ in range(80):
        residual, jacobian = evaluate(value)
        norm = np.max(np.abs(residual), initial=0.0)
        if norm <= tolerance:
            return value
        change = np.linalg.solve(jacobian, residual)
        for backtrack in range(25):
            candidate = value - change / 2**backtrack
            candidate_norm = np.max(np.abs(evaluate(candidate)[0]), initial=0.0)
            if candidate_norm < norm:
                value = candidate
                break
        else:
            raise ValueError("Newton iteration could not reduce the physical residual")
    raise ValueError("Hydraulic solve did not converge")


def network(spec):
    nodes, pipes = spec["nodes"], spec["pipes"]
    node_ids = identifiers(nodes, "Node")
    pipe_ids = identifiers(pipes, "Pipe")
    node_fields = {"id", "fixed_head_m", "demand_m3_s"}
    pipe_fields = {
        "id",
        "from",
        "to",
        "length_m",
        "diameter_m",
        "roughness_m",
        "minor_loss_K",
    }
    if any(set(node) - node_fields for node in nodes) or any(
        set(pipe) - pipe_fields for pipe in pipes
    ):
        raise ValueError(
            "Unsupported node or pipe attributes in passive fixed-demand network"
        )
    if any(
        key in spec
        for key in ("pumps", "valves", "controls", "tanks", "emitters", "demands")
    ):
        raise ValueError("Only passive pipes with fixed junction demands are supported")
    reservoirs = [i for i, row in enumerate(nodes) if "fixed_head_m" in row]
    if not reservoirs or len(nodes) < 2:
        raise ValueError("At least one prescribed head and two connected nodes required")
    fixed_heads = np.array([number(nodes[i]["fixed_head_m"]) for i in reservoirs])
    datum = fixed_heads[0]
    demands = np.array([number(n.get("demand_m3_s", 0)) for n in nodes])
    if np.any(demands[reservoirs] != 0):
        raise ValueError(
            "Reservoir supply is calculated; put known demands at junctions"
        )
    viscosity = number(spec.get("kinematic_viscosity_m2_s", 1e-6))
    if viscosity <= 0:
        raise ValueError("Kinematic viscosity must be positive")
    index = {name: i for i, name in enumerate(node_ids)}
    incidence = np.zeros((len(nodes), len(pipes)))
    for k, p in enumerate(pipes):
        a, b = index[p["from"]], index[p["to"]]
        if a == b:
            raise ValueError("Pipe cannot connect a node to itself")
        incidence[a, k], incidence[b, k] = 1.0, -1.0
    if np.linalg.matrix_rank(incidence[1:]) != len(nodes) - 1:
        raise ValueError("All nodes must belong to one connected network")
    junctions = np.ones(len(nodes), dtype=bool)
    junctions[reservoirs] = False
    A = incidence[junctions]
    boundary_drop = incidence[reservoirs].T @ (fixed_heads - datum)
    length = np.array([number(p["length_m"]) for p in pipes])
    diameter = np.array([number(p["diameter_m"]) for p in pipes])
    roughness = np.array([number(p.get("roughness_m", 0)) for p in pipes])
    minor = np.array([number(p.get("minor_loss_K", 0)) for p in pipes])
    if (
        (length <= 0).any()
        or (diameter <= 0).any()
        or (roughness < 0).any()
        or (roughness >= diameter).any()
        or (minor < 0).any()
    ):
        raise ValueError(
            "Require positive pipe dimensions, 0 <= roughness < diameter and nonnegative minor losses"
        )
    law, inverse, re_per_flow = pipe_law(length, diameter, roughness, minor, viscosity)
    d = demands[junctions]
    scale = max(float(np.abs(d).sum()), 1e-12)

    # In a one-reservoir tree, each pipe drop is an independent pressure
    # coordinate. Retain these local drops while balancing the junctions:
    # subtracting two large node heads can erase a small downstream drop.
    radial = len(reservoirs) == 1 and len(pipes) == len(nodes) - 1
    if radial:
        initial = np.zeros(len(pipes))
    else:
        # The harmonic initial head includes imposed reservoir through-flow.
        initial = np.linalg.lstsq(A.T, -boundary_drop, rcond=None)[0]
        if np.any(boundary_drop):
            initial_flow = inverse(A.T @ initial + boundary_drop, scale)
            scale = max(scale, float(np.max(np.abs(initial_flow))))

    def pipe_drops(pressure):
        return pressure if radial else A.T @ pressure + boundary_drop

    def nodal_equations(pressure):
        flow = inverse(pipe_drops(pressure), scale)
        residual = A @ flow + d
        derivative = A / law(flow)[1]
        return residual, derivative if radial else derivative @ A.T

    pressure = newton(initial, nodal_equations, min(1e-12, 5e-13 * scale))
    qnode = inverse(pipe_drops(pressure), scale)
    head = np.linalg.solve(A.T, pressure) if radial else pressure
    # The second formulation begins with continuity satisfied, and may change
    # only circulations and flows between reservoirs. Their prescribed head
    # differences supply the energy term on those boundary-to-boundary paths.
    feasible = np.linalg.lstsq(A, -d, rcond=None)[0]
    cycles = null_space(A)

    def loop_equations(amount):
        drop, slope = law(feasible + cycles @ amount)
        return cycles.T @ (drop - boundary_drop), (cycles.T * slope) @ cycles

    loss_scale = np.max(np.abs(law(feasible)[0] - boundary_drop))
    loop_tolerance = max(32 * np.finfo(float).eps * loss_scale,
                         min(1e-13, max(1e-26, 1e-13 * loss_scale)))
    amplitude = newton(np.zeros(cycles.shape[1]), loop_equations, loop_tolerance)
    qloop = feasible + cycles @ amplitude
    drop, _ = law(qloop)
    total_head = np.full(len(nodes), datum)
    total_head[reservoirs] = fixed_heads
    total_head[junctions] += head
    supply = incidence[reservoirs] @ qloop
    continuity = float(np.max(np.abs(A @ qloop + d), initial=0.0))
    energy = float(np.max(np.abs(incidence.T @ total_head - drop)))
    agreement = float(np.max(np.abs(qnode - qloop)))

    def proposed(key, labels):
        supplied = spec[key]
        if not isinstance(supplied, dict) or set(supplied) != set(labels):
            raise ValueError(f"{key} must contain every matching ID exactly once")
        return np.array([number(supplied[name]) for name in labels])

    registered = {}
    eps = np.finfo(float).eps
    if "proposed_flows_m3_s" in spec:
        flows = proposed("proposed_flows_m3_s", pipe_ids)
        # Check every pipe and junction. A large main cannot conceal a missing
        # demand or a reversed flow on a much smaller branch.
        pipe_tolerance = np.minimum(1e-6, 1e-7 * np.abs(qloop) + 32 * eps * scale)
        incident_flow = np.abs(A) @ np.abs(flows)
        mass_tolerance = np.minimum(
            1e-6, 1e-7 * np.abs(d) + 32 * eps * (incident_flow + np.abs(d))
        )
        registered["flows_match"] = bool(
            np.all(np.abs(flows - qloop) <= pipe_tolerance)
        )
        registered["junction_demands_match"] = bool(
            np.all(np.abs(A @ flows + d) <= mass_tolerance)
        )
    if "proposed_heads_m" in spec:
        heads = proposed("proposed_heads_m", node_ids)
        # Verify pressure differences in the pipe's physical scale. The two
        # rounded endpoint heads can contribute at most one ulp each.
        roundoff = np.abs(incidence.T) @ np.abs(np.spacing(heads))
        drop_tolerance = np.minimum(1e-6, 1e-7 * np.abs(drop) + roundoff)
        registered["reservoir_head_matches"] = bool(
            np.all(np.abs(heads[reservoirs] - fixed_heads) <=
                   np.minimum(1e-6, np.abs(np.spacing(fixed_heads))))
        )
        registered["pipe_head_drops_match"] = bool(
            np.all(np.abs(incidence.T @ heads - drop) <= drop_tolerance)
        )
    verified = (
        continuity < min(1e-6, 1e-7 * scale)
        and energy < 1e-6
        and agreement < min(1e-6, 1e-7 * scale)
        and all(registered.values())
    )
    return dict(
        ok=True,
        conservation_verified=bool(verified),
        proposed_checks=registered,
        calculation="pipe_network",
        flows_m3_s=dict(zip(pipe_ids, qloop.tolist())),
        heads_m=dict(zip(node_ids, total_head.tolist())),
        reynolds=dict(zip(pipe_ids, (re_per_flow * np.abs(qloop)).tolist())),
        reservoir_supply_m3_s=float(supply.sum()),
        reservoir_supplies_m3_s=dict(zip((node_ids[i] for i in reservoirs), supply.tolist())),
        independent_cycles=len(pipes) - len(nodes) + 1,
        max_mass_residual_m3_s=continuity,
        max_energy_residual_m=energy,
        max_nodal_cycle_flow_difference_m3_s=agreement,
    )


def verify_pipe_network(spec):
    """Check a steady fixed-demand network, with optional complete proposed flow and head maps."""
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return network(spec)
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:
        return dict(
            ok=False,
            conservation_verified=False,
            calculation="pipe_network",
            reason=str(exc),
        )


def equipped_branch_head(branch, flow_per_pump):
    """Common stage head at total flow plus one parallel path's hose and strainer."""
    fields = {"common_head_m", "hose_length_m", "hose_diameter_m", "hose_roughness_m",
              "kinematic_viscosity_m2_s", "strainer_reference_flow_m3_s",
              "strainer_reference_head_m"}
    if not isinstance(branch, dict) or branch.keys() - {"gravity_m_s2"} != fields:
        raise ValueError("Supply the common head, hose geometry, viscosity and strainer rating")
    common = number(branch["common_head_m"])
    length = number(branch["hose_length_m"])
    diameter = number(branch["hose_diameter_m"])
    roughness = number(branch["hose_roughness_m"])
    viscosity = number(branch["kinematic_viscosity_m2_s"])
    gravity = number(branch.get("gravity_m_s2", 9.80665))
    reference_flow = number(branch["strainer_reference_flow_m3_s"])
    reference_head = number(branch["strainer_reference_head_m"])
    if min(length, diameter, viscosity, gravity, reference_flow) <= 0:
        raise ValueError("Hose dimensions, viscosity, gravity and reference flow must be positive")
    if not 0 <= roughness < diameter or min(common, reference_head) < 0:
        raise ValueError("Require nonnegative head and roughness less than the hose diameter")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        law, _, _ = pipe_law(length, diameter, roughness, 0, viscosity, gravity)
        hose = float(law(flow_per_pump)[0])
        strainer = reference_head * (flow_per_pump / reference_flow) ** 2
        required = number(common + hose + strainer)
    return {"common_head_m": common, "hose_head_loss_m": hose,
            "strainer_head_loss_m": strainer, "required_head_m": required}


def verify_pump_duty(spec):
    """Check fixed-speed duty for identical parallel pumps at their tabulated head.

    Supplying both electrical fields declares a connected input rating per pump
    and the panel capacity. Positive hydraulic work requires a positive electrical
    connection. The ratings do not measure the power drawn at the operating point.
    Omitting both fields requests a hydraulic head check only.
    """
    result = {"calculation": "pump_duty", "mode": "thermal_fluid",
              "tool": "hamon_conservation_cad_verifier", "ok": False, "duty_verified": False}
    try:
        fields = {"mode", "calculation", "curve", "total_flow_m3_s", "required_head_m",
                  "identical_pumps", "per_pump_input_W", "available_panel_W", "equipped_branch"}
        if spec.keys() - fields:
            raise ValueError("Unsupported inputs for fixed-speed identical parallel pump duty")
        count = spec.get("identical_pumps", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("identical_pumps must be a positive integer")
        flow = number(spec["total_flow_m3_s"])
        if flow < 0:
            raise ValueError("Expected nonnegative design flow")
        if ("required_head_m" in spec) == ("equipped_branch" in spec):
            raise ValueError("Supply exactly one of required_head_m and equipped_branch")
        per_pump = flow / count
        result["flow_per_pump_m3_s"] = per_pump
        if "equipped_branch" in spec:
            result.update(equipped_branch_head(spec["equipped_branch"], per_pump))
            required = result["required_head_m"]
        else:
            required = number(spec["required_head_m"])
            if required < 0:
                raise ValueError("Expected nonnegative required head")
        curve = spec["curve"]
        if not isinstance(curve, list) or len(curve) < 2:
            raise ValueError("The pump curve needs at least two tabulated points")
        flows, heads = [], []
        for point in curve:
            if not isinstance(point, dict) or point.keys() != {"flow_m3_s", "head_m"}:
                raise ValueError("Each curve point must specify flow_m3_s and head_m")
            q, h = number(point["flow_m3_s"]), number(point["head_m"])
            if q < 0 or h < 0 or (flows and q <= flows[-1]):
                raise ValueError("Curve points require increasing nonnegative flow and nonnegative head")
            flows.append(q)
            heads.append(h)
        checks = {}
        if "per_pump_input_W" in spec or "available_panel_W" in spec:
            draw = number(spec["per_pump_input_W"])
            capacity = number(spec["available_panel_W"])
            if min(draw, capacity) < 0:
                raise ValueError("Electrical inputs must be nonnegative")
            total_draw = count * draw
            if not math.isfinite(total_draw):
                raise ValueError("Total electrical draw exceeds the numerical range")
            result.update(total_input_W=total_draw, panel_margin_W=capacity - total_draw)
            checks["panel_adequate"] = total_draw <= capacity
        if not flows[0] <= per_pump <= flows[-1]:
            raise ValueError("Flow per pump is outside the tabulated curve")
        i = min(bisect_left(flows, per_pump, lo=1), len(flows) - 1)
        fraction = (per_pump - flows[i - 1]) / (flows[i] - flows[i - 1])
        head = (1.0 - fraction) * heads[i - 1] + fraction * heads[i]
        margin = head - required
        checks["head_adequate"] = margin >= 0
        if flow > 0 and head > 0 and result.get("total_input_W") == 0:
            checks["connected_power_consistent"] = False
            result["reason"] = (
                "Positive flow and developed pump head require positive connected "
                "electrical power for this electrically driven duty"
            )
        return {**result, "ok": True, "checks": checks,
                "available_head_m": head, "head_margin_m": margin, "duty_verified": all(checks.values())}
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return {**result, "reason": str(exc)}
