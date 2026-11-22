"""Steady, fixed-demand Darcy–Weisbach flow in a connected passive water network."""

import math

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


def pipe_law(length, diameter, roughness, minor, viscosity):
    """Signed loss in m and positive derivative in m per (m³/s)."""
    area = np.pi * diameter**2 / 4
    re_per_flow = diameter / (viscosity * area)
    geometry = length / diameter
    scale = 1 / (2 * 9.80665 * area**2)
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
    if len(reservoirs) != 1 or len(nodes) < 2:
        raise ValueError(
            "One fixed-head reservoir and at least one demand junction required"
        )
    reservoir = reservoirs[0]
    datum = number(nodes[reservoir]["fixed_head_m"])
    demands = np.array([number(n.get("demand_m3_s", 0)) for n in nodes])
    if demands[reservoir] != 0:
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
    junctions = np.arange(len(nodes)) != reservoir
    A = incidence[junctions]
    if np.linalg.matrix_rank(A) != A.shape[0]:
        raise ValueError("Every junction must be connected to the reservoir")
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

    def nodal_equations(head):
        flow = inverse(A.T @ head, scale)
        residual = A @ flow + d
        jacobian = (A / law(flow)[1]) @ A.T
        return residual, jacobian

    head = newton(np.zeros(len(d)), nodal_equations, min(1e-10, 1e-10 * scale))
    qnode = inverse(A.T @ head, scale)
    # The second formulation begins with continuity satisfied, and may change
    # only circulations. This is the coupled form of the Hardy Cross correction.
    feasible = np.linalg.lstsq(A, -d, rcond=None)[0]
    cycles = null_space(A)

    def loop_equations(amount):
        drop, slope = law(feasible + cycles @ amount)
        return cycles.T @ drop, (cycles.T * slope) @ cycles

    loss_scale = np.max(np.abs(law(feasible)[0]))
    loop_tolerance = min(1e-11, max(1e-24, 1e-10 * loss_scale))
    amplitude = newton(np.zeros(cycles.shape[1]), loop_equations, loop_tolerance)
    qloop = feasible + cycles @ amplitude
    drop, _ = law(qloop)
    total_head = np.full(len(nodes), datum)
    total_head[junctions] += head
    continuity = float(np.max(np.abs(A @ qloop + d)))
    energy = float(np.max(np.abs(incidence.T @ total_head - drop)))
    agreement = float(np.max(np.abs(qnode - qloop)))
    registered = []
    for key, labels, actual in (
        ("proposed_flows_m3_s", pipe_ids, qloop),
        ("proposed_heads_m", node_ids, total_head),
    ):
        if key in spec:
            supplied = spec[key]
            if not isinstance(supplied, dict) or set(supplied) != set(labels):
                raise ValueError(f"{key} must contain every matching ID exactly once")
            values = np.array([number(supplied[name]) for name in labels])
            tolerance = (
                min(1e-6, 1e-7 * scale) if key == "proposed_flows_m3_s" else 1e-6
            )
            registered.append(float(np.max(np.abs(values - actual))) < tolerance)
    verified = (
        continuity < min(1e-6, 1e-7 * scale)
        and energy < 1e-6
        and agreement < min(1e-6, 1e-7 * scale)
        and all(registered)
    )
    return dict(
        ok=True,
        conservation_verified=bool(verified),
        calculation="pipe_network",
        flows_m3_s=dict(zip(pipe_ids, qloop.tolist())),
        heads_m=dict(zip(node_ids, total_head.tolist())),
        reynolds=dict(zip(pipe_ids, (re_per_flow * np.abs(qloop)).tolist())),
        reservoir_supply_m3_s=float(incidence[reservoir] @ qloop),
        independent_cycles=cycles.shape[1],
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
