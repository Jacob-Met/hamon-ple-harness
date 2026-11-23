"""Steady, fixed-demand Darcy–Weisbach flow in a connected passive water network."""

import math
from bisect import bisect_left
from fractions import Fraction

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


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


def newton(initial, evaluate, tolerance, solve=np.linalg.solve):
    value = initial.copy()
    for _ in range(80):
        residual, jacobian = evaluate(value)
        norm = np.max(np.abs(residual), initial=0.0)
        if norm <= tolerance:
            return value
        change = solve(jacobian, residual)
        for backtrack in range(25):
            candidate = value - change / 2**backtrack
            candidate_norm = np.max(np.abs(evaluate(candidate)[0]), initial=0.0)
            if candidate_norm < norm:
                value = candidate
                break
        else:
            raise ValueError("Newton iteration could not reduce the physical residual")
    raise ValueError("Hydraulic solve did not converge")


def pressure_forest(node_count, endpoints, reservoirs, resistance):
    """Choose the lowest-resistance forest with exactly one reservoir per tree."""
    parents = list(range(node_count))
    owner = np.full(node_count, -1, dtype=int)
    owner[reservoirs] = np.arange(len(reservoirs))

    def component(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    branches = []
    for k in np.argsort(resistance, kind="stable"):
        start, end = endpoints[k]
        left, right = component(start), component(end)
        if left != right and (owner[left] < 0 or owner[right] < 0):
            parents[left] = right
            owner[right] = max(owner[left], owner[right])
            branches.append(k)
    return branches, owner[[component(node) for node in range(node_count)]]


def forest_walk(endpoints, edges, roots, node_count):
    """Orient each selected pipe away from the root of its tree."""
    adjacency = [[] for _ in range(node_count)]
    for edge in edges:
        a, b = endpoints[edge]
        adjacency[a].append((b, edge, 1))
        adjacency[b].append((a, edge, -1))
    visited, order, branches = set(roots), list(roots), []
    for parent in order:
        for child, edge, direction in adjacency[parent]:
            if child not in visited:
                visited.add(child)
                order.append(child)
                branches.append((parent, child, edge, direction))
    if len(order) != node_count:
        raise ValueError("All nodes must belong to one connected network")
    return branches


def single_reservoir_loops(endpoints, root, demands, datum, incidence, law, inverse, scale):
    """Solve continuity in tree-drop coordinates and energy in cycle coordinates."""
    edge_count = len(endpoints)
    basis, _ = pressure_forest(len(demands), endpoints, [root], law(np.zeros(edge_count))[1])
    branches = forest_walk(endpoints, basis, [root], len(demands))
    chords = np.setdiff1d(np.arange(edge_count), basis)
    # A chord needs only the signed drops between its endpoints and their
    # common ancestor. Build these rows without a dense incidence inverse.
    parent = np.zeros(len(demands), dtype=int)
    depth = np.zeros(len(demands), dtype=int)
    column = np.zeros(len(demands), dtype=int)
    sign = np.zeros(len(demands), dtype=int)
    columns = {edge: i for i, edge in enumerate(basis)}
    for p, child, edge, direction in branches:
        parent[child] = p
        depth[child] = depth[p] + 1
        column[child] = columns[edge]
        sign[child] = direction
    row, col, data = list(basis), list(range(len(basis))), [1] * len(basis)
    for edge in chords:
        left, right = endpoints[edge]
        while left != right:
            if depth[left] >= depth[right]:
                row.append(edge)
                col.append(column[left])
                data.append(-sign[left])
                left = parent[left]
            else:
                row.append(edge)
                col.append(column[right])
                data.append(sign[right])
                right = parent[right]
    paths = coo_matrix((data, (row, col)), shape=(edge_count, len(basis))).tocsr()
    walks = paths[chords].tocoo()
    cycles = coo_matrix(
        (np.r_[-walks.data, np.ones(len(chords))],
         (np.r_[np.asarray(basis)[walks.col], chords],
          np.r_[walks.row, np.arange(len(chords))])),
        shape=(edge_count, len(chords)),
    ).tocsr()
    junctions = np.arange(len(demands)) != root
    A = incidence[junctions]
    d = demands[junctions]

    def nodal_equations(tree_drop):
        flow = inverse(paths @ tree_drop, scale)
        jacobian = A.multiply(1 / law(flow)[1]) @ paths
        return A @ flow + d, jacobian

    pressure = newton(np.zeros(len(basis)), nodal_equations, min(1e-12, 5e-13 * np.abs(d).sum()), spsolve)
    qnode = inverse(paths @ pressure, scale)
    # Accumulate rounded drops and demands exactly on the tree. The cut
    # flow starts the independent cycle solve with continuity satisfied.
    heads = [Fraction(float(datum))] * len(demands)
    for p, child, edge, direction in branches:
        heads[child] = heads[p] - direction * Fraction(float(pressure[columns[edge]]))
    cut = [Fraction(float(v)) for v in demands]
    feasible = np.zeros(edge_count)
    for p, child, edge, direction in reversed(branches):
        feasible[edge] = float(direction * cut[child])
        cut[p] += cut[child]
    loss_scale = float(np.max(np.abs(law(feasible)[0])))
    tolerance = max(32 * np.finfo(float).eps * loss_scale, min(1e-13, 1e-13 * loss_scale))

    use_nodal_projection = len(chords) > len(basis)

    def loop_equations(amount):
        drop, slope = law(feasible + cycles @ amount)
        residual = cycles.T @ drop
        if use_nodal_projection:
            return residual, slope
        return residual, cycles.T @ cycles.multiply(slope[:, None])

    def projected_step(slope, residual):
        # C is the cycle basis with C[chords] = I. A vector z supported on
        # the chords with z[chords] = residual therefore has C.T @ z = residual.
        # Project D^-1 z into the continuity nullspace with a sparse nodal
        # solve. Its chord entries solve C.T D C delta = residual exactly
        # in real arithmetic, without constructing that high-rank matrix.
        trial = np.zeros(edge_count)
        trial[chords] = residual / slope[chords]
        conductance = A.multiply(1 / slope)
        correction = spsolve(conductance @ A.T, A @ trial)
        return trial[chords] - (A[:, chords].T @ correction) / slope[chords]

    linear_step = projected_step if use_nodal_projection else spsolve
    amount = newton(np.zeros(len(chords)), loop_equations, tolerance, linear_step)
    qloop = feasible + cycles @ amount
    return qloop, qnode, np.array(list(map(float, heads))), law(qloop)[0], basis


def resolved_laminar_exchange(endpoints, basis, cycles, reservoirs, fixed_heads, demands,
                              length, diameter, viscosity, gravity, affected, flow, drop):
    """Solve the laminar forest equations as exact demand and boundary contributions."""
    from sympy import Matrix, Rational, SparseMatrix, pi
    from sympy.core.evalf import PrecisionExhausted

    # With zero minor losses, pi * head_drop = r * q. All coefficients except
    # pi are rational for the actual binary64 inputs. Keep q = a + b*pi
    # separate until publication, including on a nearly balanced exchange.
    resistance = [128 * Rational(float(viscosity)) * Rational(float(pipe_length))
                  / (Rational(float(gravity)) * Rational(float(pipe_diameter))**4)
                  for pipe_length, pipe_diameter in zip(length, diameter)]
    forest = forest_walk(endpoints, basis, reservoirs, len(demands))
    root_heads = {node: Rational(float(h)) for node, h in zip(reservoirs, fixed_heads)}
    roots = {node: node for node in reservoirs}
    for parent, child, _, _ in forest:
        roots[child] = roots[parent]
        root_heads[child] = root_heads[parent]
    cut = [Rational(float(d)) for d in demands]
    initial = Matrix.zeros(len(endpoints), 1)
    for parent, child, edge, direction in reversed(forest):
        initial[edge] = direction * cut[child]
        cut[parent] += cut[child]
    demand_flow, boundary_flow = initial, Matrix.zeros(len(endpoints), 1)
    if cycles.shape[1]:
        chord = SparseMatrix(cycles.astype(int))
        weighted = chord.T * SparseMatrix.diag(*resistance)
        boundary = Matrix([root_heads[a] - root_heads[b] for a, b in endpoints])
        rhs = (-weighted * initial).row_join(chord.T * boundary)
        # Exact elimination of both right-hand sides shares the factorization.
        matrix = (weighted * chord).to_DM().to_field()
        solution, divisor = matrix.solve_den(rhs.to_DM().convert_to(matrix.domain))
        amplitudes = solution.to_Matrix() / matrix.domain.to_sympy(divisor)
        demand_flow = initial + chord * amplitudes[:, 0]
        boundary_flow = chord * amplitudes[:, 1]

    def rounded(value):
        try:
            return float(value.evalf(25, maxn=768, strict=True))
        except PrecisionExhausted as exc:
            raise ValueError("Could not resolve a near-balanced reservoir exchange") from exc

    flow = flow.copy()
    for edge in np.flatnonzero(affected):
        flow[edge] = rounded(demand_flow[edge] + boundary_flow[edge]*pi)
    heads = dict(root_heads)
    for parent, child, edge, direction in forest:
        loss = (resistance[edge] * (boundary_flow[edge] + demand_flow[edge]/pi)
                if affected[edge] else Rational(float(drop[edge])))
        heads[child] = heads[parent] - direction * loss
    # A reservoir supplies its own tree's exact demands, plus transfers to
    # other trees. Summing its published incident flows can erase a small
    # net supply between individually rounded through-flows.
    supply = {node: cut[node] for node in reservoirs}
    for edge, (a, b) in enumerate(endpoints):
        left, right = roots[a], roots[b]
        if left != right:
            q = (demand_flow[edge] + boundary_flow[edge]*pi if affected[edge]
                 else Rational(float(flow[edge])))
            supply[left] += q
            supply[right] -= q
    return (flow, np.array([rounded(heads[i]) for i in range(len(demands))]),
            np.array([rounded(supply[node]) for node in reservoirs]))


def resolved_minor_exchange(endpoints, basis, cycles, reservoirs, fixed_heads, demands,
                            length, diameter, minor, viscosity, gravity, affected,
                            flow, drop, amplitude, hidden_supply):
    """Refine coupled laminar exchanges with signed quadratic minor losses."""
    from mpmath import mp

    forest = forest_walk(endpoints, basis, reservoirs, len(demands))
    roots = {node: node for node in reservoirs}
    for parent, child, _, _ in forest:
        roots[child] = roots[parent]
    edges = np.flatnonzero(affected)
    ctx = mp.clone()

    def refine(digits):
        ctx.dps = digits

        def exact(value):
            return ctx.mpf(float(value))

        head = {node: exact(h) for node, h in zip(reservoirs, fixed_heads)}
        cut = list(map(exact, demands))
        feasible = [ctx.zero] * len(flow)
        for parent, child, edge, direction in reversed(forest):
            feasible[edge] = direction * cut[child]
            cut[parent] += cut[child]

        chord = ctx.matrix(cycles[edges].astype(int).tolist())
        base = ctx.matrix([feasible[e] for e in edges])
        boundary = ctx.matrix([head[roots[endpoints[e][0]]] - head[roots[endpoints[e][1]]]
                               for e in edges])
        linear = [128 * exact(viscosity) * exact(length[e]) /
                  (ctx.pi * exact(gravity) * exact(diameter[e])**4) for e in edges]
        quadratic = [8 * exact(minor[e]) /
                     (ctx.pi**2 * exact(gravity) * exact(diameter[e])**4) for e in edges]

        # Each selected chord has an identity row, and every laminar slope
        # is positive, so the coupled circulation Jacobian is positive definite.
        x = ctx.matrix(list(map(exact, amplitude)))
        for _ in range(40):
            q = base + chord * x
            loss = ctx.matrix([v * (r + k*abs(v)) for v, r, k in zip(q, linear, quadratic)])
            slopes = [r + 2*k*abs(v) for v, r, k in zip(q, linear, quadratic)]
            residual = chord.T * (loss - boundary)
            step = ctx.lu_solve(chord.T * ctx.diag(slopes) * chord, residual)
            x -= step
            if ctx.norm(step, 'inf') <= ctx.eps * max(ctx.norm(x, 'inf'), ctx.norm(base, 'inf')):
                break
        else:
            raise ValueError("Could not refine a coupled minor-loss exchange")

        q = base + chord * x
        refined = dict(zip(edges, q))
        losses = {e: v*(r + k*abs(v)) for e, v, r, k in zip(edges, q, linear, quadratic)}
        for parent, child, edge, direction in forest:
            head[child] = head[parent] - direction * losses.get(edge, exact(drop[edge]))

        supply = {node: cut[node] for node in reservoirs}
        for edge, (a, b) in enumerate(endpoints):
            left, right = roots[a], roots[b]
            if left != right:
                v = refined.get(edge, exact(flow[edge]))
                supply[left] += v
                supply[right] -= v
        return (tuple(map(float, q)), tuple(float(head[i]) for i in range(len(demands))),
                tuple(float(supply[node]) for node in reservoirs))

    previous = None
    for digits in (50, 100, 200, 400):
        result = refine(digits)
        # A zero in a selected circulation or a cancelling boundary supply
        # can mask a contribution below the precision used. Leave it unresolved.
        supplies_resolved = all(v for v, hidden in zip(result[2], hidden_supply) if hidden)
        if result == previous and all(result[0]) and supplies_resolved:
            flow = flow.copy()
            flow[edges] = result[0]
            return flow, np.array(result[1]), np.array(result[2])
        previous = result
    raise ValueError("Could not resolve rounding on a coupled minor-loss exchange")


def radial_solution(endpoints, root, demands, datum, law, inverse, re_per_flow):
    """A tree edge supplies exactly the sum of demands in the cut below it."""
    branches = forest_walk(endpoints, range(len(endpoints)), [root], len(demands))
    cut = [Fraction(float(value)) for value in demands]
    flow = np.empty(len(endpoints))
    for parent, child, edge, direction in reversed(branches):
        flow[edge] = float(direction * cut[child])
        cut[parent] += cut[child]
    drop = law(flow)[0]
    # Preserve the individual pipe drops when the datum or accumulated
    # upstream loss is much larger than the local pressure difference.
    heads = [Fraction(float(datum))] * len(demands)
    for parent, child, edge, direction in branches:
        heads[child] = heads[parent] - direction * Fraction(float(drop[edge]))
    differences = np.array([float(heads[a] - heads[b]) for a, b in endpoints])
    return flow, inverse(differences, 2000 / re_per_flow), np.array(list(map(float, heads))), drop


def radial_balance(endpoints, root, demands, flows):
    """Replay mass balance from the reported binary64 values at every junction."""
    load = [Fraction(float(d)) for d in demands]
    error = load.copy()
    rounding = [Fraction(0)] * len(load)
    for (a, b), flow in zip(endpoints, flows):
        q = Fraction(float(flow))
        error[a] += q
        error[b] -= q
        half_ulp = Fraction(math.ulp(float(flow))) / 2 if q else 0
        rounding[a] += half_ulp
        rounding[b] += half_ulp
    # The root cut is the net demand exactly. Its incident flows are rounded
    # separately and can lose a small net supply between opposing branches.
    supply = sum(load)
    error[root] -= supply
    verified = True
    for i, mismatch in enumerate(error):
        demand = supply if i == root else load[i]
        tolerance = abs(demand) / 10**7 + rounding[i]
        if abs(mismatch) > min(Fraction(1, 10**6), tolerance):
            verified = False
    return float(max(map(abs, error))), verified, float(supply)



def net_reservoir_supply(endpoints, junctions, demands, flows):
    """Return net fixed-head supply and replay its boundary flux after rounding."""
    demand = sum(Fraction(float(value)) for value in demands)
    flux = rounding = Fraction(0)
    for (a, b), value in zip(endpoints, flows):
        # Direct reservoir exchanges do not change net supply.
        direction = int(junctions[b]) - int(junctions[a])
        if direction:
            q = float(value)
            flux += direction * Fraction(q)
            rounding += Fraction(math.ulp(q)) / 2
    tolerance = min(Fraction(1, 10**6), abs(demand) / 10**7 + rounding)
    return float(demand), abs(flux - demand) <= tolerance


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
    gravity = number(spec.get("gravity_m_s2", 9.80665))
    if gravity <= 0:
        raise ValueError("Gravity must be positive")
    index = {name: i for i, name in enumerate(node_ids)}
    endpoints = [(index[p["from"]], index[p["to"]]) for p in pipes]
    if any(a == b for a, b in endpoints):
        raise ValueError("Pipe cannot connect a node to itself")
    radial = len(reservoirs) == 1 and len(pipes) == len(nodes) - 1
    if len(reservoirs) == 1:
        # Both single-reservoir formulations check connectivity by traversing the tree.
        row = np.asarray(endpoints).ravel()
        col = np.repeat(np.arange(len(pipes)), 2)
        signs = np.tile([1., -1.], len(pipes))
        incidence = coo_matrix((signs, (row, col)), shape=(len(nodes), len(pipes))).tocsr()
    else:
        incidence = np.zeros((len(nodes), len(pipes)))
        for k, (a, b) in enumerate(endpoints):
            incidence[a, k], incidence[b, k] = 1.0, -1.0
        if np.linalg.matrix_rank(incidence[1:]) != len(nodes) - 1:
            raise ValueError("All nodes must belong to one connected network")
    junctions = np.ones(len(nodes), dtype=bool)
    junctions[reservoirs] = False
    A = incidence[junctions]
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
    law, inverse, re_per_flow = pipe_law(length, diameter, roughness, minor, viscosity, gravity)
    d = demands[junctions]
    scale = max(float(np.abs(d).sum()), 1e-12)

    resolved_supply = None
    if radial:
        qloop, qnode, total_head, drop = radial_solution(
            endpoints, reservoirs[0], demands, datum, law, inverse, re_per_flow)
    elif len(reservoirs) == 1:
        qloop, qnode, total_head, drop, basis = single_reservoir_loops(
            endpoints, reservoirs[0], demands, datum, incidence, law, inverse, scale)
    else:
        # Use pressure drops on a forest rooted at the reservoirs as continuous
        # coordinates. Prefer low-resistance pipes so a small pressure gradient
        # survives even when both endpoint heads are far from the datum.
        initial = np.zeros(len(d))
        basis, roots = pressure_forest(len(nodes), endpoints, reservoirs, law(np.zeros(len(pipes)))[1])
        basis_incidence = A[:, basis]
        # Each pipe drop is a signed sum along a forest path, plus the
        # prescribed head difference between roots if it connects two trees.
        paths = np.linalg.solve(basis_incidence, A).T
        root_head = fixed_heads[roots]
        fixed_drop = np.array([root_head[a] - root_head[b] for a, b in endpoints])

        def pipe_drops(tree_drop):
            return paths @ tree_drop + fixed_drop

        if np.any(fixed_drop):
            # The harmonic initial head includes imposed reservoir through-flow.
            initial = np.linalg.lstsq(paths, -fixed_drop, rcond=None)[0]
            initial_flow = inverse(pipe_drops(initial), scale)
            scale = max(scale, float(np.max(np.abs(initial_flow))))

        def nodal_equations(tree_drop):
            flow = inverse(pipe_drops(tree_drop), scale)
            residual = A @ flow + d
            derivative = A / law(flow)[1]
            return residual, derivative @ paths

        tree_drop = newton(initial, nodal_equations, min(1e-12, 5e-13 * scale))
        qnode = inverse(pipe_drops(tree_drop), scale)
        head = np.linalg.solve(basis_incidence.T, tree_drop)
        # The second formulation begins with continuity satisfied, and may change
        # only circulations and flows between reservoirs. Their prescribed head
        # differences supply the energy term on those boundary-to-boundary paths.
        chords = np.setdiff1d(np.arange(len(pipes)), basis)
        feasible = np.zeros(len(pipes))
        feasible[basis] = np.linalg.solve(basis_incidence, -d)
        cycles = np.zeros((len(pipes), len(chords)))
        cycles[basis, :] = -paths[chords, :].T
        cycles[chords, np.arange(len(chords))] = 1.0

        initial_loss = np.abs(law(feasible)[0])
        # Each exchange uses its own root-to-root head scale. The nodal
        # estimate also accounts for a loop driven by other exchanges.
        path_loss = np.maximum(initial_loss, np.abs(law(qnode)[0]))
        path_scale = np.abs(cycles).T @ path_loss + np.abs(fixed_drop[chords])
        path_scale = np.where(path_scale > 0, path_scale, 1.)
        loop_tolerance = 1e-13

        def loop_equations(amount):
            drop, slope = law(feasible + cycles @ amount)
            error = cycles.T @ drop - fixed_drop[chords]
            jacobian = (cycles.T * slope) @ cycles
            return error / path_scale, jacobian / path_scale[:, None]

        amplitude = newton(np.zeros(len(chords)), loop_equations, loop_tolerance)
        qloop = feasible + cycles @ amplitude
        drop, _ = law(qloop)
        total_head = root_head.copy()
        total_head[junctions] += head
        # Nearly balanced reservoirs can cancel either a chord's driving
        # head or a tree pipe's demand and exchange flows.
        guard = math.sqrt(np.finfo(float).eps)
        path_drop = np.abs(cycles).T @ np.abs(drop) + np.abs(fixed_drop[chords])
        hidden_drop = np.abs(drop[chords]) < guard * path_drop
        path_flow = np.abs(feasible) + np.abs(cycles) @ np.abs(amplitude)
        hidden_flow = np.abs(qloop[basis]) < guard * path_flow[basis]
        root_incidence = incidence[reservoirs]
        root_flow = root_incidence @ qloop
        hidden_supply = np.abs(root_flow) < guard * (np.abs(root_incidence) @ np.abs(qloop))
        if np.any(hidden_drop) or np.any(hidden_flow) or np.any(hidden_supply):
            # Exchanges are coupled only when their forest paths share a
            # pipe. Independent nonlinear branches keep their own law.
            support = cycles != 0
            selected = hidden_drop | np.any(support[np.asarray(basis, dtype=int)[hidden_flow]], axis=0)
            incident = np.any(root_incidence[hidden_supply] != 0, axis=0)
            selected |= np.any(support[incident], axis=0)
            while True:
                affected = np.any(support[:, selected], axis=1)
                neighboring = np.any(support[affected], axis=0)
                if np.array_equal(neighboring, selected):
                    break
                selected = neighboring
            reynolds = re_per_flow * np.maximum(np.abs(qnode), np.abs(qloop))
            if np.any(reynolds[affected] >= 2000):
                raise ValueError("Near-balanced reservoir exchange is unresolved by transitional or turbulent losses")
            if np.any(minor[affected]):
                qloop, total_head, resolved_supply = resolved_minor_exchange(
                    endpoints, basis, cycles[:, selected], reservoirs, fixed_heads, demands,
                    length, diameter, minor, viscosity, gravity, affected, qloop, drop,
                    amplitude[selected], hidden_supply)
            else:
                qloop, total_head, resolved_supply = resolved_laminar_exchange(
                    endpoints, basis, cycles[:, selected], reservoirs, fixed_heads, demands,
                    length, diameter, viscosity, gravity, affected, qloop, drop)
            if np.any((re_per_flow * np.abs(qloop))[affected] >= 2000):
                raise ValueError("Resolved flow exceeds the laminar regime")
            drop = law(qloop)[0]
    if radial:
        continuity, local_mass_ok, root_supply = radial_balance(endpoints, reservoirs[0], demands, qloop)
        supply = np.array([root_supply])
    else:
        local_mass_ok = True
        supply = incidence[reservoirs] @ qloop if resolved_supply is None else resolved_supply
        continuity = float(np.max(np.abs(A @ qloop + d), initial=0.0))
    if len(reservoirs) > 1:
        net_supply, local_mass_ok = net_reservoir_supply(endpoints, junctions, d, qloop)
    else:
        net_supply = float(supply.sum())
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
        pipe_scale = np.abs(qloop).copy()
        if not radial and len(reservoirs) == 1:
            pipe_scale[basis] = scale
        pipe_tolerance = np.minimum(1e-6, 1e-7 * np.abs(qloop) + 32 * eps * pipe_scale)
        registered["flows_match"] = bool(
            np.all(np.abs(flows - qloop) <= pipe_tolerance)
        )
        if radial:
            _, registered["junction_demands_match"], _ = radial_balance(endpoints, reservoirs[0], demands, flows)
        else:
            incident_flow = np.abs(A) @ np.abs(flows)
            mass_tolerance = np.minimum(
                1e-6, 1e-7 * np.abs(d) + 32 * eps * (incident_flow + np.abs(d))
            )
            registered["junction_demands_match"] = bool(np.all(np.abs(A @ flows + d) <= mass_tolerance))
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
    local_flow_ok = (not radial or np.all(np.abs(qnode - qloop) <= np.minimum(1e-6, 1e-7*np.abs(qloop))))
    verified = (
        local_flow_ok and local_mass_ok and continuity < min(1e-6, 1e-7 * scale)
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
        reservoir_supply_m3_s=net_supply,
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


def pump_curve(curve):
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
    return flows, heads


def _duty_quantity(result, key, value):
    """Publish a rounded display value without changing the exact decision."""
    try:
        rounded = float(value)
    except OverflowError:
        rounded = math.copysign(math.inf, 1 if value > 0 else -1)
    if math.isfinite(rounded) and (rounded or not value):
        result[key] = rounded
    else:
        result[key] = None
        result.setdefault("numeric_range", {})[key] = (
            "below_binary64" if rounded == 0 else "above_binary64"
        )


def verify_pump_duty(spec):
    """Check a tabulated fixed-speed duty and its declared power constraints.

    Electrical inputs are connected ratings. With no measured wire-to-fluid
    efficiency, the ideal first-law limit is a necessary lower bound on draw.
    No density is assumed. At positive hydraulic duty, a rating yields an
    inverse limit on density even when the liquid has not been identified.
    """
    result = {
        "calculation": "pump_duty",
        "mode": "thermal_fluid",
        "tool": "hamon_conservation_cad_verifier",
        "ok": False,
        "duty_verified": False,
    }
    try:
        fields = {
            "mode",
            "calculation",
            "curve",
            "total_flow_m3_s",
            "required_head_m",
            "identical_pumps",
            "per_pump_input_W",
            "available_panel_W",
            "equipped_branch",
            "gravity_m_s2",
            "fluid_density_kg_m3",
            "wire_to_fluid_efficiency",
        }
        if spec.keys() - fields:
            raise ValueError("Unsupported inputs for fixed-speed identical parallel pump duty")
        count = spec.get("identical_pumps", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("identical_pumps must be a positive integer")
        flow = Fraction(number(spec["total_flow_m3_s"]))
        if flow < 0:
            raise ValueError("Expected nonnegative design flow")
        per_pump = flow / count
        _duty_quantity(result, "flow_per_pump_m3_s", per_pump)
        if ("required_head_m" in spec) == ("equipped_branch" in spec):
            raise ValueError("Supply exactly one of required_head_m and equipped_branch")
        branch = spec.get("equipped_branch", {})
        if not isinstance(branch, dict):
            raise ValueError("equipped_branch must be a hydraulic branch object")
        gravity = number(spec.get("gravity_m_s2", branch.get("gravity_m_s2", 9.80665)))
        if gravity <= 0 or ("gravity_m_s2" in branch and number(branch["gravity_m_s2"]) != gravity):
            raise ValueError("Use one positive site gravity for head losses and power")
        if "equipped_branch" in spec:
            result.update(
                equipped_branch_head({**branch, "gravity_m_s2": gravity}, float(per_pump))
            )
            required = Fraction(result["required_head_m"])
        else:
            required = Fraction(number(spec["required_head_m"]))
            if required < 0:
                raise ValueError("Expected nonnegative required head")

        density = None
        if "fluid_density_kg_m3" in spec:
            density = Fraction(number(spec["fluid_density_kg_m3"]))
            if density <= 0:
                raise ValueError("fluid_density_kg_m3 must be positive")
        efficiency = Fraction(number(spec.get("wire_to_fluid_efficiency", 1)))
        if not 0 < efficiency <= 1:
            raise ValueError("wire_to_fluid_efficiency must lie in (0, 1]")
        rating = None
        checks = {}
        if "per_pump_input_W" in spec or "available_panel_W" in spec:
            rating = Fraction(number(spec["per_pump_input_W"]))
            capacity = Fraction(number(spec["available_panel_W"]))
            if min(rating, capacity) < 0:
                raise ValueError("Electrical inputs must be nonnegative")
            total_rating = count * rating
            _duty_quantity(result, "total_input_W", total_rating)
            _duty_quantity(result, "panel_margin_W", capacity - total_rating)
            checks["panel_adequate"] = total_rating <= capacity
        if density is None and rating is None and "wire_to_fluid_efficiency" in spec:
            raise ValueError("Efficiency needs a fluid density or a connected rating")

        flows, heads = pump_curve(spec["curve"])
        if not flows[0] <= per_pump <= flows[-1]:
            raise ValueError("Flow per pump is outside the tabulated curve")
        i = min(bisect_left(flows, per_pump, lo=1), len(flows) - 1)
        q0, q1 = map(Fraction, flows[i - 1 : i + 1])
        h0, h1 = map(Fraction, heads[i - 1 : i + 1])
        head = h0 + (h1 - h0) * (per_pump - q0) / (q1 - q0)
        _duty_quantity(result, "available_head_m", head)
        _duty_quantity(result, "head_margin_m", head - required)
        checks["head_adequate"] = head >= required

        # The pump develops the tabulated head at the stated discharge even
        # when the stated minimum required head is smaller.
        specific_power = Fraction(gravity) * per_pump * head
        if density is not None or rating is not None:
            result["power_basis"] = (
                "declared_wire_to_fluid_efficiency"
                if "wire_to_fluid_efficiency" in spec
                else "ideal_lower_bound"
            )
            result["hydraulic_rating_verified"] = None
        if rating is not None and specific_power:
            _duty_quantity(
                result, "max_fluid_density_at_rating_kg_m3", efficiency * rating / specific_power
            )
            if rating == 0:
                checks["connected_power_consistent"] = False
                result["hydraulic_rating_verified"] = False
                result["reason"] = (
                    "Positive flow and developed head require positive connected power"
                )
        if density is not None:
            hydraulic = density * specific_power
            electrical = hydraulic / efficiency
            _duty_quantity(result, "total_hydraulic_power_W", count * hydraulic)
            _duty_quantity(result, "required_input_per_pump_W", electrical)
            _duty_quantity(result, "required_total_input_W", count * electrical)
            if rating is not None:
                _duty_quantity(result, "per_pump_power_margin_W", rating - electrical)
                enough = rating >= electrical
                checks["connected_power_consistent"] = enough
                # Passing an ideal bound cannot establish real efficiency.
                if not enough or "wire_to_fluid_efficiency" in spec:
                    result["hydraulic_rating_verified"] = enough
        return {**result, "ok": True, "checks": checks, "duty_verified": all(checks.values())}
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return {**result, "reason": str(exc)}


def _isolate_pump_root(residual, low, high, a, slope, constant, coordinates):
    """Enclose one crossing until each reported quantity rounds unambiguously."""
    if not a:
        return constant / slope

    # A rational root can coincide with a rounding midpoint in any coordinate.
    # Compute it exactly so that ties round to even without indefinite bisection.
    discriminant = slope**2 - 4*a*constant
    num = math.isqrt(discriminant.numerator)
    den = math.isqrt(discriminant.denominator)
    if num*num == discriminant.numerator and den*den == discriminant.denominator:
        radical = Fraction(num, den)
        smaller = (slope - radical) / (2*a)
        return smaller if low < smaller < high else (slope + radical) / (2*a)

    def rounded(point):
        values = []
        for value in coordinates(point):
            try:
                values.append(float(value))
            except OverflowError:
                # A wide initial bracket may extend beyond the output range.
                values.append(math.inf)
        return values

    low_positive = residual(low) > 0
    while rounded(low) != rounded(high):
        mid = (low + high) / 2
        if (residual(mid) > 0) == low_positive:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _pump_output(value):
    result = float(value)
    if result == 0 and value:
        raise ValueError("A nonzero operating point value is below the output range")
    return result


def _heterogeneous_pump_bank(spec):
    """Independent decreasing pump curves on a common header with check valves."""
    allowed = {"mode", "calculation", "pump_curves", "check_valves",
               "static_head_m", "system_resistance_s2_m5"}
    if spec.keys() - allowed or spec.get("check_valves", True) is not True:
        raise ValueError("Independent parallel curves require a check valve on each pump")
    tables = spec["pump_curves"]
    if not isinstance(tables, list) or not tables:
        raise ValueError("Supply at least one pump curve")
    static = Fraction(number(spec["static_head_m"]))
    resistance = Fraction(number(spec["system_resistance_s2_m5"]))
    if resistance < 0:
        raise ValueError("System curve resistance must be nonnegative")

    curves = []
    for table in tables:
        q, h = pump_curve(table)
        if q[0] != 0 or any(left <= right for left, right in zip(h, h[1:])):
            raise ValueError("Each independent curve must begin at shutoff and decrease strictly in head")
        curves.append((list(map(Fraction, reversed(h))), list(map(Fraction, reversed(q)))))

    def rates(head):
        flows = []
        for heads, discharges in curves:
            if head >= heads[-1]:
                flows.append(Fraction(0))
                continue
            j = bisect_left(heads, head, lo=1)
            t = (head-heads[j-1])/(heads[j]-heads[j-1])
            flows.append(discharges[j-1]+t*(discharges[j]-discharges[j-1]))
        return flows

    def residual(head):
        return head-static-resistance*sum(rates(head))**2

    # Below this head at least one open pump would exceed its final rated
    # flow. Above every shutoff the closed bank can still support zero flow.
    floor = max(h[0] for h, _ in curves)
    ceiling = max(static, max(h[-1] for h, _ in curves))
    knots = sorted({floor, ceiling} | {x for heads, _ in curves for x in heads if x >= floor})
    base = {"ok": True, "continuous_total_flow_intervals_m3_s": []}
    if residual(floor) > 0:
        return dict(base, has_rated_intersection=False, operating_points=[],
                    reason="Common-header intersection would require pump-curve extrapolation")

    # Q(H) is continuous, nonincreasing and piecewise affine. The system
    # residual is strictly increasing, so exactly one interval can contain it.
    j = bisect_left(knots, 0, key=residual)
    if residual(knots[j]) == 0:
        head = knots[j]
        flows = rates(head)
    else:
        lower, upper = knots[j-1:j+1]
        flow_low, flow_high = rates(lower), rates(upper)
        total_low, total_high = sum(flow_low), sum(flow_high)
        slope = (upper-lower)/(total_high-total_low)
        intercept = lower-slope*total_low
        constant = static-intercept

        def at_total(total):
            h = intercept+slope*total
            t = (total-total_high)/(total_low-total_high)
            qs = [b+t*(a-b) for a,b in zip(flow_low,flow_high)]
            return h, qs

        def coordinates(total):
            h, qs = at_total(total)
            return (total, h, *qs, *(heads[-1]-h for heads, _ in curves))

        total = _isolate_pump_root(lambda q: (resistance*q-slope)*q+constant,
                                   total_high, total_low, resistance, slope, constant, coordinates)
        head, flows = at_total(total)

    units = []
    for i, ((heads, _), flow) in enumerate(zip(curves, flows)):
        margin = heads[-1]-head
        state = "open" if flow else "closed" if margin < 0 else "at_shutoff"
        units.append({"pump_index": i, "flow_m3_s": _pump_output(flow),
                      "check_valve_state": state, "shutoff_head_minus_header_m": _pump_output(margin)})
    point = {"head_m": _pump_output(head), "total_flow_m3_s": _pump_output(sum(flows)),
             "delivering_flow": any(flows), "pumps": units}
    return dict(base, has_rated_intersection=True, operating_points=[point])


def verify_pump_operating_point(spec):
    """All rated intersections of a fixed-speed pump bank with Hstatic + K Q².

    Q is total flow; each identical parallel pump carries Q/count. Curve
    segments may rise. A flat coincident segment is reported as an interval.
    """
    result = {"calculation": "pump_operating_point", "mode": "thermal_fluid",
              "tool": "hamon_conservation_cad_verifier", "ok": False}
    try:
        if "pump_curves" in spec:
            return {**result, **_heterogeneous_pump_bank(spec)}
        allowed = {"mode", "calculation", "curve", "identical_pumps", "static_head_m",
                   "system_resistance_s2_m5"}
        if spec.keys() - allowed:
            raise ValueError("Unsupported inputs for a quadratic pump-system operating point")
        count = spec.get("identical_pumps", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("identical_pumps must be a positive integer")
        static = Fraction(number(spec["static_head_m"]))
        resistance = Fraction(number(spec["system_resistance_s2_m5"]))
        if resistance < 0:
            raise ValueError("System curve resistance must be nonnegative")
        flows, heads = pump_curve(spec["curve"])
        points = list(zip(map(Fraction, flows), map(Fraction, heads)))
        roots, intervals = set(), []
        a = resistance * count**2
        for (left, h0), (right, h1) in zip(points, points[1:]):
            slope = (h1 - h0) / (right - left)
            constant = static - h0 + slope * left

            def residual(q):
                return (a*q - slope)*q + constant

            if a == 0 and slope == 0 and constant == 0:
                if intervals and intervals[-1][1] == left:
                    intervals[-1] = (intervals[-1][0], right)
                else:
                    intervals.append((left, right))
                continue
            def coordinates(q):
                return q, count*q, h0 + slope*(q-left)

            boundaries = [left, right]
            if a and left < slope/(2*a) < right:
                boundaries.insert(1, slope/(2*a))
            for lo, hi in zip(boundaries, boundaries[1:]):
                flo, fhi = residual(lo), residual(hi)
                if flo == 0:
                    roots.add(lo)
                if fhi == 0:
                    roots.add(hi)
                if flo * fhi < 0:
                    roots.add(_isolate_pump_root(residual, lo, hi, a, slope, constant, coordinates))

        # Only shared table knots can occur in two adjacent segments.
        for lo, hi in intervals:
            roots.discard(lo)
            roots.discard(hi)
        operating_points = []
        for q in sorted(roots):
            total = count*q
            j = min(bisect_left(flows, q, lo=1), len(flows)-1)
            (q0, h0), (q1, h1) = points[j-1:j+1]
            head = h0 + (h1-h0)*(q-q0)/(q1-q0)
            operating_points.append({"flow_per_pump_m3_s": _pump_output(q),
                                     "total_flow_m3_s": _pump_output(total),
                                     "head_m": _pump_output(head)})
        ranges = [[_pump_output(count*lo), _pump_output(count*hi)] for lo, hi in intervals]
        return {**result, "ok": True, "has_rated_intersection": bool(operating_points or ranges),
                "operating_points": operating_points,
                "continuous_total_flow_intervals_m3_s": ranges}
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return {**result, "reason": str(exc)}
