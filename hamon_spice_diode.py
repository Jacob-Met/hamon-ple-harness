"""Charge-free Shockley junctions in passive, constant-temperature SPICE networks."""
from collections import deque
from fractions import Fraction
import math
import re

import numpy as np


def configure_junctions(branches, lines, parse_number):
    """Resolve diode model/area at its nominal temperature; refuse unsupported physics."""
    literal = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[a-zA-Z]*")
    def number(text):
        if not literal.fullmatch(text):
            raise ValueError(f"Expected a literal SPICE value, got {text!r}")
        value = parse_number(text)
        if not math.isfinite(value):
            raise ValueError("Nonfinite junction model or circuit value")
        return value

    def options(text, allowed):
        parsed = {}
        for word in re.sub(r"\s*=\s*", "=", text.strip()).replace(",", " ").split():
            key, sep, value = word.partition("=")
            key = key.casefold()
            if not sep or key not in allowed or key in parsed:
                raise ValueError(f"Unsupported or duplicate junction option {word!r}")
            parsed[key] = number(value)
        return parsed

    models = {}
    temperatures = {}
    for line in lines:
        if "*" in line:
            raise ValueError("Expressions are unsupported in diode circuit cards")
        toks = line.split()
        tag = toks[0].casefold()
        if tag == ".model":
            match = re.fullmatch(r"\.model\s+(\S+)\s+d(?:\s*\(([^()]*)\)|\s+(.*))?", line, re.I)
            if not match:
                raise ValueError("Only ideal junction diode models are supported in diode decks")
            name = match[1].casefold()
            if name in models:
                raise ValueError("Ambiguous diode model name")
            models[name] = options(match[2] or match[3] or "", {"is", "n", "tnom"})
        elif tag in {".temp", ".options"}:
            if tag == ".temp" and len(toks) != 2:
                raise ValueError("One operating temperature is required")
            opts = {"temp": number(toks[1])} if tag == ".temp" else options(
                " ".join(toks[1:]), {"temp", "tnom"})
            if temperatures.keys() & opts.keys():
                raise ValueError("Duplicate operating or nominal temperature")
            temperatures.update(opts)
        elif tag in {".ac", ".op", ".end", ".title", ".save", ".print", ".plot"}:
            continue
        elif tag[0] in "rlc":
            if len(toks) != 4 or number(toks[3]) <= 0:
                raise ValueError("Diode circuits require passive constant R/L/C values")
        elif tag[0] not in "vid":
            raise ValueError("Diode bias supports independent sources and passive R/L/C/junctions")

    temperature = temperatures.get("temp", 27.)
    kelvin = temperature + 273.15
    if not math.isfinite(kelvin) or kelvin <= 0:
        raise ValueError("Junction temperature must be finite and above absolute zero")
    names = set()
    canonical = {}
    for br in branches:
        name = br["name"].casefold()
        if name in names:
            raise ValueError("Ambiguous device name in diode circuit")
        names.add(name)
        for key in ("n_pos", "n_neg"):
            node = br[key].casefold()
            br[key] = "0" if node in {"0", "gnd"} else canonical.setdefault(node, br[key])
        if br["type"] != "D":
            continue
        model = models.get(br["model"].casefold())
        if model is None:
            raise ValueError(f"Unknown diode model {br['model']!r}")
        if model.get("tnom", temperatures.get("tnom", 27.)) != temperature:
            raise ValueError("Temperature-dependent model extrapolation is unsupported")
        area, saturation = number(br["area"]), model.get("is", 1e-14)
        if area <= 0 or saturation <= 0:
            raise ValueError("Positive finite junction area and saturation current required")
        br["value"] = saturation * area
        br["thermal"] = model.get("n", 1.) * (1.380649e-23 / 1.602176634e-19) * kelvin
        if not all(math.isfinite(v) and v > 0 for v in (br["value"], br["thermal"])):
            raise ValueError("Positive finite junction area, saturation current and ideality required")


def _finite_bias_possible(branches):
    """Test whether KCL permits I_d + Is > 0 on every diode between linear components."""
    parents = {}
    def root(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for br in branches:
        a, b = root(br["n_pos"]), root(br["n_neg"])
        if br["type"] in {"R", "V", "L"}:
            parents[a] = b
    demand = {root(n): Fraction(0) for n in parents}
    edges = []
    for br in branches:
        a, b = root(br["n_pos"]), root(br["n_neg"])
        if a == b or br["type"] not in {"I", "D"}:
            continue
        if br["type"] == "D":
            value = Fraction(br["value"])
            edges.append((a, b))
        else:
            value = -Fraction(br["value"])
        demand[a] += value
        demand[b] -= value

    connected = {u: set() for u in demand}
    for a, b in edges:
        connected[a].add(b)
        connected[b].add(a)
    if "0" not in parents:
        raise ValueError("Diode circuit needs a DC reference")
    seen, pending = {root("0")}, [root("0")]
    while pending:
        u = pending.pop()
        for v in connected[u] - seen:
            seen.add(v)
            pending.append(v)
    if len(seen) != len(demand):
        raise ValueError("Diode circuit has a floating DC component")

    # Incidence cut bounds are integer multiples of the common binary quantum.
    # A smaller lower bound on every edge is equivalent to strict feasibility.
    quantum = max((v.denominator for v in demand.values()), default=1)
    epsilon = Fraction(1, quantum * (len(edges) + 1))
    for a, b in edges:
        demand[a] -= epsilon
        demand[b] += epsilon
    total = sum(v for v in demand.values() if v > 0)
    if not total:
        return
    source, sink = object(), object()
    graph = {u: {} for u in (*demand, source, sink)}
    def edge(a, b, cap):
        graph[a][b] = graph[a].get(b, 0) + cap
        graph[b].setdefault(a, 0)
    for a, b in edges:
        edge(a, b, total)
    for node, value in demand.items():
        if value > 0:
            edge(source, node, value)
        elif value < 0:
            edge(node, sink, -value)
    while total:
        previous = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            u = queue.popleft()
            for v, capacity in graph[u].items():
                if capacity and v not in previous:
                    previous[v] = u
                    queue.append(v)
        if sink not in previous:
            raise ValueError("No finite junction operating point satisfies the imposed DC currents")
        v, amount = sink, total
        while v is not source:
            u = previous[v]
            amount = min(amount, graph[u][v])
            v = u
        v = sink
        while v is not source:
            u = previous[v]
            graph[u][v] -= amount
            graph[v][u] += amount
            v = u
        total -= amount


def junction_law(br, voltage):
    ratio = voltage / br["thermal"]
    try:
        current = br["value"] * math.expm1(ratio)
        conductance = br["value"] * math.exp(ratio) / br["thermal"]
    except OverflowError as exc:
        raise ValueError("Junction state exceeds representable range") from exc
    if not math.isfinite(current) or not math.isfinite(conductance) or conductance <= 0:
        raise ValueError("Junction current or incremental conductance exceeds representable range")
    return current, conductance


def precise_linear(matrix, inputs, corrections=None):
    """Solve original branch stamps without aggregating away weak conductance."""
    import sympy as sp
    from mpmath import mp

    a, b, _, _ = matrix(*inputs, exact=True)
    if corrections is not None:
        b = sp.Matrix([sp.Rational(float(v)) for v in corrections])
    reduced, pivots = a.row_join(b).to_DM().rref()
    if tuple(pivots) != tuple(range(a.rows)):
        raise ValueError("Undetermined junction circuit")
    exact = list(reduced.to_Matrix()[:, -1])
    rounded = np.array([complex(v.evalf(40)) for v in exact])
    if not np.all(np.isfinite(rounded)):
        raise ValueError("Unrepresentable junction circuit solution")
    # Keep enough bits to distinguish the individual rational coefficients
    # already combined in an entry. An independent LU checks the solution.
    ctx = mp.clone()
    parts = [sp.Rational(v) for x in a for v in x.as_real_imag() if v]
    ctx.prec = 100 + max((max(abs(int(v.p)).bit_length(), int(v.q).bit_length())
                          for v in parts), default=1)
    def convert(x):
        a, b = map(sp.Rational, x.as_real_imag())
        return ctx.mpc(ctx.mpf(int(a.p))/int(a.q), ctx.mpf(int(b.p))/int(b.q))
    numeric = ctx.matrix([[convert(a[i,j]) for j in range(a.cols)] for i in range(a.rows)])
    checked = ctx.lu_solve(numeric, ctx.matrix([convert(v) for v in b]))
    again = np.array([complex(v) for v in checked])
    discrepancy = float(np.linalg.norm(again-rounded) / max(np.linalg.norm(rounded), 1e-30))
    return rounded, exact, discrepancy


def solve_bias(branches, n_idx, matrix, second_solve):
    """Damped nodal Newton solve, replayed with original currents and source constraints."""
    dc = [dict(br) for br in branches]
    for br in dc:
        if br["type"] in {"I", "V"}:
            br.update(value=br["dc_value"], phase_deg=0.)
    _finite_bias_possible(dc)
    constraints = [i for i, br in enumerate(dc) if br["type"] in {"L", "V"}]
    rows = {index: len(n_idx) + offset for offset, index in enumerate(constraints)}
    base, rhs, _, _ = matrix(dc, 0., n_idx, rows, {}, {})
    base, rhs = base.real, rhs.real
    devices = [br for br in dc if br["type"] == "D"]
    ports = np.zeros((len(rhs), len(devices)))
    for i, br in enumerate(devices):
        for key, sign in (("n_pos", 1), ("n_neg", -1)):
            if br[key] in n_idx:
                ports[n_idx[br[key]], i] += sign
    thermal = np.array([br["thermal"] for br in devices])
    fixed_currents = [[] for _ in n_idx]
    for br in dc:
        if br["type"] == "I":
            for key, sign in (("n_pos", 1), ("n_neg", -1)):
                if br[key] in n_idx:
                    fixed_currents[n_idx[br[key]]].append(sign * br["value"])
    state = np.zeros(len(rhs))
    for _ in range(256):
        volts = ports.T @ state
        laws = [junction_law(br, float(v)) for br, v in zip(devices, volts)]
        current, slope = np.array(laws).T
        # Keep the reverse leakage and the smaller exponential contribution
        # separate in KCL. A current one ULP above -Is determines a unique
        # voltage even though rounding the total diode current erases its slope.
        terms = [[*(fixed_currents[j] if j < len(n_idx) else [-float(rhs[j])]),
                  *(float(a)*float(x) for a, x in zip(row, state) if a)]
                 for j, row in enumerate(base)]
        for k, (br, v) in enumerate(zip(devices, volts)):
            ratio = v / br["thermal"]
            parts = (-br["value"], br["value"] * math.exp(ratio)) if ratio < -1 else (current[k],)
            for j in np.flatnonzero(ports[:, k]):
                terms[j].extend(ports[j, k] * x for x in parts)
        residual = np.array([math.fsum(row) for row in terms])
        jacobian = base + (ports * slope) @ ports.T
        row_scale = np.max(np.abs(jacobian), axis=1)
        if np.any(row_scale == 0):
            raise ValueError("Diode bias has an undetermined DC equation")
        scaled = jacobian / row_scale[:, None]
        correction_rhs = -residual / row_scale
        precise_discrepancy = None
        if np.linalg.cond(scaled) > 1e10:
            for br, g in zip(devices, slope):
                br["incremental_S"] = float(g)
            solved, _, precise_discrepancy = precise_linear(matrix, (dc, 0., n_idx, rows, {}, {}), -residual)
            step = solved.real
        else:
            step = np.linalg.solve(scaled, correction_rhs)
        if not np.all(np.isfinite(step)):
            raise ValueError("Nonfinite diode bias step")
        delta_v = ports.T @ step
        scale = np.abs(base) @ np.abs(state) + np.abs(rhs) + np.abs(ports) @ np.abs(current)
        if (np.all(np.abs(residual) <= 1e-12 * scale) and
                np.all(np.abs(delta_v) <= 1e-12 * thermal)):
            discrepancy = precise_discrepancy
            if discrepancy is None:
                second = second_solve(scaled, correction_rhs)
                discrepancy = float(np.linalg.norm(second-step) / max(np.linalg.norm(state), 1e-30))
            reported = {br["name"]: {"voltage_V": float(v), "current_A": float(i), "conductance_S": float(g)}
                        for br, v, i, g in zip(devices, volts, current, slope)}
            return reported, state.astype(np.complex128), discrepancy
        growth = float(np.max(delta_v/thermal))
        state += step * (min(1., 2./growth) if growth > 0 else 1.)
    raise ValueError("Nonlinear diode operating point did not converge")
