"""Balanced per-unit AC load flow on a connected network with one slack bus."""

import math

import numpy as np
from scipy.optimize import root


def number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite per-unit network input")
    return float(value)


def names(rows, label):
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(x, dict) for x in rows)
    ):
        raise ValueError(f"{label} must be a nonempty array of objects")
    ids = [x["id"] for x in rows]
    if any(not isinstance(k, str) or not k for k in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{label} must have unique nonempty string IDs")
    return ids


def network(spec):
    buses, branches = spec["buses"], spec["lines"]
    ids = names(buses, "buses")
    line_ids = names(branches, "lines")
    index = {name: i for i, name in enumerate(ids)}
    nb = len(buses)
    bus_types = np.array([b["type"] for b in buses])
    if (
        not np.isin(bus_types, ["slack", "PV", "PQ"]).all()
        or np.sum(bus_types == "slack") != 1
        or nb < 2
    ):
        raise ValueError("Use exactly one slack bus and at least one PQ or PV bus")
    slack = int(np.flatnonzero(bus_types == "slack")[0])
    nonslack = np.flatnonzero(bus_types != "slack")
    p = np.zeros(nb)
    q = np.zeros(nb)
    vset = np.ones(nb)
    shunt = np.zeros(nb, dtype=complex)
    limits = {}
    for i, b in enumerate(buses):
        typ = b["type"]
        allowed = {"id", "type", "shunt_g_pu", "shunt_b_pu"}
        if typ != "slack":
            allowed.add("p_injection_pu")
            p[i] = number(b["p_injection_pu"])
        if typ == "PQ":
            allowed.add("q_injection_pu")
            q[i] = number(b["q_injection_pu"])
        else:
            allowed.add("vm_pu")
            vset[i] = number(b["vm_pu"])
            if vset[i] <= 0:
                raise ValueError("Voltage setpoints must be positive")
        if typ == "slack":
            allowed.add("va_deg")
        if typ == "PV":
            allowed.update(("q_min_injection_pu", "q_max_injection_pu"))
            if "q_min_injection_pu" in b or "q_max_injection_pu" in b:
                lower, upper = (
                    number(b[k]) for k in ("q_min_injection_pu", "q_max_injection_pu")
                )
                if lower > upper:
                    raise ValueError("Reversed PV reactive limits")
                limits[i] = (lower, upper)
        if set(b) - allowed:
            raise ValueError(f"Unsupported {typ} bus attributes")
        shunt[i] = complex(
            number(b.get("shunt_g_pu", 0)), number(b.get("shunt_b_pu", 0))
        )
    reference = math.radians(number(buses[slack].get("va_deg", 0)))
    ybus = np.diag(shunt)
    a = []
    b = []
    admittance = []
    ratio = []
    charging = []
    adjacency = [set() for _ in buses]
    for row in branches:
        if set(row) - {
            "id",
            "from",
            "to",
            "r_pu",
            "x_pu",
            "line_charging_b_pu",
            "tap_ratio",
            "phase_shift_deg",
        }:
            raise ValueError("Unsupported line attributes")
        f, t = index[row["from"]], index[row["to"]]
        impedance = complex(number(row["r_pu"]), number(row["x_pu"]))
        tap = number(row.get("tap_ratio", 1))
        if f == t or impedance == 0 or impedance.real < 0 or tap <= 0:
            raise ValueError(
                "Require distinct endpoints, finite nonzero series impedance, nonnegative resistance and positive tap"
            )
        y = 1 / impedance
        z = tap * np.exp(1j * math.radians(number(row.get("phase_shift_deg", 0))))
        yc = 0.5j * number(row.get("line_charging_b_pu", 0))
        ybus[f, f] += (y + yc) / tap**2
        ybus[t, t] += y + yc
        ybus[f, t] -= y / z.conjugate()
        ybus[t, f] -= y / z
        a.append(f)
        b.append(t)
        admittance.append(y)
        ratio.append(z)
        charging.append(yc)
        adjacency[f].add(t)
        adjacency[t].add(f)
    seen = {slack}
    stack = [slack]
    while stack:
        neighbors = adjacency[stack.pop()] - seen
        seen.update(neighbors)
        stack.extend(neighbors)
    if len(seen) != nb:
        raise ValueError("Every bus must be connected to the slack bus")
    f, t, y, tap, yc = map(np.asarray, (a, b, admittance, ratio, charging))

    def branch_currents(voltage):
        tapped = voltage[f] / tap
        series = (tapped - voltage[t]) * y
        send = (series + yc * tapped) / np.conjugate(tap)
        receive = -series + yc * voltage[t]
        nodal = shunt * voltage
        np.add.at(nodal, f, send)
        np.add.at(nodal, t, receive)
        return nodal, send, receive

    def power_tolerances(v, targets, positions):
        # Terminal currents can cancel even when the specified net injection
        # is exactly zero. Allow local floating-point cancellation, not a
        # whole-network relative error that can erase a small bus load.
        rounding = 64 * np.finfo(float).eps * np.abs(v) * (np.abs(ybus) @ np.abs(v))
        return np.minimum(1e-6, 1e-7 * np.abs(targets[positions]) + rounding[positions])

    def polar(pq, voltage):
        def terms(v):
            current = ybus @ v
            power = v * np.conj(current)
            mismatch = np.r_[power.real[nonslack] - p[nonslack], power.imag[pq] - q[pq]]
            return current, mismatch

        for _ in range(40):
            current, res = terms(voltage)
            norm = np.linalg.norm(res, np.inf)
            limits = np.r_[
                power_tolerances(voltage, p, nonslack), power_tolerances(voltage, q, pq)
            ]
            if norm < 1e-11 and np.all(np.abs(res) <= limits):
                return voltage
            vd = np.diag(voltage)
            iv = np.diag(current)
            da = 1j * vd @ np.conj(iv - ybus @ vd)
            unit = np.diag(voltage / np.abs(voltage))
            dm = vd @ np.conj(ybus @ unit) + np.conj(iv) @ unit
            jac = np.block(
                [
                    [
                        da.real[np.ix_(nonslack, nonslack)],
                        dm.real[np.ix_(nonslack, pq)],
                    ],
                    [da.imag[np.ix_(pq, nonslack)], dm.imag[np.ix_(pq, pq)]],
                ]
            )
            correction = np.linalg.solve(jac, res)
            for step in range(20):
                angles = np.angle(voltage)
                magnitudes = np.abs(voltage)
                angles[nonslack] -= correction[: len(nonslack)] / 2**step
                magnitudes[pq] -= correction[len(nonslack) :] / 2**step
                if (magnitudes <= 0).any():
                    continue
                updated = magnitudes * np.exp(1j * angles)
                if np.linalg.norm(terms(updated)[1], np.inf) < norm:
                    voltage = updated
                    break
            else:
                raise ValueError("AC Newton step failed to reduce power mismatch")
        raise ValueError("AC Newton power flow did not converge")

    initial = vset * np.exp(1j * reference)
    voltage = initial.copy()
    limited = {}
    for _ in range(len(limits) + 1):
        pq = np.flatnonzero(bus_types == "PQ")
        voltage = polar(pq, voltage)
        injection = voltage * np.conj(ybus @ voltage)
        violations = []
        for idx, (lower, upper) in limits.items():
            if (
                idx not in limited
                and not lower - 1e-9 <= injection[idx].imag <= upper + 1e-9
            ):
                bound = float(np.clip(injection[idx].imag, lower, upper))
                violations.append((abs(injection[idx].imag - bound), idx, bound))
        if not violations:
            break
        _, idx, bound = max(violations)
        limited[idx] = bound
        bus_types[idx] = "PQ"
        q[idx] = bound
    pv = np.flatnonzero(bus_types == "PV")

    # Independent rectangular current-balance solve. Form the currents at each
    # line terminal explicitly, avoiding the polar solver's bus admittance matrix.
    def unpack(z):
        v = initial.copy()
        v[nonslack] = z[: len(nonslack)] + 1j * z[len(nonslack) :]
        return v

    def balance(z):
        v = unpack(z)
        current = branch_currents(v)[0]
        if np.any(np.abs(v[pq]) < 1e-12):
            return np.full(2 * len(nonslack), 1e12)
        pq_res = current[pq] - np.conj((p[pq] + 1j * q[pq]) / v[pq])
        pv_p = (v[pv] * np.conj(current[pv])).real - p[pv]
        return np.r_[pq_res.real, pq_res.imag, pv_p, np.abs(v[pv]) ** 2 - vset[pv] ** 2]

    z0 = np.r_[initial[nonslack].real, initial[nonslack].imag]
    independent = root(balance, z0, tol=1e-11)
    other = unpack(independent.x)
    independent_res = float(np.linalg.norm(balance(independent.x), np.inf))
    current, send, receive = branch_currents(voltage)
    sf = voltage[f] * np.conj(send)
    st = voltage[t] * np.conj(receive)
    total_lines = np.sum(sf + st)
    total_shunts = np.sum(np.abs(voltage) ** 2 * np.conj(shunt))
    tellegen = float(abs(np.sum(injection) - total_lines - total_shunts))
    current_mismatch = float(np.max(np.abs(ybus @ voltage - current)))
    difference = float(np.max(np.abs(voltage - other)))
    complementarity = all(
        (limits[i][0] == limits[i][1])
        or (bound == limits[i][1] and abs(voltage[i]) <= vset[i] + 1e-8)
        or (bound == limits[i][0] and abs(voltage[i]) >= vset[i] - 1e-8)
        for i, bound in limited.items()
    )
    verified = (
        complementarity
        and independent_res < 1e-6
        and difference < 1e-6
        and tellegen < 1e-6
        and current_mismatch < 1e-6
    )
    if "proposed_voltages" in spec:
        claims = spec["proposed_voltages"]
        if not isinstance(claims, dict) or set(claims) != set(ids):
            raise ValueError("A voltage map must cover every bus ID")
        values = []
        for name in ids:
            mag = number(claims[name]["vm_pu"])
            deg = number(claims[name]["va_deg"])
            if mag <= 0:
                raise ValueError("Proposed voltage magnitude must be positive")
            values.append(mag * np.exp(1j * math.radians(deg)))
        claimed = np.asarray(values)
        claim_error = np.max(np.abs(claimed - voltage))
        power = claimed * np.conj(branch_currents(claimed)[0])
        verified = (
            verified
            and claim_error < 1e-6
            and abs(claimed[slack] - initial[slack])
            <= 16 * np.finfo(float).eps * vset[slack]
            and np.all(
                np.abs(power.real[nonslack] - p[nonslack])
                <= power_tolerances(claimed, p, nonslack)
            )
            and np.all(
                np.abs(power.imag[pq] - q[pq]) <= power_tolerances(claimed, q, pq)
            )
            and np.all(np.abs(np.abs(claimed[pv]) - vset[pv]) < 1e-6)
        )
    return {
        "ok": True,
        "calculation": "ac_power_flow",
        "conservation_verified": bool(verified),
        "voltages": {
            name: {"vm_pu": float(abs(v)), "va_deg": float(np.angle(v, deg=True))}
            for name, v in zip(ids, voltage)
        },
        "bus_injections": {
            name: {"p_pu": float(s.real), "q_pu": float(s.imag)}
            for name, s in zip(ids, injection)
        },
        "lines": {
            name: {
                "p_from_pu": float(a.real),
                "q_from_pu": float(a.imag),
                "p_to_pu": float(b.real),
                "q_to_pu": float(b.imag),
            }
            for name, a, b in zip(line_ids, sf, st)
        },
        "pv_limits_consistent": bool(complementarity),
        "clamped_pv_q_pu": {ids[i]: x for i, x in limited.items()},
        "max_rectangular_mismatch_pu": independent_res,
        "max_voltage_disagreement_pu": difference,
        "tellegen_residual_pu": tellegen,
        "max_terminal_current_mismatch_pu": current_mismatch,
    }


def verify_ac_power_flow(spec):
    try:
        with np.errstate(invalid="raise", over="raise", divide="raise"):
            return network(spec)
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:
        return {
            "ok": False,
            "calculation": "ac_power_flow",
            "conservation_verified": False,
            "reason": str(exc),
        }
