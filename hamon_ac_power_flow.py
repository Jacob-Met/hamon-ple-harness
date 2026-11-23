"""Balanced AC load flow and terminal faults of three-sequence Thevenin equivalents."""

import cmath
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

    def power_roundoff(v):
        return 32 * np.finfo(float).eps * np.abs(v) * (np.abs(ybus) @ np.abs(v))

    def power_tolerances(v, targets, positions):
        # Terminal currents can cancel even when the specified net injection
        # is exactly zero. Allow local floating-point cancellation, not a
        # whole-network relative error that can erase a small bus load.
        rounding = power_roundoff(v)
        return np.minimum(1e-6, 1e-7 * np.abs(targets[positions]) + rounding[positions])

    def reactive_violation(value, lower, upper, roundoff):
        bound = float(np.clip(value, lower, upper))
        tolerance = min(1e-6, roundoff + 4 * math.ulp(bound))
        return bound if abs(value - bound) > tolerance else None

    def polar(pq, voltage):
        def terms(v):
            current = ybus @ v
            power = v * np.conj(current)
            mismatch = np.r_[power.real[nonslack] - p[nonslack], power.imag[pq] - q[pq]]
            return current, mismatch

        for _ in range(40):
            current, res = terms(voltage)
            norm = np.linalg.norm(res, np.inf)
            q_tolerance = power_tolerances(voltage, q, pq)
            # A saturated regulator must meet the same roundoff bound used by
            # the final reactive-limit check, including after another release.
            rounding = power_roundoff(voltage)
            for pos, idx in enumerate(pq):
                if idx in limits:
                    q_tolerance[pos] = min(q_tolerance[pos], rounding[idx] + 4 * math.ulp(q[idx]))
            tolerances = np.r_[power_tolerances(voltage, p, nonslack), q_tolerance]
            if norm < 1e-11 and np.all(np.abs(res) <= tolerances):
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

    def voltage_violation(idx, bound, magnitude):
        lower, upper = limits[idx]
        if lower == upper:
            return 0.0
        excess = magnitude - vset[idx] if bound == upper else vset[idx] - magnitude
        roundoff = 16 * np.finfo(float).eps * max(magnitude, vset[idx])
        return excess if excess > roundoff else 0.0

    def coupled_regulation():
        """Solve Q limits and voltage targets as one complementarity system.

        Q = clip(Q + k*(Vset - |V|), Qmin, Qmax) expresses free regulation,
        lower saturation and upper saturation without enumerating bus types.
        A generalized Jacobian chooses the derivative of the current face.
        """
        slope = np.maximum(np.abs(np.diag(ybus)) * vset, 1e-12)
        pq_original = bus_types == "PQ"

        def terms(v, derivative=False):
            current = ybus @ v
            injection = v * np.conj(current)
            magnitude = np.abs(v)
            second = injection.imag - q
            fixed_q = pq_original.copy()
            bounds = {}
            for idx in nonslack:
                if pq_original[idx]:
                    continue
                if idx in limits:
                    lower, upper = limits[idx]
                    trial = injection[idx].imag + slope[idx] * (vset[idx] - magnitude[idx])
                    if trial <= lower or trial >= upper:
                        bound = lower if trial <= lower else upper
                        bounds[idx] = bound
                        fixed_q[idx] = True
                        second[idx] = injection[idx].imag - bound
                        continue
                second[idx] = slope[idx] * (magnitude[idx] - vset[idx])
            residual = np.r_[injection.real[nonslack] - p[nonslack], second[nonslack]]
            rounding = power_roundoff(v)
            qtol = power_tolerances(v, q, nonslack)
            for position, idx in enumerate(nonslack):
                if idx in bounds:
                    qtol[position] = min(1e-11, rounding[idx] + 4 * math.ulp(bounds[idx]))
                elif not pq_original[idx]:
                    qtol[position] = slope[idx] * 16 * np.finfo(float).eps * max(magnitude[idx], vset[idx])
            tolerance = np.r_[power_tolerances(v, p, nonslack), qtol]
            if not derivative:
                return residual, tolerance, bounds
            vd = np.diag(v)
            da = 1j * vd @ np.conj(np.diag(current) - ybus @ vd)
            unit = np.diag(v / magnitude)
            dm = vd @ np.conj(ybus @ unit) + np.conj(np.diag(current)) @ unit
            qa, qm = da.imag.copy(), dm.imag.copy()
            for idx in nonslack:
                if not fixed_q[idx]:
                    qa[idx] = 0
                    qm[idx] = 0
                    qm[idx, idx] = slope[idx]
            jacobian = np.block([
                [da.real[np.ix_(nonslack, nonslack)], dm.real[np.ix_(nonslack, nonslack)]],
                [qa[np.ix_(nonslack, nonslack)], qm[np.ix_(nonslack, nonslack)]],
            ])
            return residual, tolerance, bounds, jacobian

        iterations = 0

        def solve(seed):
            nonlocal iterations
            v = seed.copy()
            for _ in range(80):
                iterations += 1
                residual, tolerance, bounds, jacobian = terms(v, True)
                norm = np.linalg.norm(residual, np.inf)
                if norm < 1e-11 and np.all(np.abs(residual) <= tolerance):
                    return v, bounds
                try:
                    correction = np.linalg.solve(jacobian, residual)
                except np.linalg.LinAlgError:
                    correction = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
                angles, magnitudes = np.angle(v), np.abs(v)
                for step in range(24):
                    updated_angles, updated_magnitudes = angles.copy(), magnitudes.copy()
                    updated_angles[nonslack] -= correction[:len(nonslack)] / 2**step
                    updated_magnitudes[nonslack] -= correction[len(nonslack):] / 2**step
                    if (updated_magnitudes <= 0).any():
                        continue
                    updated = updated_magnitudes * np.exp(1j * updated_angles)
                    if np.linalg.norm(terms(updated)[0], np.inf) < norm:
                        v = updated
                        break
                else:
                    raise ValueError("Coupled AC regulator step did not reduce residual")
            raise ValueError("Coupled AC regulation did not converge")

        # A global phase rotation must not select a different derivative at Q=0.
        flat = np.asarray(vset, dtype=complex)
        try:
            voltage, bounds = solve(flat)
            return voltage * np.exp(1j * reference), bounds, iterations
        except ValueError:
            pass

        # A physical PQ start permits voltage movement before reimposing the
        # generator targets. It does not assume those provisional Q values solve
        # the original constrained problem.
        saved_q = q.copy()
        try:
            for idx, (lower, upper) in limits.items():
                q[idx] = np.clip(0.0, lower, upper)
            pq_seed = np.asarray([idx for idx in nonslack
                                  if pq_original[idx] or idx in limits], dtype=int)
            seed = polar(pq_seed, flat.copy())
        except (ValueError, np.linalg.LinAlgError):
            seed = None
        finally:
            q[:] = saved_q
        if seed is not None:
            try:
                voltage, bounds = solve(seed)
                return voltage * np.exp(1j * reference), bounds, iterations
            except ValueError:
                pass

        # Complementarity Newton is local: a small ill-conditioned network may
        # need another equilibrium branch. Exhaust only <=4 regulator states,
        # never a truncated exponential search of the full coupled network.
        if len(limits) <= 4:
            from itertools import product
            ordered = sorted(limits, key=lambda idx: ids[idx])
            choices = [(None, *dict.fromkeys(limits[idx])) for idx in ordered]
            for values in product(*choices):
                bounds = {idx: value for idx, value in zip(ordered, values)
                          if value is not None}
                try:
                    q[:] = saved_q
                    for idx, bound in bounds.items():
                        q[idx] = bound
                    pq_trial = np.asarray([idx for idx in nonslack
                                           if pq_original[idx] or idx in bounds], dtype=int)
                    voltage = polar(pq_trial, flat.copy())
                    power = voltage * np.conj(ybus @ voltage)
                    rounding = power_roundoff(voltage)
                    if any(
                        reactive_violation(power[idx].imag, *limit, rounding[idx]) is not None
                        or (idx in bounds and voltage_violation(idx, bounds[idx], abs(voltage[idx])))
                        for idx, limit in limits.items()
                    ):
                        continue
                    return voltage * np.exp(1j * reference), bounds, iterations
                except (ValueError, np.linalg.LinAlgError):
                    continue
                finally:
                    q[:] = saved_q
        raise ValueError("Coupled AC regulation did not converge; feasibility undetermined")

    regulation_iterations = 0
    if limits:
        voltage, limited, regulation_iterations = coupled_regulation()
        for idx, bound in limited.items():
            bus_types[idx] = "PQ"
            q[idx] = bound
        pq = np.flatnonzero(bus_types == "PQ")
        # Polish the selected physical equations at their original strict tolerances.
        voltage = polar(pq, voltage)
    else:
        limited = {}
        pq = np.flatnonzero(bus_types == "PQ")
        voltage = polar(pq, initial.copy())
    injection = voltage * np.conj(ybus @ voltage)
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

    def regulated_state(v, s):
        rounding = power_roundoff(v)
        for idx, (lower, upper) in limits.items():
            if reactive_violation(s[idx].imag, lower, upper, rounding[idx]) is not None:
                return False
            if idx in limited and voltage_violation(idx, limited[idx], abs(v[idx])):
                return False
        return True

    complementarity = regulated_state(voltage, injection)
    proposed_limits = None
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
        proposed_limits = regulated_state(claimed, power)
        verified = (
            verified
            and proposed_limits
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
            and np.all(
                np.abs(np.abs(claimed[pv]) - vset[pv])
                <= 16 * np.finfo(float).eps * vset[pv]
            )
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
        "proposed_pv_limits_consistent": proposed_limits,
        "clamped_pv_q_pu": {ids[i]: x for i, x in limited.items()},
        "complementarity_iterations": regulation_iterations,
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


def phasor(data):
    if not isinstance(data, dict) or set(data) != {"real", "imag"}:
        raise ValueError("A phasor requires finite real and imag components")
    if any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
        for x in data.values()
    ):
        raise ValueError("A phasor requires finite real and imag components")
    return complex(data["real"], data["imag"])


def symmetrical_fault(spec):
    fault = spec["fault_type"]
    if fault not in ("3PH", "SLG", "LL", "DLG"):
        raise ValueError("Use 3PH, SLG, LL or DLG")
    labels = ("A", "B", "C")
    faulted = spec.get(
        "faulted_phases",
        {"3PH": ["A", "B", "C"], "SLG": ["A"], "LL": ["B", "C"], "DLG": ["B", "C"]}[
            fault
        ],
    )
    if (
        not isinstance(faulted, list)
        or not all(p in labels for p in faulted)
        or len(set(faulted)) != len(faulted)
    ):
        raise ValueError("faulted_phases must be a list of distinct phase names")
    count = {"3PH": 3, "SLG": 1, "LL": 2, "DLG": 2}[fault]
    if len(faulted) != count:
        raise ValueError("Incorrect number of faulted phases")
    sequence_impedances = spec["sequence_impedances_pu"]
    if not isinstance(sequence_impedances, dict) or set(sequence_impedances) != {
        "zero",
        "positive",
        "negative",
    }:
        raise ValueError("Specify zero-, positive- and negative-sequence impedances")
    z0, z1, z2 = (
        phasor(sequence_impedances[k]) for k in ("zero", "positive", "negative")
    )
    zf = phasor(spec.get("fault_impedance_pu", {"real": 0, "imag": 0}))
    emf = phasor(spec["prefault_positive_voltage_pu"])
    if min(z.real for z in (z0, z1, z2, zf)) < 0 or emf == 0:
        raise ValueError(
            "Require nonnegative resistances and nonzero prefault positive-sequence voltage"
        )
    # Preserve the ABC positive sequence when rotating the fault to canonical
    # A (SLG) or BC (LL/DLG) positions.
    offset = (
        0
        if fault == "3PH"
        else labels.index(
            faulted[0]
            if fault == "SLG"
            else next(p for p in labels if p not in faulted)
        )
    )
    order = np.roll(np.arange(3), -offset)
    alpha = cmath.exp(2j * math.pi / 3)
    rotation = emf * alpha ** (-offset)
    transform = np.array([[1, 1, 1], [1, alpha**2, alpha], [1, alpha, alpha**2]])
    if fault == "3PH":
        seq = (0, rotation / (z1 + zf), 0)
    elif fault == "SLG":
        i = rotation / (z0 + z1 + z2 + 3 * zf)
        seq = (i, i, i)
    elif fault == "LL":
        i = rotation / (z1 + z2 + zf)
        seq = (0, i, -i)
    else:
        # BC meet at a common point connected to ground by zf. The parallel
        # negative/zero-sequence paths can be solved without dividing by their sum.
        zg = z0 + 3 * zf
        determinant = z1 * (zg + z2) + zg * z2
        seq = (
            -rotation * z2 / determinant,
            rotation * (zg + z2) / determinant,
            -rotation * zg / determinant,
        )
    canonical = transform @ np.asarray(seq)
    iabc = np.empty(3, dtype=complex)
    iabc[order] = canonical
    # Independent phase-domain system: v = e - Zabc i, with three linear
    # terminal boundary conditions. No sequence-fault formulas enter this solve.
    zabc = (transform * np.array([z0, z1, z2])) @ np.conjugate(transform.T) / 3
    source = emf * transform[:, 1]
    voltage = source - zabc @ iabc
    zero_v = np.zeros((3, 3), dtype=complex)
    zero_i = np.zeros_like(zero_v)
    a, b, c = order
    if fault == "3PH":
        zero_v = np.eye(3, dtype=complex)
        zero_i = -zf * zero_v
    elif fault == "SLG":
        zero_v[0, a] = 1
        zero_i[0, a] = -zf
        zero_i[1, b] = 1
        zero_i[2, c] = 1
    elif fault == "LL":
        zero_i[0, a] = 1
        zero_i[1, b] = zero_i[1, c] = 1
        zero_v[2, b] = 1
        zero_v[2, c] = -1
        zero_i[2, b] = -zf
    else:
        zero_i[0, a] = 1
        zero_v[1, b] = 1
        zero_v[1, c] = -1
        zero_v[2, b] = 1
        zero_i[2, b] = zero_i[2, c] = -zf
    iphase = np.linalg.solve(zero_v @ zabc - zero_i, zero_v @ source)
    residual = zero_v @ voltage + zero_i @ iabc
    scale = max(abs(iabc))
    precision = 32 * np.finfo(float).eps * scale
    tol = np.minimum(1e-6, 1e-7 * np.abs(iabc) + precision)
    verified = bool(
        np.all(np.abs(iphase - iabc) <= tol) and np.max(np.abs(residual)) < 1e-6
    )
    if "proposed_phase_currents_pu" in spec:
        proposed = spec["proposed_phase_currents_pu"]
        if not isinstance(proposed, dict) or set(proposed) != set(labels):
            raise ValueError("Proposed currents must contain A, B and C")
        claimed = np.array([phasor(proposed[k]) for k in labels])
        verified = verified and bool(np.all(np.abs(claimed - iabc) <= tol))

    def pack(z):
        return {"real": float(z.real), "imag": float(z.imag)}

    def vector(names, vec):
        return dict(zip(names, map(pack, vec)))

    seq_names = ("zero", "positive", "negative")
    return {
        "ok": True,
        "calculation": "symmetrical_fault",
        "conservation_verified": verified,
        "fault_type": fault,
        "phase_currents_pu": vector(labels, iabc),
        "phase_voltages_pu": vector(labels, voltage),
        "sequence_currents_pu": vector(seq_names, np.conjugate(transform.T) @ iabc / 3),
        "sequence_voltages_pu": vector(
            seq_names, np.conjugate(transform.T) @ voltage / 3
        ),
        "ground_return_current_pu": pack(np.sum(iabc)),
        "max_phase_sequence_difference_pu": float(np.max(np.abs(iphase - iabc))),
        "max_terminal_boundary_residual_pu": float(np.max(np.abs(residual))),
    }


def verify_symmetrical_fault(spec):
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return symmetrical_fault(spec)
    except (
        ValueError,
        TypeError,
        KeyError,
        ZeroDivisionError,
        OverflowError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:
        return {
            "ok": False,
            "calculation": "symmetrical_fault",
            "conservation_verified": False,
            "reason": str(exc),
        }
