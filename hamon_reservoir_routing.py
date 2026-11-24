"""Modified Puls level-pool routing on a supplied stage/storage/outflow rating."""
from bisect import bisect_left
from decimal import Decimal, localcontext
from math import isfinite


def _number(value, name):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return Decimal(value)


def _table(spec, name, fields):
    values = spec.get(name)
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"{name} needs at least two rows")
    rows = []
    for item in values:
        if not isinstance(item, dict) or set(item) != set(fields):
            raise ValueError(f"{name} rows must contain {', '.join(fields)}")
        rows.append(tuple(_number(item[key], key) for key in fields))
    return rows


def _between(rows, coordinate, key):
    """Bracket one coordinate on a strictly increasing curve, without extrapolation."""
    index = bisect_left(rows, coordinate, key=key)
    if index < len(rows) and key(rows[index]) == coordinate:
        return rows[index], rows[index], Decimal(0)
    if index == 0 or index == len(rows):
        raise ValueError("outside the supplied rating curve")
    low, high = rows[index-1], rows[index]
    return low, high, (coordinate - key(low)) / (key(high) - key(low))


def _interpolate(low, high, fraction):
    return tuple(a + fraction*(b-a) for a, b in zip(low, high))


def _publish(number):
    value = float(number)
    if not isfinite(value) or (not value and number):
        raise ValueError("Hydrograph result is outside the finite output range")
    return value


def _relative(error, scale):
    return abs(error)/scale if scale else abs(error)


def _route(spec):
    keys = {"mode", "calculation", "rating_curve", "inflow_hydrograph", "initial_stage_m"}
    if set(spec) - keys:
        raise ValueError("Unsupported reservoir inputs; use one fixed total outlet rating")
    curve = _table(spec, "rating_curve", ("stage_m", "storage_m3", "outflow_m3_s"))
    inflows = _table(spec, "inflow_hydrograph", ("time_s", "inflow_m3_s"))
    start = _number(spec.get("initial_stage_m"), "initial_stage_m")
    if any(s < 0 or q < 0 for _, s, q in curve) or any(i < 0 for _, i in inflows):
        raise ValueError("Storage, outflow and inflow must be nonnegative")
    if any(b[0] <= a[0] or b[1] <= a[1] or b[2] < a[2] for a, b in zip(curve, curve[1:])):
        raise ValueError("Stage and storage must increase; rated outflow must not decrease")
    if any(b[0] <= a[0] for a, b in zip(inflows, inflows[1:])):
        raise ValueError("Hydrograph times must increase strictly")

    # Maintain enough significant digits for offsets in the supplied absolute
    # datums. Published binary64 states are checked independently below.
    scales = [n.adjusted() for table in (curve, inflows) for row in table for n in row if n]
    with localcontext() as ctx:
        ctx.prec = 70 + (max(scales)-min(scales) if scales else 0)
        stage_key = lambda row: row[0]
        stage, storage, outflow = _interpolate(*_between(curve, start, stage_key))
        series = []
        inflow_volume = outflow_volume = Decimal(0)
        step_error = curve_error = Decimal(0)
        for n, (time, inflow) in enumerate(inflows):
            if n:
                half_dt = (time - inflows[n-1][0]) / 2
                target = storage + half_dt*(inflows[n-1][1] + inflow - outflow)
                try:
                    segment = _between(curve, target, lambda row: row[1] + half_dt*row[2])
                except ValueError as error:
                    raise ValueError(f"Storage indication at time {time:g} s is {error}") from error
                stage, storage, outflow = _interpolate(*segment)

            # Replay the rating on the wire stage, and the conservation equation
            # on wire volumes/flows. A large elevation or dead-storage datum
            # can mask an excursion when the result is rounded for JSON.
            h, s, q = map(_publish, (stage, storage, outflow))
            wire_h, wire_s, wire_q = map(Decimal, (h, s, q))
            low, high, f = _between(curve, wire_h, stage_key)
            _, rated_s, rated_q = _interpolate(low, high, f)
            span = high[1] - low[1]
            if not span:
                knot = bisect_left(curve, wire_h, key=stage_key)
                left = max(0, min(knot, len(curve)-2)) if stage >= wire_h else max(0, knot-1)
                span = curve[left+1][1] - curve[left][1]
            curve_error = max(curve_error, _relative(wire_s-rated_s, span),
                              _relative(wire_q-rated_q, max(wire_q, rated_q)))
            if n:
                volume_in = half_dt*(inflows[n-1][1] + inflow)
                volume_out = half_dt*(previous_q + wire_q)
                change = wire_s - previous_s
                step_error = max(step_error, _relative(change + volume_out - volume_in,
                                                       max(abs(change), volume_in, volume_out)))
                inflow_volume += volume_in
                outflow_volume += volume_out
            previous_q, previous_s = wire_q, wire_s
            series.append({"time_s": spec["inflow_hydrograph"][n]["time_s"], "stage_m": h,
                           "storage_m3": s, "outflow_m3_s": q})

        change = previous_s - Decimal(series[0]["storage_m3"])
        balance = change + outflow_volume - inflow_volume
        cumulative = _relative(balance, max(abs(change), inflow_volume, outflow_volume))
        peak = max(series, key=lambda row: row["outflow_m3_s"])
        peak_stage = max(series, key=lambda row: row["stage_m"])
        peak_inflow = max(inflows, key=lambda row: row[1])
        residual = max(step_error, curve_error, cumulative)
        verified = residual <= Decimal("1e-10")
        return {"ok": verified, "conservation_verified": verified, "hydrograph": series,
                "peak_stage_m": peak_stage["stage_m"], "peak_stage_time_s": peak_stage["time_s"],
                "peak_outflow_m3_s": peak["outflow_m3_s"], "peak_outflow_time_s": peak["time_s"],
                "peak_inflow_m3_s": _publish(peak_inflow[1]),
                "peak_lag_s": _publish(Decimal(peak["time_s"])-peak_inflow[0]),
                "inflow_volume_m3": _publish(inflow_volume), "outflow_volume_m3": _publish(outflow_volume),
                "storage_change_m3": _publish(change), "mass_balance_error_m3": float(balance),
                "max_step_continuity_relative_error": float(step_error),
                "max_rating_relative_error": float(curve_error), "residual": float(residual)}


def verify_reservoir_routing(spec):
    identity = {"tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
                "calculation": "level_pool_routing", "method": "modified_puls"}
    try:
        return {**identity, **_route(spec)}
    except (ValueError, ArithmeticError, TypeError) as error:
        return {**identity, "ok": False, "conservation_verified": False, "reason": str(error)}
