"""Equal-volume CSTR design from all coupled steady material balances.

For h = V_stage/F_A0, recover the preceding outlet from the specified last
outlet: X_previous = X_next - h*r(X_next).  Interval values and derivatives
isolate every physical zero of the recovered feed conversion.  A grid or a
single forward branch cannot establish this design's existence or uniqueness.
"""
from __future__ import annotations

import math
from fractions import Fraction

from mpmath import mp
from mpmath.ctx_iv import MPIntervalContext


def solve_equal_volume_cstr_design(params):
    if params.get("steady_state_branch") is not None:
        raise ValueError("use design_branch for a prescribed conversion; steady_state_branch selects fixed-volume outlets")
    count = params.get("n_stages", 2)
    stages = int(count)
    if isinstance(count, bool) or stages < 1 or Fraction(count) != stages:
        raise ValueError("n_stages must be a positive integer")
    target = float(params["conversion"])
    order = float(params.get("order", 1.0))
    c0 = float(params.get("C_A0", 1.0))
    flow = float(params.get("v0", 1.0))
    feed = float(params.get("F_A0", c0 * flow))
    epsilon = float(params.get("epsilon_A", 0.0))
    t0 = float(params.get("T0", 300.0))
    top = float(params.get("T", t0))
    delta = float(params.get("delta_T_adiabatic", 0.0))
    delta = delta if abs(delta) > 1e-15 else 0.0
    inlet_t = t0 if delta else top
    gas = bool(params.get("gas_phase", abs(epsilon) > 1e-15))
    values = (target, order, c0, flow, feed, epsilon, t0, top, delta)
    if not all(map(math.isfinite, values)):
        raise ValueError("CSTR design inputs must be finite")
    if not 0 <= target <= 1 or min(c0, flow, feed, t0, inlet_t) <= 0:
        raise ValueError("CSTR design requires positive feed and 0 <= conversion <= 1")
    if epsilon <= -1 or inlet_t + delta <= 0:
        raise ValueError("CSTR temperature and expansion must remain positive")
    if target == 1 and order != 0:
        raise ValueError("complete CSTR conversion has no finite regular positive-order design")

    if "A_preexp" in params and "E_a" in params:
        kind = "preexponential"
        rate = float(params["A_preexp"])
        activation = float(params["E_a"])
        reference = 1.0
    elif all(key in params for key in ("k_ref", "T_ref", "E_a")):
        kind = "reference"
        rate = float(params["k_ref"])
        activation = float(params["E_a"])
        reference = float(params["T_ref"])
    else:
        kind = "constant"
        rate, activation, reference = float(params["k"]), 0.0, 1.0
    if not all(map(math.isfinite, (rate, activation, reference))) or min(rate, reference) <= 0:
        raise ValueError("CSTR kinetic rate and reference temperature must be finite and positive")

    scalar = mp.clone()
    scalar.dps = 100
    iv = MPIntervalContext()
    iv.dps = 80
    S, I = scalar.mpf, iv.mpf
    xt = S(target)
    rg = 8.31446261815324

    def reaction(ctx, x):
        m = ctx.mpf
        temp = m(inlet_t) + m(delta)*x
        log_k = ctx.ln(m(rate))
        if kind == "preexponential":
            log_k -= m(activation)/m(rg)/temp
        elif kind == "reference":
            log_k -= m(activation)/m(rg)*(1/temp - 1/m(reference))
        k = m(rate) if activation == 0 else ctx.exp(log_k)
        concentration = m(c0)/(1+m(epsilon)*x)
        if order:
            concentration *= 1-x
        if gas:
            concentration *= m(t0)/temp
        value = k*concentration**m(order)
        log_derivative = m(activation)/m(rg)*m(delta)/(temp*temp)
        if order:
            log_derivative -= m(order)/(1-x)
            log_derivative -= m(order)*m(epsilon)/(1+m(epsilon)*x)
            if gas:
                log_derivative -= m(order)*m(delta)/temp
        return value, value*log_derivative, temp, k

    def endpoints(value):
        return S(value.a), S(value.b)

    if target == 0:
        hscale = S(1)
        root_brackets = [(S(0), S(0))]
        visited = 0
    elif stages == 1:
        # A specified outlet gives one linear material balance. Its root is
        # at the physical endpoint, where an interval sign need not be strict.
        hscale = xt/reaction(scalar, xt)[0]
        root_brackets = [(S(1), S(1))]
        visited = 0
    else:
        terminal_rate = reaction(iv, I(target))[0]
        hscale = endpoints(I(target)/terminal_rate)[1]
        if not scalar.isfinite(hscale) or hscale <= 0:
            raise ValueError("CSTR design has no finite positive volume scale")

        def enclosure(lo, hi):
            t = I([lo, hi])
            h = t*I(hscale)
            x, derivative = I(target), I(0)
            complete_domain = True
            for step in range(stages):
                low, high = endpoints(x)
                if high < 0 or low > xt:
                    return None
                if low < 0 or high > xt:
                    complete_domain = False
                    x = I([max(S(0), low), min(xt, high)])
                r, dr, _, _ = reaction(iv, x)
                derivative = derivative*(1-h*dr) - I(hscale)*r
                x = x-h*r
            return x, derivative, complete_domain

        def sign_at(t):
            result = enclosure(t, t)
            if result is None or not result[2]:
                return None
            a, b = endpoints(result[0])
            return -1 if b < 0 else 1 if a > 0 else 0

        pending = [(S(0), S(1), 0)]
        root_brackets = []
        visited = 0
        while pending:
            lo, hi, depth = pending.pop()
            if any(a <= lo and hi <= b for a, b in root_brackets):
                continue
            visited += 1
            if visited > 32768 or depth > 1200:
                raise ValueError("all CSTR design branches are unresolved within the interval budget")
            result = enclosure(lo, hi)
            if result is None:
                continue
            value, derivative, complete_domain = result
            a, b = endpoints(value)
            if a > 0 or b < 0:
                continue
            dl, dh = endpoints(derivative)
            if complete_domain and (dl > 0 or dh < 0):
                left, right = sign_at(lo), sign_at(hi)
                exact = None
                for point, sign in ((lo, left), (hi, right)):
                    if sign == 0 and endpoints(enclosure(point, point)[0]) == (S(0), S(0)):
                        exact = point
                        break
                if exact is not None:
                    root_brackets.append((exact, exact))
                    continue
                if left in (-1, 1) and right in (-1, 1):
                    if left != right:
                        root_brackets.append((lo, hi))
                    continue
            mid = (lo+hi)/2
            if mid == lo or mid == hi:
                raise ValueError("CSTR design tangency or endpoint is unresolved at supported precision")
            # Overlap prevents a root at a subdivision boundary from being
            # mistaken for an unresolved endpoint in both children. Refined
            # duplicate enclosures are merged only after proving uniqueness
            # on their union with the interval derivative.
            overlap = (hi-lo)/16
            pending.extend(((mid-overlap, hi, depth+1),
                            (lo, mid+overlap, depth+1)))

        # Refine each already unique, physically enclosed root. Re-evaluating
        # interval signs avoids treating an inexact scalar zero as an oracle.
        refined = []
        for lo, hi in sorted(root_brackets):
            if lo == hi:
                refined.append((lo, hi))
                continue
            left = sign_at(lo)
            for _ in range(1200):
                mid = (lo+hi)/2
                middle = sign_at(mid)
                if middle is None:
                    raise ValueError("CSTR design lost its certified physical interval")
                if middle == 0:
                    lower, upper = (lo+mid)/2, (mid+hi)/2
                    sl, sh = sign_at(lower), sign_at(upper)
                    if sl in (-1, 1) and sh in (-1, 1) and sl != sh:
                        lo, hi, left = lower, upper, sl
                    else:
                        raise ValueError("CSTR design root sign is unresolved at supported precision")
                elif middle == left:
                    lo = mid
                else:
                    hi = mid
                if hi-lo <= min(lo, 1-hi if hi < 1 else lo)*S("1e-35"):
                    break
            else:
                raise ValueError("CSTR design root refinement did not converge")
            refined.append((lo, hi))
        root_brackets = []
        for lo, hi in sorted(refined):
            if root_brackets and lo <= root_brackets[-1][1]:
                a, b = root_brackets[-1]
                result = enclosure(min(a, lo), max(b, hi))
                dl, dh = endpoints(result[1])
                if not result[2] or not (dl > 0 or dh < 0):
                    raise ValueError("adjacent CSTR designs are not provably separable")
                root_brackets[-1] = (max(a, lo), min(b, hi))
            else:
                root_brackets.append((lo, hi))

    designs = []
    for lo, hi in root_brackets:
        h = (lo+hi)/2*hscale
        volume = float(h*S(feed)*stages)
        if not math.isfinite(volume) or (target > 0 and volume <= 0):
            raise ValueError("CSTR design volume is not representable")
        space_time = volume/flow
        if not math.isfinite(space_time) or (target > 0 and space_time <= 0):
            raise ValueError("CSTR design space time is not representable")
        x = xt
        reverse = []
        for _ in range(stages):
            r, _, temp, k = reaction(scalar, x)
            if not math.isfinite(float(k)) or float(k) <= 0:
                raise ValueError("CSTR exit rate constant is not representable")
            reverse.append({"conversion_X": float(x), "unreacted_fraction": float(1-x),
                            "exit_T_K": float(temp), "k_exit": float(k)})
            x -= h*r
        history = list(reversed(reverse))
        physical_stage_volume = S(volume)/stages
        previous = S(0)
        residual = S(0)
        for row in history:
            # Preserve small unreacted fractions independently of X rounding.
            outlet = S(row["conversion_X"]) if row["conversion_X"] <= row["unreacted_fraction"] else 1-S(row["unreacted_fraction"])
            r, _, temp, _ = reaction(scalar, outlet)
            consumed = S(feed)*(outlet-previous)
            reacted = physical_stage_volume*r
            scale = max(abs(consumed), abs(reacted))
            error = abs(consumed-reacted)/scale if scale else S(0)
            error = max(error, abs(S(row["exit_T_K"])-temp)/temp)
            row.update(residual=float(error), verified=bool(error < S("1e-9")))
            residual = max(residual, error)
            previous = outlet
        residual = max(residual, abs(previous-xt)/max(abs(xt), S(1)))
        if residual >= S("1e-9") or not all(row["verified"] for row in history):
            raise ValueError("rounded CSTR design does not satisfy every tank balance")
        designs.append({"volume_m3": volume, "space_time_s": space_time,
                        "conversion_X": target, "unreacted_fraction": float(1-xt),
                        "exit_T_K": history[-1]["exit_T_K"], "k_exit": history[-1]["k_exit"],
                        "stages": history, "residual": float(residual), "verified": True})
    if not designs:
        raise ValueError("no certified physical equal-volume CSTR design")
    designs.sort(key=lambda row: row["volume_m3"])
    if len({row["volume_m3"] for row in designs}) != len(designs):
        raise ValueError("distinct CSTR designs are not separable at output precision")
    branch = params.get("design_branch")
    if branch is None and len(designs) > 1:
        raise ValueError("multiple physical equal-volume CSTR designs; set design_branch to 'all' or an ascending volume index")
    certificate = {"method": ("direct_single_tank_balance" if stages == 1 else
                              "backward_material_balances_with_interval_derivative"),
                   "isolated_designs": len(designs), "intervals_examined": visited}
    if branch == "all":
        return {"domain": "reactor_kinetics", "reactor_type": "cstr_series",
                "conversion_X": target, "design_branch": "all", "design_count": len(designs),
                "designs": designs, "certificate": certificate,
                "residual": max(row["residual"] for row in designs), "verified": True}
    if branch is None:
        branch = 0
    if type(branch) is not int or not 0 <= branch < len(designs):
        raise ValueError("design_branch must be 'all' or an ascending zero-based volume index")
    return {"domain": "reactor_kinetics", "reactor_type": "cstr_series", **designs[branch],
            "design_count": len(designs), "design_branch": branch,
            "designs": designs, "certificate": certificate}
