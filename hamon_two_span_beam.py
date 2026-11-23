"""Level supports on a continuous Euler–Bernoulli beam, prismatic on each span.

Positive distributed load is downward. Reactions and vertical displacements
are positive upward, and bending moments are positive in sagging.

The support_conditions map accepts A, B and C. Missing entries are bilateral
(vertical displacement fixed, signed reaction). A negative bilateral reaction
is reported as hold_down. A compression_only bearing can carry positive force
at zero displacement (bearing), separate with positive displacement and zero
force (lifted), or have zero displacement and zero force (touching).
"""
from fractions import Fraction
from itertools import combinations


ZERO = Fraction(0)
NAMES = ("A", "B", "C")


def _reported(value):
    """Keep every signed nonzero force, gap and moment nonzero on the wire."""
    try:
        shown = float(value)
    except OverflowError as exc:
        raise ValueError("two-span result exceeds the finite reporting range") from exc
    if value and shown == 0:
        raise ValueError("two-span nonzero result is below the reporting range")
    return shown


def _reactions(x, loads, rigidities, active):
    l1, l2 = x[1], x[2] - x[1]
    q1, q2 = loads
    total = q1 * l1 + q2 * l2
    if len(active) == 3:
        # The rotations at B must match, with each span weighted by its flexibility.
        e1, e2 = rigidities
        middle = -(q1 * l1**3 / e1 + q2 * l2**3 / e2) / (8 * (l1 / e1 + l2 / e2))
        left = q1 * l1 / 2 + middle / l1
        right = q2 * l2 / 2 + middle / l2
        return (left, total - left - right, right)
    # Two supports define a statically determinate beam, including overhangs.
    i, j = active
    first_moment = q1 * l1**2 / 2 + q2 * l2 * (l1 + l2 / 2)
    reactions = [ZERO] * 3
    reactions[j] = (first_moment - x[i] * total) / (x[j] - x[i])
    reactions[i] = total - reactions[j]
    return tuple(reactions)


def _station_motion(x, loads, rigidities, active, reactions):
    # Integrate v'' = M/EI from the left. Carry the previous span's end
    # displacement and rotation across B, including when its rigidity jumps.
    offsets, slopes = [ZERO], [ZERO]
    moment = shear = ZERO
    for i, (q, rigidity) in enumerate(zip(loads, rigidities)):
        length = x[i + 1] - x[i]
        shear += reactions[i]
        offsets.append(offsets[-1] + slopes[-1] * length + (
            moment * length**2 / 2 + shear * length**3 / 6 - q * length**4 / 24
        ) / rigidity)
        slopes.append(slopes[-1] + (
            moment * length + shear * length**2 / 2 - q * length**3 / 6
        ) / rigidity)
        moment += shear * length - q * length**2 / 2
        shear -= q * length
    i, j = active[0], active[-1]
    tilt = (offsets[i] - offsets[j]) / (x[j] - x[i])
    shift = -offsets[i] - tilt * x[i]
    return ([v + tilt * s + shift for v, s in zip(offsets, x)],
            [slope + tilt for slope in slopes])


def _verify_elements(x, loads, rigidities, displacements, rotations, reactions):
    # Independently assemble cubic Hermite endpoint forces and moments from
    # the original nodal motion, EI and consistent distributed loads.
    nodal_forces = [ZERO] * 3
    nodal_moments = [ZERO] * 3
    for i, (q, ei) in enumerate(zip(loads, rigidities)):
        length = x[i + 1] - x[i]
        secant = (displacements[i + 1] - displacements[i]) / length
        left, right = rotations[i], rotations[i + 1]
        shear = 6 * ei / length**2 * (left + right - 2 * secant)
        nodal_forces[i] += shear + q * length / 2
        nodal_forces[i + 1] += -shear + q * length / 2
        nodal_moments[i] += ei / length * (4 * left + 2 * right - 6 * secant) + q * length**2 / 12
        nodal_moments[i + 1] += ei / length * (2 * left + 4 * right - 6 * secant) - q * length**2 / 12
    if tuple(nodal_forces) != reactions or any(nodal_moments):
        raise ValueError("two-span statics and Hermite nodal equilibrium disagree")


def _unique_motion(x, bilateral, reactions, gaps):
    # Bending energy is strictly convex except for rigid translation/tilt.
    # A positive bearing reaction or a bilateral support pins its rigid motion.
    pinned = [i for i in range(3) if i in bilateral or reactions[i] > 0]
    if len(pinned) != 1:
        return len(pinned) >= 2
    # With one pin, touching bearings on opposite sides prevent both tilts.
    contact = [s for s, gap in zip(x, gaps) if gap == 0]
    return min(contact) < x[pinned[0]] < max(contact)


def _signed_moments(x, loads, reactions):
    # Start each span from the outside (zero end moment). The only possible
    # interior extremum is where the shear from that end vanishes.
    left, _, right = reactions
    middle = left * x[1] - loads[0] * x[1]**2 / 2
    extrema = [ZERO, middle]
    for length, reaction, q in zip((x[1], x[2] - x[1]), (left, right), loads):
        if q and 0 < reaction / q < length:
            extrema.append(reaction**2 / (2 * q))
    return middle, max(extrema), -min(extrema)


def two_span_beam(l1, l2, q1, q2, ei, conditions):
    if not isinstance(conditions, dict) or any(
        key not in NAMES or value not in ("bilateral", "compression_only")
        for key, value in conditions.items()
    ):
        raise ValueError("support_conditions maps A, B, C to bilateral or compression_only")
    bilateral = tuple(i for i, name in enumerate(NAMES) if conditions.get(name, "bilateral") == "bilateral")
    bearings = tuple(i for i in range(3) if i not in bilateral)
    rigidities = ei if isinstance(ei, (tuple, list)) else (ei, ei)
    x, loads = (ZERO, l1, l1 + l2), (q1, q2)
    for count in range(len(bearings), -1, -1):
        for contacts in combinations(bearings, count):
            active = sorted((*bilateral, *contacts))
            if len(active) < 2:
                continue
            reactions = _reactions(x, loads, rigidities, active)
            if any(reactions[i] < 0 for i in bearings):
                continue
            gaps, rotations = _station_motion(x, loads, rigidities, active, reactions)
            if any(gaps[i] != 0 for i in active):
                raise ValueError("two-span contact support compatibility mismatch")
            if any(gaps[i] < 0 for i in bearings):
                continue
            if not _unique_motion(x, bilateral, reactions, gaps):
                raise ValueError("two-span bearing equilibrium has nonunique rigid motion")
            _verify_elements(x, loads, rigidities, gaps, rotations, reactions)
            middle, sagging, hogging = _signed_moments(x, loads, reactions)
            states = {}
            for i, name in enumerate(NAMES):
                if i in bilateral:
                    state = "hold_down" if reactions[i] < 0 else "bilateral"
                elif reactions[i] > 0:
                    state = "bearing"
                else:
                    state = "lifted" if gaps[i] > 0 else "touching"
                states[name] = state
            return {
                **{f"R_{name}_N": _reported(r) for name, r in zip(NAMES, reactions)},
                "M_B_Nm": _reported(middle),
                "M_B_abs_Nm": _reported(abs(middle)),
                "M_B_sagging_Nm": _reported(max(ZERO, middle)),
                "M_B_hogging_Nm": _reported(max(ZERO, -middle)),
                "M_max_sagging_Nm": _reported(sagging),
                "M_max_hogging_Nm": _reported(hogging),
                "M_max_Nm": _reported(max(sagging, hogging)),
                "support_displacements_m": dict(zip(NAMES, map(_reported, gaps))),
                "support_states": states,
                "equilibrium_residual_N": float(abs(sum(reactions) - q1 * l1 - q2 * l2)),
                "dual_rel_discrepancy": 0.0,
                "dual_verified": True,
            }
    raise ValueError("two-span loads and support conditions admit no restrained contact equilibrium")
