"""Euler–Bernoulli beam on vertical seats with independent unilateral contact."""
from fractions import Fraction
from itertools import accumulate, combinations

from hamon_two_span_beam import _reported, _station_motion


ZERO = Fraction(0)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("expected a finite number")
    return Fraction(str(value))


def _per_seat(raw, labels, default):
    if isinstance(raw, dict) and not raw.keys() - set(labels):
        return [raw.get(name, default) for name in labels]
    if isinstance(raw, list) and len(raw) == len(labels):
        return raw
    raise ValueError("each support needs a valid station label or one entry per seat")


def _assemble(lengths, loads, rigidities):
    size = 2*(len(lengths)+1)
    stiffness = [[ZERO]*size for _ in range(size)]
    force = [ZERO]*size
    for i, (length, q, ei) in enumerate(zip(lengths, loads, rigidities)):
        scale = ei/length**3
        rotation = length**2
        block = ((12, 6*length, -12, 6*length),
                 (6*length, 4*rotation, -6*length, 2*rotation),
                 (-12, -6*length, 12, -6*length),
                 (6*length, 2*rotation, -6*length, 4*rotation))
        applied = (-q*length/2, -q*rotation/12, -q*length/2, q*rotation/12)
        for j, f in enumerate(applied):
            force[2*i+j] += f
            for k, entry in enumerate(block[j]):
                stiffness[2*i+j][2*i+k] += scale*entry
    return stiffness, force


def _solve(stiffness, applied, active, levels):
    prescribed = {2*i:levels[i] for i in active}
    free = [i for i in range(len(applied)) if i not in prescribed]
    matrix = [[stiffness[i][j] for j in free] for i in free]
    rhs = [applied[i]-sum(stiffness[i][k]*v for k,v in prescribed.items()) for i in free]
    # Two distinct vertical restraints remove both rigid modes. The free
    # matrix is positive definite and retains half-bandwidth at most three.
    for j,row in enumerate(matrix):
        if row[j] <= 0:
            raise ValueError("unrestrained vertical translation or rotation")
        stop = min(len(free), j+4)
        for i in range(j+1, stop):
            factor = matrix[i][j]/row[j]
            rhs[i] -= factor*rhs[j]
            for k in range(j+1, stop):
                matrix[i][k] -= factor*row[k]
    solution = [ZERO]*len(free)
    for j in reversed(range(len(free))):
        other = sum(matrix[j][k]*solution[k] for k in range(j+1,min(len(free),j+4)))
        solution[j] = (rhs[j]-other)/matrix[j][j]
    motion = [ZERO]*len(applied)
    for i,value in prescribed.items():
        motion[i] = value
    for i,value in zip(free,solution):
        motion[i] = value
    reactions = [sum(k*v for k,v in zip(row,motion))-f for row,f in zip(stiffness,applied)]
    if any(reactions[i] for i in free):
        raise ValueError("free nodal degree of freedom is out of equilibrium")
    return motion[::2], motion[1::2], reactions[::2]


def _certify(x, loads, rigidities, levels, active, bilateral, v, rotations, reactions):
    gaps = [position-level for position,level in zip(v,levels)]
    for i in range(len(x)):
        if i in bilateral:
            if gaps[i]:
                raise ValueError("bilateral support displacement mismatch")
        elif gaps[i] < 0 or reactions[i] < 0 or gaps[i]*reactions[i] != 0:
            return None
    fixed = [i for i in range(len(x)) if i in bilateral or reactions[i] > 0]
    touching = [i for i,gap in enumerate(gaps) if gap == 0]
    if not fixed or (len(fixed)==1 and not min(touching)<fixed[0]<max(touching)):
        raise ValueError("contact permits nonunique rigid motion")

    # Independent from FEM: integrate shear and moment along each loaded
    # span and reconstruct the complete nodal displacement and rotation.
    checked_v, checked_r = _station_motion(x, loads, rigidities, active, reactions, levels)
    if v != checked_v or rotations != checked_r:
        raise ValueError("integrated beam shape disagrees with Hermite solution")
    shear = moment = ZERO
    extrema = [ZERO]
    node_moments = [ZERO]
    for i,q in enumerate(loads):
        length = x[i+1]-x[i]
        shear += reactions[i]
        if q and 0 < shear/q < length:
            extrema.append(moment+shear**2/(2*q))
        moment += shear*length-q*length**2/2
        shear -= q*length
        extrema.append(moment)
        node_moments.append(moment)
    if moment != 0 or shear+reactions[-1] != 0:
        raise ValueError("global moment or force imbalance")
    return gaps, node_moments, max(extrema), -min(extrema)


def solve_multispan(spec):
    for name in ("spans_m","udl_N_m","span_EI_Nm2"):
        if not isinstance(spec.get(name), list):
            raise ValueError(f"{name} must be a per-span array")
    lengths, loads, rigidities = (list(map(_number,spec[name])) for name in ("spans_m","udl_N_m","span_EI_Nm2"))
    if not 2 <= len(lengths) <= 12 or len(loads) != len(lengths) or len(rigidities) != len(lengths):
        raise ValueError("supply two to twelve matching spans, loads and rigidities")
    if min(lengths) <= 0 or min(rigidities) <= 0:
        raise ValueError("all spans and flexural rigidities must be positive")
    labels = [chr(ord("A")+i) for i in range(len(lengths)+1)]
    conditions = _per_seat(spec.get("support_conditions", {}),labels,"bilateral")
    if any(c not in ("bilateral","compression_only") for c in conditions):
        raise ValueError("supports must be bilateral or compression_only vertical seats")
    levels = [-_number(s) for s in _per_seat(spec.get("support_settlements_m",{}),labels,0)]
    bilateral = {i for i,c in enumerate(conditions) if c=="bilateral"}
    bearings = tuple(i for i,c in enumerate(conditions) if c=="compression_only")
    if len(bearings)>8:
        raise ValueError("at most eight unilateral seats are supported")
    x = list(accumulate(lengths, initial=ZERO))
    stiffness, applied = _assemble(lengths,loads,rigidities)
    # Most loaded beams retain every seat. Enumerate the remaining
    # contacts as distinct active sets, checking the actual sign and gap.
    for count in reversed(range(len(bearings)+1)):
        for chosen in combinations(bearings,count):
            active = sorted(bilateral.union(chosen))
            if len(active)<2:
                continue
            v, rotation, reaction = _solve(stiffness,applied,active,levels)
            checked = _certify(x,loads,rigidities,levels,active,bilateral,v,rotation,reaction)
            if checked is None:
                continue
            gap, moments, sag, hog = checked
            state = ["hold_down" if i in bilateral and f<0 else "bilateral" if i in bilateral
                     else "bearing" if f>0 else "lifted" if gap[i]>0 else "touching"
                     for i,f in enumerate(reaction)]
            def named(values):
                return dict(zip(labels,map(_reported,values)))
            return dict(dual_verified=True, support_reactions_N=named(reaction),
                        support_displacements_m=named(v), support_gaps_m=named(gap),
                        support_rotations_rad=named(rotation), support_moments_Nm=named(moments),
                        support_states=dict(zip(labels,state)), M_max_sagging_Nm=_reported(sag),
                        M_max_hogging_Nm=_reported(hog), M_max_Nm=_reported(max(sag,hog)),
                        force_equilibrium_residual_N=0., moment_equilibrium_residual_Nm=0.)
    raise ValueError("no restrained equilibrium satisfies every unilateral contact")
