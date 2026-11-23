"""Exact closest features of triangles in represented binary64 coordinates."""
from fractions import Fraction
import math
from mpmath.ctx_iv import MPIntervalContext
from hamon_step_analytic import _round_public

def point(values):
    if len(values) != 3:
        raise ValueError("mesh point must have three coordinates")
    values = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("mesh coordinates must be finite")
    return tuple(Fraction(v) for v in values)

def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))

def add_scaled(a, t, b):
    return tuple(x+t*y for x, y in zip(a, b))

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def squared(a, b):
    delta = sub(a, b)
    return dot(delta, delta)

def edges(triangle):
    return ((triangle[0],triangle[1]), (triangle[1],triangle[2]),
            (triangle[2],triangle[0]))

def closest_segment(p, a, b):
    direction = sub(b, a)
    norm = dot(direction, direction)
    parameter = max(Fraction(0), min(Fraction(1), dot(sub(p, a), direction)/norm)) if norm else Fraction(0)
    return add_scaled(a, parameter, direction)

def in_triangle(p, triangle, normal):
    return all(dot(cross(sub(b, a), sub(p, a)), normal) >= 0 for a, b in edges(triangle))

def closest_triangle(p, triangle):
    a, b, c = triangle
    normal = cross(sub(b,a),sub(c,a))
    norm = dot(normal,normal)
    if norm:
        projected = add_scaled(p, -dot(sub(p,a),normal)/norm, normal)
        if in_triangle(projected, triangle, normal):
            return projected
    candidates = [closest_segment(p,a,b) for a,b in edges(triangle)]
    return min(candidates, key=lambda q: squared(p,q))

def interior_edge_pair(a,b,c,d):
    u,v,w = sub(b,a),sub(d,c),sub(a,c)
    aa,bb,cc,dd,ee = dot(u,u),dot(u,v),dot(v,v),dot(u,w),dot(v,w)
    determinant = aa*cc-bb*bb
    if determinant <= 0:
        return None
    s,t = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
    if 0 <= s <= 1 and 0 <= t <= 1:
        return add_scaled(a,s,u),add_scaled(c,t,v)
    return None

def segment_triangle(a,b,triangle):
    normal = cross(sub(triangle[1],triangle[0]),sub(triangle[2],triangle[0]))
    direction = sub(b,a)
    denominator = dot(direction,normal)
    if not denominator:
        return None
    parameter = dot(sub(triangle[0],a),normal)/denominator
    if 0 <= parameter <= 1:
        candidate = add_scaled(a,parameter,direction)
        if in_triangle(candidate,triangle,normal):
            return candidate
    return None

def triangle_distance(a,b):
    for start,end in edges(a):
        hit = segment_triangle(start,end,b)
        if hit is not None:
            return Fraction(0),hit,hit
    for start,end in edges(b):
        hit = segment_triangle(start,end,a)
        if hit is not None:
            return Fraction(0),hit,hit
    candidates = [(p,closest_triangle(p,b)) for p in a]
    candidates.extend((closest_triangle(p,a),p) for p in b)
    # Endpoint edge minima are already covered by the vertex/face candidates.
    for start,end in edges(a):
        for other_start,other_end in edges(b):
            pair = interior_edge_pair(start,end,other_start,other_end)
            if pair is not None:
                candidates.append(pair)
    best = min(candidates,key=lambda pair:squared(*pair))
    return squared(*best),*best

def bounds(vertices):
    return tuple(min(p[i] for p in vertices) for i in range(3)), tuple(
        max(p[i] for p in vertices) for i in range(3))

def bounds_distance_squared(a,b):
    low_a,high_a=a
    low_b,high_b=b
    return sum(max(Fraction(0), low_a[i]-high_b[i], low_b[i]-high_a[i])**2
               for i in range(3))

def wire_distance(value):
    if value < 0:
        raise ValueError("mesh squared distance must be nonnegative")
    try:
        return _round_public(lambda iv:[iv.sqrt(iv.mpf(value.numerator)/value.denominator)])[0]
    except (ValueError,OverflowError) as exc:
        raise ValueError("mesh distance exceeds qualified binary64 range") from exc

def lower_distance_bound(value):
    if value == 0:
        return 0.
    iv = MPIntervalContext()
    iv.dps = 80
    lower = iv.sqrt(iv.mpf(value.numerator)/value.denominator).a
    sign, mantissa, exponent, bits = lower._mpi_[0]
    if exponent + bits < -1200:
        return 0.
    exact_lower = (Fraction(mantissa << exponent) if exponent >= 0 else
                   Fraction(mantissa, 1 << -exponent))
    try:
        reported = float(exact_lower)
    except OverflowError:
        reported = float.fromhex("0x1.fffffffffffffp+1023")
    if not math.isfinite(reported):
        reported = float.fromhex("0x1.fffffffffffffp+1023")
    return math.nextafter(reported, 0.) if Fraction(reported)**2 > value else reported

def requested_clearance(value):
    if isinstance(value,bool):
        raise ValueError("mesh clearance must be a finite nonnegative scalar")
    try:
        exact = Fraction(value)
        reported = value if isinstance(value,int) else float(value)
        finite = math.isfinite(float(reported))
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("mesh clearance must be finite") from exc
    if not finite or exact < 0 or (exact and not reported):
        raise ValueError("mesh clearance must be finite, nonnegative and representable")
    if exact and abs(Fraction(reported)-exact) > abs(exact)*Fraction(1,10**12):
        raise ValueError("mesh clearance is not accurately representable")
    return exact,reported

def _upper_sqrt(value):
    if value == 0:
        return 0.
    iv=MPIntervalContext()
    iv.dps=80
    upper=iv.sqrt(iv.mpf(value.numerator)/value.denominator).b
    sign,mantissa,exponent,bits=upper._mpi_[0]
    if bits < 0:
        raise ValueError("mesh witness error bound is not finite")
    if exponent+bits < -1200:
        rounded=0.
    else:
        dyadic=(Fraction(mantissa << exponent) if exponent >= 0 else
                Fraction(mantissa,1 << -exponent))
        rounded=float(-dyadic if sign else dyadic)
    result=math.nextafter(rounded,math.inf)
    if not math.isfinite(result):
        raise ValueError("mesh witness error bound exceeds binary64 range")
    return result

def wire_point(value):
    try:
        reported=tuple(float(v) for v in value)
    except OverflowError as exc:
        raise ValueError("mesh witness exceeds binary64 range") from exc
    if not all(math.isfinite(v) for v in reported):
        raise ValueError("mesh witness must be finite")
    error=sum((Fraction(w)-v)**2 for w,v in zip(reported,value))
    return reported,_upper_sqrt(error)

def point_inside(p, triangles, coherent):
    """Signed exact ray crossings, or orientation-independent shell parity."""
    prepared=[]
    for triangle in triangles:
        normal=cross(sub(triangle[1],triangle[0]),sub(triangle[2],triangle[0]))
        if not any(normal):
            continue
        offset=dot(sub(triangle[0],p),normal)
        if offset == 0 and in_triangle(p,triangle,normal):
            return True
        prepared.append((triangle,normal,offset))
    # Every nondegenerate rational edge/face excludes at most two directions
    # on (1,k,k²). More than eight candidates per face suffices to find a
    # ray with no coplanar face or positive-parameter edge/vertex hit.
    for k in range(1,8*len(prepared)+2):
        direction=(Fraction(1),Fraction(k),Fraction(k*k))
        winding=0
        count=0
        degenerate=False
        for triangle,normal,offset in prepared:
            denominator=dot(direction,normal)
            if not denominator:
                if not offset:
                    degenerate=True
                    break
                continue
            parameter=offset/denominator
            if parameter <= 0:
                continue
            hit=add_scaled(p,parameter,direction)
            orientations=[dot(cross(sub(b,a),sub(hit,a)),normal)
                          for a,b in edges(triangle)]
            if all(value >= 0 for value in orientations):
                if any(value == 0 for value in orientations):
                    degenerate=True
                    break
                count += 1
                winding += 1 if denominator > 0 else -1
        if not degenerate:
            return winding != 0 if coherent else count % 2 == 1
    raise ValueError("mesh containment could not find a nondegenerate exact ray")

def segment_distance(a,b,c,d):
    pairs=[(a,closest_segment(a,c,d)),(b,closest_segment(b,c,d)),
           (closest_segment(c,a,b),c),(closest_segment(d,a,b),d)]
    interior=interior_edge_pair(a,b,c,d)
    if interior is not None:
        pairs.append(interior)
    best=min(pairs,key=lambda pair:squared(*pair))
    return squared(*best),*best

def mesh_triangles(vertices,faces):
    vertices=[point(v) for v in vertices]
    triangles=[]
    for face in faces:
        if len(face) != 3:
            raise ValueError("mesh faces must be triangles")
        indices=[]
        for index in face:
            if isinstance(index,bool) or int(index) != index or not 0 <= int(index) < len(vertices):
                raise ValueError("mesh triangle has an invalid vertex index")
            indices.append(int(index))
        triangles.append(tuple(vertices[i] for i in indices))
    if not vertices or not triangles:
        raise ValueError("mesh clearance needs nonempty represented surfaces")
    return vertices,triangles

def positive_overlap(a,b):
    return all(min(a[1][i],b[1][i]) > max(a[0][i],b[0][i]) for i in range(3))

def overlap_volume(a,b):
    exact=math.prod(max(Fraction(0),min(a[1][i],b[1][i])-max(a[0][i],b[0][i]))
                    for i in range(3))
    try:
        reported=float(exact)
    except OverflowError:
        return None
    return reported if math.isfinite(reported) and (reported or not exact) else None
