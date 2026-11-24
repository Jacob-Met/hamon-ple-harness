"""Source-area qualification for clamped convex rational quadratic IFC profiles."""
import math
from fractions import Fraction


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("Nonfinite rational profile parameter")
    return Fraction(value)


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _curve_fraction(weight):
    """Fraction of the oriented control triangle between a span and its chord.

    Under an affine map to endpoints (0,0), (0,1) and control (1,0),
    Green's formula gives integral[0,1] 2*w*t^2 / D^2 dt, where
    D=1+2*(w-1)*t*(1-t). Integrate the binomial expansion term by term.
    The remainder bound uses |D-1| <= |w-1|/2 <= 1/2 over the entire span.
    """
    if not Fraction(1, 2) <= weight <= 2:
        raise ValueError("Rational middle weight outside the qualified range")
    terms = 64
    term = 2*weight/3
    value = term
    for k in range(terms-1):
        term *= -2*(weight-1)*(k+2)*(k+3) / ((2*k+5)*(2*k+4))
        value += term
    q = abs(weight-1)/2
    error = 2*weight/3 * q**terms * ((terms+1)-terms*q) / (1-q)**2
    return value, error


def rational_profile_area(curve):
    points = [tuple(_number(v) for v in p.Coordinates) for p in curve.ControlPointsList]
    weights = [_number(w) for w in curve.WeightsData]
    knots = [_number(k) for k in curve.Knots]
    if (curve.Degree != 2 or curve.ClosedCurve is not True or curve.SelfIntersect is not False
            or not 7 <= len(points) <= 65 or len(points) % 2 != 1
            or len(weights) != len(points) or any(w <= 0 for w in weights)
            or any(len(p) != 2 for p in points) or points[0] != points[-1]
            or len(knots) != (len(points)+1)//2 or any(a >= b for a,b in zip(knots, knots[1:]))
            or list(curve.KnotMultiplicities) != [3]+[2]*(len(knots)-2)+[3]):
        raise ValueError("Expected a closed clamped piecewise quadratic rational profile")
    boundary = points[:-1]
    if len(set(boundary)) != len(boundary):
        raise ValueError("Repeated rational profile control vertex")
    direction = 0
    # Every directed side of the control polygon must support its convex hull.
    # Positive weights keep each span in its control triangle. A quadratic
    # rational span has the signed curvature of that triangle and follows the
    # adjacent control edges at its endpoints. The cyclic support test below
    # therefore implies one convex boundary; only then are chords inner bounds.
    for a,b in zip(boundary, boundary[1:]+boundary[:1]):
        for c in boundary:
            turn = _cross(a,b,c)
            if turn:
                sign = 1 if turn > 0 else -1
                if direction and direction != sign:
                    raise ValueError("Nonconvex or retraced rational profile control polygon")
                direction = sign
    if not direction:
        raise ValueError("Collinear rational profile")
    twice_area = error = Fraction(0)
    origin = points[0]
    factors = {}
    for i in range(0, len(points)-1, 2):
        a,b,c = points[i:i+3]
        if weights[i] != weights[i+2]:
            raise ValueError("Rational span endpoint weights must agree")
        w = weights[i+1]/weights[i]
        if w not in factors:
            factors[w] = _curve_fraction(w)
        fraction, bound = factors[w]
        triangle = _cross(a,b,c)
        twice_area += _cross(origin,a,c) + triangle*fraction
        error += abs(triangle)*bound
    if twice_area*direction <= 0 or error > abs(twice_area)*Fraction(1,10**14):
        raise ValueError("Rational profile area is unresolved")
    result = float(abs(twice_area)/2)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Rational profile area is outside the finite output range")
    return result



def _subdivide(spans):
    def midpoint(p, q):
        return tuple((a+b)/2 for a,b in zip(p,q))
    result = []
    for a,b,c in spans:
        left, right = midpoint(a,b), midpoint(b,c)
        mid = midpoint(left,right)
        result.extend(((a,left,mid), (mid,right,c)))
    return result


def _boundaries(spans):
    def planar(p):
        return p[0]/p[2], p[1]/p[2]
    return ([planar(a) for a,_,_ in spans],
            [planar(p) for a,b,_ in spans for p in (a,b)])


def _contains(poly, point):
    cross = [_cross(a,b,point) for a,b in zip(poly,poly[1:]+poly[:1])]
    return min(cross) >= 0 or max(cross) <= 0


def _edge_point(poly, approximate):
    # GEOS suggests a witness. Any clamped point on an inscribed chord supplies
    # a valid upper bound; the selected edge need not be the nearest one.
    x,y = approximate
    best_dist = math.inf
    for a,b in zip(poly,poly[1:]+poly[:1]):
        ax,ay = map(float,a)
        dx,dy = float(b[0]-a[0]),float(b[1]-a[1])
        squared = dx*dx+dy*dy
        if not squared:
            continue
        t = min(1., max(0., ((x-ax)*dx+(y-ay)*dy)/squared))
        dist = math.hypot(x-(ax+t*dx),y-(ay+t*dy))
        if dist < best_dist:
            best_dist, edge = dist, (a,b,Fraction(t))
    if not math.isfinite(best_dist):
        raise ValueError("Rational profile witness is outside the numeric search range")
    a,b,t = edge
    return tuple(x+t*(y-x) for x,y in zip(a,b))


def _planar_bounds(inner_a, outer_a, inner_b, outer_b):
    from shapely.geometry import Polygon
    from shapely.ops import nearest_points

    def polygon(points):
        return Polygon([(float(x),float(y)) for x,y in points])

    a,b = polygon(inner_a),polygon(inner_b)
    overlap = a.intersection(b)
    if not overlap.is_empty:
        approx = overlap.representative_point()
        witness = Fraction(approx.x),Fraction(approx.y)
        if _contains(inner_a,witness) and _contains(inner_b,witness):
            return Fraction(0),Fraction(0)
    p,q = nearest_points(a,b)
    p,q = _edge_point(inner_a,(p.x,p.y)),_edge_point(inner_b,(q.x,q.y))
    upper = (p[0]-q[0])**2+(p[1]-q[1])**2
    p,q = nearest_points(polygon(outer_a),polygon(outer_b))
    dx,dy = Fraction(q.x)-Fraction(p.x),Fraction(q.y)-Fraction(p.y)
    if not dx and not dy:
        return Fraction(0),upper
    # A support gap on any axis gives an exact lower bound on distance.
    gap = min(x*dx+y*dy for x,y in outer_b)-max(x*dx+y*dy for x,y in outer_a)
    return max(Fraction(0),gap)**2/(dx*dx+dy*dy),upper


def _root(value, upward):
    squared = float(value)
    if value and squared < float.fromhex("0x1p-1022"):
        raise ValueError("Distance bound below supported output scale")
    x = math.sqrt(squared)
    while (Fraction(x)**2 < value if upward else Fraction(x)**2 > value):
        x = math.nextafter(x, math.inf if upward else 0.)
    return x


def _distance(left, right, clearance):
    za,zb = left["z"],right["z"]
    dz = max(Fraction(0),za[0]-zb[1],zb[0]-za[1])
    def status(sq):
        if sq <= Fraction(1e-9)**2:
            return "hard_clash"
        threshold = clearance-1e-9
        return "clearance_violation" if threshold>0 and sq<Fraction(threshold)**2 else "clear"

    spans = [left["spans"],right["spans"]]
    for level in range(10):
        pairs = [_boundaries(s) if s else (v["rect"],v["rect"]) for s,v in zip(spans,(left,right))]
        low,high = _planar_bounds(*pairs[0],*pairs[1])
        low,high = low+dz*dz, high+dz*dz
        if low>high:
            raise ValueError("Inconsistent rational distance enclosure")
        lo,hi = _root(low,False),_root(high,True)
        if status(low)==status(high) and hi-lo<=1e-8:
            return status(low),lo,hi
        if any(len(s)*2>4096 for s in spans):
            break
        spans = [_subdivide(s) if s else [] for s in spans]
    raise ValueError("Rational clearance threshold or distance remains unresolved")


def rational_scene(model, bodies, scale, required_clearance, cross_only, discipline):
    from ifcopenshell.util.placement import get_axis2placement, get_local_placement
    length = Fraction(str(scale))
    if not 0 < length or len(bodies)>32 or not math.isfinite(required_clearance) or required_clearance < 0:
        raise ValueError("Outside bounded rational scene or units")
    products = []
    for product,kind,args,reps,items,ring in bodies:
        if (len(reps)!=1 or str(reps[0].RepresentationIdentifier).upper()!="BODY"
                or str(reps[0].RepresentationType).upper()!="SWEPTSOLID" or len(items)!=1
                or items[0].is_a()!="IfcExtrudedAreaSolid"):
            raise ValueError("Rational scenes require one direct swept Body")
        item = items[0]
        profile = item.SweptArea
        if profile.ProfileType != "AREA":
            raise ValueError("Expected a material AREA profile")
        placement = product.ObjectPlacement
        seen = set()
        while placement is not None:
            if not placement.is_a("IfcLocalPlacement") or placement.id() in seen:
                raise ValueError("Rational products require acyclic local placements")
            seen.add(placement.id())
            placement = placement.PlacementRelTo
        matrix = get_local_placement(product.ObjectPlacement)
        if item.Position is not None:
            matrix = matrix @ get_axis2placement(item.Position)
        if (ring is not None and ring.is_a()=="IfcRationalBSplineCurveWithKnots"
                and profile.is_a()=="IfcArbitraryClosedProfileDef"):
            area = rational_profile_area(ring)
            coords = [p.Coordinates for p in ring.ControlPointsList]
            weights = [_number(w) for w in ring.WeightsData]
        elif profile.is_a()=="IfcRectangleProfileDef":
            dx,dy = _number(profile.XDim)/2,_number(profile.YDim)/2
            if dx<=0 or dy<=0:
                raise ValueError("Invalid rectangular peer")
            if profile.Position is not None:
                matrix = matrix @ get_axis2placement(profile.Position)
            area = float(4*dx*dy)
            coords = [(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
            weights = []
        else:
            raise ValueError("Rational scene contains an unsupported peer profile")
        # Only signed cardinal world frames and a right vertical extrusion are
        # qualified. This makes every transformed hull coordinate rational.
        linear = matrix[:3,:3]
        if (any(v not in (-1.,0.,1.) for v in linear.flat)
                or any(sum(abs(v) for v in linear[i,:])!=1 for i in range(3))
                or any(sum(abs(v) for v in linear[:,i])!=1 for i in range(3))
                or abs(linear[2,2])!=1):
            raise ValueError("Rational prisms require a cardinal planar world frame")
        direction = item.ExtrudedDirection.DirectionRatios
        if (len(direction)!=3 or direction[0]!=0 or direction[1]!=0 or not _number(direction[2])
                or _number(item.Depth)<=0):
            raise ValueError("Rational prisms require a positive right extrusion")
        xy = [(length*(Fraction(float(matrix[0,3]))+int(linear[0,0])*Fraction(x)+int(linear[0,1])*Fraction(y)),
               length*(Fraction(float(matrix[1,3]))+int(linear[1,0])*Fraction(x)+int(linear[1,1])*Fraction(y))) for x,y in coords]
        z0 = length*Fraction(float(matrix[2,3]))
        depth = length*_number(item.Depth)
        z1 = z0+int(linear[2,2])*(1 if direction[2]>0 else -1)*depth
        h = [(x*w,y*w,w) for (x,y),w in zip(xy,weights)]
        spans = [tuple(h[i:i+3]) for i in range(0,len(h)-1,2)]
        volume = area*float(length**2)*float(depth)
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("Rational volume outside finite output range")
        products.append(dict(guid=product.GlobalId, discipline=discipline(kind,str(args[2] or "")),
                             spans=spans,rect=[] if spans else xy,z=sorted((z0,z1)),
                             volume=volume))
    rows = []
    for i,a in enumerate(products):
        for b in products[i+1:]:
            if cross_only and a["discipline"]==b["discipline"]:
                continue
            status,lo,hi = _distance(a,b,required_clearance)
            rows.append(dict(element_a=a["guid"],element_b=b["guid"],status=status,
                             min_distance_m=(lo+hi)/2,distance_bounds_m=[lo,hi],
                             required_clearance_m=required_clearance,overlap_aabb_volume_m3=None,
                             witness_point_m=None,closest_points_m=None))
    hard = [r for r in rows if r["status"]=="hard_clash"]
    near = [r for r in rows if r["status"]=="clearance_violation"]
    return dict(solid_count=len(products),pair_count=len(rows),hard_clash_count=len(hard),
                clearance_violation_count=len(near),min_clearance_m=min((r["min_distance_m"] for r in rows),default=None),
                total_solid_volume_m3=math.fsum(p["volume"] for p in products),clashes=hard+near,all_pairs=rows,
                geometry_basis="source rational quadratic areas and nested rational control polygons")
