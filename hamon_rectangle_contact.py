"""One positive-affine pressure field on a rigid rectangular base.

Moments are first moments: integral(x*p)=Mx and integral(y*p)=My.
This supplies normal contact equilibrium, not bearing capacity or settlement.
"""
import math
from fractions import Fraction
from hamon_bearing_capacity import _strip_quantity


def _clip_pressure(rx, ry):
    """Clip [0,2]^2 against 1-rx*x-ry*y >= 0, from the loaded corner."""
    points = [(0., 0.), (2., 0.), (2., 2.), (0., 2.)]
    values = [math.fsum((1., -rx*x, -ry*y)) for x, y in points]
    out = []
    def append(point):
        if not out or point[:2] != out[-1][:2]: out.append(point)
    for i, point in enumerate(points):
        j = (i-1) % 4
        before, after = values[j], values[i]
        if (before >= 0) != (after >= 0):
            inside, outside = (j, i) if before >= 0 else (i, j)
            p, q = points[inside], points[outside]
            t = values[inside]/(values[inside]-values[outside])
            # Start inside: tiny patches must not subtract two base widths.
            append((p[0]+t*(q[0]-p[0]), p[1]+t*(q[1]-p[1]), 0.))
        if after >= 0: append((*point, after))
    if len(out)>1 and out[0][:2] == out[-1][:2]: out.pop()
    corner = next(i for i, point in enumerate(out) if point[:2] == (0., 0.))
    return out[corner:]+out[:corner]


def _shape_integrals(points):
    """Polynomial triangle integrals in coordinates from the loaded corner."""
    terms = [[] for _ in range(9)]
    for a, b in zip(points[1:-1], points[2:]):
        x,y,p = a; X,Y,q = b
        area = (x*Y-X*y)/2
        if area < 0: raise ArithmeticError('Inconsistent contact winding')
        ps = 1+p+q
        values = (area, area*(x+X)/3, area*(y+Y)/3,
                  area*(x*x+x*X+X*X)/6,
                  area*(2*x*y+x*Y+X*y+2*X*Y)/12,
                  area*(y*y+y*Y+Y*Y)/6, area*ps/3,
                  area*((x+X)*ps+x*p+X*q)/12,
                  area*((y+Y)*ps+y*p+Y*q)/12)
        for term,value in zip(terms,values): term.append(value)
    return tuple(math.fsum(term) for term in terms)


def _rectangle_shape(u, v):
    """Inverse centroid problem for exact reflected normalized offsets."""
    if u+v <= Fraction(1,3):
        peak = 1+3*u+3*v
        return float(3*u/peak),float(3*v/peak),'full_rectangle',0
    dx,dy = float(1-u),float(1-v)
    if u >= Fraction(1,2) and v >= Fraction(1,2):
        return 1/(4*dx),1/(4*dy),'corner_triangle',0
    for swapped in (False,True):
        along,across = (v,u) if swapped else (u,v)
        if across > Fraction(1,2): continue
        cross = float(across)
        ratio = 3*cross/(1+math.sqrt(1-3*cross*cross))
        width = float(1-along)*(3+ratio*ratio)/(1+ratio*ratio)
        variation = ratio*width
        if width+variation <= 2:
            a,b = 1/(width+variation),variation/(width+variation)
            return (b,a,'edge_trapezoid',0) if swapped else (a,b,'edge_trapezoid',0)
    den = 1+3*u+3*v
    rx,ry = min(float(3*u/den),.5),min(float(3*v/den),.5)
    for iteration in range(40):
        _,ix,iy,ixx,ixy,iyy,f,mx,my = _shape_integrals(_clip_pressure(rx,ry))
        cx,cy = mx/f,my/f
        ex,ey = cx-dx,cy-dy
        error = max(abs(ex),abs(ey))
        if error <= 3e-15: return rx,ry,'clipped_pentagon',iteration
        a,b = (cx*ix-ixx)/f,(cx*iy-ixy)/f
        c,d = (cy*ix-ixy)/f,(cy*iy-iyy)/f
        det = a*d-b*c
        if det <= 0 or not math.isfinite(det): raise ArithmeticError('Singular contact inverse')
        sx,sy = (-d*ex+b*ey)/det,(c*ex-a*ey)/det
        step = 1.
        for _ in range(45):
            nx,ny = rx+step*sx,ry+step*sy
            if 0 <= nx <= .5 and 0 <= ny <= .5:
                m = _shape_integrals(_clip_pressure(nx,ny))
                residual = max(abs(m[7]/m[6]-dx),abs(m[8]/m[6]-dy))
                if residual < error:
                    rx,ry = nx,ny
                    break
            step /= 2
        else: raise ArithmeticError('Contact inverse did not converge')
    raise ArithmeticError('Contact inverse exceeded its iteration limit')


def _integrate_displayed(points, pressures):
    """Integrate the actual displayed polygon/pressures using rational arithmetic."""
    a = tuple(map(Fraction, points[0])); pa = Fraction(pressures[0])
    area_total = force = mx = my = Fraction(0)
    winding = None
    for i in range(1, len(points)-1):
        b, c = (tuple(map(Fraction, points[k])) for k in (i, i+1))
        pb, pc = map(Fraction, pressures[i:i+2])
        cross = (b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1])
        if not cross or (winding is not None and (cross > 0) != winding):
            raise ValueError('Contact polygon is degenerate or has inconsistent winding')
        winding = cross > 0
        area = abs(cross)/2; ps = pa+pb+pc
        area_total += area
        force += area*ps/3
        mx += area*((a[0]+b[0]+c[0])*ps+a[0]*pa+b[0]*pb+c[0]*pc)/12
        my += area*((a[1]+b[1]+c[1])*ps+a[1]*pa+b[1]*pb+c[1]*pc)/12
    return area_total, force, mx, my


def _quantity(value, name, unit):
    return Fraction(_strip_quantity(value, name, unit))


def _render(value, name):
    if value is None: return None
    result = float(value)
    if not math.isfinite(result) or (value and not result):
        raise ValueError(name+' is outside the representable numeric range')
    return result


def _close(value, target):
    # An absolute ULP floor can approve zero or negative subnormal pressure
    # against a positive load. Rational comparisons need no such floor.
    return abs(value-target) <= abs(target)/10**9


def verify_rigid_rectangle_contact(spec):
    """Solve integral(p)=N, integral(x*p)=Mx, integral(y*p)=My on a rectangle."""
    calculation = 'rigid_rectangle_contact'
    failure = dict(ok=False, calculation=calculation, conservation_verified=False)
    claims = ('contact_area_m2','q_min_Pa','q_max_Pa','eccentricity_m',
              'corner_pressure_Pa','contact_polygon_m','contact_vertex_pressure_Pa',
              'contact_regime')
    allowed = {'mode','kind','calculation','pressure_model','base_width_m','base_center_m',
               'base_length_m','vertical_resultant_N','moment_x_Nm','moment_y_Nm'}
    allowed.update('proposed_'+key for key in claims)
    try:
        unknown = set(spec)-allowed
        if unknown: raise ValueError('Unsupported rectangle-contact inputs: '+', '.join(sorted(unknown)))
        if spec.get('calculation',calculation)!=calculation: raise ValueError('Expected calculation='+calculation)
        if spec.get('pressure_model','positive_part_affine')!='positive_part_affine':
            raise ValueError('Only pressure_model=positive_part_affine is represented')
        if 'proposed_contact_vertex_pressure_Pa' in spec and 'proposed_contact_polygon_m' not in spec:
            raise ValueError('Vertex pressure claims require a proposed contact polygon')
        b,l = (_quantity(spec[key],key,'m') for key in ('base_width_m','base_length_m'))
        if b<=0 or l<=0: raise ValueError('Base dimensions must be positive')
        center = spec.get('base_center_m',[0,0])
        if not isinstance(center,(list,tuple)) or len(center)!=2:
            raise ValueError('base_center_m must contain two coordinates')
        cx,cy = (_quantity(value,'base_center_m','m') for value in center)
        n = _quantity(spec['vertical_resultant_N'],'vertical_resultant_N','N')
        mx,my = (_quantity(spec[key],key,'N*m') for key in ('moment_x_Nm','moment_y_Nm'))
        if n<0 or (n==0 and (mx or my)) or (n>0 and
                (2*abs(mx-n*cx)>=n*b or 2*abs(my-n*cy)>=n*l)):
            return {**failure,'equilibrium_status':'no_finite_equilibrium',
                    'reason':'The normal load cannot be balanced by finite nonnegative area pressure'}
        out,exact = _solve_rectangle(b,l,n,mx,my,cx,cy)
        out.update(calculation=calculation,pressure_model='positive_part_affine',
                   base_width_m=float(b),base_length_m=float(l),vertical_resultant_N=float(n),
                   base_center_m=[float(cx),float(cy)],
                   moment_x_Nm=float(mx),moment_y_Nm=float(my),
                   moment_convention='Mx=integral(x*p); My=integral(y*p)')
        checks = _claim_checks(spec,out,exact,(b,l),n,(mx,my))
        out['claim_checks'] = checks
        out['conservation_verified'] = out['output_geometry_verified'] and all(checks.values())
        return out
    except (KeyError,TypeError,ValueError,ArithmeticError,SyntaxError) as exc:
        return {**failure,'reason':str(exc)}


def _solve_rectangle(b,l,n,mx,my,cx,cy):
    if not n:
        exact = dict(contact_area_m2=None,q_min_Pa=Fraction(0),q_max_Pa=Fraction(0))
        return dict(ok=True,equilibrium_status='finite',contact_regime='unloaded',
                    footprint_shape=None,eccentricity_m=None,contact_polygon_m=None,
                    contact_vertex_pressure_Pa=[],corner_pressure_Pa=[0.,0.,0.,0.],
                    contact_area_m2=None,q_min_Pa=0.,q_max_Pa=0.,pressure_plane=None,
                    integrated_vertical_resultant_N=0.,integrated_moment_x_Nm=0.,
                    integrated_moment_y_Nm=0.,force_residual_N=0.,moment_residual_Nm=[0.,0.],
                    output_geometry_verified=True),exact
    local_mx,local_my = mx-n*cx,my-n*cy
    u,v = 2*abs(local_mx)/(n*b),2*abs(local_my)/(n*l)
    rx,ry,shape,_ = _rectangle_shape(u,v)
    if not all(math.isfinite(x) for x in (rx,ry)): raise ValueError('Unresolved contact gradients')
    polygon = _clip_pressure(rx,ry)
    values = _shape_integrals(polygon)
    if values[6]<=0: raise ValueError('Contact area is unresolved in numeric precision')
    jacobian = b*l/4
    peak = n/(jacobian*Fraction(values[6]))
    sx,sy = (1 if local_mx>=0 else -1),(1 if local_my>=0 else -1)
    points = [[_render(cx+sx*b/2*(1-Fraction(x)),'contact_polygon_m'),
               _render(cy+sy*l/2*(1-Fraction(y)),'contact_polygon_m')] for x,y,_ in polygon]
    pressures = [_render(peak*Fraction(p),'contact_vertex_pressure_Pa') for _,_,p in polygon]
    if sx*sy<0:
        points = points[:1]+points[:0:-1]
        pressures = pressures[:1]+pressures[:0:-1]
    area,f,fx,fy = _integrate_displayed(points,pressures)
    expected_area = jacobian*Fraction(values[0])
    plane = dict(reference_point_m=[_render(cx+sx*b/2,'reference_point'),_render(cy+sy*l/2,'reference_point')],
                 pressure_at_reference_Pa=_render(peak,'peak pressure'),
                 gradient_Pa_per_m=[_render(sx*peak*Fraction(rx)/(b/2),'gradient'),
                                    _render(sy*peak*Fraction(ry)/(l/2),'gradient')])
    corners = [_render(peak*Fraction(max(0.,math.fsum((1.,-rx*(1-sx*x),-ry*(1-sy*y))))),'corner pressure')
               for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    exact = dict(contact_area_m2=expected_area,q_min_Pa=Fraction(min(corners)),q_max_Pa=peak)
    geometry = (_close(area,expected_area) and abs(f-n)<=n/10**9
                and abs(fx-mx)<=n*b/10**9 and abs(fy-my)<=n*l/10**9)
    for point,p in zip(points,pressures):
        affine = Fraction(plane['pressure_at_reference_Pa'])+sum(
            Fraction(g)*(Fraction(x)-Fraction(x0)) for g,x,x0 in
            zip(plane['gradient_Pa_per_m'],point,plane['reference_point_m']))
        geometry = geometry and abs(max(Fraction(0),affine)-Fraction(p))<=peak/10**9
    out = dict(ok=True,equilibrium_status='finite',
               contact_regime='full_contact' if u+v<=Fraction(1,3) else 'partial_contact',
               footprint_shape=shape,eccentricity_m=[_render(local_mx/n,'eccentricity'),_render(local_my/n,'eccentricity')],
               contact_polygon_m=points,contact_vertex_pressure_Pa=pressures,
               corner_pressure_Pa=corners,pressure_plane=plane,
               **{key:_render(value,key) for key,value in exact.items()},
               integrated_vertical_resultant_N=_render(f,'integrated force'),
               integrated_moment_x_Nm=_render(fx,'integrated moment'),
               integrated_moment_y_Nm=_render(fy,'integrated moment'),
               force_residual_N=float(f-n),moment_residual_Nm=[float(fx-mx),float(fy-my)],
               output_geometry_verified=bool(geometry))
    return out,exact


def _claim_checks(spec,out,exact,dimensions,normal,moments):
    checks = {}
    for field,unit in [('contact_area_m2','m**2'),('q_min_Pa','Pa'),('q_max_Pa','Pa')]:
        key = 'proposed_'+field
        if key in spec:
            checks[key] = spec[key] is None if exact[field] is None else _close(
                _quantity(spec[key],key,unit),exact[field])
    for field,unit in [('eccentricity_m','m'),('corner_pressure_Pa','Pa')]:
        key = 'proposed_'+field
        if key not in spec: continue
        target,value = out[field],spec[key]
        if target is None:
            checks[key] = value is None
        else:
            checks[key] = (isinstance(value,(list,tuple)) and len(value)==len(target)
                           and all(_close(_quantity(v,key,unit),Fraction(t))
                                   for v,t in zip(value,target)))
    key = 'proposed_contact_polygon_m'
    if key in spec:
        if out['contact_polygon_m'] is None:
            checks[key] = spec[key] is None and spec.get('proposed_contact_vertex_pressure_Pa') in (None,[])
        else:
            checks[key] = _polygon_claim(spec,out,exact,dimensions,normal,moments)
    if 'proposed_contact_regime' in spec:
        checks['proposed_contact_regime'] = spec['proposed_contact_regime']==out['contact_regime']
    return checks


def _polygon_claim(spec,out,exact,dimensions,normal,moments):
    key = 'proposed_contact_polygon_m'
    proposed = spec[key]
    if not isinstance(proposed,(list,tuple)) or not 3<=len(proposed)<=6: return False
    if not all(isinstance(p,(list,tuple)) and len(p)==2 for p in proposed): return False
    points = [[_quantity(v,key,'m') for v in p] for p in proposed]
    pressures = spec.get('proposed_contact_vertex_pressure_Pa')
    if 'proposed_contact_vertex_pressure_Pa' in spec and pressures is None: return False
    if pressures is not None:
        if not isinstance(pressures,(list,tuple)) or len(pressures)!=len(points): return False
        pressures = [_quantity(v,'proposed_contact_vertex_pressure_Pa','Pa') for v in pressures]
    if points[0]==points[-1]:
        points.pop()
        if pressures is not None and pressures.pop()!=pressures[0]: return False
    expected = [tuple(map(Fraction,p)) for p in out['contact_polygon_m']]
    n = len(expected)
    if len(points)!=n: return False
    extents = [max(p[k] for p in expected)-min(p[k] for p in expected) for k in (0,1)]
    tolerance = [v/10**9+8*Fraction(math.ulp(float(v))) for v in extents]
    order = None
    for direction in (1,-1):
        for start in range(n):
            trial = [(start+direction*j)%n for j in range(n)]
            if all(abs(points[j][k]-expected[index][k])<=tolerance[k]
                   for j,index in enumerate(trial) for k in (0,1)):
                order = trial
                break
        if order is not None: break
    if order is None: return False
    defaults = [Fraction(out['contact_vertex_pressure_Pa'][i]) for i in order]
    if pressures is None:
        peak = _quantity(spec.get('proposed_q_max_Pa',out['q_max_Pa']),'proposed_q_max_Pa','Pa')
        pressures = [p*peak/Fraction(out['q_max_Pa']) for p in defaults]
    if not all(p>=0 and (p==0 if q==0 else _close(p,q)) for p,q in zip(pressures,defaults)): return False
    area,f,mx,my = _integrate_displayed(points,pressures)
    return (_close(area,exact['contact_area_m2']) and abs(f-normal)<=abs(normal)/10**9
            and all(abs(a-b)<=abs(normal*d)/10**9 for a,b,d in zip((mx,my),moments,dimensions)))
