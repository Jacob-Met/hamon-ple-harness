"""Compression-only affine contact over three actual support vertices."""
from fractions import Fraction as F

from hamon_rectangle_contact import _claim_checks, _close, _integrate_displayed, _quantity, _render


def _det(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _coordinates(vertices, point):
    a, b, c = vertices
    det = _det(a, b, c)
    return (_det(point, b, c)/det, _det(a, point, c)/det, _det(a, b, point)/det)


def _centroid_terms(width, ratio):
    # A is unloaded, B has unit corner pressure, C has pressure ratio.
    # width and k*width are the contact depths on BA and CA. Factoring out
    # width keeps the centroid test conditioned for small representable edge depths.
    if width == 1.:
        return 1., (2.+ratio)/(1.+ratio)
    k = ratio / ((1.-width) + ratio*width)
    ccut = k*width
    total = k*(1.+ratio) + (1.-ccut)
    a = (k*k*(1.+ratio) + (1.-ccut)*(1.+k))/total
    b = (k*(2.+ratio) + (1.-ccut)*(3.-width))/total
    return a, b  # 4*lambda_A/width and 4*lambda_B


def _bisect(lo, hi, fn):
    # Endpoints bracket zero in increasing orientation. Any zero supplies a
    # candidate. The separate displayed force/three-coordinate gate is final.
    if fn(lo) >= 0:
        return lo
    if fn(hi) <= 0:
        return hi
    for _ in range(64):
        mid = (lo+hi)/2
        if mid == lo or mid == hi:
            break
        if fn(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def _pressures(centroid):
    """Nodal affine values up to a positive scalar, in the supplied vertex order."""
    low, middle, high = sorted(range(3), key=centroid.__getitem__)
    a, b = centroid[low], centroid[high]
    quarter = F(1, 4)
    if a >= quarter:
        return tuple(4*x-1 for x in centroid), 'full_triangle'
    if centroid[middle] <= quarter:
        # The positive corner carries the linear tent over a subtriangle.
        # Its two zero intersections are 4*lambda_i along the original edges.
        p = [F(1)]*3
        for i in (low, middle):
            p[i] -= 1/(4*centroid[i])
        return p, 'one_positive_corner'

    depth = float(4*a)
    if not depth:
        raise ValueError('The contact depth is outside binary64 range')
    def width_at(r):
        # width/4 <= lambda_A <= width/3. Keep the interval local to the
        # measured depth, then solve on dimensionless width/(4*lambda_A).
        t = _bisect(0., 1., lambda t: t*_centroid_terms(depth*t, r)[0]-1.)
        return depth*t
    # At r=0, lambda_B=3/4-lambda_A. At r=1, lambda_B=(1-lambda_A)/2.
    ratio = _bisect(0., 1., lambda r: float(4*b)-_centroid_terms(width_at(r), r)[1])
    width = F(width_at(ratio))
    if not width:
        raise ValueError('Unresolved contact edge')
    p = [F(0)]*3
    p[low], p[middle], p[high] = -(1-width)/width, F(ratio), F(1)
    return p, 'two_positive_corners'


def _positive_polygon(vertices, p):
    """Clip the triangle to its positive affine nodal field in exact coordinates."""
    points, pressures = [], []
    for i, point in enumerate(vertices):
        j = (i-1) % 3
        if p[j]*p[i] < 0:
            fraction = p[j]/(p[j]-p[i])
            points.append(tuple(x+fraction*(y-x) for x, y in zip(vertices[j], point)))
            pressures.append(F(0))
        if p[i] >= 0:
            points.append(point)
            pressures.append(p[i])
    return points, pressures


def _normalized_balance(vertices, points, pressures, centroid):
    # Axis-aligned moment tolerances alone can hide a large error across
    # an oblique skinny triangle. Check all three intrinsic first moments.
    bary = [_coordinates(vertices, p) for p in points]
    if any(min(q) < -F(1, 10**9) for q in bary):
        return False
    _, force, mx, my = _integrate_displayed([q[1:] for q in bary], pressures)
    if force <= 0:
        return False
    actual = (1-(mx+my)/force, mx/force, my/force)
    return all(_close(x, y) for x, y in zip(actual, centroid))


def verify_rigid_triangle_contact(spec):
    calculation = 'rigid_triangle_contact'
    failure = dict(ok=False, calculation=calculation, conservation_verified=False)
    claims = ('contact_area_m2', 'q_min_Pa', 'q_max_Pa', 'eccentricity_m', 'corner_pressure_Pa',
              'contact_polygon_m', 'contact_vertex_pressure_Pa', 'contact_regime')
    allowed = {'mode', 'kind', 'calculation', 'pressure_model', 'base_vertices_m', 'vertical_resultant_N',
               'moment_x_Nm', 'moment_y_Nm'} | {'proposed_'+k for k in claims}
    try:
        if set(spec)-allowed or spec.get('calculation', calculation) != calculation:
            raise ValueError('Unsupported triangular-contact input or calculation')
        if spec.get('pressure_model', 'positive_part_affine') != 'positive_part_affine':
            raise ValueError('Expected positive_part_affine pressure')
        if 'proposed_contact_vertex_pressure_Pa' in spec and 'proposed_contact_polygon_m' not in spec:
            raise ValueError('Vertex pressure claims require a proposed contact polygon')
        raw = spec['base_vertices_m']
        if not isinstance(raw, (list, tuple)) or len(raw) != 3 or any(
            not isinstance(p, (list, tuple)) or len(p) != 2 for p in raw):
            raise ValueError('Expected three planar vertices')
        vertices = [tuple(_quantity(x, 'base_vertices_m', 'm') for x in p) for p in raw]
        if not _det(*vertices):
            raise ValueError('Collinear vertices have zero support area')
        n = _quantity(spec['vertical_resultant_N'], 'vertical_resultant_N', 'N')
        moments = tuple(_quantity(spec[k], k, 'N*m') for k in ('moment_x_Nm', 'moment_y_Nm'))
        if n < 0 or (not n and any(moments)):
            return {**failure, 'equilibrium_status':'no_finite_equilibrium'}
        dims = [max(p[j] for p in vertices)-min(p[j] for p in vertices) for j in (0,1)]
        if not n:
            exact = dict(contact_area_m2=None, q_min_Pa=F(0), q_max_Pa=F(0))
            out = dict(contact_area_m2=None, q_min_Pa=0., q_max_Pa=0., contact_regime='unloaded',
                       footprint_shape=None, eccentricity_m=None, corner_pressure_Pa=[0.]*3,
                       contact_polygon_m=None, contact_vertex_pressure_Pa=[], pressure_plane=None,
                       integrated_vertical_resultant_N=0., integrated_moment_x_Nm=0.,
                       integrated_moment_y_Nm=0., output_geometry_verified=True)
        else:
            center = tuple(x/n for x in moments)
            centroid = _coordinates(vertices, center)
            if min(centroid) <= 0:
                return {**failure, 'equilibrium_status':'no_finite_equilibrium',
                        'reason':'A positive finite area load has a centroid strictly inside the triangle'}
            p, shape = _pressures(centroid)
            clipped, heights = _positive_polygon(vertices, p)
            area, volume, *_ = _integrate_displayed(clipped, heights)
            scale = n/volume
            corner = [max(F(0), h)*scale for h in p]
            exact = dict(contact_area_m2=area, q_min_Pa=min(corner), q_max_Pa=max(corner))
            points = [[_render(x, 'contact vertex') for x in xy] for xy in clipped]
            pressures = [_render(h*scale, 'vertex pressure') for h in heights]
            rendered = [tuple(map(F, xy)) for xy in points]
            ra, rf, rx, ry = _integrate_displayed(rendered, pressures)
            # Recover the global affine field from three nodal values. Anchor it
            # at the maximal corner to avoid a translated intercept subtraction.
            va, vb, vc = vertices
            db, dc = (p[1]-p[0])*scale, (p[2]-p[0])*scale
            det = _det(*vertices)
            gx = (db*(vc[1]-va[1])-dc*(vb[1]-va[1]))/det
            gy = ((vb[0]-va[0])*dc-(vc[0]-va[0])*db)/det
            high = max(range(3), key=p.__getitem__)
            reference = [float(x) for x in vertices[high]]
            gradient = [_render(g, 'gradient') for g in (gx, gy)]
            peak = _render(exact['q_max_Pa'], 'peak pressure')
            verified = _close(ra, area) and _close(rf, n) and _normalized_balance(
                vertices, rendered, pressures, centroid)
            verified &= all(abs(x-y) <= n*d/10**9 for x, y, d in zip((rx, ry), moments, dims))
            for xy, pressure in list(zip(rendered, pressures)) + list(zip(vertices, corner)):
                affine = F(peak)+sum(F(g)*(x-F(ref)) for g, x, ref in zip(gradient, xy, reference))
                verified &= abs(max(0,affine)-F(pressure)) <= exact['q_max_Pa']/10**9
            out = dict(**{k:_render(val,k) for k,val in exact.items()},
                       contact_regime='full_contact' if shape=='full_triangle' else 'partial_contact',
                       footprint_shape=shape, corner_pressure_Pa=[_render(h,'corner pressure') for h in corner],
                       eccentricity_m=[_render(center[j]-sum(p[j] for p in vertices)/3,'eccentricity') for j in (0,1)],
                       contact_polygon_m=points, contact_vertex_pressure_Pa=pressures,
                       pressure_plane=dict(reference_point_m=reference, pressure_at_reference_Pa=peak,
                                           gradient_Pa_per_m=gradient),
                       integrated_vertical_resultant_N=_render(rf,'force'),
                       integrated_moment_x_Nm=_render(rx,'first moment'),
                       integrated_moment_y_Nm=_render(ry,'first moment'),
                       output_geometry_verified=bool(verified))
        out.update(ok=True, calculation=calculation, equilibrium_status='finite',
                   pressure_model='positive_part_affine', moment_convention='Mx=integral(x*p); My=integral(y*p)')
        checks = _claim_checks(spec, out, exact, dims, n, moments)
        key = 'proposed_contact_polygon_m'
        if n and checks.get(key):
            proposed = [tuple(_quantity(v,key,'m') for v in xy) for xy in spec[key]]
            if proposed[-1] == proposed[0]:
                proposed.pop()
            # Independently require that proposed vertices match in the intrinsic
            # triangle, including the narrow coordinate of an oblique support.
            actual = [_coordinates(vertices, xy) for xy in rendered]
            claimed = [_coordinates(vertices, xy) for xy in proposed]
            span = [max(q[j] for q in actual)-min(q[j] for q in actual) for j in range(3)]
            checks[key] = any(all(abs(q[j]-actual[(start+step*i)%len(actual)][j]) <= span[j]/10**9
                                  for i,q in enumerate(claimed) for j in range(3))
                              for start in range(len(actual)) for step in (-1,1))
        out['claim_checks'] = checks
        out['conservation_verified'] = out['output_geometry_verified'] and all(checks.values())
        return out
    except (KeyError, ValueError, ArithmeticError, TypeError, SyntaxError) as exc:
        return {**failure, 'reason':str(exc)}
