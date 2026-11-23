"""One shared compression-only affine field over a polygon with material holes."""
from fractions import Fraction as F
from hamon_rectangle_contact import _integrate_displayed, _quantity, _render
from hamon_triangle_contact import _det


def _edges(loop):
    return zip(loop, loop[1:] + loop[:1])


def _on_segment(a, b, p):
    return (_det(a, b, p) == 0 and
            all(min(x,y) <= z <= max(x,y) for x,y,z in zip(a,b,p)))


def _intersects(a, b, c, d):
    if _det(a,b,c)*_det(a,b,d) < 0 and _det(c,d,a)*_det(c,d,b) < 0:
        return True
    return any((_on_segment(a,b,c), _on_segment(a,b,d), _on_segment(c,d,a), _on_segment(c,d,b)))


def _inside(loop, point):
    x, y = point
    return sum((b[1] > y) != (a[1] > y) and x < a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
               for a,b in _edges(loop)) % 2 == 1


def _loop(raw):
    if not isinstance(raw, (list,tuple)) or any(
            not isinstance(p,(list,tuple)) or len(p) != 2 for p in raw):
        raise ValueError('Expected a loop of planar coordinates')
    ring = [tuple(_quantity(v,'material vertex','m') for v in p) for p in raw]
    if len(ring)>1 and ring[-1] == ring[0]:
        ring.pop()
    if len(set(ring)) != len(ring):
        raise ValueError('Repeated boundary vertex')
    while len(ring)>=3:
        i = next((i for i,b in enumerate(ring) if _det(ring[i-1], b, ring[(i+1)%len(ring)])==0),None)
        if i is None:
            break
        if not _on_segment(ring[i-1],ring[(i+1)%len(ring)],ring[i]):
            raise ValueError('Overlapping adjacent edges')
        ring.pop(i)
    if len(ring)<3:
        raise ValueError('Degenerate boundary loop')
    edges=list(_edges(ring))
    for i,(a,b) in enumerate(edges):
        for j in range(i+2,len(edges)):
            if i==0 and j==len(edges)-1:
                continue
            if _intersects(a,b,*edges[j]):
                raise ValueError('Self-intersecting boundary loop')
    return ring


def _patches(outer, holes):
    if not isinstance(holes,(list,tuple)):
        raise ValueError('Expected a list of hole boundaries')
    rings=[_loop(outer), *(_loop(h) for h in holes)]
    for i, hole in enumerate(rings[1:],1):
        for j,other in enumerate(rings[:i]):
            if any(_intersects(*ab,*cd) for ab in _edges(hole) for cd in _edges(other)):
                raise ValueError('Material boundaries must be disjoint')
            if j and (_inside(other,hole[0]) or _inside(hole,other[0])):
                raise ValueError('Hole interiors overlap')
        if not _inside(rings[0],hole[0]):
            raise ValueError('A hole is outside the outer boundary')
    # In an open slab, no boundary segment starts, ends or crosses another.
    # Consecutive bottom/top intersections sweep exactly the occupied material.
    edges=[(a,b) for ring in rings for a,b in _edges(ring) if a[0]!=b[0]]
    xs=sorted({p[0] for ring in rings for p in ring})
    patches=[]
    def ordinate(e,x):
        a,b=e
        return a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0])
    for left,right in zip(xs,xs[1:]):
        mid=(left+right)/2
        active=sorted((e for e in edges if min(e[0][0],e[1][0])<mid<max(e[0][0],e[1][0])),
                      key=lambda e:ordinate(e,mid))
        if len(active)%2:
            raise ValueError('Unpaired material boundary')
        for lo,hi in zip(active[::2],active[1::2]):
            poly=list(dict.fromkeys([(left,ordinate(lo,left)),(right,ordinate(lo,right)),
                                     (right,ordinate(hi,right)),(left,ordinate(hi,left))]))
            if len(poly)>=3:
                patches.append(poly)
    return rings,patches


def _hull(points):
    pts=sorted(set(points))
    def chain(seq):
        out=[]
        for p in seq:
            while len(out)>1 and _det(out[-2],out[-1],p)<=0:
                out.pop()
            out.append(p)
        return out
    return chain(pts)[:-1]+chain(reversed(pts))[:-1]


def _clip(poly, plane):
    def height(p):
        return plane[0]+plane[1]*p[0]+plane[2]*p[1]
    out=[]
    prev=poly[-1]
    pa=height(prev)
    for p in poly:
        pb=height(p)
        if pa*pb < 0:
            inside,outside,hi,ho=(prev,p,pa,pb) if pa>0 else (p,prev,pb,pa)
            t=hi/(hi-ho)
            out.append((tuple(x+t*(y-x) for x,y in zip(inside,outside)),0))
        if pb >= 0:
            out.append((p,pb))
        prev,pa=p,pb
    return out


def _gram(polygons):
    """Integral of (1,x,y)(1,x,y)^T over convex counterclockwise patches."""
    h = [[0]*3 for _ in range(3)]
    for poly in polygons:
        for i in range(1,len(poly)-1):
            pts=(poly[0],poly[i],poly[i+1])
            area=_det(*pts)/2
            vectors=[(1,*p) for p in pts]
            sums=[sum(col) for col in zip(*vectors)]
            for j in range(3):
                for k in range(j,3):
                    h[j][k] += area*(sums[j]*sums[k]+sum(v[j]*v[k] for v in vectors))/12
    for j in range(3):
        for k in range(j):
            h[j][k]=h[k][j]
    return h


def _solve3(h, rhs):
    rows=[[*row,val] for row,val in zip(h,rhs)]
    for j in range(3):
        pivot=max(range(j,3),key=lambda i:abs(rows[i][j]))
        rows[j],rows[pivot]=rows[pivot],rows[j]
        if not rows[j][j]:
            raise ValueError('Insufficient numerical contact rank')
        for i in range(j+1,3):
            scale=rows[i][j]/rows[j][j]
            rows[i]=[a-scale*b for a,b in zip(rows[i],rows[j])]
    answer=[0]*3
    for j in (2,1,0):
        answer[j]=(rows[j][3]-sum(rows[j][i]*answer[i] for i in range(j+1,3)))/rows[j][j]
    return answer


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def _inverse(patches,target,weights=None):
    if weights is not None and (len(weights)!=len(patches) or any(w<=0 for w in weights)):
        raise ValueError('Every contact patch requires one positive stiffness weight')
    def gram(polygons):
        if weights is None:
            return _gram(polygons)
        matrices=[_gram([poly]) for poly in polygons]
        return [[sum(w*h[j][k] for w,h in zip(weights,matrices))
                 for k in range(3)] for j in range(3)]
    theta=_solve3(gram(patches),target)
    if all(_dot(theta,(1,*p))>=0 for poly in patches for p in poly):
        return theta
    # Put the prescribed centroid at zero and two supporting lines at -1.
    # Compute their small distances before converting to floating point.
    # Maximizing the scaled normal determinant is invariant under affine
    # changes of coordinates, and resolves a thin edge or corner contact.
    center=tuple(x/target[0] for x in target[1:])
    hull=_hull([p for poly in patches for p in poly])
    normals=[]
    for a,b in _edges(hull):
        slack=_det(a,b,center)
        if slack<=0:
            raise ValueError('Positive contact requires an interior centroid')
        normals.append(((a[1]-b[1])/slack,(b[0]-a[0])/slack))
    def cross(a,b):
        return a[0]*b[1]-a[1]*b[0]
    u,v=max(((a,b) for i,a in enumerate(normals) for b in normals[i+1:]),
            key=lambda pair:abs(cross(*pair)))
    jacobian=cross(u,v)
    if jacobian<0:
        u,v=v,u
        jacobian=-jacobian
    def coordinates(p):
        offset=tuple(x-c for x,c in zip(p,center))
        return _dot(u,offset),_dot(v,offset)
    transformed=[[coordinates(p) for p in poly] for poly in patches]
    local_target=(F(1),F(0),F(0))
    local=_solve3(gram(transformed),local_target)
    polys=[[tuple(map(float,p)) for p in poly] for poly in transformed]
    theta=list(map(float,local))
    # One variational objective couples every occupied patch, including
    # disconnected contact islands. Its Hessian is the active second moment.
    def state(t):
        h=gram([[p for p,_ in _clip(poly,t)] for poly in polys])
        load=[_dot(row,t) for row in h]
        return h,[load[0]-1,load[1],load[2]],_dot(t,load)/2-t[0]
    h,g,obj=state(theta)
    for _ in range(200):
        norm=max(map(abs,g))
        if norm<2e-12:
            a,b,c=map(F,theta)
            gradients=tuple(b*x+c*y for x,y in zip(u,v))
            factor=jacobian*target[0]
            return [factor*(a-_dot(gradients,center)),*(factor*x for x in gradients)]
        step=_solve3(h,g)
        decrease=_dot(g,step)
        for j in range(40):
            fraction=2.**-j
            trial=[x-fraction*y for x,y in zip(theta,step)]
            h2,g2,obj2=state(trial)
            if (obj2<obj-1e-4*fraction*decrease or
                    (norm<1e-6 and max(map(abs,g2))<norm*(1-.1*fraction))):
                theta,h,g,obj=trial,h2,g2,obj2
                break
        else:
            break
    raise ValueError('Numerical inverse could not certify the requested equilibrium')


def verify_rigid_polygon_contact(spec):
    kind='rigid_polygon_contact'
    failure=dict(ok=False,calculation=kind,conservation_verified=False)
    claims={'contact_area_m2':'m^2','material_area_m2':'m^2','q_max_Pa':'Pa'}
    allowed={'mode','kind','calculation','pressure_model','outer_vertices_m','hole_vertices_m',
             'vertical_resultant_N','moment_x_Nm','moment_y_Nm'} | {'proposed_'+key for key in claims}
    try:
        if set(spec)-allowed or spec.get('calculation',kind)!=kind:
            raise ValueError('Unsupported polygon-contact input')
        if spec.get('pressure_model','positive_part_affine')!='positive_part_affine':
            raise ValueError('Expected a shared positive-part affine pressure field')
        rings,patches=_patches(spec['outer_vertices_m'],spec.get('hole_vertices_m',[]))
        def area(ring):
            return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in _edges(ring)))/2
        material=area(rings[0])-sum(area(h) for h in rings[1:])
        if material<=0 or material!=sum(area(poly) for poly in patches):
            raise ValueError('Material partition failed polygon area conservation')
        n=_quantity(spec['vertical_resultant_N'],'vertical_resultant_N','N')
        m=tuple(_quantity(spec[k],k,'N*m') for k in ('moment_x_Nm','moment_y_Nm'))
        if n<0 or (n==0 and any(m)):
            return {**failure,'equilibrium_status':'no_finite_equilibrium'}
        out=dict(material_area_m2=_render(material,'material area'),contact_patches=[],
                 pressure_plane=None,contact_area_m2=None,q_max_Pa=0.,
                 integrated_vertical_resultant_N=0.,integrated_moment_x_Nm=0.,integrated_moment_y_Nm=0.)
        if n:
            center=tuple(z/n for z in m)
            hull=_hull(rings[0])
            if any(_det(a,b,center)<=0 for a,b in _edges(hull)):
                return {**failure,'equilibrium_status':'no_finite_equilibrium',
                        'reason':'Positive finite pressure requires a centroid strictly inside the support hull'}
            axes=list(zip(*rings[0]))
            origin=tuple((min(a)+max(a))/2 for a in axes)
            scale=tuple((max(a)-min(a))/2 for a in axes)
            def normalized(p):
                return tuple((x-c)/d for x,c,d in zip(p,origin,scale))
            t=_inverse([[normalized(p) for p in poly] for poly in patches],(F(1),*normalized(center)))
            pressure_scale=n/(scale[0]*scale[1])
            reference_origin=origin
            if any(_dot(t,(1,*normalized(p)))<0 for p in rings[0]):
                reference_origin=max(rings[0],key=lambda p:_dot(t,(1,*normalized(p))))
            reference=[_render(c,'plane reference') for c in reference_origin]
            exact_gradients=[pressure_scale*x/d for x,d in zip(t[1:],scale)]
            exact_constant=pressure_scale*_dot(t,(1,*normalized(tuple(map(F,reference)))))
            exact_plane=(exact_constant-_dot(exact_gradients,map(F,reference)),*exact_gradients)
            gradients=[_render(x,'pressure gradient') for x in exact_gradients]
            constant=_render(exact_constant,'pressure intercept')
            plane=(F(constant)-_dot(map(F,gradients),map(F,reference)),*map(F,gradients))
            out['pressure_plane']=dict(reference_point_m=reference,pressure_at_reference_Pa=constant,
                                       gradient_Pa_per_m=gradients)
            # Separate rounding of oblique gradient components can rotate a
            # thin strip by more than its width. Preserve the common plane
            # exactly when binary64 cannot represent its field on the support.
            peak_exact=max(_dot(exact_plane,(1,*p)) for p in hull)
            if any(abs(_dot(tuple(a-b for a,b in zip(plane,exact_plane)),(1,*p)))>
                   peak_exact*F('1e-12') for p in hull):
                plane=exact_plane
                out['pressure_plane']=dict(encoding='rational',reference_point_m=reference,
                                           pressure_at_reference_Pa=str(exact_constant),
                                           gradient_Pa_per_m=list(map(str,exact_gradients)))
            total=force=mx=my=peak=F(0)
            for poly in patches:
                clipped=_clip(poly,plane)
                if len(clipped)<3 or not any(q>0 for _,q in clipped):
                    continue
                points=[[_render(c,'contact coordinate') for c in p] for p,_ in clipped]
                pressures=[_render(q,'contact pressure') for _,q in clipped]
                display=[tuple(map(F,p)) for p in points]
                # Each vertex is an intersection of material, slab or release
                # boundaries. Moving it even slightly can fill a real hole or
                # separate adjacent patches while preserving integrated loads.
                # Round-trip the heights as well, so adjacent patches remain
                # exact restrictions of the single reported pressure plane.
                compact=all(xy==p and F(h)==q
                            for (p,q),xy,h in zip(clipped,display,pressures))
                if compact:
                    encoding='binary64'
                else:
                    encoding='rational'
                    points=[[str(c) for c in p] for p,_ in clipped]
                    pressures=[str(q) for _,q in clipped]
                a,f,x,y=_integrate_displayed(points,pressures)
                total+=a
                force+=f
                mx+=x
                my+=y
                peak=max(peak,*map(F,pressures))
                out['contact_patches'].append(dict(encoding=encoding,vertices_m=points,vertex_pressure_Pa=pressures))
            if (abs(force-n)>n*F('1e-8') or
                any(abs((v-o*force)-(w-o*n))>n*d*F('1e-8') for v,w,o,d in zip((mx,my),m,origin,scale))):
                raise ValueError('Displayed contact geometry does not reconstruct the requested loads')
            # Check the actual centroid-to-edge distances. A full-footprint
            # scale can conceal a large relative contact-width or peak error.
            for p,q in _edges(hull):
                width=_det(p,q,center)
                residual=(q[0]-p[0])*(my-m[1]-p[1]*(force-n))-(q[1]-p[1])*(mx-m[0]-p[0]*(force-n))
                if abs(residual)>n*width*F('1e-8'):
                    raise ValueError('Displayed geometry failed a support-normal moment check')
            out.update(contact_area_m2=_render(total,'contact area'),q_max_Pa=_render(peak,'peak'),
                       integrated_vertical_resultant_N=_render(force,'force'),
                       integrated_moment_x_Nm=_render(mx,'moment'),integrated_moment_y_Nm=_render(my,'moment'))
        checks={}
        for key,unit in claims.items():
            if 'proposed_'+key in spec:
                supplied=_quantity(spec['proposed_'+key],key,unit)
                actual=out[key]
                checks['proposed_'+key]=(actual is not None and
                    abs(supplied-F(actual))<=F('1e-8')*max(abs(supplied),abs(F(actual))))
        return dict(**out,ok=True,calculation=kind,conservation_verified=all(checks.values()),
                    claim_checks=checks,output_geometry_verified=True,pressure_model='positive_part_affine',
                    moment_convention='Mx=integral(x*p); My=integral(y*p)',
                    equilibrium_status='finite' if n else 'unloaded')
    except (KeyError,TypeError,ValueError,ZeroDivisionError,OverflowError) as exc:
        return {**failure,'equilibrium_status':'unverified','reason':str(exc)}
