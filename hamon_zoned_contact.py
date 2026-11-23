"""One rigid affine displacement over disjoint linear normal support regions."""
from fractions import Fraction as F

from hamon_polygon_contact import _clip, _det, _dot, _edges, _hull, _inverse, _patches
from hamon_rectangle_contact import _integrate_displayed, _quantity, _render


def _area(poly):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in _edges(poly)))/2


def _positive_overlap(left,right):
    # Exact separating-axis test on the convex material patches. A shared
    # boundary is allowed; a positive-area overlap would count stiffness twice.
    for poly in (left,right):
        for a,b in _edges(poly):
            normal=(a[1]-b[1],b[0]-a[0])
            x=[_dot(normal,p) for p in left]
            y=[_dot(normal,p) for p in right]
            if max(x)<=min(y) or max(y)<=min(x):
                return False
    return True


def _displacement_plane(theta,origin,scale,factor,hull):
    def normalized(point):
        return tuple((x-c)/d for x,c,d in zip(point,origin,scale))
    anchor=max(hull,key=lambda p:_dot(theta,(1,*normalized(p))))
    reference=[_render(x,'displacement reference') for x in anchor]
    slopes=[factor*t/d for t,d in zip(theta[1:],scale)]
    intercept=factor*_dot(theta,(1,*normalized(tuple(map(F,reference)))))
    exact=(intercept-_dot(slopes,map(F,reference)),*slopes)
    peak=max(_dot(exact,(1,*p)) for p in hull)
    try:
        values=[_render(intercept,'displacement'),*(_render(g,'displacement slope') for g in slopes)]
        rounded=(F(values[0])-_dot(map(F,values[1:]),map(F,reference)),*map(F,values[1:]))
        representable=all(abs(_dot(tuple(a-b for a,b in zip(rounded,exact)),(1,*p)))<=
                          peak*F('1e-12') for p in hull)
    except (ValueError,OverflowError):
        representable=False
    if representable:
        return rounded,dict(reference_point_m=reference,displacement_at_reference_m=values[0],
                            gradient_m_per_m=values[1:])
    return exact,dict(encoding='rational',reference_point_m=reference,
                      displacement_at_reference_m=str(intercept),gradient_m_per_m=list(map(str,slopes)))


def _region_witness(region,plane,scale):
    stiffness=region['stiffness']
    total=force=mx=my=peak=closure=F(0)
    patches=[]
    for poly in region['patches']:
        cut=_clip(poly,plane)
        if len(cut)<3 or not any(w>0 for _,w in cut):
            continue
        points=[[_render(x,'contact coordinate') for x in p] for p,_ in cut]
        displacement=[_render(w,'contact displacement') for _,w in cut]
        pressure=[_render(stiffness*w,'contact pressure') for _,w in cut]
        displayed=[tuple(map(F,p)) for p in points]
        local_peak=max(w for _,w in cut)
        exact_area=_area([p for p,_ in cut])
        compact=(len(set(displayed))==len(displayed) and
                 all(_det(displayed[i-1],p,displayed[(i+1)%len(displayed)])>0
                     for i,p in enumerate(displayed)) and
                 abs(_area(displayed)-exact_area)<=exact_area*F('1e-8') and
                 all(abs(max(F(0),_dot(plane,(1,*p)))-F(w))<=local_peak*F('1e-10')
                     and abs(F(q)/stiffness-F(w))<=local_peak*F('1e-10')
                     and all(abs(a-b)<=d*F('1e-9') for a,b,d in zip(original,p,scale))
                     for (original,_),p,w,q in zip(cut,displayed,displacement,pressure)))
        if compact:
            encoding='binary64'
        else:
            encoding='rational'
            points=[[str(x) for x in p] for p,_ in cut]
            displacement=[str(w) for _,w in cut]
            pressure=[str(stiffness*w) for _,w in cut]
        for p,w,q in zip(points,displacement,pressure):
            kinematic=max(F(0),_dot(plane,(1,*map(F,p))))
            closure=max(closure,abs(F(w)-kinematic),abs(F(q)/stiffness-F(w)))
        a,n,x,y=_integrate_displayed(points,pressure)
        total+=a
        force+=n
        mx+=x
        my+=y
        peak=max(peak,*map(F,pressure))
        patches.append(dict(encoding=encoding,vertices_m=points,
                            vertex_displacement_m=displacement,vertex_pressure_Pa=pressure))
    out=dict(id=region['id'],vertical_stiffness_N_m3=_render(stiffness,'support stiffness'),
             material_area_m2=_render(region['area'],'material area'),
             contact_area_m2=_render(total,'contact area'),q_max_Pa=_render(peak,'peak pressure'),
             integrated_vertical_resultant_N=_render(force,'region force'),
             integrated_moment_x_Nm=_render(mx,'region first moment'),
             integrated_moment_y_Nm=_render(my,'region first moment'),
             max_closure_residual_m=_render(closure,'constitutive closure'),
             contact_patches=patches,constitutive_verified=True)
    return out,(total,force,mx,my,peak,closure)


def verify_rigid_zoned_contact(spec):
    kind='rigid_zoned_contact'
    failure=dict(ok=False,calculation=kind,conservation_verified=False,
                 common_displacement_verified=False,equilibrium_status='unverified')
    claims={'contact_area_m2':'m**2','material_area_m2':'m**2','q_max_Pa':'Pa'}
    allowed={'mode','kind','calculation','pressure_model','support_regions',
             'vertical_resultant_N','moment_x_Nm','moment_y_Nm'}|{'proposed_'+k for k in claims}
    try:
        if set(spec)-allowed or spec.get('calculation',kind)!=kind:
            raise ValueError('Unsupported zoned-contact input')
        if spec.get('pressure_model','positive_part_affine_displacement')!='positive_part_affine_displacement':
            raise ValueError('Expected one positive-part affine displacement over all supports')
        raw=spec['support_regions']
        if not isinstance(raw,(list,tuple)) or not raw:
            raise ValueError('Expected a nonempty list of support regions')
        regions=[]
        identities=set()
        for item in raw:
            if not isinstance(item,dict) or set(item)-{
                    'id','outer_vertices_m','hole_vertices_m','vertical_stiffness_N_m3'}:
                raise ValueError('Unsupported support-region input')
            label=item['id']
            if not isinstance(label,str) or not label.strip() or label in identities:
                raise ValueError('Each support region needs a distinct nonempty id')
            identities.add(label)
            stiffness=_quantity(item['vertical_stiffness_N_m3'],'vertical stiffness','N/m**3')
            if stiffness<=0:
                raise ValueError('Vertical stiffness must be strictly positive')
            rings,patches=_patches(item['outer_vertices_m'],item.get('hole_vertices_m',[]))
            area=_area(rings[0])-sum(_area(hole) for hole in rings[1:])
            if area<=0 or area!=sum(_area(p) for p in patches):
                raise ValueError('Support material partition failed area conservation')
            for prior in regions:
                if any(_positive_overlap(p,q) for p in patches for q in prior['patches']):
                    raise ValueError('Support-region interiors overlap')
            regions.append(dict(id=label,stiffness=stiffness,rings=rings,patches=patches,area=area))
        n=_quantity(spec['vertical_resultant_N'],'vertical resultant','N')
        moments=tuple(_quantity(spec[k],k,'N*m') for k in ('moment_x_Nm','moment_y_Nm'))
        if n<0 or (not n and any(moments)):
            return {**failure,'equilibrium_status':'no_finite_equilibrium'}
        material=sum(r['area'] for r in regions)
        out=dict(material_area_m2=_render(material,'material area'),contact_area_m2=None,
                 q_max_Pa=0.,displacement_plane=None,regions=[],
                 integrated_vertical_resultant_N=0.,integrated_moment_x_Nm=0.,
                 integrated_moment_y_Nm=0.,max_closure_residual_m=0.)
        if not n:
            for region in regions:
                row,_=_region_witness(region,(F(0),F(0),F(0)),(F(1),F(1)))
                row['contact_area_m2']=None
                out['regions'].append(row)
        else:
            vertices=[p for r in regions for p in r['rings'][0]]
            hull=_hull(vertices)
            center=tuple(m/n for m in moments)
            if any(_det(a,b,center)<=0 for a,b in _edges(hull)):
                return {**failure,'equilibrium_status':'no_finite_equilibrium',
                        'reason':'Positive finite pressure requires an interior support-hull centroid'}
            axes=list(zip(*vertices))
            origin=tuple((min(a)+max(a))/2 for a in axes)
            scale=tuple((max(a)-min(a))/2 for a in axes)
            def normalized(point):
                return tuple((x-c)/d for x,c,d in zip(point,origin,scale))
            reference_stiffness=max(r['stiffness'] for r in regions)
            patches=[poly for r in regions for poly in r['patches']]
            weights=[r['stiffness']/reference_stiffness for r in regions for _ in r['patches']]
            theta=_inverse([[normalized(p) for p in poly] for poly in patches],
                           (F(1),*normalized(center)),weights=weights)
            plane,description=_displacement_plane(theta,origin,scale,
                                                  n/(reference_stiffness*scale[0]*scale[1]),hull)
            out['displacement_plane']=description
            area=force=mx=my=peak=closure=F(0)
            for region in regions:
                row,values=_region_witness(region,plane,scale)
                a,f,x,y,q,c=values
                area+=a
                force+=f
                mx+=x
                my+=y
                peak=max(peak,q)
                closure=max(closure,c)
                out['regions'].append(row)
            if abs(force-n)>n*F('1e-8'):
                raise ValueError('Displayed region pressures failed total force equilibrium')
            for a,b in _edges(hull):
                slack=_det(a,b,center)
                residual=(b[0]-a[0])*(my-moments[1]-a[1]*(force-n))-(b[1]-a[1])*(mx-moments[0]-a[0]*(force-n))
                if abs(residual)>n*slack*F('1e-8'):
                    raise ValueError('Displayed region pressures failed a support-distance moment check')
            maximum_displacement=max(_dot(plane,(1,*p)) for p in hull)
            if closure>maximum_displacement*F('1e-8'):
                raise ValueError('Displayed regions do not share the reported displacement field')
            out.update(contact_area_m2=_render(area,'contact area'),q_max_Pa=_render(peak,'peak pressure'),
                       integrated_vertical_resultant_N=_render(force,'force'),
                       integrated_moment_x_Nm=_render(mx,'first moment'),
                       integrated_moment_y_Nm=_render(my,'first moment'),
                       max_closure_residual_m=_render(closure,'constitutive closure'))
        checks={}
        for key,unit in claims.items():
            if 'proposed_'+key in spec:
                supplied=_quantity(spec['proposed_'+key],key,unit)
                actual=out[key]
                checks['proposed_'+key]=(actual is not None and
                    abs(supplied-F(actual))<=F('1e-8')*max(abs(supplied),abs(F(actual))))
        return dict(**out,ok=True,calculation=kind,conservation_verified=all(checks.values()),
                    common_displacement_verified=True,output_geometry_verified=True,claim_checks=checks,
                    pressure_model='positive_part_affine_displacement',
                    moment_convention='Mx=integral(x*p); My=integral(y*p)',
                    equilibrium_status='finite' if n else 'unloaded')
    except (KeyError,TypeError,ValueError,SyntaxError,ZeroDivisionError,OverflowError) as exc:
        return {**failure,'reason':str(exc)}
