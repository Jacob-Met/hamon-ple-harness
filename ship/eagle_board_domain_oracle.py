"""Frozen W19 nominal board-domain oracle. Units: mm. Python standard library."""
from fractions import Fraction as F
from math import sqrt

def Q(x):
    return F(str(x))

def P(x, y):
    return Q(x), Q(y)

def sub(a, b):
    return a[0]-b[0], a[1]-b[1]

def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]

def det(a, b):
    return a[0]*b[1]-a[1]*b[0]

def at(a, b, t):
    return tuple(x+(y-x)*t for x,y in zip(a,b))

def norm2(a, b):
    u=sub(a,b)
    return dot(u,u)

def pseg2(p, a, b):
    u=sub(b,a)
    if a==b:
        return norm2(p,a)
    t=max(F(0),min(F(1),dot(sub(p,a),u)/dot(u,u)))
    return norm2(p,at(a,b,t))

def intersects(a,b,c,d):
    def cross(p,q,r):
        return det(sub(q,p),sub(r,p))
    for j in (0,1):
        if max(min(a[j],b[j]),min(c[j],d[j]))>min(max(a[j],b[j]),max(c[j],d[j])):
            return False
    return cross(a,b,c)*cross(a,b,d)<=0 and cross(c,d,a)*cross(c,d,b)<=0

def seg2(a,b,c,d):
    if intersects(a,b,c,d):
        return F(0)
    return min(pseg2(a,c,d),pseg2(b,c,d),pseg2(c,a,b),pseg2(d,a,b))

def arc_profile(p,q,arc):
    """Endpoint distance squared plus radial squared-distance interval in the sweep."""
    a,c,b=arc
    u,v=sub(a,c),sub(b,c)
    assert dot(u,v)==0 and dot(u,u)==dot(v,v)>0
    assert u[0]*u[1]==v[0]*v[1]==0
    radius=max(abs(u[0]),abs(u[1]))
    direction=1 if det(u,v)>0 else -1
    lo,hi=F(0),F(1)
    for f in (lambda w:direction*det(u,w),lambda w:direction*det(w,v)):
        s,e=f(sub(p,c)),f(sub(q,c))
        if s<0 and e<0:
            return min(pseg2(a,p,q),pseg2(b,p,q)),radius,None
        if s<0:
            lo=max(lo,s/(s-e))
        elif e<0:
            hi=min(hi,s/(s-e))
    d2=min(pseg2(a,p,q),pseg2(b,p,q))
    if lo>hi:
        return d2,radius,None
    x,y=at(p,q,lo),at(p,q,hi)
    return d2,radius,(pseg2(c,x,y),max(norm2(c,x),norm2(c,y)))

def arc_clear(p,q,arc,r):
    d2,R,interval=arc_profile(p,q,arc)
    if d2<=r*r:
        return False
    if interval is None:
        return True
    m,M=interval
    return m>(R+r)**2 or (R>r and M<(R-r)**2)

def segment_arc_distance(p,q,arc):
    """Reporting only. Exact squared comparisons above decide pass/fail."""
    d2,R,interval=arc_profile(p,q,arc)
    if interval is None:
        return sqrt(d2)
    m,M=interval
    radial=max(0.0,sqrt(m)-float(R),float(R)-sqrt(M))
    return min(sqrt(d2),radial)

def inside_polygon(p,vertices):
    x,y=p
    inside=False
    for a,b in zip(vertices,vertices[1:]+vertices[:1]):
        if pseg2(p,a,b)==0:
            return True
        if (a[1]>y)!=(b[1]>y):
            hit=a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if hit>x:
                inside=not inside
    return inside

def rounded(virtual,corners):
    """Known simple sharp polygon and its disjoint convex corner replacements."""
    arcs=list(corners.values())
    lines=[]
    for i,v in enumerate(virtual):
        w=virtual[(i+1)%len(virtual)]
        lines.append((corners[v][2] if v in corners else v,
                      corners[w][0] if w in corners else w))
    def material(p):
        if not inside_polygon(p,virtual):
            return False
        for vertex,(a,c,b) in corners.items():
            inbox=all(min(v,k)<=x<=max(v,k) for x,v,k in zip(p,vertex,c))
            if inbox and norm2(p,c)>norm2(a,c):
                return False
        return True
    return lines,arcs,material

def rect(x0,y0,x1,y1):
    return rounded([P(x0,y0),P(x1,y0),P(x1,y1),P(x0,y1)],{})

def tube(domain,a,b,r):
    r=Q(r)
    if r<0:
        raise ValueError("negative tube radius")
    lines,arcs,inside=domain
    return (inside(a) and all(seg2(a,b,c,d)>r*r for c,d in lines)
            and all(arc_clear(a,b,arc,r) for arc in arcs))

def disk(domain,p,r):
    if Q(r)<=0:
        raise ValueError("positive witness radius required")
    return tube(domain,p,p,r)

def cases():
    V=[P(*v) for v in [(97.536,53.34),(99.06,51.816),(99.06,40.386),
       (101.6,37.846),(101.6,5.08),(99.06,2.54),(99.06,0),(0,0),(0,53.34)]]
    due_arc=(P(1,0),P(1,1),P(0,1))
    due=rounded(V,{P(99.06,0):(P(99.06,1),P(98.06,1),P(98.06,0)),
                   P(0,0):due_arc,P(0,53.34):(P(0,52.34),P(1,52.34),P(1,53.34))})
    V=[P(0,0),P(50.8,0),P(50.8,22.86),P(0,22.86)]
    feather=rounded(V,{V[0]:(P(0,2.54),P(2.54,2.54),P(2.54,0)),
               V[1]:(P(48.26,0),P(48.26,2.54),P(50.8,2.54)),
               V[2]:(P(50.8,20.32),P(48.26,20.32),P(48.26,22.86)),
               V[3]:(P(2.54,22.86),P(2.54,20.32),P(0,20.32))})
    p=P(.46,.28)
    assert [disk(due,p,r) for r in (.09,.1,.11)]==[True,False,False]
    assert [tube(due,P(.70,.60),p,r) for r in (.09,.1,.11)]==[True,False,False]
    assert [disk(feather,P(1.1684,.7112),r) for r in (.25,.254,.26)]==[True,False,False]
    for arc in (due_arc,due_arc[::-1]):
        assert arc_clear(P(.70,.60),p,arc,Q(.09))
        assert not arc_clear(P(.70,.60),p,arc,Q(.1))
        assert arc_clear(P(1.6,1.6),P(1.8,1.8),arc,Q(1))
    a,b=P(3,3.8),P(3.8,3)
    ring=[P(0,0),P(4,0),P(4,3),P(3,4),P(0,4)]
    bite=(P(4,3),P(4,4),P(3,4))
    lines=[(ring[i],ring[(i+1)%5]) for i in (0,1,3,4)]
    domain=(lines,[bite],lambda z:all(0<=v<=4 for v in z) and norm2(z,P(4,4))>=1)
    assert disk(domain,a,.01) and disk(domain,b,.01) and not tube(domain,a,b,.01)
    h=Q(".099998")
    def clipped_area(half,xmin,xmax):
        return 2*half*max(F(0),min(half,Q(xmax))-max(-half,Q(xmin)))
    assert clipped_area(h,-2,2)==Q(".039998400016")
    assert clipped_area(h,10,14)==0
    assert clipped_area(h,.05,2)==Q(".009999400008")
    assert clipped_area(Q(".1"),.05,2)==Q(".01")
    assert clipped_area(Q(".1"),.1,2)==0
    witness=P(.075,0)
    assert all(disk(board,witness,.012) for board in
               (rect(.05,-2,2,2),rect(-h,-h,h,h)))
    assert disk(rect(-2,-2,2,2),P(0,0),.049999)
    assert not disk(rect(10,-2,14,2),P(0,0),.049999)
    print("PASS: W19 rectangles, Due/Feather disks, complete segments and reversed arcs")
    print("Due segment-to-quarter distance, mm:",segment_arc_distance(P(.70,.60),p,due_arc))

if __name__=="__main__":
    cases()


from fractions import Fraction

NM_IN_MM = Fraction(1, 1_000_000)

def authentic_domains():
    due_vertices = [P(*xy) for xy in (
        ("97.536", "53.34"), ("99.06", "51.816"), ("99.06", "40.386"),
        ("101.6", "37.846"), ("101.6", "5.08"), ("99.06", "2.54"),
        ("99.06", 0), (0, 0), (0, "53.34"),
    )]
    due_corners = {
        P("99.06", 0): (P("99.06", 1), P("98.06", 1), P("98.06", 0)),
        P(0, 0): (P(1, 0), P(1, 1), P(0, 1)),
        P(0, "53.34"): (P(0, "52.34"), P(1, "52.34"), P(1, "53.34")),
    }
    feather_vertices = [P(*xy) for xy in (
        (0, 0), ("50.8", 0), ("50.8", "22.86"), (0, "22.86"),
    )]
    feather_corners = {
        P(0, 0): (P(0, "2.54"), P("2.54", "2.54"), P("2.54", 0)),
        P("50.8", 0): (P("48.26", 0), P("48.26", "2.54"), P("50.8", "2.54")),
        P("50.8", "22.86"): (P("50.8", "20.32"), P("48.26", "20.32"), P("48.26", "22.86")),
        P(0, "22.86"): (P("2.54", "22.86"), P("2.54", "20.32"), P(0, "20.32")),
    }
    return {"Due": rounded(due_vertices, due_corners),
            "Feather": rounded(feather_vertices, feather_corners)}

def point_nm_to_mm(xy):
    x, y = xy
    return Q(x) * NM_IN_MM, Q(y) * NM_IN_MM

def material_path_on_board(domain, vertices_nm):
    """Certify the complete 1 nm tube of a nonconstant, piecewise-linear path."""
    points = [point_nm_to_mm(xy) for xy in vertices_nm]
    if len(set(points)) < 2:
        return False
    return all(tube(domain, a, b, NM_IN_MM) for a, b in zip(points, points[1:]))

def reported_disk_on_board(domain, center_nm, radius_nm):
    """Use the reported disk radius, for either contact or clear-collar evidence."""
    radius_mm = Q(radius_nm) * NM_IN_MM
    return radius_mm > 0 and disk(domain, point_nm_to_mm(center_nm), radius_mm)
