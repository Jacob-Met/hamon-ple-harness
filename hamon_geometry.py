"""Conservative geometry-aware IFC clash screening.

Independent of IFC until the loader: AABB BVH -> triangle surface SAT -> closed
shell containment. "Contact" means triangulated surfaces touch/intersect, not
verified physical interference; failing or non-manifold models stay unresolved.
Based on separating-axis testing of triangles (Moller 1997) and Moller-Trumbore
(1997) ray/triangle crossings. Built-in Python only for geometry core.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import re


def minus(a,b): return tuple(x-y for x,y in zip(a,b))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def bounds(points):
    return (tuple(min(p[i] for p in points) for i in range(3)),
            tuple(max(p[i] for p in points) for i in range(3)))


def overlaps(box_a,box_b,tol=0.0):
    return all(box_a[1][i]+tol >= box_b[0][i] and
               box_b[1][i]+tol >= box_a[0][i] for i in range(3))


def tri_overlap(a,b,epsilon=1e-9):
    """SAT triangle surface contact including coplanar plane-edge axes."""
    ae=[minus(a[(i+1)%3],a[i]) for i in range(3)]
    be=[minus(b[(i+1)%3],b[i]) for i in range(3)]
    an=cross(ae[0],minus(a[2],a[0]))
    bn=cross(be[0],minus(b[2],b[0]))
    if dot(an,an) <= 1e-28 or dot(bn,bn) <= 1e-28:
        raise ValueError("degenerate triangle: cannot certify contact")
    axes=[an,bn]
    axes.extend(cross(x,y) for x in ae for y in be)
    # Normals cross in-plane edges close the coplanar SAT gap.
    axes.extend(cross(an,x) for x in ae+be)
    axes.extend(cross(bn,x) for x in ae+be)
    for axis in axes:
        length=math.sqrt(dot(axis,axis))
        if length <= 1e-14: continue
        pa=[dot(p,axis) for p in a]
        pb=[dot(p,axis) for p in b]
        allowance=epsilon*length
        if max(pa)+allowance < min(pb) or max(pb)+allowance < min(pa):
            return False
    return True


def node_for(triangles,ids):
    pts=[p for index in ids for p in triangles[index]]
    bb=bounds(pts)
    if len(ids) <= 8: return (bb,None,None,tuple(ids))
    span=minus(bb[1],bb[0])
    axis=max(range(3),key=lambda i:span[i])
    order=sorted(ids,key=lambda i:sum(p[axis] for p in triangles[i])/3)
    middle=len(order)//2
    return (bb,node_for(triangles,order[:middle]),
            node_for(triangles,order[middle:]),None)


def surface_contact(mesh_a,mesh_b,budget=250000,epsilon=1e-9):
    """Returns whether triangles contact, or fails loudly if pair budget ends."""
    counts=[0]
    def visit(left,right):
        if not overlaps(left[0],right[0],epsilon): return False
        if left[3] is not None and right[3] is not None:
            for i in left[3]:
                ai=mesh_a.triangles[i]
                ai_box=mesh_a.triangle_boxes[i]
                for j in right[3]:
                    if not overlaps(ai_box,mesh_b.triangle_boxes[j],epsilon):
                        continue
                    counts[0]+=1
                    if counts[0]>budget:
                        raise ValueError("triangle comparison budget exhausted; unresolved, increase budget")
                    if tri_overlap(ai,mesh_b.triangles[j],epsilon):
                        return True
            return False
        if left[3] is not None: return visit(left,right[1]) or visit(left,right[2])
        if right[3] is not None: return visit(left[1],right) or visit(left[2],right)
        return any((visit(left[1],right[1]),visit(left[1],right[2]),
                    visit(left[2],right[1]),visit(left[2],right[2])))
    if not overlaps(mesh_a.bbox,mesh_b.bbox,epsilon):
        return False,0
    hit=visit(mesh_a.bvh,mesh_b.bvh)
    return hit,counts[0]


def _ray_triangle(point,direction,tri,epsilon=1e-10):
    """Return t>0, 0 no hit, or -1 ambiguous boundary/grazing intersection."""
    a,b,c=tri
    e1,e2=minus(b,a),minus(c,a)
    h=cross(direction,e2)
    det=dot(e1,h)
    if abs(det) < epsilon: return 0
    inv=1.0/det
    s=minus(point,a)
    u=dot(s,h)*inv
    if u < -epsilon or u > 1+epsilon: return 0
    q=cross(s,e1)
    v=dot(direction,q)*inv
    if v < -epsilon or u+v > 1+epsilon: return 0
    t=dot(e2,q)*inv
    if t < -epsilon: return 0
    if t <= epsilon or u <= epsilon or v <= epsilon or u+v >= 1-epsilon:
        return -1
    return t


@dataclass(frozen=True)
class Mesh:
    vertices: tuple
    triangles: tuple
    triangle_boxes: tuple
    bbox: tuple
    bvh: tuple
    closed_oriented: bool
    representatives: tuple


def make_mesh(flat_vertices,flat_faces,max_triangles=20000):
    vv=list(flat_vertices); ff=list(flat_faces)
    if len(vv)%3 or len(ff)%3 or not vv or not ff:
        raise ValueError("nonempty 3D vertices and triangular indices required")
    if len(ff)//3 > max_triangles:
        raise ValueError("mesh triangle budget exceeded")
    if len(vv)//3 > 50000:
        raise ValueError("mesh vertex budget exceeded")
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>1e9
           for v in vv):
        raise ValueError("mesh has non-finite or implausible world coordinates")
    verts=tuple(tuple(float(vv[i+j]) for j in range(3)) for i in range(0,len(vv),3))
    triangles=[]; tr_boxes=[]; edges=Counter(); oriented=Counter()
    parent=list(range(len(verts)))
    def root(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]]
            k=parent[k]
        return k
    def union(i,j):
        ri,rj=root(i),root(j)
        if ri!=rj:parent[rj]=ri
    for k in range(0,len(ff),3):
        abc=ff[k:k+3]
        if any(type(x) is not int or x<0 or x>=len(verts) for x in abc):
            raise ValueError("mesh triangle references invalid vertex index")
        if len(set(abc))!=3:
            raise ValueError("mesh has repeated-index degenerate triangle")
        tri=tuple(verts[x] for x in abc)
        an=cross(minus(tri[1],tri[0]),minus(tri[2],tri[0]))
        if dot(an,an) <= 1e-28:
            raise ValueError("mesh has degenerate zero-area triangle")
        triangles.append(tri);tr_boxes.append(bounds(tri))
        for i in range(3):
            u,v=abc[i],abc[(i+1)%3]
            union(u,v)
            edges[tuple(sorted((u,v)))]+=1
            oriented[(u,v)]+=1
    closed=all(n==2 for n in edges.values()) and all(
        oriented[(u,v)]==oriented[(v,u)]
        for u,v in edges)
    reps={}
    active={index for k in range(0,len(ff),3) for index in ff[k:k+3]}
    for index in sorted(active): reps.setdefault(root(index),index)
    return Mesh(verts,tuple(triangles),tuple(tr_boxes),bounds(verts),
                node_for(triangles,list(range(len(triangles)))),
                closed,tuple(verts[i] for i in reps.values()))


def inside_closed(point,mesh):
    """Watertight oriented surface point-in-solid by multiple generic rays."""
    if not mesh.closed_oriented:
        return None
    rays=((1.0,0.416234,0.271828),(0.357,1.0,0.618034),
          (0.12345,0.517,1.0),(1.0,0.927,0.211))
    votes=[]
    for direction in rays:
        crossings=0
        ambiguous=False
        for tri in mesh.triangles:
            t=_ray_triangle(point,direction,tri)
            if t==-1: ambiguous=True;break
            if t!=0:crossings+=1
        if not ambiguous:
            votes.append(bool(crossings%2))
        if len(votes)>=2 and votes[-1]==votes[-2]:
            return votes[-1]
    return None


def mesh_relation(mesh_a,mesh_b,triangle_budget=250000):
    """Qualified triangulated geometry relation; no certified design clearance."""
    if not overlaps(mesh_a.bbox,mesh_b.bbox):
        return {"relation":"aabb_disjoint","triangle_tests":0}
    contact,tested=surface_contact(mesh_a,mesh_b,triangle_budget)
    if contact:
        return {"relation":"surface_contact","triangle_tests":tested}
    if not mesh_a.closed_oriented or not mesh_b.closed_oriented:
        return {"relation":"unresolved_nonwatertight","triangle_tests":tested}
    contain=[]
    for name,points,container in (("a_in_b",mesh_a.representatives,mesh_b),
                                  ("b_in_a",mesh_b.representatives,mesh_a)):
        for p in points:
            result=inside_closed(p,container)
            if result is None:
                return {"relation":"unresolved_containment_ray","triangle_tests":tested}
            if result:contain.append(name)
    if contain:
        return {"relation":"solid_containment","triangle_tests":tested,
                "evidence":sorted(set(contain))}
    return {"relation":"no_surface_contact_found","triangle_tests":tested}


def ifc_narrowphase(filename,class_a,class_b,limit=400,
                    triangle_budget=250000):
    """Source-qualified IFC tessellation screen; NO automatic clash approval."""
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError as exc:
        raise RuntimeError("ifcopenshell absent in task container; optional IFC kernel required") from exc
    if not 1<=limit<=2000 or not 100<=triangle_budget<=5000000:
        raise ValueError("element limit 1..2000; triangle budget 100..5000000")
    def names(expr):
        values=[s.strip() for s in expr.split(",")]
        if not values or any(not re.fullmatch(r"Ifc[A-Za-z0-9]{2,50}",v) for v in values):
            raise ValueError("invalid IFC class selector")
        return list(dict.fromkeys(values))
    src=ifcopenshell.open(filename)
    opts=ifcopenshell.geom.settings()
    opts.set(opts.USE_WORLD_COORDS,True)
    skipped=[]; cache={}
    def elements(types):
        result=[]
        for name in types:
            for element in src.by_type(name):
                uid=str(element.GlobalId)
                if uid not in cache:
                    if len(cache)>=limit*2:
                        raise ValueError("IFC element scan budget exceeded; chunk class selection")
                    try:
                        shape=ifcopenshell.geom.create_shape(opts,element)
                        cache[uid]=make_mesh(shape.geometry.verts,shape.geometry.faces)
                    except (RuntimeError,ValueError,AttributeError,TypeError) as exc:
                        skipped.append({"guid":uid,"class":name,
                                        "reason":type(exc).__name__,"detail":str(exc)[:120]})
                        cache[uid]=None
                if cache[uid] is not None and uid not in {x[0] for x in result}:
                    result.append((uid,name,cache[uid]))
        return result
    left,right=elements(names(class_a)),elements(names(class_b))
    summary=Counter(); witnesses=[]; examined=set()
    comparisons=0; ambiguous=0
    for aid,ac,am in left:
        for bid,bc,bm in right:
            if aid==bid:continue
            key=tuple(sorted((aid,bid)))
            if key in examined:continue
            examined.add(key);comparisons+=1
            if not overlaps(am.bbox,bm.bbox):
                summary["aabb_disjoint"]+=1
                continue
            try:
                decision=mesh_relation(am,bm,triangle_budget)
            except ValueError as exc:
                decision={"relation":"unresolved_budget_or_geometry",
                          "reason":str(exc)[:140]}
            relation=decision["relation"]
            summary[relation]+=1
            if relation.startswith("unresolved"): ambiguous+=1
            if len(witnesses)<100:
                witnesses.append({"a":aid,"a_class":ac,"b":bid,"b_class":bc,
                                  **decision})
    return {
        "kind":"ifc-triangulated-contact-screen",
        "element_a_count":len(left),"element_b_count":len(right),
        "elements_skipped":len(skipped),"skipped_samples":skipped[:30],
        "compared_pairs":comparisons,
        "broadphase_overlaps":comparisons-summary["aabb_disjoint"],
        "by_relation":dict(sorted(summary.items())),
        "surface_contact_pairs":summary["surface_contact"],
        "solid_containment_pairs":summary["solid_containment"],
        "inconclusive_pairs":ambiguous+len(skipped),
        "all_pairs_qualified":ambiguous==0 and not skipped,
        "witnesses":witnesses,
        "warning":"Surface SAT detects contact, not physical penetration or code clashes. IFC tessellation is approximate; all results require project-unit confirmation, tolerances, BRep validation, and design review. No clearance distance established.",
    }
