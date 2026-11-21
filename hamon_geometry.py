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
        return (visit(left[1],right[1]) or visit(left[1],right[2]) or
                visit(left[2],right[1]) or visit(left[2],right[2]))
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
    vertex_links={}; duplicated_link=False
    # IFC tessellators may emit one vertex index per *face corner*. An
    # indexed-edge test then calls a perfectly closed shell non-watertight.
    # Weld only bit-identical world coordinates for topological evidence:
    # approximate welding could invent closure across a real, narrow gap.
    first_at={}
    welded=[]
    for index,position in enumerate(verts):
        welded.append(first_at.setdefault(position,index))
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
        canonical=[welded[i] for i in abc]
        if len(set(canonical))!=3:
            raise ValueError("mesh triangle collapses after exact-coordinate welding")
        for i in range(3):
            u,v,w=canonical[i],canonical[(i+1)%3],canonical[(i+2)%3]
            union(u,v)
            edges[tuple(sorted((u,v)))]+=1
            oriented[(u,v)]+=1
            link=vertex_links.setdefault(u,{})
            if w in link:
                duplicated_link=True
            else:
                link[w]=v
    def one_vertex_cycle(link):
        # Edge-pair closure alone misses two solid shells touching at one
        # vertex. Each closed orientable manifold vertex has exactly one
        # directed neighbor ring, not two cycles joined at one coordinate.
        if len(link)<3 or set(link)!=set(link.values()):
            return False
        seen=set()
        start=next(iter(link))
        cur=start
        while cur not in seen:
            seen.add(cur)
            cur=link[cur]
        return cur==start and len(seen)==len(link)
    closed=not duplicated_link and all(n==2 for n in edges.values()) and all(
        oriented[(u,v)]==oriented[(v,u)]
        for u,v in edges) and all(one_vertex_cycle(link)
                                for link in vertex_links.values())
    reps={}
    active={welded[index] for index in ff}
    for index in sorted(active): reps.setdefault(root(index),index)
    return Mesh(verts,tuple(triangles),tuple(tr_boxes),bounds(verts),
                node_for(triangles,list(range(len(triangles)))),
                closed,tuple(verts[i] for i in reps.values()))


def _ray_hits_box(point,direction,box,epsilon=1e-9):
    """Conservative forward ray/AABB slab screen in world metres.

    Expand both sides by a positive epsilon so rounding can increase work,
    but cannot silently prune a triangle whose boundary is on the ray.
    """
    near,far=0.0,float("inf")
    for i in range(3):
        lo,hi=box[0][i]-epsilon,box[1][i]+epsilon
        speed=direction[i]
        if speed==0.0:
            if point[i]<lo or point[i]>hi:
                return False
            continue
        t0,t1=(lo-point[i])/speed,(hi-point[i])/speed
        near=max(near,min(t0,t1))
        far=min(far,max(t0,t1))
        if far<near:
            return False
    return far>=0.0


def _ray_crossings_bvh(point,direction,mesh,stats=None):
    """Count positive ray/surface hits or return None on a grazing witness.

    Indexed BVH traversal replaces a whole-mesh scan; leaf triangle bounds
    are checked again. No parity decision is made from an unresolved ray.
    """
    hits=0
    stack=[mesh.bvh]
    while stack:
        node=stack.pop()
        if stats is not None:
            stats["bvh_nodes"]=stats.get("bvh_nodes",0)+1
        if not _ray_hits_box(point,direction,node[0]):
            continue
        if node[3] is None:
            stack.append(node[2])
            stack.append(node[1])
            continue
        for tri_index in node[3]:
            if not _ray_hits_box(point,direction,mesh.triangle_boxes[tri_index]):
                continue
            if stats is not None:
                stats["triangle_tests"]=stats.get("triangle_tests",0)+1
            t=_ray_triangle(point,direction,mesh.triangles[tri_index])
            if t==-1:
                return None
            if t!=0:
                hits+=1
    return hits


def inside_closed(point,mesh,stats=None):
    """Watertight oriented point-in-solid, BVH-pruned with 3-ray consensus.

    stats, when passed, records actual BVH nodes and triangle tests without
    affecting classification. Distinct ray crossings must agree or the
    caller receives unresolved rather than an invented clearance.
    """
    if not mesh.closed_oriented:
        return None
    rays=((1.0,0.416234,0.271828),(0.357,1.0,0.618034),
          (0.12345,0.517,1.0),(1.0,0.927,0.211))
    votes=[]
    for direction in rays:
        if stats is not None:
            stats["rays"]=stats.get("rays",0)+1
        crossings=_ray_crossings_bvh(point,direction,mesh,stats)
        if crossings is not None:
            votes.append(bool(crossings%2))
    if len(votes)>=3 and len(set(votes))==1:
        return votes[0]
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



# Closest-point and branch-and-bound surface gap in metres. The separation
# test uses all triangle vertex-face and edge-edge feature pairs, with an
# explicit SAT zero-distance branch for interior surface crossings.
# See Ericson, Real-Time Collision Detection (2005), sections 5.1.5/5.1.9/5.1.11.
def _sum(a,b): return tuple(x+y for x,y in zip(a,b))
def _scaled(a,s): return tuple(x*s for x in a)
def _d2(a,b): return dot(minus(a,b),minus(a,b))
def _clamp01(x): return max(0.0,min(1.0,x))


def _closest_on_triangle(p,tri):
    a,b,c=tri
    ab,ac=minus(b,a),minus(c,a)
    ap=minus(p,a)
    d1,d2=dot(ab,ap),dot(ac,ap)
    if d1<=0 and d2<=0:return a
    bp=minus(p,b)
    d3,d4=dot(ab,bp),dot(ac,bp)
    if d3>=0 and d4<=d3:return b
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        return _sum(a,_scaled(ab,d1/(d1-d3)))
    cp=minus(p,c)
    d5,d6=dot(ab,cp),dot(ac,cp)
    if d6>=0 and d5<=d6:return c
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        return _sum(a,_scaled(ac,d2/(d2-d6)))
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:
        bc=minus(c,b)
        return _sum(b,_scaled(bc,(d4-d3)/((d4-d3)+(d5-d6))))
    denom=va+vb+vc
    if denom<=0:
        raise ValueError("numerically unresolved closest triangle feature")
    v,w=vb/denom,vc/denom
    return _sum(a,_sum(_scaled(ab,v),_scaled(ac,w)))


def _closest_segments(p1,q1,p2,q2):
    u,v=minus(q1,p1),minus(q2,p2)
    w=minus(p1,p2)
    a,e=dot(u,u),dot(v,v)
    if a<=0 or e<=0:
        raise ValueError("zero-length mesh segment")
    b,c,f=dot(u,v),dot(u,w),dot(v,w)
    denom=a*e-b*b
    if denom>1e-12*a*e:
        s=_clamp01((b*f-c*e)/denom)
    else:s=0.0
    t=(b*s+f)/e
    if t<0:
        t=0.0;s=_clamp01(-c/a)
    elif t>1:
        t=1.0;s=_clamp01((b-c)/a)
    return _sum(p1,_scaled(u,s)),_sum(p2,_scaled(v,t))


def _tri_pair_gap(a,b):
    if tri_overlap(a,b):
        # Real surfaces intersect/touch; no unique closest-point witness.
        return 0.0,None,None
    best=(float("inf"),None,None)
    for p in a:
        q=_closest_on_triangle(p,b)
        cand=(_d2(p,q),p,q)
        if cand[0]<best[0]:best=cand
    for q in b:
        p=_closest_on_triangle(q,a)
        cand=(_d2(p,q),p,q)
        if cand[0]<best[0]:best=cand
    for i in range(3):
        for j in range(3):
            p,q=_closest_segments(a[i],a[(i+1)%3],
                                  b[j],b[(j+1)%3])
            cand=(_d2(p,q),p,q)
            if cand[0]<best[0]:best=cand
    return best


def _box_gap2(a,b):
    gaps=(max(a[0][i]-b[1][i],b[0][i]-a[1][i],0.0)
          for i in range(3))
    return sum(g*g for g in gaps)


def surface_distance(mesh_a,mesh_b,triangle_budget=250000):
    """BVH best-first exact triangulated-surface distance, no voxel sampling.

    On a budget exceed, never substitute AABB bound as the actual surface gap.
    Returns exact source-coordinate closest points for separated triangles.
    """
    import heapq
    if triangle_budget<1:
        raise ValueError("triangle-distance budget must be positive")
    best2=float("inf")
    winner=(None,None)
    comparisons=0
    heap=[]
    seq=0
    def push(a,b):
        nonlocal seq
        lo=_box_gap2(a[0],b[0])
        if lo<best2:
            seq+=1
            heapq.heappush(heap,(lo,seq,a,b))
    push(mesh_a.bvh,mesh_b.bvh)
    while heap:
        lower,_,na,nb=heapq.heappop(heap)
        if lower>=best2:break
        if na[3] is not None and nb[3] is not None:
            for i in na[3]:
                for j in nb[3]:
                    if _box_gap2(mesh_a.triangle_boxes[i],
                                 mesh_b.triangle_boxes[j])>=best2:
                        continue
                    comparisons+=1
                    if comparisons>triangle_budget:
                        raise ValueError("triangle-distance budget exhausted")
                    d2,p,q=_tri_pair_gap(mesh_a.triangles[i],mesh_b.triangles[j])
                    if d2<best2:best2=d2;winner=(p,q)
            continue
        if na[3] is not None:
            push(na,nb[1]);push(na,nb[2])
        elif nb[3] is not None:
            push(na[1],nb);push(na[2],nb)
        else:
            sa=sum((na[0][1][i]-na[0][0][i])**2 for i in range(3))
            sb=sum((nb[0][1][i]-nb[0][0][i])**2 for i in range(3))
            if sa>=sb:
                push(na[1],nb);push(na[2],nb)
            else:
                push(na,nb[1]);push(na,nb[2])
    if not math.isfinite(best2):
        raise ValueError("no closest triangle pair qualified")
    return {
        "surface_distance_m":math.sqrt(max(0.0,best2)),
        "closest_a_world_m":list(winner[0]) if winner[0] is not None else None,
        "closest_b_world_m":list(winner[1]) if winner[1] is not None else None,
        "distance_triangle_tests":comparisons,
    }


def ifc_narrowphase(filename,class_a,class_b,limit=400,
                    triangle_budget=250000,min_clearance_mm=None):
    """Source-qualified IFC tessellation screen; NO automatic clash approval."""
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError as exc:
        raise RuntimeError("ifcopenshell absent in task container; optional IFC kernel required") from exc
    if not 1<=limit<=2000 or not 100<=triangle_budget<=5000000:
        raise ValueError("element limit 1..2000; triangle budget 100..5000000")
    if min_clearance_mm is not None and (
        isinstance(min_clearance_mm,bool) or not isinstance(min_clearance_mm,(int,float))
        or not math.isfinite(min_clearance_mm) or min_clearance_mm<0
        or min_clearance_mm>1000000):
        raise ValueError("minimum clearance must be finite millimetres within [0,1000000]")
    clearance_m=None if min_clearance_mm is None else min_clearance_mm/1000.0
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
        selected=set()
        result=[]
        for name in types:
            for element in src.by_type(name):
                uid=str(element.GlobalId)
                if uid in selected:
                    continue
                selected.add(uid)
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
                if cache[uid] is not None:
                    result.append((uid,name,cache[uid]))
        return result,selected
    left,left_ids=elements(names(class_a))
    right,right_ids=elements(names(class_b))
    shared=len(left_ids & right_ids)
    # Remove self-pairs and the second ordering of every pair in the intersection.
    requested_pairs=len(left_ids)*len(right_ids)-shared*(shared+1)//2
    summary=Counter(); witnesses=[]; examined=set()
    comparisons=0; ambiguous=0; distance_queries=0
    clearance_failures=0; clearance_unknown=0
    for aid,ac,am in left:
        for bid,bc,bm in right:
            if aid==bid:continue
            key=tuple(sorted((aid,bid)))
            if key in examined:continue
            examined.add(key);comparisons+=1
            if comparisons > 100000:
                raise ValueError("100000 IFC element-pair budget exhausted; narrow class selectors")
            aabb_overlap=overlaps(am.bbox,bm.bbox)
            if not aabb_overlap:
                decision={"relation":"aabb_disjoint","triangle_tests":0}
            else:
                try:
                    decision=mesh_relation(am,bm,triangle_budget)
                except ValueError as exc:
                    decision={"relation":"unresolved_budget_or_geometry",
                              "reason":str(exc)[:140]}
            relation=decision["relation"]
            summary[relation]+=1
            required=clearance_m
            # Without an explicit tolerance, quantify all envelope overlaps
            # where the previous broad phase had only a false-positive box.
            need_distance=aabb_overlap
            if required is not None and not aabb_overlap:
                # AABBs farther than the requested threshold are safe lower
                # bounds; near but disjoint AABBs MUST get triangle distances.
                need_distance=(_box_gap2(am.bbox,bm.bbox) <= required**2)
            if need_distance and relation!="surface_contact":
                try:
                    metric=surface_distance(am,bm,triangle_budget)
                    decision.update(metric)
                    decision["surface_distance_mm"]=metric["surface_distance_m"]*1000
                    distance_queries+=1
                except ValueError as exc:
                    decision["distance_error"]=str(exc)[:140]
            elif relation=="surface_contact":
                decision["surface_distance_mm"]=0.0
            if required is not None:
                if relation in ("surface_contact","solid_containment"):
                    decision["clearance_status"]="contact_or_solid_overlap"
                    clearance_failures+=1
                elif relation.startswith("unresolved") or "distance_error" in decision:
                    decision["clearance_status"]="unresolved"
                    clearance_unknown+=1
                elif not aabb_overlap and not need_distance:
                    decision["clearance_status"]="meets_from_aabb_lower_bound"
                elif not am.closed_oriented or not bm.closed_oriented:
                    # Distance between two incomplete surfaces does not prove
                    # minimum volume clearance; retain uncertainty.
                    decision["clearance_status"]="unresolved_open_surface"
                    clearance_unknown+=1
                elif decision.get("surface_distance_mm") is None:
                    decision["clearance_status"]="unresolved_missing_distance"
                    clearance_unknown+=1
                elif decision["surface_distance_mm"] < min_clearance_mm:
                    decision["clearance_status"]="below_specified_gap"
                    clearance_failures+=1
                else:
                    decision["clearance_status"]="meets_triangulated_gap"
            if relation.startswith("unresolved") or decision.get("clearance_status","").startswith("unresolved"):
                ambiguous+=1
            if len(witnesses)<100 or (required is not None and
                                     decision.get("clearance_status") not in
                                     ("meets_from_aabb_lower_bound","meets_triangulated_gap")):
                # Preserve failing witnesses even after 100 safe pairs; bound
                # reporting to 1000 to avoid the evaluator flooding the context.
                if len(witnesses)<1000:
                    witnesses.append({"a":aid,"a_class":ac,"b":bid,"b_class":bc,
                                      **decision})
    return {
        "kind":"ifc-triangulated-contact-screen",
        "element_a_count":len(left),"element_b_count":len(right),
        "elements_skipped":len(skipped),"skipped_samples":skipped[:30],
        "requested_pairs":requested_pairs,"unmeasured_pairs":requested_pairs-comparisons,
        "compared_pairs":comparisons,
        "broadphase_overlaps":comparisons-summary["aabb_disjoint"],
        "by_relation":dict(sorted(summary.items())),
        "surface_contact_pairs":summary["surface_contact"],
        "solid_containment_pairs":summary["solid_containment"],
        "inconclusive_pairs":requested_pairs-comparisons+ambiguous,
        "distance_pairs_evaluated":distance_queries,
        "min_clearance_required_mm":min_clearance_mm,
        "clearance_failures":clearance_failures,
        "clearance_unresolved_pairs":clearance_unknown,
        "all_pairs_qualified":ambiguous==0 and clearance_unknown==0 and not skipped,
        "witnesses":witnesses,
        "warning":"All gaps are on emitted SI-metre triangle surfaces, not validated BRep solids. Surface contact/containment can represent physical overlap; verify design semantics, tolerance, openings, source units and governing code independently.",
    }
