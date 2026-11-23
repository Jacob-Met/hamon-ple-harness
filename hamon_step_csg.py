"""Exact regularized Boolean material sets for rectilinear STEP blocks."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
import math
import re

def _ref(token):
    if not isinstance(token,str) or not re.fullmatch(r'#[1-9][0-9]*',token.strip()):
        raise ValueError('STEP CSG requires a complete entity reference')
    return int(token.strip()[1:])

def _number(token):
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?',token.strip()):
        raise ValueError('STEP CSG requires finite decimal coordinates and dimensions')
    return Fraction(token.strip())

def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def select_csg_root(entities, selector):
    roots={i for i,(kind,_) in entities.items() if kind=='CSG_SOLID'}
    if selector is not None:
        if isinstance(selector,bool) or not isinstance(selector,int):
            raise ValueError('STEP solid_entity_id must be an integer solid definition')
        return selector if selector in roots else None
    if not roots:return None
    referenced=set()
    for kind,args in entities.values():
        if kind=='BOOLEAN_RESULT' and len(args)==4:
            referenced.update(_ref(v) for v in args[2:])
    candidates=roots-referenced
    if len(candidates)!=1:
        raise ValueError('STEP has ambiguous or cyclic CSG roots; specify solid_entity_id')
    selected=next(iter(candidates))
    if any(kind in {'MANIFOLD_SOLID_BREP','FACETED_BREP','BREP_WITH_VOIDS',
                    'MAPPED_ITEM','SOLID_REPLICA','CONTEXT_DEPENDENT_SHAPE_REPRESENTATION',
                    'REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION'}
           or (kind.endswith('_SOLID') and kind!='CSG_SOLID') for kind,_ in entities.values()):
        raise ValueError('STEP has other solid definitions or occurrences; select a CSG definition explicitly')
    for kind,args in entities.values():
        if kind=='SHAPE_REPRESENTATION' or kind.endswith('_SHAPE_REPRESENTATION'):
            if len(args)!=3 or not (args[1].startswith('(') and args[1].endswith(')')):
                raise ValueError('STEP CSG representation is incomplete')
            items=[_ref(v) for v in args[1][1:-1].split(',')]
            if any(i!=selected and entities.get(i,(None,[]))[0]!='AXIS2_PLACEMENT_3D' for i in items):
                raise ValueError('STEP representation includes other material; select a CSG definition explicitly')
    return selected

def _block(entities,args,scale):
    if len(args)!=5:raise ValueError('STEP CSG block is incomplete')
    kind,placement=entities.get(_ref(args[1]),(None,[]))
    if kind!='AXIS2_PLACEMENT_3D' or len(placement)!=4:
        raise ValueError('STEP CSG block requires a three-dimensional placement')
    def vector(token,kind,default=None):
        if token=='$' and default is not None:return default
        actual,values=entities.get(_ref(token),(None,[]))
        if actual!=kind or len(values)!=2 or not (values[1].startswith('(') and values[1].endswith(')')):
            raise ValueError('STEP CSG placement references a missing or wrong coordinate type')
        result=tuple(_number(v) for v in values[1][1:-1].split(','))
        if len(result)!=3 or (default is not None and not any(result)):
            raise ValueError('STEP CSG placement requires three coordinates and nonzero directions')
        return result
    origin=vector(placement[1],'CARTESIAN_POINT')
    z=vector(placement[2],'DIRECTION',(0,0,1))
    x=vector(placement[3],'DIRECTION',(1,0,0))
    y=_cross(z,x)
    if not any(y):
        if placement[3]!='$':raise ValueError('STEP CSG has parallel declared placement axes')
        y=_cross(z,(0,1,0))
    axes=(_cross(y,z),y,z)
    directions=[]
    for axis in axes:
        if sum(v!=0 for v in axis)!=1:
            raise ValueError('STEP CSG block placement must be exactly rectilinear')
        directions.append(tuple((1 if v>0 else -1) if v else 0 for v in axis))
    lengths=tuple(_number(v) for v in args[2:])
    if any(v<=0 for v in lengths):raise ValueError('STEP CSG block dimensions must be positive')
    delta=[sum(lengths[j]*directions[j][i] for j in range(3)) for i in range(3)]
    return tuple((min(origin[i],origin[i]+delta[i])*scale,
                  max(origin[i],origin[i]+delta[i])*scale) for i in range(3))

def _material(entities,root,scale):
    expressions={};leaves={};visiting=set()
    def parse(eid):
        if eid in visiting:raise ValueError('STEP CSG expression contains a cycle')
        if eid in expressions:return
        if len(expressions)+len(visiting)>=128:raise ValueError('STEP CSG exceeds 128 expression nodes')
        visiting.add(eid)
        kind,args=entities.get(eid,(None,[]))
        if kind=='BLOCK':
            if len(leaves)>=32:raise ValueError('STEP CSG exceeds 32 distinct block leaves')
            leaves[eid]=_block(entities,args,scale)
            expressions[eid]=('BLOCK',)
        elif kind=='CSG_SOLID' and len(args)==2:
            child=_ref(args[1])
            if entities.get(child,(None,[]))[0] not in {'BLOCK','BOOLEAN_RESULT'}:
                raise ValueError('STEP CSG root must be a supported primitive or Boolean expression')
            parse(child);expressions[eid]=('ROOT',child)
        elif kind=='BOOLEAN_RESULT' and len(args)==4:
            if args[1] not in {'.UNION.','.INTERSECTION.','.DIFFERENCE.'}:
                raise ValueError('STEP CSG Boolean operator is unsupported')
            children=tuple(_ref(v) for v in args[2:])
            for child in children:parse(child)
            expressions[eid]=(args[1],*children)
        else:raise ValueError('STEP CSG contains an unsupported or incomplete operand')
        visiting.remove(eid)
    parse(root)
    planes=[sorted({v for box in leaves.values() for v in box[k]}) for k in range(3)]
    if math.prod(len(p)-1 for p in planes)>32768:
        raise ValueError('STEP CSG arrangement exceeds 32768 exact cells')
    positions=[{v:i for i,v in enumerate(p)} for p in planes]
    cache={}
    def evaluate(eid):
        if eid in cache:return cache[eid]
        expr=expressions[eid]
        if expr[0]=='BLOCK':
            box=leaves[eid]
            result=set(product(*(range(positions[k][box[k][0]],positions[k][box[k][1]]) for k in range(3))))
        elif expr[0]=='ROOT':result=evaluate(expr[1])
        else:
            a,b=evaluate(expr[1]),evaluate(expr[2])
            result=a|b if expr[0]=='.UNION.' else (a&b if expr[0]=='.INTERSECTION.' else a-b)
        cache[eid]=result
        return result
    occupied=evaluate(root)
    if not occupied:raise ValueError('STEP CSG regularized result has empty interior')
    return planes,occupied,len(leaves)

def _neighbor(cell,axis,step):
    return tuple(v+step if k==axis else v for k,v in enumerate(cell))

def _boundary(planes,occupied):
    reached={next(iter(occupied))};pending=list(reached)
    while pending:
        cell=pending.pop()
        for axis,step in product(range(3),(-1,1)):
            other=_neighbor(cell,axis,step)
            if other in occupied and other not in reached:
                reached.add(other);pending.append(other)
    if len(reached)!=len(occupied):
        raise ValueError('STEP CSG solid must have one connected interior')
    vertices=[];indices={};triangles=[];area=Fraction(0);face_count=0
    area_normal=[Fraction(0) for _ in range(3)]
    for cell in sorted(occupied):
        for axis,step in product(range(3),(-1,1)):
            if _neighbor(cell,axis,step) in occupied:continue
            a,b=(axis+1)%3,(axis+2)%3
            face_area=(planes[a][cell[a]+1]-planes[a][cell[a]])*(planes[b][cell[b]+1]-planes[b][cell[b]])
            area+=face_area;area_normal[axis]+=step*face_area
            corners=[]
            for u,v in ((0,0),(1,0),(1,1),(0,1)):
                point=list(cell);point[axis]+=step==1;point[a]+=u;point[b]+=v;point=tuple(point)
                if point not in indices:
                    indices[point]=len(vertices)
                    vertices.append(tuple(planes[k][point[k]] for k in range(3)))
                corners.append(indices[point])
            if step<0:corners.reverse()
            triangles.extend(((corners[0],corners[1],corners[2]),(corners[0],corners[2],corners[3])))
            face_count+=1
    if any(area_normal):raise ValueError('STEP CSG boundary has an unbalanced oriented area')
    return vertices,triangles,area,face_count

def _boundary_components(triangles):
    edges=defaultdict(list);stars=defaultdict(lambda:defaultdict(set))
    parents=list(range(len(triangles)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for index,(a,b,c) in enumerate(triangles):
        for u,v,w in ((a,b,c),(b,c,a),(c,a,b)):
            edges[min(u,v),max(u,v)].append((index,u<v))
            stars[u][v].add(w);stars[u][w].add(v)
    for incident in edges.values():
        if len(incident)!=2 or incident[0][1]==incident[1][1]:
            raise ValueError('STEP CSG boundary has a nonmanifold edge')
        parents[root(incident[0][0])]=root(incident[1][0])
    for link in stars.values():
        reached={next(iter(link))};pending=list(reached)
        while pending:
            for n in link[pending.pop()]:
                if n not in reached:reached.add(n);pending.append(n)
        if len(reached)!=len(link) or any(len(v)!=2 for v in link.values()):
            raise ValueError('STEP CSG boundary has a nonmanifold vertex')
    return len({root(i) for i in range(len(triangles))})

def _cell_moments(planes,occupied,density):
    volume=Fraction(0);first=[Fraction(0) for _ in range(3)]
    second=[[Fraction(0) for _ in range(3)] for _ in range(3)]
    for cell in occupied:
        limits=[(planes[k][cell[k]],planes[k][cell[k]+1]) for k in range(3)]
        v=math.prod(b-a for a,b in limits)
        center=[(a+b)/2 for a,b in limits]
        volume+=v
        for i in range(3):
            first[i]+=v*center[i]
            for j in range(3):
                a,b=limits[i]
                second[i][j]+=v*((a*a+a*b+b*b)/3 if i==j else center[i]*center[j])
    centroid=[f/volume for f in first]
    covariance=[[second[i][j]-first[i]*first[j]/volume for j in range(3)] for i in range(3)]
    trace=sum(covariance[i][i] for i in range(3))
    inertia=[[density*((trace if i==j else 0)-covariance[i][j]) for j in range(3)] for i in range(3)]
    return volume,centroid,inertia

def csg_mass_properties(entities,selected,unit_scale,density):
    if not math.isfinite(density) or density<0:
        raise ValueError('STEP CSG density must be finite and nonnegative')
    if not math.isfinite(unit_scale) or unit_scale<=0:
        raise ValueError('STEP CSG length scale must be finite and positive')
    planes,occupied,leaf_count=_material(entities,selected,Fraction(unit_scale))
    vertices,triangles,area,face_count=_boundary(planes,occupied)
    shell_count=_boundary_components(triangles)
    from hamon_step_surface import integrate_closed_surface
    properties=integrate_closed_surface(vertices,triangles,density)
    if 'reason' in properties:return properties,0.
    volume,center,inertia=_cell_moments(planes,occupied,Fraction(density))
    if (properties['volume_m3']!=float(volume)
            or properties['centroid_m']!=list(map(float,center))
            or properties['inertia_tensor_centroid_kg_m2']!=[list(map(float,row)) for row in inertia]):
        raise ValueError('STEP CSG cell and boundary moment integrals disagree')
    try:surface_area=float(area)
    except OverflowError as exc:raise ValueError('STEP CSG surface area is outside the finite range') from exc
    if not math.isfinite(surface_area) or surface_area<=0:
        raise ValueError('STEP CSG surface area is outside the finite range')
    properties.update(surface_area_m2=surface_area,watertight_manifold=True,
                      embedded_surface_verified=True,connected_component_count=shell_count,
                      void_shell_count=shell_count-1,csg_interior_component_count=1,
                      csg_leaf_count=leaf_count,csg_cell_count=len(occupied),
                      csg_boundary_face_count=face_count,triangle_count=len(triangles),
                      net_oriented_area_residual_m2=0.,net_oriented_area_tolerance_m2=0.,
                      csg_cell_boundary_moments_verified=True,
                      mass_moment_method='exact_decimal_csg_cells_and_boundary')
    return properties,float(volume)
