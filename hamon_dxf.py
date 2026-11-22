"""Bounded source-native ASCII DXF dimension geometry, not PDF OCR or design approval."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path

_UNITS={1:"inch",2:"foot",4:"mm",5:"cm",6:"m",7:"km",10:"yard"}
_KINDS={0:"rotated_linear",1:"aligned_linear",2:"angular",3:"diameter",4:"radius",5:"angular_3point",6:"ordinate"}

def _groups(data):
    if b"\x00" in data[:100] or data.startswith(b"AutoCAD Binary DXF"):
        raise ValueError("binary DXF not supported; supply ASCII DXF")
    try: text=data.decode("utf-8-sig")
    except UnicodeDecodeError: text=data.decode("latin-1")
    lines=text.splitlines()
    if len(lines)%2 or len(lines)//2>800000:
        raise ValueError("DXF group codes malformed or file exceeds group bound")
    for i in range(0,len(lines),2):
        try: code=int(lines[i].strip())
        except ValueError as exc: raise ValueError("invalid DXF group code") from exc
        if code < -5 or code > 1071:
            raise ValueError("DXF group code out of bounds")
        yield i//2,code,lines[i+1].strip()

def _get(groups,code):
    return next((v for c,v in groups if c==code),None)

def _float(groups,code,default=None):
    v=_get(groups,code)
    if v is None: return default
    try: n=float(v)
    except ValueError: return None
    return n if math.isfinite(n) else None

def _point(groups,x,y,z):
    a,b,c=_float(groups,x),_float(groups,y),_float(groups,z,0.0)
    return [a,b,c] if None not in (a,b,c) else None

def _model_space(groups):
    # Layout flags belong to the entity, outside application-defined 102 groups.
    depth=0
    for code,value in groups:
        if code==102 and value.startswith("{"): depth+=1
        elif code==102 and value=="}":
            if not depth: return False
            depth-=1
        elif not depth and code==67:
            try:
                if int(value)!=0: return False
            except ValueError: return False
        elif not depth and code==410 and value.casefold()!="model":
            return False
    return depth==0

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def _ccw(start, end):
    return (end - start) % math.tau


def _dimension_picture_sweep(picture, center, height, pointer, rays, kind):
    """Select the oriented sweep witnessed by the linked ARC and its two end rays."""
    if not picture or picture.get('invalid') or not picture['arcs']:
        return None
    radius = abs(pointer)
    tol = max(radius * 1e-7, 16 * math.ulp(max(abs(center), abs(height), radius)))
    if not math.isfinite(radius) or radius <= tol:
        return None
    angular_tol = max(1e-8, tol / radius)
    if angular_tol > 1e-4:
        return None
    phases = [math.atan2(z.imag, z.real) for z in rays]
    def ray_at(phase):
        found = [i for i,t in enumerate(phases) if abs(math.remainder(phase-t, math.tau)) <= angular_tol]
        return found[0] if len(found) == 1 else None

    anchors, intervals = set(), []
    for (x,y,z), r, start, end in picture['arcs']:
        if abs(complex(x,y)-center) > tol or abs(z-height) > tol or abs(r-radius) > tol:
            return None
        start, end = math.radians(start), math.radians(end)
        extent = _ccw(start,end)
        if not angular_tol < extent < math.tau-angular_tol:
            return None
        intervals.append((start,extent))
        for phase in (start,end):
            ray = ray_at(phase)
            if ray is not None:
                anchors.add(ray)
    for vertices in picture['arrows']:
        corners = list(dict.fromkeys(tuple(p) for p in vertices))
        if len(corners) != 3 or any(abs(z-height) > tol for _,_,z in corners):
            continue
        points = [complex(x,y) for x,y,_ in corners]
        tips = set()
        for i, tip in enumerate(points):
            radial = tip - center
            if abs(abs(radial) - radius) > tol:
                continue
            ray = ray_at(math.atan2(radial.imag, radial.real))
            if ray is None:
                continue
            # The base vertices must straddle the tip's tangent line and lie on the
            # same tangent side of the radial line. This rejects flat and radial solids.
            a = (points[(i+1) % 3] - tip) * radial.conjugate() / abs(radial)
            b = (points[(i+2) % 3] - tip) * radial.conjugate() / abs(radial)
            area2 = abs((a.conjugate() * b).imag)
            if (area2 > tol * max(abs(a), abs(b), abs(a-b)) and
                    min(a.real,b.real) < -tol and max(a.real,b.real) > tol and
                    (min(a.imag,b.imag) > tol or max(a.imag,b.imag) < -tol)):
                tips.add(ray)
        if len(tips) > 1:
            return None
        anchors.update(tips)
    if len(anchors) != 2:
        return None
    first, second = sorted(anchors)
    if kind == 'angular' and first//2 == second//2:
        return None
    matches = []
    marker = math.atan2(pointer.imag, pointer.real)
    for i,j in ((first,second),(second,first)):
        start = phases[i]
        sweep = _ccw(start, phases[j])
        if not angular_tol < _ccw(start,marker) < sweep-angular_tol:
            continue
        for arc_start,extent in intervals:
            offset = _ccw(start,arc_start)
            if offset > math.tau-angular_tol:
                offset = 0.
            if offset + extent > sweep + angular_tol:
                break
        else:
            matches.append(math.degrees(sweep))
    return matches[0] if len(matches) == 1 else None


def _dimension_pictures(data, names):
    """Read bounded native dimension-block witnesses. Never use annotation text as geometry."""
    pictures = {}
    section = block = entity = None
    groups = []
    def finish():
        nonlocal block
        if entity == 'BLOCK':
            name = _get(groups, 2)
            block = name if name in names else None
            if block:
                # An anonymous dimension reference uses the model coordinates directly.
                if (block in pictures or not block.upper().startswith('*D') or
                        _point(groups,10,20,30) != [0.,0.,0.]):
                    pictures[block] = None
                else:
                    pictures[block] = {'arcs': [], 'arrows': []}
        elif entity == 'ENDBLK':
            block = None
        elif block and pictures.get(block) is not None:
            p = pictures[block]
            if entity in ('ARC','SOLID'):
                normal = [_float(groups,c,v) for c,v in ((210,0.),(220,0.),(230,1.))]
                if normal != [0.,0.,1.] or len(p['arcs'])+len(p['arrows']) >= 128:
                    pictures[block] = None
                    return
                if entity == 'ARC':
                    arc = [_point(groups,10,20,30),_float(groups,40),_float(groups,50),_float(groups,51)]
                    if None in arc or arc[1] <= 0:
                        pictures[block] = None
                    else:
                        p['arcs'].append(arc)
                else:
                    corners = [_point(groups,c,c+10,c+20) for c in (10,11,12,13)]
                    if None in corners or len(set(map(tuple,corners))) != 3:
                        pictures[block] = None
                    else:
                        p['arrows'].append(corners)
            elif entity not in ('LINE','TEXT','MTEXT','POINT'):
                pictures[block] = None
    it = iter(_groups(data))
    for _,code,value in it:
        if code == 0:
            if section == 'BLOCKS':
                finish()
            entity, groups = value.upper(), []
            if entity == 'SECTION':
                _,key,section = next(it)
                if key != 2:
                    raise ValueError('invalid DXF section marker')
                section, block = section.upper(), None
            elif entity in ('ENDSEC','EOF'):
                section = block = None
        elif section == 'BLOCKS' and (entity == 'BLOCK' or block):
            groups.append((code,value))
    return pictures


def _angular_sector(groups, kind, picture=None):
    """Resolve the open sector containing the authored dimension-arc point."""
    normal = [_float(groups, code, default) for code, default in ((210, 0.), (220, 0.), (230, 1.))]
    if None in normal or not 0 < math.hypot(*normal) < math.inf:
        return None
    normal = tuple(x / math.hypot(*normal) for x in normal)
    # Autodesk's arbitrary-axis rule defines the orientation of the OCS group-16 marker.
    seed = (0., 1., 0.) if max(abs(normal[0]), abs(normal[1])) < 1/64 else (0., 0., 1.)
    axis_x = _cross(seed, normal)
    axis_x = tuple(x / math.hypot(*axis_x) for x in axis_x)
    axis_y = _cross(normal, axis_x)
    points = [_point(groups, c, c+10, c+20) for c in (13, 14, 15, 10)]
    if kind == 'angular':
        marker = _point(groups, 16, 26, 36)
        if marker is None:
            return None
        points.append(tuple(_dot(marker, components) for components in zip(axis_x, axis_y, normal)))
    if any(p is None or not all(map(math.isfinite, p)) for p in points):
        return None
    origin = points[2]
    local = [tuple(x-y for x,y in zip(p, origin)) for p in points]
    scale = max(math.hypot(*p) for p in local)
    epsilon = max(scale * 1e-12, 8 * max(math.ulp(x) for p in points for x in p))
    if not math.isfinite(scale) or scale <= epsilon or any(abs(_dot(p, normal)) > epsilon for p in local):
        return None
    # Local complex coordinates keep the plane's orientation and remove length units from angles.
    a, b, _, d, *marker = [complex(_dot(p, axis_x), _dot(p, axis_y)) for p in local]
    phase = lambda z: math.atan2(z.imag, z.real) % math.tau
    if kind == 'angular':
        # Lines are (13,14) and (15,10). Both rays of each line bound candidate sectors.
        u, v = b-a, d
        if min(abs(u), abs(v)) <= epsilon:
            return None
        u, v = u/abs(u), v/abs(v)
        determinant = (u.conjugate() * v).imag
        if abs(determinant) <= 1e-10:
            return None
        center = a - u * ((a.conjugate() * v).imag / determinant)
        pointer = marker[0] - center
        directions = (u, -u, v, -v)
    else:
        center = 0j
        # The supplied vertex (15) is the local origin. Point 10 selects minor or reflex.
        if min(abs(a), abs(b)) <= epsilon:
            return None
        pointer, directions = d, (a, b)
    if abs(pointer) <= epsilon or not math.isfinite(abs(pointer)):
        return None
    if picture is not None:
        # The block witness is qualified for untransformed +Z model-space graphics only.
        if normal != (0.,0.,1.):
            return None
        return _dimension_picture_sweep(picture, complex(origin[0],origin[1])+center, origin[2],
                                        pointer, directions, kind)
    boundaries = sorted(phase(z) for z in directions)
    angle = phase(pointer)
    spans = [(b-a) % math.tau for a,b in zip(boundaries,boundaries[1:]+boundaries[:1])]
    if min(spans) <= 1e-10:
        return None
    for start, span in zip(boundaries, spans):
        offset = (angle-start) % math.tau
        if 1e-10 < offset < span-1e-10:
            return math.degrees(span)
    return None


def _ordinate(groups, flags, row):
    """Measure an axis-aligned ordinate from its UCS datum, never its leader."""
    axis = 0 if flags & 64 else 1
    datum = _point(groups, 10, 20, 30)
    feature = _point(groups, 13, 23, 33)
    leader = _point(groups, 14, 24, 34)
    evidence = {"axis": ("x", "y")[axis], "datum_wcs": datum, "feature_wcs": feature,
                "leader_endpoint_wcs": leader, "signed_coordinate_model_units": None}
    row["ordinate"] = evidence
    # Ambiguous or application-defined group values cannot establish a datum or axis.
    relevant = {10, 20, 30, 13, 23, 33, 14, 24, 34, 42, 51, 70, 210, 220, 230}
    codes = [c for c, _ in groups if c in relevant]
    if (flags & ~(6 | 32 | 64 | 128) or len(codes) != len(set(codes)) or _get(groups, 102) is not None
            or _float(groups, 51, 0.0) != 0.0
            or tuple(_float(groups, c, v) for c, v in ((210, 0.), (220, 0.), (230, 1.))) != (0., 0., 1.)
            or datum is None or feature is None or leader is None
            or not datum[2] == feature[2] == leader[2] or datum == feature):
        return
    delta = feature[axis] - datum[axis]
    if not math.isfinite(delta):
        return
    evidence["signed_coordinate_model_units"] = delta
    magnitude = abs(delta)
    row["computed_model_units"] = magnitude
    cache = row["cad_measurement_model_units"]
    if _get(groups, 42) is None:
        row["geometry_status"] = "geometry_only"
    elif cache is None or cache < 0 or not math.isclose(cache, magnitude, rel_tol=1e-8, abs_tol=1e-8):
        row["geometry_status"] = "cad_measurement_mismatch"
        return
    else:
        row["geometry_status"] = "geometry_and_cad_agree"
    row["usable_qualified_value"] = magnitude


def _dimension(groups,ordinal,offset,picture=None):
    try: t=int(_get(groups,70) or "0")
    except ValueError: t=-1
    kind=_KINDS.get(t&15,"unknown") if t>=0 else "unknown"
    p1,p2=_point(groups,13,23,33),_point(groups,14,24,34)
    if kind in ("radius","diameter"):
        p1,p2=_point(groups,10,20,30),_point(groups,15,25,35)
    stored=_float(groups,42)
    calculated=None
    if p1 is not None and p2 is not None:
        dx,dy=p2[0]-p1[0],p2[1]-p1[1]
        if kind=="aligned_linear": calculated=math.dist(p1,p2)
        elif kind=="rotated_linear":
            angle=_float(groups,50)
            normal=[_float(groups,210,0.0),_float(groups,220,0.0),_float(groups,230,1.0)]
            if angle is not None and all(
                    v is not None and math.isclose(v,e,abs_tol=1e-10)
                    for v,e in zip(normal,(0.0,0.0,1.0))):
                radians=math.radians(angle)
                calculated=abs(dx*math.cos(radians)+dy*math.sin(radians))
        elif kind in ("radius","diameter"):
            calculated=math.dist(p1,p2)
    if calculated is not None and (calculated<=0 or not math.isfinite(calculated)):
        calculated=None
    if calculated is None: status="geometry_unavailable"
    elif stored is None: status="geometry_only"
    elif math.isclose(calculated,abs(stored),rel_tol=1e-4,abs_tol=1e-6):
        status="geometry_and_cad_agree"
    else: status="cad_measurement_mismatch"
    raw=_get(groups,1)
    row={"ordinal":ordinal,"group_offset":offset,
      "handle":_get(groups,5),"layer":_get(groups,8),
      "kind":kind,"geometry_status":status,
      "axis_origin":_point(groups,10,20,30) if kind=="rotated_linear" else None,
      "definition_points":[p1,p2] if p1 and p2 else None,
      "angle_degrees":_float(groups,50) if kind=="rotated_linear" else None,
      "computed_model_units":calculated,"cad_measurement_model_units":stored,
      "text_override":raw[:240] if raw is not None else None,
      "text_override_is_unverified":True,
      "usable_qualified_value":calculated if status in ("geometry_only","geometry_and_cad_agree") else None}

    if kind == "ordinate":
        _ordinate(groups, t, row)

    if kind in ("angular", "angular_3point"):
        marker = _angular_sector(groups, kind)
        drawn = _angular_sector(groups, kind, picture) if picture else None
        degrees = drawn if drawn is not None else marker
        valid_cache = stored is not None and 0 < stored <= math.tau
        if degrees is None:
            status = "geometry_unavailable"
        elif picture and (picture.get("invalid") or picture["arcs"]) and drawn is None:
            status = "angular_picture_unresolved"
        elif stored is None or stored == -1.0:
            status = "angular_picture_verified" if drawn is not None else "angular_unverified_sector"
        elif valid_cache and math.isclose(math.radians(degrees), stored, rel_tol=1e-6, abs_tol=1e-9):
            status = "angular_picture_and_cad_agree" if drawn is not None else "angular_geometry_and_cad_agree"
        else:
            status = "cad_measurement_mismatch"
        verified = status in ("angular_picture_verified", "angular_picture_and_cad_agree", "angular_geometry_and_cad_agree")
        row.update(geometry_status=status, cad_measurement_model_units=None, angular={
            "marker_sector_degrees": marker, "picture_sweep_degrees": drawn,
            "picture_block": _get(groups, 2), "picture_arc_count": len(picture.get("arcs", [])) if picture else 0,
            "cad_measurement_raw": stored, "cad_measurement_radians": stored if valid_cache else None,
            "usable_degrees": degrees if verified else None,
        })
    return row


def scan_dxf(filename,limit=1000):
    if not 1<=limit<=5000: raise ValueError("dimension limit outside [1,5000]")
    p=Path(filename)
    if not p.is_file() or p.stat().st_size>32000000:
        raise ValueError("missing DXF or larger than 32 MB inspection bound")
    data=p.read_bytes()
    section=None;var=None;units=None;entity=None;offset=0;groups=[]
    results=[];angular_rows=[];seen=0;excluded=0
    def flush():
        nonlocal seen,excluded
        if entity!="DIMENSION":return
        seen+=1
        if not _model_space(groups):
            excluded+=1
            return
        if len(results)<limit:
            row = _dimension(groups,seen,offset)
            if "angular" in row and _get(groups,2):
                angular_rows.append((len(results),groups,seen,offset,_get(groups,2)))
            results.append(row)
    it=iter(_groups(data))
    for idx,code,value in it:
        upper=value.upper() if code in (0,2,9) else value
        if code==0 and upper=="SECTION":
            flush();entity=None;groups=[]
            try: _,c,n=next(it)
            except StopIteration: raise ValueError("incomplete DXF section marker")
            if c!=2:raise ValueError("invalid DXF section marker")
            section=n.upper();var=None
            continue
        if code==0 and upper in ("ENDSEC","EOF"):
            flush();entity=None;groups=[];section=None;var=None
            if upper=="EOF":break
            continue
        if section=="HEADER":
            if code==9:var=upper
            elif code==70 and var=="$INSUNITS":
                try:units=int(value)
                except ValueError:units=None
                var=None
            continue
        if section!="ENTITIES":continue
        if code==0:
            flush()
            entity=upper;groups=[];offset=idx
        elif entity is not None:
            groups.append((code,value))
    if angular_rows:
        pictures = _dimension_pictures(data, {name for _,_,_,_,name in angular_rows})
        for index, record, ordinal, group_offset, name in angular_rows:
            if name in pictures:
                results[index] = _dimension(record,ordinal,group_offset,pictures[name] or {"invalid": True})
    return {"kind":"dxf-native-dimension-evidence",
      "schema":"hamon.dxf.native-dimensions.v1",
      "file":str(p),"source_sha256":hashlib.sha256(data).hexdigest(),
      "source_bytes":len(data),"declared_insunits":units,
      "declared_unit":_UNITS.get(units),
      "unit_status":"declared" if units in _UNITS else "unknown",
      "total_dimensions":seen,"reported_dimensions":len(results),
      "excluded_non_model_dimensions":excluded,
      "dimension_limit_hit":seen-excluded>limit,
      "cad_geometry_agreement_count":sum(d["geometry_status"]=="geometry_and_cad_agree" for d in results),
      "geometry_supported_count":sum(d["usable_qualified_value"] is not None for d in results),
      "angular_geometry_supported_count":sum(d.get("angular", {}).get("usable_degrees") is not None for d in results),
      "dimensions":results,
      "pdf_equivalence_verified":False,"drawing_scale_verified":False,
      "warning":"DXF model-space entity evidence only. CAD measurement 42 is not engineering approval. Text override is unverified. PDF equivalence requires independent registration."}
