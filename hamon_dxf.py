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

def _float(groups,code):
    v=_get(groups,code)
    if v is None: return None
    try: n=float(v)
    except ValueError: return None
    return n if math.isfinite(n) else None

def _point(groups,x,y,z):
    a,b,c=_float(groups,x),_float(groups,y),_float(groups,z)
    return [a,b,c if c is not None else 0.0] if a is not None and b is not None else None

def _dimension(groups,ordinal,offset):
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
            normal=[_float(groups,210),_float(groups,220),_float(groups,230)]
            normal=[v if v is not None else default
                    for v,default in zip(normal,(0.0,0.0,1.0))]
            if angle is not None and all(
                    math.isclose(v,e,abs_tol=1e-10)
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
    return {"ordinal":ordinal,"group_offset":offset,
      "handle":_get(groups,5),"layer":_get(groups,8),
      "kind":kind,"geometry_status":status,
      "axis_origin":_point(groups,10,20,30) if kind=="rotated_linear" else None,
      "definition_points":[p1,p2] if p1 and p2 else None,
      "angle_degrees":_float(groups,50) if kind=="rotated_linear" else None,
      "computed_model_units":calculated,"cad_measurement_model_units":stored,
      "text_override":raw[:240] if raw is not None else None,
      "text_override_is_unverified":True,
      "usable_qualified_value":calculated if status in ("geometry_only","geometry_and_cad_agree") else None}

def scan_dxf(filename,limit=1000):
    if not 1<=limit<=5000: raise ValueError("dimension limit outside [1,5000]")
    p=Path(filename)
    if not p.is_file() or p.stat().st_size>32000000:
        raise ValueError("missing DXF or larger than 32 MB inspection bound")
    data=p.read_bytes()
    section=None;var=None;units=None;entity=None;offset=0;groups=[]
    results=[];seen=0
    def flush():
        nonlocal seen
        if entity!="DIMENSION":return
        seen+=1
        if len(results)<limit:
            results.append(_dimension(groups,seen,offset))
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
    return {"kind":"dxf-native-dimension-evidence",
      "schema":"hamon.dxf.native-dimensions.v1",
      "file":str(p),"source_sha256":hashlib.sha256(data).hexdigest(),
      "source_bytes":len(data),"declared_insunits":units,
      "declared_unit":_UNITS.get(units),
      "unit_status":"declared" if units in _UNITS else "unknown",
      "total_dimensions":seen,"reported_dimensions":len(results),
      "dimension_limit_hit":seen>limit,
      "cad_geometry_agreement_count":sum(d["geometry_status"]=="geometry_and_cad_agree" for d in results),
      "geometry_supported_count":sum(d["usable_qualified_value"] is not None for d in results),
      "dimensions":results,
      "pdf_equivalence_verified":False,"drawing_scale_verified":False,
      "warning":"DXF model-space entity evidence only. CAD measurement 42 is not engineering approval. Text override is unverified. PDF equivalence requires independent registration."}
