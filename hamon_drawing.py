"""Bounded PDF text, vector inventory and coordinate-grounded raster windows.

Text boxes are in the PDF-page coordinate frame, vector paths in Cairo SVG
viewBox coordinates. Any physical drawing scale is an unverified annotation,
never automatically applied to measurements. Detail-link candidates can use local
Tesseract on bounded rendered captions. No network calls.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET


_FLOAT=r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PART=re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
_NUMBER=re.compile(_FLOAT)
_TOKENS=re.compile(r"([A-Za-z])|("+_FLOAT+r")")
_ID=(1.0,0.0,0.0,1.0,0.0,0.0)
_SVG_FRAME = "poppler-cropbox-rotated-no-fit-v1"


def local(tag):return str(tag).rsplit("}",1)[-1]


def mmul(a,b):
    """Affine a after b: x'=a*x+c*y+e, y'=b*x+d*y+f."""
    return (a[0]*b[0]+a[2]*b[1],a[1]*b[0]+a[3]*b[1],
            a[0]*b[2]+a[2]*b[3],a[1]*b[2]+a[3]*b[3],
            a[0]*b[4]+a[2]*b[5]+a[4],
            a[1]*b[4]+a[3]*b[5]+a[5])


def transform_point(mat,p):
    x,y=p
    return (mat[0]*x+mat[2]*y+mat[4],
            mat[1]*x+mat[3]*y+mat[5])


def parse_transform(source):
    if not source: return _ID
    value=_ID
    consumed=""
    for match in _PART.finditer(source):
        consumed+=source[len(consumed):match.start()] if len(consumed)<match.start() else ""
        kind=match.group(1).lower()
        nums=[float(x.group()) for x in _NUMBER.finditer(match.group(2))]
        if not all(math.isfinite(v) for v in nums):return None
        if kind=="matrix" and len(nums)==6:mat=tuple(nums)
        elif kind=="translate" and len(nums) in (1,2):
            mat=(1,0,0,1,nums[0],nums[1] if len(nums)==2 else 0)
        elif kind=="scale" and len(nums) in (1,2):
            mat=(nums[0],0,0,nums[1] if len(nums)==2 else nums[0],0,0)
        elif kind=="rotate" and len(nums) in (1,3):
            rad=math.radians(nums[0]);co,si=math.cos(rad),math.sin(rad)
            rot=(co,si,-si,co,0,0)
            if len(nums)==3:
                mat=mmul(mmul((1,0,0,1,nums[1],nums[2]),rot),
                         (1,0,0,1,-nums[1],-nums[2]))
            else:mat=rot
        else:return None
        value=mmul(value,mat)
    if not list(_PART.finditer(source)):return None
    # Fail closed on any unrecognized transform material between functions.
    if re.sub(r"\s+|,","",_PART.sub("",source)):
        return None
    return value


def _path_points(data):
    """Exact bounds ONLY for straight M/L/H/V/Z sequences, no curve guesses."""
    tokens=[c if c else n for c,n in _TOKENS.findall(data)]
    if not tokens:return None
    cmd=None;index=0;cursor=(0.,0.);start=None;pts=[]
    while index<len(tokens):
        token=tokens[index]
        if token.isalpha():
            cmd=token;index+=1
            if cmd not in "MmLlHhVvZz":return None
            if cmd in "Zz":
                if start is None:return None
                cursor=start;pts.append(cursor)
                cmd=None
                continue
        if cmd is None or index>=len(tokens):return None
        typ=cmd.upper();rel=cmd.islower()
        if typ in ("M","L"):
            if index+1>=len(tokens):return None
            try:x,y=float(tokens[index]),float(tokens[index+1])
            except ValueError:return None
            index+=2
            if rel:x,y=x+cursor[0],y+cursor[1]
            cursor=(x,y)
            if typ=="M":
                start=cursor
                cmd="l" if rel else "L"
        elif typ in ("H","V"):
            try:value=float(tokens[index])
            except ValueError:return None
            index+=1
            if typ=="H":cursor=(cursor[0]+value if rel else value,cursor[1])
            else:cursor=(cursor[0],cursor[1]+value if rel else value)
        else:return None
        pts.append(cursor)
    if not pts or not all(math.isfinite(q) for p in pts for q in p):
        return None
    return pts


def _svg_curve_bbox(data, matrix):
    """Exact affine-transformed M/L/H/V/C/S/Q/T/Z SVG centerline bounds.

    Curves are bounded at true interior derivative roots, not at control
    polygons, endpoints or arbitrary sampled points. Unsupported arc and
    malformed commands are unlocalized rather than silently approximated.
    """
    if matrix is None or not data or len(data)>500000:
        return None
    if re.sub(r"[\s,]+","",_TOKENS.sub("",data)):
        return None
    tokens=[c if c else n for c,n in _TOKENS.findall(data)]
    if len(tokens)>100000:
        return None
    bounds=[math.inf,math.inf,-math.inf,-math.inf]
    cur=(0.0,0.0)
    start=None
    op=None
    cmd=None
    previous=None
    previous_cubic=None
    previous_quadratic=None
    index=0
    steps=0
    def add_point(p):
        q=transform_point(matrix,p)
        if not all(math.isfinite(v) for v in q):
            raise ArithmeticError("SVG coordinate overflow")
        bounds[0]=min(bounds[0],q[0])
        bounds[1]=min(bounds[1],q[1])
        bounds[2]=max(bounds[2],q[0])
        bounds[3]=max(bounds[3],q[1])
    def roots(p0,p1,p2,p3):
        a=-p0+3*p1-3*p2+p3
        b=2*(p0-2*p1+p2)
        c=p1-p0
        if abs(a)<1e-12:
            return [-c/b] if abs(b)>1e-12 else []
        disc=b*b-4*a*c
        if disc<0:return []
        sqrt=math.sqrt(disc)
        q=-0.5*(b+math.copysign(sqrt,b))
        if abs(q)<1e-12:
            return [-b/(2*a)]
        return [q/a,c/q]
    def add_curve(p0,p1,p2,p3):
        pts=[transform_point(matrix,p) for p in (p0,p1,p2,p3)]
        if any(not math.isfinite(v) for p in pts for v in p):
            raise ArithmeticError("SVG curve overflow")
        ts=[0.0,1.0]
        for axis in (0,1):
            v=[p[axis] for p in pts]
            for t in roots(*v):
                if 0<t<1:ts.append(t)
        for t in ts:
            r=1-t
            q=(r*r*r*pts[0][0]+3*r*r*t*pts[1][0]+
               3*r*t*t*pts[2][0]+t*t*t*pts[3][0],
               r*r*r*pts[0][1]+3*r*r*t*pts[1][1]+
               3*r*t*t*pts[2][1]+t*t*t*pts[3][1])
            if not all(math.isfinite(v) for v in q):
                raise ArithmeticError("SVG evaluated curve overflow")
            bounds[0]=min(bounds[0],q[0]);bounds[1]=min(bounds[1],q[1])
            bounds[2]=max(bounds[2],q[0]);bounds[3]=max(bounds[3],q[1])
    def point(x,y,origin,relative):
        return (x+origin[0],y+origin[1]) if relative else (x,y)
    allowed={"M":2,"L":2,"H":1,"V":1,"C":6,"S":4,"Q":4,"T":2}
    try:
        while index<len(tokens):
            steps+=1
            if steps>20000:return None
            if tokens[index].isalpha():
                cmd=tokens[index]
                index+=1
                if cmd in "Zz":
                    if start is None:return None
                    add_point(cur);add_point(start)
                    cur=start;previous="Z";cmd=None
                    previous_cubic=None;previous_quadratic=None
                    continue
                if cmd.upper() not in allowed:return None
            if cmd is None:return None
            typ=cmd.upper()
            need=allowed[typ]
            if index+need>len(tokens) or any(t.isalpha() for t in tokens[index:index+need]):
                return None
            args=[float(v) for v in tokens[index:index+need]]
            if not all(math.isfinite(v) for v in args):return None
            index+=need
            rel=cmd.islower()
            p0=cur
            if typ=="M":
                cur=point(args[0],args[1],p0,rel)
                start=cur;add_point(cur)
                cmd="l" if rel else "L"
                previous="M";previous_cubic=None;previous_quadratic=None
                continue
            if start is None:return None
            if typ=="L":
                cur=point(args[0],args[1],p0,rel)
                add_point(p0);add_point(cur)
            elif typ=="H":
                cur=(p0[0]+args[0] if rel else args[0],p0[1])
                add_point(p0);add_point(cur)
            elif typ=="V":
                cur=(p0[0],p0[1]+args[0] if rel else args[0])
                add_point(p0);add_point(cur)
            elif typ in ("C","S"):
                if typ=="C":
                    c1=point(args[0],args[1],p0,rel)
                    c2=point(args[2],args[3],p0,rel)
                    cur=point(args[4],args[5],p0,rel)
                else:
                    c1=(2*p0[0]-previous_cubic[0],
                        2*p0[1]-previous_cubic[1]) if (
                            previous in ("C","S") and previous_cubic is not None) else p0
                    c2=point(args[0],args[1],p0,rel)
                    cur=point(args[2],args[3],p0,rel)
                add_curve(p0,c1,c2,cur)
                previous_cubic=c2;previous_quadratic=None
            else:
                if typ=="Q":
                    quad=point(args[0],args[1],p0,rel)
                    cur=point(args[2],args[3],p0,rel)
                else:
                    quad=(2*p0[0]-previous_quadratic[0],
                          2*p0[1]-previous_quadratic[1]) if (
                              previous in ("Q","T") and previous_quadratic is not None) else p0
                    cur=point(args[0],args[1],p0,rel)
                c1=(p0[0]+2*(quad[0]-p0[0])/3,p0[1]+2*(quad[1]-p0[1])/3)
                c2=(cur[0]+2*(quad[0]-cur[0])/3,cur[1]+2*(quad[1]-cur[1])/3)
                add_curve(p0,c1,c2,cur)
                previous_quadratic=quad;previous_cubic=None
            if typ not in ("C","S","Q","T"):
                previous_cubic=None;previous_quadratic=None
            previous=typ
    except (ValueError,OverflowError,ArithmeticError,ZeroDivisionError):
        return None
    return bounds if all(math.isfinite(v) for v in bounds) else None


def _svg_ellipse_bbox(shape, attrs, matrix):
    """Exact support-function bbox for affine circles and ellipses."""
    if matrix is None:return None
    try:
        cx,cy=float(attrs.get("cx","0")),float(attrs.get("cy","0"))
        if shape=="circle":
            rx=ry=float(attrs["r"])
        else:
            rx,ry=float(attrs["rx"]),float(attrs["ry"])
        if rx<=0 or ry<=0 or not all(
                math.isfinite(v) for v in (cx,cy,rx,ry)):
            return None
        center=transform_point(matrix,(cx,cy))
        hw=math.hypot(matrix[0]*rx,matrix[2]*ry)
        hh=math.hypot(matrix[1]*rx,matrix[3]*ry)
        out=[center[0]-hw,center[1]-hh,center[0]+hw,center[1]+hh]
        return out if all(math.isfinite(v) for v in out) else None
    except (KeyError,ValueError,OverflowError):
        return None


def _straight_segment_vertex_pairs(shape, attr, points):
    """Only SVG strokes actually drawn; never bridge distinct M subpaths.

    Pairs index the returned SVG vertex array. In particular, an SVG path
    "M 0 0 L 10 0 M 90 0 L 100 0" has edges (0,1),(2,3):
    the jump from 10 to 90 is NOT a source edge. Unsupported/curved
    commands have no extractable pairs and cannot certify a dimension.
    """
    if points is None:
        return None
    n = len(points)
    if shape == "line":
        return [[0,1]] if n == 2 else None
    if shape in ("polyline","polygon","rect"):
        pairs=[[i,i+1] for i in range(n-1)]
        if shape in ("polygon","rect") and n>=3:
            pairs.append([n-1,0])
        return pairs if len(pairs)<=128 else None
    if shape != "path":
        return None
    tokens=[c if c else num for c,num in _TOKENS.findall(attr.get("d",""))]
    index=0; cursor=0; previous=None; subpath=None; command=None; pairs=[]
    while index < len(tokens):
        if tokens[index].isalpha():
            command=tokens[index];index+=1
            if command not in "MmLlHhVvZz":
                return None
            if command in "Zz":
                if subpath is None or previous is None:
                    return None
                if cursor>=n:
                    return None
                if points[previous]!=points[cursor]:
                    pairs.append([previous,cursor])
                previous=cursor;cursor+=1;command=None
                continue
        if command is None:
            return None
        typ=command.upper()
        need=2 if typ in ("M","L") else 1
        if index+need>len(tokens) or cursor>=n:
            return None
        index+=need
        if typ=="M":
            previous=cursor;subpath=cursor;cursor+=1
            command="l" if command.islower() else "L"
        else:
            if previous is None:
                return None
            if points[previous]!=points[cursor]:
                pairs.append([previous,cursor])
            previous=cursor;cursor+=1
        if len(pairs)>128:
            return None
    return pairs if cursor==n else None


def _bounds(points,matrix):
    if points is None or matrix is None:return None
    moved=[transform_point(matrix,p) for p in points]
    if any(not math.isfinite(v) for point in moved for v in point):
        return None
    return [min(p[0] for p in moved),min(p[1] for p in moved),
            max(p[0] for p in moved),max(p[1] for p in moved)]


def _shape_points(tag,attr):
    if tag=="path":
        return _path_points(attr.get("d",""))
    if tag=="line":
        try:return [(float(attr["x1"]),float(attr["y1"])),
                    (float(attr["x2"]),float(attr["y2"]))]
        except (ValueError,KeyError):return None
    if tag=="rect":
        try:
            x,y=float(attr.get("x","0")),float(attr.get("y","0"))
            w,h=float(attr["width"]),float(attr["height"])
            if w<0 or h<0:return None
            return [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
        except (ValueError,KeyError):return None
    if tag in ("polyline","polygon"):
        values=[float(x.group()) for x in _NUMBER.finditer(attr.get("points",""))]
        if len(values)<4 or len(values)%2:return None
        return list(zip(values[::2],values[1::2]))
    return None


def _svg_inventory(payload,limit_paths=2000,region=None,selected_vector_ids=None,stroke_widths=False,source_repaints=None):
    if selected_vector_ids is not None:
        if (not isinstance(selected_vector_ids,set) or len(selected_vector_ids)>2048
                or any(not isinstance(i,str) or not re.fullmatch(r"v[1-9][0-9]*",i)
                       for i in selected_vector_ids)):
            raise ValueError("selected vector IDs must be <=2048 exact vN identifiers")
    root=ET.fromstring(payload)
    if local(root.tag)!="svg":raise ValueError("SVG converter output lacks root svg")
    dims=[float(t) for t in _NUMBER.findall(root.attrib.get("viewBox",""))]
    if len(dims)!=4 or dims[2]<=0 or dims[3]<=0:
        raise ValueError("SVG viewBox absent or invalid")
    # Only a flattened, opaque monochrome page can supply sidecar marker evidence.
    # Poppler emits the paper background before every painted path.
    children = list(root)
    solid_black = {"black", "#000", "#000000", "rgb(0%, 0%, 0%)"}
    source_paths = set()
    if (source_repaints is not None
            and root.attrib.keys() <= {"width", "height", "viewBox", "version", "preserveAspectRatio"}
            and not any(local(node.tag) == "style" for node in root.iter())):
        source_paths = set(children)
    seen_paint = {}
    path_paint = {"d","fill","fill-rule","stroke","stroke-width","fill-opacity",
                  "stroke-opacity","stroke-linecap","stroke-linejoin","stroke-miterlimit",
                  "transform"}
    flat = bool(children) and all(
        local(node.tag) == "path" and not list(node)
        and node.attrib.keys() <= path_paint
        and node.attrib.get("fill","black") in solid_black | {"none"}
        and node.attrib.get("stroke","none") in solid_black | {"none"}
        and node.attrib.get("fill-opacity","1") == node.attrib.get("stroke-opacity","1") == "1"
        for node in children[1:])
    flat = flat and local(children[0].tag) == "rect" and children[0].get("fill") == "rgb(100%, 100%, 100%)"
    stats=Counter();out=[];unlocalized=0
    stack=[(root,_ID)]
    shape_tags={"path","line","rect","polyline","polygon","circle","ellipse"}
    while stack:
        node,parent=stack.pop()
        name=local(node.tag)
        if name=="defs":continue  # Glyph outline definitions are not drawing vectors.
        own=parse_transform(node.attrib.get("transform"))
        matrix=mmul(parent,own) if parent is not None and own is not None else None
        if name=="use":stats["text_glyph_uses"]+=1
        if name=="image":stats["embedded_images"]+=1
        if name in shape_tags:
            stats["vector_primitives"]+=1
            points=_shape_points(name,node.attrib)
            box=_bounds(points,matrix)
            if box is None and name=="path":
                box=_svg_curve_bbox(node.attrib.get("d",""),matrix)
                if box is not None:stats["bezier_localized_paths"]+=1
            elif box is None and name in ("circle","ellipse"):
                box=_svg_ellipse_bbox(name,node.attrib,matrix)
                if box is not None:stats["ellipse_localized_paths"]+=1
            if box is None:unlocalized+=1
            retained=(region is None or box is None or
                (box[0]<=region[2] and box[2]>=region[0] and
                 box[1]<=region[3] and box[3]>=region[1]))
            selected = (selected_vector_ids is None or
                        f"v{stats['vector_primitives']}" in selected_vector_ids)
            if (name == "path" and node in source_paths and not len(node)
                    and node.attrib.keys() <= path_paint and node.get("fill") == "none"
                    and node.get("stroke") in solid_black and node.get("stroke-opacity","1") == "1"):
                # An opaque stroke painted twice at the exact same location is one
                # printed mark. Only direct siblings share this full paint context.
                paint = tuple(sorted(node.attrib.items()))
                if paint in seen_paint:
                    source_repaints.add(seen_paint[paint])
                    selected = False
                else:
                    seen_paint[paint] = f"v{stats['vector_primitives']}"
            if selected and retained and len(out)<limit_paths:
                vertices = ([transform_point(matrix, p) for p in points]
                            if points is not None and matrix is not None
                            and len(points) <= 64 else None)
                if vertices is not None and any(
                        not math.isfinite(q) for point in vertices for q in point):
                    vertices = None
                row={"id":f"v{stats['vector_primitives']}",
                     "type":name,"svg_bbox":box,
                     "bbox_exact_for_centerline":box is not None,
                     "svg_points":[list(p) for p in vertices] if vertices else None,
                     "svg_segment_vertex_pairs":(
                         _straight_segment_vertex_pairs(name,node.attrib,points)
                         if vertices is not None else None)}
                if name=="path":row["d"]=node.attrib.get("d","")[:500]
                if node.attrib.get("stroke"):row["stroke"]=node.attrib["stroke"][:80]
                if node.attrib.get("stroke-width"):
                    row["stroke_width"]=node.attrib["stroke-width"][:40]
                    if stroke_widths and matrix is not None and re.fullmatch(_NUMBER,row["stroke_width"]):
                        # A circular pen becomes an ellipse under an affine transform.
                        # Its maximum diameter is the largest singular value times the pen width.
                        a,b,c,d = matrix[:4]
                        trace = a*a+b*b+c*c+d*d
                        discriminant = max(0, trace*trace-4*(a*d-b*c)**2)
                        row["svg_pen_diameter"] = float(row["stroke_width"])*math.sqrt((trace+math.sqrt(discriminant))/2)

                out.append(row)
        stack.extend((child,matrix) for child in reversed(list(node)))
    return {"flat_opaque_marker_page":flat,"viewbox_svg_units":dims,"vector_primitives":stats["vector_primitives"],
            "reported_vectors":out,"unlocalized_vectors":unlocalized,
            "embedded_images":stats["embedded_images"],
            "text_glyph_instances":stats["text_glyph_uses"],
            "exact_bezier_centerline_bboxes":stats["bezier_localized_paths"],
            "exact_ellipse_centerline_bboxes":stats["ellipse_localized_paths"],
            "reported_count":len(out),
            "vector_limit_hit":(stats["vector_primitives"]>limit_paths
                                if selected_vector_ids is None else
                                len(out)<len(selected_vector_ids)),
            "selected_vector_ids":len(selected_vector_ids) if selected_vector_ids is not None else None,
            "region_svg_units":region}



def _poppler_text_xml(payload):
    """Keep an unreadable PDF text line out of reference matching."""
    try:
        return ET.fromstring(payload)
    except ET.ParseError:
        # Valid UTF-16, declared legacy encodings and callers supplying Unicode
        # keep ElementTree's original input semantics.
        raw = not isinstance(payload, str)
        illegal = r"[\x00-\x08\x0b\x0c\x0e-\x1f]"
        word = r"(<word(?:\s[^<>]*)?>)([^<>]*)(</word>)"
        if raw:
            illegal, word = illegal.encode("ascii"), word.encode("ascii")
        if not re.search(illegal, payload):
            raise
        marker = b"&#xFFFD;" if raw else "&#xFFFD;"

        def quarantine(match):
            if re.search(illegal, match.group(2)):
                return match.group(1) + marker + match.group(3)
            return match.group()

        # Word contents alone are eligible. Invalid markup remains a parse error.
        root = ET.fromstring(re.sub(word, quarantine, payload))
    for line in root.findall(".//{*}line"):
        words = line.findall("{*}word")
        if any(item.text and "\ufffd" in item.text for item in words):
            for item in words:
                item.text = "\ufffd"
    return root


def _word_inventory(data,max_words):
    root=_poppler_text_xml(data)
    words=[];page_box=None
    lines = (element for element in root.iter() if local(element.tag) == "line")
    source_lines = {word: index for index, line in enumerate(lines)
                    for word in line if local(word.tag) == "word"}
    for element in root.iter():
        name=local(element.tag)
        if name=="page" and page_box is None:
            try:page_box=[float(element.attrib["width"]),float(element.attrib["height"])]
            except (ValueError,KeyError):pass
        if name=="word":
            try:b=[float(element.attrib[k]) for k in ("xMin","yMin","xMax","yMax")]
            except (ValueError,KeyError):continue
            row = {"text":(element.text or "")[:160], "page_bbox_pt":b}
            if element in source_lines:
                row["source_line_id"] = source_lines[element]
            words.append(row)
            if len(words)>=max_words:break
    return words,page_box


def _scales(words):
    lines=[]
    for word in sorted(words,key=lambda w:(w["page_bbox_pt"][1],w["page_bbox_pt"][0])):
        b=word["page_bbox_pt"];cy=(b[1]+b[3])/2
        target=next((line for line in lines if abs(line["cy"]-cy)<2.5),None)
        if target is None:
            target={"cy":cy,"words":[]};lines.append(target)
        target["words"].append(word)
    found=[]
    for line in lines:
        items=sorted(line["words"],key=lambda w:w["page_bbox_pt"][0])
        text=" ".join(w["text"] for w in items)
        if not re.search(r"\bSCALE\b|\bNTS\b|\b1\s*[:/]\s*\d{1,5}\b",text,re.I):
            continue
        xs=[w["page_bbox_pt"] for w in items]
        found.append({"text":text[:200],
                      "page_bbox_pt":[min(b[0] for b in xs),min(b[1] for b in xs),
                                      max(b[2] for b in xs),max(b[3] for b in xs)],
                      "source_kind":"unverified_printed_annotation"})
    return found[:80]


def _page_geometry(path, first, last):
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        raise RuntimeError("pdfinfo required (Poppler)")
    info = subprocess.run([pdfinfo, "-f", str(first), "-l", str(last), str(path)], capture_output=True,
                          timeout=10, check=False)
    if info.returncode != 0:
        if b"Wrong page range" in info.stderr:
            raise ValueError("page range outside PDF")
        raise ValueError("pdfinfo failed: encrypted/invalid PDF")
    match = re.search(rb"^Pages:\s*(\d+)\s*$", info.stdout, re.M)
    if not match:
        raise ValueError("PDF has no readable page count")
    rotations = {int(page): int(angle) % 360 for page, angle in re.findall(
        rb"^Page\s+(\d+) rot:\s*(-?\d+)\s*$", info.stdout, re.M)}
    return int(match.group(1)), rotations


def _page_svg(path, page, directory):
    """One canonical CropBox frame, with page rotation and no paper fitting.

    Cairo's default vector printing may shrink/center fractional page sizes.
    Explicit no-fit settings keep the SVG coordinates aligned with a CropBox
    raster at dpi/72. Native word boxes use the same rotated CropBox.
    """
    renderer = shutil.which("pdftocairo")
    if not renderer:
        raise RuntimeError("pdftocairo required (Poppler)")
    output = pathlib.Path(directory) / f"page_{page}.svg"
    result = subprocess.run(
        [renderer, "-svg", "-origpagesizes", "-nocenter", "-noshrink",
         "-f", str(page), "-l", str(page), str(path), str(output)],
        capture_output=True, timeout=30, check=False)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace")[:140]
        raise ValueError(f"page {page}: SVG conversion failed: {message}")
    candidates = sorted(pathlib.Path(directory).glob(f"page_{page}*.svg"))
    if len(candidates) != 1:
        raise ValueError(f"page {page}: SVG produced missing/ambiguous page files")
    svg = candidates[0]
    if svg.stat().st_size > 15_000_000:
        raise ValueError(f"page {page}: vector SVG exceeds 15 MB")
    return svg.read_bytes()


def _crop_native_png(source, destination, full_size, pixel_box):
    """Copy exact pixels using libpng's public, versioned simplified C API.

    Poppler already links libpng. Its native decoder keeps this path independent
    of the Python test oracle and avoids handwritten PNG filtering or Pillow.
    ABI: pnggroup/libpng v1.6.43 png.h, PNG_IMAGE_VERSION and simplified APIs.
    """
    import ctypes
    import ctypes.util

    class PngImage(ctypes.Structure):
        _fields_ = [("opaque", ctypes.c_void_p), ("version", ctypes.c_uint32),
                    ("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                    ("format", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                    ("colormap_entries", ctypes.c_uint32),
                    ("warning_or_error", ctypes.c_uint32), ("message", ctypes.c_char * 64)]

    library = ctypes.util.find_library("png16")
    if not library:
        raise RuntimeError("libpng16 required for exact raster windows")
    codec = ctypes.CDLL(library)
    pointer = ctypes.POINTER(PngImage)
    try:
        codec.png_image_begin_read_from_file.argtypes = [pointer, ctypes.c_char_p]
        codec.png_image_begin_read_from_file.restype = ctypes.c_int
        codec.png_image_finish_read.argtypes = [pointer, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_int32, ctypes.c_void_p]
        codec.png_image_finish_read.restype = ctypes.c_int
        codec.png_image_write_to_file.argtypes = [pointer, ctypes.c_char_p, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
        codec.png_image_write_to_file.restype = ctypes.c_int
        codec.png_image_free.argtypes = [pointer]
        codec.png_image_free.restype = None
    except AttributeError as exc:
        raise RuntimeError("libpng16 simplified image API unavailable") from exc
    image, cropped = PngImage(), PngImage()
    image.version = cropped.version = 1
    left, top, right, bottom = pixel_box
    try:
        if not codec.png_image_begin_read_from_file(ctypes.byref(image), os.fsencode(source)):
            raise ValueError("PNG decoding failed: " + image.message.decode(errors="replace"))
        if (image.width, image.height) != tuple(full_size):
            raise ValueError("native full-page PNG dimensions disagree with source coordinates")
        if image.width * image.height > 32_000_000:
            raise ValueError("full-page raster exceeds 32 million pixel bound")
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError("pixel crop outside native full-page raster")
        # PNG_FORMAT_RGBA = COLOR (0x02) | ALPHA (0x01). Keep every channel;
        # the opaque Cairo background is not replaced or recomposited.
        image.format = 3
        stride = image.width * 4
        buffer = (ctypes.c_ubyte * (stride * image.height))()
        if not codec.png_image_finish_read(ctypes.byref(image), None, buffer, stride, None):
            raise ValueError("PNG decoding failed: " + image.message.decode(errors="replace"))
        cropped.width, cropped.height, cropped.format = right - left, bottom - top, 3
        start = ctypes.byref(buffer, top * stride + left * 4)
        if not codec.png_image_write_to_file(ctypes.byref(cropped), os.fsencode(destination),
                                             0, start, stride, None):
            raise ValueError("PNG encoding failed: " + cropped.message.decode(errors="replace"))
    finally:
        codec.png_image_free(ctypes.byref(image))
        codec.png_image_free(ctypes.byref(cropped))


def scan_pdf(filename,first=1,last=None,limit_paths=2000,
             limit_words=3000,region=None,selected_vector_ids=None):
    """Read-only exact native PDF pages; never infer real-world drawing units."""
    path=pathlib.Path(filename)
    if path.stat().st_size>50_000_000:
        raise ValueError("PDF exceeds 50 MB task-container inspection bound")
    pdfinfo,pdftotext,pdftocairo=[shutil.which(tool) for tool in
                                  ("pdfinfo","pdftotext","pdftocairo")]
    if not all((pdfinfo,pdftotext,pdftocairo)):
        raise RuntimeError("pdfinfo, pdftotext, pdftocairo required (Poppler); use document MCP")
    if not 1<=first<=10000 or (last is not None and not first<=last<=10000):
        raise ValueError("invalid PDF page bounds")
    if not 1<=limit_paths<=5000 or not 1<=limit_words<=5000:
        raise ValueError("limits must be in [1,5000]")
    if region is not None:
        if (len(region)!=4 or any(not math.isfinite(v) for v in region) or
            region[0]>=region[2] or region[1]>=region[3]):
            raise ValueError("region requires finite SVG xMin yMin xMax yMax")
    end=last if last is not None else first
    if end-first+1>20:
        raise ValueError("page range exceeds 20-sheet bound")
    total, rotations = _page_geometry(path, first, end)
    if end>total:
        raise ValueError("page range outside PDF or exceeds 20-sheet bound")
    pages=[]
    with tempfile.TemporaryDirectory(prefix="hamon-pdf-") as tmp:
        for page in range(first,end+1):
            text=subprocess.run([pdftotext,"-f",str(page),"-l",str(page),
                                 "-cropbox","-bbox-layout",str(path),"-"],
                                capture_output=True,timeout=20,check=False)
            if text.returncode!=0 or len(text.stdout)>8_000_000:
                raise ValueError(f"page {page}: PDF word extraction failed/budget exceeded")
            words,box=_word_inventory(text.stdout,limit_words)
            # Poppler rotates the glyph boxes but reports the unrotated CropBox size.
            rotation = rotations.get(page)
            if rotation not in (0, 90, 180, 270):
                raise ValueError(f"page {page}: missing or invalid PDF rotation")
            if rotation in (90, 270):
                box = box[::-1]
            inventory=_svg_inventory(_page_svg(path,page,tmp),limit_paths,region,selected_vector_ids)
            pages.append({"page":page,"pdf_page_size_pt":box,"page_rotation_degrees":rotation,
                          "svg_coordinate_frame":_SVG_FRAME,
                          "word_boxes_reported":len(words),
                          "words":words,"scale_annotations":_scales(words),
                          **inventory})
    return {"kind":"pdf-multisheet-vector-inventory",
            "file":str(path),"document_pages":total,
            "first_page":first,"last_page":end,"pages":pages,
            "physical_drawing_scale_verified":False,
            "warning":"Native PDF word boxes and SVG vector centerlines use the rotated CropBox without paper fitting. Printed scale text is annotation only. SVG M/L/H/V/C/S/Q/T/Z curve centerline bboxes are exact under affine transforms; unsupported arc paths remain unlocalized, crop is exploratory, and no OCR, plotted geometric calibration, or code compliance is inferred."}


def render_window(filename, output, page=1, region=None, dpi=120, limit_paths=128):
    """Render a bounded PNG and return its exact pixel-to-SVG provenance.

    The supplied region uses the same rotated CropBox frame as ``scan_pdf``.
    Its edges are rounded outward to raster pixels; the returned inverse
    transform describes those actual pixel edges. A private source snapshot
    binds the vectors and image to one digest. Existing files are never replaced.
    """
    source, destination = pathlib.Path(filename), pathlib.Path(output)
    if type(page) is not int or not 1 <= page <= 10000:
        raise ValueError("invalid PDF page bounds")
    if not isinstance(dpi, (int, float)) or not math.isfinite(dpi) or not 36 <= dpi <= 600:
        raise ValueError("dpi must be finite and in [36,600]")
    if type(limit_paths) is not int or not 1 <= limit_paths <= 5000:
        raise ValueError("limits must be in [1,5000]")
    if region is not None:
        if (len(region) != 4 or any(not math.isfinite(v) for v in region)
                or region[0] >= region[2] or region[1] >= region[3]):
            raise ValueError("region requires finite SVG xMin yMin xMax yMax")
        region = list(region)
    if destination.suffix.lower() != ".png":
        raise ValueError("output must be a new .png file")
    if os.path.lexists(destination) or destination.resolve() == source.resolve():
        raise ValueError("output already exists or aliases the source; choose a new PNG path")
    if not destination.parent.is_dir():
        raise ValueError("output parent directory does not exist")
    if source.stat().st_size > 50_000_000:
        raise ValueError("PDF exceeds 50 MB task-container inspection bound")
    renderer = shutil.which("pdftocairo")
    if not renderer:
        raise RuntimeError("pdftocairo required (Poppler)")

    with tempfile.TemporaryDirectory(prefix="hamon-pdf-window-") as temporary:
        root = pathlib.Path(temporary)
        snapshot = root / "source.pdf"
        digest, source_size = hashlib.sha256(), 0
        with source.open("rb") as incoming, snapshot.open("xb") as copied:
            while chunk := incoming.read(131072):
                source_size += len(chunk)
                if source_size > 50_000_000:
                    raise ValueError("PDF exceeds 50 MB task-container inspection bound")
                digest.update(chunk)
                copied.write(chunk)
        total, _ = _page_geometry(snapshot, page, page)
        if page > total:
            raise ValueError("page range outside PDF")
        svg = _page_svg(snapshot, page, root)
        inventory = _svg_inventory(svg, limit_paths, region)
        viewbox = inventory["viewbox_svg_units"]
        if (viewbox is None or len(viewbox) != 4
                or any(not math.isfinite(v) for v in viewbox)
                or viewbox[:2] != [0.0, 0.0]
                or viewbox[2] <= 0 or viewbox[3] <= 0):
            raise ValueError("unsupported SVG page coordinate frame")
        width_pt, height_pt = viewbox[2:]
        requested = region if region is not None else [0.0, 0.0, width_pt, height_pt]
        if (requested[0] < 0 or requested[1] < 0
                or requested[2] > width_pt or requested[3] > height_pt):
            raise ValueError("region outside the rotated PDF CropBox")
        scale = dpi / 72
        full_width, full_height = math.ceil(width_pt * scale), math.ceil(height_pt * scale)
        if max(full_width, full_height) > 32767:
            raise ValueError("page raster axis exceeds 32767 pixels; choose lower dpi")
        if full_width * full_height > 32_000_000:
            raise ValueError("full-page raster exceeds 32 million pixels; choose lower dpi")
        left, top = math.floor(requested[0] * scale), math.floor(requested[1] * scale)
        right, bottom = math.ceil(requested[2] * scale), math.ceil(requested[3] * scale)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0 or max(width, height) > 8192 or width * height > 4_000_000:
            raise ValueError("window exceeds 4 million pixels or 8192-pixel side; choose a smaller region or lower dpi")
        prefix = root / "window"
        rendered = subprocess.run(
            [renderer, "-png", "-cropbox", "-singlefile", "-f", str(page), "-l", str(page),
             "-r", str(dpi), str(snapshot), str(prefix)],
            capture_output=True, timeout=30, check=False)
        if rendered.returncode != 0:
            message = rendered.stderr.decode(errors="replace")[:140]
            raise ValueError(f"page {page}: PNG conversion failed: {message}")
        png = prefix.with_suffix(".png")
        if png.stat().st_size > 128_000_000:
            raise ValueError("full-page PNG exceeds 128 MB intermediate bound")
        with png.open("rb") as native:
            header = native.read(33)
        if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[12:16] != b"IHDR"
                or struct.unpack(">II", header[16:24]) != (full_width, full_height)):
            raise ValueError("native full-page PNG dimensions disagree with source coordinates")
        if (left, top, right, bottom) != (0, 0, full_width, full_height):
            cropped_png = root / "cropped.png"
            _crop_native_png(png, cropped_png, (full_width, full_height), (left, top, right, bottom))
            png = cropped_png
        if png.stat().st_size > 8_000_000:
            raise ValueError("window PNG exceeds 8 MB; choose a smaller region or lower dpi")
        pixels = png.read_bytes()
        if (len(pixels) < 33 or pixels[:8] != b"\x89PNG\r\n\x1a\n"
                or pixels[12:16] != b"IHDR" or struct.unpack(">II", pixels[16:24]) != (width, height)):
            raise ValueError("native PNG dimensions disagree with the declared coordinate transform")

        # A same-directory temporary plus an exclusive hard link is atomic and
        # cannot truncate an existing output, including a concurrent writer.
        pending = None
        try:
            with tempfile.NamedTemporaryFile(prefix="hamon-window-", suffix=".png",
                                             dir=destination.parent, delete=False) as staged:
                pending = pathlib.Path(staged.name)
                staged.write(pixels)
                staged.flush()
                os.fsync(staged.fileno())
            os.link(pending, destination)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)

    return {
        "kind": "pdf-coordinate-window", "schema": "hamon.pdf.window.v1",
        "file": str(source), "source_pdf_sha256": digest.hexdigest(),
        "source_pdf_bytes": source_size, "document_pages": total, "page": page,
        "svg_sha256": hashlib.sha256(svg).hexdigest(),
        "image": {"path": str(destination), "mime_type": "image/png",
                  "sha256": hashlib.sha256(pixels).hexdigest(), "bytes": len(pixels),
                  "width": width, "height": height},
        "coordinate_frame": {
            "name": _SVG_FRAME, "viewbox_svg_units": viewbox,
            "requested_region_svg": requested,
            "covered_region_svg": [left * 72 / dpi, top * 72 / dpi,
                                   right * 72 / dpi, bottom * 72 / dpi],
            "dpi": dpi, "pixel_box_on_full_raster": [left, top, right, bottom],
            "full_raster_size_pixels": [full_width, full_height],
            "raster_selection": "exact native full-page pixels; no local re-render",
            "svg_to_pixel": [scale, 0, 0, scale, -left, -top],
            "pixel_to_svg": [72 / dpi, 0, 0, 72 / dpi, left * 72 / dpi, top * 72 / dpi],
            "pixel_convention": "integer pixel edges; centers at (x+0.5,y+0.5)",
        },
        "vector_inventory": inventory, "ocr_performed": False,
        "physical_drawing_scale_verified": False,
        "warning": "Read the emitted PNG to inspect the actual marks. Pixel-to-SVG mapping is page geometry, not physical calibration. This raster window does not establish a dimension label, OCR result, engineering measurement or code approval.",
    }


# Explicit detail references and headings use different namespaces. An unlabelled
# number on a target sheet, including a grid mark, is never a detail definition.
_SHEET_ID = r"[A-Z]{1,4}[-.]?\d{1,4}(?:[.-]\d{1,3})?[A-Z]?"
_DETAIL_ID = r"\d{1,3}[A-Z]?"
_DETAIL_LIST = rf"{_DETAIL_ID}(?:\s*(?:&|,|AND)\s*{_DETAIL_ID})*"
_SHEET_LABEL = re.compile(rf"SHEET\s+(?:NO\.?|NUMBER|ID|#)\s*:?\s*({_SHEET_ID})?", re.I)
_DETAIL_REF = re.compile(
    rf"\bDETAILS?\s+({_DETAIL_LIST})\s*(?:/|(?:ON\s+)?SHEET\s+)\s*({_SHEET_ID})(?!\w|[.\-]\w)", re.I)
_BUBBLE_TEXT = re.compile(rf"({_DETAIL_ID})\s*/\s*({_SHEET_ID})", re.I)
_DETAIL_HEADING = re.compile(rf"DETAIL\s+({_DETAIL_ID})\s*[:–—-]\s*(\S.+)", re.I)


def _drawing_text_lines(page, page_no):
    rows = []
    for line in page.findall(".//{*}line"):
        text = " ".join(word.text or "" for word in line.findall("{*}word")).strip()
        if "\ufffd" in text:
            text = "\ufffd"
        box = [float(line.attrib[k]) for k in ("xMin", "yMin", "xMax", "yMax")]
        if not all(math.isfinite(v) for v in box) or box[0] >= box[2] or box[1] >= box[3]:
            continue
        if text:
            rows.append({"page": page_no, "line": len(rows) + 1, "text": text, "box": box})
    return rows


def _sheet_stamps(rows):
    """Accept an explicit inline label or a nearby standalone printed identifier."""
    ids = [row for row in rows if re.fullmatch(_SHEET_ID, row["text"], re.I)]
    stamps = {}
    for row in rows:
        label = _SHEET_LABEL.fullmatch(row["text"])
        if not label:
            continue
        if label[1]:
            stamps.setdefault(label[1].upper(), []).append([row])
            continue
        x0, y0, x1, y1 = row["box"]
        height = y1 - y0
        for value in ids:
            left, top, right, bottom = value["box"]
            if (y0 - height / 2 <= top <= y1 + 2 * height and
                    x0 - height <= left <= x1 + 5 * height and bottom >= y0):
                stamps.setdefault(value["text"].upper(), []).append([row, value])
    return stamps


def _stacked_detail_candidates(rows, stamps):
    stamp_lines = {r["line"] for witnesses in stamps.values() for pair in witnesses for r in pair}
    numbers = sorted((((r["box"][0] + r["box"][2]) / 2, r["line"], r) for r in rows
                      if re.fullmatch(_DETAIL_ID, r["text"], re.I)))
    centers = [n[0] for n in numbers]
    candidates, numerator_uses = [], Counter()
    for row in rows:
        if row["line"] in stamp_lines or not re.fullmatch(_SHEET_ID, row["text"], re.I):
            continue
        left, top, right, bottom = row["box"]
        h = bottom - top
        center = (left + right) / 2
        choices = []
        for _, _, number in numbers[bisect_left(centers, center - .35*h):bisect_right(centers, center + .35*h)]:
            x0, y0, x1, y1 = number["box"]
            if .75*h <= y1-y0 <= 1.3*h and -.1*h <= top-y1 <= .8*h:
                choices.append(number)
                numerator_uses[number["line"]] += 1
        if choices:
            candidates.append((row, choices))
    for row, choices in candidates:
        unique = len(choices) == 1 and numerator_uses[choices[0]["line"]] == 1
        yield {"source": row, "source_sheet_ids": sorted(stamps),
               "reference_form": "stacked_text_candidate", "numerator_evidence": choices,
               "target_sheet": row["text"].upper(),
               "target_detail": choices[0]["text"].upper() if unique else None}


def _detail_graph(pages):
    sheet_pages = {}
    headings = {}
    requested = []
    for number, rows in enumerate(pages, 1):
        stamps = _sheet_stamps(rows)
        for sheet, evidence in stamps.items():
            sheet_pages.setdefault(sheet, []).append(
                {"page": number, "sheet_ids_on_page": sorted(stamps), "stamp_evidence": evidence})
        for row in rows:
            line = row["text"]
            references = list(_DETAIL_REF.finditer(line))
            bubble = _BUBBLE_TEXT.fullmatch(line)
            if bubble:
                references.append(bubble)
            for ref in references:
                for detail in re.findall(_DETAIL_ID, ref[1], re.I):
                    requested.append({"source": row, "source_sheet_ids": sorted(stamps),
                                      "reference_form": "explicit_text",
                                      "reference_span": list(ref.span()), "target_sheet": ref[2].upper(),
                                      "target_detail": detail.upper()})
            heading = _DETAIL_HEADING.fullmatch(line)
            if heading and not references:
                headings.setdefault((number, heading[1].upper()), []).append(row)
        requested.extend(_stacked_detail_candidates(rows, stamps))

    for ref in requested:
        destinations = sheet_pages.get(ref["target_sheet"], [])
        ref["candidate_pages"] = [d["page"] for d in destinations]
        if ref["target_detail"] is None:
            ref["status"] = "ambiguous_stacked_source"
        elif not destinations:
            ref["status"] = "sheet_not_found"
        elif len(destinations) != 1 or len(destinations[0]["sheet_ids_on_page"]) != 1:
            ref["status"] = "ambiguous_sheet"
        else:
            candidates = headings.get((destinations[0]["page"], ref["target_detail"]), [])
            ref["detail_evidence"] = candidates
            ref["status"] = ("resolved_detail" if len(candidates) == 1 else
                             "ambiguous_detail" if candidates else "sheet_only")
            if ref["status"] == "resolved_detail" and ref["reference_form"] == "stacked_text_candidate":
                ref["status"] = "candidate_detail"
    return {"sheets": sheet_pages, "references": requested}


def _numbered_caption_candidates(rows, detail):
    """Find a separate printed number and heading positioned for a ruled caption."""
    for number in rows:
        if number["text"].upper() != detail:
            continue
        x0, y0, x1, y1 = number["box"]
        height = y1 - y0
        for title in rows:
            left, top, right, bottom = title["box"]
            if (0 < left-x1 < 3*height and -.15*height < bottom-y0 < .65*height
                    and .8*height < bottom-top < 1.2*height and right-left > 2*height
                    and title["text"].isupper() and len(title["text"].split()) >= 2):
                yield number, title


def _round_marker(vector):
    if vector["type"] in ("circle", "ellipse"):
        return True
    # Poppler commonly emits a circular view label as four cubic quadrants.
    path = vector.get("d", "")
    if not re.fullmatch(r"MCCCCZ?", "".join(re.findall(r"[MmLlHhVvCcSsQqTtAaZz]", path))):
        return False
    coords = [float(x) for x in _NUMBER.findall(path)]
    if len(coords) != 26:
        return False
    ends = [tuple(coords[i:i+2]) for i in (0, 6, 12, 18, 24)]
    cx = (ends[0][0] + ends[2][0]) / 2
    cy = (ends[0][1] + ends[2][1]) / 2
    tol = .015 * max(math.dist(ends[0], ends[2]), math.dist(ends[1], ends[3]))
    if not tol or math.dist(ends[0], ends[4]) > tol:
        return False
    axis = [abs(x-cx) < tol for x, y in ends[:4]]
    if axis not in ([True, False, True, False], [False, True, False, True]):
        return False
    # Elliptic quarter-arc cubic controls lie on the endpoint tangents.
    for i, vertical in enumerate(axis):
        x, y = ends[i]
        nx, ny = ends[i+1]
        dx, dy = nx-x, ny-y
        if abs(dx) < tol or abs(dy) < tol or (abs(y-cy) < tol) == vertical:
            return False
        c1, c2 = coords[2+6*i:4+6*i], coords[4+6*i:6+6*i]
        k = .55228475
        expected1 = (x+k*dx, y) if vertical else (x, y+k*dy)
        expected2 = (nx, ny-k*dy) if vertical else (nx-k*dx, ny)
        if math.dist(c1, expected1) > tol or math.dist(c2, expected2) > tol:
            return False
    return True


def _caption_geometry(vectors, number, title):
    x0, y0, x1, y1 = number["box"]
    height = y1-y0
    left, top, right, bottom = title["box"]
    markers, rules = [], []
    for vector in vectors:
        if vector.get("stroke") in (None, "none") or not vector["bbox_exact_for_centerline"]:
            continue
        a, b, c, d = vector["svg_bbox"]
        if (a < x0 < x1 < c and b < y0 < y1 < d and 1.25*height < c-a < 2.5*height
                and abs(c-a-(d-b)) < .1*height and abs(a+c-x0-x1) < .4*height
                and abs(b+d-y0-y1) < .5*height and _round_marker(vector)):
            markers.append(vector)
        if (vector.get("svg_segment_vertex_pairs") == [[0, 1]] and abs(b-d) < .05*height
                and bottom-.15*height <= b <= bottom+.3*height and c >= right and a < left):
            rules.append(vector)
    pairs = [(m, r) for m in markers for r in rules
             if abs(m["svg_bbox"][2]-r["svg_bbox"][0]) < .2*height
             and abs((m["svg_bbox"][1]+m["svg_bbox"][3])/2-r["svg_bbox"][1]) < .2*height]
    if len(pairs) != 1:
        return None
    return {"number": number, "caption": title,
            "vectors": [{"id": v["id"], "box": v["svg_bbox"]} for v in pairs[0]]}


def _resolve_ruled_captions(result, pages, target, source, directory):
    destinations = result["sheets"].get(target, [])
    if len(destinations) != 1 or destinations[0]["sheet_ids_on_page"] != [target]:
        return
    page = destinations[0]["page"]
    refs = [r for r in result["references"] if r["target_sheet"] == target
            and r["target_detail"] is not None]
    if not refs:
        return
    vectors = None
    for detail in sorted({r["target_detail"] for r in refs}):
        candidates = list(_numbered_caption_candidates(pages[page-1], detail))
        if not candidates:
            continue
        if vectors is None:
            inventory = _svg_inventory(_page_svg(source, page, directory), 30000)
            if inventory["vector_limit_hit"] or inventory["unlocalized_vectors"]:
                return
            vectors = inventory["reported_vectors"]
        matches = [proof for n, t in candidates if (proof := _caption_geometry(vectors, n, t))]
        if not matches:
            continue
        for ref in (r for r in refs if r["target_detail"] == detail):
            ref["ruled_caption_evidence"] = matches
            if len(matches) != 1 or ref.get("detail_evidence"):
                ref["status"] = "ambiguous_detail"
            else:
                ref["status"] = ("resolved_detail" if ref["reference_form"] == "explicit_text"
                                 else "candidate_detail")


def _detail_symbol_inventory(source, page, directory, stroke_vectors=None, *, source_repaints=None):
    try:
        inventory = _svg_inventory(_page_svg(source, page, directory), 30000,
                                   stroke_widths=stroke_vectors is not None,
                                   source_repaints=source_repaints)
    except (OSError, ValueError, subprocess.TimeoutExpired, ET.ParseError):
        return [], []
    if inventory["vector_limit_hit"] or inventory["unlocalized_vectors"]:
        return [], []
    markers, horizontal = [], []
    for v in inventory["reported_vectors"]:
        if v.get("stroke") in (None, "none") or not v["bbox_exact_for_centerline"]:
            continue
        a, b, c, d = v["svg_bbox"]
        if stroke_vectors is not None and 0 <= c-a < 80 and 0 <= d-b < 80 and (c > a or d > b):
            stroke_vectors.append(v)
        if 5 <= c-a <= 80 and abs(c-a-(d-b)) < .03*(c-a) and _round_marker(v):
            markers.append({"id": v["id"], "box": [a,b,c,d]})
        points = v.get("svg_points") or []
        for i,j in v.get("svg_segment_vertex_pairs") or []:
            x,y = points[i]
            xx,yy = points[j]
            if abs(y-yy) < .2 and abs(xx-x) > 4:
                horizontal.append((min(x,xx), max(x,xx), (y+yy)/2, v["id"]))
    return markers, horizontal


def _divided_source_mark(ref, markers, horizontal):
    numerators = ref.get("numerator_evidence", [])
    if len(numerators) != 1 or ref.get("target_detail") != numerators[0]["text"].upper():
        return None
    n = numerators[0]["box"]
    s = ref["source"]["box"]
    matches = []
    for m in markers:
        a,b,c,d = m["box"]
        x,y,r = (a+c)/2, (b+d)/2, (c-a)/2
        # Font ascent and descent boxes can overlap a divider even when their
        # printed ink does not. Bind the text centres to opposite semicircles.
        ny, sy = (n[1]+n[3])/2, (s[1]+s[3])/2
        if not (b < ny <= y-.18*r and y+.18*r <= sy < d):
            continue
        if any(box[1] < b-.15*r or box[3] > d+.15*r
               or abs((box[0]+box[2])/2-x) >= .2*r
               or any(((xx-x)/r)**2+((cy-y)/r)**2 >= 1.0 for xx in (box[0],box[2]))
               for box,cy in ((n,ny),(s,sy))):
            continue
        rules = [(lo, hi, cy, v) for lo, hi, cy, v in horizontal
                 if abs(cy-y) < .08*r and lo <= a+.12*r and hi >= c-.12*r]
        if len(rules) == 1:
            matches.append({"ring":m, "divider":rules[0][3], "numerator_line":numerators[0]["line"]})
        elif len(rules) > 1 and len(set(rule[:3] for rule in rules)) == len(rules):
            # Multiple vector segments are provisional until the raster establishes
            # that they occupy one printed band across this particular ring.
            matches.append({"ring": m, "divider_segments": rules,
                            "numerator_line": numerators[0]["line"]})
    return matches[0] if len(matches)==1 else None


def _ruled_detail_regions(markers, horizontal):
    regions=[]
    for marker in markers:
        a,b,c,d = marker["box"]
        mid=(b+d)/2
        diameter=c-a
        rules=[(right,y,uid) for left,right,y,uid in horizontal
               if abs(left-c)<.10*diameter and abs(y-mid)<.08*diameter
               and 3*diameter < right-c < 30*diameter]
        if len(rules)==1:
            right,y,uid=rules[0]
            regions.append({"ring":marker,"rule":uid,
                            "caption_box":[c, y-.92*diameter, right, y-.045*diameter]})
    return regions


def _caption_raster(source, page, box, scale, erase_ring=False, pixel_limit=2500000, *, hide_annotations=False):
    a,b,c,d=box
    left,top=math.floor(a*scale),math.floor(b*scale)
    width,height=math.ceil(c*scale)-left, math.ceil(d*scale)-top
    if min(left,top)<0 or min(width,height)<1 or width*height>pixel_limit:
        return None
    annotation_flags = ["-hide-annotations"] if hide_annotations else []
    call=subprocess.run(["pdftoppm",*annotation_flags,"-cropbox","-r",str(72*scale),"-f",str(page),"-l",str(page),
                         "-singlefile","-x",str(left),"-y",str(top),"-W",str(width),"-H",str(height),
                         "-gray",str(source)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    if call.returncode:
        return None
    header=re.match(rb"P5\s+(\d+)\s+(\d+)\s+255\s",call.stdout)
    if not header or tuple(map(int,header.groups()))!=(width,height):
        return None
    pixels=bytearray(call.stdout[header.end():])
    if len(pixels)!=width*height:
        return None
    if erase_ring:
        rx=(c-a)*scale/2
        ry=(d-b)*scale/2
        cx=(a+c)*scale/2-left
        cy=(b+d)*scale/2-top
        for y in range(height):
            for x in range(width):
                if ((x+.5-cx)/rx)**2+((y+.5-cy)/ry)**2>.91**2:
                    pixels[y*width+x]=255
    return f"P5\n{width} {height}\n255\n".encode()+pixels


def _single_divider_band(source, page, evidence):
    """Verify multiple vector witnesses against a single continuous printed band."""
    ring = evidence["ring"]["box"]
    cx = (ring[0] + ring[2]) / 2
    radius = (ring[2] - ring[0]) / 2
    heights = [line[2] for line in evidence["divider_segments"]]
    lo, hi = min(heights), max(heights)
    scale = 8
    box = [cx - .8*radius, lo - .2*radius, cx + .8*radius, hi + .2*radius]
    try:
        raster = _caption_raster(source, page, box, scale)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if raster is None:
        return None
    _, shape, _, ink = raster.split(b"\n", 3)
    width, height = map(int, shape.split())
    left, top = math.floor(box[0]*scale), math.floor(box[1]*scale)
    y0, y1 = (round(y*scale - top - .5) for y in (lo, hi))
    # The centerline can lie on a pixel edge. Allow one pixel of coverage on
    # each outside boundary, but reject a white row *between* distinct strokes.
    mid = (y0 + y1) // 2
    contrast_checks = {round((cx + t*radius)*scale - left - .5)
                       for t in (-.7, -.45, .45, .7)}
    x0, x1 = min(contrast_checks), max(contrast_checks)
    previous = None
    for x in range(x0, x1+1):
        if not (0 <= x < width and 0 < mid < height-1):
            return None
        def dark(y):
            return ink[y*width+x] < 170
        anchor = next((y for y in (mid, mid-1, mid+1) if dark(y)), None)
        if anchor is None:
            return None
        first = last = anchor
        while first and dark(first-1):
            first -= 1
        while last+1 < height and dark(last+1):
            last += 1
        if first > y0+1 or last < y1-1:
            return None
        if x in contrast_checks and (first == 0 or last == height-1):
            return None
        if previous is not None and (first > previous[1] or last < previous[0]):
            return None
        previous = first, last
    return {"dpi": 72*scale, "columns": x1-x0+1, "vector_segments": len(heights)}


def _ink_window(source, page, markers):
    """Obtain one bounded physical raster for all marked regions on this page."""
    if not markers:
        return None
    boxes=[m["box"] for m in markers]
    left=max(0,math.floor(min(a-.25*(c-a) for a,b,c,d in boxes)))
    top=max(0,math.floor(min(b-.25*(d-b) for a,b,c,d in boxes)))
    right=math.ceil(max(c+1.75*(c-a) for a,b,c,d in boxes))
    bottom=math.ceil(max(d+.25*(d-b) for a,b,c,d in boxes))
    scale=2
    try:
        pgm=_caption_raster(source,page,[left,top,right,bottom],scale,pixel_limit=22000000)
    except (OSError,subprocess.TimeoutExpired):
        return None
    if pgm is None:
        return None
    header,shape,maximum,pixels=pgm.split(b"\n",3)
    width,height=map(int,shape.split())
    return left,top,scale,width,height,pixels


def _printed_mark(ink, marker, right_rule=False):
    """Require visible two-sided contrast around the ring and through its rule."""
    if ink is None:return None
    left,top,scale,width,height,pixels=ink
    a,b,c,d=marker["box"]
    cx,cy,r=(a+c)/2,(b+d)/2,(c-a)/2
    def profile(x,y):
        i=round((x-left)*scale-.5);j=round((y-top)*scale-.5)
        if not 1<=i<width-1 or not 1<=j<height-1:return 255,255
        values=sorted(pixels[(j+dj)*width+i+di] for di in (-1,0,1) for dj in (-1,0,1))
        return values[0],values[4]
    arc_hits=0
    for i in range(16):
        theta=(i+.5)*math.tau/16
        x,y=math.cos(theta),math.sin(theta)
        core=profile(cx+r*x,cy+r*y)[0]
        inside=profile(cx+.78*r*x,cy+.78*r*y)[1]
        outside=profile(cx+1.22*r*x,cy+1.22*r*y)[1]
        arc_hits+=core+35<=min(inside,outside)
    offsets=(1.35,1.85,2.5,3.1) if right_rule else (-.72,-.4,.4,.72)
    rule_hits=[]
    for dx in offsets:
        x=cx+dx*r
        core=profile(x,cy)[0]
        rule_hits.append(core+35<=min(profile(x,cy-.23*r)[1],profile(x,cy+.23*r)[1]))
    if arc_hits<10 or not(any(rule_hits[:2]) and any(rule_hits[2:])):
        return None
    return {"ring_contrast_samples":arc_hits, "ring_samples":16,
            "rule_contrast_samples":sum(rule_hits), "rule_samples":4, "dpi":72*scale}


def _printed_source_tokens(ink, marker, ref):
    """Require two-dimensional ink in each text box, away from the outline and rule."""
    if ink is None:
        return None
    left, top, scale, width, height, pixels = ink
    a, b, c, d = marker["box"]
    cx, cy, radius = (a + c) / 2, (b + d) / 2, (c - a) / 2
    evidence = {}
    rows = (("numerator", ref["numerator_evidence"][0]), ("sheet", ref["source"]))
    for name, row in rows:
        x0, y0, x1, y1 = row["box"]
        xlo = max(0, math.floor((x0 - left) * scale))
        xhi = min(width, math.ceil((x1 - left) * scale))
        ylo = max(0, math.floor((y0 - top) * scale))
        yhi = min(height, math.ceil((y1 - top) * scale))
        cells = {}
        for j in range(ylo, yhi):
            y = top + (j + .5) / scale
            distance = cy - y if name == "numerator" else y - cy
            if distance < .17 * radius:
                continue
            for i in range(xlo, xhi):
                x = left + (i + .5) / scale
                if (x - cx) ** 2 + (y - cy) ** 2 < (.83 * radius) ** 2:
                    cells[i, j] = pixels[j * width + i]
        # A one-dimensional construction line or flat fill does not establish
        # visible character strokes. Compare adjacent pixels inside the mask.
        dark = [(i, j) for (i, j), value in cells.items() if value < 150]
        if not dark:
            return None
        horizontal_edges = set()
        vertical_edges = set()
        for (i, j), value in cells.items():
            if value >= 150:
                continue
            if any(cells.get((i + step, j), 0) >= value + 50 for step in (-1, 1)):
                horizontal_edges.add(j)
            if any(cells.get((i, j + step), 0) >= value + 50 for step in (-1, 1)):
                vertical_edges.add(i)
        if (len(dark) < .07 * len(cells) or len(horizontal_edges) < 2
                or len(vertical_edges) < 2):
            return None
        evidence[name] = {"interior_pixels": len(cells), "dark_pixels": len(dark),
                          "horizontal_edge_rows": len(horizontal_edges),
                          "vertical_edge_columns": len(vertical_edges)}
    return evidence


def _read_caption_words(pgm, psm, digits=False, *, boxes=False):
    if pgm is None:
        return []
    cmd=["tesseract","stdin","stdout","--psm",str(psm)]
    if digits:
        cmd.extend(["-c","tessedit_char_whitelist=0123456789"])
    call=subprocess.run(cmd+["-c", "tessedit_create_tsv=1"],input=pgm,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    words=[]
    if call.returncode==0:
        for line in call.stdout.decode(errors="replace").splitlines()[1:]:
            columns=line.split("\t",11)
            if len(columns)==12 and columns[0]=="5" and columns[11].strip():
                word = (columns[11].strip(), float(columns[10]))
                if boxes:
                    word += (tuple(int(v) for v in columns[6:10]),)
                words.append(word)
    return words


def _frame_detail_digits(pgm):
    """Give the same printed numeral a consistent white border without changing the glyphs."""
    if not pgm or not pgm.startswith(b"P5\n"):
        return None
    _, shape, _, pixels = pgm.split(b"\n", 3)
    width, height = map(int, shape.split())
    ink = [(i % width, i // width) for i, value in enumerate(pixels) if value <= 160]
    if not ink:
        return None
    xs, ys = zip(*ink)
    left, right = min(xs), max(xs) + 1
    top, bottom = min(ys), max(ys) + 1
    if right - left < 2 or bottom - top < .22 * height:
        return None
    margin = max(8, round((bottom - top) / 4))
    w, h = right - left + 2 * margin, bottom - top + 2 * margin
    frame = bytearray(b"\xff" * (w * h))
    for y in range(top, bottom):
        offset = (y - top + margin) * w + margin
        frame[offset:offset + right - left] = pixels[y * width + left:y * width + right]
    return f"P5\n{w} {h}\n255\n".encode() + frame


def _read_repainted_source(source, page, ref):
    rows = (ref["numerator_evidence"][0], ref["source"])
    observed = []
    try:
        for row in rows:
            for scale in (6, 8):
                crop = _frame_detail_digits(_caption_raster(source, page, row["box"], scale))
                words = _read_caption_words(crop, 7)
                if len(words) != 1 or words[0][0] != row["text"] or words[0][1] < 80:
                    return None
                observed.append(words[0])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return observed


def _read_ruled_detail(source, page, region, *, record_number_only=False):
    recognized = []
    views = []
    for scale, mode, digits_only in ((4, 8, True), (6, 10, False)):
        crop = _caption_raster(source, page, region["ring"]["box"], scale, True)
        words = _read_caption_words(crop, mode, digits_only)
        if len(words) != 1 or not re.fullmatch(r"[1-9]\d?", words[0][0]):
            return None
        recognized.append(words[0])
        views.append((crop, mode, digits_only))
    detail = recognized[0][0]
    if recognized[1][0] != detail:
        return None
    if min(confidence for _, confidence in recognized) < 80:
        # Retry confidence only. Both original reads must already agree on the complete numeral.
        recognized = []
        for crop, mode, digits_only in views:
            framed = _frame_detail_digits(crop)
            if framed is None:
                return None
            words = _read_caption_words(framed, mode, digits_only)
            if len(words) != 1 or words[0][0] != detail or words[0][1] < 80:
                return None
            recognized.append(words[0])
    label = dict(region, detail=detail, digit_confidences=[x[1] for x in recognized], page=page)
    return _qualify_detail_title(source, page, region, label, record_number_only)


def _qualify_detail_title(source, page, region, label, record_number_only):
    title=_read_caption_words(_caption_raster(source,page,region["caption_box"],3),7)
    if sum(bool(re.search(r"[A-Za-z]{2}", word)) and confidence >= 70
           for word, confidence in title) < 2:
        # A printed number can veto uniqueness without qualifying its caption as a destination.
        return label if record_number_only else None
    return dict(label, caption=" ".join(word for word, confidence in title))


def _detail_strokes(region, vectors):
    """Nominate complete vector glyphs for the full-resolution printed-label check."""
    a, b, c, d = region["ring"]["box"]
    diameter = d - b
    glyphs = []
    for v in vectors:
        x0, y0, x1, y1 = v["svg_bbox"]
        if not (a < x0 <= x1 < c and b < y0 <= y1 < d):
            continue
        points = v.get("svg_points")
        pairs = v.get("svg_segment_vertex_pairs")
        height = y1 - y0
        if not (points and pairs and 0.24 * diameter <= height <= 0.69 * diameter
                and 0.08 * height <= x1 - x0 <= 1.2 * height):
            return []
        edges = [(points[i], points[j]) for i, j in pairs if points[i] != points[j]]
        if not 1 <= len(edges) <= 64:
            return []
        shape = tuple(tuple((x-x0) / height if k == 0 else (x-y0) / height
                            for pt in edge for k, x in enumerate(pt)) for edge in edges)
        glyphs.append((x0, x1, height, shape, v["id"]))
    glyphs.sort(key=lambda g: g[0])
    if not 1 <= len(glyphs) <= 3 or any(g[1] >= h[0] for g,h in zip(glyphs, glyphs[1:])):
        return []
    if max(g[2] for g in glyphs) > 1.12 * min(g[2] for g in glyphs):
        return []
    return [(g[3],g[4]) for g in glyphs]


# OpenCV 4.10.0, modules/imgproc/src/hershey_fonts.cpp and drawing.cpp.
# The following table is the complete printable ASCII Hershey Simplex face.
# Source content SHA-256 values are 5fa0d65264f72e5b166d07973aabd49f27cc54d0074062a8efc312bf9ab95432
# and fdfb8bb141c7138ce0e448f0e034c4f83cded209d7501f737404bbaf3fbbcba4, respectively.
#
#  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
#
#  By downloading, copying, installing or using the software you agree to this license.
#  If you do not agree to this license, do not download, install,
#  copy or use the software.
#
#
#                           License Agreement
#                For Open Source Computer Vision Library
#
# Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
# Copyright (C) 2009, Willow Garage Inc., all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   * Redistribution's of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistribution's in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * The name of the copyright holders may not be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall the Intel Corporation or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#
#

_SIMPLEX_PRINTABLE = (
    'JZ',
    'MWRFRT RYQZR[SZRY',
    'JZNFNM VFVM',
    'G]OFOb UFUb JQZQ JWZW',
    'H\\PBP_ TBT_ YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX',
    'F^[FYGVHSHPGNFLFJGIIIKKMMMOLPJPHNF [FI[ YTWTUUTWTYV[X[ZZ[X[VYT',
    'E_\\O\\N[MZMYNXPVUTXRZP[L[JZIYHWHUISJRQNRMSKSIRGPFNGMIMKNNPQUXWZY[[[\\Z\\Y',
    'NVRFRM',
    'KYVBTDRGPKOPOTPYR]T`Vb',
    'KYNBPDRGTKUPUTTYR]P`Nb',
    'JZRLRX MOWU WOMU',
    'E_RIR[ IR[R',
    'MWSZR[QZRYSZS\\R^Q_',
    'E_IR[R',
    'MWRYQZR[SZRY',
    'G][BIb',
    'H\\QFNGLJKOKRLWNZQ[S[VZXWYRYOXJVGSFQF',
    'H\\NJPISFS[',
    'H\\LKLJMHNGPFTFVGWHXJXLWNUQK[Y[',
    'H\\MFXFRNUNWOXPYSYUXXVZS[P[MZLYKW',
    'H\\UFKTZT UFU[',
    'H\\WFMFLOMNPMSMVNXPYSYUXXVZS[P[MZLYKW',
    'H\\XIWGTFRFOGMJLOLTMXOZR[S[VZXXYUYTXQVOSNRNOOMQLT',
    'H\\YFO[ KFYF',
    'H\\PFMGLILKMMONSOVPXRYTYWXYWZT[P[MZLYKWKTLRNPQOUNWMXKXIWGTFPF',
    'H\\XMWPURRSQSNRLPKMKLLINGQFRFUGWIXMXRWWUZR[P[MZLX',
    'MWRMQNROSNRM RYQZR[SZRY',
    'MWRMQNROSNRM SZR[QZRYSZS\\R^Q_',
    'F^ZIJRZ[',
    'E_IO[O IU[U',
    'F^JIZRJ[',
    'I[LKLJMHNGPFTFVGWHXJXLWNVORQRT RYQZR[SZRY',
    'DaWNVLTKQKOLNMMOMRNTOUQVTVVUWS WKWSXUYV[V\\U]S]O\\L[JYHWGTFQFNGLHJJILHOHRIUJWLYNZQ[T[WZYY',
    'I[RFJ[ RFZ[ MTWT',
    'G\\KFK[ KFTFWGXHYJYLXNWOTP KPTPWQXRYTYWXYWZT[K[',
    'H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZV',
    'G\\KFK[ KFRFUGWIXKYNYSXVWXUZR[K[',
    'H[LFL[ LFYF LPTP L[Y[',
    'HZLFL[ LFYF LPTP',
    'H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZVZS USZS',
    'G]KFK[ YFY[ KPYP',
    'NVRFR[',
    'JZVFVVUYTZR[P[NZMYLVLT',
    'G\\KFK[ YFKT POY[',
    'HYLFL[ L[X[',
    'F^JFJ[ JFR[ ZFR[ ZFZ[',
    'G]KFK[ KFY[ YFY[',
    'G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF',
    'G\\KFK[ KFTFWGXHYJYMXOWPTQKQ',
    'G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF SWY]',
    'G\\KFK[ KFTFWGXHYJYLXNWOTPKP RPY[',
    'H\\YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX',
    'JZRFR[ KFYF',
    'G]KFKULXNZQ[S[VZXXYUYF',
    'I[JFR[ ZFR[',
    'F^HFM[ RFM[ RFW[ \\FW[',
    'H\\KFY[ YFK[',
    'I[JFRPR[ ZFRP',
    'H\\YFK[ KFYF K[Y[',
    'KYOBOb OBVB ObVb',
    'G]IL[b',
    'KYUBUb NBUB NbUb',
    'G]JTROZT JTRPZT',
    'I[J[Z[',
    'LXPFUL PFOGUL',
    'I\\XMX[ XPVNTMQMONMPLSLUMXOZQ[T[VZXX',
    'H[LFL[ LPNNPMSMUNWPXSXUWXUZS[P[NZLX',
    'I[XPVNTMQMONMPLSLUMXOZQ[T[VZXX',
    'I\\XFX[ XPVNTMQMONMPLSLUMXOZQ[T[VZXX',
    'I[LSXSXQWOVNTMQMONMPLSLUMXOZQ[T[VZXX',
    'MYWFUFSGRJR[ OMVM',
    'I\\XMX]W`VaTbQbOa XPVNTMQMONMPLSLUMXOZQ[T[VZXX',
    'I\\MFM[ MQPNRMUMWNXQX[',
    'NVQFRGSFREQF RMR[',
    'MWRFSGTFSERF SMS^RaPbNb',
    'IZMFM[ WMMW QSX[',
    'NVRFR[',
    'CaGMG[ GQJNLMOMQNRQR[ RQUNWMZM\\N]Q][',
    'I\\MMM[ MQPNRMUMWNXQX[',
    'I\\QMONMPLSLUMXOZQ[T[VZXXYUYSXPVNTMQM',
    'H[LMLb LPNNPMSMUNWPXSXUWXUZS[P[NZLX',
    'I\\XMXb XPVNTMQMONMPLSLUMXOZQ[T[VZXX',
    'KXOMO[ OSPPRNTMWM',
    'J[XPWNTMQMNNMPNRPSUTWUXWXXWZT[Q[NZMX',
    'MYRFRWSZU[W[ OMVM',
    'I\\MMMWNZP[S[UZXW XMX[',
    'JZLMR[ XMR[',
    'G]JMN[ RMN[ RMV[ ZMV[',
    'J[MMX[ XMM[',
    'JZLMR[ XMR[P_NaLbKb',
    'J[XMM[ MMXM M[X[',
    'KYTBQEPHPJQMSOSPORSTSUQWPZP\\Q_Tb',
    'NVRBRb',
    'KYPBSETHTJSMQOQPURQTQUSWTZT\\S_Pb',
    'F^IUISJPLONOPPTSVTXTZS[Q ISJQLPNPPQTTVUXUZT[Q[O',
)


def _visible_complete_label(source, page, region, glyphs, vectors):
    """Check the complete printed label, resolving subpixel gaps only when physically plausible."""
    identifiers = {identifier for _, identifier in glyphs}
    ring = region["ring"]["box"]
    selected = [v for v in vectors if v["id"] in identifiers]
    if _visible_label_at_scale(source, page, ring, selected, 8):
        return True
    # A lettering stroke and a marker rim can occupy adjacent pixels at 8 px/pt
    # even if the vector pen footprints leave a small positive gap. Choose one
    # finer observation using the two pen widths, then repeat the original
    # connected-component and complete-edge proof on the unmodified PDF.
    ring_vector = next((v for v in vectors if v["id"] == region["ring"].get("id")), None)
    if not selected or not ring_vector or not ring_vector.get("svg_pen_diameter"):
        return False
    cx, cy = (ring[0]+ring[2])/2, (ring[1]+ring[3])/2
    inner_radius = min(ring[2]-ring[0], ring[3]-ring[1])/2 - ring_vector["svg_pen_diameter"]/2
    extent = 0.
    for v in selected:
        pen = v.get("svg_pen_diameter")
        if not pen or not v.get("svg_points"):
            return False
        extent = max(extent, max(math.hypot(x-cx,y-cy) for x,y in v["svg_points"])+pen/2)
    clearance = inner_radius - extent
    if not 0 < clearance < (2+math.sqrt(2))/8:
        return False
    scale = math.ceil((2+math.sqrt(2))/clearance)
    width = math.ceil(ring[2]*scale)-math.floor(ring[0]*scale)
    height = math.ceil(ring[3]*scale)-math.floor(ring[1]*scale)
    if width*height > 2_500_000:
        return False
    return _visible_label_at_scale(source, page, ring, selected, scale)


def _visible_label_at_scale(source, page, ring, selected, scale):
    """Every predicted edge must be visibly inked outside the ring's own stroke."""
    cx, cy = (ring[0]+ring[2])/2, (ring[1]+ring[3])/2
    radius = (ring[2]-ring[0])/2
    raster = _caption_raster(source, page, ring, scale)
    if raster is None:
        return False
    _, size, _, pixels = raster.split(b"\n", 3)
    width, height = map(int, size.split())
    left, top = math.floor(ring[0]*scale), math.floor(ring[1]*scale)
    dark_pixels = {(i % width, i // width) for i, pixel in enumerate(pixels) if pixel <= 150}
    def distance(pixel):
        ix, iy = pixel
        return math.hypot((left+ix+.5)/scale-cx, (top+iy+.5)/scale-cy)
    # Ring-connected ink cannot prove a glyph stroke. Follow it into the circle
    # as well, so thick outlines, joins and protrusions cannot serve as witnesses.
    rim = {pixel for pixel in dark_pixels if distance(pixel) >= radius-.25}
    frontier = list(rim)
    while frontier:
        ix, iy = frontier.pop()
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                neighbor = (ix+dx, iy+dy)
                if neighbor in dark_pixels and neighbor not in rim:
                    rim.add(neighbor)
                    frontier.append(neighbor)
    interior = dark_pixels - rim
    expected_paint = set()
    for v in selected:
        pen = v.get("svg_pen_diameter")
        if pen is None or not 0 < pen <= .2*radius:
            return False
        reach = pen*scale/2+1
        step = math.ceil(reach)
        footprint = [(dx,dy) for dx in range(-step,step+1) for dy in range(-step,step+1)
                     if math.hypot(dx,dy) <= reach]
        centerline = set()
        points = v["svg_points"]
        for i, j in v["svg_segment_vertex_pairs"]:
            a, b = points[i], points[j]
            if a == b:
                continue
            steps = max(1, math.ceil(math.dist(a,b)*scale*2))
            for n in range(steps+1):
                x = a[0]+(b[0]-a[0])*n/steps
                y = a[1]+(b[1]-a[1])*n/steps
                if math.hypot(x-cx,y-cy) >= radius:
                    return False
                ix, iy = math.floor(x*scale)-left, math.floor(y*scale)-top
                if min(ix,iy) < 1 or ix >= width-1 or iy >= height-1:
                    return False
                if not any((ix+dx,iy+dy) in interior
                           for dx in (-1,0,1) for dy in (-1,0,1)):
                    return False
                centerline.add((ix,iy))
        expected_paint.update((ix+dx,iy+dy) for ix,iy in centerline for dx,dy in footprint)
    return bool(interior) and len(interior & expected_paint) >= .95*len(interior)


def _simplex_strokes():
    """Decode the reference face independent of any destination sheet or OCR output."""
    symbols = []
    for i, record in enumerate(_SIMPLEX_PRINTABLE):
        strokes = [[(ord(x)-82, ord(y)-82) for x,y in zip(line[::2],line[1::2])]
                   for line in record[2:].split()]
        points = [p for stroke in strokes for p in stroke]
        if not points:
            continue
        x0 = min(x for x,y in points)
        y0 = min(y for x,y in points)
        height = max(y for x,y in points) - y0
        if height == 0:
            continue
        segments = tuple(((a[0]-x0)/height, (a[1]-y0)/height,
                          (b[0]-x0)/height, (b[1]-y0)/height)
                         for line in strokes for a,b in zip(line,line[1:]) if a != b)
        width = (max(x for x,y in points)-x0)/height
        if segments and width:
            symbols.append((chr(i+32),width,segments))
    return symbols


def _same_stroke_graph(observed, prototype, stretch):
    """Match every edge once, in either drawing direction, including both endpoints."""
    if len(observed) != len(prototype):
        return False
    edges = [(a*stretch,b,c*stretch,d) for a,b,c,d in prototype]
    def near(p, q):
        return all(abs(x-y) <= .028 for x,y in zip(p,q))
    neighbors = [[j for j,q in enumerate(edges) if near(p,q) or near(p,(q[2],q[3],q[0],q[1]))]
                 for p in observed]
    if any(not matches for matches in neighbors):
        return False
    owners = {}
    def assign(i, seen):
        for j in neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j],seen):
                owners[j] = i
                return True
        return False
    return all(assign(i,set()) for i in range(len(observed)))


def _recognize_stroke_label(glyphs, alphabet):
    """One horizontal text scale must explain the complete token without a second reading."""
    if not glyphs:
        return None
    choices = []
    for observed,_ in glyphs:
        width = max(max(e[0],e[2]) for e in observed)
        fits = []
        for char,ref_width,prototype in alphabet:
            if len(observed) != len(prototype):
                continue
            ratio = width/ref_width
            if .65 <= ratio <= 1.5 and _same_stroke_graph(observed,prototype,ratio):
                fits.append((char,ratio,prototype))
        if not fits:
            return None
        choices.append(fits)
    labels = set()
    for _,scale,_ in choices[0]:
        compatible = []
        for (observed, _), fits in zip(glyphs, choices):
            characters = {char for char, ratio, prototype in fits
                          if abs(ratio - scale) <= .03 and _same_stroke_graph(observed, prototype, scale)}
            if not characters:
                break
            compatible.append(characters)
        if len(compatible) != len(glyphs):
            continue
        if any(len(characters) != 1 for characters in compatible):
            return None
        labels.add("".join(next(iter(characters)) for characters in compatible))
    if len(labels) == 1:
        label = next(iter(labels))
        if re.fullmatch(r"[1-9]\d?[A-Z]?",label):
            return label
    return None


def _caption_face():
    """Retain each glyph's vertical position in the published face's 21-unit cap height."""
    shapes = {}
    for index, record in enumerate(_SIMPLEX_PRINTABLE):
        runs = [[(ord(x)-82, ord(y)-82) for x, y in zip(run[::2], run[1::2])]
                for run in record[2:].split()]
        edges = [(a, b) for run in runs for a, b in zip(run, run[1:]) if a != b]
        if not edges:
            continue
        left = min(x for edge in edges for x, _ in edge)
        shapes[chr(index+32)] = tuple(((a[0]-left)/21, (a[1]+12)/21,
                                       (b[0]-left)/21, (b[1]+12)/21)
                                      for a,b in edges)
    return shapes


def _caption_line_geometry(region, vectors):
    """Separate the baseline immediately above the caption rule from dimension notes."""
    if not vectors or "caption_box" not in region:
        return None
    a, b, c, d = region["caption_box"]
    diameter = region["ring"]["box"][2]-region["ring"]["box"][0]
    visible = []
    bands = {}
    for vector in vectors:
        box = vector["svg_bbox"]
        if box[0] < a or box[2] > c or box[1] < b or box[3] > d+.12*diameter:
            continue
        pairs = vector.get("svg_segment_vertex_pairs")
        pts = vector.get("svg_points")
        if not pairs or not pts:
            continue
        edges = [(pts[i], pts[j]) for i,j in pairs if pts[i] != pts[j]]
        if not edges:
            continue
        visible.append((vector, edges))
        height = box[3]-box[1]
        if .14*diameter <= height <= .52*diameter and 0 < box[2]-box[0] < 1.7*height:
            band = round(box[1], 1), round(box[3], 1)
            bands[band] = bands.get(band, 0)+1
    if not bands:
        return None
    (top, bottom), votes = max(bands.items(), key=lambda item:item[1])
    if votes < 3 or sum(count==votes for count in bands.values())>1:
        return None
    cap = bottom-top
    objects = []
    for vector,edges in visible:
        x0,y0,x1,y1 = vector["svg_bbox"]
        if y1 < top-.25*cap or y0 > bottom+.38*cap:
            continue
        if y0 < top-.28*cap or y1 > bottom+.38*cap or x1-x0 > 2.2*cap:
            continue
        pen = vector.get("svg_pen_diameter")
        if pen is None or not 0 < pen < .2*cap:
            return None
        objects.append((vector,edges))
    if not 3 <= len(objects) <= 180:
        return None
    # A character may consist of multiple touching or horizontally overlapping primitives.
    groups = []
    for vector,edges in sorted(objects,key=lambda item:item[0]["svg_bbox"][0]):
        lo,_,hi,_ = vector["svg_bbox"]
        if not groups or lo > groups[-1][0]+.01:
            groups.append([hi,[],[]])
        groups[-1][0] = max(groups[-1][0],hi)
        groups[-1][1].extend(edges)
        groups[-1][2].append(vector["id"])
    return top,bottom,objects,groups


def _caption_graph_options(groups, face, top, cap, stretch):
    """Exhaust the published alphabet and preserve competitors, including raised commas."""
    options = {}
    for i in range(len(groups)):
        combo = []
        for n in range(1,min(4,len(groups)-i)+1):
            combo.extend(groups[i+n-1][1])
            left = min(x for edge in combo for x,_ in edge)
            observed = tuple(((u[0]-left)/cap,(u[1]-top)/cap,
                              (v[0]-left)/cap,(v[1]-top)/cap) for u,v in combo)
            choices = {ch for ch,prototype in face.items()
                       if len(prototype)==len(observed) and _same_stroke_graph(observed,prototype,stretch)}
            if n==1 and len(observed)==len(face[","]):
                upper = min(min(y,w) for _,y,_,w in observed)
                lower = max(max(y,w) for _,y,_,w in observed)
                comma_top = min(min(y,w) for _,y,_,w in face[","])
                shifted = tuple((x,y-upper+comma_top,z,w-upper+comma_top) for x,y,z,w in observed)
                if -.28 <= upper <= .12 and lower < .5 and _same_stroke_graph(shifted,face[","],stretch):
                    choices.add("’")
            if choices:
                options[i,n] = choices
    for i in range(len(groups)-1):
        if "’" not in options.get((i,1),()) or "’" not in options.get((i+1,1),()):
            continue
        first = groups[i][1];second = groups[i+1][1]
        left = min(x for edge in second for x,_ in edge)
        right = max(x for edge in first for x,_ in edge)
        y0 = min(y for edge in first for _,y in edge)
        y1 = min(y for edge in second for _,y in edge)
        if .02*cap < left-right < .5*cap and abs(y0-y1) <= .04*cap:
            options.setdefault((i,2),set()).add("”")
    return options


def _caption_token_route(text, groups, options, cap):
    """Every physical group must belong to exactly one glyph of the independently read line."""
    tokens=text.split()
    chars = [(char, i) for i,token in enumerate(tokens) for char in token]
    frontier = [(0, [])]
    for k,(ch,word) in enumerate(chars):
        next_paths = []
        for gi,path in frontier:
            for count in range(1,5):
                choices=options.get((gi,count),())
                if ch not in choices:
                    continue
                # Multiple printed forms such as I/l need matching evidence from the two
                # word reads. Ambiguous numeric values, unit styles, or punctuation veto.
                if len(choices)>1 and not set(choices)<=set("Il"):
                    continue
                if gi and path:
                    right=groups[gi-1][0]
                    left=min(x for edge in groups[gi][1] for x,_ in edge)
                    gap=(left-right)/cap
                    if (word != chars[k-1][1] and gap < .42) or (word == chars[k-1][1] and gap > .85):
                        continue
                next_paths.append((gi+count,[*path,(ch,gi,count,sorted(choices))]))
        frontier=next_paths
        if not frontier or len(frontier)>48:
            return None
    complete=[p for i,p in frontier if i==len(groups)]
    return complete[0] if len(complete)==1 else None


def _caption_edge_ink(source, page, region, bounds, objects):
    """Confirm painted strokes at native geometry, excluding the long caption rule."""
    scale = 8
    max_pen = max(v["svg_pen_diameter"] for v,_ in objects)
    pad = max_pen+1/scale
    # Use the complete title span. An added printed mark past the nominated
    # first or last glyph also needs its own physical drawing operation.
    start, _, end, _ = region["caption_box"]
    crop = [min(bounds[0]-pad,start+pad),bounds[1]-pad,
            max(bounds[2]+pad,end-pad),bounds[3]+pad]
    raw = _caption_raster(source,page,crop,scale)
    if raw is None:
        return None
    _, shape, _, pixels = raw.split(b"\n",3)
    width,height = map(int,shape.split())
    left,top = math.floor(crop[0]*scale),math.floor(crop[1]*scale)
    actual = {(i%width,i//width) for i,pixel in enumerate(pixels) if pixel<=150}
    diameter = region["ring"]["box"][2]-region["ring"]["box"][0]
    rule_y = region["caption_box"][3]+.045*diameter
    rule_rows = {y for y in range(height)
                 if abs((y+top+.5)/scale-rule_y)<=max_pen+1/scale
                 and sum((x,y) in actual for x in range(width)) >= .8*width}
    actual_text = {(x,y) for x,y in actual if y not in rule_rows}
    expected = set()
    covered_edges = 0
    occluded_operations = set()
    for vector,edges in objects:
        reach = vector["svg_pen_diameter"]*scale/2+1
        size = math.ceil(reach)
        footprint = [(dx,dy) for dx in range(-size,size+1) for dy in range(-size,size+1)
                     if math.hypot(dx,dy)<=reach]
        for a,b in edges:
            per_edge = 0
            count = max(1,math.ceil(math.dist(a,b)*scale*2))
            for k in range(count+1):
                x=math.floor((a[0]+(b[0]-a[0])*k/count)*scale)-left
                y=math.floor((a[1]+(b[1]-a[1])*k/count)*scale)-top
                if not any((x+dx,y+dy) in actual for dx in (-1,0,1) for dy in (-1,0,1)):
                    return None
                direct=any((x+dx,y+dy) in actual_text for dx in (-1,0,1) for dy in (-1,0,1))
                if y in rule_rows:
                    occluded_operations.add(vector["id"])
                if not direct:
                    if y not in rule_rows:
                        return None
                else:
                    per_edge+=1
                expected.update((x+dx,y+dy) for dx,dy in footprint)
            # A rule cannot supply an otherwise hidden edge.
            if per_edge<.70*(count+1):
                return None
            covered_edges+=1
    # The true transformed nib has a one-pixel envelope for sampling.
    # A foreign mark can occupy arbitrarily little of a long caption.
    if not actual_text or actual_text.difference(expected):
        return None
    return {"visible_edges":covered_edges,"rule_rows_excluded":len(rule_rows),
            "ink_agreement":len(actual_text & expected)/len(actual_text),
            "rule_intersection_operations":sorted(occluded_operations),
            "rule_raster_y":[top+y for y in sorted(rule_rows)]}


def _rule_hides_competitor(route, groups, face, cap, top, stretch, hidden_ids, hidden_rows):
    """Compare glyph centerlines after removing rows owned by the printed rule."""
    scale = 8
    radius = max(1, math.ceil(.028*cap*scale))
    offsets = [(dx,dy) for dx in range(-radius,radius+1)
               for dy in range(-radius,radius+1) if math.hypot(dx,dy)<=radius]
    def samples(edges):
        result=set()
        for a,b in edges:
            count=max(1,math.ceil(math.dist(a,b)*scale*2))
            for k in range(count+1):
                x=math.floor((a[0]+(b[0]-a[0])*k/count)*scale)
                y=math.floor((a[1]+(b[1]-a[1])*k/count)*scale)
                if y not in hidden_rows:
                    result.add((x,y))
        return result
    for char,start,count,competitors in route:
        span=groups[start:start+count]
        if not any(key in hidden_ids for _,_,ids in span for key in ids):
            continue
        edges=[edge for _,word_edges,_ in span for edge in word_edges]
        actual=samples(edges)
        if not actual:
            return True
        support={(x+dx,y+dy) for x,y in actual for dx,dy in offsets}
        lo=min(x for edge in edges for x,y in edge)
        hi=max(x for edge in edges for x,y in edge)
        for rival,shape in face.items():
            if rival==char:
                continue
            width=max(max(x,z) for x,y,z,w in shape)*cap*stretch
            for origin in {lo,hi-width,(hi+lo-width)/2}:
                alternative=[((origin+x*cap*stretch,top+y*cap),(origin+z*cap*stretch,top+w*cap))
                             for x,y,z,w in shape]
                alt=samples(alternative)
                if not alt or not alt<=support:
                    continue
                alt_support={(x+dx,y+dy) for x,y in alt for dx,dy in offsets}
                if actual<=alt_support:
                    return True
    return False


def _quantitative_caption(model, caption):
    """A printed number or numeric transcription makes the whole quantitative title mandatory."""
    if re.search(r"\d", caption):
        return True
    if model is None:
        return False
    top,bottom,_,groups=model
    cap=bottom-top
    face=_caption_face()
    for _,edges,_ in groups:
        left=min(x for edge in edges for x,y in edge)
        obs=tuple(((a[0]-left)/cap,(a[1]-top)/cap,
                   (b[0]-left)/cap,(b[1]-top)/cap) for a,b in edges)
        width=max(max(x,z) for x,y,z,w in obs)
        for digit in "0123456789":
            shape=face[digit]
            if len(shape)!=len(obs):
                continue
            expected=max(max(x,z) for x,y,z,w in shape)
            stretch=width/expected
            if .65<=stretch<=1.5 and _same_stroke_graph(obs,shape,stretch):
                if not any(ch.isalpha() and len(alt)==len(obs) and
                           _same_stroke_graph(obs,alt,stretch) for ch,alt in face.items()):
                    return True
    return False


def _caption_token_views(source, page, box, groups, cap):
    """Locate words on two line views; verify any disputed token in its own physical crop."""
    rows=[]
    for scale in (4,6):
        raster=_caption_raster(source,page,box,scale)
        if raster is None:
            return None
        obs=_read_caption_words(raster,7,boxes=True)
        if not 2<=len(obs)<=24:
            return None
        left=math.floor(box[0]*scale)
        rows.append([(token,conf,(left+x)/scale,(left+x+w)/scale) for token,conf,(x,y,w,h) in obs])
    if len(rows[0])!=len(rows[1]):
        return None
    records=[]
    for first,second in zip(*rows):
        if min(first[3],second[3]) <= max(first[2],second[2]):
            return None
        if first[0]==second[0] and min(first[1],second[1])>=80:
            records.append((first[0],[first[1],second[1]],"line"))
            continue
        left=min(first[2],second[2]);right=max(first[3],second[3])
        spans=[]
        for _,edges,_ in groups:
            pts=[p for edge in edges for p in edge]
            x0=min(x for x,y in pts);x1=max(x for x,y in pts)
            if left-.06*cap <= (x0+x1)/2 <= right+.06*cap:
                if x0<left-.22*cap or x1>right+.22*cap:
                    return None
                spans.extend(pts)
        if not spans:
            return None
        gap=.26*cap
        small=[min(x for x,y in spans)-gap,min(y for x,y in spans)-gap,
               max(x for x,y in spans)+gap,max(y for x,y in spans)+gap]
        confirmations=[]
        for scale,mode in ((4,7),(6,8)):
            read=_read_caption_words(_caption_raster(source,page,small,scale),mode)
            if len(read)!=1 or read[0][1]<80:
                return None
            confirmations.append(read[0])
        if (confirmations[0][0]!=confirmations[1][0] or
                any(conf>=80 and token!=confirmations[0][0] for token,conf,*_ in (first,second))):
            return None
        records.append((confirmations[0][0],[q[1]for q in confirmations],"word"))
    return records


def _external_caption_metrics():
    """The two characters preceding each stroke stream are its Hershey bearings."""
    metrics = {}
    for index, record in enumerate(_SIMPLEX_PRINTABLE):
        left_bearing, right_bearing = (ord(code) - 82 for code in record[:2])
        xs = [ord(x) - 82 for run in record[2:].split() for x in run[::2]]
        metrics[chr(index + 32)] = (right_bearing - left_bearing,
                                    min(xs) - left_bearing if xs else 0)
    return metrics


def _shared_tracking_partitions(residuals, space, intervals=None):
    """Intersect every observed displacement with one common tracking interval."""
    # A quarter of an external en-space keeps neighboring space-count models
    # separated. Each measured advance also includes the common tracking, twice
    # when an additional space glyph appears between two printed characters.
    tolerance = space / 4
    paths = [(-tolerance, tolerance, ())]
    for delta in residuals:
        updated = []
        for lower, upper, cuts in paths:
            for extra_space in (False, True):
                count = 1 + int(extra_space)
                offset = space if extra_space else 0
                lo = max(lower, (delta - offset - tolerance) / count)
                hi = min(upper, (delta - offset + tolerance) / count)
                if hi > lo:
                    updated.append((lo, hi, cuts + (extra_space,)))
        paths = updated
        if len(paths) > 64:
            return None
        if not paths:
            return set()
    if intervals is not None:
        intervals.update((cuts, (lo, hi)) for lo, hi, cuts in paths)
    return {cuts for _, _, cuts in paths}


def _font_spaced_words(groups, cap, top, proof=None):
    """Only one word partition may fit the complete alphabet and line metrics."""
    face = _caption_face()
    metrics = _external_caption_metrics()
    # The OCR-independent scale is measured by fitting actual full stroke
    # graphs with a nonzero horizontal extent to the published alphabet.
    scales = set()
    group_boxes = []
    for _, edges, _ in groups:
        points = [p for edge in edges for p in edge]
        left = min(x for x, _ in points)
        right = max(x for x, _ in points)
        group_boxes.append((left, min(y for _, y in points),
                            right, max(y for _, y in points)))
        observed = tuple(((a[0] - left) / cap, (a[1] - top) / cap,
                          (b[0] - left) / cap, (b[1] - top) / cap) for a, b in edges)
        for shape in face.values():
            if len(shape) != len(edges):
                continue
            width = max(max(x, z) for x, _, z, _ in shape)
            if width <= .12:
                continue
            stretch = round((right - left) / (cap * width), 2)
            if .65 <= stretch <= 1.5 and _same_stroke_graph(observed, shape, stretch):
                scales.add(stretch)
    partitions = set()
    compatible_metrics = set()
    for stretch in scales:
        unit = cap * stretch / 21
        options = _caption_graph_options(groups, face, top, cap, stretch)
        # Keep every compatible glyph/advance sequence. Equivalent I/l glyphs
        # have identical metrics and therefore do not multiply layout paths.
        paths = [(0, ())]
        while paths:
            advanced = []
            for index, path in paths:
                if index == len(groups):
                    residuals = [(b[0] - a[0]) / unit - (a[2] + b[3] - a[3])
                                 for a, b in zip(path, path[1:])]
                    intervals = {}
                    fits = _shared_tracking_partitions(residuals, metrics[" "][0], intervals)
                    if fits is None:
                        return None
                    for cuts in fits:
                        starts = tuple(path[k + 1][1] for k, cut in enumerate(cuts) if cut)
                        partitions.add(starts)
                        compatible_metrics.add((stretch, *intervals[cuts]))
                    if len(partitions) > 1:
                        return None
                    continue
                for count in range(1, min(4, len(groups) - index) + 1):
                    choices = options.get((index, count), ())
                    dimensions = set()
                    for ch in choices:
                        if ch in metrics:
                            dimensions.add(metrics[ch])
                        elif ch == "’":
                            dimensions.add(metrics[","])
                        elif ch == "”" and count == 2:
                            advance, bearing = metrics[","]
                            offset = (group_boxes[index + 1][0] - group_boxes[index][0]) / unit
                            dimensions.add((advance + offset, bearing))
                    for advance, bearing in dimensions:
                        entry = (group_boxes[index][0], index, advance, bearing)
                        advanced.append((index + count, path + (entry,)))
            if len(advanced) > 128:
                return None
            paths = advanced
    if len(partitions) != 1:
        return None
    stops = (0, *next(iter(partitions)), len(groups))
    if not 2 <= len(stops) - 1 <= 24:
        return None
    words = []
    for start, end in zip(stops, stops[1:]):
        boxes = group_boxes[start:end]
        words.append([min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes)])
    if proof is not None:
        proof.update(external_face="Hershey Simplex", space_advance=metrics[" "][0],
                     unit="Hershey font coordinate", word_starts=list(stops[:-1]),
                     word_boxes=words, compatible_tracking=[
                         {"horizontal_scale": scale, "min_tracking": lo, "max_tracking": hi}
                         for scale, lo, hi in sorted(compatible_metrics)])
    return words


def _annotation_preserves_ink(annotated, underlying):
    """Only light was added if both dark masks retain their physical strokes."""
    if not annotated or not underlying:
        return False
    left = annotated.split(b"\n", 3)
    right = underlying.split(b"\n", 3)
    if left[:3] != right[:3] or len(left[3]) != len(right[3]):
        return False
    width, height = map(int, left[1].split())
    # Removing annotations must visibly change light background, not just a
    # letter's anti-aliased boundary. A dark correction remains authoritative.
    changed_background = sum(a < b - 12 and a > 150 and b >= 245
                             for a, b in zip(left[3], right[3]))
    if changed_background < max(4, width):
        return False
    for observed, other in ((left[3], right[3]), (right[3], left[3])):
        for i, value in enumerate(observed):
            if value > 150 or other[i] <= 150:
                continue
            # Only anti-aliased boundary samples may cross the ink threshold.
            if value < 100 or other[i] > 200:
                return False
            x, y = i % width, i // width
            if not any(other[yy*width+xx] <= 150
                       for yy in range(max(0, y-1), min(height, y+2))
                       for xx in range(max(0, x-1), min(width, x+2))):
                return False
    return True


def _physically_spaced_tokens(source, page, line_box, groups, cap, top, layout):
    """Use a uniquely fitted tracking-and-space layout to isolate each word."""
    words = _font_spaced_words(groups, cap, top, layout)
    if not words:
        return None
    # A confident complete word from either original line view can veto the
    # proposed spelling. A box spanning multiple separated words has no such
    # authority over individual letters or the physical whitespace.
    checks = [[] for _ in words]
    for scale in (4, 6):
        image = _caption_raster(source, page, line_box, scale)
        if image is None:
            return None
        line_left = math.floor(line_box[0] * scale)
        for token, confidence, (x, _, width, _) in _read_caption_words(image, 7, boxes=True):
            if confidence < 80:
                continue
            start = (line_left + x) / scale
            end = (line_left + x + width) / scale
            hits = [index for index, (left, _, right, _) in enumerate(words)
                    if min(end, right) > max(start, left)]
            if len(hits) == 1:
                checks[hits[0]].append((scale, token, confidence))

    result = []
    padding = .26 * cap
    for (left, top, right, bottom), confident in zip(words, checks):
        if len(confident) == 2 and {record[0] for record in confident} == {4, 6}:
            if confident[0][1] != confident[1][1]:
                return None
            result.append((confident[0][1], [record[2] for record in confident], "line"))
            continue
        box = [left - padding, top - padding, right + padding, bottom + padding]
        reads = []
        strong = []
        annotated_views = []
        raw_observations = []
        for scale, mode in ((4, 7), (6, 8)):
            image = _caption_raster(source, page, box, scale)
            if image is None:
                return None
            annotated_views.append((scale, mode, image))
            raw = _read_caption_words(image, mode)
            raw_observations.append({"scale": scale, "psm": mode, "words": raw})
            if len(raw) == 1 and raw[0][1] >= 80:
                strong.append(raw[0][0])
            # The exact same dark-ink classification is subsequently verified
            # against every native pen stroke and every pixel in the title span.
            header, shape, maximum, pixels = image.split(b"\n", 3)
            binary = pixels.translate(bytes([0] * 151 + [255] * 105))
            read = _read_caption_words(b"\n".join((header, shape, maximum, binary)), mode)
            if len(read) != 1 or read[0][1] < 80:
                return None
            reads.append(read[0])
        token = reads[0][0]
        if reads[1][0] != token or any(record[1] != token for record in confident):
            return None
        method = "external_font_spacing"
        if any(word != token for word in strong):
            control_observations = []
            for scale, mode, annotated in annotated_views:
                underlying = _caption_raster(source, page, box, scale, hide_annotations=True)
                if not _annotation_preserves_ink(annotated, underlying):
                    return None
                read = _read_caption_words(underlying, mode)
                if len(read) != 1 or read[0][0] != token or read[0][1] < 80:
                    return None
                control_observations.append({"scale": scale, "psm": mode, "words": read})
            layout.setdefault("annotation_controls", []).append({
                "word": token, "annotated_reads": raw_observations,
                "without_annotations": control_observations,
                "dark_ink_reads": reads, "annotated_and_underlying_ink_match": True,
            })
            method = "external_font_spacing_and_annotation_control"
        result.append((token, [read[1] for read in reads], method))
    return result


def _complete_caption_title(source, page, region, model):
    if not model:
        return None
    top,bottom,objects,groups=model
    cap=bottom-top
    bounds=[min(v["svg_bbox"][0] for v,_ in objects),min(v["svg_bbox"][1] for v,_ in objects),
            max(v["svg_bbox"][2] for v,_ in objects),max(v["svg_bbox"][3] for v,_ in objects)]
    crop_box=[bounds[0]-.26*cap,bounds[1]-.32*cap,bounds[2]+.26*cap,bounds[3]+.32*cap]
    layout = {}
    readings=_caption_token_views(source,page,crop_box,groups,cap)
    if not readings:
        readings = _physically_spaced_tokens(source, page, crop_box, groups, cap, top, layout)
    if not readings:
        return None
    text=" ".join(word for word,_,_ in readings)
    if len(text)>150:
        return None
    face=_caption_face()
    # Estimate common horizontal stretch from all presumptive glyph widths; the
    # winning scale has to explain the entire string and its component partition.
    scales={}
    for _,edges,_ in groups:
        xmin=min(x for edge in edges for x,y in edge);xmax=max(x for edge in edges for x,y in edge)
        actual=(xmax-xmin)/cap
        for char in set(text):
            proto=face.get(char,())
            if not proto or len(proto)!=len(edges):
                continue
            expected=max(max(x,z) for x,y,z,w in proto)
            if expected<=.12:
                continue
            ratio=round(actual/expected,2)
            if .65<=ratio<=1.5:
                scales[ratio]=scales.get(ratio,0)+1
    if not scales:
        return None
    best=max(scales,key=scales.get)
    options=_caption_graph_options(groups,face,top,cap,best)
    route=_caption_token_route(text,groups,options,cap)
    if route is None:
        return None
    physical=_caption_edge_ink(source,page,region,bounds,objects)
    if not physical or _rule_hides_competitor(route,groups,face,cap,top,best,
                                              set(physical["rule_intersection_operations"]),
                                              set(physical["rule_raster_y"])):
        return None
    return {"caption":text,"complete_caption_glyphs":route,"caption_text_stretch":best,"caption_ink":physical,
            "caption_token_observations":readings, **({"caption_spacing_evidence": layout} if layout else {}),
            "caption_vector_operations":[v["id"]for v,_ in objects]}


def _resolve_rendered_stacked(result, target, source, directory):
    refs=[r for r in result["references"] if r["target_sheet"]==target
          and r["reference_form"]=="stacked_text_candidate" and r.get("target_detail")
          and r["status"] in ("sheet_only","candidate_detail")]
    places=result["sheets"].get(target,[])
    if not refs or len(places)!=1 or places[0]["sheet_ids_on_page"]!=[target]:
        return
    target_page=places[0]["page"]
    by_source={}
    for ref in refs:
        by_source.setdefault(ref["source"]["page"],[]).append(ref)
    verified=[]
    for page, group in by_source.items():
        source_repaints = set()
        markers,lines=_detail_symbol_inventory(source,page,directory,source_repaints=source_repaints)
        matches=[(ref,evidence) for ref in group
                 if (evidence:=_divided_source_mark(ref,markers,lines))]
        ink=_ink_window(source,page,[e["ring"] for ref,e in matches])
        for ref,evidence in matches:
            if "divider_segments" in evidence:
                band = _single_divider_band(source, page, evidence)
                if band is None:
                    continue
                evidence["printed_divider_band"] = band
            visible=_printed_mark(ink,evidence["ring"])
            text_ink = _printed_source_tokens(ink, evidence["ring"], ref) if visible else None
            repaint_ids = {evidence["ring"].get("id"), evidence.get("divider")}
            repaint_ids.update(line[3] for line in evidence.get("divider_segments", []))
            if text_ink and source_repaints.intersection(repaint_ids):
                # Overprinted symbols can carry different text at one location.
                # Read the final visible numerator and sheet before binding either.
                observed = _read_repainted_source(source, page, ref)
                if observed is None:
                    continue
                evidence["repainted_source_text"] = observed
            if text_ink:
                ref["source_symbol_evidence"]=dict(evidence,visible_ink=visible, token_ink=text_ink)
                verified.append(ref)
    if not verified or not shutil.which("pdftoppm") or not shutil.which("tesseract"):
        return
    stroke_vectors=[]
    markers,lines=_detail_symbol_inventory(source,target_page,directory,stroke_vectors)
    regions=_ruled_detail_regions(markers,lines)
    if not 2<=len(regions)<=64:
        return
    identified={}
    unconfirmed={}
    alphabet = _simplex_strokes()
    ink=_ink_window(source,target_page,[reg["ring"] for reg in regions])
    for region in regions:
        visible=_printed_mark(ink,region["ring"],right_rule=True)
        if not visible:
            continue
        region=dict(region,visible_ink=visible)
        try:
            evidence=_read_ruled_detail(source,target_page,region,record_number_only=True)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
        strokes = _detail_strokes(region,stroke_vectors)
        detail = _recognize_stroke_label(strokes,alphabet)
        if detail and not _visible_complete_label(source, target_page, region, strokes, stroke_vectors):
            if evidence and evidence["detail"] != detail:
                for value in {detail, evidence["detail"]}:
                    unconfirmed.setdefault(value, []).append(dict(region, detail=value, page=target_page))
                continue
            # An independent two-view OCR result retains its own provenance.
            # The external font supplies no positive evidence for this site.
            detail = None
        if detail and evidence and evidence["detail"] != detail:
            for value in (detail,evidence["detail"]):
                unconfirmed.setdefault(value,[]).append(dict(region,detail=value,page=target_page))
            continue
        if detail and not evidence:
            label = dict(region,detail=detail,page=target_page,stroke_vectors=[v for _,v in strokes],
                         reference_face="Hershey Simplex")
            try:
                evidence = _qualify_detail_title(source,target_page,region,label,True)
            except (OSError,ValueError,subprocess.TimeoutExpired):
                evidence = label
        if evidence:
            model = _caption_line_geometry(region, stroke_vectors)
            must_verify = _quantitative_caption(model, evidence.get("caption", ""))
            if must_verify or "caption" not in evidence:
                try:
                    extra = _complete_caption_title(source, target_page, region, model)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    extra = None
                if extra:
                    evidence = dict(evidence, **extra)
                elif must_verify:
                    # A partial OCR title cannot override a quantitative mismatch.
                    evidence.pop("caption", None)
        if evidence:
            pool = identified if "caption" in evidence else unconfirmed
            pool.setdefault(evidence["detail"], []).append(evidence)
    for ref in verified:
        hits=identified.get(ref["target_detail"],[])
        if hits:
            ref["rendered_caption_evidence"]=hits
            # Optical recognition is explicitly a review candidate; the original native-text proofs stay authoritative.
            conflicts = unconfirmed.get(ref["target_detail"], [])
            if conflicts:
                ref["rendered_number_conflicts"] = conflicts
            if len(hits)>1 or conflicts:
                ref["status"]="ambiguous_detail"
            elif ref["status"]=="sheet_only":
                ref["status"]="candidate_detail"


def detail_links(filename, target_sheet=None):
    """Document-wide native text index. All boxes refer to Poppler crop-box text coordinates."""
    source = pathlib.Path(filename)
    with source.open("rb") as stream:
        data = stream.read(50_000_001)
    if len(data) > 50_000_000:
        raise ValueError("PDF exceeds 50 MB task-container inspection bound")
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("pdftotext required (Poppler)")
    # Hash and extract precisely the same bytes even if the supplied path changes.
    with tempfile.TemporaryDirectory(prefix="hamon-detail-") as directory:
        frozen = pathlib.Path(directory) / "source.pdf"
        layout = frozen.with_suffix(".xml")
        frozen.write_bytes(data)
        total, _ = _page_geometry(frozen, 1, 1)
        if not 1 <= total <= 256:
            raise ValueError("document-wide detail index supports at most 256 pages")
        extract = subprocess.run([exe, "-cropbox", "-bbox-layout", str(frozen), str(layout)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=45)
        if extract.returncode:
            raise ValueError("pdftotext failed to extract the document's native sheet text")
        if layout.stat().st_size > 64_000_000:
            raise ValueError("PDF native text layout exceeds 64 MB")
        pages = _poppler_text_xml(layout.read_bytes()).findall(".//{*}page")
        if len(pages) != total:
            raise ValueError("incomplete native page index")
        rows = [_drawing_text_lines(page, n) for n, page in enumerate(pages, 1)]
        result = _detail_graph(rows)
        if target_sheet is not None:
            _resolve_ruled_captions(result, rows, target_sheet.upper(), frozen, directory)
            _resolve_rendered_stacked(result, target_sheet.upper(), frozen, directory)
    result.update(kind="pdf-detail-links", source_sha256=hashlib.sha256(data).hexdigest(),
                  page_count=total, coordinate_frame="Poppler pdftotext -cropbox -bbox-layout",
                  coverage="Native-text headings and numbered vector captions are indexed. A selected target sheet may also include verified split source symbols and OCR-nominated caption boxes. Inspect optical candidates against the printed detail before use.")
    if target_sheet is not None:
        target = target_sheet.upper()
        selected = [r for r in result["references"] if r["target_sheet"] == target]
        needed = {target} | {s for r in selected for s in r["source_sheet_ids"]}
        result["references"] = selected
        result["sheets"] = {s: pages for s, pages in result["sheets"].items() if s in needed}
        result["target_sheet_filter"] = target
    return result
