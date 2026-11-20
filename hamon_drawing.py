"""Bounded PDF text, vector inventory and coordinate-grounded raster windows.

Text boxes are in the PDF-page coordinate frame, vector paths in Cairo SVG
viewBox coordinates. Any physical drawing scale is an unverified annotation,
never automatically applied to measurements. No OCR or network calls.
"""
from __future__ import annotations
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


def _svg_inventory(payload,limit_paths=2000,region=None,selected_vector_ids=None):
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
                out.append(row)
        stack.extend((child,matrix) for child in reversed(list(node)))
    return {"viewbox_svg_units":dims,"vector_primitives":stats["vector_primitives"],
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


def _word_inventory(data,max_words):
    root=ET.fromstring(data)
    words=[];page_box=None
    for element in root.iter():
        name=local(element.tag)
        if name=="page" and page_box is None:
            try:page_box=[float(element.attrib["width"]),float(element.attrib["height"])]
            except (ValueError,KeyError):pass
        if name=="word":
            try:b=[float(element.attrib[k]) for k in ("xMin","yMin","xMax","yMax")]
            except (ValueError,KeyError):continue
            words.append({"text":(element.text or "")[:160],"page_bbox_pt":b})
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


def _page_count(path):
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        raise RuntimeError("pdfinfo required (Poppler)")
    info = subprocess.run([pdfinfo, str(path)], capture_output=True,
                          timeout=10, check=False)
    if info.returncode != 0:
        raise ValueError("pdfinfo failed: encrypted/invalid PDF")
    match = re.search(rb"^Pages:\s*(\d+)\s*$", info.stdout, re.M)
    if not match:
        raise ValueError("PDF has no readable page count")
    return int(match.group(1))


def _page_svg(path, page, directory):
    """One canonical CropBox frame, with page rotation and no paper fitting.

    Cairo's default vector printing may shrink/center fractional page sizes.
    Explicit no-fit settings keep the SVG coordinates aligned with a CropBox
    raster at dpi/72. PDF word boxes retain their separate coordinate frame.
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
    total=_page_count(path)
    end=last if last is not None else first
    if end>total or end-first+1>20:
        raise ValueError("page range outside PDF or exceeds 20-sheet bound")
    pages=[]
    with tempfile.TemporaryDirectory(prefix="hamon-pdf-") as tmp:
        for page in range(first,end+1):
            text=subprocess.run([pdftotext,"-f",str(page),"-l",str(page),
                                 "-bbox-layout",str(path),"-"],
                                capture_output=True,timeout=20,check=False)
            if text.returncode!=0 or len(text.stdout)>8_000_000:
                raise ValueError(f"page {page}: PDF word extraction failed/budget exceeded")
            words,box=_word_inventory(text.stdout,limit_words)
            inventory=_svg_inventory(_page_svg(path,page,tmp),limit_paths,region,selected_vector_ids)
            pages.append({"page":page,"pdf_page_size_pt":box,
                          "svg_coordinate_frame":_SVG_FRAME,
                          "word_boxes_reported":len(words),
                          "words":words,"scale_annotations":_scales(words),
                          **inventory})
    return {"kind":"pdf-multisheet-vector-inventory",
            "file":str(path),"document_pages":total,
            "first_page":first,"last_page":end,"pages":pages,
            "physical_drawing_scale_verified":False,
            "warning":"Native PDF word boxes and SVG vector centerlines use separate page coordinate frames; SVG uses the rotated CropBox without paper fitting. Printed scale text is annotation only. SVG M/L/H/V/C/S/Q/T/Z curve centerline bboxes are exact under affine transforms; unsupported arc paths remain unlocalized, crop is exploratory, and no OCR, plotted geometric calibration, or code compliance is inferred."}


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
        total = _page_count(snapshot)
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
        "warning": "Read the emitted PNG to inspect the actual marks. Pixel-to-SVG mapping is page geometry, not physical calibration. Native word boxes use a separate frame; no dimension label, OCR result, engineering measurement or code approval is inferred.",
    }
