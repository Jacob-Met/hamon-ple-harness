"""Bounded multi-sheet PDF text, vector and scale-annotation inventory.

Text boxes are in the PDF-page coordinate frame, vector paths in Cairo SVG
viewBox coordinates. Any physical drawing scale is an unverified annotation,
never automatically applied to measurements. No raster/OCR or network calls.
"""
from __future__ import annotations
from collections import Counter
import math
import pathlib
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


_FLOAT=r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PART=re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
_NUMBER=re.compile(_FLOAT)
_TOKENS=re.compile(r"([A-Za-z])|("+_FLOAT+r")")
_ID=(1.0,0.0,0.0,1.0,0.0,0.0)


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


def _svg_inventory(payload,limit_paths=2000,region=None):
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
            if box is None:unlocalized+=1
            retained=(region is None or box is None or
                (box[0]<=region[2] and box[2]>=region[0] and
                 box[1]<=region[3] and box[3]>=region[1]))
            if retained and len(out)<limit_paths:
                row={"type":name,"svg_bbox":box,"bbox_exact_for_centerline":box is not None}
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
            "reported_count":len(out),
            "vector_limit_hit":stats["vector_primitives"]>limit_paths,
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


def scan_pdf(filename,first=1,last=None,limit_paths=2000,
             limit_words=3000,region=None):
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
    info=subprocess.run([pdfinfo,str(path)],capture_output=True,
                        timeout=10,check=False)
    if info.returncode!=0:raise ValueError("pdfinfo failed: encrypted/invalid PDF")
    match=re.search(rb"^Pages:\s*(\d+)\s*$",info.stdout,re.M)
    if not match:raise ValueError("PDF has no readable page count")
    total=int(match.group(1))
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
            output=pathlib.Path(tmp)/f"page_{page}.svg"
            render=subprocess.run([pdftocairo,"-svg","-f",str(page),
                                   "-l",str(page),
                                   str(path),str(output)],
                                  capture_output=True,timeout=30,check=False)
            if render.returncode!=0:
                raise ValueError(f"page {page}: SVG conversion failed: {render.stderr.decode(errors='replace')[:140]}")
            candidates=sorted(pathlib.Path(tmp).glob(f"page_{page}*.svg"))
            if not candidates:
                raise ValueError(f"page {page}: SVG conversion emitted no page file")
            if len(candidates)>1:
                raise ValueError(f"page {page}: SVG produced ambiguous page files")
            svg=candidates[0]
            if svg.stat().st_size>15_000_000:
                raise ValueError(f"page {page}: vector SVG exceeds 15 MB")
            inventory=_svg_inventory(svg.read_bytes(),limit_paths,region)
            pages.append({"page":page,"pdf_page_size_pt":box,
                          "word_boxes_reported":len(words),
                          "words":words,"scale_annotations":_scales(words),
                          **inventory})
    return {"kind":"pdf-multisheet-vector-inventory",
            "file":str(path),"document_pages":total,
            "first_page":first,"last_page":end,"pages":pages,
            "physical_drawing_scale_verified":False,
            "warning":"Native PDF word boxes and SVG vector centerlines use separate page coordinate frames; printed scale text is annotation only. SVG curves without straight-line bounds remain unlocalized, crop is exploratory, and no OCR, plotted geometric calibration, or code compliance is inferred."}
