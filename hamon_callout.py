"""Source-verified PDF dimension text ↔ actual SVG dimension-line geometry.

Unlike nearby-text heuristics, a certified association requires:
  - an exact native text word box and a compatible page coordinate frame;
  - a REAL SVG straight stroke, never an across-M phantom edge;
  - two distinct source strokes intersecting its opposite endpoints,
    with non-collinear endpoint/tick/extension geometry;
  - no equal-plausibility competing line or competing label assignment.

This proves a bounded *on-sheet annotation geometry link*, not that the
architect specified correct physical dimensions or a safe construction.
Unknown or ambiguous information is reported, never imputed.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

import hamon_drawing
import hamon_scale

_LABEL = re.compile(
    r"(?P<num>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
    r"(?:\.[0-9]+)?|\.[0-9]+)"
    r"\s*(?P<unit>inch|mm|cm|ft|in|yd|m)", re.IGNORECASE)
_LABEL_FULL = re.compile(r"(?:" + _LABEL.pattern + r")\Z", re.IGNORECASE)
_UNIT = {"inch":"inch","in":"inch","mm":"mm","cm":"cm",
         "ft":"ft","yd":"yd","m":"m"}
_MAX_SEGMENTS = 12000


def _box(value):
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise ValueError("source PDF label must carry an exact four-point box")
    if any(isinstance(v,bool) or not isinstance(v,(int,float))
           or not math.isfinite(v) for v in value):
        raise ValueError("source PDF label bbox contains invalid coordinates")
    a,b,c,d=map(float,value)
    if c<=a or d<=b:
        raise ValueError("source PDF word box has nonpositive extent")
    return (a,b,c,d)


def _page_map(page):
    if not isinstance(page,dict) or page.get("vector_limit_hit") is not False:
        raise ValueError("incomplete source vector inventory; scan all strokes before pairing callouts")
    dims=page.get("pdf_page_size_pt")
    vb=page.get("viewbox_svg_units")
    if not isinstance(dims,(list,tuple)) or len(dims)!=2 or (
            not isinstance(vb,(list,tuple)) or len(vb)!=4):
        raise ValueError("PDF/SVG coordinate frames are unavailable")
    vals=tuple(dims)+tuple(vb)
    if any(isinstance(v,bool) or not isinstance(v,(float,int))
           or not math.isfinite(v) for v in vals):
        raise ValueError("PDF/SVG coordinate frame dimensions nonfinite")
    pw,ph=map(float,dims); vx,vy,vw,vh=map(float,vb)
    if min(pw,ph,vw,vh)<=0 or max(pw,ph,vw,vh)>100000:
        raise ValueError("PDF/SVG coordinate frame dimensions outside bounds")
    sx,sy=vw/pw,vh/ph
    # Both native inventories cover the rotated CropBox. Reject incompatible extents.
    if abs(sx/sy-1)>0.005 or not 0.1<=sx<=10:
        raise ValueError("PDF/SVG page aspect or coordinate frames mismatch")
    return vx,vy,sx,sy,pw,ph


def _label(text):
    if not isinstance(text,str):
        return None
    t=text.strip()
    match=_LABEL_FULL.fullmatch(t)
    if not match:
        return None
    raw=match.group("num").replace(",","")
    try:
        value=Decimal(raw)
    except InvalidOperation:
        return None
    if not value.is_finite() or not Decimal("0")<=value<Decimal("1e9"):
        return None
    unit=_UNIT[match.group("unit").lower()]
    return {"source_label":t,
            "length":f"{raw}*{unit}",
            "physical_value":str(value),
            "physical_unit":unit}


def _adjacent_word_boxes(left, right):
    height = min(left[3] - left[1], right[3] - right[1])
    gap = right[0] - left[2]
    return (-0.5 <= gap <= max(2, 0.5 * height)
            and abs(left[1] + left[3] - right[1] - right[3]) <= 0.6 * height)


def _join_feet_inches(labels, words):
    """Treat a neighbouring ft/in pair on one extracted line as one dimension."""
    labels.sort(key=lambda row: row["word_ids"][0])
    merged = []
    index = 0
    while index < len(labels):
        first = labels[index]
        index += 1
        if index < len(labels):
            second = labels[index]
            end = first["word_ids"][-1]
            start = second["word_ids"][0]
            line = words[end].get("source_line_id")
            if ({first["physical_unit"], second["physical_unit"]} == {"ft", "inch"}
                    and start == end + 1 and line is not None
                    and all(words[i].get("source_line_id") == line
                            for i in first["word_ids"] + second["word_ids"])
                    and _adjacent_word_boxes(first["bbox_pdf_pt"], second["bbox_pdf_pt"])):
                index += 1
                feet, inches = Decimal(first["physical_value"]), Decimal(second["physical_value"])
                if first["physical_unit"] != "ft" or feet != feet.to_integral_value() or inches >= 12:
                    continue
                total = feet * 12 + inches
                if not 0 < total < Decimal("1e9"):
                    continue
                ids = first["word_ids"] + second["word_ids"]
                combined = dict(source_label=" ".join(words[i]["text"] for i in ids),
                                word_ids=ids, physical_value=str(total), physical_unit="inch",
                                length=f"{total}*inch")
                for key in ("bbox_pdf_pt", "bbox_svg"):
                    a, b = first[key], second[key]
                    combined[key] = [min(a[0], b[0]), min(a[1], b[1]),
                                     max(a[2], b[2]), max(a[3], b[3])]
                merged.append(combined)
                continue
        if Decimal(first["physical_value"]) > 0:
            merged.append(first)
    return merged


def _text_rects(page,frame):
    vx,vy,sx,sy,pw,ph=frame
    words=page.get("words")
    if not isinstance(words,list) or len(words)>5000:
        raise ValueError("bounded Poppler PDF word boxes required")
    observed=[]
    for i,word in enumerate(words):
        if not isinstance(word,dict):
            raise ValueError("malformed native PDF word row")
        box=_box(word.get("page_bbox_pt"))
        if box[0]<-1 or box[1]<-1 or box[2]>pw+1 or box[3]>ph+1:
            raise ValueError("PDF word box outside declared source page")
        item=_label(word.get("text"))
        if item is None:
            continue
        mapped=(vx+box[0]*sx,vy+box[1]*sy,
                vx+box[2]*sx,vy+box[3]*sy)
        observed.append({
            **item,"word_ids":[i],"bbox_pdf_pt":list(box),
            "bbox_svg":list(mapped),
        })
    # Compound "1000" "mm" is valid only when the unit is an adjacent word
    # on the *same* baseline. Never join separated title/table columns.
    for i in range(len(words)-1):
        first,second=words[i],words[i+1]
        x,y=first.get("text"),second.get("text")
        if not isinstance(x,str) or not isinstance(y,str):
            continue
        proposed=_label(x.strip()+" "+y.strip())
        if proposed is None:
            continue
        a,b=_box(first.get("page_bbox_pt")),_box(second.get("page_bbox_pt"))
        if not _adjacent_word_boxes(a, b):
            continue
        combined=(a[0],min(a[1],b[1]),b[2],max(a[3],b[3]))
        observed.append({
            **proposed,"word_ids":[i,i+1],
            "bbox_pdf_pt":list(combined),
            "bbox_svg":[vx+combined[0]*sx,vy+combined[1]*sy,
                        vx+combined[2]*sx,vy+combined[3]*sy]})
    # Poppler may split "100 ± 0.1 mm" into four words. A magnitude
    # following the tolerance sign cannot calibrate the whole dimension line.
    def follows_tolerance(label):
        index = label["word_ids"][0]
        if not index:
            return False
        prior = words[index - 1]
        text = prior.get("text")
        if not isinstance(text, str) or not text.rstrip().endswith(("±", "+/-")):
            return False
        value_word = words[index]
        left = _box(prior["page_bbox_pt"])
        right = _box(value_word["page_bbox_pt"])
        if "source_line_id" in prior and "source_line_id" in value_word:
            return prior["source_line_id"] == value_word["source_line_id"] and right[0] >= left[2] - 0.5
        return _adjacent_word_boxes(left, right)

    observed = [label for label in _join_feet_inches(observed, words)
                if not follows_tolerance(label)]
    observed.sort(key=lambda e:(e["bbox_svg"][1],e["bbox_svg"][0],
                                e["word_ids"]))
    if len(observed)>128:
        raise ValueError("over 128 dimension callouts; specify a smaller drawing")
    return observed


def _point(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError("invalid SVG source vertex")
    if any(not isinstance(v,(int,float)) or isinstance(v,bool)
           or not math.isfinite(v) for v in value):
        raise ValueError("nonfinite SVG source vertex")
    return (float(value[0]),float(value[1]))


def _strokes(page):
    rows=page.get("reported_vectors")
    if not isinstance(rows,list) or len(rows)>5000:
        raise ValueError("source path inventory not bounded")
    result=[]
    seen_ids=set()
    for vector in rows:
        if not isinstance(vector,dict):
            raise ValueError("malformed vector primitive")
        vid=vector.get("id")
        if not isinstance(vid,str) or not re.fullmatch(r"v[1-9][0-9]*",vid):
            raise ValueError("invalid source vector identifier")
        if vid in seen_ids:
            raise ValueError("duplicate source vector identifier")
        seen_ids.add(vid)
        pts=vector.get("svg_points")
        edges=vector.get("svg_segment_vertex_pairs")
        if pts is None or edges is None:
            continue
        if not isinstance(pts,list) or not isinstance(edges,list):
            raise ValueError("malformed SVG source points/segment table")
        coords=[_point(p) for p in pts]
        for pair in edges:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2:
                raise ValueError("malformed SVG stroke provenance")
            i,j=pair
            if (type(i) is not int or type(j) is not int
                    or not 0<=i<len(coords) or not 0<=j<len(coords)):
                raise ValueError("SVG stroke vertex index outside source vector")
            a,b=coords[i],coords[j]
            n=math.dist(a,b)
            if n<2 or n>100000:
                continue
            result.append({
                "from":{"vector":vid,"vertex":i},
                "to":{"vector":vid,"vertex":j},
                "a":a,"b":b,"length":n,"id":vid,
            })
            if len(result)>_MAX_SEGMENTS:
                raise ValueError("over 12000 actual strokes; bound the page")
    if not result:
        raise ValueError("no exact straight source SVG strokes found")
    return result


def _perpendicular_to(point,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    den=dx*dx+dy*dy
    if den<1e-14:
        return float("inf"),0.0
    t=((point[0]-a[0])*dx+(point[1]-a[1])*dy)/den
    perp=abs((point[0]-a[0])*dy-(point[1]-a[1])*dx)/math.sqrt(den)
    return perp,t


def _point_segment_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    den=dx*dx+dy*dy
    t=0 if den<1e-14 else max(0.0,min(1.0,
        ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/den))
    return math.dist(p,(a[0]+t*dx,a[1]+t*dy))


def _witness_grid(strokes):
    """Index original short line segments by intersected 16-unit SVG cells.

    An endpoint witness can only be <=48 units long and <=3.5 units from
    an actual segment. Index each exact segment bbox expanded by 3.5, so
    the matched cell is complete for all allowed endpoint distances.
    This collapses dense-sheet all-stroke O(label*paths**2) witnesses
    into bounded local lookups without an approximate nearest neighbor.
    """
    grid={}
    for stroke in strokes:
        if stroke["length"]>48:
            continue
        a,b=stroke["a"],stroke["b"]
        lo_x=math.floor((min(a[0],b[0])-3.5)/16)
        hi_x=math.floor((max(a[0],b[0])+3.5)/16)
        lo_y=math.floor((min(a[1],b[1])-3.5)/16)
        hi_y=math.floor((max(a[1],b[1])+3.5)/16)
        if (hi_x-lo_x+1)*(hi_y-lo_y+1)>49:
            raise ValueError("short source stroke exceeds bounded witness grid")
        for x in range(lo_x,hi_x+1):
            for y in range(lo_y,hi_y+1):
                grid.setdefault((x,y),[]).append(stroke)
    return grid


def _endpoint_witnesses(stroke,strokes,box_height,witness_grid):
    base_a,base_b=stroke["a"],stroke["b"]
    ux=(base_b[0]-base_a[0])/stroke["length"]
    uy=(base_b[1]-base_a[1])/stroke["length"]
    found=[]
    eps=max(2.3,min(3.5,0.16*box_height))
    for p in (base_a,base_b):
        options=[]
        local=witness_grid.get((math.floor(p[0]/16),math.floor(p[1]/16)),())
        for s in local:
            if s is stroke or s["length"]>min(48,max(12,0.35*stroke["length"])):
                continue
            vx=(s["b"][0]-s["a"][0])/s["length"]
            vy=(s["b"][1]-s["a"][1])/s["length"]
            if abs(ux*vy-uy*vx)<0.45:
                continue
            d=_point_segment_distance(p,s["a"],s["b"])
            if d<=eps:
                options.append((d,abs(s["length"]-10)/max(10,s["length"]),
                                s["id"],s))
        if not options:
            return None
        options.sort(key=lambda x:(x[0],x[1],x[2]))
        chosen=options[0][3]
        found.append({
            "vector":chosen["id"],
            "stroke_from":chosen["from"],
            "stroke_to":chosen["to"],
            "distance_svg":options[0][0],
        })
    # A single short stroke must never "witness" both ends of a line.
    if (found[0]["stroke_from"],found[0]["stroke_to"])==(
            found[1]["stroke_from"],found[1]["stroke_to"]):
        return None
    return found


def _geom_key(stroke):
    ends=(stroke["a"],stroke["b"])
    return tuple(sorted((tuple(round(q,4) for q in p) for p in ends)))


def associate_page(page):
    """Finite, source-backed, deliberately fail-closed annotation graph."""
    frame=_page_map(page)
    labels=_text_rects(page,frame)
    strokes=_strokes(page)
    witness_grid=_witness_grid(strokes)
    matches=[]
    rejected=[]
    claimed={}
    for label in labels:
        x0,y0,x1,y1=label["bbox_svg"]
        width=x1-x0; height=y1-y0
        middle=((x0+x1)/2,(y0+y1)/2)
        candidates=[]
        unique=set()
        for stroke in strokes:
            if stroke["length"]<max(20,1.15*width):
                continue
            perp,t=_perpendicular_to(middle,stroke["a"],stroke["b"])
            if not 0.12<=t<=0.88:
                continue
            if perp>min(36.0,max(9.0,2.8*height)):
                continue
            signature=_geom_key(stroke)
            if signature in unique:
                continue
            unique.add(signature)
            links=_endpoint_witnesses(stroke,strokes,height,witness_grid)
            if links is None:
                continue
            score=(perp/max(4,height)+1.25*abs(t-0.5))
            candidates.append((score,stroke,links,perp,t,signature))
        candidates.sort(key=lambda x:(x[0],x[1]["id"]))
        if not candidates:
            rejected.append({"source_label":label["source_label"],
                             "word_ids":label["word_ids"],
                             "reason":"no_two_endpoint_witnessed_line"})
            continue
        leading=candidates[0]
        if len(candidates)>1 and candidates[1][0]-leading[0] <= max(
                0.18,0.2*leading[0]):
            rejected.append({"source_label":label["source_label"],
                             "word_ids":label["word_ids"],
                             "reason":"ambiguous_dimension_line",
                             "competing_vectors":[leading[1]["id"],
                                                  candidates[1][1]["id"]]})
            continue
        _,line,links,perp,t,key=leading
        row={**label,"from":line["from"],"to":line["to"],
             "source_vector_ids":[line["id"],links[0]["vector"],links[1]["vector"]],
             "endpoint_witnesses":links,
             "annotation_axis_svg_units":line["length"],
             "label_line_perpendicular_svg_units":perp,
             "label_projection_fraction":t,
             "geometry_evidence":"both_endpoint_cross_strokes",
             "association_margin":(
                 candidates[1][0]-leading[0] if len(candidates)>1 else None)}
        matches.append(row)
        claimed.setdefault(key,[]).append(row)
    # Even if each text has a unique local best, two different labels
    # selecting the same dimensional stroke are NOT independent evidence.
    collisions={key for key,rows in claimed.items() if len(rows)>1}
    if collisions:
        for key in collisions:
            for m in claimed[key]:
                rejected.append({"source_label":m["source_label"],
                                 "word_ids":m["word_ids"],
                                 "reason":"multiple_labels_claim_same_dimension_line"})
        matches=[m for m in matches if _geom_key(
            {"a":_stroke_for_match(m,strokes)["a"],
             "b":_stroke_for_match(m,strokes)["b"]}) not in collisions]
    return {
        "kind":"pdf-dimension-callout-associations-v1",
        "page":page.get("page"),
        "matches":matches,"rejected":rejected,
        "dimension_word_count":len(labels),
        "source_stroke_count":len(strokes),
        "coordinate_frame":"PDF page pt ↔ native Poppler SVG viewBox, axis-aligned",
        "claim_of_engineering_approval":False,
        "warning":"A two-endpoint SVG witness only certifies local 2D annotation geometry; it does not certify source design intent, projection, correct print dimensions, 3D geometry or building-code compliance.",
    }


def _stroke_for_match(match,strokes):
    a,b=match["from"],match["to"]
    for s in strokes:
        if s["from"]==a and s["to"]==b:
            return s
    raise ValueError("Matched line cannot be resolved in original SVG source")


def auto_measure_page(page,targets,quantity,metre,calibration_model="affine_spd",
                      max_relative_residual=0.01):
    if calibration_model not in ("affine_spd","similarity"):
        raise ValueError("calibration_model must be affine_spd or similarity")
    links=associate_page(page)
    source=links["matches"]
    if len(source)<(4 if calibration_model=="affine_spd" else 2):
        raise ValueError("insufficient unambiguous two-endpoint dimensions; inspect rejected callouts and source strokes")
    # Registration is geometric alone: no target size is consulted.
    # For SPD, take the first distinct four-or-more directions (angle <~5deg
    # families discarded). A bad numerical anchor cannot be silently
    # optimized away after scale-fitting.
    refs=[];axes=[];label_texts=set()
    source_strokes=_strokes(page)
    for m in source:
        if m["source_label"] in label_texts:
            raise ValueError("duplicate source label text; disambiguate independently")
        label_texts.add(m["source_label"])
        if calibration_model=="affine_spd":
            targetline=_stroke_for_match(m,source_strokes)
            a,b=targetline["a"],targetline["b"]
            unit=((b[0]-a[0])/targetline["length"],
                  (b[1]-a[1])/targetline["length"])
            if any(abs(unit[0]*other[0]+unit[1]*other[1])>0.996
                   for other in axes):
                continue
            axes.append(unit)
        refs.append({
            "from":m["from"],"to":m["to"],
            "length":m["length"],"source_label":m["source_label"]})
        if len(refs)==64:
            break
    if len(refs)<(4 if calibration_model=="affine_spd" else 2):
        raise ValueError("insufficient independent source-verified dimension directions")
    spec={
        "schema":"hamon.pdf.measure.v1",
        "page":page.get("page"),
        "references":refs,
        "targets":targets,
        "calibration_model":calibration_model,
        "max_relative_residual":max_relative_residual,
    }
    result=hamon_scale.calibrate_inventory(page,spec,quantity,metre)
    result["annotation_binding"]="two_endpoint_source_witness"
    result["annotation_evidence"]=[{
        "source_label":m["source_label"],
        "bbox_pdf_pt":m["bbox_pdf_pt"],
        "from":m["from"],"to":m["to"],
        "endpoint_witnesses":m["endpoint_witnesses"],
    } for m in source if m["source_label"] in {
        r["source_label"] for r in refs}]
    result["annotation_rejected"]=links["rejected"]
    result["warning"] += (
        " Annotation matches are machine-qualified local geometry associations,"
        " not engineering approvals; a real expert must confirm drawing intent.")
    return result


def associate_pdf(pdf_file,page_number=1):
    inventory=hamon_drawing.scan_pdf(
        pdf_file,first=page_number,last=page_number,
        limit_paths=5000,limit_words=5000)
    result=associate_page(inventory["pages"][0])
    result["source_pdf_sha256"]=hashlib.sha256(Path(pdf_file).read_bytes()).hexdigest()
    return result


def auto_measure_pdf(pdf_file,spec_file,quantity,metre):
    source=Path(spec_file)
    if source.stat().st_size>500000:
        raise ValueError("auto measurement spec exceeds 500 KB")
    spec=json.loads(source.read_text())
    if not isinstance(spec,dict) or spec.get("schema")!="hamon.pdf.auto-measure.v1":
        raise ValueError("expected hamon.pdf.auto-measure.v1")
    pn=spec.get("page")
    if type(pn) is not int or not 1<=pn<=10000:
        raise ValueError("positive PDF page number required")
    scan=hamon_drawing.scan_pdf(pdf_file,first=pn,last=pn,
                               limit_paths=5000,limit_words=5000)
    out=auto_measure_page(
        scan["pages"][0],spec.get("targets"),quantity,metre,
        calibration_model=spec.get("calibration_model","affine_spd"),
        max_relative_residual=spec.get("max_relative_residual",0.01))
    out["source_pdf_sha256"]=hashlib.sha256(Path(pdf_file).read_bytes()).hexdigest()
    out["auto_spec_sha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
    return out
