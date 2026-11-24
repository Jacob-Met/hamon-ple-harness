"""Finish source-backed Word and ZIP deliverables using a participant's fresh JSON draft."""
import hashlib
import math
import io
import json
import os
from pathlib import Path
import re
import tempfile
from xml.dom import minidom
from zipfile import ZipFile, ZIP_DEFLATED
import posixpath
from archive_checkpoint import verify_typed_members
from hamon_witness_package import example, paragraphs, write_docx

TASK_ROOTS = (Path("/filesystem"), Path("/app"))


def safe_file(path):
    return (path.is_absolute() and any(path.is_relative_to(root) for root in TASK_ROOTS)
            and ".." not in path.parts
            and all(not parent.is_symlink() for parent in (path, *path.parents))
            and path.is_file())

def workbook(template, changes):
    """Patch existing template cells while preserving all other ZIP parts and cell styles."""
    location = Path(template)
    if not safe_file(location) or location.suffix.lower() != ".xlsx":
        raise ValueError("expected a task-local XLSX template")
    if not isinstance(changes, dict) or not changes:
        raise ValueError("expected per-sheet template cell values")
    def children(node, name):
        return [child for child in node.childNodes
                if child.nodeType == child.ELEMENT_NODE and child.localName == name]
    def address(ref):
        m = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]{0,6})", ref)
        if not m: raise ValueError("invalid A1 coordinate")
        col = 0
        for c in m[1]: col = col*26 + ord(c)-64
        row = int(m[2])
        if col > 16384 or row > 1048576: raise ValueError("cell outside worksheet")
        return row, col
    output=io.BytesIO()
    with ZipFile(location) as src, ZipFile(output,"w",ZIP_DEFLATED) as dest:
        if len(set(src.namelist())) != len(src.namelist()):
            raise ValueError("duplicate template part")
        wb=minidom.parseString(src.read("xl/workbook.xml"))
        rel=minidom.parseString(src.read("xl/_rels/workbook.xml.rels"))
        table={r.getAttribute("Id"):r.getAttribute("Target")
               for r in rel.getElementsByTagNameNS("*","Relationship")
               if r.getAttribute("TargetMode") != "External" and r.getAttribute("Type").endswith("/worksheet")}
        files={}
        for sheet in wb.getElementsByTagNameNS("*","sheet"):
            name=sheet.getAttribute("name")
            if name not in changes:continue
            rid=next((a.value for a in sheet.attributes.values() if a.localName=="id" and a.namespaceURI),None)
            target=table[rid]
            part=posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join("xl",target))
            if part.startswith("../") or part not in src.namelist() or part in files: raise ValueError("invalid worksheet relationship")
            files[part]=name
        if set(files.values()) != set(changes): raise ValueError("unknown worksheet name")
        for item in src.infolist():
            data=src.read(item)
            if item.filename in files:
                sheetname=files[item.filename];patch=changes[sheetname]
                if not isinstance(patch,dict) or not patch:raise ValueError("empty worksheet update")
                doc=minidom.parseString(data);root=doc.documentElement
                rows=children(children(root,"sheetData")[0],"row")
                cells={cell.getAttribute("r"):cell for row in rows for cell in children(row,"c")}
                for ref,value in patch.items():
                    number,column=address(ref)
                    if ref not in cells:
                        area=children(root,"sheetData")[0]
                        by_number={int(row.getAttribute("r")):row for row in children(area,"row")}
                        row=by_number.get(number)
                        def tag(name):
                            prefix=root.prefix+":" if root.prefix else ""
                            return doc.createElementNS(root.namespaceURI,prefix+name)
                        if row is None:
                            row=tag("row");row.setAttribute("r",str(number))
                            later=next((a for n,a in by_number.items() if n>number),None)
                            area.insertBefore(row,later)
                        cell=tag("c");cell.setAttribute("r",ref)
                        later=next((child for child in row.childNodes if child.nodeType==child.ELEMENT_NODE and
                                    (child.localName!="c" or address(child.getAttribute("r"))[1]>column)),None)
                        row.insertBefore(cell,later);cells[ref]=cell
                    cell=cells[ref]
                    if children(cell, "f"):
                        raise ValueError("refusing to replace an owner template formula")
                    if not isinstance(value,(str,int,float,bool)) or isinstance(value,float) and not math.isfinite(value):
                        raise ValueError("expected a finite literal cell value")
                    if isinstance(value,str) and not value.strip():raise ValueError("empty text cell")
                    for child in list(cell.childNodes):
                        if child.nodeType==child.ELEMENT_NODE and child.localName in ("f","v","is"):cell.removeChild(child)
                    def el(parent,local):
                        qname=(cell.prefix+":" if cell.prefix else "")+local
                        node=doc.createElementNS(cell.namespaceURI,qname);parent.appendChild(node);return node
                    if isinstance(value,str):
                        cell.setAttribute("t","inlineStr")
                        node=el(el(cell,"is"),"t")
                        node.setAttributeNS("http://www.w3.org/XML/1998/namespace","xml:space","preserve")
                        node.appendChild(doc.createTextNode(value))
                    else:
                        cell.setAttribute("t","b" if isinstance(value,bool) else "n")
                        el(cell,"v").appendChild(doc.createTextNode(str(int(value)) if isinstance(value,bool) else str(value)))
                dims=children(root,"dimension")
                if dims:
                    # Dimensions are an optional optimization. Remove a stale
                    # range after appending rows or cells so readers infer it.
                    root.removeChild(dims[0])
                data=doc.toxml(encoding="utf-8")
            dest.writestr(item,data)
    return output.getvalue()

def render(draft, target, names, fingerprint):
    draft, target = Path(draft), Path(target)
    if (draft != target.with_name(target.name + ".draft.json") or not safe_file(draft)
            or target.exists() or target.is_symlink()):
        raise ValueError("expected a fresh sibling draft and a missing destination")
    raw = draft.read_bytes()
    if hashlib.sha256(str(len(raw)).encode() + raw).hexdigest() != fingerprint:
        raise ValueError("draft changed after observation")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("invalid archive contract")
    manifest=json.loads(raw)
    if not isinstance(manifest,dict):raise ValueError("expected an authored object")
    if target.suffix.lower()==".docx":
        if names or manifest == example():
            raise ValueError("supply a project witness record")
        result, content = paragraphs(manifest)
        if not result["schedule_matches"]:raise ValueError("case and schedule do not reconcile")
        write_docx(target,content)
        return {"state":"executed"}
    members=manifest.get("members")
    if (target.suffix.lower()!=".zip" or not names or not isinstance(members,dict)
            or len(names)!=len(set(names)) or set(names)!=set(members)):
        raise ValueError("draft must supply every declared archive member")
    with tempfile.NamedTemporaryFile(dir=target.parent) as tmp:
        with ZipFile(tmp,"w",ZIP_DEFLATED) as out:
            for name in names:
                if Path(name).name!=name or name in (".",".."):raise ValueError("invalid member")
                row=members[name];suffix=Path(name).suffix.lower()
                if not isinstance(row,dict):raise ValueError("invalid authored member")
                if suffix==".json" and set(row)=={"json"} and isinstance(row["json"],(dict,list)) and row["json"]:
                    data=json.dumps(row["json"],allow_nan=False,ensure_ascii=False,indent=2).encode()
                elif suffix in (".md",".txt",".csv") and set(row)=={"text"} and isinstance(row["text"],str) and row["text"].strip():
                    data=row["text"].encode()
                elif suffix==".xlsx" and set(row)=={"template","cells"}:
                    data=workbook(row["template"],row["cells"])
                else:raise ValueError("unsupported or empty member: "+name)
                out.writestr(name,data)
        tmp.flush()
        with ZipFile(tmp) as z:
            if z.testzip() is not None:raise ValueError("archive checksum")
            verify_typed_members(z,names)
        os.link(tmp.name,target)
    return {"state":"executed"}
