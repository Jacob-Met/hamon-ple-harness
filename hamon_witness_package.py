"""Reconcile a dated commissioning case and write a Word witness package."""
from collections import Counter
from datetime import date
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED


def example():
    """A constructed record showing the required fields, not a commissioning determination."""
    def ref(source):
        return {"source": source, "date": "2025-01-02"}
    return {"case": "Example municipal case", "as_of": "2025-01-02",
            "scope_roster": ref("Illustrative batch roster"), "case_snapshot": ref("Illustrative case export"),
            "arrays": [{"id": "Example array", "scope": "in", "scope_basis": ref("Illustrative scope memo"),
                        "installed_equipment": "Example make and exact installed variant",
                        "installation_basis": ref("Illustrative installation record"),
                        "carriage": "not carried", "carriage_basis": ref("Illustrative dated list and exclusions"),
                        "release_request": "hold", "release_basis": ref("Illustrative open disposition"),
                        "tests": [{"name": "Illustrative witness activity", "basis": ref("Illustrative procedure"),
                                   "performer": "Designated test provider", "witness": "Designated witness",
                                   "prerequisites": "Approved procedure and designated parties on file"}]}],
            "scheduling_items": [{"id": "EX-1", "case": "Example municipal case", "array": "Example array",
                                  "standing": "yes", "basis": ref("Illustrative case item EX-1")}]}


def text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing text field {key!r}")
    if any(ord(c) < 32 and c not in "\t\n\r" or 0xD800 <= ord(c) <= 0xDFFF or ord(c) in (0xFFFE, 0xFFFF) for c in value):
        raise ValueError(f"invalid XML character in {key!r}")
    return value.strip()


def rows(obj, key):
    result = obj.get(key)
    if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
        raise ValueError(f"expected records in {key!r}")
    return result


def evidence(row, field, cutoff):
    ref = row.get(field)
    if not isinstance(ref, dict):
        raise ValueError(f"missing evidence for {field}")
    observed = date.fromisoformat(text(ref, "date"))
    if observed > cutoff:
        raise ValueError(f"{field} evidence postdates the case cutoff")
    return f'{text(ref, "source")} ({observed.isoformat()})'


def paragraphs(spec):
    cutoff = date.fromisoformat(text(spec, "as_of"))
    case = text(spec, "case")
    arrays = rows(spec, "arrays")
    ids = [text(a, "id") for a in arrays]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("expected a nonempty roster of distinct arrays")
    scope_source = evidence(spec, "scope_roster", cutoff)
    case_source = evidence(spec, "case_snapshot", cutoff)
    if spec["case_snapshot"]["date"] != cutoff.isoformat():
        raise ValueError("case snapshot must represent the as-of date")

    needed = set()
    releases = []
    sections = []
    for array, ident in zip(arrays, ids):
        sections.append(("Heading1", ident))
        scope = text(array, "scope")
        if scope not in ("in", "out"):
            raise ValueError(f"{ident}: unresolved scope")
        sections.append(("Normal", f'Scope: {scope}. {evidence(array, "scope_basis", cutoff)}'))
        if scope == "out":
            continue
        equipment = text(array, "installed_equipment")
        carriage = text(array, "carriage")
        if carriage not in ("carried", "not carried"):
            raise ValueError(f"{ident}: unresolved certified equipment determination")
        sections.extend([
            ("Normal", f'Installed equipment: {equipment}. {evidence(array, "installation_basis", cutoff)}'),
            ("Normal", f'Certified Equipment List: {carriage}. {evidence(array, "carriage_basis", cutoff)}'),
        ])
        release = text(array, "release_request")
        if release not in ("supported", "hold"):
            raise ValueError(f"{ident}: unresolved release request")
        sections.append(("Normal", f'Release request: {release}. {evidence(array, "release_basis", cutoff)}'))
        if release == "supported":
            releases.append(ident)
        if carriage == "not carried":
            needed.add(ident)
            tests = rows(array, "tests")
            if not tests:
                raise ValueError(f"{ident}: missing witness scope")
            for test in tests:
                sections.extend([
                    ("Heading2", text(test, "name")),
                    ("Normal", f'Test basis: {evidence(test, "basis", cutoff)}'),
                    ("Normal", f'Performer: {text(test, "performer")}. Witness: {text(test, "witness")}.'),
                    ("Normal", f'Required before scheduling: {text(test, "prerequisites")}'),
                ])

    observed = []
    item_ids = set()
    for item in rows(spec, "scheduling_items"):
        if text(item, "case") != case:
            raise ValueError("scheduling item belongs to a different case")
        item_id = text(item, "id")
        if item_id in item_ids:
            raise ValueError(f"duplicate scheduling item {item_id}")
        item_ids.add(item_id)
        state = text(item, "standing")
        if state not in ("yes", "no"):
            raise ValueError("unresolved scheduling item status")
        source = evidence(item, "basis", cutoff)
        if state == "yes":
            observed.append((text(item, "array"), item_id, source))
    counts = Counter(ident for ident, _, _ in observed)
    issues = {"missing": sorted(needed - counts.keys()), "unexpected": sorted(counts.keys() - needed),
              "duplicate": sorted(ident for ident, count in counts.items() if count > 1)}
    result = {"case": case, "as_of": cutoff.isoformat(), "schedule_matches": not any(issues.values()),
              "schedule_issues": issues, "witness_required": sorted(needed),
              "supported_release_requests": sorted(releases)}
    intro = [("Title", "Municipal witness test package"),
             ("Normal", f"Case {case} as of {cutoff.isoformat()}. Standards Engineering retains waiver disposition."),
             ("Heading1", "Batch determination"),
             ("Normal", "Witness scheduling required: " + (", ".join(sorted(needed)) or "none") + "."),
             ("Normal", "Release requests supported by the cited records: " + (", ".join(sorted(releases)) or "none") + "."),
             ("Normal", f"Scope roster: {scope_source}. Case snapshot: {case_source}.")]
    schedule = [("Heading1", "Standing scheduling items")]
    schedule.extend(("Normal", f"{ident} — {item_id}. {source}") for ident, item_id, source in sorted(observed))
    if not observed:
        schedule.append(("Normal", "The case snapshot lists no standing scheduling items."))
    return result, intro + sections + schedule


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"


def element(parent, name, **kwargs):
    return ET.SubElement(parent, "{"+W+"}"+name, {"{"+W+"}"+k: v for k, v in kwargs.items()})


def write_docx(path, content):
    ET.register_namespace("w", W)
    doc = ET.Element("{"+W+"}document")
    body = element(doc, "body")
    for style, value in content:
        paragraph = element(body, "p")
        element(element(paragraph, "pPr"), "pStyle", val=style)
        element(element(paragraph, "r"), "t").text = value
    section = element(body, "sectPr")
    element(section, "pgSz", w="12240", h="15840")
    element(section, "pgMar", top="1100", bottom="1100", left="1200", right="1200")
    styles = ET.Element("{"+W+"}styles")
    for name, label, size in (("Normal","Normal",22),("Title","Title",38),
                               ("Heading1","heading 1",29),("Heading2","heading 2",24)):
        style = element(styles, "style", type="paragraph", styleId=name)
        element(style, "name", val=label)
        props = element(style, "pPr")
        if name != "Normal":
            element(props, "keepNext")
        element(props, "spacing", before="120" if name != "Normal" else "0", after="80")
        run = element(style, "rPr")
        element(run, "rFonts", ascii="Calibri", hAnsi="Calibri")
        if name != "Normal":
            element(run, "b")
        element(run, "sz", val=str(size))
    types = ET.Element("Types", xmlns=CT)
    ET.SubElement(types, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
    ET.SubElement(types, "Default", Extension="xml", ContentType="application/xml")
    for part, kind in (("document","document.main"),("styles","styles")):
        ET.SubElement(types, "Override", PartName=f"/word/{part}.xml",
                      ContentType=f"application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml")
    rootrel, docrel = (ET.Element("Relationships", xmlns=REL) for _ in range(2))
    ET.SubElement(rootrel, "Relationship", Id="rId1", Type=OFFICE+"/officeDocument", Target="word/document.xml")
    ET.SubElement(docrel, "Relationship", Id="rId1", Type=OFFICE+"/styles", Target="styles.xml")
    with NamedTemporaryFile(dir=path.parent) as tmp:
        with ZipFile(tmp, "w", ZIP_DEFLATED) as z:
            for member, node in (("[Content_Types].xml", types),("_rels/.rels", rootrel),
                                 ("word/document.xml", doc),("word/styles.xml", styles),
                                 ("word/_rels/document.xml.rels", docrel)):
                z.writestr(member, ET.tostring(node, encoding="utf-8", xml_declaration=True))
        tmp.flush()
        os.link(tmp.name, path)  # Publish a complete new file without replacing an earlier package.


def witness_package(file, output=None):
    spec = json.loads(Path(file).read_text())
    if not isinstance(spec, dict):
        raise ValueError("expected a witness-package object")
    result, content = paragraphs(spec)
    if output is not None and result["schedule_matches"]:
        write_docx(Path(output), content)
        result["document"] = str(output)
    return result

def match_installation(pages, catalog):
    """Report an exact equipment-row join only for an explicit installed-model field."""
    import re
    claims = []
    pattern = re.compile(r"^\s*(?:inverter|power conversion equipment|PCS)\s+model\b(?P<role>\s*\([^)]*\))?(?:\s*:\s*|\s{2,})(?P<name>.+)$", re.I)
    for page, content in enumerate(pages.split("\f"), 1):
        for line, raw in enumerate(content.splitlines(), 1):
            found = pattern.match(raw)
            if not found: continue
            role = " ".join((found["role"] or "").split()).casefold()
            value = re.split(r"\s+·\s+",found["name"],maxsplit=1)[0].strip()
            claims.append({"role":"installed" if role=="(as installed)" else "reference",
                           "text":raw.strip(),"equipment":value,"page":page,"line":line})
    installed = {" ".join(c["equipment"].split()).casefold() for c in claims if c["role"]=="installed"}
    matches = []
    if len(installed)>1: status="conflicting_installation_evidence"
    elif not installed: status="installation_unresolved"
    else:
        matches = catalog.get(next(iter(installed)), [])
        status = "matched_installed_equipment" if len(matches)==1 else "ambiguous_catalog_rows" if matches else "no_exact_catalog_row"
    return {"status":status, "evidence":claims, "catalog_rows":matches}


