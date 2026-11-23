"""Office package relationships, content types and XML parts. Standard library only."""
import xml.etree.ElementTree as ET


_SPREADSHEET_NS = {
    "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "http://purl.oclc.org/ooxml/spreadsheetml/main",
}
_DOCUMENT_RELS = {
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "http://purl.oclc.org/ooxml/officeDocument/relationships",
}
_PACKAGE_RELS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _xlsx_xml(archive, name, limit):
    try:
        part = archive.getinfo(name)
    except KeyError as exc:
        raise ValueError(f"Missing OOXML part: {name}") from exc
    if limit is not None and part.file_size > limit:
        raise ValueError(f"OOXML part exceeds {limit} bytes: {name}")
    return ET.fromstring(archive.read(part))


def _xlsx_target(source, target):
    """Resolve an internal OPC part URI without consulting a filesystem or network."""
    from urllib.parse import urlsplit
    uri = urlsplit(target)
    if (not target or uri.scheme or uri.netloc or uri.query or uri.fragment
            or "\\" in target or target.startswith("//")):
        raise ValueError(f"Invalid internal OOXML part target: {target!r}")
    segments = [] if target.startswith("/") else source.split("/")[:-1]
    for segment in uri.path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not segments:
                raise ValueError(f"OOXML part target leaves package: {target!r}")
            segments.pop()
        else:
            segments.append(segment)
    if not segments:
        raise ValueError("Empty OOXML part target")
    return "/".join(segments)


def _xlsx_relationships(archive, name, limit=2_000_000):
    root = _xlsx_xml(archive, name, limit)
    if root.tag != _PACKAGE_RELS + "Relationships":
        raise ValueError(f"Invalid OOXML relationship part: {name}")
    result = {}
    for rel in root.findall(_PACKAGE_RELS + "Relationship"):
        rid = rel.get("Id")
        if not rid or rid in result:
            raise ValueError(f"Missing or repeated relationship Id in {name}")
        result[rid] = rel
    return result


_CONTENT_TYPES_NS = "{http://schemas.openxmlformats.org/package/2006/content-types}"
_WORD_DOCUMENT_TYPES = {
    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document",
    "{http://purl.oclc.org/ooxml/wordprocessingml/main}document",
}
_WORD_MAIN_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
_RELATIONSHIPS_TYPE = "application/vnd.openxmlformats-package.relationships+xml"


def _content_type_index(types):
    defaults, overrides = {}, {}
    for tag, attribute, table in (("Default", "Extension", defaults),
                                  ("Override", "PartName", overrides)):
        for entry in types.findall(_CONTENT_TYPES_NS + tag):
            key = entry.get(attribute, "").lower()
            if key in table:
                raise ValueError("ambiguous_content_type")
            table[key] = entry.get("ContentType", "").lower()
    return defaults, overrides


def _opc_content_type(defaults, overrides, name):
    key = name.lower()
    _, dot, extension = key.rsplit("/", 1)[-1].rpartition(".")
    return overrides.get("/" + key, defaults.get(extension, "") if dot else "")


class _RootTag:
    """Keep only the root tag while Expat checks the complete XML stream."""
    tag = None

    def start(self, tag, attrs):
        if self.tag is None:
            self.tag = tag

    def close(self):
        return self.tag


def check_office_package(archive, suffix):
    """Validate the main relationship and all parts declared as XML."""
    members = {}
    for item in archive.infolist():
        if item.is_dir():
            continue
        key = item.filename.lower()
        if key in members:
            raise ValueError("duplicate_part")
        members[key] = item.filename
    if "[content_types].xml" not in members or "_rels/.rels" not in members:
        raise ValueError("missing_package_metadata")
    types = _xlsx_xml(archive, members["[content_types].xml"], None)
    if types.tag != _CONTENT_TYPES_NS + "Types":
        raise ValueError("invalid_content_types_root")
    defaults, overrides = _content_type_index(types)
    if _opc_content_type(defaults, overrides, "_rels/.rels") != _RELATIONSHIPS_TYPE:
        raise ValueError("incorrect_relationships_content_type")
    office = {ns + "/officeDocument" for ns in _DOCUMENT_RELS}
    relationships = _xlsx_relationships(archive, members["_rels/.rels"], limit=None)
    starts = [rel for rel in relationships.values() if rel.get("Type") in office]
    if len(starts) != 1 or starts[0].get("TargetMode", "Internal") != "Internal":
        raise ValueError("expected_internal_main_relationship")
    name = _xlsx_target("", starts[0].get("Target", ""))
    main = members.get(name.lower())
    if main is None:
        raise ValueError("missing_main_part")
    if suffix == ".docx":
        expected_type = _WORD_MAIN_TYPE
        roots = _WORD_DOCUMENT_TYPES
    elif suffix in (".xlsx", ".xlsm"):
        expected_type = ("application/vnd.ms-excel.sheet.macroenabled.main+xml" if suffix == ".xlsm"
                         else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")
        roots = {"{" + ns + "}workbook" for ns in _SPREADSHEET_NS}
    else:
        raise ValueError(f"unsupported Office suffix: {suffix}")
    if _opc_content_type(defaults, overrides, main) != expected_type:
        raise ValueError("incorrect_word_main_content_type" if suffix == ".docx" else "incorrect_workbook_content_type")

    # The two metadata parts were already parsed through EOF above.
    for key, name in members.items():
        if key in ("[content_types].xml", "_rels/.rels"):
            continue
        mime = _opc_content_type(defaults, overrides, name).partition(";")[0].strip()
        if mime not in ("application/xml", "text/xml") and not mime.endswith("+xml"):
            continue
        with archive.open(name) as stream:
            root_tag = ET.parse(stream, parser=ET.XMLParser(target=_RootTag())).getroot()
        if name == main and root_tag not in roots:
            raise ValueError("incorrect_word_document_root" if suffix == ".docx" else "incorrect_workbook_root")
