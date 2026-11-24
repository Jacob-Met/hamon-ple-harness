"""Office package relationships, content types and XML parts. Standard library only."""
import pathlib
import re
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


_SHEET_ID_ATTRIBUTES = {"{" + ns + "}id" for ns in _DOCUMENT_RELS}
_SHEET_KINDS = {
    namespace + "/" + kind: (
        f"application/vnd.openxmlformats-officedocument.spreadsheetml.{kind}+xml",
        {"{" + ns + "}" + kind for ns in _SPREADSHEET_NS},
    )
    for namespace in _DOCUMENT_RELS for kind in ("worksheet", "chartsheet", "dialogsheet")
}
for _relation, _kind in (("xlMacrosheet", "macrosheet"), ("xlIntlMacrosheet", "intlmacrosheet")):
    _SHEET_KINDS["http://schemas.microsoft.com/office/2006/relationships/" + _relation] = (
        f"application/vnd.ms-excel.{_kind}+xml", {"{http://schemas.microsoft.com/office/excel/2006/main}macrosheet"},
    )


class _WorkbookSheets:
    """Keep only sheet relationship IDs while parsing the whole workbook."""
    def __init__(self):
        self.path = []
        self.ids = []

    def start(self, tag, attrs):
        if not self.path:
            if tag not in {"{" + ns + "}workbook" for ns in _SPREADSHEET_NS}:
                raise ValueError("incorrect_workbook_root")
            ns = tag.removesuffix("workbook")
            self.sheet_path = [tag, ns + "sheets", ns + "sheet"]
        self.path.append(tag)
        if self.path == self.sheet_path:
            keys = _SHEET_ID_ATTRIBUTES.intersection(attrs)
            if len(keys) != 1:
                raise ValueError("missing_or_ambiguous_sheet_relationship")
            self.ids.append(attrs[keys.pop()])

    def end(self, tag):
        self.path.pop()

    def close(self):
        if not self.ids:
            raise ValueError("workbook_has_no_sheets")
        return self.ids


def _workbook_sheet_parts(archive, main, members, defaults, overrides):
    from posixpath import basename, dirname, join

    with archive.open(main) as stream:
        sheet_ids = ET.parse(stream, parser=ET.XMLParser(target=_WorkbookSheets())).getroot()
    rel_name = join(dirname(main), "_rels", basename(main) + ".rels").lower()
    if rel_name not in members or _opc_content_type(defaults, overrides, rel_name) != _RELATIONSHIPS_TYPE:
        raise ValueError("missing_or_incorrect_workbook_relationships")
    relationships = _xlsx_relationships(archive, members[rel_name], limit=None)
    expected = {}
    for rid in sheet_ids:
        rel = relationships.get(rid)
        if rel is None or rel.get("TargetMode", "Internal") != "Internal":
            raise ValueError("missing_or_external_sheet_relationship")
        kind = _SHEET_KINDS.get(rel.get("Type"))
        target = _xlsx_target(main, rel.get("Target", "")).lower()
        if (kind is None or target not in members or target in expected or
                _opc_content_type(defaults, overrides, target) != kind[0]):
            raise ValueError("missing_or_incorrect_sheet_part")
        expected[target] = kind[1]
    return expected, rel_name


def check_office_package(archive, suffix):
    """Validate the main part, workbook sheet graph and all parts declared as XML."""
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
    elif suffix in (".xlsx", ".xlsm"):
        expected_type = ("application/vnd.ms-excel.sheet.macroenabled.main+xml" if suffix == ".xlsm"
                         else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")
    else:
        raise ValueError(f"unsupported Office suffix: {suffix}")
    if _opc_content_type(defaults, overrides, main) != expected_type:
        raise ValueError("incorrect_word_main_content_type" if suffix == ".docx" else "incorrect_workbook_content_type")

    checked = {"[content_types].xml", "_rels/.rels"}
    if suffix == ".docx":
        expected_roots = {main.lower(): _WORD_DOCUMENT_TYPES}
    else:
        expected_roots, workbook_rels = _workbook_sheet_parts(archive, main, members, defaults, overrides)
        checked.update((main.lower(), workbook_rels))

    for key, name in members.items():
        if key in checked and key not in expected_roots:
            continue
        mime = _opc_content_type(defaults, overrides, name).partition(";")[0].strip()
        if mime not in ("application/xml", "text/xml") and not mime.endswith("+xml"):
            continue
        with archive.open(name) as stream:
            root_tag = ET.parse(stream, parser=ET.XMLParser(target=_RootTag())).getroot()
        if key in expected_roots and root_tag not in expected_roots[key]:
            raise ValueError("incorrect_word_document_root" if suffix == ".docx" else "incorrect_sheet_root")


def workbook_parts(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate OOXML package member")
    if "_rels/.rels" not in names and "xl/workbook.xml" not in names:
        # Earlier callers also audit standalone worksheet fragments.
        parts = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        if not parts:
            raise ValueError("No workbook or standalone worksheet parts")
        return None, [{"part": n, "kind": "worksheet", "name": None} for n in parts]
    if "_rels/.rels" not in names:
        workbook = "xl/workbook.xml"
    else:
        office = [r for r in _xlsx_relationships(archive, "_rels/.rels").values()
                  if r.get("Type") in {base + "/officeDocument" for base in _DOCUMENT_RELS}]
        if len(office) != 1 or office[0].get("TargetMode", "Internal") != "Internal":
            raise ValueError("Expected exactly one internal workbook relationship")
        workbook = _xlsx_target("", office[0].get("Target", ""))
    root = _xlsx_xml(archive, workbook, 2_000_000)
    ns = root.tag.removesuffix("workbook")
    if ns not in {"{" + v + "}" for v in _SPREADSHEET_NS}:
        raise ValueError(f"Invalid spreadsheet workbook: {workbook}")
    sheets = root.find(ns + "sheets")
    if sheets is None:
        raise ValueError("Workbook is missing its sheets collection")
    source = pathlib.PurePosixPath(workbook)
    rels = _xlsx_relationships(archive, str(source.parent / "_rels" / (source.name + ".rels")))
    result = []
    for sheet in sheets.findall(ns + "sheet"):
        ids = [sheet.get("{" + prefix + "}id") for prefix in _DOCUMENT_RELS]
        ids = [rid for rid in ids if rid is not None]
        if len(ids) != 1 or ids[0] not in rels:
            raise ValueError(f"Unresolved relationship for sheet {sheet.get('name')!r}")
        rel = rels[ids[0]]
        if rel.get("TargetMode", "Internal") != "Internal":
            raise ValueError(f"Workbook sheet {sheet.get('name')!r} has an external target")
        base, _, kind = rel.get("Type", "").rpartition("/")
        if base not in _DOCUMENT_RELS:
            kind = rel.get("Type", "unknown")
        result.append({"name": sheet.get("name"), "sheet_id": sheet.get("sheetId"),
                       "state": sheet.get("state", "visible"), "kind": kind,
                       "part": _xlsx_target(workbook, rel.get("Target", ""))})
    return workbook, result



def _table_text(element, ns):
    """Rich or plain spreadsheet text, excluding phonetic annotations."""
    if element is None:
        return ""
    return "".join(child.text or "" for child in element.findall(ns + "t")) + "".join(
        run.findtext(ns + "t", "") for run in element.findall(ns + "r"))


def table_signatures(archive):
    """Return bounded structural fingerprints for visible row-one worksheet tables."""
    import hashlib
    import json
    from posixpath import basename, dirname, join

    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()

    if sum(info.file_size for info in archive.infolist()) > 8_000_000:
        raise OverflowError("table inspection budget")
    workbook, sheets = workbook_parts(archive)
    if not workbook:
        raise ValueError("table requires a complete workbook")
    rels = _xlsx_relationships(archive, join(dirname(workbook), "_rels", basename(workbook) + ".rels"))
    string_parts = [rel for rel in rels.values()
                    if rel.get("Type") in {r + "/sharedStrings" for r in _DOCUMENT_RELS}]
    strings = []
    if string_parts:
        rel = string_parts[0]
        if len(string_parts) != 1 or rel.get("TargetMode", "Internal") != "Internal":
            raise ValueError("ambiguous shared strings")
        root = _xlsx_xml(archive, _xlsx_target(workbook, rel["Target"]), 8_000_000)
        ns = root.tag.removesuffix("sst")
        strings = [_table_text(si, ns) for si in root.findall(ns + "si")]
    tables = {}
    for sheet in sheets:
        if sheet["kind"] != "worksheet" or sheet["state"] != "visible":
            continue
        root = _xlsx_xml(archive, sheet["part"], 8_000_000)
        ns = root.tag.removesuffix("worksheet")
        header, formulas, records = {}, set(), set()
        previous_row = 0
        for row in root.findall(ns + "sheetData/" + ns + "row"):
            number = int(row.get("r", previous_row + 1))
            if number <= previous_row:
                raise ValueError("ambiguous worksheet rows")
            previous_row = number
            values = {}
            for cell in row.findall(ns + "c"):
                ref = cell.get("r", "")
                match = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]*)", ref)
                if not match:
                    raise OverflowError("implicit cell positions unsupported for table comparison")
                if int(match[2]) != number or match[1] in values:
                    raise ValueError("ambiguous worksheet cells")
                formula = cell.find(ns + "f")
                kind = cell.get("t")
                value = cell.findtext(ns + "v", "")
                if formula is not None and ((formula.text or "").strip() or formula.get("t") == "shared"):
                    formulas.add(ref)
                    value = "=" + (formula.text or "")
                elif kind == "s" and value:
                    if not value.isdigit() or int(value) >= len(strings):
                        raise ValueError("invalid shared string index")
                    value = strings[int(value)]
                elif kind == "inlineStr":
                    value = _table_text(cell.find(ns + "is"), ns)
                elif kind == "e":
                    value = ""
                if value.strip():
                    values[match[1]] = value.strip()
            if number == 1:
                header = values
            elif len(header) >= 2:
                record = sorted((k, v) for k, v in values.items() if k in header)
                if len(record) >= 2:
                    records.add(fingerprint(record))
        if len(header) >= 2:
            key = fingerprint([sheet["name"], sorted(header.items())])
            if key in tables:
                raise ValueError("duplicate table identity")
            tables[key] = {"formulas": sorted(formulas), "rows": sorted(records)}
    return tables


def discover_table_contracts(root, roles):
    """Freeze an unambiguous matching input before participant execution."""
    import os
    import zipfile

    results = {name: {"unavailable": True} for name in roles}
    candidates = {name: [] for name in roles}
    if not root.is_dir() or root == root.parent or root.is_symlink():
        return results
    def fail(exc):
        raise exc
    def words(stem):
        return re.findall(r"[a-z0-9]+", stem.casefold())
    visited = 0
    try:
        for directory, folders, files in os.walk(root, onerror=fail):
            visited += len(files) + len(folders)
            if visited > 4096:
                return results
            folders[:] = [d for d in folders if not d.startswith(".") and
                          not (pathlib.Path(directory) / d).is_symlink()]
            for filename in files:
                path = pathlib.Path(directory) / filename
                tokens = words(path.stem)
                if "template" not in tokens or path.is_symlink():
                    continue
                for name in roles:
                    out = pathlib.PurePosixPath(name)
                    role = words(out.stem)
                    if path.suffix.lower() == out.suffix.lower() and any(
                            tokens[i:i+len(role)] == role for i in range(len(tokens))):
                        candidates[name].append(path)
    except OSError:
        return results
    for name, matches in candidates.items():
        if len(matches) != 1:
            continue
        try:
            source = matches[0]
            if not source.is_file() or source.stat().st_size > 2_000_000:
                continue
            with zipfile.ZipFile(source) as book:
                tables = table_signatures(book)
            if len(tables) == 1:
                key, table = next(iter(tables.items()))
                if len(table["formulas"]) <= 128 and len(table["rows"]) <= 128:
                    results[name] = {"key": key, **table, "minimum": roles[name]}
        except (ValueError, OverflowError, LookupError, OSError, ET.ParseError, zipfile.BadZipFile):
            continue
    return results


def verify_table_contract(archive, contract):
    table = table_signatures(archive).get(contract["key"])
    if table is None:
        raise ValueError("template worksheet or ordered headers differ")
    if not set(contract["formulas"]).issubset(table["formulas"]):
        raise ValueError("template formula cells missing")
    if len(table["rows"]) < contract["minimum"] or not (set(table["rows"]) - set(contract["rows"])):
        raise ValueError("insufficient populated template rows")
