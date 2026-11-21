"""Route available engineering evidence to the declared model before its first turn."""
import asyncio
import json
from pathlib import PurePosixPath
import shlex

TOOL = "/tmp/hamon-engineering-toolkit.py"
GUIDE = "/tmp/hamon-engineering-toolkit-guide.txt"
FAMILIES = {
    ".pdf": "pdf-words, pdf-vector, pdf-window, pdf-measure, pdf-dimension-links, pdf-auto-measure",
    ".ifc": "ifc-broadphase, ifc-narrowphase",
    ".dxf": "dxf-dimensions",
    ".xlsx": "xlsx-audit",
    ".xlsm": "xlsx-audit",
}


async def prepare(session, discovered, full_guide):
    """Keep full usage nearby; proactively verify one bounded workbook when unambiguous."""
    if (left := session.time_left()) is not None and left < 35:
        return full_guide
    available = discovered.get("by_extension", {})
    families = dict.fromkeys(commands for ext, commands in FAMILIES.items() if available.get(ext))
    # Mixed CAD and workbook tasks need the full cross-format interpretation guidance.
    if len(families) > 1:
        return full_guide
    guide_bytes = full_guide.encode()
    try:
        size = await asyncio.wait_for(session.write_file(GUIDE, guide_bytes), timeout=4)
        if size != len(guide_bytes):
            return full_guide
    except (TimeoutError, OSError, RuntimeError, ValueError, KeyError):
        return full_guide
    guide = (f"\nEngineering CLI: python3 {TOOL}. Use quantity, check, calc-pack for dimensional "
             f"calculations. Additional commands for detected files: {', '.join(families) or 'none'}. "
             f"Read {GUIDE} for schemas and worked examples. PDF drawing scale and IFC physical clearance require independent verification.")
    sheets = [item for item in discovered["files"] if isinstance(item, dict)
              and item.get("ext") in (".xlsm", ".xlsx")]
    if sum(available.get(ext, 0) for ext in (".xlsx", ".xlsm")) != 1 or len(sheets) != 1:
        return guide
    item = sheets[0]
    if not isinstance(item.get("bytes"), int) or not 0 < item["bytes"] <= 2_000_000:
        return guide
    name = item.get("path")
    if not isinstance(name, str) or not name or len(name) >= 260:
        return guide
    file = PurePosixPath(name)
    if file.is_absolute() or ".." in file.parts:
        return guide
    command = "cd " + shlex.quote(session.workdir) + " && " + shlex.join(
        ["python3", TOOL, "xlsx-audit", "./" + name])
    try:
        result = await asyncio.wait_for(session.exec(command, timeout_sec=6), timeout=8)
        if result.get("timed_out") or result.get("truncated") or result.get("exit_code") != 0:
            return guide
        output = result.get("output", "")
        if len(output) > 64000:
            return guide
        audit = json.loads(output)
        if audit["kind"] != "xlsx-formula-audit":
            return guide
        keys = ("cells", "formulas", "formulas_without_cached_values", "error_count")
        counts = {k: sum(row[k] for row in audit["sheets"]) for k in keys}
    except (TimeoutError, OSError, RuntimeError, ValueError, KeyError, TypeError):
        return guide
    return (guide + "\nWorkbook structural audit (untrusted file name): "
            + json.dumps({"path": name, "sheets": len(audit["sheets"]), **counts})
            + ". Cached values can be stale. Inspect the workbook for affected cells and physical assumptions.")
