"""Independent SymPy units and LibreOffice replay for the saved XLSX graph.

This is a qualification fixture, not an official PLE model run or submission.
Run on a host with at least 4 GiB effective memory and libreoffice-calc installed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engineering_toolkit import audit_local_deliverable, xlsx_audit

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def effective_memory():
    capacity = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    limits = [capacity]
    for path in (Path("/sys/fs/cgroup/memory.max"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.exists():
            value = path.read_text().strip()
            if value.isdecimal():
                limits.append(int(value))
    return min(limits)


def cases():
    import sympy as sp
    from sympy.physics import units as u
    from sympy.physics.units import convert_to

    units = {
        "m": u.meter, "mm": u.millimeter, "cm": u.centimeter,
        "km": u.kilometer, "in": u.inch, "ft": u.foot, "yd": u.yard,
        "g": u.gram, "kg": u.kilogram, "lbm": u.pound,
        "sec": u.second, "hr": u.hour,
        "Pa": u.pascal, "kPa": 1000*u.pascal, "MPa": 10**6*u.pascal,
    }

    def ratio(source, target):
        value = sp.simplify(convert_to(units[source], units[target]) / units[target])
        if not value.is_Rational:
            raise AssertionError((source, target, value))
        return value

    rows = [
        ('CONVERT(Inputs!A1*2,"m","cm")', sp.Rational(3)*ratio("m","cm")),
        ('CONVERT(Inputs!A2+Inputs!A2,"cm","in")', sp.Rational(508,100)*ratio("cm","in")),
        ('CONVERT(CONVERT(Inputs!A3,"hr","sec")/3,"sec","hr")', sp.Rational(2,3)),
        ('Inputs!A4*7.5', sp.Rational(45,2)),
        ('CONVERT(A4,"kg","lbm")', sp.Rational(45,2)*ratio("kg","lbm")),
        ('CONVERT(2*3,"m","cm")', sp.Rational(6)*ratio("m","cm")),
        ('CONVERT(-1,"m","cm")', -ratio("m","cm")),
        ('CONVERT(1+1,"m","cm")', 2*ratio("m","cm")),
        ('CONVERT(CONVERT(Inputs!A1,"m","cm")*CONVERT(Inputs!A2,"cm","m")/'
         'CONVERT(Inputs!A1,"m","cm"),"m","in")', sp.Rational(254,100)*ratio("cm","in")),
        ('CONVERT(Inputs!A1,"m","cm")*CONVERT(Inputs!A2,"cm","m")', sp.Rational(381,100)),
        ('A10/CONVERT(Inputs!A1,"m","cm")', sp.Rational(127,5000)),
        ('CONVERT(A11,"m","in")', sp.Rational(1)),
        ('CONVERT(Inputs!A1,"m","cm")+CONVERT(Inputs!A2,"cm","cm")', sp.Rational(7627,50)),
        ('CONVERT(A13,"cm","m")', sp.Rational(7627,5000)),
    ]
    # Independent reproduction of the pinned Calc 24.2 conversion table,
    # separate from SymPy's exact SI definition and the production evaluator.
    # https://github.com/LibreOffice/core/blob/libreoffice-24.2.7.2/scaddins/source/analysis/analysishelper.cxx
    legacy_lbm_per_kg = sp.Rational("2.20462291469134")
    legacy_expected = {"A5": sp.Rational(45,2)*legacy_lbm_per_kg}
    for group in (("m","mm","cm","km","in","ft","yd"), ("g","kg","lbm"), ("sec","hr"), ("Pa","kPa","MPa")):
        for source in group:
            for target in group:
                rows.append((f'CONVERT(5/2,"{source}","{target}")',
                             sp.Rational(5,2)*ratio(source,target)))
                if "lbm" in (source,target) and source != target:
                    factors = {"g":sp.Rational(1,1000), "kg":sp.Rational(1),
                               "lbm":1/legacy_lbm_per_kg}
                    legacy_expected[f"A{len(rows)}"] = sp.Rational(5,2)*factors[source]/factors[target]
    # NASA's worked value is rounded to 30.2 kPa. Compare the unrounded
    # pressure with a dimensioned SymPy expression at the published inputs.
    # https://www.nasa.gov/wp-content/uploads/2023/06/principles-of-flight-in-action-9-12.pdf
    density = sp.Rational("0.74") * u.kilogram / u.meter**3
    speed = sp.Rational("285.8") * u.meter / u.second
    pressure = convert_to(density * speed**2 / 2, u.pascal) / (1000 * u.pascal)
    rows.extend([
        ('CONVERT((CONVERT(0.74,"kg","kg")/CONVERT(1,"m","m")^3)*'
         '(CONVERT(285.8,"m","m")/CONVERT(1,"sec","sec"))^2/2,"Pa","kPa")', pressure),
        ("-2^2", sp.Integer(4)), ("2^3^2", sp.Integer(64)),
        ("2^(3^2)", sp.Integer(512)), ("4/2^-2", sp.Integer(16)),
    ])
    # Independent quantities establish the composite unit frames as well as numbers.
    assert sp.simplify(convert_to(u.centimeter*u.meter/u.centimeter, u.meter)/u.meter) == 1
    assert sp.simplify(convert_to(u.inch*u.centimeter/u.inch, u.meter)/u.meter) == sp.Rational(1,100)
    return rows, sp.__version__, legacy_expected


def sheet(rows, formulas):
    root = ET.Element("worksheet", xmlns=MAIN)
    data = ET.SubElement(root, "sheetData")
    for index, value in enumerate(rows, 1):
        row = ET.SubElement(data, "row", r=str(index))
        cell = ET.SubElement(row, "c", r=f"A{index}")
        ET.SubElement(cell, "f" if formulas else "v").text = str(value)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_workbook(path, rows):
    types = ('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
             + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in (1,2))
             + '</Types>')
    workbook = (f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>'
                '<sheet name="Inputs" sheetId="1" r:id="rId1"/>'
                '<sheet name="Checks" sheetId="2" r:id="rId2"/>'
                '</sheets><calcPr fullCalcOnLoad="1"/></workbook>')
    relationships = (f'<Relationships xmlns="{PKG}">'
                     + ''.join(f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in (1,2))
                     + '</Relationships>')
    parts = {
        "[Content_Types].xml": types,
        "_rels/.rels": f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": workbook,
        "xl/_rels/workbook.xml.rels": relationships,
        "xl/worksheets/sheet1.xml": sheet([1.5,2.54,2,3], False),
        "xl/worksheets/sheet2.xml": sheet([formula for formula,_ in rows], True),
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as book:
        for name, contents in parts.items():
            book.writestr(name, contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-memory-gib", type=float, default=4)
    args = parser.parse_args()
    memory = effective_memory()
    if memory < args.min_memory_gib * 1024**3:
        raise SystemExit("Insufficient effective memory for the independent Calc qualification")
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if executable is None:
        raise SystemExit("LibreOffice Calc is required; this gate does not skip a missing engine")
    rows, sympy_version, legacy_expected = cases()
    args.out.mkdir(parents=True, exist_ok=True)
    source, output = args.out/"source.xlsx", args.out/"recalculated"
    output.mkdir()
    write_workbook(source, rows)
    before = xlsx_audit(source)
    assert sum(s["formulas_without_cached_values"] for s in before["sheets"]) == len(rows)
    # Calc evaluates the spreadsheet numbers even when the stated input unit
    # contradicts a preceding conversion. Keep this control separate from the ordinary formulas.
    source_frames = args.out / "conversion-source-frames.xlsx"
    converted_cm = 'CONVERT(Inputs!A3,"m","cm")*2'
    controls = [
        (f'CONVERT({converted_cm},"m","mm")', 400000, "unit_scale_mismatch"),
        (f'CONVERT({converted_cm},"kg","g")', 400000, "dimension_mismatch"),
        (f'CONVERT({converted_cm},"cm","mm")', 4000, "matched"),
        ('CONVERT(Inputs!A2^0,"Pa","kPa")', .001, "dimension_mismatch"),
    ]
    write_workbook(source_frames, [(formula, value) for formula, value, _ in controls])
    command = [executable, f"-env:UserInstallation={(args.out/'calc-profile').resolve().as_uri()}",
               "--headless", "--convert-to", "xlsx", "--outdir", str(output), str(source), str(source_frames)]
    calc = subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)
    (args.out/"calc.log").write_text(calc.stdout+calc.stderr)
    frames_recalculated = output / source_frames.name
    frame_audit = xlsx_audit(frames_recalculated)
    frame_guard = audit_local_deliverable(frames_recalculated)
    assert not frame_guard["ok"], frame_guard
    frames = {r["cell"]: r for r in frame_audit["conversion_audit"]["examples"] if r["sheet"] == "Checks"}
    assert set(frames) == {f"A{i}" for i in range(1,len(controls)+1)}
    for index, (_, expected_value, expected_status) in enumerate(controls, 1):
        observed = frames[f"A{index}"]
        assert observed["cached"] == expected_value and observed["status"] == expected_status, observed
    recalculated = output/source.name
    audit = xlsx_audit(recalculated)
    guard = audit_local_deliverable(recalculated)
    (args.out/"audit-fresh.json").write_text(json.dumps(audit,indent=2,default=str)+"\n")
    assert guard["ok"], guard
    checks = next(s["part"] for s in audit["sheets"] if s["name"] == "Checks")
    with ZipFile(recalculated) as book:
        parts = {name:book.read(name) for name in book.namelist()}
    root = ET.fromstring(parts[checks])
    cells = {cell.get("r"):cell for cell in root.iter(f"{{{MAIN}}}c")}
    comparisons, engine_variants = [], []
    for index, (formula, expected) in enumerate(rows, 1):
        cell = cells[f"A{index}"]
        assert cell.get("t", "n") == "n", ET.tostring(cell).decode()
        actual = float(cell.find(f"{{{MAIN}}}v").text)
        expected_float = float(expected)
        exact_match = math.isclose(actual, expected_float, rel_tol=1e-12, abs_tol=0)
        alternate = legacy_expected.get(f"A{index}")
        legacy_match = alternate is not None and math.isclose(actual,float(alternate),rel_tol=1e-12,abs_tol=0)
        assert exact_match or legacy_match, (formula,actual,expected,alternate)
        if not exact_match:
            engine_variants.append({"cell":f"A{index}", "sympy_exact":str(expected),
                                    "calc_legacy_exact":str(alternate), "calc_value":actual})
        comparisons.append({"cell":f"A{index}", "source_formula":formula,
                            "sympy_exact":str(expected), "calc_value":actual})
    matched = sum(audit[kind]["counts"]["matched"] for kind in ("arithmetic_audit","conversion_audit"))
    unverified = sum(audit[kind]["counts"]["unverified"] for kind in ("arithmetic_audit","conversion_audit"))
    assert matched+len(engine_variants) == len(rows) and unverified == len(engine_variants), audit
    # Alter only a separate control copy; preserve the engine-produced package unchanged.
    stale_cells = ("A1","A5","A6","A9",f"A{len(rows)-4}")
    for address in stale_cells:
        value = cells[address].find(f"{{{MAIN}}}v")
        value.text = str(float(value.text)*1.125)
    parts[checks] = ET.tostring(root)
    stale = args.out/"stale-control.xlsx"
    with ZipFile(stale, "w", compression=ZIP_DEFLATED) as book:
        for name, contents in parts.items():
            book.writestr(name, contents)
    stale_audit, stale_guard = xlsx_audit(stale), audit_local_deliverable(stale)
    assert not stale_guard["ok"], stale_guard
    mismatches = [row["cell"] for kind in ("arithmetic_audit","conversion_audit")
                  for row in stale_audit[kind]["examples"]
                  if row["sheet"] == "Checks" and row["status"] == "mismatch"]
    assert set(mismatches) == set(stale_cells), mismatches
    result = {
        "qualified":True, "effective_memory_gib":memory/1024**3,
        "libreoffice":subprocess.check_output([executable,"--version"],text=True).strip(),
        "sympy":sympy_version, "independent_pair_conversions":71,
        "fresh_formulas_compared":len(rows), "fresh_formulas_matched":matched,
        "engine_specific_lbm_variants":engine_variants, "fresh_guard":guard["ok"],
        "stale_guard":stale_guard["ok"], "stale_mismatch_cells":mismatches,
        "source_frame_controls":frames,
        "comparisons":comparisons,
    }
    (args.out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key != "comparisons"}))


if __name__ == "__main__":
    main()

