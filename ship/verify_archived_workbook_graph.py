"""Replay the archived MV conveyance workbook against two verifier versions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def run_tool(checkout, command, workbook):
    result = subprocess.run([sys.executable, str(checkout/"engineering_toolkit.py"),
                             command, str(workbook)], capture_output=True, text=True, timeout=30)
    if result.returncode not in (0, 3):
        raise RuntimeError(result.stderr + result.stdout)
    return json.loads(result.stdout)


def corrupt_caches(clean, target, sheets):
    # Keep both late cost caches mutually consistent with the same false total.
    # The verifier must reconstruct the upstream arithmetic from original inputs.
    changed = {}
    with ZipFile(clean) as src, ZipFile(target, "w", compression=ZIP_DEFLATED) as dst:
        for name in src.namelist():
            data = src.read(name)
            if name in sheets:
                sheet_name, addresses = sheets[name]
                root = ET.fromstring(data)
                for address in addresses:
                    cell = root.find(f".//s:c[@r='{address}']", NS)
                    assert cell is not None and cell.find("s:f", NS) is not None
                    value = cell.find("s:v", NS)
                    observed = float(value.text)
                    assert math.isfinite(observed) and observed != 0
                    changed[(sheet_name, address)] = observed
                    value.text = str(observed * 1.25)
                data = ET.tostring(root)
            dst.writestr(name, data)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archive, = (ROOT/"artifacts/mv-conveyance-study/files").rglob("submission.zip")
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        original = work/"conveyance_study.xlsx"
        with ZipFile(archive) as source:
            original.write_bytes(source.read(original.name))
        original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
        before = run_tool(ROOT, "xlsx-audit", original)
        assert sum(s["formulas_without_cached_values"] for s in before["sheets"]) == 383
        out = work/"recalculated"
        out.mkdir()
        subprocess.run(["soffice", f"-env:UserInstallation={(work/'profile').as_uri()}",
                        "--headless", "--convert-to", "xlsx", "--outdir", str(out), str(original)],
                       check=True, capture_output=True, text=True, timeout=90)
        assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
        clean = out/original.name
        baseline = run_tool(args.reference, "xlsx-audit", clean)["arithmetic_audit"]["counts"]
        current = run_tool(ROOT, "xlsx-audit", clean)
        counts = current["arithmetic_audit"]["counts"]
        assert baseline == {"matched": 256, "mismatch": 0, "unverified": 127}, baseline
        assert counts == {"matched": 383, "mismatch": 0, "unverified": 0}, counts
        assert run_tool(ROOT, "format-guard", clean)["ok"]
        parts = {sheet["name"]: sheet["part"] for sheet in current["sheets"]}
        corrupted = work/"corrupted.xlsx"
        expected = corrupt_caches(clean, corrupted, {
            parts["Alternatives"]: ("Alternatives", ["T35", "U35"]),
            parts["LaborRates"]: ("LaborRates", ["G10"]),
        })
        assert run_tool(args.reference, "format-guard", corrupted)["ok"]
        assert not run_tool(ROOT, "format-guard", corrupted)["ok"]
        stale = run_tool(ROOT, "xlsx-audit", corrupted)["arithmetic_audit"]
        failures = {(row["sheet"], row["cell"]): row for row in stale["examples"]
                    if row["status"] == "mismatch"}
        assert stale["counts"] == {"matched": 380, "mismatch": 3, "unverified": 0}, stale["counts"]
        assert failures.keys() == expected.keys()
        for address, row in failures.items():
            assert math.isclose(row["expected"], expected[address], rel_tol=1e-12), row
        cost_roots = ({("Alternatives", f"{column}35") for column in "GHIJ"}
                      | {("Markups", f"B{i}") for i in range(2, 9)})
        expected_roots = {
            ("Alternatives", "T35"): cost_roots,
            ("Alternatives", "U35"): cost_roots | {("Alternatives", "E35")},
            ("LaborRates", "G10"): {("LaborRates", f"{column}10") for column in "DEF"},
        }
        for key, roots in expected_roots.items():
            assert {tuple(ref) for ref in failures[key]["root_cells"]} == roots
            assert failures[key]["root_cell_count"] == len(roots)
        report = {
            "archive": str(archive.relative_to(ROOT)), "workbook": original.name,
            "source_sha256": original_hash, "recalculated_sha256": hashlib.sha256(clean.read_bytes()).hexdigest(),
            "libreoffice": subprocess.check_output(["soffice","--version"], text=True).strip(),
            "reference": baseline, "candidate": counts, "corrupted": stale["counts"],
            "rejected_cells": {f"{sheet}!{cell}": failures[(sheet, cell)]["root_cells"]
                               for sheet, cell in sorted(failures)},
            "reference_accepted_corruption": True, "candidate_accepted_corruption": False,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
