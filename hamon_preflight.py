"""Validate PLE deliverables using only Python's standard library."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


MAX_PLE_BYTES = 512 * 1024 * 1024
MAX_PLE_FILES = 10_000


def sanitize_scalar_value(val: Any) -> float:
    if isinstance(val, bool) or val is None:
        raise ValueError(f"Invalid boolean/null scalar answer: {val!r}")
    if isinstance(val, (int, float)):
        f = float(val)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite scalar answer: {val!r}")
        return f
    if isinstance(val, str):
        s = val.strip()
        if not re.match(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$", s):
            raise ValueError(
                f"Scalar string contains prose or unstripped units: {val!r}"
            )
        f = float(s)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite parsed scalar: {val!r}")
        return f
    raise ValueError(f"Unsupported scalar type: {type(val).__name__}")


def check_bundle_preflight(
    bundle_path: str | Path,
    expected_task_ids: Optional[Sequence[str]] = None,
    max_bytes: int = MAX_PLE_BYTES,
    max_files: int = MAX_PLE_FILES,
) -> Dict[str, Any]:
    path = Path(bundle_path)
    errors: List[str] = []
    total_uncompressed = 0
    compressed_bytes = 0
    file_count = 0
    seen_names: set = set()
    scalar_payloads_checked = 0

    if not path.exists():
        return {
            "ok": False,
            "errors": [f"Missing bundle: {path}"],
            "file_count": 0,
            "total_bytes": 0,
        }

    if path.is_file() and path.suffix.lower() == ".zip":
        compressed_bytes = path.stat().st_size
        if compressed_bytes > max_bytes:
            errors.append(
                f"Compressed zip size {compressed_bytes} exceeds {max_bytes} bytes"
            )
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                name = info.filename
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if unix_mode and stat.S_ISLNK(unix_mode):
                    errors.append(f"Symlink prohibited in archive: {name}")
                if (
                    name.startswith("/")
                    or name.startswith("\\")
                    or ".." in Path(name).parts
                ):
                    errors.append(f"Unsafe archive member path: {name}")
                if info.is_dir():
                    continue
                norm = name.replace("\\", "/").lstrip("./")
                if norm in seen_names:
                    errors.append(f"Duplicate archive member path: {norm}")
                seen_names.add(norm)
                file_count += 1
                total_uncompressed += info.file_size
                if norm.endswith(
                    (
                        "predictions.json",
                        "answers.json",
                        "scalar_answers.json",
                        "wave3_scalar_predictions.json",
                    )
                ):
                    raw_txt = zf.read(info).decode("utf-8")
                    _validate_scalar_json_text(raw_txt, norm, expected_task_ids, errors)
                    scalar_payloads_checked += 1
    elif path.is_dir():
        for root, dirs, files in os.walk(path, followlinks=False):
            root_p = Path(root)
            for d in dirs:
                dp = root_p / d
                if dp.is_symlink():
                    errors.append(
                        f"Symlink directory prohibited: {dp.relative_to(path)}"
                    )
            for fn in files:
                fp = root_p / fn
                rel = fp.relative_to(path).as_posix()
                if fp.is_symlink():
                    errors.append(f"Symlink file prohibited: {rel}")
                    continue
                file_count += 1
                total_uncompressed += fp.stat().st_size
                if rel.endswith(
                    (
                        "predictions.json",
                        "answers.json",
                        "scalar_answers.json",
                        "wave3_scalar_predictions.json",
                    )
                ):
                    _validate_scalar_json_text(
                        fp.read_text(encoding="utf-8"), rel, expected_task_ids, errors
                    )
                    scalar_payloads_checked += 1
    else:
        errors.append(f"Unsupported deliverable path type: {path}")

    if file_count > max_files:
        errors.append(f"File count {file_count} exceeds {max_files}")
    if total_uncompressed > max_bytes:
        errors.append(
            f"Uncompressed size {total_uncompressed} exceeds {max_bytes} bytes"
        )
    if file_count == 0:
        errors.append("Deliverable archive contains 0 files")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "file_count": file_count,
        "compressed_bytes": compressed_bytes,
        "total_uncompressed_bytes": total_uncompressed,
        "scalar_payloads_checked": scalar_payloads_checked,
    }


def _validate_scalar_json_text(
    raw_txt: str,
    label: str,
    expected_task_ids: Optional[Sequence[str]],
    errors: List[str],
) -> None:
    def bad_const(tok: str) -> Any:
        raise ValueError(f"Non-standard JSON constant {tok} in {label}")

    try:
        data = json.loads(raw_txt, parse_constant=bad_const)
    except Exception as exc:
        errors.append(f"Invalid JSON in {label}: {exc}")
        return

    if not isinstance(data, dict):
        errors.append(f"Scalar JSON root in {label} must be an object")
        return

    if expected_task_ids is not None:
        missing = [tid for tid in expected_task_ids if tid not in data]
        if missing:
            errors.append(
                f"Missing {len(missing)} required task IDs in {label}: {missing[:5]}"
            )

    for k, v in data.items():
        val = v.get("answer") if isinstance(v, dict) and "answer" in v else v
        try:
            sanitize_scalar_value(val)
        except ValueError as exc:
            errors.append(f"Task {k} in {label}: {exc}")


def check_deliverable_preflight(target_path, expected_task_ids=None, **kwargs):
    p = Path(target_path)
    if not p.exists():
        return {
            "ok": False,
            "passed": False,
            "valid": False,
            "path": str(p),
            "size_bytes": 0,
            "issues": ["missing_deliverable"],
            "errors": ["missing_deliverable"],
        }
    if p.is_file() and p.suffix.lower() != ".zip":
        sz = p.stat().st_size
        issues = []
        if sz == 0:
            issues.append("empty_file")
        suf = p.suffix.lower()
        if suf == ".json" and sz > 0:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))

                def _scan_num(v):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        issues.append("non_finite_float")
                    elif isinstance(v, dict):
                        for vv in v.values():
                            _scan_num(vv)
                    elif isinstance(v, list):
                        for vv in v:
                            _scan_num(vv)

                _scan_num(data)
                if expected_task_ids and isinstance(data, dict):
                    missing = [
                        tid
                        for tid in expected_task_ids
                        if tid not in data or data[tid] is None
                    ]
                    if missing:
                        issues.append(f"missing_task_ids:{missing}")
            except Exception as exc:
                issues.append(f"invalid_json:{exc}")
        elif suf == ".csv" and sz > 0:
            try:
                rows = list(
                    csv.reader(
                        p.read_text(encoding="utf-8", errors="replace").splitlines()
                    )
                )
                if not rows:
                    issues.append("empty_csv")
            except Exception as exc:
                issues.append(f"invalid_csv:{exc}")
        elif suf == ".ifc" and sz > 0:
            txt = p.read_text(encoding="utf-8", errors="replace")
            if "ISO-10303-21" not in txt and "DATA;" not in txt:
                issues.append("invalid_ifc_header")
        elif suf == ".pdf" and sz > 0:
            raw = p.read_bytes()
            if not raw.startswith(b"%PDF"):
                issues.append("invalid_pdf_header")
        ok = len(issues) == 0
        return {
            "ok": ok,
            "passed": ok,
            "valid": ok,
            "path": str(p),
            "size_bytes": sz,
            "file_count": 1,
            "issues": issues,
            "errors": issues,
        }

    limits = {key: kwargs[key] for key in ("max_bytes", "max_files") if key in kwargs}
    result = check_bundle_preflight(target_path, expected_task_ids, **limits)
    result["passed"] = result["valid"] = result["ok"]
    return result


def run_router_preflight(payload: dict) -> dict:
    target = payload.get("path") or payload.get("target_path") or ""
    result = check_deliverable_preflight(
        target, expected_task_ids=payload.get("expected_task_ids")
    )
    result.setdefault("path", str(Path(target)))
    result["tool"] = "check_deliverable_preflight"
    return result


if __name__ == "__main__":
    request = json.loads(sys.argv[1])
    print(json.dumps(run_router_preflight(request), allow_nan=False))
