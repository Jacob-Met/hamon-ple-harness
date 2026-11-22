"""Publish complete root-member ZIPs without replacing a previous valid checkpoint mid-write."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


def verify(archive: Path, names: list[str]) -> None:
    with zipfile.ZipFile(archive) as z:
        if sorted(z.namelist()) != sorted(names) or any(z.getinfo(n).file_size == 0 for n in names):
            raise ValueError("missing, empty, extra or duplicate ZIP member")
        if z.testzip() is not None:
            raise ValueError("invalid ZIP checksum")
        for name in names:
            suffix = Path(name).suffix.lower()
            if suffix == ".json":
                with z.open(name) as f:
                    json.load(f)
            elif suffix in {".xlsx", ".docx"}:
                key = "xl/workbook.xml" if suffix == ".xlsx" else "word/document.xml"
                with z.open(name) as f, zipfile.ZipFile(f) as doc:
                    if not {"[Content_Types].xml", key}.issubset(doc.namelist()) or doc.testzip():
                        raise ValueError(f"invalid Office document {name}")
            elif suffix == ".csv":
                with z.open(name) as f, io.TextIOWrapper(f, encoding="utf-8-sig") as text:
                    if not next(csv.reader(text), None):
                        raise ValueError(f"missing CSV header in {name}")
            elif suffix == ".pdf":
                with z.open(name) as f:
                    head = f.read(8)
                    if not head.startswith(b"%PDF-"):
                        raise ValueError(f"invalid PDF header in {name}")
            elif suffix == ".dxf":
                with z.open(name) as f:
                    raw = f.read().decode("utf-8", errors="strict")
                    if "EOF" not in raw:
                        raise ValueError(f"truncated DXF in {name}")
            elif suffix == ".md":
                with z.open(name) as f:
                    if not f.read().decode("utf-8", errors="strict").strip():
                        raise ValueError(f"blank Markdown in {name}")


def verify_checkpoint(stage: Path, target: Path, names: list[str]) -> None:
    """Confirm that the graded archive is the exact first version this session staged."""
    verify(target, names)
    with zipfile.ZipFile(target) as archive:
        for name in names:
            member = stage / name
            if member.is_symlink() or not member.is_file():
                raise ValueError(f"missing staged version of {name}")
            with member.open("rb") as staged, archive.open(name) as saved:
                if hashlib.file_digest(staged, "sha256").digest() != hashlib.file_digest(saved, "sha256").digest():
                    raise ValueError(f"staged and published versions differ for {name}")


def _valid_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0


def _hydrate_stage_from_output_dir(stage: Path, output_dir: Path, names: list[str]) -> None:
    if not output_dir.is_dir() or output_dir.is_symlink():
        return
    if stage.exists() and stage.is_symlink():
        return
    if not all(_valid_regular_file(stage / n) or _valid_regular_file(output_dir / n) for n in names):
        return
    stage.mkdir(parents=True, exist_ok=True)
    for n in names:
        dst = stage / n
        src = output_dir / n
        if not _valid_regular_file(dst) and _valid_regular_file(src):
            shutil.copy2(src, dst)


def publish(stage: Path, target: Path, names: list[str]) -> None:
    if not names or len(names) != len(set(names)) or any(
        not n or n in {".", ".."} or "/" in n or "\\" in n for n in names
    ):
        raise ValueError("expected distinct root filenames")
    if not target.exists():
        _hydrate_stage_from_output_dir(stage, target.parent, names)
    sources = [stage / n for n in names]
    if any(not _valid_regular_file(p) for p in sources):
        raise ValueError("required staged members are missing or empty")
    def stamp(p):
        st = p.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    initial = [stamp(p) for p in sources]
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ple-zip-", delete=False) as f:
        candidate = Path(f.name)
    try:
        with zipfile.ZipFile(candidate, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, path.name)
        verify(candidate, names)
        if initial != [stamp(p) for p in sources]:
            raise ValueError("staged members changed during publication")
        with candidate.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(candidate, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        candidate.unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    stage, target, members = json.loads(sys.argv[1])
    if sys.argv[2:] == ["--verify"]:
        verify_checkpoint(Path(stage), Path(target), members)
    else:
        publish(Path(stage), Path(target), members)
