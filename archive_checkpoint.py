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
from xml.etree.ElementTree import ParseError

from office_package import check_office_package


def verify_typed_members(z, names, max_unpacked_bytes=None):
    """Validate declared file formats. Return False if the inspection budget is exhausted."""
    if max_unpacked_bytes is not None:
        max_unpacked_bytes -= sum(info.file_size for info in z.infolist())
        if max_unpacked_bytes < 0:
            return False
    for name in names:
        suffix = os.path.splitext(name)[1].lower()
        if suffix == ".json":
            with z.open(name) as f:
                json.load(f)
        elif suffix in {".xlsx", ".xlsm", ".docx"}:
            with z.open(name) as f, zipfile.ZipFile(f) as doc:
                if max_unpacked_bytes is not None:
                    max_unpacked_bytes -= sum(info.file_size for info in doc.infolist())
                    if max_unpacked_bytes < 0:
                        return False
                try:
                    check_office_package(doc, suffix)
                except (ParseError, LookupError) as exc:
                    raise ValueError(f"invalid Office XML in {name}") from exc
                if doc.testzip():
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
    return True


def verify(archive: Path, names: list[str]) -> None:
    with zipfile.ZipFile(archive) as z:
        if sorted(z.namelist()) != sorted(names) or any(z.getinfo(n).file_size == 0 for n in names):
            raise ValueError("missing, empty, extra or duplicate ZIP member")
        if z.testzip() is not None:
            raise ValueError("invalid ZIP checksum")
        verify_typed_members(z, names)


def verify_checkpoint(stage: Path, target: Path, names: list[str]) -> None:
    """Confirm that the archive matches the current staged member bytes."""
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


def _replace_synced(candidate: Path, target: Path) -> None:
    with candidate.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(candidate, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _copy_verified(source: Path, target: Path, names: list[str], *, staged: Path | None = None) -> None:
    if not _valid_regular_file(source):
        raise ValueError("missing regular archive checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ple-copy-", delete=False) as f:
        candidate = Path(f.name)
    try:
        shutil.copyfile(source, candidate)
        if staged is None:
            verify(candidate, names)
        else:
            verify_checkpoint(staged, candidate, names)
        _replace_synced(candidate, target)
    finally:
        candidate.unlink(missing_ok=True)


def checkpoint(stage: Path, target: Path, names: list[str]) -> None:
    """Save a verified complete version outside the members being refined."""
    _copy_verified(target, stage.with_name(stage.name + ".checkpoint.zip"), names, staged=stage)


def recover(stage: Path, target: Path, names: list[str]) -> str:
    """Keep a complete result, publish complete staging, or restore the saved copy."""
    invalid = (ValueError, OSError, zipfile.BadZipFile, EOFError, RuntimeError, csv.Error)
    try:
        verify(target, names)
    except invalid:
        pass
    else:
        return "kept"
    try:
        publish(stage, target, names)
    except invalid:
        _copy_verified(stage.with_name(stage.name + ".checkpoint.zip"), target, names)
        return "restored"
    return "published"


def publish(stage: Path, target: Path, names: list[str], *, after_empty=False, prior_members=None) -> None:
    if not names or len(names) != len(set(names)) or any(
        not n or n in {".", ".."} or "/" in n or "\\" in n for n in names
    ):
        raise ValueError("expected distinct root filenames")
    if prior_members is not None and set(prior_members) != set(names):
        raise ValueError("incomplete prior observation of generated members")
    if prior_members is None and not after_empty and not target.exists():
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
        if prior_members is not None:
            # Compare the verified, immutable candidate bytes with what was on
            # disk before the generator, including same-size replacements.
            with zipfile.ZipFile(candidate) as archive:
                for name, previous in prior_members.items():
                    if previous is None:
                        continue
                    digest = hashlib.sha256(str(archive.getinfo(name).file_size).encode())
                    with archive.open(name) as member:
                        while block := member.read(1_048_576):
                            digest.update(block)
                    if digest.hexdigest() == previous:
                        raise ValueError(f"unchanged generator output: {name}")
        if after_empty or prior_members is not None:
            _install_over_empty(candidate, target, names)
        else:
            _replace_synced(candidate, target)
    finally:
        candidate.unlink(missing_ok=True)


def _install_over_empty(candidate, target, names):
    """Keep an independent atomic publication arriving before or during installation."""
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("archive destination is not a regular file")
    if target.exists() and target.stat().st_size:
        verify(target, names)
        return
    with candidate.open("rb") as stream:
        os.fsync(stream.fileno())
    # Renaming onto an existing regular tempfile cannot move a directory.
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ple-prior-", delete=False) as f:
        prior = Path(f.name)
    parked = False
    try:
        try:
            os.replace(target, prior)
            parked = True
        except FileNotFoundError:
            pass
        use_prior = parked and (prior.is_symlink() or not prior.is_file() or prior.stat().st_size != 0)
        try:
            os.link(prior if use_prior else candidate, target, follow_symlinks=False)
        except FileExistsError:
            pass
        if target.is_symlink() or not target.is_file():
            raise ValueError("archive destination is not a regular file")
        verify(target, names)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if parked:
            try:
                os.link(prior, target, follow_symlinks=False)
            except FileExistsError:
                pass
        prior.unlink(missing_ok=True)


def publish_loose(stage: Path, target: Path, names: list[str]) -> None:
    """Publish the acknowledged output members; prior private staging cannot supersede them."""
    if target.parent.is_symlink():
        raise ValueError("refusing linked output directory")
    publish(target.parent, target, names)
    # Staging remains available for refinement; never overwrite existing work.
    _hydrate_stage_from_output_dir(stage, target.parent, names)


def publish_after_empty(stage: Path, target: Path, names: list[str]) -> None:
    """Recover a task-created empty target using a single complete source set."""
    if target.is_symlink() or target.parent.is_symlink():
        raise ValueError("refusing linked archive destination")
    if target.exists() and not target.is_file():
        raise ValueError("archive destination is not a regular file")
    if target.exists() and target.stat().st_size != 0:
        verify(target, names)
        return
    if not stage.is_symlink() and all(_valid_regular_file(stage / name) for name in names):
        source = stage
    else:
        source = target.parent
    publish(source, target, names, after_empty=True)


if __name__ == "__main__":
    import sys
    stage, target, members = json.loads(sys.argv[1])
    if len(sys.argv) == 4 and sys.argv[2] == "--from-generated-output":
        target = Path(target)
        if target.parent.is_symlink():
            raise ValueError("refusing linked output directory")
        publish(target.parent, target, members, prior_members=json.loads(sys.argv[3]))
    elif sys.argv[2:] == ["--verify"]:
        verify_checkpoint(Path(stage), Path(target), members)
    elif sys.argv[2:] == ["--checkpoint"]:
        checkpoint(Path(stage), Path(target), members)
    elif sys.argv[2:] == ["--recover"]:
        print(json.dumps({"action": recover(Path(stage), Path(target), members)}))
    elif sys.argv[2:] == ["--first-only"]:
        # An earlier publisher may still finish after an interrupted request.
        publish(Path(stage), Path(target), members, after_empty=True)
        print("PLE_ARCHIVE_PUBLISHED", end="")
    elif sys.argv[2:] == ["--after-empty"]:
        publish_after_empty(Path(stage), Path(target), members)
    elif sys.argv[2:] == ["--from-output"]:
        publish_loose(Path(stage), Path(target), members)
        print("PLE_ARCHIVE_PUBLISHED", end="")
    elif not sys.argv[2:]:
        publish(Path(stage), Path(target), members)
        print("PLE_ARCHIVE_PUBLISHED", end="")
    else:
        raise ValueError("unknown archive operation")
