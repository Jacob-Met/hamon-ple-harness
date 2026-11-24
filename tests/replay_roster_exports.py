"""Replay capture/export provenance on the public roster with fixture bytes.

This exercises the organizer's unscored artifact interface. Trials, receipts
and the deliberate message loss exist only in the temporary directory.
"""
import asyncio
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ship.check_artifact_provenance import evaluate_official
from roster_tool_write_cancel import produce


async def replay(kit, scratch):
    sys.path.insert(0, str(kit))
    from kit import artifacts, capture, export

    roster = artifacts.load_dev_set(artifacts.DEV_SET_FILE)
    assert len(roster) == 29, "the pinned public development roster changed"
    entry = scratch / "entry"
    entry.mkdir()
    subprocess.run(["git", "init", "-q", str(entry)], check=True)
    (entry / "fixture.txt").write_text("Unscored export provenance test.\n")
    source = artifacts.worktree_entries(entry)
    harness = {"tree": artifacts.harness_tree(source), "files": artifacts.harness_files(source)}

    archived = ROOT / "artifacts"
    docx = archived / "backfeed-energization-reply/files/filesystem/Halbray_Backfeed_Reply.docx"
    ifc = archived / "data-center-site-model/files/app/files/frontier__DCBim__p6_002/submission/model.ifc"
    with zipfile.ZipFile(docx) as package:
        assert "word/document.xml" in package.namelist()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("fixture.txt", "Unscored export control.\n")
    payloads = {".docx": docx.read_bytes(), ".ifc": ifc.read_bytes(),
                ".zip": archive.getvalue(), ".py": b"CAPTURE_FIXTURE = True\n"}
    rows = []
    witnesses = {}
    for task_id, task in roster.items():
        target = task.deliverables[0]
        if target.endswith("/"):
            target += "capture-fixture.docx"
        assert task.covers(target)
        payload = payloads[Path(target).suffix]
        trial = scratch / "trials" / task_id
        cap = trial / capture.FOLDER
        cap.mkdir(parents=True)
        saved, message = await produce(task_id, target, payload, scratch / "live-writes")
        with saved.open("rb") as source_file:
            assert capture._save(source_file, cap, target, [])
        relative = target.lstrip("/")
        (cap / capture.INFO).write_text(json.dumps({"files": {relative: {"mode": "644"}}}))
        (cap / artifacts.FINAL_MESSAGE).write_text(message, encoding="utf8")
        rows.append({"task": task_id, "trial": str(trial), "measured": True, "run": "capture-fixture"})
        witnesses[task_id] = (relative, artifacts.blob_id(payload))

    dest = entry / artifacts.ROOT
    results = export.export_run(rows, dest, harness, artifacts.dev_set_digest(artifacts.DEV_SET_FILE))
    assert len(results) == len(roster) and all(result.entry for result in results)
    for result in results:
        path, sha = witnesses[result.task]
        assert result.entry["files"] == {path: sha}
        assert artifacts.file_blob_id(dest / result.task / artifacts.FILES / path) == sha
    baseline = evaluate_official(entry, kit)
    assert baseline["status"] == "fresh" and baseline["kit_review"]["counts"]["ok"] == 29

    # The rubric reads the final message as well as the captured workbook/document.
    focal = "fire-test-validation-memo"
    assert roster[focal].final_message
    message = dest / focal / artifacts.FINAL_MESSAGE
    original = message.read_bytes()
    message.unlink()
    damaged = evaluate_official(entry, kit)
    assert damaged["kit_review"]["accepted"] and damaged["kit_review"]["counts"]["ok"] == 29
    assert damaged["kit_review"]["provenance"] == baseline["kit_review"]["provenance"]
    assert damaged["kit_review"]["provenance"]["edited"] == []
    assert damaged["status"] == "stale" and damaged["final_message_mismatches"] == [focal]
    message.write_bytes(original)
    assert evaluate_official(entry, kit)["status"] == "fresh"
    print(json.dumps({"registered_tasks": len(roster), "cold_file_hashes": len(witnesses),
                      "registered_writes_followed_by_cancel": len(rows),
                      "missing_message": focal, "raw_organizer_usable_after_deletion": 29,
                      "local_freshness": ["fresh", "stale", "fresh"]}))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(replay(Path(sys.argv[1]).resolve(), Path(tmp)))
