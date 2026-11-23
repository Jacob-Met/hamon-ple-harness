"""Check two controlled solve recoveries with the official capture/export code.

    python tests/replay_recovery_exports.py /path/to/organizer/competition

Writes only a temporary trial and export for each structural fixture. Does not
run an official dev trial, update the submitted artifacts or file any document.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_archive_checkpoint import ArchiveRoute
from test_terminal_delivery import TerminalReceipt


async def main(kit_root):
    sys.path.insert(0, str(kit_root))
    from kit import artifacts, capture, export
    tasks = artifacts.load_dev_set(artifacts.DEV_SET_FILE)

    def check(task, actual, virtual):
        assert artifacts.covered(tasks[task].deliverables, virtual), virtual
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "trial"
            cap = trial / capture.FOLDER
            cap.mkdir(parents=True)
            with actual.open("rb") as stream:
                assert capture._save(stream, cap, virtual, [])
            rel = virtual.lstrip("/")
            (cap / capture.INFO).write_text(json.dumps({"files": {rel: {"mode": "644"}}}))
            output = Path(tmp) / "export"
            content, reason = export.task_folder(trial, output)
            assert not reason, reason
            exported = output / artifacts.FILES / rel
            assert content["files"][rel] == artifacts.file_blob_id(exported)
            assert actual.read_bytes() == exported.read_bytes()
            print(json.dumps({"task": task, "path": virtual, "bytes": exported.stat().st_size,
                              "sha256": hashlib.sha256(exported.read_bytes()).hexdigest()}))

    zip_path = "/filesystem/dc_structural__foundation_strategy__blind/output/submission.zip"
    await ArchiveRoute().exercise(complete=False, late_foundation=True,
                                  inspect=lambda path: check("foundation-strategy", path, zip_path))
    name = "Municipal_Witness_Test_Package.docx"
    word_path = "/filesystem/work/.ple-delivery/" + name
    def check_word(root):
        local = root / word_path.lstrip("/")
        assert local.read_bytes() == (root / "fixture-document-store" / name).read_bytes()
        check("solar-witness-test-package", local, word_path)
    await TerminalReceipt().recover_named_filing(check_word)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
