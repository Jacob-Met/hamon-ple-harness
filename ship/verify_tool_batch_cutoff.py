"""Cold paired verification of acknowledged tool-boundary delivery.

Runs the same real-process and installed-Pi controls on a fixed reference and
this checkout. Loopback outcomes are protocol evidence, never scored PLE data.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run_control(root, destination):
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests",
         "-p", "test_tool_batch_cutoff.py", "-v"],
        cwd=root, capture_output=True, text=True, timeout=150,
    )
    output = result.stdout + "\n" + result.stderr
    destination.write_text(output)
    observations = []
    for line in output.splitlines():
        if "REGISTERED_READ_BOUNDARY " in line:
            observations.append(json.loads(line.split("REGISTERED_READ_BOUNDARY ", 1)[1]))
    if len(observations) != 1:
        raise RuntimeError("Expected one complete registered-read observation; inspect " + str(destination))
    return {"returncode": result.returncode, "registered_read": observations[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.baseline):
        parser.error("--baseline must be an exact commit")
    if not (ROOT / "node_modules/.bin/pi").is_file():
        raise SystemExit("Install the actual submitted Pi with npm ci first.")
    args.out.mkdir(parents=True, exist_ok=True)
    candidate = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="ple-boundary-reference-") as temp:
        reference = Path(temp) / "checkout"
        subprocess.run(["git", "worktree", "add", "--detach", str(reference), args.baseline],
                       cwd=ROOT, check=True)
        try:
            if (reference / "package-lock.json").read_bytes() != (ROOT / "package-lock.json").read_bytes():
                raise RuntimeError("Reference and candidate dependency locks differ.")
            (reference / "node_modules").symlink_to(ROOT / "node_modules", target_is_directory=True)
            shutil.copy2(ROOT / "tests/test_tool_batch_cutoff.py",
                         reference / "tests/test_tool_batch_cutoff.py")
            before = run_control(reference, args.out / "reference.log")
            after = run_control(ROOT, args.out / "candidate.log")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(reference)],
                           cwd=ROOT, check=True)
    old = before["registered_read"]
    new = after["registered_read"]
    expected_loss = (
        before["returncode"] != 0 and old.get("first_stopped") is True
        and old.get("read_requests") == 1 and old.get("work_requests") == 1
        and old.get("results_recorded") == 0 and old.get("history_verified") is False
        and old.get("delivered") is False and "handler_error" not in old
    )
    complete = (
        after["returncode"] == 0 and new.get("first_stopped") is False
        and new.get("read_requests") == 1 and new.get("work_requests") == 1
        and new.get("results_recorded") == 1 and new.get("history_verified") is True
        and new.get("delivered") is True and new.get("total_model_requests") == 3
        and new.get("boundary_markers") == 1 and new.get("empty_abort_tail") is True
    )
    report = {"scope": "real Pi and task protocol with deterministic loopback provider; no PLE grade",
              "baseline_commit": args.baseline, "candidate_commit": candidate,
              "reference": before, "candidate": after,
              "reference_loss_reproduced": expected_loss, "candidate_qualified": complete}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not expected_loss or not complete:
        raise SystemExit("Paired boundary evidence did not meet both frozen gates.")


if __name__ == "__main__":
    main()
