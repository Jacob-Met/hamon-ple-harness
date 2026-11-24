"""Compare the registered IFC reader with independently integrated original FZK solids."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_wired_router_e2e import _start_mock_relay
from oracle import SOURCE_SHA256

def save(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")

def by_step(rows):
    result = {row["step_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate source occurrence identifiers")
    return result

def main(source, out):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Different IFC input")
    manifest = json.loads((out / "manifest.json").read_bytes())
    native = json.loads((out / "native-results.json").read_bytes())
    if not (manifest["source_sha256"] == native["source_sha256"] == SOURCE_SHA256):
        raise ValueError("Source digest differs between measurements")
    calls = []
    server = _start_mock_relay(source.parent, calls)
    runner = """
      import register from './task-tools.ts';
      const registered = new Map();
      register({registerTool(t){registered.set(t.name,t)}, on(){}});
      const reader = registered.get('hamon_ifc_spatial_clash');
      if (reader.parameters.properties.allow_partial.type !== 'boolean') throw Error('Missing public selector');
      const result = await reader.execute('fzk-qualification', {path:process.argv[1], allow_partial:true});
      console.log(JSON.stringify(result.details));
    """
    env = dict(os.environ, TASK_RELAY_URL=f"http://127.0.0.1:{server.server_port}",
               TASK_WORKDIR=str(source.parent), TASK_HARNESS_SRC=str(ROOT), TASK_HARNESS_PYTHON=sys.executable)
    try:
        proc = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", runner,
                               source.name], cwd=ROOT, env=env, capture_output=True, timeout=600)
    finally:
        server.shutdown()
        server.server_close()
    (out / "registered-reader.stdout").write_bytes(proc.stdout)
    (out / "registered-reader.stderr").write_bytes(proc.stderr)
    proc.check_returncode()
    if calls != ["read_file"]:
        raise ValueError(f"Unexpected registered-reader source operations: {calls}")
    report = json.loads(proc.stdout)
    save(out / "registered-reader.json", report)
    physical = by_step([r for r in report["products"] if r["role"] == "physical"])
    products = by_step(manifest["products"])
    geometry = {int(k): v for k, v in native["per_product"].items()}
    failures = []
    if physical.keys() != products.keys() or geometry.keys() != products.keys() or len(products) != 82:
        failures.append("physical STEP inventory differs")
    compared = []
    for step in sorted(physical.keys() & geometry.keys() & products.keys()):
        row, expected, ident = physical[step], geometry[step], products[step]
        same_id = row["ifc_type"].upper() == ident["ifc_type"].upper() and row["guid"] == ident["guid"]
        actual, reference = row["volume_m3"], expected["volume_m3"]
        matches = (row["volume_status"] == expected["status"] and
                   ((reference is None and actual is None) or
                    (type(actual) in (int,float) and reference is not None and math.isfinite(actual)
                     and math.isclose(actual, reference, rel_tol=1e-7, abs_tol=1e-9))))
        compared.append(dict(step_id=step, matching_identity=same_id, matching_volume=matches,
                             candidate_m3=actual, native_m3=reference, reason=row.get("reason")))
        if not (matches and same_id):
            failures.append(f"STEP #{step} identity/status/volume")
    if (report.get("status") != "partial" or report.get("ok") is not False
            or any(k not in report or report[k] is not None for k in
                   ("total_solid_volume_m3", "hard_clash_count", "clearance_violation_count"))):
        failures.append("incomplete model quantities must remain unknown")
    digest = hashlib.sha256((out / "registered-reader.stdout").read_bytes()).hexdigest()
    summary = dict(source_sha256=SOURCE_SHA256, tested_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), reader_stdout_sha256=digest,
        native_results_sha256=hashlib.sha256((out / "native-results.json").read_bytes()).hexdigest(),
        source_read_operations=calls, relative_tolerance=1e-7, absolute_tolerance_m3=1e-9,
        physical_count=len(physical), compared=compared, failures=failures)
    save(out / "source-comparison.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "compared"}), flush=True)
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
