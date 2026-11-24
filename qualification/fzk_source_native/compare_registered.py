"""Compare the real registered reader on the intact FZK source with independent native quantities."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_wired_router_e2e import _start_mock_relay

SOURCE_SHA256 = "ea6f04eaf92fac4d7ad0038bc3d2dfea4c094dd3f516ecc33c50bf1835ca108d"

def main(source, native_path, output):
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == SOURCE_SHA256
    native = json.loads(native_path.read_text())
    assert native["source_sha256"] == SOURCE_SHA256
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        shutil.copyfile(source, work / "FZK-Haus.ifc")
        calls = []
        relay = _start_mock_relay(work, calls)
        try:
            script = """
import register from './task-tools.ts';
const tools = new Map();
register({registerTool(t) { tools.set(t.name,t); }, on() {}});
const result = await tools.get('hamon_ifc_spatial_clash').execute(
    'fzk-original', {path:'FZK-Haus.ifc',allow_partial:true,cross_discipline_only:true},
    null, () => {}, {});
console.log(JSON.stringify(result.details));
"""
            env = {**os.environ, "TASK_WORKDIR": folder,
                   "TASK_HARNESS_PYTHON": sys.executable, "TASK_HARNESS_SRC": str(ROOT),
                   "TASK_RELAY_URL": f"http://127.0.0.1:{relay.server_address[1]}"}
            completed = subprocess.run(["node", "--experimental-strip-types", "--input-type=module",
                                        "-e", script], cwd=ROOT, env=env, text=True,
                                       capture_output=True, timeout=300, check=True)
            report = json.loads(completed.stdout.strip().splitlines()[-1])
            (output / "registered-reader.json").write_text(json.dumps(report, indent=2))
            assert calls == ["read_file"], calls
        finally:
            relay.shutdown()
            relay.server_close()
    assert report["status"] == "partial" and report["ok"] is False, report
    assert report["physical_element_count"] == 82
    assert report["known_mesh_count"] == 81
    assert report["unresolved_geometry_step_ids"] == [14502]
    assert report["total_solid_volume_m3"] is None and report["hard_clash_count"] is None
    assert len(report["void_relationships"]) == 17 and len(report["filling_relationships"]) == 16
    assert report["requested_pair_count"] == 1197
    assert report["evaluated_pair_count"] == 1134
    assert report["missing_geometry_pair_count"] == 63
    assert report["unresolved_pair_count"] == 0 and report["certified_empty_pair_count"] == 0
    assert report["bound_only_pair_count"] + report["measured_pair_count"] == 1134

    rows = {row["step_id"]: row for row in report["products"] if row["role"] == "physical"}
    qualified = {int(key): row["volume_m3"] for key, row in native["per_product"].items()
                 if row["status"] == "known"}
    assert set(report["known_volume_step_ids"]) == set(qualified)
    errors = []
    for step, reference in qualified.items():
        row = rows[step]
        assert row["geometry_status"] == row["volume_status"] == "known", row
        actual = row["volume_m3"]
        error = abs(actual - reference)
        assert math.isfinite(actual) and math.isclose(actual, reference, rel_tol=1e-9, abs_tol=1e-9), (step, actual, reference)
        errors.append((error, step, actual, reference))
    assert rows[14502]["volume_m3"] is None
    assert math.isclose(report["known_occurrence_volume_sum_m3"],
                        math.fsum(qualified.values()), rel_tol=1e-9, abs_tol=1e-8)
    pair_receipts = []
    for host, window in ((60012,66459),(67828,74280)):
        pair, = [row for row in report["all_pairs"]
                 if {row["element_a"],row["element_b"]} == {f"#{host}",f"#{window}"}]
        # Surface touching is not a positive material-intersection volume.
        assert pair["min_distance_m"] == 0 and pair["material_interference"] == "unverified"
        reference = native["openings"][str(host)]
        native_distances = [part["net"]["distance_m"] for part in reference["window_parts"]]
        assert math.isclose(pair["min_distance_m"], min(native_distances), rel_tol=0, abs_tol=1e-9)
        assert sum(distance <= 1e-9 for distance in native_distances) == 2
        assert math.isclose(rows[window]["volume_m3"], reference["window_union_m3"],
                            rel_tol=0, abs_tol=1e-11)
        assert math.isclose(rows[host]["volume_m3"] + rows[window]["volume_m3"],
                            reference["wall_window_union_m3"], rel_tol=0, abs_tol=1e-9)
        assert all(abs(part["net"]["common_m3"]) <= 1e-12 for part in reference["window_parts"])
        pair_receipts.append({"host_step_id":host,"window_step_id":window,
                              "candidate":pair,"native_material_common_m3":0.0})
    from hamon_ifc_spatial_clash import run_router_ifc_clash
    complete_only = run_router_ifc_clash({"ifc_text":original.decode()})
    assert complete_only["ok"] is False and complete_only["total_solid_volume_m3"] is None
    worst = sorted(errors, reverse=True)[:5]
    receipt = {"source_sha256":SOURCE_SHA256,
               "candidate_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               "scope":"81 per-occurrence quantities and two wall/window relations; not a native oracle for the full pair matrix",
               "registered_task_reads":calls,"physical_count":82,"quantity_comparisons":len(errors),
               "unresolved_geometry_step_ids":[14502],"worst_volume_errors":worst,
               "pair_coverage":{"requested":1197,"evaluated":1134,"missing_geometry":63,
                               "measured":report["measured_pair_count"],"bounds_only":report["bound_only_pair_count"]},
               "wall_window_receipts":pair_receipts,"whole_model_volume":None,
               "organizer_score":None,"portal_submissions":0}
    (output / "source-comparison.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__ == "__main__":
    main(*(Path(value) for value in sys.argv[1:4]))
