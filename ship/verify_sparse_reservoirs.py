"""Cold paired source benchmark; analytic physics is required in both arms."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_pipe_sparse_reservoirs import manufactured_case


def case(name):
    if name == "long_ring":
        return manufactured_case(1025, [(i, (i + 1) % 1025) for i in range(1025)], {0, 341, 1024})
    if name == "dense_mesh":
        return manufactured_case(33, [(i, j) for i in range(33) for j in range(i + 1, 33)], {0, 11, 32})
    if name == "parallel_exchanges":
        return manufactured_case(3, [(0, 1), (2, 1)] * 257, {0, 2})
    raise ValueError(name)


def worker(module_path, name):
    module_spec = importlib.util.spec_from_file_location("measured_hydraulics", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    problem, expected_flows, expected_heads, expected_supplies = case(name)
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    result = module.verify_pipe_network(problem)
    cpu_s, wall_s = time.process_time() - cpu_start, time.perf_counter() - wall_start
    if not result["conservation_verified"]:
        raise RuntimeError(result)
    flow_error = max(abs(result["flows_m3_s"][k] - v) for k, v in expected_flows.items())
    head_error = max(abs(result["heads_m"][str(i)] - v) for i, v in enumerate(expected_heads))
    supply_error = max(abs(result["reservoir_supplies_m3_s"][k] - v)
                       for k, v in expected_supplies.items())
    if flow_error > 1e-14 or head_error > 2e-13 or supply_error > 1e-14:
        raise RuntimeError({"flow_error": flow_error, "head_error": head_error,
                            "supply_error": supply_error})
    print(json.dumps({"case": name, "cpu_s": cpu_s, "wall_s": wall_s,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "max_flow_error": flow_error, "max_head_error": head_error,
                      "max_supply_error": supply_error,
                      "mass_residual": result["max_mass_residual_m3_s"],
                      "energy_residual": result["max_energy_residual_m"],
                      "independent_flow_difference": result["max_nodal_cycle_flow_difference_m3_s"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--worker-module")
    parser.add_argument("--case")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker_module:
        worker(args.worker_module, args.case)
        return
    if not args.baseline or not args.output:
        parser.error("--baseline and --output are required for the paired course")
    sources = {"baseline": str(Path(args.baseline).resolve()),
               "candidate": str(ROOT / "hamon_pipe_network.py")}
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    records = []
    for name in ("long_ring", "dense_mesh", "parallel_exchanges"):
        for arm in ("baseline", "candidate", "candidate", "baseline"):
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                  "--worker-module", sources[arm], "--case", name],
                                 capture_output=True, text=True, env=env, timeout=120)
            if run.returncode:
                raise RuntimeError({"arm": arm, "case": name, "stdout": run.stdout,
                                    "stderr": run.stderr, "returncode": run.returncode})
            record = json.loads(run.stdout)
            record["arm"] = arm
            records.append(record)
            print(json.dumps(record), flush=True)
    report = {"scope": "synthetic analytic hydraulics; not an official PLE score",
              "order_per_case": ["baseline", "candidate", "candidate", "baseline"],
              "threads": 1, "records": records}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
