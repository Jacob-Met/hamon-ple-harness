"""Independent public RLC endpoint controls with a fixed process deadline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import mpmath as mp

CHILD = """import json,sys,time
from hamon_wave6_physics_verifiers import solve_transmission_line_rlc
p=json.load(sys.stdin)
started=time.perf_counter()
result=solve_transmission_line_rlc(p)
print(json.dumps(dict(result=result,solver_seconds=time.perf_counter()-started),allow_nan=False))
"""


def reference(case):
    with mp.workdps(80):
        r, l, c, vs, t = [mp.mpf(str(case[key])) for key in ("R", "L", "C", "V_s", "t")]
        v0, i0 = mp.mpf(str(case.get("v_c0", 0))), mp.mpf(str(case.get("i_l0", 0)))
        # Independent general-purpose high-precision matrix exponential in
        # physical voltage/current coordinates, without the candidate's roots.
        matrix = mp.matrix([[0, 1/c], [-1/l, -r/l]])
        state = mp.expm(matrix*t) * mp.matrix([v0-vs, i0])
        return dict(voltage=float(vs+state[0]), current=float(state[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=4.)
    parser.add_argument("--without-scipy", action="store_true")
    args=parser.parse_args()
    if args.workers not in (1, 2):
        parser.error("workers must be one or two")
    cases=json.loads(args.cases.read_text())
    references=[reference(case) for case in cases]
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    child=CHILD
    if args.without_scipy:
        child='''import importlib.abc,sys
class RejectScipy(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=="scipy" or fullname.startswith("scipy."):
   raise ModuleNotFoundError("SciPy deliberately absent",name=fullname)
sys.meta_path.insert(0,RejectScipy())
'''+child
    def evaluate(pair):
        case,expected=pair
        started=time.perf_counter()
        row=dict(case=case,reference=expected)
        try:
            process=subprocess.run([sys.executable,"-c",child],input=json.dumps(case),
                cwd=args.source,env=env,capture_output=True,text=True,timeout=args.timeout)
            row.update(wall_seconds=time.perf_counter()-started,returncode=process.returncode)
            if process.returncode:
                row.update(passed=False,error=process.stderr[-1000:])
            else:
                output=json.loads(process.stdout)
                actual=output["result"]
                impedance=math.sqrt(case["L"]/case["C"])
                voltage_scale=max(abs(case["V_s"]),abs(case.get("v_c0",0)),
                                  abs(case.get("i_l0",0)*impedance),1e-300)
                current_scale=max(abs(expected["current"]),
                                  voltage_scale/max(case["R"],impedance),1e-300)
                error=max(abs(actual["v_c_V"]-expected["voltage"])/voltage_scale,
                          abs(actual["i_l_A"]-expected["current"])/current_scale)
                row.update(output=output,scaled_error=error,
                           passed=bool(actual["verified"] and error<1e-10))
        except subprocess.TimeoutExpired:
            row.update(passed=False,error="process deadline",wall_seconds=time.perf_counter()-started)
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(evaluate,zip(cases,references)))
    version=subprocess.check_output(["git","rev-parse","HEAD"],cwd=args.source,text=True).strip()
    source=args.source/"hamon_wave6_physics_verifiers.py"
    report=dict(source_commit=version,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        python=platform.python_version(),mpmath=mp.__version__,without_scipy=args.without_scipy,
        process_deadline_seconds=args.timeout,workers=args.workers,
        passed=sum(row["passed"] for row in results),total=len(results),
        max_scaled_error=max((r["scaled_error"] for r in results if "scaled_error" in r),default=None),
        timeouts=sum(row.get("error")=="process deadline" for row in results),
        scope="Manufactured public linear-circuit controls; no organizer task data, model run or contest score.",
        results=results)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key!="results"}),flush=True)
    for row in results:
        if not row["passed"]:
            print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
