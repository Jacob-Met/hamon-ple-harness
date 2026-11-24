"""Run the local admission audit against a supplied PLE harness checkout."""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkout", type=Path)
args = parser.parse_args()
root = args.checkout.resolve()
sys.path.insert(0, str(root))
import hamon_pareto as router

cover = "Run the calculation below and return its JSON result."
inventory = {"schema": "hamon.ple.intake.v1", "scanned_files": 0,
             "engineering_file_count": 0, "files": [], "warnings": [], "by_extension": {}}


def request(matrix, **fields):
    return {"mode": "control", "spec": {"mode": "state_space", "timebase": "discrete", "A": matrix, **fields}}


def certify(payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    result = router._wave8_certificate(cover + "\n" + text, inventory, 0, 0)
    return None if result is None else json.loads(result)


def faults(payload, edits):
    original = router.dispatch_ple_tool("hamon_conservation_cad_verifier", payload)
    accepted = {}
    for name, changes in edits:
        bad = deepcopy(original)
        bad.update(changes)
        with patch.object(router, "dispatch_ple_tool", return_value=bad):
            accepted[name] = certify(payload) is not None
    return accepted


literal = json.dumps(request([[88]]))
boundaries = {
    "positive_half": (request([[.5]]), True), "negative_two": (request([[-2]]), True),
    "jordan_half": (request([[.5, 2], [0, .5]]), True),
    "unit_rotation": (request([[0, -1], [1, 0]]), False),
    "reciprocal": (request([[2, 0], [0, .5]]), False),
    "decimal_stochastic": (request([[.1, .9], [.3, .7]]), False),
    "literal_near_one": (literal.replace("88", "0.99999999999999999"), False),
    "tiny_literal": (literal.replace("88", "1e-999999999"), False),
    "magnitude_small": (request([[1e-101]]), False), "magnitude_large": (request([[1e7]]), False),
    "extra_Q": (request([[.5]], Q=[[1]]), False),
    "CTMS": (request([[.999, .0095], [-.1903, .9039]]), True),
    "SciPy": (request([[.2, .7], [.5, -.9]]), True),
    "empty": (request([]), False), "three_states": (request([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), False),
    "boolean": (request([[True]]), False), "inside": (request([[math.nextafter(1., 0)]]), True),
    "outside": (request([[math.nextafter(1., math.inf)]]), True),
    "complex": (request([[.5, -.5], [.5, .5]]), True),
}
boundary_errors = [name for name, (payload, expected) in boundaries.items()
                   if (certify(payload) is not None) != expected]
admitted = singular = 0
grid_errors = []
for a, b, c, d in itertools.product((-1.5, -.5, .5, 1.5), repeat=4):
    trace = Fraction(a) + Fraction(d)
    det = Fraction(a) * Fraction(d) - Fraction(b) * Fraction(c)
    factors = (1 - trace + det, 1 + trace + det, 1 - det)
    unique = all(factors)
    stable = min(factors) > 0
    result = certify(request([[a, b], [c, d]]))
    admitted += int(result is not None)
    singular += int(not unique)
    if bool(result) != unique or (result and result["schur_stable"] != stable):
        grid_errors.append([a, b, c, d])

half = request([[.5]], sample_time_s=1)
original = router.dispatch_ple_tool("hamon_conservation_cad_verifier", half)
duplicate_p = deepcopy(original["stability_certificate"])
duplicate_p["P_exact"] = [["8/6"]]
original_faults = faults(half, [
    ("truthy_flag", {"schur_stable": 1}), ("wrong_timebase", {"timebase": "continuous"}),
    ("period_bool", {"sample_time_s": True}), ("period_float_for_int", {"sample_time_s": 1.0}),
    ("inexact_poly_encoding", {"char_poly_exact": [True, -.5]}),
    ("unreduced_poly", {"char_poly_exact": ["1", "-2/4"]}),
    ("unreduced_P", {"lyapunov_P_exact": [["8/6"]], "stability_certificate": duplicate_p}),
    ("wrong_float_P", {"lyapunov_P": [[0.0]]}), ("wrong_exact_P", {"lyapunov_P_exact": [["1"]]}),
    ("wrong_pole", {"eigenvalues_real_imag": [[200, 0]]}),
    ("wrong_float_status", {"lyapunov_float_status": "outside_float_range"}),
])
reciprocal = request([[2, 0], [0, .5]])
wrong = router.dispatch_ple_tool("hamon_conservation_cad_verifier", reciprocal)
p = [["-1/3", "0"], ["0", "4/3"]]
wrong.update(lyapunov_solution_status="unique", lyapunov_P_exact=p, lyapunov_pd=False, lyapunov_residual_exact=True)
wrong["stability_certificate"].update(solution_status="unique", P_exact=p, positive_definite=False, residual_exact=True)
with patch.object(router, "dispatch_ple_tool", return_value=wrong) as dispatch:
    original_faults["false_unique_reciprocal"] = certify(reciprocal) is not None
    assert not dispatch.called
report = {
    "router_sha256": hashlib.sha256((root / "hamon_pareto.py").read_bytes()).hexdigest(),
    "boundaries": len(boundaries), "boundary_errors": boundary_errors,
    "grid_total": 256, "grid_admitted": admitted, "grid_singular": singular, "grid_errors": grid_errors,
    "corruptions_admitted": original_faults,
}
print(json.dumps(report, indent=2))
raise SystemExit(bool(boundary_errors or grid_errors or any(original_faults.values())))
