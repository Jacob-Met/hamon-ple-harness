"""Exact-rank controls with analytically known modes for the registered state-space solver."""
import hashlib
import json
import math
from pathlib import Path
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


PROTOCOL = Path(__file__).parent / "control_rank" / "protocol.json"
RANK_FIELDS = ("controllability_rank", "observability_rank")
FLAGS = ("controllable", "observable")


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def power_of_two(text):
    mantissa, exponent = text.split("^")
    if mantissa not in ("2", "-2"):
        raise ValueError(f"Unexpected dyadic fixture {text}")
    return math.ldexp(-1.0 if mantissa == "-2" else 1.0, int(exponent))


def family_cases(family):
    # The expected ranks are the algebraic results in the frozen protocol.
    # No numerical rank routine or candidate solver constructs these labels.
    name = family["name"]
    diagonal = [[-1.0, 0.0], [0.0, -2.0]]
    negative_identity = [[-1.0, 0.0], [0.0, -1.0]]
    if "k" in family:
        for k in family["k"]:
            high, low = math.ldexp(1., k), math.ldexp(1., -k)
            if name == "diagonal_coordinate_units":
                inputs, outputs = [[1.], [high]], [[1., low]]
            elif name == "mimo_channel_units":
                inputs, outputs = [[high, 0.], [0., low]], [[low, 0.], [0., high]]
            elif name == "absent_second_mode":
                inputs, outputs = [[high], [0.]], [[low, 0.]]
            else:
                raise ValueError(name)
            yield str(k), diagonal, inputs, outputs, family["ranks"]
    elif name == "zero_channels":
        yield "zero", diagonal, [[0.], [0.]], [[0., 0.]], family["ranks"]
    elif name in ("coupling_before_rounding", "uncoupled_direction"):
        for value in family["delta"]:
            a = [[-1., 0.], [power_of_two(value), -1.]]
            if name == "coupling_before_rounding":
                inputs, outputs = [[1.], [1.]], [[1., 1.]]
            else:
                inputs, outputs = [[0.], [1.]], [[1., 0.]]
            yield value, a, inputs, outputs, family["ranks"]
    elif name == "zero_coupling":
        yield "zero", negative_identity, [[1.], [1.]], [[1., 1.]], family["ranks"]
    elif name == "instant_mimo_near_dependence":
        for value, ranks in zip(family["delta"], family["ranks_by_delta"], strict=True):
            b = [[1., 1.], [1., 1. + power_of_two(value)]]
            yield value, negative_identity, b, [list(column) for column in zip(*b)], ranks
    elif name == "jordan_chain":
        for n in family["n"]:
            shift = math.ldexp(1., -55)
            a = [[-1. if i == j else shift if i == j + 1 else 0.
                  for j in range(n)] for i in range(n)]
            e1, en = [1.] + [0.] * (n - 1), [0.] * (n - 1) + [1.]
            yield f"{n}-full", a, [[v] for v in e1], [en], [n, n]
            yield f"{n}-deficient", a, [[v] for v in en], [e1], family["deficient"]["ranks"]
    else:
        raise ValueError(name)


def route(spec):
    return run_router_conservation_cad_verifier({"mode": "control", "spec": spec})


class StateSpaceAlgebraicRankTests(unittest.TestCase):
    def test_frozen_rank_pairs_and_correct_and_opposite_claims(self):
        protocol = json.loads(PROTOCOL.read_text())
        rows = []
        for family in protocol["families"]:
            cases = list(family_cases(family))
            self.assertEqual(len(cases), family["count"], family["name"])
            for label, a, b, c, expected in cases:
                name = f"{family['name']}/{label}"
                spec = {"mode": "state_space", "A": a, "B": b, "C": c}
                flags = [rank == len(a) for rank in expected]
                row = {"case": name, "expected": expected, "expected_flags": flags,
                       "claims": [], "errors": []}
                try:
                    base = route(spec)
                    row["ranks"] = [base[field] for field in RANK_FIELDS]
                    row["flags"] = [base[field] for field in FLAGS]
                    row["rank_pair_correct"] = (
                        base["passed"] is True and row["ranks"] == expected
                        and all(type(value) is int for value in row["ranks"])
                        and all(value is target for value, target in zip(row["flags"], flags))
                    )
                    row["nonrank_sha256"] = digest({
                        key: value for key, value in base.items()
                        if key not in RANK_FIELDS and key not in FLAGS
                    })
                except Exception as error:
                    row["errors"].append(["base", type(error).__name__, str(error)])
                for field, expected_flag in zip(FLAGS, flags):
                    for supplied in (expected_flag, not expected_flag):
                        right = supplied is expected_flag
                        try:
                            result = route({**spec, "claimed_field": field, "claimed_value": supplied})
                            check = result["claim_check"]
                            row["claims"].append(
                                result["passed"] is right and check["within_tolerance"] is right
                                and check["field"] == field and check["actual"] == float(expected_flag)
                                and check["claimed"] == float(supplied)
                            )
                        except Exception as error:
                            row["errors"].append([field, type(error).__name__, str(error)])
                            row["claims"].append(False)
                rows.append(row)
                print("CONTROL_RANK_CASE " + json.dumps(row, sort_keys=True, separators=(",", ":")))
        pairs = sum(row.get("rank_pair_correct", False) for row in rows)
        claims = sum(sum(row["claims"]) for row in rows)
        total_claims = sum(len(row["claims"]) for row in rows)
        errors = sum(len(row["errors"]) for row in rows)
        # Per-case hashes and the digest of the full ordered map allow exact
        # comparison across the two original CI jobs, including non-rank fields.
        nonrank = digest([[row["case"], row.get("nonrank_sha256")] for row in rows])
        print(f"CONTROL_RANK_TOTAL physical={len(rows)} pairs={pairs} claims={claims}/{total_claims} "
              f"exceptions={errors} nonrank_sha256={nonrank}")
        self.assertEqual(len(rows), protocol["physical_case_count"])
        self.assertEqual((len(rows), pairs, claims, total_claims, errors), (49, 49, 196, 196, 0))

    def test_omitted_and_zero_width_channels(self):
        cases = (
            ({}, (None, None), (None, None)),
            ({"B": [[1.], [1.]]}, (2, None), (True, None)),
            ({"C": [1., 1.]}, (None, 2), (None, True)),
            ({"B": [[], []]}, (0, None), (False, None)),
            ({"C": []}, (None, 0), (None, False)),
            ({"B": [[], []], "C": []}, (0, 0), (False, False)),
        )
        for ports, ranks, flags in cases:
            with self.subTest(ports=ports):
                result = route({"mode": "state_space", "A": [[-1., 0.], [0., -2.]], **ports})
                self.assertTrue(result["passed"], result)
                self.assertEqual(tuple(result[field] for field in RANK_FIELDS), ranks)
                for field, expected in zip(FLAGS, flags):
                    self.assertIs(result[field], expected)
