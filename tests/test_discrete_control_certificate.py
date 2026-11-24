"""Independent admission and corrupted-output controls for direct control JSON."""
import copy
import json
import unittest
from unittest.mock import patch

import hamon_pareto as router
from hamon_pde_control_psychro_fem import verify_control_system


COVER = "Run the calculation below and return its JSON result.\n"
EMPTY = {"schema": "hamon.ple.intake.v1", "scanned_files": 0,
         "engineering_file_count": 0, "files": [], "warnings": [], "by_extension": {}}


def request(a, **extra):
    return {"mode": "control", "spec": {
        "mode": "state_space", "timebase": "discrete", "A": a, **extra,
    }}


class DiscreteControlCertificate(unittest.TestCase):
    def route(self, text, inventory=None, **context):
        return router.route_from_intake_prompt(
            json.dumps(EMPTY if inventory is None else inventory), instruction=text, **context)

    def test_decimal_and_unsupported_requests_abstain(self):
        # The first matrix fixes [1,1] in exact source-decimal arithmetic.
        # Binary64 instead has det(I-A)=3/72057594037927936.
        values = [
            request([[.1, .9], [.3, .7]]), request([[1]]),
            request([[2, 0], [0, .5]]), request([[True]]), request([["0.5"]]),
            request([[.5]], Q=[[1]]), request([[.5]], B=[[1]]),
            request([[.5]], claimed_field="schur_stable", claimed_value=True),
            request([[.5]], timebase="continuous"), request([[.5]], sample_time_s=True),
            request([[.5, 0, 0], [0, .5, 0], [0, 0, .5]]),
        ]
        texts = [COVER + json.dumps(value) for value in values]
        texts += [COVER + '{"mode":"control","spec":{"mode":"state_space",'
                  '"timebase":"discrete","A":[[' + literal + ']]}}'
                  for literal in ("0.99999999999999999", "1e-999999999")]
        for text in texts:
            with self.subTest(text=text):
                self.assertNotIn("presolve_result_json", self.route(text))

    def test_additional_obligations_keep_participant_turn(self):
        text = COVER + json.dumps(request([[.5]]))
        self.assertTrue(self.route(text)["deliver_verified_json"])
        for context in ({"mcp_server_count": 1}, {"deliverable_count": 1}):
            self.assertNotIn("presolve_result_json", self.route(text, **context))
        inventory = dict(EMPTY, scanned_files=1, files=[{"path": "notes.txt"}])
        self.assertNotIn("presolve_result_json", self.route(text, inventory))
        self.assertNotIn("presolve_result_json", self.route(text + "\nExplain assumptions."))
        bare = self.route(json.dumps(request([[.5]])))
        self.assertNotIn("deliver_verified_json", bare)

    def test_corrupted_full_result_cannot_be_delivered(self):
        req = request([[.5]], sample_time_s=1)
        original = verify_control_system(req["spec"])
        self.assertIsNotNone(router._wave8_result(req))
        changes = (
            {"sample_time_s": True}, {"char_poly_exact": [True, -.5]},
            {"char_poly_exact": ["1", "-0.5"]}, {"lyapunov_P": [[0.0]]},
            {"eigenvalues_real_imag": [[99.0, 0.0]]}, {"schur_stable": 1},
            {"lyapunov_residual_exact": False}, {"timebase": "continuous"},
            {"lyapunov_P_exact": [["0"]]},
        )
        for change in changes:
            forged = copy.deepcopy(original)
            forged.update(change)
            with self.subTest(change=change), patch.object(router, "dispatch_ple_tool", return_value=forged):
                self.assertIsNone(router._wave8_result(req))

    def test_nonunique_equation_refused_before_solver(self):
        # P=diag(-1/3,4/3) solves A^T P A-P=-I, but arbitrary P12 also solves it.
        # A forged 'unique' status must not turn one witness into uniqueness.
        with patch.object(router, "dispatch_ple_tool") as dispatch:
            self.assertIsNone(router._wave8_result(request([[2, 0], [0, .5]])))
            dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
