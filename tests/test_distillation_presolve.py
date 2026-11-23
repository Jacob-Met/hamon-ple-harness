import json
import unittest
from unittest.mock import patch

import hamon_pareto


class DistillationPresolveTests(unittest.TestCase):
    inventory = {"schema": "hamon.ple.intake.v1", "scanned_files": 0,
                 "engineering_file_count": 0, "files": [], "warnings": [],
                 "by_extension": {}}
    spec = {"subkind": "distillation", "z_F": 0.5, "x_D": 0.95, "x_B": 0.05,
            "alpha": 2.5, "q": 0.6, "R_mult": 1.3}

    def request(self, **changes):
        return {"mode": "electrochem_separations", "spec": {**self.spec, **changes}}

    def route(self, request, inventory=None, **counts):
        return hamon_pareto.route_from_intake_prompt(
            json.dumps(self.inventory if inventory is None else inventory),
            instruction=request if isinstance(request, str) else json.dumps(request),
            **counts)

    def assert_abstains(self, request, **context):
        plan = self.route(request, **context)
        self.assertFalse(plan["presolve_verified"])
        self.assertNotIn("presolve_result_json", plan)

    def test_fractional_feed_uses_registered_result(self):
        request = self.request()
        plan = self.route(request)
        self.assertEqual(plan["thinking_tier"], "low")
        result = json.loads(plan["presolve_result_json"])
        self.assertEqual(result, hamon_pareto.dispatch_ple_tool(
            "hamon_conservation_cad_verifier", request))
        self.assertEqual(result["N_stages_integer"], 13)
        self.assertAlmostEqual(result["N_theoretical"], 12.72977479823362)
        self.assertLess(result["residual"], 1e-10)

    def test_ordered_batch_with_explicit_reflux_and_feed(self):
        second = self.request(R=3.0, F_mol_s=70)
        del second["spec"]["R_mult"]
        results = json.loads(self.route([self.request(), second])["presolve_result_json"])
        self.assertEqual([row["D_mol_s"] for row in results], [50, 35])
        self.assertEqual(results[1]["R_operating"], 3.0)

    def test_pinches_bad_numbers_and_ambiguous_requests_abstain(self):
        for invalid in (self.request(q=1.0, R_mult=0.95), self.request(alpha=True),
                        self.request(R=3.0), self.request(F_mol_s=-1),
                        self.request(q=float("nan")), self.request(x_D=.4),
                        self.request(domain="thermo_cycles"), self.request(comments="draw column"),
                        {**self.request(), "units": "kg/s"}, self.spec):
            with self.subTest(invalid=invalid):
                self.assert_abstains(invalid)
        self.assert_abstains([self.request(), self.request(q=1.0, R_mult=0.95)])
        wire = json.dumps(self.request())
        self.assert_abstains(wire.replace('"q":', '"q": 0.8, "q":'))
        self.assert_abstains(wire + "\nAlso prepare a PDF.")

    def test_files_services_and_deliverables_preserve_existing_route(self):
        self.assert_abstains(self.request(), inventory={**self.inventory, "files": ["feed.csv"]})
        self.assert_abstains(self.request(), mcp_server_count=1)
        self.assert_abstains(self.request(), deliverable_count=1)

    def test_solver_flag_alone_cannot_certify_wrong_streams(self):
        request = self.request()
        correct = hamon_pareto.dispatch_ple_tool("hamon_conservation_cad_verifier", request)
        for tamper in ({"D_mol_s": 51, "B_mol_s": 49}, {"L_rect_mol_s": 0},
                       {"V_strip_mol_s": -1}, {"L_strip_mol_s": 1},
                       {"R_operating": 3}, {"residual": 1e-8}, {"verified": False}):
            with self.subTest(tamper=tamper), patch.object(
                    hamon_pareto, "dispatch_ple_tool", return_value={**correct, **tamper}):
                self.assert_abstains(request)

    def table_request(self, reflux=12):
        spec = {"subkind": "distillation", "x_B": .05, "z_F": .5, "x_D": .95,
                "q": .5, "R": reflux,
                "equilibrium_table": [[0,0],[.25,.46],[.5,.63],[.7,.72],[.85,.94],[1,1]]}
        return {"mode": "electrochem_separations", "spec": spec}

    def test_table_pinches_abstain_after_registered_call(self):
        request = self.table_request(11.5)
        received = []
        real_dispatch = hamon_pareto.dispatch_ple_tool

        def observe(tool, payload):
            result = real_dispatch(tool, payload)
            received.append((tool, payload, result))
            return result

        with patch.object(hamon_pareto, "dispatch_ple_tool", side_effect=observe):
            plan = self.route(request)
        self.assertEqual(len(received), 1)
        tool, payload, result = received[0]
        self.assertEqual((tool,payload), ("hamon_conservation_cad_verifier",request))
        self.assertFalse(result["verified"])
        self.assertNotIn("N_stages_integer", result)
        self.assertEqual((result["min_operating_gap"],result["min_gap_composition"]), (0,.7))
        self.assertFalse(plan["presolve_verified"])
        self.assertEqual(plan["thinking_tier"], "medium")
        self.assertNotIn("presolve_result_json", plan)
        self.assert_abstains([self.table_request(),request])

    def test_table_certificate_preserves_the_registered_result(self):
        request = self.table_request()
        plan = self.route(request)
        self.assertTrue(plan["presolve_verified"])
        self.assertEqual(plan["thinking_tier"], "low")
        result = json.loads(plan["presolve_result_json"])
        self.assertEqual(result, hamon_pareto.dispatch_ple_tool(
            "hamon_conservation_cad_verifier", request))
        self.assertEqual((result["R_min"],result["N_stages_integer"],result["feed_stage"]),
                         (11.5,23,19))
        self.assertAlmostEqual(result["N_theoretical"],22.620006621335445)
        duplicate = self.route([request,request])
        self.assertEqual(json.loads(duplicate["presolve_result_json"]), [result,result])

    def test_incomplete_tables_and_additional_obligations_abstain(self):
        request = self.table_request()
        for change in ({"alpha":2.5},{"equilibrium_points":[[0,0],[1,1]]},
                       {"R_mult":1.3},{"equilibrium_table":None},
                       {"equilibrium_table":[[0,0],[.7,.72]]},
                       {"equilibrium_table":[[0,0],[.5,.7],[.6,.65],[1,1]]},
                       {"equilibrium_table":[[0,0],[".5",.7],[1,1]]}):
            self.assert_abstains({**request, "spec":{**request["spec"],**change}})
        self.assert_abstains(json.dumps(request)+"\nDraw each stage in a PDF.")
        self.assert_abstains(request,inventory={**self.inventory,"files":["vle.csv"]})

    def test_wrong_model_and_unusable_table_receipts_abstain(self):
        request = self.table_request()
        result = hamon_pareto.dispatch_ple_tool("hamon_conservation_cad_verifier", request)
        for correction in ({"equilibrium_model":"constant_alpha"}, {"min_operating_gap":0},
                           {"N_stages_integer":24}, {"feed_stage":24}, {"D_mol_s":51},
                           {"section_mode":"unknown"}):
            with patch.object(hamon_pareto,"dispatch_ple_tool", return_value={**result,**correction}):
                self.assert_abstains(request)
        flash = self.table_request(0)
        flash["spec"].update(x_B=.25,x_D=.75,z_F=.5,q=.5,equilibrium_table=[[0,0],[.25,.75],[1,1]])
        accepted = self.route(flash)
        self.assertTrue(accepted["presolve_verified"])
        self.assertEqual(json.loads(accepted["presolve_result_json"])["section_mode"],
                         "single_equilibrium_flash")
