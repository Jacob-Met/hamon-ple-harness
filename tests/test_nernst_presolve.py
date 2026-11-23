"""The low budget requires a complete, independently checked open-circuit request."""
import hashlib
import json
import unittest
from unittest.mock import patch

import hamon_pareto as router


EMPTY = dict(schema="hamon.ple.intake.v1", scanned_files=0, engineering_file_count=0,
             by_extension={}, files=[], warnings=[])
CELL = dict(mode="electrochem_separations", spec=dict(
    subkind="electrochemistry", E0_V=0, n_electrons=2, T_K=298.15,
    activity_terms=[dict(species="Zn anode", stoich=1, activity=.1),
                    dict(species="Zn cathode", stoich=-1, activity=.5)]))


def route(text, inventory=EMPTY, **kwargs):
    return router.route_from_intake_prompt(json.dumps(inventory), instruction=text, **kwargs)


class NernstPresolve(unittest.TestCase):
    def test_complete_cell(self):
        text = json.dumps(CELL)
        result = route(text)
        self.assertEqual(result["thinking_tier"], "low")
        self.assertEqual(result["presolve_source_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertAlmostEqual(json.loads(result["presolve_result_json"])["E_nernst_V"], .0206753054531724)

    def test_abstains_on_incomplete_source_or_external_obligations(self):
        text = json.dumps(CELL)
        spec = CELL["spec"]
        changes = [dict(spec, current_A=1), dict(spec, activity_terms=[spec["activity_terms"][0]]*2),
                   {k:v for k,v in spec.items() if k != "T_K"}]
        for s in changes:
            self.assertFalse(route(json.dumps(dict(CELL,spec=s)))["presolve_verified"])
        for t in (text.replace('"T_K":', '"T_K":0,"T_K":'), text+'\nAlso report loaded voltage.'):
            self.assertFalse(route(t)["presolve_verified"])
        for inventory, opts in ((dict(EMPTY, warnings=["incomplete"]),{}),(EMPTY,dict(deliverable_count=1)),
                                (EMPTY,dict(mcp_server_count=1))):
            self.assertFalse(route(text,inventory,**opts)["presolve_verified"])

    def test_near_equilibrium_lexical_activity_is_not_rounded_away(self):
        spec = dict(CELL["spec"], activity_terms=[dict(species="ion", stoich=1, activity=1.0)])
        text = json.dumps(dict(CELL, spec=spec))
        self.assertTrue(route(text)["presolve_verified"])
        self.assertFalse(route(text.replace('"activity": 1.0', '"activity": 1.0000000000000001'))["presolve_verified"])

    def test_mixed_activity_and_distillation_requests_keep_their_own_verifiers(self):
        column = {"mode": "electrochem_separations", "spec": {
            "subkind": "distillation", "z_F": 0.5, "x_D": 0.95, "x_B": 0.05,
            "alpha": 2.5, "q": 0.6, "R_mult": 1.3}}
        result = route(json.dumps([CELL, column]))
        self.assertTrue(result["presolve_verified"])
        cell, tower = json.loads(result["presolve_result_json"])
        self.assertAlmostEqual(cell["E_nernst_V"], .0206753054531724)
        self.assertEqual(tower["N_stages_integer"], 13)

    def test_inconsistent_numeric_certificate(self):
        for k in ("E_nernst_V", "V_cell_V", "ln_Q_reaction", "delta_G_J_mol", "product_moles", "residual"):
            good=router.dispatch_ple_tool("hamon_conservation_cad_verifier", CELL)
            with patch.object(router, "dispatch_ple_tool", return_value=dict(good, **{k:1.0})):
                self.assertFalse(route(json.dumps(CELL))["presolve_verified"])
