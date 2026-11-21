"""Comprehensive multi-discipline tests for Dual-Solver Verifier, CAD Takeoff, and Pareto Router."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engineering_toolkit as tk
import hamon_cad_takeoff as ct
import hamon_dual_verifier as dv
import hamon_pareto as hp


class DualSolverVerifierTests(unittest.TestCase):
    def test_heat_exchanger_saturated_forward_rating(self):
        base = dict(m_hot_kg_s=1, cp_hot_J_kgK=150, Th_in_K=345,
                    m_cold_kg_s=1, cp_cold_J_kgK=270, Tc_in_K=302)
        # Independent boundary-value solutions for a known conductance.
        cases = [
            ("parallel", 270, 12000, 4146.428571428571, 317.35714285714283, 317.35714285714283),
            ("counterflow", 270, 12000, 6449.999999999999, 302.0, 325.8888888888889),
            ("counterflow", 150.000000075, 12000, 6370.3703719433015, 302.53086418704464, 344.46913579172076),
        ]
        for flow, cold_capacity, ua, duty, hot, cold in cases:
            result = dv.solve_domain_problem("heat_exchanger_ntu", dict(
                base, arrangement=flow, cp_cold_J_kgK=cold_capacity, UA_W_K=ua))
            assert result["dual_verified"]
            for key, expected in [("Q_duty_W", duty), ("Th_out_K", hot), ("Tc_out_K", cold)]:
                assert math.isclose(result[key], expected, rel_tol=2e-12)

    def test_heat_exchanger_rating_at_equal_and_nearly_equal_inlets(self):
        spec = {
            "m_hot_kg_s": 1, "cp_hot_J_kgK": 150, "Th_in_K": 310.0,
            "m_cold_kg_s": 1, "cp_cold_J_kgK": 270, "Tc_in_K": 310.0,
            "UA_W_K": 120.0,
        }
        for arrangement in ("counterflow", "parallel"):
            inputs = dict(spec, arrangement=arrangement)
            equal = dv.solve_heat_exchanger_ntu_dual(inputs)
            assert equal["dual_verified"] and 0 < equal["effectiveness"] < 1
            assert equal["Q_duty_W"] == 0
            assert equal["Th_out_K"] == equal["Tc_out_K"] == 310.0
            inputs["Th_in_K"] = math.nextafter(310.0, math.inf)
            near = dv.solve_heat_exchanger_ntu_dual(inputs)
            assert near["dual_verified"] and near["effectiveness"] == equal["effectiveness"]
            expected_duty = 150 * equal["effectiveness"] * (inputs["Th_in_K"] - 310.0)
            assert math.isclose(near["Q_duty_W"], expected_duty, rel_tol=1e-12)

    def test_si_7dim_rational_exponents_and_transcendental_guards(self):
        # Rational intermediate exponents: Manning R^(2/3), fire D*^(2/5), spillway H^(3/2)
        q_manning = dv.eval_quantity("(8*m)**(2/3)")
        self.assertAlmostEqual(q_manning.val, 4.0, places=9)
        self.assertEqual(q_manning.dim_list(), [0, 2 / 3, 0, 0, 0, 0, 0])

        q_fire = dv.eval_quantity("(32*m**(5/2))**(2/5)")
        self.assertAlmostEqual(q_fire.val, 4.0, places=9)
        self.assertEqual(q_fire.dim_list(), [0, 1, 0, 0, 0, 0, 0])

        q_trig = dv.eval_quantity("sin(30*deg) + cos(60*deg)")
        self.assertAlmostEqual(q_trig.val, 1.0, places=9)

        # Reject dimensional arguments to transcendental functions & mismatched additions
        for bad in ["sin(3*m)", "exp(2*s)", "ln(5*kg)", "10*kN + 5*kPa", "3*lb"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    dv.eval_quantity(bad)

    def test_dual_linear_and_complex_ac_nodal_kcl(self):
        # Complex AC nodal admittance matrix Y * V = I
        res = dv.solve_nodal_kcl({
            "num_nodes": 2,
            "admittances": [
                {"from": 1, "to": 0, "Y": [0.1, -0.05]},
                {"from": 1, "to": 2, "Y": [0.2, -0.1]},
                {"from": 2, "to": 0, "Y": [0.05, 0.025]},
            ],
            "current_injections": [
                {"node": 1, "I": [10.0, 0.0]},
            ],
        })
        self.assertTrue(res["dual_verified"])
        self.assertLess(res["dual_rel_discrepancy"], 1e-10)
        self.assertLess(res["kcl_max_residual_A"], 1e-10)

        # Pure-Python Householder QR fallback agreement when HAS_SCIPY=False
        orig_scipy = dv.HAS_SCIPY
        try:
            dv.HAS_SCIPY = False
            fallback = dv.dual_solve_linear([[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0])
            self.assertAlmostEqual(fallback["solution"][0], 1.0 / 11.0, places=10)
            self.assertAlmostEqual(fallback["solution"][1], 7.0 / 11.0, places=10)
            self.assertLess(fallback["dual_rel_discrepancy"], 1e-11)
        finally:
            dv.HAS_SCIPY = orig_scipy

    def test_beam_symbolic_macaulay_matches_64el_hermite_fem(self):
        # Random-parameterized cantilever, fixed-fixed, propped-cantilever, simply-supported beams
        rng = random.Random(20261121)
        for support in ("simply_supported", "cantilever", "fixed_fixed", "propped_cantilever"):
            L = rng.uniform(4.0, 14.0)
            q = rng.uniform(5e3, 35e3)
            P = rng.uniform(10e3, 80e3)
            xp = rng.uniform(0.2 * L, 0.8 * L)
            res = dv.solve_beam_dual({
                "support": support,
                "span": f"{L}*m",
                "udl": f"{q}*N/m",
                "E": "200*GPa",
                "I": "8.5e-5*m**4",
                "point_loads": [{"P": f"{P}*N", "x": f"{xp}*m"}],
            })
            self.assertTrue(res["dual_verified"])
            self.assertLess(res["dual_rel_discrepancy"], 1e-6)
            if support != "cantilever":
                self.assertLess(res["equilibrium_residual_N"], 1e-3)

    def test_colebrook_and_manning_nonlinear_dual_roots(self):
        hyd = dv.solve_pipe_colebrook_dual({
            "Q_m3_s": 0.045, "D_m": 0.15, "L_m": 220.0,
            "roughness_m": 4.5e-5, "rho_kg_m3": 998.0, "mu_Pa_s": 1.002e-3,
            "minor_loss_K": 3.5, "pump_efficiency": 0.78,
        })
        self.assertTrue(hyd["dual_verified"])
        self.assertLess(hyd["dual_rel_discrepancy"], 1e-9)
        self.assertGreater(hyd["shaft_power_W"], 0.0)

        chan = dv.solve_manning_channel_dual({
            "Q_m3_s": 18.5, "bottom_width_m": 3.2, "side_slope_z": 1.5,
            "manning_n": 0.015, "slope_S": 0.0018,
        })
        self.assertTrue(chan["dual_verified"])
        self.assertLess(chan["dual_rel_discrepancy"], 1e-9)
        self.assertGreater(chan["normal_depth_m"], 0.1)

    def test_dxf_spatial_takeoff_and_toolkit_cli(self):
        with tempfile.TemporaryDirectory() as td:
            dxf_file = Path(td) / "plan.dxf"
            lines = [
                "0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
                "0", "SECTION", "2", "ENTITIES",
                "0", "LINE", "8", "BEAMS", "10", "0", "20", "0", "11", "6000", "21", "0",
                "0", "LWPOLYLINE", "8", "SLAB", "70", "1",
                "10", "0", "20", "0", "10", "6000", "20", "0",
                "10", "6000", "20", "4000", "10", "0", "20", "4000",
                "0", "CIRCLE", "8", "CORES", "10", "1000", "20", "1000", "40", "500",
                "0", "INSERT", "8", "COLS", "2", "W14X90", "10", "0", "20", "0",
                "0", "INSERT", "8", "COLS", "2", "W14X90", "10", "6000", "20", "0",
                "0", "ENDSEC", "0", "EOF",
            ]
            dxf_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            cli = Path(__file__).resolve().parents[1] / "engineering_toolkit.py"
            cp = subprocess.run([sys.executable, str(cli), "dxf-takeoff", str(dxf_file)],
                                capture_output=True, text=True, check=True)
            out = json.loads(cp.stdout)
            self.assertTrue(out["ok"])
            self.assertAlmostEqual(out["line_length_m_by_layer"]["BEAMS"], 6.0, places=6)
            self.assertAlmostEqual(out["closed_area_m2_by_layer"]["SLAB"], 24.0, places=6)
            self.assertAlmostEqual(out["closed_area_m2_by_layer"]["CORES"], math.pi * 0.25, places=6)
            self.assertEqual(out["insert_count_by_block"]["W14X90"], 2)

            spec_file = Path(td) / "dv.json"
            spec_file.write_text(json.dumps({
                "mode": "si-eval",
                "expression": "12*kip / (6*in_**2)",
                "to": "ksi",
            }))
            cp2 = subprocess.run([sys.executable, str(cli), "dual-verify", str(spec_file)],
                                 capture_output=True, text=True, check=True)
            out2 = json.loads(cp2.stdout)
            self.assertTrue(out2["ok"])
            self.assertAlmostEqual(out2["value"], 2.0, places=9)

    def test_pareto_router_and_format_guard(self):
        # Simple single-calc task -> medium ($5.35/task vs $14.87/task)
        b_low = hp.route_task_budget(by_extension={".md": 1}, engineering_file_count=1)
        self.assertEqual(b_low["thinking_tier"], "medium")
        self.assertLess(b_low["estimated_cost_usd"], 6.0)

        # Drawing + workbook task -> high
        b_mid = hp.route_task_budget(by_extension={".pdf": 1, ".xlsx": 1}, engineering_file_count=3)
        self.assertEqual(b_mid["thinking_tier"], "high")

        # 3D BIM IFC + multi-sheet drawing -> max (or high on gemini-4-argon)
        b_max = hp.route_task_budget(by_extension={".ifc": 1, ".pdf": 3}, engineering_file_count=6)
        self.assertEqual(b_max["thinking_tier"], "max")
        b_argon = hp.route_task_budget(
            by_extension={".ifc": 1, ".pdf": 3},
            engineering_file_count=6,
            model_api="google-cloud/gemini-4-argon",
        )
        self.assertEqual(b_argon["thinking_tier"], "high")

        self.assertEqual(hp.sanitize_scalar_or_json_answer("```json\n{\"b\":1,\"a\":2}\n```", "json"), '{"a": 2, "b": 1}')
        self.assertEqual(hp.sanitize_scalar_or_json_answer("Answer: 1,234.56 kN", "number"), "1234.56")

        with tempfile.TemporaryDirectory() as td:
            bad_xlsx = Path(td) / "bad.xlsx"
            with zipfile.ZipFile(bad_xlsx, "w") as zf:
                zf.writestr("xl/worksheets/sheet1.xml",
                            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                            '<sheetData><row r="1"><c r="A1" t="e"><v>#DIV/0!</v></c></row></sheetData></worksheet>')
            audit = hp.audit_local_deliverable(bad_xlsx)
            self.assertFalse(audit["ok"])
            self.assertIn("xlsx_errors", audit["reason"])



    def test_wave2_dxf_bulge_and_multidiscipline_solvers(self):
        # Curved stadium bulge area/perimeter + paper-space group 67 exclusion
        verts = [(0.0, 0.0, 0.0), (30.0, 0.0, 1.0), (30.0, 8.0, 0.0), (0.0, 8.0, 1.0)]
        area, perim = ct.bulge_area_and_perimeter(verts, closed=True)
        self.assertAlmostEqual(area, 240.0 + math.pi * 16.0, places=9)
        self.assertAlmostEqual(perim, 60.0 + 8.0 * math.pi, places=9)

        # Complex symmetric non-Hermitian AC nodal admittance + exact 1e-16 V Decimal guard
        ac = dv.solve_nodal_kcl({
            "num_nodes": 3,
            "admittances": [
                {"from": 1, "to": 0, "Y": [0.12, -0.08]},
                {"from": 1, "to": 2, "Y": [0.25, -0.40]},
                {"from": 2, "to": 3, "Y": [0.05, 0.35]},
                {"from": 3, "to": 0, "Y": [0.08, -0.15]},
            ],
            "current_injections": [{"node": 1, "I": [12.0, -3.5]}, {"node": 3, "I": [-2.0, 4.0]}],
        })
        self.assertTrue(ac["dual_verified"])
        self.assertLess(ac["dual_rel_discrepancy"], 1e-11)
        self.assertFalse(dv.verify_quantity_inequality("1.0000000000000001*V", "<=", "1*V")["passes"])

        # Eccentric bolt group, 3D MEP clearance, NTU heat exchanger, Bélanger jump, HVAC coil+duct
        bg = dv.solve_bolt_group_eccentric_dual({
            "bolts_xy_m": [[-0.1, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]],
            "Px_N": 0.0, "Py_N": -120e3, "load_x_m": 0.25, "load_y_m": 0.0, "bolt_phi_Rn_N": 140e3,
        })
        self.assertTrue(bg["dual_verified"])
        mep = dv.solve_mep_routing_clearance_3d({
            "min_clearance_m": 0.10, "min_slope_ratio": 0.01,
            "segments": [{"run_id": 1, "p1": [0.0, 0.0, 3.52], "q1": [8.0, 0.0, 3.40], "outer_radius_m": 0.10}],
            "obstacles_aabb": [{"min": [3.5, -2.0, 2.50], "max": [4.5, 2.0, 3.00]}],
        })
        self.assertTrue(mep["dual_verified"] and mep["passes_routing"])
        hx = dv.solve_heat_exchanger_ntu_dual({
            "m_hot_kg_s": 2.0, "cp_hot_J_kgK": 4180.0, "Th_in_K": 365.0,
            "m_cold_kg_s": 2.5, "cp_cold_J_kgK": 4180.0, "Tc_in_K": 295.0,
            "UA_W_K": 14000.0, "arrangement": "counterflow",
        })
        self.assertTrue(hx["dual_verified"])
        hj = dv.solve_hydraulic_jump_dual({"Q_m3_s": 12.0, "bottom_width_m": 3.5, "side_slope_z": 0.0, "y1_m": 0.45})
        self.assertTrue(hj["dual_verified"])
        b_pre = hp.route_task_budget(by_extension={".md": 1}, engineering_file_count=1, presolve_verified=True)
        self.assertEqual(b_pre["thinking_tier"], "low")


if __name__ == "__main__":
    unittest.main()
