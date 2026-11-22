from __future__ import annotations
import base64, json, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import hamon_conservation_cad_verifier as ccv
import hamon_pareto

SAMPLE_STEP_BREP = """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
#10=CARTESIAN_POINT('',(0.,0.,0.));
#11=CARTESIAN_POINT('',(200.,0.,0.));
#12=CARTESIAN_POINT('',(200.,100.,0.));
#13=CARTESIAN_POINT('',(0.,100.,0.));
#14=CARTESIAN_POINT('',(0.,0.,50.));
#15=CARTESIAN_POINT('',(200.,0.,50.));
#16=CARTESIAN_POINT('',(200.,100.,50.));
#17=CARTESIAN_POINT('',(0.,100.,50.));
#20=POLY_LOOP('',(#10,#13,#12,#11));
#21=POLY_LOOP('',(#14,#15,#16,#17));
#22=POLY_LOOP('',(#10,#11,#15,#14));
#23=POLY_LOOP('',(#13,#17,#16,#12));
#24=POLY_LOOP('',(#10,#14,#17,#13));
#25=POLY_LOOP('',(#11,#12,#16,#15));
#30=FACE_OUTER_BOUND('',#20,.T.);
#31=FACE_OUTER_BOUND('',#21,.T.);
#32=FACE_OUTER_BOUND('',#22,.T.);
#33=FACE_OUTER_BOUND('',#23,.T.);
#34=FACE_OUTER_BOUND('',#24,.T.);
#35=FACE_OUTER_BOUND('',#25,.T.);
#40=ADVANCED_FACE('',(#30),$,.T.);
#41=ADVANCED_FACE('',(#31),$,.T.);
#42=ADVANCED_FACE('',(#32),$,.T.);
#43=ADVANCED_FACE('',(#33),$,.T.);
#44=ADVANCED_FACE('',(#34),$,.T.);
#45=ADVANCED_FACE('',(#35),$,.T.);
#50=CLOSED_SHELL('',(#40,#41,#42,#43,#44,#45));
#51=MANIFOLD_SOLID_BREP('Block200x100x50mm',#50);
ENDSEC;
END-ISO-10303-21;
"""

def test_step_brep_volume_centroid_and_principal_inertia():
    res = ccv.verify_step_cad_geometry({"step_text": SAMPLE_STEP_BREP, "density_kg_m3": 7850.0})
    assert res["conservation_verified"] is True and res["watertight_manifold"] is True
    exp_mass = 7.85
    assert abs(res["mass_kg"] - exp_mass) < 1e-10
    exp_I = sorted([exp_mass*(0.1**2+0.05**2)/12.0, exp_mass*(0.2**2+0.05**2)/12.0, exp_mass*(0.2**2+0.1**2)/12.0])
    for got, exp in zip(res["principal_moments_kg_m2"], exp_I):
        assert abs(got - exp) / exp < 1e-12

def test_structural_beam_virtual_work_stress_deflection():
    res = ccv.verify_structural_mechanics({"boundary": "simply_supported_udl", "span_m": 6.0, "w_N_m": 12000.0, "E_Pa": 200e9, "I_m4": 8.5e-5, "c_m": 0.15, "area_m2": 0.0062, "yield_strength_Pa": 345e6, "safety_factor": 1.67})
    assert res["conservation_verified"] is True and res["dual_deflection_rel_error"] < 1e-10 and res["passes_stress"] is True

def test_spice_ac_kcl_kvl_tellegen_and_bad_voltage_rejection():
    deck = "RLC\nV1 n1 0 AC 240 0\nR1 n1 n2 6.0\nR2 n2 0 18.0\nL1 n2 n3 0.025\nC1 n3 0 80u\n.ac lin 1 60 60\n.end\n"
    res = ccv.verify_circuit_conservation({"netlist": deck})
    assert res["conservation_verified"] is True and res["kcl_max_residual_A"] < 1e-11 and res["tellegen_power_residual_VA"] < 1e-9
    bad = ccv.verify_circuit_conservation({"netlist": deck, "proposed_node_voltages": {"n2": [10.0, 0.0]}})
    assert bad["conservation_verified"] is False

def test_civil_geotech_and_thermal_fluid_and_dimensional():
    cg = ccv.verify_civil_geotech_balance({"cut_bank_m3": 8000.0, "shrink_factor": 0.9, "fill_compacted_m3": 6500.0, "footing_B_m": 2.4, "depth_Df_m": 1.5, "gamma_N_m3": 18000.0, "c_prime_Pa": 10000.0, "phi_deg": 32.0, "FS": 3.0})
    assert cg["conservation_verified"] is True and abs(cg["net_compacted_surplus_m3"] - 700.0) < 1e-9
    comb = ccv.verify_thermal_fluid_chemical_balance({"inlets": [{"mdot_kg_s": 1.6043, "ndot_mol_s": 100.0, "formula": "CH4"}, {"mdot_kg_s": 6.3998, "ndot_mol_s": 200.0, "formula": "O2"}], "outlets": [{"mdot_kg_s": 4.4010, "ndot_mol_s": 100.0, "formula": "CO2"}, {"mdot_kg_s": 3.6031, "ndot_mol_s": 200.0, "formula": "H2O"}], "rtol": 1e-3})
    assert comb["conservation_verified"] is True
    bad_hx = ccv.verify_thermal_fluid_chemical_balance({"Th_in_K": 350.0, "Th_out_K": 290.0, "Tc_in_K": 300.0, "Tc_out_K": 360.0, "Ch_W_K": 1000.0, "Cc_W_K": 1000.0})
    assert bad_hx["conservation_verified"] is False
    dim_ok = ccv.verify_dimensional_homogeneity({"lhs": {"value": 0.0, "unit": "psig"}, "rhs": {"value": 101.325, "unit": "kPa"}})
    assert dim_ok["conservation_verified"] is True

def test_router_dispatch_step_hygiene_and_zero_write_relay():
    tools = hamon_pareto.select_domain_tools("Compute STEP B-Rep inertia and SPICE KCL", ["rotor.step", "drv.cir"])
    assert "hamon_conservation_cad_verifier" in tools
    out = hamon_pareto.dispatch_ple_tool("hamon_conservation_cad_verifier", {"mode": "step_cad", "step_text": SAMPLE_STEP_BREP})
    assert out["conservation_verified"] is True
    here = Path(__file__).resolve().parent.parent
    relay_ops = []
    with tempfile.TemporaryDirectory() as td:
        wdir = Path(td)
        (wdir / "part.step").write_text(SAMPLE_STEP_BREP, encoding="utf-8")
        (wdir / "bad.step").write_text("INVALID", encoding="utf-8")
        assert hamon_pareto.audit_local_deliverable(wdir / "part.step")["ok"] is True
        assert hamon_pareto.audit_local_deliverable(wdir / "bad.step")["ok"] is False
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
                relay_ops.append(self.path)
                target = (wdir / req.get("path", "")).resolve()
                body = json.dumps({"ok": True, "content_b64": base64.b64encode(target.read_bytes()).decode("ascii")}) if (self.path == "/read_file" and target.is_file()) else json.dumps({"ok": False})
                raw = body.encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def log_message(self, *a): pass
        srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            env = dict(**__import__("os").environ, TASK_RELAY_URL=f"http://127.0.0.1:{srv.server_port}", TASK_WORKDIR=str(wdir), TASK_HARNESS_PYTHON=sys.executable, TASK_HARNESS_SRC=str(here))
            proc = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", "import registerTaskTools from './task-tools.ts'; const reg={}; registerTaskTools({registerTool(t){reg[t.name]=t;},on(){}}); const r=await reg['hamon_conservation_cad_verifier'].execute('1',{path:'part.step',spec:{density_kg_m3:7850}}); console.log(JSON.stringify(r.details));"], cwd=str(here), env=env, text=True, capture_output=True)
            assert proc.returncode == 0, proc.stderr
            assert json.loads(proc.stdout.strip().splitlines()[-1])["conservation_verified"] is True
            assert relay_ops == ["/read_file"]
        finally:
            srv.shutdown()
