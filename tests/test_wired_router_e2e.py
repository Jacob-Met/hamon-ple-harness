import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import hamon_drawing_reader
import hamon_pareto


SAMPLE_IFC = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('site.ifc','2026-11-21T00:00:00',('HAMON'),('HAMON'),'','','');
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCPROJECT('0001',$,'TestSite',$,$,$,$,$,#2);
#2=IFCUNITASSIGNMENT((#3));
#3=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);
#4=IFCCARTESIANPOINT((0.,0.));
#5=IFCDIRECTION((1.,0.));
#6=IFCAXIS2PLACEMENT2D(#4,#5);
#7=IFCCARTESIANPOINT((0.,0.,0.));
#11=IFCDIRECTION((0.,0.,1.));
#12=IFCDIRECTION((1.,0.,0.));
#8=IFCAXIS2PLACEMENT3D(#7,#11,#12);
#10=IFCCARTESIANPOINT((0.,0.,0.));
#13=IFCAXIS2PLACEMENT3D(#10,#11,#12);
#14=IFCLOCALPLACEMENT($,#13);
#20=IFCRECTANGLEPROFILEDEF(.AREA.,$,#6,2.0,1.0);
#21=IFCEXTRUDEDAREASOLID(#20,#8,#11,3.0);
#22=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#21));
#23=IFCPRODUCTDEFINITIONSHAPE($,$,(#22));
#24=IFCBEAM('B001',$,'StructuralBeam_S1',$,$,#14,#23,$,.BEAM.);
#30=IFCCARTESIANPOINT((0.5,0.2,1.0));
#31=IFCAXIS2PLACEMENT3D(#30,#11,#12);
#32=IFCLOCALPLACEMENT($,#31);
#33=IFCRECTANGLEPROFILEDEF(.AREA.,$,#6,0.6,0.6);
#34=IFCEXTRUDEDAREASOLID(#33,#8,#11,1.5);
#35=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#34));
#36=IFCPRODUCTDEFINITIONSHAPE($,$,(#35));
#37=IFCFLOWSEGMENT('M001',$,'SupplyDuct_MEP1',$,$,#32,#36,$);
ENDSEC;
END-ISO-10303-21;
"""


def _start_mock_relay(workdir: Path, log_ops: list):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            op = self.path.rsplit("/", 1)[-1]
            log_ops.append(op)
            target = (workdir / req.get("path", "")).resolve()
            if op == "exec":
                proc = subprocess.run(["bash", "-c", req["command"]], cwd=workdir,
                                      capture_output=True, text=True, timeout=20)
                body = json.dumps({"exit_code": proc.returncode,
                                   "output": proc.stdout + proc.stderr,
                                   "timed_out": False, "truncated": False})
            elif op == "read_file" and target.is_file():
                content = target.read_bytes()
                body = json.dumps({"ok": True, "size": len(content), "truncated": False,
                                   "content_b64": base64.b64encode(content).decode("ascii")})
            else:
                body = json.dumps({"ok": False, "error": "not_found"})
            raw = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def test_select_and_dispatch_ple_bim():
    here = Path(__file__).resolve().parent.parent
    tools = hamon_pareto.select_domain_tools("Check IFC4 BIM clash and PDF sheet scale", ["model.ifc", "plan.pdf"])
    assert tools == ("hamon_ifc_spatial_clash", "hamon_drawing_reader", "check_deliverable_preflight")

    real_dcbim = here / "artifacts/data-center-site-model/files/app/files/frontier__DCBim__p6_002/submission/model.ifc"
    with tempfile.TemporaryDirectory(prefix="ple-bim-test-") as td:
        sample_path = Path(td) / "sample.ifc"
        sample_path.write_text(SAMPLE_IFC, encoding="utf-8")
        targets = [sample_path]
        if real_dcbim.is_file() and "ISO-10303-21" in real_dcbim.read_text(encoding="utf-8", errors="ignore")[:1024]:
            targets.append(real_dcbim)
        for tgt in targets:
            res = hamon_pareto.dispatch_ple_tool("hamon_ifc_spatial_clash", {"path": str(tgt), "required_clearance_m": 0.05})
            assert res["ok"] is True
            pre = hamon_pareto.dispatch_ple_tool("check_deliverable_preflight", {"path": str(tgt)})
            assert pre["ok"] is True


def test_task_tools_ts_live_relay_roundtrip():
    from test_ifc_compound_mass import course

    here = Path(__file__).resolve().parent.parent
    relay_ops = []
    with tempfile.TemporaryDirectory(prefix="ple-ts-router-") as td:
        wdir = Path(td)
        (wdir / "site.ifc").write_text(SAMPLE_IFC, encoding="utf-8")
        (wdir / "reflected.ifc").write_text(course(reverse=False, reflect=True, right_scale=1.))
        (wdir / "open.ifc").write_text(course(declared_open=True, reflect=True, probe=True))
        pdf_bytes = hamon_drawing_reader.build_minimal_vector_pdf([
            {
                "width_pt": 612.0,
                "height_pt": 792.0,
                "texts": [(36.0, 36.0, "SCALE: 1:50")],
                "segments": [(72.0, 144.0, 288.0, 144.0)],
            }
        ])
        (wdir / "sheet.pdf").write_bytes(pdf_bytes)
        (wdir / "predictions.json").write_text(json.dumps({"task_a": 12.5, "task_b": 0.05}), encoding="utf-8")

        srv = _start_mock_relay(wdir, relay_ops)
        port = srv.server_address[1]
        try:
            node_script = """
import registerTaskTools, { PLE_DOMAIN_TOOL_NAMES } from './task-tools.ts';
const tools = new Map();
const pi = {
  registerTool(t) { tools.set(t.name, t); },
  on() {},
};
registerTaskTools(pi);
const ifcOut = await tools.get('hamon_ifc_spatial_clash').execute('1', { path: 'site.ifc', required_clearance_m: 0.08, cross_discipline_only: true }, null, () => {}, {});
const ifc = tools.get('hamon_ifc_spatial_clash');
const reflected = await ifc.execute('reflection', { path: 'reflected.ifc' }, null, () => {}, {});
const open = await ifc.execute('open', { path: 'open.ifc' }, null, () => {}, {});
const pdfOut = await tools.get('hamon_drawing_reader').execute('2', { path: 'sheet.pdf' }, null, () => {}, {});
const preOut = await tools.get('check_deliverable_preflight').execute('3', { path: 'predictions.json', expected_task_ids: ['task_a', 'task_b'] }, null, () => {}, {});
console.log(JSON.stringify({
  domain_names: PLE_DOMAIN_TOOL_NAMES,
  ifc_ok: ifcOut.details.ok,
  reflected_volume: reflected.details.total_solid_volume_m3,
  open_volume: open.details.total_solid_volume_m3,
  open_witness: open.details.all_pairs[0].closest_points_m,
  pdf_ok: pdfOut.details.ok,
  pre_ok: preOut.details.ok,
}));
"""
            env = {
                **dict(os.environ),
                "TASK_RELAY_URL": f"http://127.0.0.1:{port}",
                "TASK_WORKDIR": str(wdir),
                "TASK_HARNESS_PYTHON": sys.executable,
                "TASK_HARNESS_SRC": str(here),
            }
            proc = subprocess.run(
                ["node", "--experimental-strip-types", "--input-type=module", "-e", node_script],
                cwd=here,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            assert proc.returncode == 0, proc.stderr
            out = json.loads(proc.stdout.strip().splitlines()[-1])
            assert out["domain_names"] == ["hamon_ifc_spatial_clash", "hamon_drawing_reader", "hamon_conservation_cad_verifier", "check_deliverable_preflight"]
            assert out["ifc_ok"] is True
            assert abs(out["reflected_volume"] - 1/3) < 1e-12
            assert out["open_volume"] is None
            assert len(out["open_witness"]) == 2
            assert out["pdf_ok"] is True
            assert out["pre_ok"] is True
            assert len(relay_ops) >= 2
        finally:
            srv.shutdown()
