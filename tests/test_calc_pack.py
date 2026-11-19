import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engineering_toolkit as t

def sample():
    return {"schema":"hamon.engineering.calcpack.v1",
            "quantities":{
                "Fy":{"expr":"250*MPa","source":"material specification"},
                "S":{"expr":"0.001*m**3","source":"section modulus"},
                "FS":{"expr":"1.5","source":"chosen factor"},
                "Mallow":{"expr":"Fy*S/FS","to":"kN*m","source":"strength model"},
                "q":{"expr":"10*kN/m","source":"input line load"},
                "L":{"expr":"5*m","source":"input span"},
                "M":{"expr":"q*L**2/2","to":"kN*m","source":"cantilever UDL"},
                "Vshear":{"expr":"q*L","source":"cantilever shear"},
                "Vallow":{"expr":"30*kN","source":"shear limit"}},
            "checks":[{"id":"bending","lhs":"M","rhs":"Mallow","relation":"<=",
                       "source":"provided strength criterion"},
                      {"id":"shear","lhs":"Vshear","rhs":"Vallow","relation":"<=",
                       "source":"provided shear criterion"}]}

class CalculationPack(unittest.TestCase):
    def audit(self,data):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"pack.json";p.write_text(json.dumps(data))
            return t.calc_pack(str(p))
    def test_dependency_graph_and_mixed_failures(self):
        r=self.audit(sample())
        self.assertEqual(t.D(r["quantities"]["M"]["value"]),t.D("125"))
        self.assertEqual([x["passed"] for x in r["checks"]],[True,False])
        self.assertFalse(r["all_checks_pass"])
        self.assertFalse(r["code_compliance_checked"])
        self.assertEqual(len(r["input_sha256"]),64)
    def test_electrical_voltage_drop(self):
        d={"schema":"hamon.engineering.calcpack.v1",
           "quantities":{"I":{"expr":"100*A"},"R":{"expr":"0.5*ohm/km"},
                         "L":{"expr":"200*m"},"drop":{"expr":"2*I*R*L","to":"V"}},
           "checks":[{"id":"max_drop","lhs":"drop","rhs":"12*V","relation":"<="}]}
        r=self.audit(d)
        self.assertEqual(t.D(r["quantities"]["drop"]["value"]),t.D("20"))
        self.assertFalse(r["all_checks_pass"])
        self.assertIn("I: supplied quantity lacks a source",r["warnings"])
    def test_dag_cycle_and_ambiguous_units(self):
        d={"schema":"hamon.engineering.calcpack.v1",
           "quantities":{"a":{"expr":"b+1"},"b":{"expr":"a+1"}},"checks":[]}
        with self.assertRaisesRegex(ValueError,"circular"):self.audit(d)
        d["quantities"]={"a":{"expr":"2*lb"}}
        with self.assertRaisesRegex(ValueError,"ambiguous"):self.audit(d)
    def test_expression_injection_and_unit_collisions(self):
        d=sample();d["quantities"]["kN"]={"expr":"42"}
        with self.assertRaisesRegex(ValueError,"reserved"):self.audit(d)
        del d["quantities"]["kN"]
        d["quantities"]["evil"]={"expr":"__import__('os').system('whoami')"}
        with self.assertRaises(ValueError):self.audit(d)
    def test_dimensions_and_tolerance(self):
        d={"schema":"hamon.engineering.calcpack.v1",
           "quantities":{"x":{"expr":"100*kN"}},
           "checks":[{"id":"tolerance","lhs":"x","rhs":"100001*N",
                      "relation":"~=","relative_tolerance":"0.00002"}]}
        self.assertTrue(self.audit(d)["all_checks_pass"])
        d["checks"][0]["rhs"]="1*m"
        with self.assertRaisesRegex(ValueError,"dimension mismatch"):self.audit(d)
    def test_cli_exit_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"pack.json";p.write_text(json.dumps(sample()))
            cli=Path(__file__).resolve().parents[1]/"engineering_toolkit.py"
            cp=subprocess.run([sys.executable,str(cli),"calc-pack",str(p)],
                              text=True,capture_output=True)
            self.assertEqual(cp.returncode,3,cp.stderr)
            self.assertFalse(json.loads(cp.stdout)["all_checks_pass"])

if __name__=="__main__":unittest.main()
