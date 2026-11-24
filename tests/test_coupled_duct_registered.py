"""Exercise actual Pi registration and TypeScript/Python dispatch for ducts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RegisteredCoupledDuctTests(unittest.TestCase):
    def test_registered_coupled_flow_and_rejected_separate_models(self):
        p = dict(subkind='fanno_rayleigh',M1=.5,T01=315.,P1=101325.,
                 L=.5,D_h=.1,f=.01,q_J_kg=40000.)
        specs = [p,dict(p,M1=2.),dict(p,subkind='fanno'),
                 dict(p,subkind='rayleigh'),dict(p,q_J_kg=-40000.),
                 dict(p,gamma=1.4,R=287.05,cp=500.)]
        calls = [{'mode':'compressible_acoustics','spec':s} for s in specs]+[{'mode':'reference'}]
        source = r"""
import registerTaskTools from './task-tools.ts';
const registered = new Map();
registerTaskTools({registerTool(t){registered.set(t.name,t)},on(){}});
const tool = registered.get('hamon_conservation_cad_verifier');
const results=[];
for (const [i,payload] of JSON.parse(process.env.COUPLED_DUCT_CASES).entries()) {
  results.push(await tool.execute('duct-'+i,payload,null,()=>{},{}));
}
console.log(JSON.stringify(results));
"""
        with tempfile.TemporaryDirectory() as taskdir:
            env = {**os.environ,'TASK_RELAY_URL':'http://127.0.0.1:1','TASK_WORKDIR':taskdir,
                   'TASK_HARNESS_SRC':str(ROOT),'TASK_HARNESS_PYTHON':sys.executable,
                   'COUPLED_DUCT_CASES':json.dumps(calls)}
            proc = subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',source],
                                  cwd=ROOT,env=env,capture_output=True,text=True,timeout=45)
            self.assertEqual(proc.returncode,0,proc.stderr)
            results = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(list(Path(taskdir).iterdir()),[])
        self.assertEqual([r['details']['verified'] for r in results[:6]],
                         [True,False,False,False,True,False])
        self.assertAlmostEqual(results[0]['details']['M2'],.6069166972327382,places=12)
        self.assertTrue(results[1]['details']['choked'])
        self.assertNotIn('M2',results[1]['details'])
        self.assertIn('fanno_rayleigh',results[6]['content'][0]['text'])
        self.assertIn('uniform heat',results[6]['content'][0]['text'])


if __name__ == '__main__':
    unittest.main()
