"""Exercise the real Pi registration and TypeScript/Python boundary for Lambert."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RegisteredLambertTests(unittest.TestCase):
    def test_registered_solver_reference_and_failure_controls(self):
        positive = dict(subkind='lambert_transfer',mu=1,r1_m=[1,0,0],r2_m=[0,1,0],tof_s=math.pi/2)
        calls = [
            {'mode':'lagrangian_orbital','spec':positive},
            {'mode':'lagrangian_orbital','spec':{**positive,'transfer_path':'long','tof_s':3*math.pi/2}},
            {'mode':'lagrangian_orbital','spec':{**positive,'proposed_v1_m_s':[0,-1,0]}},
            {'mode':'lagrangian_orbital','spec':{**positive,'subkind':'orbital_transfer','r1':1,'r2':1}},
            {'mode':'lagrangian_orbital','spec':{**positive,'revolutions':1}},
            {'mode':'reference'},
        ]
        source = r"""
import registerTaskTools from './task-tools.ts';
const registered = new Map();
registerTaskTools({registerTool(t){registered.set(t.name,t)},on(){}});
const tool = registered.get('hamon_conservation_cad_verifier');
const results=[];
for (const [i,payload] of JSON.parse(process.env.LAMBERT_REGISTERED_CASES).entries()) {
  results.push(await tool.execute('lambert-'+i,payload,null,()=>{},{}));
}
console.log(JSON.stringify(results));
"""
        with tempfile.TemporaryDirectory() as taskdir:
            env = {**os.environ,'TASK_RELAY_URL':'http://127.0.0.1:1','TASK_WORKDIR':taskdir,
                   'TASK_HARNESS_SRC':str(ROOT),'TASK_HARNESS_PYTHON':sys.executable,
                   'LAMBERT_REGISTERED_CASES':json.dumps(calls)}
            proc = subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',source],
                cwd=ROOT,env=env,capture_output=True,text=True,timeout=45)
            self.assertEqual(proc.returncode,0,proc.stderr)
            results = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(list(Path(taskdir).iterdir()),[])
        self.assertEqual([r['details']['verified'] for r in results[:5]],[True,True,False,False,False])
        self.assertAlmostEqual(results[0]['details']['v1_m_s'][1],1.)
        self.assertAlmostEqual(results[1]['details']['v1_m_s'][1],-1.)
        self.assertIn('lambert_transfer',results[5]['content'][0]['text'])
        self.assertIn('endpoint_verified',results[5]['content'][0]['text'])

if __name__ == '__main__':
    unittest.main()
