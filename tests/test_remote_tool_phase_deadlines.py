"""Check the registered remote exporter and preflight phase metadata."""
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r"""
import assert from 'node:assert/strict';
import http from 'node:http';
const seen = [];
const budget = JSON.parse(process.env.PHASE_TEST_CONFIG);
if (budget) process.env.PLE_DELIVERY_BUDGET = JSON.stringify(budget);
else delete process.env.PLE_DELIVERY_BUDGET;
const server = http.createServer((req, res) => {
  let text = '';
  req.on('data', part => { text += part; });
  req.on('end', () => {
    seen.push({route:req.url, body:JSON.parse(text)});
    res.writeHead(200, {'Content-Type':'application/json'});
    res.end(JSON.stringify({ok:true, exit_code:0, output:'{"ok":true}'}));
  });
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
process.env.TASK_RELAY_URL = 'http://127.0.0.1:' + server.address().port;
try {
  const {default:register} = await import('./task-tools.ts');
  const tools = new Map();
  register({registerTool(tool){tools.set(tool.name,tool)},on(){}});
  await tools.get('compile_witness_package').execute('w', {spec_path:'/fixture/source.json'});
  await tools.get('check_deliverable_preflight').execute('p', {path:'/fixture/receipt.json'});
  assert.equal(seen.length, 2);
  for (const r of seen) {
    assert.equal(r.route, '/exec');
    assert.equal(r.body.timeout_sec, 120);
    assert.equal(r.body.phase_deadline_ms, budget?.phase_deadline_ms);
  }
  console.log(JSON.stringify({phase:budget?.phase??null, calls:seen.length}));
} finally {
  await new Promise(resolve => server.close(resolve));
}
"""

@unittest.skipUnless((ROOT / "node_modules/@earendil-works/pi-coding-agent").is_dir(),
                     "install the submitted Pi via npm ci")
class RegisteredExecDeadlines(unittest.TestCase):
    def test_every_remote_exec_uses_the_current_phase(self):
        for phase in ("work", "delivery", None):
            with self.subTest(phase=phase):
                budget = {"phase":phase,"phase_deadline_ms":2000000000000} if phase else None
                result = subprocess.run(
                    ["node","--experimental-strip-types","--input-type=module","-e",SCRIPT],
                    cwd=ROOT,env={**os.environ,"PHASE_TEST_CONFIG":json.dumps(budget)},
                    capture_output=True,text=True,timeout=25)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                print(result.stdout.strip(),flush=True)

if __name__ == "__main__":
    unittest.main()
