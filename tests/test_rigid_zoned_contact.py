import copy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from hamon_zoned_contact import verify_rigid_zoned_contact
from ship.verify_zoned_contact import reference_cases, validate


def rectangle(left,right,bottom=-1,top=1):
    return [(left,bottom),(right,bottom),(right,top),(left,top)]


class RigidZonedContactTests(unittest.TestCase):
    def test_48_preregistered_rational_references_and_boundary_integrals(self):
        cases=list(reference_cases())
        self.assertEqual(len(cases),48)
        for case in cases:
            with self.subTest(case=case['name']):
                validate(case,verify_rigid_zoned_contact(case['spec']))

    def test_common_stiffness_scaling_preserves_pressure_and_rescales_displacement(self):
        for factor in (F(1,8),F(10),F(1024)):
            for case in reference_cases(factor):
                if case['origin']!=(0,0) or case['matrix']!=(1,0,0,1):
                    continue
                with self.subTest(factor=str(factor),case=case['name']):
                    validate(case,verify_rigid_zoned_contact(case['spec']))

    def test_region_subdivision_and_reordering_preserve_one_field(self):
        for case in reference_cases():
            if case['origin']!=(0,0) or case['matrix']!=(1,0,0,1):
                continue
            original=verify_rigid_zoned_contact(case['spec'])
            spec=copy.deepcopy(case['spec'])
            right=spec['support_regions'].pop()
            spec['support_regions'] += [
                {**right,'id':'right-lower','outer_vertices_m':rectangle(0,2,-1,0)},
                {**right,'id':'right-upper','outer_vertices_m':rectangle(0,2,0,1)}]
            for regions in (spec['support_regions'],list(reversed(spec['support_regions']))):
                out=verify_rigid_zoned_contact({**spec,'support_regions':regions})
                self.assertTrue(out['conservation_verified'],out)
                for key in ('contact_area_m2','q_max_Pa','integrated_vertical_resultant_N',
                            'integrated_moment_x_Nm','integrated_moment_y_Nm'):
                    self.assertAlmostEqual(out[key],original[key],delta=1e-8)
                p,q=out['displacement_plane'],original['displacement_plane']
                for point in ((-2,-1),(0,0),(2,1)):
                    def value(plane):
                        return F(plane['displacement_at_reference_m'])+sum(
                            F(g)*(F(x)-F(r)) for g,x,r in zip(
                                plane['gradient_m_per_m'],point,plane['reference_point_m']))
                    self.assertAlmostEqual(float(value(p)),float(value(q)),delta=1e-8)

    def test_disconnected_support_and_fully_unloaded_region(self):
        spec=dict(calculation='rigid_zoned_contact',support_regions=[
            dict(id='left',vertical_stiffness_N_m3=1,outer_vertices_m=rectangle(-3,-2)),
            dict(id='right',vertical_stiffness_N_m3=4,outer_vertices_m=rectangle(2,3))],
            vertical_resultant_N=20,moment_x_Nm=152/3,moment_y_Nm=0)
        out=verify_rigid_zoned_contact(spec)
        self.assertTrue(out['conservation_verified'],out)
        self.assertAlmostEqual(out['contact_area_m2'],2)
        self.assertAlmostEqual(out['q_max_Pa'],12)
        self.assertEqual(out['regions'][0]['contact_patches'],[])
        self.assertEqual(out['regions'][0]['integrated_vertical_resultant_N'],0)
        self.assertAlmostEqual(out['regions'][1]['integrated_vertical_resultant_N'],20)

    def test_region_exactly_fills_another_region_hole(self):
        spec=dict(calculation='rigid_zoned_contact',support_regions=[
            dict(id='outer',vertical_stiffness_N_m3=1,outer_vertices_m=rectangle(-2,2,-2,2),
                 hole_vertices_m=[rectangle(-1,1)]),
            dict(id='inner',vertical_stiffness_N_m3=4,outer_vertices_m=rectangle(-1,1))],
            vertical_resultant_N=28,moment_x_Nm=0,moment_y_Nm=0)
        out=verify_rigid_zoned_contact(spec)
        self.assertTrue(out['conservation_verified'],out)
        self.assertEqual(out['contact_area_m2'],16)
        self.assertEqual(out['q_max_Pa'],4)
        self.assertEqual([r['integrated_vertical_resultant_N'] for r in out['regions']],[12,16])

    def test_invalid_stiffness_overlap_unknown_fields_and_claims(self):
        spec=next(c['spec'] for c in reference_cases() if '-k4-r0-o0-0' in c['name'])
        cases=[]
        for value in (0,-1,True,float('nan'),float('inf')):
            bad=copy.deepcopy(spec)
            bad['support_regions'][0]['vertical_stiffness_N_m3']=value
            cases.append(bad)
        missing=copy.deepcopy(spec)
        del missing['support_regions'][0]['vertical_stiffness_N_m3']
        cases.append(missing)
        duplicate=copy.deepcopy(spec)
        duplicate['support_regions'][1]['id']='left'
        cases.append(duplicate)
        extra=copy.deepcopy(spec)
        extra['support_regions'][0]['unknown']=1
        cases += [extra,{**spec,'unknown':1},{**spec,'pressure_model':'independent_region_planes'}]
        overlap=copy.deepcopy(spec)
        # One binary64 step inside the other region remains positive-area overlap.
        overlap['support_regions'][1]['outer_vertices_m']=rectangle(-math.ulp(1.),2)
        cases.append(overlap)
        for bad in cases:
            with self.subTest(input=bad):
                out=verify_rigid_zoned_contact(bad)
                self.assertFalse(out['ok'],out)
                self.assertFalse(out['conservation_verified'],out)
        edge=verify_rigid_zoned_contact({**spec,'vertical_resultant_N':1,'moment_x_Nm':2,'moment_y_Nm':0})
        self.assertEqual(edge['equilibrium_status'],'no_finite_equilibrium')
        unloaded=verify_rigid_zoned_contact({**spec,'vertical_resultant_N':0,'moment_x_Nm':0,'moment_y_Nm':0})
        self.assertTrue(unloaded['conservation_verified'])
        self.assertEqual(unloaded['equilibrium_status'],'unloaded')
        self.assertIsNone(unloaded['contact_area_m2'])
        wrong=verify_rigid_zoned_contact({**spec,'proposed_q_max_Pa':4.75})
        self.assertTrue(wrong['ok'])
        self.assertFalse(wrong['conservation_verified'])
        self.assertAlmostEqual(wrong['q_max_Pa'],4)

    def test_registered_python_dual_cli_and_dimensioned_stiffness(self):
        from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as verify
        from hamon_dual_verifier import run_dual_verify_spec as dual_verify
        case=next(c for c in reference_cases() if c['name']=='horizontal-k4-r0-o0-0')
        expected=verify_rigid_zoned_contact(case['spec'])
        self.assertEqual(verify(case['spec']),expected)
        dual=dual_verify(dict(kind='rigid_zoned_contact',spec=case['spec']))
        self.assertTrue(dual['dual_verified'],dual)
        validate(case,dual)
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/
            'hamon_conservation_cad_verifier.py'),'--json-stdin'],
            input=json.dumps(case['spec']),capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertEqual(json.loads(completed.stdout),expected)
        units=copy.deepcopy(case['spec'])
        units['support_regions'][0]['vertical_stiffness_N_m3']='1*N/m**3'
        units['support_regions'][1]['vertical_stiffness_N_m3']='4*Pa/m'
        self.assertEqual(verify(units),expected)
        claimed=verify({**units,'proposed_contact_area_m2':'8*m**2',
                        'proposed_material_area_m2':'8*m**2','proposed_q_max_Pa':'4*Pa'})
        self.assertTrue(claimed['conservation_verified'],claimed)
        units['support_regions'][1]['vertical_stiffness_N_m3']='4*s'
        self.assertFalse(verify(units)['ok'])
        units['support_regions'][1]['vertical_stiffness_N_m3']='4 Pa/m'
        self.assertFalse(verify(units)['ok'])

    def test_registered_typescript_tool_boundary(self):
        import os
        import tempfile
        root=Path(__file__).resolve().parents[1]
        if not (root/'node_modules'/'@earendil-works'/'pi-coding-agent').exists():
            self.skipTest('registered Pi runtime not installed')
        cases=[c for c in reference_cases() if c['origin']==(0,0) and c['matrix']==(1,0,0,1)]
        calls=[dict(id=c['name'],spec=c['spec']) for c in cases]
        calls.append(dict(id='wrong-peak',spec={**cases[1]['spec'],'proposed_q_max_Pa':4.75}))
        calls.append(dict(id='unknown-model',spec={**cases[1]['spec'],'pressure_model':'independent_planes'}))
        source=r"""
import registerTaskTools from './task-tools.ts';
const tools = new Map();
registerTaskTools({ registerTool(t) { tools.set(t.name,t); }, on() {} });
const tool = tools.get('hamon_conservation_cad_verifier');
const results = {};
for (const c of JSON.parse(process.env.ZONED_TOOL_CASES)) {
  const result = await tool.execute(c.id,{mode:'auto',spec:c.spec},null,()=>{},{});
  results[c.id] = result.details;
}
console.log(JSON.stringify(results));
"""
        with tempfile.TemporaryDirectory() as work:
            env={**os.environ,'TASK_RELAY_URL':'http://127.0.0.1:1',
                 'TASK_WORKDIR':work,'TASK_HARNESS_SRC':str(root),
                 'TASK_HARNESS_PYTHON':sys.executable,'ZONED_TOOL_CASES':json.dumps(calls)}
            run=subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',source],
                               cwd=root,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(run.returncode,0,run.stderr)
        results=json.loads(run.stdout.strip().splitlines()[-1])
        self.assertEqual(set(results),{c['id'] for c in calls})
        for case in cases:
            validate(case,results[case['name']])
        self.assertTrue(results['wrong-peak']['ok'])
        self.assertFalse(results['wrong-peak']['conservation_verified'])
        self.assertFalse(results['unknown-model']['ok'])
        self.assertFalse(results['unknown-model']['conservation_verified'])


if __name__ == '__main__':
    unittest.main()
