"""Joint area contact, with independent forward fields and public tool routes."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hamon_rectangle_contact import verify_rigid_rectangle_contact as solve
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route
from hamon_dual_verifier import run_dual_verify_spec
from ship.verify_rectangular_contact import run as independent_course

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(calculation='rigid_rectangle_contact',base_width_m=2,base_length_m=2,
            vertical_resultant_N=4,moment_x_Nm=2,moment_y_Nm=2)


class RigidRectangleContact(unittest.TestCase):
    def test_diagonal_field_and_all_reflections(self):
        for sx in (-1,1):
            for sy in (-1,1):
                out = solve({**BASE,'moment_x_Nm':2*sx,'moment_y_Nm':2*sy})
                self.assertTrue(out['conservation_verified'],out)
                self.assertEqual(out['contact_area_m2'],2)
                self.assertEqual(out['q_max_Pa'],6)
                self.assertEqual(out['force_residual_N'],0)
                self.assertEqual(out['moment_residual_Nm'],[0,0])
                self.assertEqual(out['contact_vertex_pressure_Pa'],[6,0,0])
                self.assertEqual(out['pressure_plane']['gradient_Pa_per_m'],[3*sx,3*sy])

    def test_independent_cross_section_course(self):
        report = independent_course()
        self.assertEqual(report['total'],675)
        self.assertEqual(report['failures'],[])
        self.assertLess(report['worst_relative_error'],2e-12)

    def test_translation_and_dimensioned_inputs(self):
        for cx,cy in [(0,0),(8,-32),(-256,128)]:
            out = solve({**BASE,'base_center_m':[cx,cy],
                         'moment_x_Nm':2+4*cx,'moment_y_Nm':2+4*cy})
            self.assertTrue(out['conservation_verified'],out)
            self.assertEqual(out['eccentricity_m'],[.5,.5])
            self.assertEqual(out['contact_polygon_m'],[[cx+1,cy+1],[cx-1,cy+1],[cx+1,cy-1]])
        converted = solve({**BASE,'base_width_m':'2000*mm','base_length_m':'200*cm',
                           'vertical_resultant_N':'.004*kN','moment_x_Nm':'2000*N*mm'})
        self.assertEqual(converted,solve(BASE))
        for field,value in [('vertical_resultant_N','4*N/m'),('moment_x_Nm','2*N'),
                            ('base_width_m','2*s'),('base_center_m',['1*Pa',0])]:
            self.assertFalse(solve({**BASE,field:value})['ok'])

    def test_product_of_strips_is_not_the_joint_contact(self):
        for claim in [{'proposed_contact_area_m2':2.25},{'proposed_q_max_Pa':64/9},
                      {'proposed_contact_polygon_m':[[-.5,-.5],[1,-.5],[1,1],[-.5,1]]}]:
            self.assertFalse(solve({**BASE,**claim})['conservation_verified'])

    def test_cyclic_and_reversed_polygon_claims(self):
        out = solve(BASE)
        points,pressures = out['contact_polygon_m'],out['contact_vertex_pressure_Pa']
        self.assertFalse(solve({**BASE,'proposed_contact_polygon_m':points,
                                'proposed_contact_vertex_pressure_Pa':None})['conservation_verified'])
        self.assertTrue(solve({**BASE,'proposed_contact_polygon_m':points})['conservation_verified'])
        for order in [(0,1,2),(1,2,0),(0,2,1)]:
            spec = {**BASE,'proposed_contact_polygon_m':[points[i] for i in order],
                    'proposed_contact_vertex_pressure_Pa':[pressures[i] for i in order]}
            self.assertTrue(solve(spec)['conservation_verified'])
            spec['proposed_contact_vertex_pressure_Pa'][0] += 1
            self.assertFalse(solve(spec)['conservation_verified'])

    def test_thin_corner_support_cannot_collapse_or_translate(self):
        for exponent in (20,34,48,52):
            gap = 2.**-exponent
            spec = {**BASE,'vertical_resultant_N':1,'moment_x_Nm':1-gap,'moment_y_Nm':1-gap}
            out = solve(spec)
            self.assertTrue(out['conservation_verified'],out)
            self.assertTrue(solve({**spec,'proposed_contact_polygon_m':out['contact_polygon_m'],
                                  'proposed_contact_vertex_pressure_Pa':out['contact_vertex_pressure_Pa']})
                            ['conservation_verified'])
            bad = [[[1,1],[1,1],[1,1]],
                   [[1,1],[1-2*gap,1],[1,1-2*gap]],
                   [[x-gap,y-gap] for x,y in out['contact_polygon_m']]]
            for polygon in bad:
                self.assertFalse(solve({**spec,'proposed_contact_polygon_m':polygon})
                                 ['conservation_verified'])

    def test_boundary_and_invalid_inputs(self):
        for key,value in [('moment_x_Nm',4),('moment_y_Nm',-4),('vertical_resultant_N',-1),
                          ('base_width_m',0),('base_width_m',True),('moment_x_Nm',math.nan),
                          ('base_length_m',math.inf),('pressure_model','independent_strips'),
                          ('horizontal_resultant_N',1),('base_center_m',[0]),
                          ('proposed_contact_vertex_pressure_Pa',[6,0,0])]:
            self.assertFalse(solve({**BASE,key:value})['conservation_verified'])
        for key in ('base_width_m','base_length_m','vertical_resultant_N','moment_x_Nm','moment_y_Nm'):
            spec = dict(BASE); del spec[key]
            self.assertFalse(solve(spec)['ok'])
        unloaded = solve({**BASE,'vertical_resultant_N':0,'moment_x_Nm':0,'moment_y_Nm':0})
        self.assertTrue(unloaded['conservation_verified'])
        self.assertIsNone(unloaded['contact_polygon_m'])
        self.assertIsNone(unloaded['contact_area_m2'])

    def test_middle_diamond_and_shape_transitions(self):
        # N=12, B=L=2: the full-contact boundary is |Mx|+|My|=4.
        for value,regime in [(math.nextafter(2.,0.),'full_contact'),(2.,'full_contact'),
                             (math.nextafter(2.,math.inf),'partial_contact')]:
            out = solve({**BASE,'vertical_resultant_N':12,'moment_x_Nm':2,'moment_y_Nm':value})
            self.assertTrue(out['conservation_verified'],out)
            self.assertEqual(out['contact_regime'],regime)
        for mx,my,shape in [(0,0,'full_rectangle'),(.6,.1,'edge_trapezoid'),
                            (.4,.2,'clipped_pentagon'),(.6,.6,'corner_triangle')]:
            out = solve({**BASE,'vertical_resultant_N':1,'moment_x_Nm':mx,'moment_y_Nm':my})
            self.assertTrue(out['conservation_verified'],out)
            self.assertEqual(out['footprint_shape'],shape)

    def test_unrepresentable_output_is_not_certified(self):
        for spec in [{**BASE,'base_width_m':1e-300,'vertical_resultant_N':1e300,
                      'moment_x_Nm':0,'moment_y_Nm':0},
                     {**BASE,'base_center_m':[2.**60,2.**60],
                      'moment_x_Nm':4*2.**60,'moment_y_Nm':4*2.**60}]:
            self.assertFalse(solve(spec)['conservation_verified'])

    def test_router_and_domain(self):
        for payload in (BASE,{'mode':'civil_geotech','spec':BASE}):
            self.assertEqual(route(payload),solve(BASE))
        dual = run_dual_verify_spec({'mode':'domain','kind':'rigid_rectangle_contact','spec':BASE})
        self.assertTrue(dual['dual_verified'],dual)
        self.assertEqual(dual['contact_polygon_m'],solve(BASE)['contact_polygon_m'])

    def test_inline_source_without_checkout(self):
        names = ('hamon_bearing_capacity','hamon_rectangle_contact','hamon_dual_verifier',
                 'hamon_conservation_cad_verifier')
        bundle = {'sources':{n:(ROOT/(n+'.py')).read_text() for n in names},
                  'payload':{**BASE,'base_width_m':'2000*mm'}}
        runner = """
import json,sys,types
bundle=json.loads(sys.stdin.read())
for name,source in bundle['sources'].items():
    module=types.ModuleType(name);module.__file__='/inline/'+name+'.py'
    sys.modules[name]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
print(json.dumps(run_router_conservation_cad_verifier(bundle['payload']),allow_nan=False))
"""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable,'-I','-c',runner],cwd=tmp,
                                    input=json.dumps(bundle),text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),solve(BASE))

    def test_actual_registered_pi_callback(self):
        if not (ROOT/'node_modules/@earendil-works/pi-coding-agent').exists():
            self.skipTest('registered Pi runtime not installed')
        cases = [BASE,{**BASE,'moment_x_Nm':-2},{**BASE,'moment_y_Nm':-2},
                 {**BASE,'base_center_m':[8,-32],'moment_x_Nm':34,'moment_y_Nm':-126},
                 {**BASE,'proposed_contact_area_m2':2.25},
                 {**BASE,'proposed_q_max_Pa':64/9},{**BASE,'vertical_resultant_N':'4*N/m'},
                 {**BASE,'moment_x_Nm':4},
                 {**BASE,'vertical_resultant_N':1,'moment_x_Nm':1-2.**-34,
                  'moment_y_Nm':1-2.**-34,'proposed_contact_polygon_m':[[1,1],[1,1],[1,1]]}]
        script = """
import registerTaskTools from './task-tools.ts';
const tools=new Map();
registerTaskTools({registerTool(t){tools.set(t.name,t);},on(){}});
const tool=tools.get('hamon_conservation_cad_verifier');
const results=[];
for(const spec of JSON.parse(process.env.CONTACT_CASES)){
 const result=await tool.execute('contact',{mode:'civil_geotech',spec},null,()=>{},{});
 results.push(result.details);
}
console.log(JSON.stringify(results));
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ,'TASK_RELAY_URL':'http://127.0.0.1:1','TASK_WORKDIR':tmp,
                   'TASK_HARNESS_SRC':str(ROOT),'TASK_HARNESS_PYTHON':sys.executable,
                   'CONTACT_CASES':json.dumps(cases)}
            result = subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',script],
                                    cwd=ROOT,env=env,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        # JavaScript may spell a large binary64 value as an integer decimal.
        # Interpret returned JSON numbers in the callback's binary64 domain.
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1],parse_int=float),
                         [solve(s) for s in cases])


if __name__ == '__main__':
    unittest.main()
