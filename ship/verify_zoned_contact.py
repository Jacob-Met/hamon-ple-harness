"""Independent rational regional references through the ordinary JSON CLI.

The reference integrates in x first; the output receiver uses Green's theorem.
Neither shares the solver's clipping, triangulation, inverse, or moment code.
"""
import argparse
from fractions import Fraction as F
from math import comb
import json
from pathlib import Path
import subprocess
import sys

REFERENCES = [
    ('horizontal', 1, '8', '8', '0', '0', '1'),
    ('horizontal', 4, '8', '20', '12', '0', '4'),
    ('one_axis', 1, '9/2', '81/16', '405/64', '0', '9/4'),
    ('one_axis', 4, '9/2', '321/16', '1621/64', '0', '9'),
    ('two_axes', 1, '9/2', '247/48', '1211/192', '3/4', '11/4'),
    ('two_axes', 4, '9/2', '3871/192', '77747/3072', '1401/512', '11'),
]
ROTATIONS = ((1,0,0,1), (0,-1,1,0), (-1,0,0,-1), (0,1,-1,0))


def powers(w, c, m):
    if not c:
        return 2*max(w, F(0))**m
    return (max(w+c, F(0))**(m+1)-max(w-c, F(0))**(m+1))/(c*(m+1))


def ymoment(w, c, m):
    return (powers(w,c,m+1)-w*powers(w,c,m))/c if c else F(0)


def zone_reference(name, k, left, right):
    k, left, right = map(F, (k,left,right))
    if name == 'horizontal':
        return 2*(right-left), 2*k*(right-left), k*(right*right-left*left), F(0), k
    a = F(1,4)
    c = F(1,2) if name == 'two_axes' else F(0)
    delta = lambda fun,m: fun(a+right,c,m)-fun(a+left,c,m)
    return (delta(powers,1), k*delta(powers,2)/2,
            k*(delta(powers,3)/3-a*delta(powers,2)/2-c*delta(ymoment,2)/2),
            k*delta(ymoment,2)/2, k*max(F(0),right+a+c))


def reference_cases(stiffness_factor=F(1)):
    for name, right_k, *values in REFERENCES:
        regional = [zone_reference(name,1,-2,0), zone_reference(name,right_k,0,2)]
        expected = [sum(v[j] for v in regional) for j in range(4)]
        expected.append(max(v[4] for v in regional))
        assert expected == list(map(F,values)), (name,right_k,expected,values)
        area, force, mx, my, peak = expected
        for rotation, matrix in enumerate(ROTATIONS):
            a,b,c,d = matrix
            for ox,oy in ((0,0), (1000,-2000)):
                def transform(x,y):
                    return ox+a*x+b*y, oy+c*x+d*y
                regions = []
                for label,k,left,right in (('left',1,-2,0), ('right',right_k,0,2)):
                    regions.append(dict(id=label, vertical_stiffness_N_m3=float(k*stiffness_factor),
                        outer_vertices_m=[transform(left,-1),transform(right,-1),
                                          transform(right,1),transform(left,1)]))
                yield dict(name=f'{name}-k{right_k}-r{rotation}-o{ox}-{oy}',
                    displacement=name, matrix=matrix, origin=(ox,oy),
                    expected=expected, regional=regional, stiffness_factor=stiffness_factor,
                    spec=dict(calculation='rigid_zoned_contact', support_regions=regions,
                        vertical_resultant_N=float(force),
                        moment_x_Nm=float(ox*force+a*mx+b*my),
                        moment_y_Nm=float(oy*force+c*mx+d*my)))


def boundary_moment(points, i, j):
    """Exact area integral x**i*y**j by integrating x**(i+1)*y**j dy."""
    result = F(0)
    for (x,y),(X,Y) in zip(points,points[1:]+points[:1]):
        dx,dy = X-x,Y-y
        for r in range(i+2):
            for s in range(j+1):
                result += (F(comb(i+1,r)*comb(j,s), (i+1)*(r+s+1)) *
                           x**(i+1-r)*dx**r*y**(j-s)*dy**s*dy)
    return result


def validate(case, out):
    assert out.get('ok') is True and out.get('conservation_verified') is True, out
    assert out.get('common_displacement_verified') is True, out
    assert out.get('output_geometry_verified') is True, out
    assert out.get('equilibrium_status') == 'finite', out
    assert out.get('pressure_model') == 'positive_part_affine_displacement', out
    assert [r['id'] for r in out['regions']] == ['left','right'], out
    errors = []
    def check(actual, expected):
        error = abs(F(actual)-expected)/max(abs(expected),F(1))
        errors.append(error)
        assert error <= F('1e-8'), (case['name'], actual, str(expected), float(error))
    area,force,mx,my,peak = case['expected']
    ox,oy = map(F,case['origin'])
    a,b,c,d = case['matrix']
    check(out['material_area_m2'],F(8))
    check(out['contact_area_m2'],area)
    check(out['q_max_Pa'],peak)
    check(out['integrated_vertical_resultant_N'],force)
    check(F(out['integrated_moment_x_Nm'])-ox*F(out['integrated_vertical_resultant_N']),a*mx+b*my)
    check(F(out['integrated_moment_y_Nm'])-oy*F(out['integrated_vertical_resultant_N']),c*mx+d*my)
    plane = out['displacement_plane']
    reference = list(map(F,plane['reference_point_m']))
    gx,gy = map(F,plane['gradient_m_per_m'])
    constant = F(plane['displacement_at_reference_m']) + gx*(ox-reference[0]) + gy*(oy-reference[1])
    def displacement(point):
        return constant+gx*(F(point[0])-ox)+gy*(F(point[1])-oy)
    for source,region,truth in zip(case['spec']['support_regions'],out['regions'],case['regional']):
        assert region['constitutive_verified'] is True
        ar,n,x,y,q = truth
        k = F(source['vertical_stiffness_N_m3'])
        check(region['vertical_stiffness_N_m3'],k)
        check(region['material_area_m2'],F(4))
        check(region['contact_area_m2'],ar)
        check(region['q_max_Pa'],q)
        check(region['integrated_vertical_resultant_N'],n)
        check(F(region['integrated_moment_x_Nm'])-ox*F(region['integrated_vertical_resultant_N']),a*x+b*y)
        check(F(region['integrated_moment_y_Nm'])-oy*F(region['integrated_vertical_resultant_N']),c*x+d*y)
        totals = [F(0)]*4
        observed_peak = F(0)
        for patch in region['contact_patches']:
            points = [(F(p[0])-ox,F(p[1])-oy) for p in patch['vertices_m']]
            m = {(i,j):boundary_moment(points,i,j)
                 for i,j in ((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))}
            assert m[0,0] > 0
            integrals = (m[0,0], k*(constant*m[0,0]+gx*m[1,0]+gy*m[0,1]),
                k*(constant*m[1,0]+gx*m[2,0]+gy*m[1,1]),
                k*(constant*m[0,1]+gx*m[1,1]+gy*m[0,2]))
            totals = [v+t for v,t in zip(totals,integrals)]
            assert len(points) == len(patch['vertex_pressure_Pa']) == len(patch['vertex_displacement_m'])
            for point,w,p in zip(patch['vertices_m'],patch['vertex_displacement_m'],patch['vertex_pressure_Pa']):
                expected_w = max(F(0),displacement(point))
                check(w,expected_w)
                check(p,k*expected_w)
                assert F(w) >= 0 and F(p) >= 0
                observed_peak = max(observed_peak,F(p))
        for actual,expected in zip(totals,(ar,n,a*x+b*y,c*x+d*y)):
            check(actual,expected)
        check(observed_peak,q)
        check(region['max_closure_residual_m'],F(0))
        for point in source['outer_vertices_m']:
            x,y = F(point[0])-ox,F(point[1])-oy
            u,v = a*x+c*y,b*x+d*y
            expected_w = (F(1) if case['displacement']=='horizontal' else
                          u+F(1,4)+(v/2 if case['displacement']=='two_axes' else 0))
            check(displacement(point),expected_w/case['stiffness_factor'])
    check(out['max_closure_residual_m'],F(0))
    return float(max(errors))


def replay(repo, output):
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    for case in reference_cases():
        completed = subprocess.run([sys.executable,str(repo/'hamon_conservation_cad_verifier.py'),
            '--json-stdin'],input=json.dumps(case['spec']),capture_output=True,text=True)
        out = json.loads(completed.stdout)
        error = validate(case,out)
        assert completed.returncode == 0, completed.stderr
        rows.append(dict(name=case['name'],spec=case['spec'],received=out,max_error=error))
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary = dict(cases=len(rows),passed=len(rows),worst_error=max(r['max_error'] for r in rows))
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    replay(args.repo.resolve(),args.out)
