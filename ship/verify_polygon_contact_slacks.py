"""Independent exact edge/corner references through the ordinary JSON CLI."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys


def cases():
    square=[(-1,-1),(1,-1),(1,1),(-1,1)]
    hole=[(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]
    rows=[]
    def add(name,rings,mx,my,area,peak):
        rows.append(dict(name=name,spec=dict(calculation='rigid_polygon_contact',
                    outer_vertices_m=rings[0],hole_vertices_m=rings[1:],
                    vertical_resultant_N=1,moment_x_Nm=mx,moment_y_Nm=my),
                    area=area,peak=peak,expected_verified=True))
    for power in (2,4,6,8,10,12,14,16,20,24,28,32,36,40,44):
        moment=1-2.**(-power)/3
        width=3*(1-F(moment))
        add(f'input-edge-{power}',[square],moment,0,2*width,1/width)
    for power,origin in ((14,(0,0)),(32,(1024,-2048)),(44,(0,0))):
        centroid=1-2.**(-power)
        width=3*(1-F(centroid))
        for matrix in ((1,0,0,1),(0,-1,1,0),(3,1,2,1),(-2,0,0,.5),
                       (3,2,1,3),(2,7,5,1)):
            a,b,c,d=matrix
            determinant=abs(F(a)*F(d)-F(b)*F(c))
            def transform(point):
                x,y=point
                return origin[0]+a*x+b*y,origin[1]+c*x+d*y
            for rings in ([square],[square,hole]):
                moved=[[transform(p) for p in ring] for ring in rings]
                mx,my=transform((centroid,0))
                add(f'affine-edge-{power}-{matrix}-holes{len(rings)-1}',
                    moved,mx,my,2*width*determinant,1/(width*determinant))
    for power in (4,14,24,36,44):
        centroid=1-2.**(-power)
        leg=4*(1-F(centroid))
        add(f'corner-{power}',[square],centroid,centroid,leg*leg/2,6/(leg*leg))
    for power in (10,14,16,24,32,44):
        source=next(r for r in rows if r['name']==f'input-edge-{power}')
        rows.append(dict(source,name=f'wrong-peak-{power}',expected_verified=False,
                    spec={**source['spec'],'proposed_q_max_Pa':float(source['peak'])*(1+3e-8)}))
    return rows


def replay(repo,output,require_all=False):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in cases():
        completed=subprocess.run([sys.executable,str(repo/'hamon_conservation_cad_verifier.py'),
                                  '--json-stdin'],input=json.dumps(case['spec']),
                                 capture_output=True,text=True)
        try:
            received=json.loads(completed.stdout)
        except json.JSONDecodeError:
            received=dict(ok=False,reason=completed.stderr[-1000:])
        verified=received.get('conservation_verified') is True
        area_error=(abs(received['contact_area_m2']/float(case['area'])-1)
                    if received.get('ok') else None)
        peak_error=(abs(received['q_max_Pa']/float(case['peak'])-1)
                    if received.get('ok') else None)
        correct=bool(received.get('ok') and verified==case['expected_verified']
                     and area_error<1e-8 and peak_error<1e-8)
        row={**case,'area':str(case['area']),'peak':str(case['peak']),
             'exit_code':completed.returncode,'received':received,
             'area_relative_error':area_error,'peak_relative_error':peak_error,'correct':correct}
        rows.append(row)
        if not correct:
            print(json.dumps(dict(failure=case['name'],reason=received.get('reason'),
                                  verified=verified,area_error=area_error,peak_error=peak_error)),flush=True)
    positives=[r for r in rows if r['expected_verified']]
    negatives=[r for r in rows if not r['expected_verified']]
    summary=dict(cases=len(rows),positive_correct=sum(r['correct'] for r in positives),
                 positives=len(positives),negative_correct=sum(r['correct'] for r in negatives),
                 negatives=len(negatives),
                 certified=sum(r['received'].get('conservation_verified') is True for r in positives),
                 worst_certified_peak_relative_error=max(
                     (r['peak_relative_error'] for r in positives
                      if r['received'].get('conservation_verified') is True),default=None),
                 worst_certified_area_relative_error=max(
                     (r['area_relative_error'] for r in positives
                      if r['received'].get('conservation_verified') is True),default=None))
    (output/'results.json').write_text(json.dumps(dict(summary=summary,cases=rows),indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if require_all and not all(r['correct'] for r in rows):
        raise SystemExit('The ordinary CLI failed an independent contact reference')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--require-all',action='store_true')
    args=parser.parse_args()
    replay(args.repo.resolve(),args.out.resolve(),args.require_all)
