#!/usr/bin/env python3
"""Independent rational cross-section controls, not a judged PLE score."""
import argparse
from fractions import Fraction as Q
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def exact_field(rx, ry):
    """Exact cubic cross-section quadrature, independent of polygon methods."""
    end = min(Q(2), 1/rx) if rx else Q(2)
    cuts = {Q(0), end}
    if rx:
        crossing = (1-2*ry)/rx
        if 0 < crossing < end:
            cuts.add(crossing)

    def slice_values(d):
        h = 1-rx*d
        depth = min(Q(2), h/ry) if ry else Q(2)
        force = h*depth-ry*depth**2/2
        return depth, force, d*force, h*depth**2/2-ry*depth**3/3

    result = [Q(0) for _ in range(4)]
    cuts = sorted(cuts)
    for left, right in zip(cuts, cuts[1:]):
        values = [slice_values(left+(right-left)*Q(k,3)) for k in range(4)]
        for i in range(4):
            result[i] += (right-left)*(values[0][i]+3*values[1][i]
                                      +3*values[2][i]+values[3][i])/8
    area, force, dmoment, emoment = result
    return area, force, force-dmoment, force-emoment


def cases():
    slopes = [Q(0),Q(1,8),Q(1,4),Q(3,8),Q(1,2),Q(3,4),Q(2),Q(8)]
    dimensions = [(Q(2),Q(2)),(Q(3),Q(7)),(Q(2**16),Q(1,2**12))]
    for rx, ry in itertools.product(slopes, repeat=2):
        area, force, mx, my = exact_field(rx,ry)
        for width, length in dimensions:
            jacobian = width*length/4
            for sx, sy in itertools.product((1,-1) if mx else (1,),
                                           (1,-1) if my else (1,)):
                spec = dict(calculation='rigid_rectangle_contact',
                            base_width_m=float(width),base_length_m=float(length),
                            vertical_resultant_N=float(12*jacobian*force),
                            moment_x_Nm=float(sx*12*jacobian*width/2*mx),
                            moment_y_Nm=float(sy*12*jacobian*length/2*my))
                yield dict(slopes=[str(rx),str(ry)],signs=[sx,sy],spec=spec,
                           expected_area=float(jacobian*area),expected_peak=12.)


def run():
    records = []
    for case in cases():
        result = run_router_conservation_cad_verifier(case['spec'])
        errors = {key:abs(result.get(field, float('inf'))-case[key])/case[key]
                  for key,field in [('expected_area','contact_area_m2'),
                                    ('expected_peak','q_max_Pa')]}
        passed = result.get('conservation_verified') is True and max(errors.values()) < 2e-9
        records.append({**case,'passed':passed,'relative_errors':errors,'result':result})
    return dict(scope='Independent authored controls; not a judged PLE score',
                total=len(records),passed=sum(r['passed'] for r in records),
                worst_relative_error=max(max(r['relative_errors'].values()) for r in records),
                failures=[r for r in records if not r['passed']],records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    report = run()
    if args.out:
        args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    raise SystemExit(report['passed'] != report['total'])
