import json
import sys
cases = json.load(open(sys.argv[1]))
print("all inputs", len(cases), "accepted flux and temperature", sum(x['qualified'] for x in cases),
      "incorrect claims rejected", sum(x.get('wrong_rejected', False) for x in cases),
      "exceptions", sum('error' in x for x in cases))
for r in cases:
    if r['x'] == .375 and (r['z'], r['tb'], r['tl']) in {(1.,0,50),(20.,0,50),(800.,50,0),(.6931471805599453,20,25)}:
        print(json.dumps(r,sort_keys=True))
