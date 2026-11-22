import itertools
import json
import math
from decimal import Decimal as D, localcontext
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier

P = json.loads((here / "protocol.json").read_text())
def expected(spec):
    with localcontext() as ctx:
        ctx.prec = 80
        k, area, p, h, length, tb, tl, x = [D.from_float(float(spec[v])) for v in
            ("k_w_mk","area_m2","perimeter_m","h_w_m2k","length_m","t_base_c","t_tip_c","x_eval_m")]
        m = (h*p/(k*area)).sqrt()
        z = m*length
        def sinh(a):
            return (a.exp()-(-a).exp())/2
        def cosh(a):
            return (a.exp()+(-a).exp())/2
        return (float(k*area*m*(tb*cosh(z)-tl)/sinh(z)),
                float((tb*sinh(m*(length-x))+tl*sinh(m*x))/sinh(z)))
rows=[]
for z, (tb, tl), x in itertools.product(P["mL"], P["endpoint_excesses_K"], P["positions_L"]):
    spec=dict(mode="convective_fin_1d",tip_bc="prescribed_temp",k_w_mk=2.,area_m2=.02,
              perimeter_m=.4,h_w_m2k=z*z*2*.02/.4,length_m=1.,t_inf_c=0.,
              t_base_c=tb,t_tip_c=tl,x_eval_m=x)
    q,t=expected(spec)
    row=dict(z=z, tb=tb,tl=tl,x=x,q_expected=q,t_expected=t)
    try:
        out=run_router_conservation_cad_verifier(dict(mode="pde",spec=spec))
        row.update({k:out.get(k) for k in ("passed","reason","q_fin_w","q_fvm_w","t_eval_c","fin_efficiency","fin_effectiveness")})
        row["qualified"]=bool(out.get("passed") and math.isclose(out["q_fin_w"],q,rel_tol=3e-11,abs_tol=3e-11) and math.isclose(out["t_eval_c"],t,rel_tol=3e-11,abs_tol=3e-11))
        bad = run_router_conservation_cad_verifier(dict(mode="pde",spec={**spec,"claimed_field":"q_fin_w","claimed_value":q+(-2 if x==0.375 else 2),"rtol":0.,"atol":1e-8}))
        row["wrong_rejected"] = bad["passed"] is False
    except Exception as e:
        row.update(qualified=False,error=type(e).__name__+": "+str(e))
    rows.append(row)
Path(sys.argv[2]).write_text(json.dumps(rows,indent=1))
print(json.dumps(dict(total=len(rows),qualified=sum(r['qualified'] for r in rows),sample=rows[9*3*3])))
