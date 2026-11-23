"""Check ordinary EAGLE witnesses against the full boundary of a registered cutout."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from eagle_board_domain_oracle import P, inside_polygon, rect, rounded, tube
from verify_eagle_board_domain import (
    COUNTS, boundary, inspect_walk, on_board_disk, rectangle, run_cli, small_fixture, wire,
)


def course(repo, out, observe=False):
    out.mkdir(parents=True)
    args = SimpleNamespace(repo=repo.resolve(),out=out)
    base_hole = [P(-.2,-.12),P(.2,-.12),P(.2,.12),P(-.2,.12)]
    outer = rect(-2,-2,2,2)
    records = {}
    for rotated in (False,True):
        def orient(p):
            x,y=p
            return ((3*x-4*y)/5,(4*x+3*y)/5) if rotated else (x,y)
        for name,layer_route,hit in [
                ("removed",False,False),("straddle",False,True),("retained",False,True),
                ("detour",True,False),("wide_wire",True,True),
                ("intact_collar",True,None),("cut_collar",False,None)]:
            via=name.endswith("collar")
            hole = base_hole
            if via:
                left,right = (.45,.55) if name=="intact_collar" else (.14,.17)
                hole=[P(left,-.02),P(right,-.02),P(right,.02),P(left,.02)]
            points=[orient(v) for v in hole]
            description=[(str(float(x)),str(float(y))) for x,y in points]
            if rotated: description.reverse()
            board_outline=rectangle()+boundary(description)
            if rotated: board_outline=board_outline[::2]+board_outline[1::2]
            # All points in these two rational rotations terminate as decimals.
            assert {P(*v) for v in description}==set(points)
            lines,_,_=rounded(points,{})
            valid=(outer[0]+lines,[],lambda p: outer[2](p) and not inside_polygon(p,points))
            pads=[("top",".25",0,1),("bottom",".25",0,16)] if via else [("L","-.65",0,1),("R",".65",0,1)]
            root,sch=small_fixture(pads,board_outline,via=via)
            signals=root.find("./drawing/board/signals")
            net=signals.find("signal")
            if not via:
                width=".7" if name=="wide_wire" else ".08"
                net.append(wire(("-.65",0),(".65",0),layer=1,width=width))
                if name=="detour":
                    for a,b in [(("-.65",0),("-.65",".5")),(("-.65",".5"),(".65",".5")),((".65",".5"),(".65",0))]:
                        net.append(wire(a,b,layer=1,width=".08"))
                x=".5" if name=="retained" else (".2" if name=="straddle" else "0")
                other=ET.SubElement(signals,"signal",name="B")
                span=".13" if name=="detour" else ".65"
                other.append(wire((x,"-"+span),(x,span),layer=1,width=".08"))
            key=("oblique-" if rotated else "axis-")+name
            schfile=out/(key+".sch");ET.ElementTree(sch).write(schfile,encoding="utf-8")
            query=["--routed-net","A"] + ([] if via else ["--contact-nets","A","B"])
            doc=run_cli(args,key,root,schfile,query,(2,1))
            route=doc["routed_continuity"]
            contact=doc.get("inter_net_contact",{})
            disks=contact.get("contacts",[])
            records[key]=dict(bounded=route["board_domain"]["status"],route=bool(route["witnesses"]),
                              disks=len(disks),overlap_mm2=sum(r["overlap_lower_bound_mm2"] for r in disks))
            if observe: continue
            assert route["board_domain"]["status"]=="bounded",key
            assert route["board_domain"]["hole_wire_count"]==4
            assert bool(route["witnesses"])==layer_route,(key,records[key])
            if not via:
                assert bool(disks)==hit,(key,records[key])
                for disk in disks:
                    on_board_disk(valid,disk["center_nm"],disk["disk_radius_nm"])
                    COUNTS["cross_net_disks"] += 1
                if name=="straddle" and not rotated:
                    assert abs(records[key]["overlap_mm2"]-39996*79996/1e12)<1e-14
            for witness in route["witnesses"]:
                inspect_walk(witness,valid)
            if name=="wide_wire":
                assert not tube(valid,P(-.65,0),P(.65,0),0)
                assert any(len(step["material"].get("path_nm",[]))>2
                           for witness in route["witnesses"] for step in witness["steps"])
    return dict(cases=records,checked_components=dict(COUNTS))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    p.add_argument("--observe",action="store_true")
    args=p.parse_args()
    result=course(args.repo,args.out,args.observe)
    (args.out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
