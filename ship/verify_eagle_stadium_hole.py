"""Replay ordinary EAGLE stadium openings against an exact segment-distance oracle.

The oracle checks complete returned path tubes and disk rims against the analytic
capsule; it imports neither the candidate domain builder nor Shapely.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from eagle_board_domain_oracle import P, Q, rect, seg2, tube
from verify_eagle_board_domain import run_cli, small_fixture, wire
from verify_eagle_rounded_hole import decimal_point, turn


SPINE = (P(-1,0), P(1,0))
OUTER = rect(-4,-4,4,4)


def clear(a, b, radius):
    radius=Q(radius)
    return (radius>=0 and tube(OUTER,a,b,radius)
            and seg2(a,b,*SPINE)>(1+radius)**2)


def outline(split):
    if split:
        points=[P(-1,-1),P(1,-1),P(2,0),P(1,1),P(-1,1),P(-2,0)]
        sweeps=[0,90,90,0,90,90]
    else:
        points=[P(-1,-1),P(1,-1),P(1,1),P(-1,1)]
        sweeps=[0,180,0,180]
    return list(zip(points,points[1:]+points[:1],sweeps))


def replay(args):
    args.out.mkdir(parents=True,exist_ok=True)
    counts=Counter()
    results=[]
    # Two endpoint-only or centre-only receivers must reject these controls.
    a,b=P("1.35","1.03"),P("2.03",".35")
    assert clear(a,a,Q(".006")) and clear(b,b,Q(".006"))
    assert not clear(a,b,Q(".006"))
    rim=P("2.0000005",0)
    assert clear(rim,rim,0) and not clear(rim,rim,Q(".000001"))
    for split in (False,True):
        for frame in (False,True):
            def world(p):
                return turn((-p[0],p[1]),True) if frame else p

            def local(p):
                x,y=turn(tuple(Q(v)/1_000_000 for v in p),frame,inverse=True)
                return (-x,y) if frame else (x,y)

            def edge(a,b,s=0):
                return wire(decimal_point(world(a)),decimal_point(world(b)),
                            curve=-s if frame else s)

            outer=[P(-4,-4),P(4,-4),P(4,4),P(-4,4)]
            boundary=[edge(a,b) for a,b in zip(outer,outer[1:]+outer[:1])]
            inner=[edge(a,b,s) for a,b,s in outline(split)]
            if frame:
                inner=inner[::2]+inner[1::2]
            cases=[
                ("clear-contact",P("1.8",".8"),True),
                ("inside-contact",P("1.6",".6"),False),
                ("clear-trace",(P("1.65",".95"),P("1.95",".65")),True),
                ("clipped-trace",(a,b),False),
            ]
            for label,where,expected in cases:
                routed=label.endswith("trace")
                ends=list(where) if routed else [P(-3,0),P(3,0)]
                pads=[(name,*decimal_point(world(p)),1)
                      for name,p in zip(("L","R"),ends)]
                board,schematic=small_fixture(pads,boundary+inner)
                for pad in board.findall(".//smd"):
                    pad.attrib.update(dx=".02",dy=".02")
                signals=board.find(".//signals")
                signal=signals.find("signal")
                if routed:
                    assert clear(*ends,Q(".006"))==expected
                    signal.append(wire(*[decimal_point(world(p)) for p in ends],
                                       layer=1,width=".012"))
                    query=["--routed-net","A"]
                else:
                    x,y=where
                    assert clear(where,where,Q(".006"))==expected
                    for target,dx,dy in (
                        (signal,Q(".05"),0),
                        (ET.SubElement(signals,"signal",name="B"),0,Q(".05")),
                    ):
                        target.append(wire(decimal_point(world((x-dx,y-dy))),
                                           decimal_point(world((x+dx,y+dy))),
                                           layer=1,width=".012"))
                    query=["--contact-nets","A","B"]
                name=f"stadium-{int(split)}-{int(frame)}-{label}"
                sch=args.out/(name+".sch")
                ET.ElementTree(schematic).write(sch,encoding="utf-8")
                data=run_cli(args,name,board,sch,query,(2,1))
                report=data["routed_continuity" if routed else "inter_net_contact"]
                witnesses=report["witnesses" if routed else "contacts"]
                info=report["board_domain"]
                if args.baseline:
                    assert info["status"]=="unresolved" and not witnesses,(name,report)
                else:
                    assert info["status"]=="bounded" and bool(witnesses)==expected,(name,report)
                    assert info["hole_semicircle_count"]==2
                    assert info["hole_wire_count"]==(6 if split else 4)
                    assert info["hole_arc_error_upper_bound_nm"]<=1
                    cap_sources=[tuple(source) for cap in info["hole_semicircle_sources"] for source in cap]
                    assert len(cap_sources)==len(set(cap_sources))==(4 if split else 2)
                    assert Q(info["hole_capsule_radius_squared_nm2"])==1_000_000**2
                    for witness in witnesses:
                        if not routed:
                            point=local(witness["center_nm"])
                            radius=Q(witness["disk_radius_nm"])/1_000_000
                            assert radius>0 and clear(point,point,radius)
                            counts["contact_disks"]+=1
                            continue
                        steps=witness["steps"]
                        assert steps and steps[0]["start"]==witness["start"]
                        assert steps[-1]["end"]==witness["end"]
                        assert all(left["end"]==right["start"] for left,right in zip(steps,steps[1:]))
                        for step in steps:
                            material=step["material"]
                            assert material["kind"]=="component" and material["in_plane_clearance_nm"]==1
                            path=material["path_nm"]
                            assert path
                            for start,end in zip(path,path[1:] or path):
                                assert clear(local(start),local(end),Q(".000001")),(name,path)
                                counts["complete_path_segments"]+=1
                            for key,point in (("start",path[0]),("end",path[-1])):
                                node=step[key]
                                if node[0]=="copper":
                                    assert node[1]==material["layer"] and node[2:]==point
                            if "exit_overlap" in material:
                                port=material["exit_overlap"]
                                assert port["point_nm"]==path[-1]
                                point=local(port["point_nm"])
                                radius=Q(port["radius_nm"])/1_000_000
                                assert radius>0 and clear(point,point,radius)
                                counts["overlap_disks"]+=1
                results.append(dict(case=name,physical=expected,status=info["status"],
                                    witnesses=len(witnesses)))
    summary=dict(cases=results,verified=dict(counts))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(queries=len(results),positive=sum(bool(r["witnesses"]) for r in results),
                         bounded=sum(r["status"]=="bounded" for r in results),
                         verified=dict(counts))))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--baseline",action="store_true")
    args=parser.parse_args()
    args.repo=args.repo.resolve()
    args.out=args.out.resolve()
    replay(args)
