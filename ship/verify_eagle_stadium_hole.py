"""Replay ordinary EAGLE stadium openings against an exact segment-distance oracle.

The oracle checks complete returned path tubes and disk rims against the analytic
capsule; it imports neither the candidate domain builder nor Shapely.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from eagle_board_domain_oracle import P, Q, pseg2, rect, seg2, tube
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


def fixture_copper(board):
    """Read the fixture's source geometry in board coordinates, independently of the solver."""
    layout=board.find("./drawing/board")
    element,=layout.findall("elements/element")
    assert element.get("name")=="P" and P(element.get("x"),element.get("y"))==P(0,0)
    assert element.get("rot","R0")=="R0"
    package=layout.find(f"libraries/library[@name='{element.get('library')}']/packages/"
                        f"package[@name='{element.get('package')}']")
    pads={}
    for pad in package.findall("smd"):
        assert pad.get("layer")=="1" and pad.get("rot","R0")=="R0"
        assert Q(pad.get("roundness","0"))==0 and Q(pad.get("dx"))==Q(pad.get("dy"))==Q(".02")
        key=("terminal",element.get("name"),pad.get("name"))
        assert key not in pads
        pads[key]=("square",P(pad.get("x"),pad.get("y")),Q(pad.get("dx"))/2)
    assert set(pads)=={("terminal","P","L"),("terminal","P","R")}
    wires={}
    for signal in layout.findall("signals/signal"):
        for i,wire_row in enumerate(signal.findall("wire")):
            assert wire_row.get("layer")=="1" and Q(wire_row.get("curve","0"))==0
            assert Q(wire_row.get("width"))==Q(".012") and wire_row.get("cap","round")=="round"
            a=P(wire_row.get("x1"),wire_row.get("y1"))
            b=P(wire_row.get("x2"),wire_row.get("y2"))
            wires[(signal.get("name"),"wire",i)]=("stroke",a,b,Q(wire_row.get("width"))/2)
    return pads,wires


def contains_disk(shape, point_nm, radius_nm):
    point=tuple(Q(v)/1_000_000 for v in point_nm)
    radius=Q(radius_nm)/1_000_000
    if len(point)!=2 or radius<0:
        return False
    if shape[0]=="square":
        _,center,half=shape
        return all(abs(p-c)+radius<=half for p,c in zip(point,center))
    _,a,b,half=shape
    return radius<=half and pseg2(point,a,b)<=(half-radius)**2


def inspect_copper(witness, routed, pads, wires, counts):
    """Receive the source-owned copper too, independently from the board-domain gate."""
    if not routed:
        assert witness["layer"]==1
        sources=[(x["signal"],*x["primitive"]) for x in witness["sources"]]
        assert sorted(sources)==[("A","wire",0),("B","wire",0)]
        for source in sources:
            assert contains_disk(wires[source],witness["center_nm"],witness["disk_radius_nm"])
        assert Q(witness["disk_radius_nm"])>0
        counts["contact_disks_in_both_strokes"]+=1
        return

    assert {tuple(witness["start"]),tuple(witness["end"])}=={("pad","P","L"),("pad","P","R")}
    steps=witness["steps"]
    assert steps and steps[0]["start"]==witness["start"] and steps[-1]["end"]==witness["end"]
    assert all(a["end"]==b["start"] for a,b in zip(steps,steps[1:]))
    def shape(step):
        key=tuple(step["source"])
        if key in pads:
            return pads[key]
        assert ("A",*key) in wires, key
        return wires[("A",*key)]

    for i,step in enumerate(steps):
        material=step["material"]
        assert material["kind"]=="component" and material["layer"]==1
        assert material["in_plane_clearance_nm"]==1 and "terminal_collar" not in material
        path=material["path_nm"]
        assert path
        primitive=shape(step)
        # The square and round-capped straight stroke are convex after erosion
        # by the path radius. Containment of every vertex covers every full segment.
        assert all(contains_disk(primitive,xy,1) for xy in path)
        counts["complete_segments_in_declared_source"]+=max(1,len(path)-1)
        for key,xy,is_terminal in (("start",path[0],i==0),("end",path[-1],i==len(steps)-1)):
            node=step[key]
            if is_terminal:
                assert node[0]=="pad" and tuple(step["source"])==("terminal",*node[1:])
                assert contains_disk(pads[tuple(step["source"])],xy,1)
                counts["bound_smd_landings"]+=1
            else:
                assert node==["copper",1,*xy]
        if i+1<len(steps):
            port=material["exit_overlap"]
            assert port["kind"]=="overlap" and Q(port["radius_nm"])>0
            assert port["point_nm"]==path[-1]==steps[i+1]["material"]["path_nm"][0]
            assert contains_disk(primitive,port["point_nm"],port["radius_nm"])
            assert contains_disk(shape(steps[i+1]),port["point_nm"],port["radius_nm"])
            counts["overlap_disks_in_both_step_sources"]+=1
        else:
            assert "exit_overlap" not in material


def reject_off_copper_controls(witness, routed, pads, wires, local, controls):
    """Adverse certificates clear the board but fail independent source membership."""
    def reject(bad, key):
        try:
            inspect_copper(bad,routed,pads,wires,Counter())
        except AssertionError:
            controls[key]+=1
        else:
            raise AssertionError(f"accepted fabricated {key}")
    if routed and "false_smd_landing" not in controls:
        far=[2_750_000,2_750_000]
        assert clear(local(far),local(far),Q(".000001"))
        forged=deepcopy(witness)
        forged["steps"]=[{"start":witness["start"],"end":witness["end"],
                          "source":["terminal",*witness["start"][1:]],
                          "material":{"kind":"component","layer":1,
                                      "in_plane_clearance_nm":1,"path_nm":[far]}}]
        reject(forged,"false_smd_landing")
        forged=deepcopy(witness)
        port=next(step["material"]["exit_overlap"] for step in forged["steps"]
                  if "exit_overlap" in step["material"])
        port["radius_nm"]=6_001
        assert clear(local(port["point_nm"]),local(port["point_nm"]),Q(".006001"))
        reject(forged,"overlap_rim_outside_stroke")
    elif not routed and "contact_rim_outside_strokes" not in controls:
        forged=deepcopy(witness)
        forged["disk_radius_nm"]=6_001
        point=local(forged["center_nm"])
        assert clear(point,point,Q(".006001"))
        reject(forged,"contact_rim_outside_strokes")


def replay(args):
    args.out.mkdir(parents=True,exist_ok=True)
    counts=Counter()
    copper_counts=Counter()
    controls=Counter()
    results=[]
    # Two endpoint-only or centre-only receivers must reject these controls.
    a,b=P("1.35","1.03"),P("2.03",".35")
    assert clear(a,a,Q(".006")) and clear(b,b,Q(".006"))
    assert not clear(a,b,Q(".006"))
    # Every point in the stroke's midpoint cross-section lies inside the cap.
    midpoint=tuple((x+y)/2 for x,y in zip(a,b))
    assert sum((x-y)**2 for x,y in zip(midpoint,SPINE[1]))<(1-Q(".006"))**2
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
                    pads_by_source,wires_by_source=fixture_copper(board)
                    for witness in witnesses:
                        inspect_copper(witness,routed,pads_by_source,wires_by_source,copper_counts)
                        reject_off_copper_controls(witness,routed,pads_by_source,wires_by_source,local,controls)
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
    if not args.baseline:
        assert set(controls)=={"false_smd_landing","overlap_rim_outside_stroke","contact_rim_outside_strokes"}
    summary=dict(cases=results,verified=dict(counts),copper_membership=dict(copper_counts),
                 rejected_off_copper=dict(controls))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(queries=len(results),positive=sum(bool(r["witnesses"]) for r in results),
                         bounded=sum(r["status"]=="bounded" for r in results),
                         verified=dict(counts),copper_membership=dict(copper_counts),
                         rejected_off_copper=dict(controls))))
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
