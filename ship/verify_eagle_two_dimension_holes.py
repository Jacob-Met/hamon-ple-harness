"""Check nominal copper paths against two separately identified board cutouts."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from eagle_board_domain_oracle import P, Q, rounded
from verify_eagle_board_domain import COUNTS, inspect_walk, on_board_disk, run_cli, small_fixture, wire
from verify_eagle_rounded_hole import decimal_point, turn


def corners(x0, y0, x1, y1):
    return [P(x0,y0), P(x1,y0), P(x1,y1), P(x0,y1)]


def outline(points):
    return list(zip(points, points[1:]+points[:1]))


def course(args):
    results = {}
    for oblique in (False, True):
        def world(point):
            return decimal_point(turn(point, oblique))

        loops = [corners(-2,-2,2,2), corners(-.95,-.3,-.45,.3), corners(.45,-.3,.95,.3)]
        analytic = [rounded([turn(p, oblique) for p in loop], {}) for loop in loops]
        outer, left, right = analytic
        material = (outer[0]+left[0]+right[0], [],
                    lambda p: outer[2](p) and not left[2](p) and not right[2](p))
        dims = []
        source_groups = {}
        for index in ([2,0,1] if oblique else [0,1,2]):
            edges = outline(loops[index])
            if oblique:
                edges = [(b,a) for a,b in edges[::2]+edges[1::2]]
            source_groups[index] = {("plain","wire",len(dims)+i) for i in range(len(edges))}
            dims.extend(wire(world(a),world(b)) for a,b in edges)

        cases = [
            ("left-disk","disk",P(-.70,0),False), ("right-disk","disk",P(.70,0),False),
            ("web-disk","disk",P(0,0),True), ("left-rim","disk",P(-.45,0),True),
            ("both-cut-route","cut",None,False), ("route-around-both","detour",None,True),
            ("middle-plated","via",P(0,0),True), ("removed-plated","via",P(.70,0),False),
        ]
        for label,kind,point,expected in cases:
            if kind == "via":
                vx,vy=point
                pads=[(k,*world((vx+Q(".25"),vy)),layer) for k,layer in (("T",1),("B",16))]
            else:
                pads=[(k,*world(P(x,0)),1) for k,x in (("L",-1.4),("R",1.4))]
            board, schematic = small_fixture(pads,dims,via=kind=="via")
            signals=board.find(".//signals")
            signal=signals.find("signal")
            if kind=="disk":
                x,y=point
                delta=Q(".03")
                signal.append(wire(world((x-delta,y)),world((x+delta,y)),layer=1,width=".008"))
                other=ET.SubElement(signals,"signal",name="B")
                other.append(wire(world((x,y-delta)),world((x,y+delta)),layer=1,width=".008"))
                query=["--contact-nets","A","B"]
            else:
                query=["--routed-net","A"]
                if kind=="via":
                    x,y=world(point)
                    signal.find("via").attrib.update(x=x,y=y)
                else:
                    path=[P(-1.4,0),P(1.4,0)]
                    if kind=="detour":
                        path[1:1]=[P(-1.4,.55),P(1.4,.55)]
                    for a,b in zip(path,path[1:]):
                        signal.append(wire(world(a),world(b),layer=1,width=".05"))
            key=("oblique-" if oblique else "axis-")+label
            sch=args.out/(key+".sch")
            ET.ElementTree(schematic).write(sch,encoding="utf-8")
            doc=run_cli(args,key,board,sch,query,(2,1))
            report=doc["inter_net_contact" if kind=="disk" else "routed_continuity"]
            info=report["board_domain"]
            witnesses=report["contacts" if kind=="disk" else "witnesses"]
            results[key]={"bounded":info["status"]=="bounded","witnesses":len(witnesses),
                          "board_sha256":doc["board_comparison"]["board_sha256"]}
            if args.baseline:
                assert info["status"]=="unresolved" and not witnesses,key
                continue
            assert info["status"]=="bounded" and info["hole_count"]==2,key
            groups={frozenset(map(tuple,h["sources"])) for h in info["dimension_holes"]}
            assert groups=={frozenset(source_groups[1]),frozenset(source_groups[2])}
            assert bool(witnesses)==expected,(key,report)
            for w in witnesses:
                if kind=="disk":
                    on_board_disk(material,w["center_nm"],w["disk_radius_nm"])
                    COUNTS["cross_net_disks"]+=1
                else:
                    inspect_walk(w,material)
    return {"layouts":results,"geometry_checks":dict(COUNTS)}


def manufacturer(args):
    """Keep native signal, terminal and via artwork and change only the milling drawing."""
    from hashlib import sha256
    from eagle_board_domain_oracle import authentic_domains
    from verify_eagle_finite_copper import SOURCES

    checks = {}
    for donor, label, pivot in (("due","Due",P("12.4049","28.4999")),
                               ("feather","Feather",P("32.4485","6.731"))):
        sch, brd, net, nc, nn, sch_digest, brd_digest = SOURCES[donor]
        assert sha256((args.raw/sch).read_bytes()).hexdigest()==sch_digest
        assert sha256((args.raw/brd).read_bytes()).hexdigest()==brd_digest
        remote = [P(20,12),P(22,12)] if donor=="feather" else [P(30,28),P(34,28)]
        for label2,centres,expect in (("remote",remote,True),
                                     ("severed",[pivot,remote[1]],False)):
            root=ET.parse(args.raw/brd).getroot()
            plain=root.find("./drawing/board/plain")
            holes=[]
            for x,y in centres:
                half=Q(".40")
                points=corners(x-half,y-half,x+half,y+half)
                holes.append(rounded(points,{}))
                plain.extend(wire(decimal_point(a),decimal_point(b)) for a,b in outline(points))
            key=donor+"-"+label2
            doc=run_cli(args,key,root,args.raw/sch,["--routed-net",net],(nc,nn))
            report=doc["routed_continuity"]
            witnesses=report["witnesses"]
            checks[key]={"bounded":report["board_domain"]["status"]=="bounded",
                         "witnesses":len(witnesses),"board_sha256":doc["board_comparison"]["board_sha256"]}
            if args.baseline:
                assert report["board_domain"]["status"]=="unresolved" and not witnesses
                continue
            assert report["board_domain"]["hole_count"]==2 and bool(witnesses)==expect,(key,report)
            outside=authentic_domains()[label]
            material=(outside[0]+holes[0][0]+holes[1][0],outside[1],
                      lambda p: outside[2](p) and not any(h[2](p) for h in holes))
            for w in witnesses:
                inspect_walk(w,material,feather=(donor=="feather"))
    return checks


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--baseline",action="store_true")
    parser.add_argument("--raw",type=Path,help="directory containing the four pinned manufacturer inputs")
    args=parser.parse_args()
    args.repo=args.repo.resolve()
    args.out.mkdir(parents=True)
    result=course(args)
    if args.raw:
        result["manufacturer"]=manufacturer(args)
        result["geometry_checks"]=dict(COUNTS)
    (args.out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
