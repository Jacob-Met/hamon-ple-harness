"""Verify curved-copper witnesses on the unmodified Arduino Due differential pair."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from xml.etree import ElementTree as ET


BOARD_SHA = "fd7cc5f9a14623ff62a7555c826373fe8f842cca400f05ec4f7b4c5589e1780a"
SCH_SHA = "60f449078bea22dc298a0cad5b9733affb1214f6052f778e54babb08531e9734"
PAIRS = [("D+", "37", "3"), ("D-", "38", "2")]


def arc_oracle(wire):
    """Independent centre-and-polar-angle distance to the manufacturer's swept arc."""
    x, y, u, v = [float(wire.get(k))*1e6 for k in ("x1","y1","x2","y2")]
    sweep = math.radians(float(wire.get("curve")))
    k = .5 / math.tan(sweep/2)
    cx, cy = (x+u)/2 - k*(v-y), (y+v)/2 + k*(u-x)
    radius = math.hypot(x-cx,y-cy)
    start = math.atan2(y-cy,x-cx)
    def distance(point):
        px,py=point
        a=math.atan2(py-cy,px-cx)
        turn = ((a-start)*math.copysign(1,sweep)) % math.tau
        if turn <= abs(sweep):
            return abs(math.hypot(px-cx,py-cy)-radius)
        return min(math.hypot(px-x,py-y),math.hypot(px-u,py-v))
    mid = (cx+radius*math.cos(start+sweep/2), cy+radius*math.sin(start+sweep/2))
    return distance,mid,radius*abs(sweep)


def certify_segment(a,b,distance,limit):
    # Distance to a closed set is 1-Lipschitz. Bound every point on the segment.
    mid=[(a[i]+b[i])/2 for i in (0,1)]
    margin=limit-distance(mid)
    length=math.dist(a,b)
    if length/2 <= margin:
        return
    assert margin > 0 and length > .01, (a,b,margin)
    certify_segment(a,mid,distance,limit)
    certify_segment(mid,b,distance,limit)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    sch,board=args.raw/"due-0.sch",args.raw/"due-1.brd"
    assert hashlib.sha256(sch.read_bytes()).hexdigest()==SCH_SHA
    assert hashlib.sha256(board.read_bytes()).hexdigest()==BOARD_SHA
    summary=[]
    for net,pin,usb in PAIRS:
        def route(path,pair=True):
            argv=[sys.executable,str(args.repo/"engineering_toolkit.py"),"eagle-netlist",str(sch),
                  "--board",str(path),"--routed-net",net]
            if pair:
                argv+=["--routed-pads","U1",pin,"USB1",usb]
            doc=json.loads(subprocess.check_output(argv))
            return doc["routed_continuity"]
        dom=ET.parse(board)
        wires=next(s for s in dom.findall(".//board/signals/signal") if s.get("name")==net).findall("wire")
        row=route(board)
        assert row["status"]=="witnessed" and row["net_terminal_count"]==5 and row["query"]=="pad_pair"
        assert route(board,False)["status"]=="witnessed"
        bad = subprocess.run([sys.executable,str(args.repo/"engineering_toolkit.py"),"eagle-netlist",
                              str(sch),"--board",str(board),"--routed-net",net,"--routed-pads",
                              "U1",pin,"not-a-schematic-part","1"],capture_output=True,text=True)
        assert bad.returncode != 0 and "distinct schematic pads" in bad.stdout + bad.stderr
        witness,=row["witnesses"]
        assert witness["start"]==["pad","U1",pin] and witness["end"]==["pad","USB1",usb]
        arcs={}
        for step in witness["steps"]:
            source=step["source"]
            if source[0]!="wire":
                continue
            wire=wires[source[1]]
            if not float(wire.get("curve","0")):
                continue
            distance,mid,length=arc_oracle(wire)
            limit=float(wire.get("width"))*5e5-1
            pts=step["material"]["path_nm"]
            for a,b in zip(pts,pts[1:]):
                certify_segment(a,b,distance,limit)
            arcs[source[1]]=(length,mid,float(wire.get("width")))
        assert arcs
        index,(_,mid,width)=max(arcs.items(),key=lambda pair:pair[1][0])
        # Center the severing drill on the longest arc in the independently proven route.
        controls={}
        with tempfile.TemporaryDirectory() as temp:
            for label,shift in [("sever",0),("remote",10)]:
                control=ET.parse(board)
                where=control.find(".//board/plain")
                ET.SubElement(where,"hole",x=f"{mid[0]/1e6+shift:.6f}",y=f"{mid[1]/1e6+shift:.6f}",
                              drill=f"{width*1.5:.6f}")
                bp=Path(temp)/(label+".brd")
                control.write(bp,encoding="utf-8",xml_declaration=True)
                controls[label]=route(bp)["status"]
        assert controls=={"sever":"unresolved","remote":"witnessed"},controls
        (args.out/(net+"-witness.json")).write_text(json.dumps(row,indent=2))
        summary.append(dict(net=net,signal_terminal_count=5,wire_arcs=sorted(arcs),
                            verified_arc_count=len(arcs),severed_wire=index,controls=controls))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__=="__main__":
    main()
