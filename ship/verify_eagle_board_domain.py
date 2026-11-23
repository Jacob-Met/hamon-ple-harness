"""Independently check whole ordinary-CLI witnesses against nominal board domains.

The analytic receiver imports no candidate parser or geometry backend. Frozen
manufacturer bytes and the earlier 19-case course remain independently checked.
"""
import argparse
import copy
from collections import Counter
from decimal import Decimal
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

from eagle_board_domain_oracle import (
    P, Q, rect, rounded, disk, authentic_domains, material_path_on_board,
    reported_disk_on_board, point_nm_to_mm, norm2, pseg2,
)
from eagle_board_milling_oracle import (
    verify_original, path_clears_x6, disk_clears_x6,
)
from verify_eagle_finite_copper import CASES, SOURCES, make, check_witness
from verify_eagle_inter_net import ENDS

COUNTS = Counter()


def wire(a, b, layer=20, width="0", curve="0"):
    return ET.Element("wire", x1=str(a[0]), y1=str(a[1]), x2=str(b[0]), y2=str(b[1]),
                      width=str(width), layer=str(layer), curve=str(curve))


def boundary(points):
    return [wire(a,b) for a,b in zip(points,points[1:]+points[:1])]


def rectangle(left=-2, right=2, bottom=-2, top=2):
    return boundary([(left,bottom),(right,bottom),(right,top),(left,top)])


def on_board_disk(domain, center, radius, feather=False):
    assert reported_disk_on_board(domain,center,radius), ("disk outside board",center,radius)
    if feather:
        assert disk_clears_x6(center,radius), ("disk enters X6 exclusion",center,radius)


def inspect_collar(record, domain, feather=False):
    center=P(*record["drill_center_nm"])
    wall=P(*record["barrel_wall_point_nm"])
    port=P(*record["point_nm"])
    inner,outer=Q(record["drill_radius_nm"]),Q(record["clear_collar_outer_radius_nm"])
    assert outer>inner>0
    on_board_disk(domain,record["drill_center_nm"],record["clear_collar_outer_radius_nm"],feather)
    assert norm2(center,wall)==inner*inner
    assert inner*inner<norm2(center,port)<outer*outer
    # A radial landing may meet its intended bore wall, but cannot cross its void.
    assert pseg2(center,port,wall)>=inner*inner
    return port,wall


def inspect_walk(witness, domain, feather=False):
    steps=witness["steps"]
    assert steps and steps[0]["start"]==witness["start"] and steps[-1]["end"]==witness["end"]
    assert all(a["end"]==b["start"] for a,b in zip(steps,steps[1:]))
    for step in steps:
        material=step["material"]
        if material["kind"]=="plated_barrel":
            port,wall=inspect_collar(material,domain,feather)
            assert material["source"]==step["source"]
            path=material["path"]
            assert len(path)==4 and path[0]==step["start"] and path[-1]==step["end"]
            assert [node[0] for node in path]==["copper","barrel_wall","barrel_wall","copper"]
            assert {path[0][1],path[-1][1]}=={1,16}
            assert path[0][1]==path[1][1] and path[2][1]==path[3][1]
            assert P(*path[0][2:])==P(*path[3][2:])==port
            assert P(*path[1][2:])==P(*path[2][2:])==wall
            # The two radial landings lie in the collar; the middle segment
            # stays at the same bore-wall XY through the plated layer crossing.
            COUNTS["plated_barrels"]+=1
            COUNTS["barrel_path_segments"]+=3
            continue
        assert material["kind"]=="component" and material["in_plane_clearance_nm"]==1
        path=material["path_nm"]
        assert path
        for key,point in (("start",path[0]),("end",path[-1])):
            node=step[key]
            if node[0]=="copper":
                assert node[1]==material["layer"] and node[2:]==point
        if len(set(map(tuple,path)))==1:
            on_board_disk(domain,path[0],1,feather)
            COUNTS["point_landings"]+=1
        else:
            assert material_path_on_board(domain,path), ("complete path outside board",path)
            if feather:
                assert path_clears_x6(path), ("complete path enters X6 exclusion",path)
            COUNTS["path_segments"]+=len(path)-1
        for key in ("exit_overlap","terminal_collar"):
            if key not in material:
                continue
            record=material[key]
            if key=="exit_overlap":
                assert record["kind"]=="overlap" and record["point_nm"]==path[-1]
                on_board_disk(domain,record["point_nm"],record["radius_nm"],feather)
                COUNTS["overlap_disks"]+=1
            else:
                port,_=inspect_collar(record,domain,feather)
                assert any(step[key][0]=="pad" and P(*point)==port
                           for key,point in (("start",path[0]),("end",path[-1])))
                COUNTS["terminal_collars"]+=1


def reject_translated_barrel(witness, domain, feather=False):
    """The whole serialized barrel path, not just its declared disk, is checked."""
    saved=COUNTS.copy()
    candidate=copy.deepcopy(witness)
    barrel=next(step for step in candidate["steps"] if step["material"]["kind"]=="plated_barrel")
    for node in barrel["material"]["path"]:
        node[2]+=120_000_000
    try:
        inspect_walk(candidate,domain,feather)
    except AssertionError:
        pass
    else:
        raise AssertionError("translated serialized barrel path was accepted")
    finally:
        COUNTS.clear();COUNTS.update(saved)
    COUNTS["rejected_translated_barrel_paths"]+=1


def run_cli(args, label, root, schematic, query, membership=(605,163)):
    path=args.out/(label+".brd")
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)
    doc=json.loads(subprocess.check_output([
        sys.executable,str(args.repo/"engineering_toolkit.py"),"eagle-netlist",
        str(schematic),"--board",str(path),*query],text=True))
    (args.out/(label+".json")).write_text(json.dumps(doc,indent=2)+"\n")
    flat=doc["board_comparison"]
    assert flat["board_sha256"]==sha256(path.read_bytes()).hexdigest()
    assert flat["agreement"] and (flat["exact_contacts"],len(flat["exact_nets"]))==membership
    COUNTS["ordinary_cli_queries"]+=1
    return doc


def set_outline(root, rows):
    plain=root.find("./drawing/board/plain")
    for item in list(plain):
        if item.get("layer")=="20":
            plain.remove(item)
    plain.extend(copy.deepcopy(rows))


def crossing_controls(args, original, sch):
    root=copy.deepcopy(original)
    board=root.find("./drawing/board")
    for name,a,b in (("D+",(-1,0),(1,0)),("D-",(0,-1),(0,1))):
        signal=board.find("./signals/signal[@name='%s']"%name)
        for item in list(signal):
            if item.tag!="contactref":
                signal.remove(item)
        signal.append(wire(a,b,layer=1,width=".2"))
    records={}
    for label,left,expected in (("inside",-2,True),("outside",10,False),
                                ("straddling",".05",True),("tangent",".1",False),
                                ("gap",".101",False)):
        xmax=14 if label=="outside" else 2
        set_outline(root,rectangle(left,xmax))
        doc=run_cli(args,"crossing-"+label,root,sch,["--contact-nets","D+","D-"])
        result=doc["inter_net_contact"]
        assert result["board_domain"]["status"]=="bounded"
        assert bool(result["contacts"])==expected
        assert result["status"]==("contact_witnessed" if expected else "unresolved")
        if expected:
            row,=result["contacts"]
            on_board_disk(rect(left,-2,xmax,2),row["center_nm"],row["disk_radius_nm"])
            target=F("0.039998400016") if label=="inside" else F(49996*199996,10**12)
            assert abs(Q(row["overlap_lower_bound_mm2"])-target)<F(1,10**14)
            if label=="inside":
                assert row["center_nm"]==[0,0] and row["disk_radius_nm"]==49999
            COUNTS["cross_net_disks"]+=1
        records[label]=result
    for label,rows in (
        ("open",rectangle()[:-1]),
        ("nested",rectangle()+rectangle("-.05",".05","-.05",".05")),
        ("disjoint",rectangle()+rectangle(10,14)),
        ("self_crossing",boundary([(-2,-2),(2,2),(-2,2),(2,-2)])),
        ("circle",[ET.Element("circle",x="0",y="0",radius="2",width="0",layer="20")]),
        ("absent",[]),
    ):
        set_outline(root,rows)
        result=run_cli(args,"crossing-"+label,root,sch,["--contact-nets","D+","D-"])["inter_net_contact"]
        assert result["board_domain"]["status"]==("unbounded_artwork" if label=="absent" else "unresolved")
        assert bool(result["contacts"])==(label=="absent")
        records[label]=result
    return records


def small_fixture(pads, outline_rows, via=False, notch=False):
    root=ET.fromstring("<eagle><drawing><board><plain/><libraries/><elements/><signals/></board></drawing></eagle>")
    board=root.find("./drawing/board")
    board.find("plain").extend(copy.deepcopy(outline_rows))
    lib=ET.SubElement(board.find("libraries"),"library",name="Pads")
    pkg=ET.SubElement(ET.SubElement(lib,"packages"),"package",name="P")
    for name,x,y,layer in pads:
        ET.SubElement(pkg,"smd",name=name,x=str(x),y=str(y),dx=".02" if via else ".1",
                      dy=".02" if via else ".1",layer=str(layer))
    ET.SubElement(board.find("elements"),"element",name="P",library="Pads",package="P",x="0",y="0")
    signal=ET.SubElement(board.find("signals"),"signal",name="A")
    for name,_,_,_ in pads:
        ET.SubElement(signal,"contactref",element="P",pad=name)
    if via:
        ET.SubElement(signal,"via",x="0",y="0",extent="1-16",drill=".3",diameter=".6")
    if notch:
        signal.append(wire((-.5,0),(.5,0),layer=1,width=".2"))
    schematic=ET.fromstring("<eagle><drawing><schematic><libraries/><parts/><sheets><sheet><instances/><nets/></sheet></sheets></schematic></drawing></eagle>")
    doc=schematic.find("./drawing/schematic")
    slib=copy.deepcopy(lib);doc.find("libraries").append(slib)
    symbol=ET.SubElement(ET.SubElement(slib,"symbols"),"symbol",name="S")
    for name,_,_,_ in pads:
        ET.SubElement(symbol,"pin",name=name)
    ds=ET.SubElement(ET.SubElement(slib,"devicesets"),"deviceset",name="D")
    ET.SubElement(ET.SubElement(ds,"gates"),"gate",name="G",symbol="S")
    device=ET.SubElement(ET.SubElement(ds,"devices"),"device",name="",package="P")
    connects=ET.SubElement(device,"connects")
    for name,_,_,_ in pads:
        ET.SubElement(connects,"connect",gate="G",pin=name,pad=name)
    ET.SubElement(doc.find("parts"),"part",name="P",library="Pads",deviceset="D",device="")
    sheet=doc.find("sheets/sheet")
    ET.SubElement(sheet.find("instances"),"instance",part="P",gate="G")
    segment=ET.SubElement(ET.SubElement(sheet.find("nets"),"net",name="A"),"segment")
    for name,_,_,_ in pads:
        ET.SubElement(segment,"pinref",part="P",gate="G",pin=name)
    return root,schematic


def notch_and_collar_controls(args):
    points=[(-1,-1),(1,-1),(1,1),(".15",1),(".15","-.2"),
            ("-.15","-.2"),("-.15",1),(-1,1)]
    root,schematic=small_fixture([("L","-.5",0,1),("R",".5",0,1)],boundary(points),notch=True)
    sch=args.out/"notch.sch";ET.ElementTree(schematic).write(sch,encoding="utf-8")
    domain=rounded([P(*p) for p in points],{})
    records={}
    for detour in (False,True):
        if detour:
            signal=root.find("./drawing/board/signals/signal")
            for a,b in [((-.5,0),(-.5,-.45)),((-.5,-.45),(.5,-.45)),((.5,-.45),(.5,0))]:
                signal.append(wire(a,b,layer=1,width=".2"))
        label="notch-detour" if detour else "notch-direct"
        routed=run_cli(args,label,root,sch,["--routed-net","A"],(2,1))["routed_continuity"]
        assert routed["status"]==("witnessed" if detour else "unresolved")
        for witness in routed["witnesses"]:
            inspect_walk(witness,domain)
        records[label]=routed["status"]
    for label,left,expected in (("full",-1,True),("cap",".2",False),
                                ("bore_tangent",".15",False),("collar","-.16",True)):
        root,schematic=small_fixture([("top",".25",0,1),("bottom",".25",0,16)],
                                      rectangle(left,1,-1,1),via=True)
        sch=args.out/("via-"+label+".sch");ET.ElementTree(schematic).write(sch,encoding="utf-8")
        routed=run_cli(args,"via-"+label,root,sch,["--routed-net","A"],(2,1))["routed_continuity"]
        assert routed["status"]==("witnessed" if expected else "unresolved"),routed
        for witness in routed["witnesses"]:
            inspect_walk(witness,rect(left,-1,1,1))
        records["via-"+label]=routed["status"]
    return records


def authentic_walks(args, original, sch, domain):
    records={}
    for label,shift in (("original",None),("bridge",0),("remote",10)):
        root=copy.deepcopy(original)
        if shift is not None:
            signal=root.find("./drawing/board/signals/signal[@name='D-']")
            ET.SubElement(signal,"wire",**{a:str(Decimal(v)+shift) for a,v in ENDS.items()},
                          width=".1",layer="1")
        for net in ("D+","D-"):
            doc=run_cli(args,"authentic-"+label+("-plus" if net=="D+" else "-minus"),
                        root,sch,["--routed-net",net,"--contact-nets","D+","D-"])
            routed,contact=doc["routed_continuity"],doc["inter_net_contact"]
            assert routed["status"]=="witnessed" and len(routed["witnesses"])==4
            assert routed["board_domain"]["status"]==contact["board_domain"]["status"]=="bounded"
            for witness in routed["witnesses"]:
                inspect_walk(witness,domain)
            assert len(contact["contacts"])==(4 if label=="bridge" else 0)
            for row in contact["contacts"]:
                on_board_disk(domain,row["center_nm"],row["disk_radius_nm"])
                COUNTS["cross_net_disks"]+=1
            records[label+net]={"paths":4,"contacts":len(contact["contacts"])}
    # Move only the known closed outline; preserve every original signal/pin.
    root=copy.deepcopy(original)
    for row in root.findall("./drawing/board/plain/wire"):
        if row.get("layer")=="20":
            for axis in ("x1","x2"):
                row.set(axis,str(Decimal(row.get(axis))+120))
    signal=root.find("./drawing/board/signals/signal[@name='D-']")
    ET.SubElement(signal,"wire",**ENDS,width=".1",layer="1")
    for net in ("D+","D-"):
        doc=run_cli(args,"authentic-outline-remote"+("-plus" if net=="D+" else "-minus"),root,sch,
                    ["--routed-net",net,"--contact-nets","D+","D-"])
        routed,contact=doc["routed_continuity"],doc["inter_net_contact"]
        assert routed["board_domain"]["status"]==contact["board_domain"]["status"]=="bounded"
        assert routed["status"]==contact["status"]=="unresolved"
        assert not routed["witnesses"] and not contact["contacts"]
    records["outline_only_plus_120mm"]={"queried_nets":["D+","D-"],"paths":0,"contacts":0,
                                       "contacts_declared":605,"nets_declared":163}
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--finite",type=Path,required=True)
    p.add_argument("--milling",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    domains=authentic_domains()
    for donor,(sch,brd,_,_,_,s_digest,b_digest) in SOURCES.items():
        assert sha256((args.raw/sch).read_bytes()).hexdigest()==s_digest
        assert sha256((args.raw/brd).read_bytes()).hexdigest()==b_digest
    verify_original(args.raw/SOURCES["feather"][1])
    finite=json.loads(args.finite.read_text())
    assert set(finite)==set(CASES)
    for name,row in finite.items():
        assert row["passed"] and row["observed"]==("witnessed" if CASES[name] else "unresolved")
        data,signal,bores=make(args.raw,name)
        assert sha256(data).hexdigest()==row["board_sha256"]
        if CASES[name]:
            assert check_witness(name,row["source_witness"],signal,bores)==row["witness"]
            feather=name.startswith("feather")
            inspect_walk(row["source_witness"],domains["Feather" if feather else "Due"],feather)
            if feather:
                reject_translated_barrel(row["source_witness"],domains["Feather"],True)
    milling=json.loads(args.milling.read_text())
    assert {k:len(v["contacts"]) for k,v in milling.items()}=={
        "cut":0,"partial":2,"remote":4,"drawing":4,"open":0}
    for result in milling.values():
        for row in result["contacts"]:
            on_board_disk(domains["Due"],row["center_nm"],row["disk_radius_nm"])
            COUNTS["cross_net_disks"]+=1
    original=ET.parse(args.raw/SOURCES["due"][1]).getroot()
    sch=args.raw/SOURCES["due"][0]
    report={"crossings":crossing_controls(args,original,sch),
            "notch_and_collar":notch_and_collar_controls(args),
            "authentic":authentic_walks(args,original,sch,domains["Due"])}
    report.update(independent_counts=dict(COUNTS),frozen_finite_cases=len(finite),
                  positive_finite_cases=sum(CASES.values()),scope="Nominal supported board/cutout geometry; no manufactured clearance or hosted score")
    (args.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":True,"counts":dict(COUNTS),"finite":len(finite),
                      "notch_and_collar":report["notch_and_collar"],
                      "authentic":report["authentic"]},indent=2))


if __name__=="__main__":
    main()
