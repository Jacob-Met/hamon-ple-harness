"""Nominal board-boundary tests with independent rectangle and circle controls."""
import copy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from hamon_eagle_copper import dimension_records, inter_net_contact, routed_continuity
from hamon_eagle_material import material_continuity, nominal_board_domain
from shapely.geometry import Point, LineString

def edge(a, b, layer=20, curve=0, width=0):
    text=lambda v: str(float(v)) if isinstance(v,F) else str(v)
    return ET.Element("wire", x1=text(a[0]), y1=text(a[1]), x2=text(b[0]), y2=text(b[1]),
                      width=str(width), layer=str(layer), curve=str(curve))

def outline(points, **kwargs):
    return [edge(a, b, **kwargs) for a, b in zip(points, points[1:]+points[:1])]

def rectangle(left=-2, right=2, bottom=-2, top=2, **kwargs):
    return outline([(left,bottom),(right,bottom),(right,top),(left,top)], **kwargs)

def rounded_rectangle(radius):
    r=F(radius)
    return [edge((r,0),(3*r,0)), edge((3*r,0),(4*r,r),curve=90),
            edge((4*r,r),(4*r,3*r)), edge((4*r,3*r),(3*r,4*r),curve=90),
            edge((3*r,4*r),(r,4*r)), edge((r,4*r),(0,3*r),curve=90),
            edge((0,3*r),(0,r)), edge((0,r),(r,0),curve=90)]

def reverse_rows(rows):
    result=[]
    for row in reversed(rows):
        other=copy.deepcopy(row)
        for axis in ("x","y"):
            other.set(axis+"1",row.get(axis+"2"));other.set(axis+"2",row.get(axis+"1"))
        other.set("curve",str(-F(row.get("curve","0"))))
        result.append(other)
    return result

def board(rows):
    root=ET.fromstring("<eagle><drawing><board><plain/><signals/></board></drawing></eagle>")
    b=root.find(".//board")
    b.find("plain").extend(copy.deepcopy(rows))
    for name,a,z in (("A",(-1,0),(1,0)),("B",(0,-1),(0,1))):
        ET.SubElement(b.find("signals"),"signal",name=name).append(edge(a,z,layer=1,width=".2"))
    return root,b

class BoardDomainTests(unittest.TestCase):
    def test_isolated_reflex_quarter_excludes_the_entire_circular_recess(self):
        # The left tip selects counterclockwise canonical traversal here. Its
        # reflection starts at the opposite side and selects clockwise traversal.
        corners = [(0, 0), (8, 0), (8, 8), (5, 8), (5, 5), (4, 4), (0, 4), (-1, 2)]
        for mirror in (1, -1):
            rows = outline([(mirror*x, y) for x, y in corners])
            rows[4].set("curve", str(-90*mirror))
            for edges in (rows, reverse_rows(rows), rows[::2]+rows[1::2]):
                _, document = board(edges)
                domain, evidence = nominal_board_domain(dimension_records(document))
                center = (mirror*4_000_000, 5_000_000)
                self.assertFalse(domain.covers(Point(center)))
                self.assertLessEqual(evidence["arc_sagitta_upper_bound_nm"], 1)
                self.assertTrue(domain.covers(Point(mirror*4_850_000, 4_150_000).buffer(5_000)))
                self.assertFalse(domain.intersects(Point(mirror*4_600_000, 4_400_000).buffer(5_000)))
                # The closest point on every complete returned edge must clear
                # the unit-radius recess, including the middle of each tangent.
                points = [(F(x)-center[0], F(y)-center[1]) for x,y in domain.exterior.coords]
                for (x,y),(u,v) in zip(points,points[1:]):
                    dx,dy = u-x,v-y
                    t = max(0, min(1, -(x*dx+y*dy)/(dx*dx+dy*dy)))
                    self.assertGreaterEqual((x+t*dx)**2+(y+t*dy)**2, 1_000_000**2)

    def query(self, rows):
        root,_=board(rows)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"board.brd";ET.ElementTree(root).write(p,encoding="utf-8")
            return inter_net_contact(p,("A","B"))

    def test_registered_inside_outside_and_edge_straddling_contacts(self):
        inside=self.query(rectangle())
        hit,=inside["contacts"]
        self.assertEqual(inside["board_domain"]["status"],"bounded")
        self.assertEqual(hit["center_nm"],[0.,0.])
        self.assertEqual(hit["disk_radius_nm"],49999.)
        self.assertAlmostEqual(hit["overlap_lower_bound_mm2"],.039998400016,places=14)
        outside=self.query(rectangle(10,14))
        self.assertEqual(outside["status"],"unresolved")
        self.assertEqual(outside["contacts"],[])
        partial=self.query(rectangle(.05,2))
        hit,=partial["contacts"]
        self.assertGreater(hit["center_nm"][0]-hit["disk_radius_nm"],50000)
        # Width 49996 nm after both 2 nm guards; height 199996 nm.
        self.assertAlmostEqual(hit["overlap_lower_bound_mm2"],49996*199996/1e12,places=14)
        for left in (.1,.101):
            self.assertEqual(self.query(rectangle(left,2))["contacts"],[])

    def test_order_winding_layer_alias_and_pen_do_not_change_the_domain(self):
        rows=rectangle(width=".254",layer="020")
        permutations=[rows,reverse_rows(rows),[rows[i] for i in (2,0,3,1)]]
        for value in permutations:
            hit,=self.query(value)["contacts"]
            self.assertEqual(hit["disk_radius_nm"],49999.)
            self.assertAlmostEqual(hit["overlap_lower_bound_mm2"],.039998400016,places=14)

    def test_declared_ambiguous_boundaries_never_fall_back_to_unbounded_copper(self):
        shapes=[rectangle()[:-1],rectangle()+[edge((-2,-2),(0,0))],
                rectangle()+rectangle(-.05,.05,-.05,.05),
                rectangle()+rectangle(10,14),
                outline([(-2,-2),(2,2),(-2,2),(2,-2)]),
                [ET.Element("circle",x="0",y="0",radius="2",width="0",layer="20")]]
        bad_arc=rectangle();bad_arc[0].set("curve","45");shapes.append(bad_arc)
        for rows in shapes:
            with self.subTest(rows=[r.attrib for r in rows]):
                result=self.query(rows)
                self.assertEqual(result["status"],"unresolved",result)
                self.assertEqual(result["contacts"],[],result)
                self.assertEqual(result["board_domain"]["status"],"unresolved",result)
        absent=self.query([])
        self.assertEqual(absent["board_domain"]["status"],"unbounded_artwork")
        self.assertEqual(absent["status"],"contact_witnessed")

    def test_convex_quarters_have_the_independently_known_circle_margin(self):
        for r,center,good,bad in (
            (1,(460000,280000),90000,110000),
            (F(127,50),(1168400,711200),250000,260000)):
            rows=rounded_rectangle(r)
            domains=[]
            for version in (rows,reverse_rows(rows),rows[3:]+rows[:3]):
                _,b=board(version)
                domain,info=nominal_board_domain(dimension_records(b))
                self.assertEqual(info["quarter_count"],4)
                self.assertLessEqual(info["arc_sagitta_upper_bound_nm"],1)
                self.assertTrue(domain.covers(Point(center).buffer(good,quad_segs=64)))
                self.assertFalse(domain.covers(Point(center).buffer(bad,quad_segs=64)))
                domains.append(domain)
            self.assertAlmostEqual(domains[0].area,domains[1].area,delta=100)
            self.assertAlmostEqual(domains[0].area,domains[2].area,delta=100)
        _,b=board(rounded_rectangle(1))
        domain,_=nominal_board_domain(dimension_records(b))
        self.assertTrue(domain.covers(LineString([(700000,600000),(460000,280000)]).buffer(90000)))
        self.assertFalse(domain.covers(Point(460000,280000).buffer(100000,quad_segs=256)))
        wrong=rounded_rectangle(1);wrong[-1].set("curve","-90")
        self.assertEqual(self.query(wrong)["contacts"],[])

    def test_same_net_notch_requires_the_real_in_board_detour(self):
        corners=[(-1,-1),(1,-1),(1,1),(.15,1),(.15,-.2),(-.15,-.2),(-.15,1),(-1,1)]
        root,b=board(outline(corners))
        b.find("signals").clear()
        net=ET.SubElement(b.find("signals"),"signal",name="A")
        net.append(edge((-.5,0),(.5,0),layer=1,width=".2"))
        lib=ET.SubElement(ET.SubElement(b,"libraries"),"library",name="Pads")
        pkg=ET.SubElement(ET.SubElement(lib,"packages"),"package",name="P")
        for pin,x in (("L","-.5"),("R",".5")):
            ET.SubElement(pkg,"smd",name=pin,x=x,y="0",dx=".1",dy=".1",layer="1")
        ET.SubElement(ET.SubElement(b,"elements"),"element",name="P",library="Pads",package="P",x="0",y="0")
        prediction={"schematic_sha256":"fixture","physical_contacts":[
            {"part":"P","pad":pin,"net":"A"} for pin in ("L","R")]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"board.brd"
            for detour in (False,True):
                if detour:
                    for a,z in [((-.5,0),(-.5,-.45)),((-.5,-.45),(.5,-.45)),((.5,-.45),(.5,0))]:
                        net.append(edge(a,z,layer=1,width=".2"))
                ET.ElementTree(root).write(p,encoding="utf-8")
                result=routed_continuity(prediction,p,"A")
                self.assertEqual(result["status"],"witnessed" if detour else "unresolved",result)
                if detour:
                    domain,_=nominal_board_domain(dimension_records(b))
                    for step in result["witnesses"][0]["steps"]:
                        path=step["material"]["path_nm"]
                        self.assertTrue(domain.covers(LineString(path) if len(path)>1 else Point(path[0])))

    def test_planar_via_caps_do_not_create_a_missing_plated_barrel(self):
        pads=[("pad","P","top"),("pad","P","bottom")];via=("via",0)
        primitives=[dict(source=via,layers=(1,16),shape="disk",center=(0,0),diameter=600000,plated=True)]
        for pad,layer in zip(pads,(1,16)):
            primitives.append(dict(source=pad,terminal=pad,layers=(layer,),shape="disk",
                                   center=(250000,0),diameter=20000))
        bores=[(0,0,300000,via)]
        for left,expected in ((-1,True),(.2,False),(.15,False),(-.16,True)):
            _,b=board(rectangle(left,1,-1,1))
            domain,_=nominal_board_domain(dimension_records(b))
            result=material_continuity(pads,primitives,bores,domain=domain)
            self.assertEqual(bool(result["witnesses"]),expected,(left,result))


    def test_placed_outline_sources_and_dependency_failures_remain_declared(self):
        root,b=board([])
        for row in b.findall("signals/signal/wire"):
            for name in ("x1","x2"): row.set(name,str(float(row.get(name))*.2+9.5))
            for name in ("y1","y2"): row.set(name,str(float(row.get(name))*.2+22))
        libraries=ET.SubElement(b,"libraries")
        for libname,offset in (("Actual",0),("Other",100)):
            library=ET.SubElement(libraries,"library",name=libname)
            package=ET.SubElement(ET.SubElement(library,"packages"),"package",name="EDGE")
            package.append(edge((0,0),(1,0),layer=21))
            package.extend(rectangle(1+offset,3+offset,0,1))
            for pin,y in (("L",".4"),("R",".6")):
                ET.SubElement(package,"smd",name=pin,x="2",y=y,dx=".1",dy=".1",layer="1")
        elements=ET.SubElement(b,"elements")
        placed=ET.SubElement(elements,"element",name="EDGE1",library="Actual",package="EDGE",
                             x="10",y="20",rot="R90")
        prediction={"schematic_sha256":"fixture","physical_contacts":[
            {"part":"EDGE1","pad":pin,"net":"A"} for pin in ("L","R")]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"placed.brd"
            def queries():
                ET.ElementTree(root).write(path,encoding="utf-8")
                return inter_net_contact(path,("A","B")),routed_continuity(prediction,path,"A")
            contact,route=queries()
            self.assertEqual(contact["status"],"contact_witnessed",contact)
            self.assertEqual(route["status"],"witnessed",route)
            self.assertEqual({tuple(s) for s in contact["board_domain"]["sources"]},{
                ("element","EDGE1","library","Actual","package","EDGE","wire",i)
                for i in (1,2,3,4)})
            for xpos in ("10","20"):
                placed.set("x",xpos)
                for failure in ("missing","constructor"):
                    manager=(patch.dict("sys.modules",{"hamon_eagle_material":None})
                             if failure=="missing" else
                             patch("hamon_eagle_material.nominal_board_domain",
                                   side_effect=ValueError("unresolved declared fixture")))
                    with manager:
                        for result in queries():
                            self.assertEqual(result["status"],"unresolved",result)
                            self.assertEqual(result["board_domain"]["status"],"unresolved",result)
            contact,route=queries()
            self.assertEqual(contact["contacts"],[])
            self.assertEqual(route["witnesses"],[])
            elements.remove(placed)
            ET.ElementTree(root).write(path,encoding="utf-8")
            unplaced=inter_net_contact(path,("A","B"))
            self.assertEqual(unplaced["status"],"contact_witnessed",unplaced)
            self.assertEqual(unplaced["board_domain"]["status"],"unbounded_artwork")


    def test_placed_closed_milling_capsules_use_separate_outward_boxes(self):
        points=[(-.3,-.3),(0,-.6),(.3,-.3),(.3,.3),(0,.6),(-.3,.3)]
        slot=[edge(a,b,layer=46,curve=s) for a,b,s in
              zip(points,points[1:]+points[:1],(90,90,0,90,90,0))]
        root,b=board(rectangle())
        lib=ET.SubElement(ET.SubElement(b,"libraries"),"library",name="USB")
        pkg=ET.SubElement(ET.SubElement(lib,"packages"),"package",name="SLOTS")
        pkg.extend(copy.deepcopy(slot))
        el=ET.SubElement(ET.SubElement(b,"elements"),"element",name="X6",
                        library="USB",package="SLOTS",x="0",y="0",rot="R90")
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"milled.brd"
            def run():
                ET.ElementTree(root).write(path,encoding="utf-8")
                return inter_net_contact(path,("A","B"))
            cut=run()
            self.assertEqual(cut["contacts"],[])
            self.assertIn("milling",cut,cut)
            self.assertEqual(len(cut["milling"]["conservative_capsule_boxes"]),1)
            el.set("rot","MR0")
            self.assertEqual(run()["contacts"],[])
            el.set("rot","R90")
            el.set("x","1")
            self.assertEqual(run()["status"],"contact_witnessed")
            pkg.remove(pkg[-1])
            broken=run()
            self.assertEqual(broken["contacts"],[])
            self.assertIn("open or branched",broken["detail"])
            pkg[:]=[];pkg.extend(copy.deepcopy(slot))
            pkg[0].set("curve","-90")
            self.assertEqual(run()["contacts"],[])
            pkg[:]=[]
            for shift in (-.6,.6):
                for row in slot:
                    other=copy.deepcopy(row)
                    for suffix in ("1","2"):
                        other.set("x"+suffix,str(float(F(other.get("x"+suffix))+F(str(shift)))))
                    pkg.append(other)
            el.set("x","0");el.set("rot","R0")
            separated=run()
            self.assertEqual(separated["status"],"contact_witnessed",separated)
            self.assertEqual(len(separated["milling"]["conservative_capsule_boxes"]),2)

if __name__=="__main__":
    unittest.main()
