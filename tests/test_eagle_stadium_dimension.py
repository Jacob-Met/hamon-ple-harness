"""Capsule domains retain exact source identity across semicircle subdivisions."""
from fractions import Fraction as F
import unittest
from xml.etree import ElementTree as ET

from hamon_eagle_copper import dimension_records, milling_edges

from shapely.geometry import Point
from hamon_eagle_material import nominal_board_domain, _outline_cycles, _stadium_geometry


def loop(points, sweeps=()):
    return [(a, points[(i+1) % len(points)], sweeps[i] if sweeps else 0)
            for i, a in enumerate(points)]


OUTER = loop([(-400, -400), (400, -400), (400, 400), (-400, 400)])


def stadium(split=False, center=(0, 0), half=80, radius=40):
    x, y = center
    if split:
        points = [(x-half,y-radius),(x+half,y-radius),(x+half+radius,y),
                  (x+half,y+radius),(x-half,y+radius),(x-half-radius,y)]
        return loop(points, [0,90,90,0,90,90])
    return loop([(x-half,y-radius),(x+half,y-radius),
                 (x+half,y+radius),(x-half,y+radius)], [0,180,0,180])


def rows(edges, frame=0):
    def at(p):
        x, y = map(F, p)
        if frame == 1:
            return ((3*x-4*y)/5, (4*x+3*y)/5)
        if frame == 2:
            return ((-3*x-4*y)/5, (-4*x+3*y)/5)
        return (x, y)
    return [dict(a=at(a), b=at(b), curve=-s if frame==2 else s,
                 source=("plain","wire",i)) for i,(a,b,s) in enumerate(edges)]


def reverse(records):
    return [dict(row, a=row["b"], b=row["a"], curve=-row["curve"])
            for row in records[::-1]]


def partition(edges):
    result=[]
    for a,b,s in edges:
        if s:
            result.append((a,b,s))
            continue
        fractions=(F(0),F(1,160),F(1,2),F(1))
        points=[tuple(a[k]+t*(b[k]-a[k]) for k in range(2)) for t in fractions]
        result.extend((p,q,0) for p,q in zip(points,points[1:]))
    return result


class StadiumDimensionTests(unittest.TestCase):
    def test_ordinary_dimension_parser_admits_half_caps_only_on_layer_twenty(self):
        for split in (False,True):
            for frame in (0,1,2):
                board=ET.fromstring("<board><plain/><libraries/><elements/></board>")
                for row in rows(OUTER+stadium(split),frame):
                    attributes=dict(width="0",layer="20",curve=str(row["curve"]))
                    for suffix,key in (("1","a"),("2","b")):
                        for axis,value in zip(("x","y"),row[key]):
                            attributes[axis+suffix]=str(float(value/1_000_000))
                    ET.SubElement(board.find("plain"),"wire",**attributes)
                domain,info=nominal_board_domain(dimension_records(board))
                self.assertEqual(len(domain.interiors),1)
                self.assertEqual(info["hole_semicircle_count"],2)
        for sweep in (45,120,270,360):
            board.find("plain/wire").set("curve",str(sweep))
            with self.assertRaises(ValueError):
                dimension_records(board)
        milling=ET.Element("wire",x1="0",y1="-1",x2="0",y2="1",
                           width="0",layer="46",curve="180")
        with self.assertRaises(ValueError):
            list(milling_edges([(None,[milling])]))

    def test_arc_encoding_order_winding_and_bridge_partition_preserve_domain(self):
        for frame in (0,1,2):
            reference,_=nominal_board_domain(rows(OUTER+stadium(),frame))
            for split in (False,True):
                for divide in (False,True):
                    inner=stadium(split)
                    if divide:
                        inner=partition(inner)
                    records=rows(OUTER+inner,frame)
                    for variant in (records,records[::2]+records[1::2],reverse(records)):
                        domain,info=nominal_board_domain(variant)
                        self.assertTrue(domain.equals(reference))
                        self.assertEqual(info["model"],"outer_outline_with_stadium_dimension_hole")
                        self.assertEqual(info["hole_semicircle_count"],2)
                        self.assertEqual(info["hole_wire_count"],len(inner))
                        self.assertCountEqual(map(tuple,info["sources"]),(r["source"] for r in records))
                        self.assertCountEqual(map(tuple,info["hole_sources"]),(r["source"] for r in records[4:]))
                        sources=[tuple(source) for cap in info["hole_semicircle_sources"] for source in cap]
                        self.assertCountEqual(sources,(r["source"] for r in records if r["curve"]))
                        self.assertLessEqual(info["hole_arc_error_upper_bound_nm"],1)
                        self.assertEqual(F(info["hole_capsule_radius_squared_nm2"]),1600)

    def test_every_inner_boundary_segment_stays_outside_the_exact_capsule(self):
        # The independently derived closest distance to the horizontal spine
        # checks the whole polygon edge, not merely its endpoints.
        for split in (False,True):
            domain,_=nominal_board_domain(rows(OUTER+stadium(split)))
            points=list(domain.interiors[0].coords)
            for a,b in zip(points,points[1:]):
                ax,ay=map(F,a);bx,by=map(F,b)
                dx,dy=bx-ax,by-ay
                candidates=[F(0),F(1)]
                if dx:
                    candidates.extend((F(x)-ax)/dx for x in (-80,80))
                if dy:
                    candidates.append(-ay/dy)
                denominator=dx*dx+dy*dy
                if denominator:
                    candidates.extend(-((ax-x)*dx+ay*dy)/denominator for x in (-80,80))
                distances=[]
                for t in candidates:
                    if 0<=t<=1:
                        x,y=ax+t*dx,ay+t*dy
                        nearest=min(F(80),max(F(-80),x))
                        distances.append((x-nearest)**2+y*y)
                self.assertGreater(min(distances),1600)
            self.assertTrue(domain.covers(Point(112,32).buffer(1)))
            self.assertFalse(domain.intersects(Point(104,24).buffer(1)))

    def test_stadium_outer_boundary_is_inside_the_exact_capsule(self):
        for split in (False,True):
            domain,info=nominal_board_domain(rows(stadium(split)))
            self.assertEqual(info["model"],"single_stadium_outline")
            for x,y in domain.exterior.coords:
                nearest=min(80,max(-80,x))
                self.assertLess((x-nearest)**2+y*y,40**2)

    def test_two_stadium_holes_are_separated_or_conservatively_refused(self):
        outer=loop([(-1000,-1000),(1000,-1000),(1000,1000),(-1000,1000)])
        first=stadium(center=(-200,0))
        second=stadium(True,center=(200,0))
        for records in (rows(outer+first+second),reverse(rows(outer+second+first,1))):
            domain,info=nominal_board_domain(records)
            self.assertEqual(len(domain.interiors),2)
            self.assertEqual([h["semicircle_count"] for h in info["dimension_holes"]],[2,2])
        for center in ((40,0),(30,0),(-200,0),(-80,0)):
            # Touching, crossing, duplicate and nested/overlapping cutouts.
            with self.assertRaises(ValueError):
                nominal_board_domain(rows(outer+first+stadium(True,center=center)))

    def test_nested_cap_hole_is_not_hidden_by_endpoint_chord_parity(self):
        outer=loop([(-5000,-5000),(10000,-5000),(10000,5000),(-5000,5000)])
        capsule=stadium(center=(2000,0),half=2000,radius=1000)
        def square(x):
            return loop([(x-20,-20),(x+20,-20),(x+20,20),(x-20,20)])
        for frame in (0,1,2):
            with self.assertRaises(ValueError):
                nominal_board_domain(rows(outer+capsule+square(-500),frame))
            domain,info=nominal_board_domain(rows(outer+capsule+square(-1500),frame))
            self.assertEqual(len(domain.interiors),2)
            self.assertEqual(info["hole_count"],2)

    def test_broken_tangents_mismatched_radii_and_inward_caps_abstain(self):
        for kind in ("kink","unequal","inward","unsupported"):
            edges=stadium()
            if kind=="kink":
                a,b,s=edges[0]
                edges=[(a,(0,-39),0),((0,-39),b,0)]+edges[1:]
            elif kind=="unequal":
                points=[(-80,-40),(80,-30),(80,30),(-80,40)]
                edges=loop(points,[0,180,0,180])
            elif kind=="inward":
                edges=[(a,b,-s) for a,b,s in edges]
            else:
                a,b,_=edges[1];edges[1]=(a,b,120)
            with self.assertRaises(ValueError):
                nominal_board_domain(rows(OUTER+edges))

    def test_short_real_spine_and_radius_do_not_gain_support_by_subdivision(self):
        for half,radius in ((4,40),(80,8)):
            for split in (False,True):
                with self.assertRaises(ValueError):
                    nominal_board_domain(rows(OUTER+partition(stadium(split,half=half,radius=radius))))

    def test_cap_circle_centers_are_exact_after_rational_rotation(self):
        records=rows(stadium(True),1)
        geometry=_stadium_geometry(_outline_cycles(records)[0])
        centers=geometry["centers"]
        self.assertCountEqual(centers,[(F(-48),F(-64)),(F(48),F(64))])
        self.assertEqual(geometry["radius2"],1600)


if __name__=="__main__":
    unittest.main()
