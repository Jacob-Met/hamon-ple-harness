"""Source STEP controls for partial physical IFC coverage on the ordinary route."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from hamon_ifc_spatial_clash import run_router_ifc_clash
from test_ifc_component_containment import Scene


class PartialScene(Scene):
    def __init__(self):
        super().__init__()
        point = self.add("IFCCARTESIANPOINT((0.,0.,0.))")
        frame = self.add(f"IFCAXIS2PLACEMENT3D(#{point},$,$)")
        self.place = self.add(f"IFCLOCALPLACEMENT($,#{frame})")
        self.context = self.add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-5,#{frame},$)")

    def product(self, kind, name, items=None, context=None):
        context = self.context if context is None else context
        shape = "$"
        if items is not None:
            ids = ",".join(f"#{i}" for i in items)
            rep = self.add(f"IFCSHAPEREPRESENTATION(#{context},'Body','SweptSolid',({ids}))")
            shape = "#" + str(self.add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{rep}))"))
        return self.add(f"{kind}('{name}',$,'{name}',$,$,#{self.place},{shape},$,$)")

    def text(self):
        return ("ISO-10303-21;HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;DATA;\n"
                + "\n".join(self.records) + "\nENDSEC;END-ISO-10303-21;")

    def partial(self, **kw):
        return run_router_ifc_clash(dict({"ifc_text": self.text(), "cross_discipline_only": False,
                                          "allow_partial": True}, **kw))


def rows(report):
    return {item["step_id"]: item for item in report["products"]}


class PartialIfcTests(unittest.TestCase):
    def test_original_roles_and_unknown_clash_pairs(self):
        scene = PartialScene()
        wall = scene.product("IFCWALL", "wall", [scene.box(1, 0)])
        duct = scene.product("IFCDUCTSEGMENT", "duct", [scene.box(1, 0)])
        missing = scene.product("IFCWALL", "missing")
        scene.product("IFCSPACE", "spatial", [scene.box(1, 0)])
        scene.product("IFCVIRTUALELEMENT", "virtual", [scene.box(1, 0)])
        r = scene.partial()
        self.assertEqual((r["status"], r["ok"], r["role_counts"]), ("partial", False, {"physical":3, "other":2}))
        self.assertEqual(r["known_volume_step_ids"], [wall, duct])
        self.assertEqual(r["unresolved_geometry_step_ids"], [missing])
        self.assertAlmostEqual(r["known_occurrence_volume_sum_m3"], 2.)
        self.assertIsNone(r["total_solid_volume_m3"])
        self.assertIsNone(r["hard_clash_count"])
        self.assertEqual((r["requested_pair_count"],r["evaluated_pair_count"],r["missing_geometry_pair_count"]),
                         (3, 1, 2))
        self.assertEqual((r["known_hard_clash_count"], r["min_clearance_m"]), (1, 0))
        self.assertEqual([(p["element_a"],p["element_b"]) for p in r["all_pairs"]], [(f"#{wall}", f"#{duct}")])
        self.assertEqual(scene.partial(cross_discipline_only=True)["requested_pair_count"], 2)
        self.assertNotIn("physical_element_count", scene.partial(allow_partial=False))

    def test_all_voids_preserved_and_independent_filling_survives(self):
        s = PartialScene()
        host = s.product("IFCWALL","host",[s.box(4,0)])
        good = s.product("IFCOPENINGELEMENT","first",[s.box(1,0)])
        bad = s.product("IFCOPENINGELEMENT","second",[99999])
        filling = s.product("IFCWINDOW","window",[s.box(.5,0)])
        distant = s.product("IFCBEAM","distant",[s.box(1,10)])
        links = [s.add(f"IFCRELVOIDSELEMENT('v',$,$,$,#{host},#{i})") for i in (good,bad)]
        s.add(f"IFCRELFILLSELEMENT('f',$,$,$,#{bad},#{filling})")
        r = s.partial()
        self.assertEqual(rows(r)[host]["void_relation_ids"], links)
        self.assertIsNone(rows(r)[host]["volume_m3"])
        self.assertIn(str(bad), rows(r)[host]["reason"])
        self.assertIn("99999", rows(r)[bad]["reason"])
        self.assertEqual([v["cutter_step_id"] for v in r["void_relationships"]], [good,bad])
        self.assertEqual(r["filling_relationships"][0]["filling_step_id"], filling)
        self.assertEqual(r["known_volume_step_ids"], [filling,distant])
        self.assertGreater(r["known_min_clearance_m"],0.)
        self.assertIsNone(r["min_clearance_m"])

    def test_certified_empty_and_missing_are_distinct(self):
        s = PartialScene()
        host = s.product("IFCWALL","host",[s.box(1,0)])
        opening = s.product("IFCOPENINGELEMENT","cut",[s.box(2,0)])
        s.add(f"IFCRELVOIDSELEMENT('v',$,$,$,#{host},#{opening})")
        s.product("IFCBEAM","nearby",[s.box(1,5)])
        r = s.partial()
        self.assertTrue(r["ok"], r)
        self.assertEqual((rows(r)[host]["geometry_status"],rows(r)[host]["volume_m3"]),("empty",0.))
        self.assertEqual((r["requested_pair_count"],r["certified_empty_pair_count"],r["evaluated_pair_count"]),
                         (1,1,0))
        self.assertIsNone(r["min_clearance_m"])
        self.assertAlmostEqual(r["total_solid_volume_m3"],1.)

    def test_each_shared_fault_reaches_its_dependent_occurrences(self):
        for damage in ("shared_item","placement","context","foreign_context"):
            with self.subTest(damage=damage):
                s = PartialScene()
                normal = s.place
                root = s.context
                item = s.box(1,0)
                if damage == "shared_item":
                    item = s.add("IFCCSGSOLID(#999999)")
                elif damage == "placement":
                    s.place = s.add("IFCLOCALPLACEMENT(#999999,#999998)")
                elif damage == "context":
                    root = 999999
                else:
                    root = s.add("IFCGEOMETRICREPRESENTATIONCONTEXT($,'Separate',3,1.E-5,#5,$)")
                    s.add(f"IFCPROJECT('p',$,'p',$,$,$,$,(#{s.context}),#2)")
                bad = [s.product("IFCBUILDINGELEMENTPROXY","duplicate-guid",[item],root) for _ in range(2)]
                s.place = normal
                good = s.product("IFCBEAM","good",[s.box(1,10)])
                r = s.partial()
                self.assertEqual((r["unresolved_geometry_step_ids"],r["known_volume_step_ids"]), (bad,[good]))
                self.assertEqual((r["missing_geometry_pair_count"],r["all_pairs"]),(3,[]))

    def test_unassignable_and_structurally_malformed_source_faults(self):
        records = ["IFCRELVOIDSELEMENT('v',$,$,$,#99999,#99998)", "IFCOUNKNOWNENTITY($)",
                   "IFCRELVOIDSELEMENT('v',$,$,$,,#99998)"]
        for damage in (*records,"duplicate","truncated","extra DATA","units"):
            with self.subTest(damage=damage):
                s = PartialScene()
                s.product("IFCWALL","wall",[s.box(1,0)])
                if damage in records:
                    s.add(damage)
                elif damage == "duplicate":
                    s.records.append(s.records[0])
                elif damage == "extra DATA":
                    s.records.append("ENDSEC;DATA;")
                elif damage == "units":
                    s.records[1] = "#2=IFCUNITASSIGNMENT((#99999));"
                text = s.text().removesuffix("END-ISO-10303-21;") if damage == "truncated" else s.text()
                r = s.partial(ifc_text=text)
                self.assertEqual((r["status"],r["ok"],r["total_solid_volume_m3"]),("unresolved_model",False,None))

    def test_geometry_faults_are_never_clear(self):
        for fault in ("bad_position","circle","indexed","false_closed","open","closed"):
            with self.subTest(fault=fault):
                s = PartialScene()
                probe = s.product("IFCBEAM","probe",[s.box(1,10)])
                if fault == "bad_position":
                    item = s.box(1,10)
                    pt = s.add("IFCCARTESIANPOINT((10.,0.,0.))")
                    s.records[item-1] = re.sub(r"(IFCEXTRUDEDAREASOLID\(#\d+,)#\d+",
                                                rf"\g<1>#{pt}", s.records[item-1])
                elif fault in ("false_closed","open","closed"):
                    pts = s.add("IFCCARTESIANPOINTLIST3D(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))")
                    faces = "(1,3,2),(1,2,4),(2,3,4)" + (",(3,1,4)" if fault == "closed" else "")
                    flag = ".F." if fault == "open" else ".T."
                    item = s.add(f"IFCTRIANGULATEDFACESET(#{pts},$,{flag},({faces}),$)")
                else:
                    if fault == "circle":
                        profile = s.add("IFCCIRCLEPROFILEDEF(.AREA.,$,$,1.)")
                    else:
                        pts = s.add("IFCCARTESIANPOINTLIST2D(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))")
                        curve = s.add(f"IFCINDEXEDPOLYCURVE(#{pts},(IFCLINEINDEX((1,2,3,1))),.F.)")
                        profile = s.add(f"IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{curve})")
                    item = s.sweep(profile, (0,0,0), 1.)
                wall = s.product("IFCWALL","wall",[item])
                r = s.partial()
                if fault == "closed":
                    self.assertTrue(r["ok"],r)
                    self.assertAlmostEqual(rows(r)[wall]["volume_m3"],1/6)
                else:
                    self.assertEqual(r["known_volume_step_ids"],[probe])
                    self.assertEqual(r["unresolved_geometry_step_ids"],[wall])
                    self.assertEqual((r["missing_geometry_pair_count"],r["all_pairs"]),(1,[]))

    def test_polygonal_point_list_retains_step_numeric_types(self):
        for coordinate, flag in (("1.", ".T."), (".T.", ".T."), ("'1.'", ".T."), ("1.", "'.F.'")):
            with self.subTest(coordinate=coordinate, flag=flag):
                s = PartialScene()
                pts = s.add(f"IFCCARTESIANPOINTLIST3D(((0.,0.,0.),({coordinate},0.,0.),(0.,1.,0.),(0.,0.,1.)))")
                faces = [s.add(f"IFCINDEXEDPOLYGONALFACE(({loop}))")
                         for loop in ("1,3,2", "1,2,4", "2,3,4", "3,1,4")]
                refs = ",".join(f"#{i}" for i in faces)
                body = s.add(f"IFCPOLYGONALFACESET(#{pts},{flag},({refs}),$)")
                wall = s.product("IFCWALL","polyhedral",[body])
                s.product("IFCBEAM","probe",[s.box(1,10)])
                report = s.partial()
                if coordinate == "1." and flag == ".T.":
                    self.assertTrue(report["ok"], report)
                    self.assertAlmostEqual(rows(report)[wall]["volume_m3"], 1/6)
                else:
                    self.assertEqual(report["unresolved_geometry_step_ids"], [wall])
                    self.assertEqual(report["missing_geometry_pair_count"], 1)
                    self.assertEqual(report["all_pairs"], [])
                    self.assertIsNone(report["total_solid_volume_m3"])

    def test_overlapping_cube_shells_require_explicit_material_items(self):
        # The overlap is 0.8 * 0.7 * 0.6, so adding two closed shells would overcount.
        triangles = [(1,3,2),(1,4,3),(5,6,7),(5,7,8),(1,2,6),(1,6,5),
                     (2,3,7),(2,7,6),(3,4,8),(3,8,7),(4,1,5),(4,5,8)]
        for kind in ("triangulated", "polygonal", "faceted"):
            with self.subTest(kind=kind):
                s = PartialScene()
                points = [(x+ox, y+oy, z+oz) for ox,oy,oz in [(0,0,0),(.2,.3,.4)]
                          for z in (0,1) for x,y in ((0,0),(1,0),(1,1),(0,1))]
                faces = [tuple(i+offset for i in t) for offset in (0,8) for t in triangles]
                if kind == "faceted":
                    point_ids = [s.add(f"IFCCARTESIANPOINT({pt})") for pt in points]
                    ff = []
                    for t in faces:
                        refs = ",".join(f"#{point_ids[i-1]}" for i in t)
                        loop = s.add(f"IFCPOLYLOOP(({refs}))")
                        outer = s.add(f"IFCFACEOUTERBOUND(#{loop},.T.)")
                        ff.append(s.add(f"IFCFACE((#{outer}))"))
                    refs = ",".join(f"#{f}" for f in ff)
                    shell = s.add(f"IFCCLOSEDSHELL(({refs}))")
                    item = s.add(f"IFCFACETEDBREP(#{shell})")
                else:
                    coordinates = ",".join(map(str,points))
                    pts = s.add(f"IFCCARTESIANPOINTLIST3D(({coordinates}))")
                    if kind == "triangulated":
                        face_list = ",".join(map(str,faces))
                        item = s.add(f"IFCTRIANGULATEDFACESET(#{pts},$,.T.,({face_list}),$)")
                    else:
                        ff = [s.add(f"IFCINDEXEDPOLYGONALFACE({t})") for t in faces]
                        refs = ",".join(f"#{f}" for f in ff)
                        item = s.add(f"IFCPOLYGONALFACESET(#{pts},.T.,({refs}),$)")
                invalid = s.product("IFCWALL", "single source", [item])
                valid = s.product("IFCBEAM", "separate Body items", [s.box(1,.5,z=0,y=.5),
                                                                    s.box(1,.7,z=.4,y=.8)])
                report = s.partial()
                self.assertEqual(report["unresolved_geometry_step_ids"], [invalid])
                self.assertIn("Multiple boundary components", rows(report)[invalid]["reason"])
                self.assertAlmostEqual(rows(report)[valid]["volume_m3"], 1.664, places=10)
                self.assertIsNone(report["total_solid_volume_m3"])
                self.assertEqual((report["missing_geometry_pair_count"], report["all_pairs"]), (1,[]))

    def test_project_conversion_dimensions(self):
        for dims in ("1,0,0,0,0,0,0","0,0,1,0,0,0,0",None):
            with self.subTest(dims=dims):
                s = PartialScene()
                dim = s.add(f"IFCDIMENSIONALEXPONENTS({dims})") if dims else 999999
                measure = s.add("IFCMEASUREWITHUNIT(IFCLENGTHMEASURE(0.3048),#1)")
                unit = s.add(f"IFCCONVERSIONBASEDUNIT(#{dim},.LENGTHUNIT.,'foot',#{measure})")
                s.records[1] = f"#2=IFCUNITASSIGNMENT((#{unit}));"
                s.product("IFCWALL","wall",[s.box(1,0)])
                r = s.partial()
                if dims == "1,0,0,0,0,0,0":
                    self.assertAlmostEqual(r["total_solid_volume_m3"],.3048**3)
                else:
                    self.assertEqual((r["status"],r["total_solid_volume_m3"]),("unresolved_model",None))

    def test_deep_placement_and_quoted_delimiters(self):
        s = PartialScene()
        pt = s.add("IFCCARTESIANPOINT((1.,0.,0.))")
        frame = s.add(f"IFCAXIS2PLACEMENT3D(#{pt},$,$)")
        s.product("IFCDUCTSEGMENT","probe",[s.box(1,40)])
        for _ in range(40):
            s.place = s.add(f"IFCLOCALPLACEMENT(#{s.place},#{frame})")
        guid = "aa'' /* ; ENDSEC; DATA; #99999=IFCWALL($); */"
        deep = s.product("IFCBEAM",guid,[s.box(1,0)])
        r = s.partial()
        self.assertTrue(r["ok"],r)
        self.assertEqual(rows(r)[deep]["guid"],guid.replace("''","'"))
        self.assertEqual(r["hard_clash_count"],1)

    def test_registered_input_preserves_partial_status(self):
        from test_wired_router_e2e import _start_mock_relay
        s = PartialScene()
        s.product("IFCWALL","known",[s.box(1,0)])
        s.product("IFCDUCTSEGMENT","missing")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work/"partial.ifc").write_text(s.text())
            ops = []
            srv = _start_mock_relay(work, ops)
            root = Path(__file__).resolve().parents[1]
            runner = """
              import register from './task-tools.ts';
              const tools=new Map(); register({registerTool(t){tools.set(t.name,t)},on(){}});
              const t=tools.get('hamon_ifc_spatial_clash');
              if(t.parameters.properties.allow_partial.type!=='boolean') throw Error('missing selector');
              console.log(JSON.stringify((await t.execute('test',{path:'partial.ifc',allow_partial:true})).details));
            """
            try:
                env = {**os.environ,"TASK_RELAY_URL":f"http://127.0.0.1:{srv.server_port}",
                       "TASK_WORKDIR":str(work),"TASK_HARNESS_SRC":str(root),"TASK_HARNESS_PYTHON":sys.executable}
                proc = subprocess.run(["node","--experimental-strip-types","--input-type=module","-e",runner],
                                      cwd=root, env=env, capture_output=True, text=True, timeout=60, check=True)
            finally:
                srv.shutdown()
                srv.server_close()
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(ops,["read_file"])
        self.assertEqual((r["ok"],r["status"],r["missing_geometry_pair_count"]),(False,"partial",1))
        self.assertIsNone(r["total_solid_volume_m3"])


if __name__ == "__main__":
    unittest.main()
