"""Independent measurements of the unchanged public FZK-Haus exchange model.

Use different Python interpreters for extract and measure: IfcOpenShell and OCP
must not load their separately bundled OpenCASCADE libraries in one process.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

SOURCE_SHA256 = "ea6f04eaf92fac4d7ad0038bc3d2dfea4c094dd3f516ecc33c50bf1835ca108d"
TRIPLES = ((60012, 60579, 66459), (67828, 68400, 74280))

def checked_source(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Public source differs from the pinned exchange file")

def extract(source, out):
    import ifcopenshell as ifc
    import ifcopenshell.geom as geom
    checked_source(source)
    out.mkdir(parents=True, exist_ok=True)
    model = ifc.open(str(source))
    physical = [p for p in model.by_type("IfcElement")
                if getattr(p, "Representation", None) and not p.is_a("IfcFeatureElement")]
    if len(physical) != 82:
        raise ValueError(f"Unexpected physical inventory: {len(physical)}")
    manifest = {"source_sha256": SOURCE_SHA256,
                "ifcopenshell_version": importlib.metadata.version("ifcopenshell"),
                "physical_count": len(physical), "products": [], "gross": {}}
    def export(product, gross=False, prefix=""):
        settings = geom.settings()
        settings.set("iterator-output", ifc.ifcopenshell_wrapper.NATIVE)
        settings.set("use-world-coords", True)
        settings.set("disable-opening-subtractions", gross)
        element = geom.create_shape(settings, product)
        geometry = element.geometry
        if geometry.size() < 1:
            raise ValueError("No native shape items")
        rows = []
        for index in range(geometry.size()):
            part = geometry.item(index)
            name = f"{prefix}{product.id()}-{'gross' if gross else 'net'}-{index}.brep"
            serialized = part.serialize()
            if isinstance(serialized, bytes):
                (out / name).write_bytes(serialized)
            else:
                (out / name).write_text(serialized)
            rows.append({"file": name, "native_face_count": part.num_faces()})
        return rows
    for product in sorted(physical, key=lambda p: p.id()):
        row = {"step_id": product.id(), "ifc_type": product.is_a(),
               "guid": product.GlobalId, "name": product.Name}
        try:
            row["parts"] = export(product)
        except (RuntimeError, ValueError, TypeError) as exc:
            row["export_error"] = str(exc)
        manifest["products"].append(row)
    for host, opening, window in TRIPLES:
        assert model[opening].VoidsElements[0].RelatingBuildingElement.id() == host
        assert model[opening].HasFillings[0].RelatedBuildingElement.id() == window
        manifest["gross"][str(host)] = export(model[host], True)
    # A separate importer conversion changes only source context precision.
    # The downloaded exchange bytes and every authored coordinate remain intact.
    precise = ifc.file.from_string(source.read_text())
    for context in precise.by_type("IfcGeometricRepresentationContext", False):
        if context.CoordinateSpaceDimension == 3:
            context.Precision = 1e-8
    manifest["context_precision_variant_m"] = 1e-8
    manifest["precision_variant"] = {
        str(step): export(precise[step], prefix="context-1e-8-")
        for step in (60012, 66459, 67828, 74280)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"stage": "native_export", "source_sha256": SOURCE_SHA256,
                      "physical_count": len(physical),
                      "export_errors": [r["step_id"] for r in manifest["products"] if "export_error" in r]}))

def measure(source, out):
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS, TopoDS_Shape
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.BRepLib import BRepLib
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    checked_source(source)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source_sha256"] == SOURCE_SHA256 and manifest["physical_count"] == 82
    def volume(shape, closed=True):
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props, closed, False, False)
        return props.Mass()
    def explore(shape, kind):
        it = TopExp_Explorer(shape, kind)
        while it.More():
            yield it.Current()
            it.Next()
    def valid_closed(shape):
        solids = list(explore(shape, TopAbs_SOLID))
        assert solids and BRepCheck_Analyzer(shape).IsValid()
        for solid in solids:
            shells = list(explore(solid, TopAbs_SHELL))
            assert shells and all(BRep_Tool.IsClosed_s(TopoDS.Shell_s(s)) for s in shells)
        assert sum(len(list(explore(s, TopAbs_FACE))) for s in solids) == len(list(explore(shape, TopAbs_FACE)))
        result = volume(shape)
        assert math.isfinite(result) and result > 0
        return result
    def read_closed(row):
        raw = TopoDS_Shape()
        assert BRepTools.Read_s(raw, str(out / row["file"]), BRep_Builder())
        raw_faces = list(explore(raw, TopAbs_FACE))
        assert len(raw_faces) == row["native_face_count"], "Native face inventory changed in serialization"
        solids = list(explore(raw, TopAbs_SOLID))
        if len(solids) == 1:
            solid = solids[0]
        else:
            assert not solids, "A native item contains multiple solids"
            sewing = BRepBuilderAPI_Sewing(1e-7)
            sewing.Add(raw)
            sewing.Perform()
            shells = list(explore(sewing.SewedShape(), TopAbs_SHELL))
            assert shells and sewing.NbFreeEdges() == sewing.NbMultipleEdges() == 0
            builder = BRepBuilderAPI_MakeSolid()
            for result in shells:
                shell = TopoDS.Shell_s(result)
                assert BRep_Tool.IsClosed_s(shell)
                builder.Add(shell)
            solid = builder.Solid()
            assert BRepLib.OrientClosedSolid_s(solid)
        assert len(list(explore(solid, TopAbs_FACE))) == len(raw_faces), "Orphan or lost native faces"
        result = valid_closed(solid)
        assert math.isclose(result, abs(volume(raw, False)), rel_tol=1e-8, abs_tol=1e-12)
        return solid
    def union(shapes):
        assert shapes
        current = shapes[0]
        for following in shapes[1:]:
            left, right = valid_closed(current), valid_closed(following)
            op = BRepAlgoAPI_Fuse(current, following)
            assert op.IsDone()
            current = op.Shape()
            result = valid_closed(current)
            error = 1e-10 * max(left + right, 1e-6)
            assert max(left, right) - error <= result <= left + right + error
        return current
    def relation(a, b):
        op = BRepAlgoAPI_Common(a, b)
        distance = BRepExtrema_DistShapeShape(a, b)
        assert op.IsDone() and distance.IsDone()
        common = volume(op.Shape())
        assert math.isfinite(common) and common >= -1e-12
        return {"common_m3": common, "distance_m": distance.Value()}
    report = {"source_sha256": SOURCE_SHA256,
              "ifcopenshell_version": manifest["ifcopenshell_version"],
              "ocp_version": importlib.metadata.version("cadquery-ocp"),
              "platform": platform.platform(), "per_product": {}, "openings": {},
              "scope": "Per-occurrence material unions; their sum is not a whole-building material union"}
    original_parts = {}
    for row in manifest["products"]:
        step = row["step_id"]
        try:
            if "export_error" in row:
                raise ValueError(row["export_error"])
            parts = [read_closed(part) for part in row["parts"]]
            joined = union(parts)
            report["per_product"][str(step)] = {
                "status": "known", "volume_m3": valid_closed(joined), "native_part_count": len(parts)}
            original_parts[step] = parts
        except (AssertionError, RuntimeError, ValueError, TypeError) as exc:
            report["per_product"][str(step)] = {
                "status": "unresolved", "volume_m3": None, "reason": f"{type(exc).__name__}: {exc}"}
    for host, opening, window in TRIPLES:
        gross, = [read_closed(row) for row in manifest["gross"][str(host)]]
        net, = original_parts[host]
        components = original_parts[window]
        assert len(components) == 11
        joined = union(components)
        precise_net, = [read_closed(r) for r in manifest["precision_variant"][str(host)]]
        precise_parts = [read_closed(r) for r in manifest["precision_variant"][str(window)]]
        assert len(precise_parts) == len(components)
        rows = [{"volume_m3": valid_closed(part),
                 "gross": relation(gross, part), "net": relation(net, part),
                 "net_context_precision_1e_8_m": relation(precise_net, precise_parts[index])}
                for index, part in enumerate(components)]
        report["openings"][str(host)] = {
            "opening_step_id": opening, "window_step_id": window,
            "gross_m3": valid_closed(gross), "net_m3": valid_closed(net),
            "window_union_m3": valid_closed(joined),
            "wall_window_union_m3": valid_closed(union([net, joined])),
            "window_parts": rows}
        assert math.isclose(valid_closed(gross), 5.779930866498654, rel_tol=0, abs_tol=1e-9)
        assert math.isclose(valid_closed(net), 5.544406029727043, rel_tol=0, abs_tol=1e-9)
        assert math.isclose(valid_closed(joined), 0.0298949822468136, rel_tol=0, abs_tol=1e-11)
        assert all(abs(r["net"]["common_m3"]) <= 1e-12 for r in rows)
        assert all(abs(r["net_context_precision_1e_8_m"]["common_m3"]) <= 1e-12 for r in rows)
        assert all(math.isclose(r["gross"]["common_m3"], r["volume_m3"],
                                rel_tol=1e-7, abs_tol=1e-12) for r in rows)
    known = {int(k): v for k, v in report["per_product"].items() if v["status"] == "known"}
    unknown = sorted(int(k) for k, v in report["per_product"].items() if v["status"] != "known")
    report["known_count"] = len(known)
    report["unresolved_step_ids"] = unknown
    report["known_occurrence_volume_sum_m3"] = math.fsum(v["volume_m3"] for v in known.values())
    (out / "native-results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"stage": "native_measure", "known_count": len(known),
                      "unresolved_step_ids": unknown, "openings": report["openings"]}))
    assert len(known) == 81 and unknown == [14502], "Unexpected native qualification coverage"
    for step in (67536, 75347):
        assert math.isclose(known[step]["volume_m3"], 2.056376840517137, rel_tol=0, abs_tol=1e-9)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("extract", "measure"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (extract if args.stage == "extract" else measure)(args.source, args.output)
