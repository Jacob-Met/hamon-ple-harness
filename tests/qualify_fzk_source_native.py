"""Rebuild the FZK-Haus native BReps and measure gross/net material and window intersections."""
import hashlib
import json
import math
import sys
from pathlib import Path
import ifcopenshell as ifc
import ifcopenshell.geom as geom
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCP.BRepLib import BRepLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

def volume(shape, closed=True):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, closed, False, False)
    return props.Mass()

def explore(shape, kind):
    e = TopExp_Explorer(shape, kind)
    while e.More():
        yield e.Current()
        e.Next()

def read_closed(path):
    raw = TopoDS_Shape()
    assert BRepTools.Read_s(raw, str(path), BRep_Builder())
    solids = list(explore(raw, TopAbs_SOLID))
    if len(solids) == 1:
        solid = solids[0]
    else:
        assert not solids
        sewing = BRepBuilderAPI_Sewing(1e-7)
        sewing.Add(raw)
        sewing.Perform()
        shells = list(explore(sewing.SewedShape(), TopAbs_SHELL))
        assert shells and sewing.NbFreeEdges() == sewing.NbMultipleEdges() == 0
        builder = BRepBuilderAPI_MakeSolid()
        for result in shells:
            sh = TopoDS.Shell_s(result)
            assert BRep_Tool.IsClosed_s(sh)
            builder.Add(sh)
        solid = builder.Solid()
        assert BRepLib.OrientClosedSolid_s(solid)
    assert BRepCheck_Analyzer(solid).IsValid() and volume(solid) > 0
    assert math.isclose(volume(solid), volume(raw, False), rel_tol=1e-8, abs_tol=1e-12)
    return solid

def shapes(model, step, disable, out):
    settings = geom.settings()
    settings.set("iterator-output", ifc.ifcopenshell_wrapper.NATIVE)
    settings.set("use-world-coords", True)
    settings.set("disable-opening-subtractions", disable)
    product = geom.create_shape(settings, model[step])
    geometry = product.geometry
    for i in range(geometry.size()):
        path = out / f"{step}-{int(disable)}-{i}.brep"
        path.write_text(geometry.item(i).serialize())
        yield read_closed(path)

def relation(a, b):
    op = BRepAlgoAPI_Common(a,b)
    d = BRepExtrema_DistShapeShape(a,b)
    assert op.IsDone() and d.IsDone()
    return {"common_m3": volume(op.Shape()), "surface_distance_m": d.Value()}

def main(source, out):
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "ea6f04eaf92fac4d7ad0038bc3d2dfea4c094dd3f516ecc33c50bf1835ca108d"
    out.mkdir(parents=True, exist_ok=True)
    model = ifc.open(str(source))
    results = {}
    for h,o,w in [(60012,60579,66459),(67828,68400,74280)]:
        assert model[o].VoidsElements[0].RelatingBuildingElement.id() == h
        assert model[o].HasFillings[0].RelatedBuildingElement.id() == w
        gross, = shapes(model,h,True,out)
        net, = shapes(model,h,False,out)
        components = list(shapes(model,w,False,out))
        assert len(components) == 11
        results[h] = {"gross_m3":volume(gross), "net_m3":volume(net), "window_parts":[
            {"volume_m3":volume(s), "gross":relation(gross,s), "net":relation(net,s)} for s in components]}
    for h in (67536,75347):
        solid, = shapes(model,h,False,out)
        results[h] = {"net_m3":volume(solid)}
    (out/"native.json").write_text(json.dumps(results,indent=2))

if __name__ == "__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]))
