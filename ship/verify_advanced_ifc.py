"""Cold replay of two unchanged buildingSMART IFC4 examples and refusal controls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.request

import ifcopenshell

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hamon_ifc_spatial_clash import run_router_ifc_clash


REVISION = "ef22440ec501551ccc9689309560bbc7e26aa9d7"
BASE = f"https://raw.githubusercontent.com/buildingSMART/IFC4.3.x-sample-models/{REVISION}/models/advanced-geometric-shape"
# Independent OpenCASCADE sewing (1e-7 m), closed-shell construction and
# BRepGProp integration of the original five and six analytic faces.
CASES = {
    "basin-advanced-brep": ("e947de66cadaefe167e77da75eaaeca864cfcc47f7d1ed197b760b466ae081a9",
                            0.002045022383936515, 0.49333289144135845, 5),
    "cube-advanced-brep": ("c3c5b0e6f4d659cef6efe7725ff6023d22126e3c471601804b7462f166624161",
                           0.8333333333333326, 5.820640611188515, 6),
}


def replay(name, directory):
    expected_hash, volume, area, count = CASES[name]
    path = directory / (name + ".ifc")
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(f"{BASE}/{name}/{name}.ifc", timeout=30) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"{name}: unexpected source bytes")
    source = data.decode("utf-8")
    original = run_router_ifc_clash({"ifc_text": source})
    assert original["ok"] and original["pair_count"] == 0, original
    assert original["source_face_count"] == count, original
    assert math.isclose(original["total_solid_volume_m3"], volume, rel_tol=1e-9), original
    assert math.isclose(original["total_surface_area_m2"], area, rel_tol=1e-9), original

    refusals = {}
    for damage in ("missing_face", "duplicate_face", "second_product"):
        model = ifcopenshell.file.from_string(source)
        solid = model.by_type("IfcAdvancedBrep")[0]
        faces = list(solid.Outer.CfsFaces)
        if damage == "second_product":
            product = next(p for p in model.by_type("IfcProduct") if p.Representation)
            duplicate = model.create_entity(product.is_a(), *tuple(product))
            duplicate.GlobalId = ifcopenshell.guid.new()
        elif damage == "missing_face":
            solid.Outer.CfsFaces = faces[1:]
        else:
            solid.Outer.CfsFaces = faces + faces[:1]
        result = run_router_ifc_clash({"ifc_text": model.to_string()})
        assert not result["ok"] and result["total_solid_volume_m3"] is None, result
        refusals[damage] = result["status"]
    return {"source_sha256": expected_hash, "registered": original, "refusals": refusals}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = {"source_revision": REVISION, "ifcopenshell": ifcopenshell.version,
               "cases": {name: replay(name, args.fixtures) for name in CASES}}
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    elif os.getenv("ARTIFACT_DIR"):
        (Path(os.environ["ARTIFACT_DIR"]) / "advanced-ifc.json").write_text(rendered + "\n")
