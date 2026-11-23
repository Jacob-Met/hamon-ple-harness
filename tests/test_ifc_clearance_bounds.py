import math
from unittest.mock import patch

import pytest

import hamon_ifc_spatial_clash as clash


def rotated_members(count, second=(3.0, 1.0, 0.0), competing_pair=False):
    """Square prisms whose horizontal sections satisfy |x| + |y| <= 1."""
    records = [
        "#1=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);",
        "#2=IFCUNITASSIGNMENT((#1));",
        "#3=IFCCARTESIANPOINT((0.,0.,0.));",
        "#4=IFCDIRECTION((0.,0.,1.));",
        "#5=IFCDIRECTION((1.,1.,0.));",
        "#6=IFCAXIS2PLACEMENT3D(#3,$,$);",
        f"#7=IFCRECTANGLEPROFILEDEF(.AREA.,'square',$,{math.sqrt(2)},{math.sqrt(2)});",
        "#8=IFCEXTRUDEDAREASOLID(#7,#6,#4,1.);",
        "#9=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#8));",
        "#10=IFCPRODUCTDEFINITIONSHAPE($,$,(#9));",
    ]
    for i in range(count):
        x, y, z = (
            (0.0, 0.0, 0.0)
            if i == 0
            else (second if i == 1 else (100.0 * i, 10.0, 0.0))
        )
        if competing_pair and i in (2, 3):
            x, y, z = 200.0, 0.0, 2.1 if i == 3 else 0.0
        duct = i == 1 or (competing_pair and i == 3)
        a = 20 + 5 * i
        records += [
            f"#{a}=IFCCARTESIANPOINT(({x},{y},{z}));",
            f"#{a + 1}=IFCAXIS2PLACEMENT3D(#{a},#4,#5);",
            f"#{a + 2}=IFCLOCALPLACEMENT($,#{a + 1});",
            f"#{a + 3}={'IFCDUCTSEGMENT' if duct else 'IFCBEAM'}('G{i}',$,'member',$,$,#{a + 2},#10,$,$);",
        ]
    return (
        "ISO-10303-21;\nHEADER;FILE_SCHEMA(('IFC4'));ENDSEC;\nDATA;\n"
        + "\n".join(records)
        + "\nENDSEC;END-ISO-10303-21;"
    )


@pytest.mark.parametrize("count", [2, 25, 26, 38])
def test_added_remote_geometry_preserves_measured_clearance(count):
    with patch.object(
        clash, "evaluate_pair_clash", wraps=clash.evaluate_pair_clash
    ) as refine:
        report = clash.detect_ifc_clashes(rotated_members(count), 0.05, True)
    assert report["min_clearance_m"] == pytest.approx(math.sqrt(2))
    assert report["pair_count"] == count - 1
    first, *far = report["all_pairs"]
    assert [first["element_a"], first["element_b"]] == ["G0", "G1"]
    a, b = first["closest_points_m"]
    assert math.dist(a, b) == pytest.approx(math.sqrt(2))
    assert abs(a[0]) + abs(a[1]) == pytest.approx(1)
    assert abs(b[0] - 3) + abs(b[1] - 1) == pytest.approx(1)
    assert 0 <= a[2] <= 1 and 0 <= b[2] <= 1
    if count > 25:
        assert refine.call_count == 1
        assert all(row["distance_lower_bound_m"] > math.sqrt(2) for row in far)
        assert all(
            row["min_distance_m"] is None
            and row["witness_point_m"] is None
            and row["closest_points_m"] is None
            for row in far
        )


def test_smallest_box_bound_is_not_the_closest_surface_pair():
    report = clash.detect_ifc_clashes(
        rotated_members(26, competing_pair=True), 1.5, True
    )
    assert report["min_clearance_m"] == pytest.approx(1.1)
    assert report["clearance_violation_count"] == 2
    assert report["hard_clash_count"] == 0
    assert all(
        row["min_distance_m"] > 0 and row["closest_points_m"]
        for row in report["clashes"]
    )
    assert math.dist(*report["clashes"][1]["closest_points_m"]) == pytest.approx(1.1)


def test_eligible_pairs_and_intersecting_solids():
    single = clash.detect_ifc_clashes(rotated_members(1), 0.05, True)
    assert single["all_pairs"] == [] and single["min_clearance_m"] is None
    collision = clash.detect_ifc_clashes(
        rotated_members(26, second=(0.4, 0.2, 0.0)), 0.05, True
    )
    assert collision["min_clearance_m"] == 0
    assert collision["hard_clash_count"] == 1
    assert math.dist(*collision["clashes"][0]["closest_points_m"]) == pytest.approx(0)
