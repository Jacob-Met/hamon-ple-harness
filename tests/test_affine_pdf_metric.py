"""Affine-PDF physical metric falsification controls.

A 2-axis isotropic scale can fit the horizontal and vertical dimensions
perfectly yet be WRONG for diagonal dimensions under shear. Four independently
sourced directions and leave-one-out metric testing expose that ceiling.
"""
import copy
import math
import unittest

import engineering_toolkit as unit
import hamon_affine_metric
import hamon_scale


def ep(i, n):
    return {"vector":f"v{i}","vertex":n}


def reference(i, length, label):
    return {"from":ep(i,0),"to":ep(i,1),
            "length":length,"source_label":label}


def fixture():
    spans=[(100,0),(0,100),(100,100),(100,-100),(80,40),(80,60)]
    page={"page":7,"vector_limit_hit":False,"unlocalized_vectors":0,
          "words":[{"text":label} for label in
                   ("H-1000","V-1000","POS-DIAG","NEG-DIAG","OBLIQUE")],
          "reported_vectors":[{"id":f"v{i+1}",
                               "svg_points":[[0,0],[x,y]]}
                              for i,(x,y) in enumerate(spans)]}
    references=[
        reference(1,"1*m","H-1000"),
        reference(2,"1*m","V-1000"),
        reference(3,"sqrt(2.8)*m","POS-DIAG"),
        reference(4,"sqrt(1.2)*m","NEG-DIAG"),
    ]
    spec={"schema":"hamon.pdf.measure.v1",
          "page":7,"calibration_model":"affine_spd",
          "max_relative_residual":0.01,
          "references":references,
          "targets":[{"from":ep(5,0),"to":ep(5,1),
                      "name":"true-oblique","unit":"mm"}]}
    return page,spec


def check(page,spec):
    return hamon_scale.calibrate_inventory(
        page,spec,unit.quantity,unit.UNITS["m"])


class AffineMetricContracts(unittest.TestCase):
    def test_shear_is_undetectable_from_two_orthogonal_scale_anchors(self):
        page,spec=fixture()
        isotropic=copy.deepcopy(spec)
        isotropic["calibration_model"]="similarity"
        isotropic["references"]=isotropic["references"][:2]
        wrong=check(page,isotropic)
        target=math.sqrt(1.056)*1000
        self.assertAlmostEqual(wrong["targets"][0]["estimate"],
                               math.sqrt(8000)*10,places=8)
        self.assertGreater(abs(wrong["targets"][0]["estimate"]-target)/target,
                           .12)
        self.assertTrue(wrong["assumed_isotropic"])
        self.assertIsNone(wrong["reference_max_heldout_relative_residual"])

    def test_four_cold_directions_resolve_shear_and_correct_heldout_target(self):
        page,spec=fixture()
        result=check(page,spec)
        self.assertFalse(result["assumed_isotropic"])
        self.assertFalse(result["physical_scale_verified"])
        self.assertEqual(result["calibration_method"],"affine_metric_heldout")
        tensor=result["affine_metric"]["metric_tensor_m2_per_svg_unit2"]
        self.assertAlmostEqual(tensor[0][0],1e-4,delta=1e-12)
        self.assertAlmostEqual(tensor[0][1],4e-5,delta=1e-12)
        self.assertAlmostEqual(tensor[1][1],1e-4,delta=1e-12)
        self.assertAlmostEqual(result["targets"][0]["estimate"],
                               math.sqrt(1.056)*1000,delta=1e-7)
        self.assertLess(result["reference_max_heldout_relative_residual"],1e-9)

    def test_five_independent_directions_falsify_heldout_stability(self):
        page,spec=fixture()
        spec["references"].append(
            reference(6,"sqrt(1.384)*m","OBLIQUE"))
        result=check(page,spec)
        self.assertEqual(result["reference_count"],5)
        self.assertEqual(len(result["affine_metric"]["heldout_relative_residuals"]),5)
        self.assertLess(result["reference_max_heldout_relative_residual"],1e-9)

    def test_fourth_label_corruption_fails_even_if_fit_has_three_parameters(self):
        page,spec=fixture()
        spec["references"][3]["length"]="1.4*m"
        with self.assertRaisesRegex(ValueError, "inconsistent affine dimension"):
            check(page,spec)

    def test_no_three_anchor_exact_interpolation_ever_qualifies(self):
        page,spec=fixture()
        spec["references"]=spec["references"][:3]
        with self.assertRaisesRegex(ValueError,"4..64"):
            check(page,spec)

    def test_four_citations_cannot_all_be_in_single_direction_family(self):
        page,spec=fixture()
        page["reported_vectors"][3]["svg_points"]=[[0,0],[100,0]]
        with self.assertRaisesRegex(ValueError,"distinct reference orientations"):
            check(page,spec)

    def test_near_singular_affine_design_rejected_by_heldout_condition(self):
        page,spec=fixture()
        # Two angles separated by <5 degrees: not independent enough
        page["reported_vectors"][3]["svg_points"]=[[0,0],[100,5]]
        with self.assertRaisesRegex(ValueError,
                                    "distinct reference orientations|ill-conditioned"):
            check(page,spec)

    def test_incompatible_dimension_labels_fail_before_fit(self):
        page,spec=fixture()
        spec["references"][2]["length"]="sqrt(2.8)*kN"
        with self.assertRaisesRegex(ValueError,"physical length"):
            check(page,spec)
        page,spec=fixture()
        spec["references"][2]["source_label"]="MISSING-DIAGONAL"
        with self.assertRaisesRegex(ValueError,"not found"):
            check(page,spec)

    def test_nonpositive_metric_is_never_emitted(self):
        page,spec=fixture()
        # A 3m diagonal cannot coexist with 1m horizontal and vertical
        # distances under any positive-definite 2D metric.
        spec["references"][2]["length"]="3*m"
        with self.assertRaises(ValueError):
            check(page,spec)

    def test_global_svg_rescaling_preserves_physical_length(self):
        page,spec=fixture()
        original=check(page,spec)["targets"][0]["estimate"]
        for vector in page["reported_vectors"]:
            vector["svg_points"]=[[x*3,y*3] for x,y in vector["svg_points"]]
        changed=check(page,spec)
        self.assertAlmostEqual(changed["targets"][0]["estimate"],
                               original,delta=1e-7)
        self.assertAlmostEqual(
            changed["affine_metric"]["metric_tensor_m2_per_svg_unit2"][0][1],
            4e-5/9,delta=1e-12)

    def test_invalid_affine_model_or_tolerance_rejected(self):
        page,spec=fixture()
        spec["calibration_model"]="freehand"
        with self.assertRaisesRegex(ValueError,"calibration_model"):
            check(page,spec)
        spec["calibration_model"]="affine_spd"
        spec["max_relative_residual"]=0.3
        with self.assertRaisesRegex(ValueError,"max_relative_residual"):
            check(page,spec)

    def test_robust_diagonal_holdout_gate_at_tight_tolerance(self):
        page,spec=fixture()
        spec["max_relative_residual"]=0.001
        self.assertLess(check(page,spec)["reference_max_heldout_relative_residual"],
                        .001)
        spec["references"][2]["length"]="sqrt(2.9)*m"
        with self.assertRaisesRegex(ValueError,"inconsistent affine dimension"):
            check(page,spec)


if __name__=="__main__":
    unittest.main()
