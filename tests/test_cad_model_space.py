"""Model-space quantities and counts on a drawing with a shared layout layer."""
from pathlib import Path
import tempfile
import unittest

from hamon_cad_takeoff import scan_dxf_takeoff


class ModelSpaceTakeoff(unittest.TestCase):
    def test_shared_layer_retains_implicit_and_explicit_model_entities(self):
        pairs = [(0, "SECTION"), (2, "HEADER"), (9, "$INSUNITS"), (70, 6),
                 (0, "ENDSEC"), (0, "SECTION"), (2, "ENTITIES")]
        for length, flags in [(2, []), (3, [(67, 0)]), (50, [(67, 1)])]:
            pairs.extend([(0, "LINE"), (8, "shared"), (10, 0), (20, 0),
                          (11, length), (21, 0), *flags])
        for block, space in [("model-block", 0), ("sheet-block", 1)]:
            pairs.extend([(0, "INSERT"), (8, "shared"), (2, block), (67, space)])
        pairs.extend([(0, "CIRCLE"), (8, "shared"), (40, 10), (67, 1),
                      (0, "ENDSEC"), (0, "EOF")])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.dxf"
            path.write_text("".join(f"{code}\n{value}\n" for code, value in pairs))
            report = scan_dxf_takeoff(path)

        self.assertEqual(report["line_length_m_by_layer"], {"shared": 5.0})
        self.assertEqual(report["entity_counts"], {"LINE": 2, "INSERT": 1})
        self.assertEqual(report["closed_area_m2_by_layer"], {})
        self.assertEqual(report["polyline_perimeter_m_by_layer"], {})
        self.assertEqual(report["insert_count_by_layer"], {"shared": 1})
        self.assertEqual(report["insert_count_by_block"], {"model-block": 1})
        self.assertEqual(report["total_insert_blocks"], 1)
