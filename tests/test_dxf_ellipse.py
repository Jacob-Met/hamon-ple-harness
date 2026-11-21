import math
from pathlib import Path
import tempfile
import unittest

from hamon_cad_takeoff import scan_dxf_takeoff


class EllipseTakeoffTests(unittest.TestCase):
    def test_eccentric_full_loop_and_wrapped_arc(self):
        # Reference lengths from the incomplete elliptic integral of the second kind.
        for start, end, key, expected in (
            (0, math.tau, "polyline_perimeter_m_by_layer", 40.00015588104688),
            (5.9, 0.12, "line_length_m_by_layer", 0.7971946147629847),
        ):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as directory:
                pairs = [
                    (0, "SECTION"), (2, "HEADER"), (9, "$INSUNITS"), (70, 6), (0, "ENDSEC"),
                    (0, "SECTION"), (2, "ENTITIES"), (0, "ELLIPSE"), (8, "survey"),
                    (10, 0), (20, 0), (11, 10), (21, 0), (40, 0.001),
                    (41, start), (42, end), (0, "ENDSEC"), (0, "EOF"),
                ]
                path = Path(directory) / "survey.dxf"
                path.write_text("".join(f"{code}\n{value}\n" for code, value in pairs))
                self.assertAlmostEqual(scan_dxf_takeoff(path)[key]["survey"], expected, places=10)
