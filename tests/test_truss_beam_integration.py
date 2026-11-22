"""Check the incoming beam recovery on the truss receiver's unchanged cantilever."""
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TrussBeamIntegrationTests(unittest.TestCase):
    def test_unit_tip_load_cantilever(self):
        spec = {
            "mode": "euler_bernoulli_beam_1d",
            "x_nodes_m": [0, 1],
            "EI_Nm2": 1,
            "supports": [{"node": 0, "fix_w": True, "fix_theta": True}],
            "nodal_loads": [{"node": 1, "fy_N": 1}],
        }
        record = {"case": "beam/cantilever", "full_sha256": None}
        try:
            result = run_router_conservation_cad_verifier({"mode": "fem", "spec": spec})
        except Exception as exc:
            record["solver_error"] = [type(exc).__name__, str(exc)]
            print("TRUSS_BEAM_INTEGRATION_CASE " + canonical(record))
            raise
        try:
            encoded = canonical(result).encode("utf-8")
        except (ValueError, TypeError, OverflowError) as exc:
            record["json_error"] = [type(exc).__name__, str(exc)]
            print("TRUSS_BEAM_INTEGRATION_CASE " + canonical(record))
            raise
        record["full_sha256"] = hashlib.sha256(encoded).hexdigest()
        record["observed"] = result
        print("TRUSS_BEAM_INTEGRATION_CASE " + canonical(record))

        self.assertIs(result["passed"], True)
        self.assertEqual((result["domain"], result["mode"]), ("fem", spec["mode"]))
        # For F=EI=L=1, w(x)=x²(3-x)/6 and theta(x)=x(2-x)/2.
        # The end load stores U=F*w(1)/2. The clamped support does no work.
        scalar = {
            "tip_deflection_m": 1/3,
            "strain_energy_J": 1/6,
            "external_work_J": 1/6,
            "support_work_J": 0,
        }
        vector = {
            "deflections_m": [0, 1/3],
            "rotations_rad": [0, 1/2],
            "element_midspan_deflections_m": [5/48],
        }

        def check_number(actual, expected, atol):
            self.assertIn(type(actual), (float, int))
            self.assertTrue(math.isfinite(actual))
            self.assertAlmostEqual(actual, expected, delta=atol)

        for field, expected in scalar.items():
            with self.subTest(field=field):
                check_number(result[field], expected, 16*math.ulp(expected) if expected else 0)
        for field, expected in vector.items():
            with self.subTest(field=field):
                actual = result[field]
                self.assertIsInstance(actual, list)
                self.assertEqual(len(actual), len(expected))
                for component, target in zip(actual, expected):
                    check_number(component, target, 16*math.ulp(target) if target else 0)
        reactions = result["reactions_V_M"]
        self.assertIsInstance(reactions, list)
        self.assertEqual(len(reactions), 2)
        # Reactions from recovered equilibrium use the applied unit load/moment scale.
        for reaction, expected in zip(reactions, ((-1, -1), (0, 0))):
            self.assertIsInstance(reaction, list)
            self.assertEqual(len(reaction), 2)
            for component, target in zip(reaction, expected):
                check_number(component, target, 16*math.ulp(1.0))
