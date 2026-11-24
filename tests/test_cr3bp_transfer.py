"""Numerical CR3BP transfer certificates against independent public orbit data."""
import math
import unittest

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


# NASA/JPL Three-Body Periodic Orbits API documentation, both example queries:
# https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html
# Rotating barycentric [x, y, z, vx, vy, vz], normalized mass fraction and time.
EARTH_MOON = (
    0.01215058560962404, 3.1233143922761588, 0.19516273085815472,
    [-0.41456184803140111, 0., 0.90753120433295065, -1.1555581216266303e-12,
     1.4076145460136695, 3.9979936124406781e-13],
)
SATURN_TITAN = (
    0.0002366393158331484, 3.6919029518069890, 3.0091858300914192,
    [0.97753007347844856, 0., 0.039522030056289147, 4.9658541968180441e-14,
     -0.011884747036685137, 7.2136225765117e-14],
)


def orbit(row):
    mu, time, jacobi, state = row
    return {"mass_ratio": mu, "flight_time_tu": time, "jacobi": jacobi,
            "initial_state": state, "target_state": list(state)}


class ThreeBodyTransfers(unittest.TestCase):
    def test_registered_periodic_endpoints_in_two_systems(self):
        for row in (EARTH_MOON, SATURN_TITAN):
            with self.subTest(mass_ratio=row[0]):
                result = dispatch_wave7_verifier("CR3BP", orbit(row))
                self.assertTrue(result["verified"], result)
                self.assertLess(result["frame_residual"], 1e-8)
                self.assertLess(result["declared_jacobi_residual"], 1e-11)
                self.assertEqual(len(result["endpoint"]), 6)

    def test_same_jacobi_wrong_velocity_and_wrong_flight_time(self):
        params = orbit(EARTH_MOON)
        params["target_state"][4] *= -1
        bad = dispatch_wave7_verifier("CR3BP", params)
        self.assertFalse(bad["verified"])
        self.assertLess(bad["jacobi_drift"], 1e-8)
        self.assertGreater(bad["velocity_residual"], 2.0)
        params = orbit(SATURN_TITAN)
        params["flight_time_tu"] /= 2
        self.assertFalse(dispatch_wave7_verifier("CR3BP", params)["verified"])

    def test_public_libration_points_in_rotating_frame(self):
        mu = EARTH_MOON[0]
        for point, x, y in [
            ("L1", 0.8369151257723572, 0.), ("L2", 1.155682165444884, 0.),
            ("L3", -1.005062645810278, 0.), ("L4", 0.5-mu, math.sqrt(3)/2),
            ("L5", 0.5-mu, -math.sqrt(3)/2),
        ]:
            params = {"mass_ratio": mu, "flight_time_tu": 0.3,
                      "initial_state": [x, y, 0, 0, 0, 0], "target_lagrange": point}
            with self.subTest(point=point):
                result = dispatch_wave7_verifier({"domain": "lagrangian_orbital", "params": params})
                self.assertTrue(result["verified"], result)
                self.assertAlmostEqual(result["target"][0], x, places=13)

    def test_jacobi_and_distinct_model_obligations_are_bound(self):
        params = orbit(SATURN_TITAN)
        params["jacobi"] += 0.01
        self.assertFalse(dispatch_wave7_verifier("CR3BP", params)["verified"])
        for extra in ({"burn_m_s": 3}, {"target_lagrange": "L1"}, {"subkind": "orbital_transfer"}):
            with self.subTest(extra=extra):
                self.assertFalse(dispatch_wave7_verifier("CR3BP", {**orbit(EARTH_MOON), **extra})["verified"])
        conflicting = {"mu": 3.986004418e14, "r1": 6.678e6, "r2": 4.2164e7,
                       "subkind": "orbital_transfer", "mass_ratio": EARTH_MOON[0]}
        self.assertFalse(dispatch_wave7_verifier("lagrangian_orbital", conflicting)["verified"])

    def test_collision_and_invalid_input_refuse_without_certificate(self):
        p = orbit(EARTH_MOON)
        for key, value in (("initial_state", [-p["mass_ratio"], 0, 0, 0, 0, 0]),
                           ("mass_ratio", True), ("target_state", [0]*5),
                           ("flight_time_tu", float("nan")), ("flight_time_tu", 0)):
            with self.subTest(key=key, value=value):
                result = dispatch_wave7_verifier("CR3BP", {**p, key: value})
                self.assertFalse(result["verified"])
                self.assertIn("error", result)

    def test_three_body_obligations_cannot_disappear_during_dispatch(self):
        p = orbit(EARTH_MOON)
        for extra in ({"burn_m_s": 3}, {"jacobi": 4}, {"target_state": [0]*6},
                      {"kind": "orbital_transfer"}, {"subkind":"euler_rigid_body"}):
            envelope = {"domain":"lagrangian_orbital", "params":p, **extra}
            self.assertFalse(dispatch_wave7_verifier(envelope)["verified"], extra)
        for p in ({"I":[2,3,5], "omega0":[1,0,0], "subkind":"euler_rigid_body"},
                  {"K":[[1]], "M":[[1]], "subkind":"lagrangian_modes"}):
            for extra in ({"jacobi":3}, {"mass_ratio":0.01, "target_lagrange":"L4"}):
                self.assertFalse(dispatch_wave7_verifier("lagrangian_orbital", {**p, **extra})["verified"])

    def test_free_text_three_body_request_cannot_certify_hohmann(self):
        two_body = {"mu": 3.986004418e14, "r1": 6.678e6, "r2": 4.2164e7}
        prompt = "verify a CR3BP transfer"
        self.assertFalse(dispatch_wave7_verifier(prompt, two_body)["verified"])
        self.assertFalse(dispatch_wave7_verifier({"prompt": prompt, "params": two_body})["verified"])
