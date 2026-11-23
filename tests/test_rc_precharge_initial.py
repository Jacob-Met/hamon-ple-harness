"""Compare the registered RC solver with direct numerical integration of V, work and heat."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def curve(bus, initial, target, horizon, step):
    """RK4 on dV/dt=I/C, dW/dt=Vbus*I, dH/dt=R*I² for the test fixture."""
    resistance, capacitance = 10., 100e-6

    def derivative(state):
        current = (bus - state[0]) / resistance
        return current / capacitance, bus * current, resistance * current ** 2

    def advance(state, dt):
        def shift(slopes, factor):
            return tuple(value + dt * factor * slope for value, slope in zip(state, slopes))
        k1 = derivative(state)
        k2 = derivative(shift(k1, .5))
        k3 = derivative(shift(k2, .5))
        k4 = derivative(shift(k3, 1.))
        return tuple(v + dt * (a + 2*b + 2*c + d) / 6 for v, a, b, c, d in zip(state, k1, k2, k3, k4))

    time, state = 0., (initial, 0., 0.)
    if target == initial:
        return time, state
    for _ in range(round(horizon / step)):
        future = advance(state, step)
        if (state[0] < target <= future[0]) or (state[0] > target >= future[0]):
            lo, hi = 0., step
            for _ in range(45):
                mid = (lo + hi) / 2
                if (advance(state, mid)[0] - target) * (initial - target) > 0:
                    lo = mid
                else:
                    hi = mid
            return time + hi, advance(state, hi)
        state, time = future, time + step
    return None, state


class RcInitialTrajectory(unittest.TestCase):
    def solve(self, spec):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.json"
            path.write_text(json.dumps({"mode": "domain", "kind": "rc_precharge", "spec": spec}))
            process = subprocess.run([sys.executable, str(ROOT / "engineering_toolkit.py"), "dual-verify", str(path)],
                                     capture_output=True, text=True, check=True)
            return json.loads(process.stdout)

    def test_first_crossing_and_signed_work_via_registered_cli(self):
        cases = [
            (12, 9, 6, "unreachable"), (12, 9, 13, "unreachable"), (8, 8, 4, "unreachable"),
            (12, 9, 10, "finite"), (12, 18, 15, "finite"), (12, -4, 0, "finite"),
            (-12, 4, -6, "finite"), (0, 12, 6, "finite"), (-12, -3, -3, "already_at_target"),
            (12, 12, 12, "already_at_target"), (0, 12, 0, "asymptotic"), (12, -3, 12, "asymptotic"),
            (12, 0, 6, "finite")]
        for bus, initial, target, status in cases:
            with self.subTest(bus=bus, initial=initial, target=target):
                spec = dict(V_bus_V=bus, R_precharge_ohm=10, C_uF=100, V_target_V=target)
                # Preserve the older fraction-only call and the original reported failure.
                if (bus, initial, target) in [(12, 0, 6), (12, 9, 6)]:
                    del spec["V_target_V"]
                    spec["target_fraction"] = .5
                if initial != 0:
                    spec["V_initial_V"] = initial
                result = self.solve(spec)
                t, (voltage, work, heat) = curve(bus, initial, target, horizon=.025, step=1e-5)
                self.assertEqual(result["target_time_status"], status)
                self.assertEqual(result["requested_capacitor_voltage_V"], target)
                if status == "unreachable":
                    self.assertIsNone(t)
                    self.assertFalse(result["dual_verified"])
                    for name in ("t_target_s", "target_capacitor_energy_J", "source_work_J", "resistor_heat_J"):
                        self.assertIsNone(result[name])
                    continue
                self.assertTrue(result["dual_verified"])
                if status == "asymptotic":
                    self.assertIsNone(t)
                    self.assertIsNone(result["t_target_s"])
                else:
                    self.assertAlmostEqual(result["t_target_s"], t, delta=2e-12)
                self.assertAlmostEqual(voltage, target, delta=4e-10)
                self.assertAlmostEqual(result["source_work_J"], work, delta=2e-11)
                self.assertAlmostEqual(result["resistor_heat_J"], heat, delta=2e-11)
                self.assertGreaterEqual(result["resistor_heat_J"], 0)
                self.assertAlmostEqual(result["capacitor_energy_change_J"], work - heat, delta=3e-11)

    def test_invalid_and_adjacent_inputs(self):
        from hamon_rc_precharge import rc_precharge_state
        spec = dict(V_bus_V=12., V_initial_V=9., R_precharge_ohm=10., C_uF=100.)
        for change in ({}, {"V_target_V": 10., "target_fraction": .5}, {"V_target_V": float("nan")},
                       {"V_target_V": 10., "V_initial_V": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                rc_precharge_state({**spec, **change})
        for target in (math.nextafter(9., 12.), math.nextafter(12., 9.)):
            result = rc_precharge_state({**spec, "V_target_V": target})
            self.assertEqual(result["target_time_status"], "finite")
            self.assertTrue(result["dual_verified"])
