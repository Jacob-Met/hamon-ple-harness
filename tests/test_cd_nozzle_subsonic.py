"""NASA Glenn CDV 0.89 axial profiles (ideal-flow analytic reference tables)."""
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

from hamon_wave6_physics_verifiers import solve_compressible_gas_acoustics

# https://www.grc.nasa.gov/www/wind/valid/cdv/axial.Mex.p89.gen
NASA_MACH = (
    0.2305986, 0.2311824, 0.2329429, 0.2359077, 0.2401235, 0.2456567, 0.2525952,
    0.2610493, 0.2711545, 0.2830723, 0.2969925, 0.3131352, 0.3317496, 0.3531143,
    0.3775311, 0.4053157, 0.4367796, 0.4721986, 0.5117609, 0.5554809, 0.6030496,
    0.6535671, 0.7050224, 0.7532675, 0.7904066, 0.8049845, 0.7999743, 0.7858996,
    0.7650277, 0.7398368, 0.7123653, 0.6840839, 0.6559903, 0.6287439, 0.6027672,
    0.5783231, 0.5555651, 0.5345717, 0.5153693, 0.4979494, 0.4822819, 0.4683219,
    0.4560181, 0.4453161, 0.4361631, 0.4285094, 0.4223110, 0.4175303, 0.4141374,
    0.4121103, 0.4114360,
)
# https://www.grc.nasa.gov/www/wind/valid/cdv/axial.Pex.p89.gen
NASA_PRESSURE = (
    0.9636506, 0.9634708, 0.9629258, 0.9619994, 0.9606641, 0.9588791, 0.9565899,
    0.9537245, 0.9501908, 0.9458735, 0.9406303, 0.9342855, 0.9266265, 0.9173993,
    0.9063088, 0.8930188, 0.8771661, 0.8583872, 0.8363656, 0.8109176, 0.7821286,
    0.7505741, 0.7176939, 0.6864491, 0.6622707, 0.6527760, 0.6560383, 0.6652070,
    0.6787986, 0.6951735, 0.7129585, 0.7311446, 0.7490404, 0.7661875, 0.7823024,
    0.7972215, 0.8108676, 0.8232222, 0.8343071, 0.8441703, 0.8528739, 0.8604878,
    0.8670827, 0.8727270, 0.8774845, 0.8814117, 0.8845576, 0.8869622, 0.8886572,
    0.8896652, 0.8900000,
)
BASE = {"subkind": "cd_nozzle_subsonic", "gamma": 1.4,
        "exit_to_throat_area": 1.5, "back_to_total_pressure": 0.89}


def solve(**overrides):
    return solve_compressible_gas_acoustics({**BASE, **overrides})


class UnchokedCDNozzle(unittest.TestCase):
    def test_registered_nasa_axial_profile(self):
        # NASA CDV uses x in inches, from 0 to 10 in 0.2 inch increments.
        areas = []
        for index in range(51):
            x = index / 5
            center, amplitude = (1.75, 0.75) if x < 5 else (1.25, 0.25)
            areas.append(center - amplitude * math.cos((0.2 * x - 1) * math.pi))
        request = {"mode": "compressible_acoustics", "spec": {**BASE, "station_areas_over_throat": areas}}
        cli = Path(__file__).resolve().parents[1] / "hamon_conservation_cad_verifier.py"
        run = subprocess.run([sys.executable, str(cli), "--json-stdin"], check=True,
                             input=json.dumps(request), capture_output=True, text=True)
        result = json.loads(run.stdout)
        self.assertTrue(result["verified"], result)
        rows = result["stations"]
        self.assertEqual(len(rows), len(NASA_MACH))
        mach_error = max(abs(row["mach"]-m) for row, m in zip(rows, NASA_MACH))
        pressure_error = max(abs(row["p_over_total"]-p) for row, p in zip(rows, NASA_PRESSURE))
        self.assertLess(mach_error, 3e-6)
        self.assertLess(pressure_error, 3e-6)
        self.assertEqual(result["throat"], rows[25])
        self.assertEqual(result["exit"], rows[-1])
        self.assertLess(result["residual"], 1e-14)
        print("NASA CDV 0.89, 51 stations:", json.dumps(
            {"max_mach_error": mach_error, "max_pressure_error": pressure_error,
             "throat": result["throat"], "exit": result["exit"]}))

    def test_fixed_throat_with_variable_mass_flow(self):
        flows = []
        for back in (0.9, 0.95, 0.99, math.nextafter(1.0, 0.0)):
            state = solve(back_to_total_pressure=back, station_areas_over_throat=[3, 1, 3, 1.5])
            self.assertTrue(state["verified"], state)
            me = math.sqrt(5 * math.expm1(-2 * math.log(back) / 7))
            expected = 1.5 * me * (1.2 / (1 + 0.2 * me * me))**3
            flows.append(state["virtual_sonic_area_over_throat"])
            self.assertAlmostEqual(flows[-1], expected, delta=2e-15)
            self.assertGreater(state["throat"]["mach"], state["exit"]["mach"])
            self.assertLess(state["throat"]["mach"], 1)
            self.assertEqual(state["stations"][0], state["stations"][2])
        self.assertTrue(all(a > b for a, b in zip(flows, flows[1:])))

    def test_choking_boundary_and_wrong_regimes(self):
        # With gamma=3 and exit Mach=1/2, Ae/A*=5/4 at a sonic throat.
        critical = (1.25)**-1.5
        self.assertTrue(solve(gamma=3, exit_to_throat_area=1.25,
                              back_to_total_pressure=critical + 1e-10)["verified"])
        for state in (solve(gamma=3, exit_to_throat_area=1.25, back_to_total_pressure=critical-1e-10),
                      solve(back_to_total_pressure=0.75), solve(back_to_total_pressure=0.16)):
            self.assertEqual(state["regime"], "no_strictly_subsonic_solution")
            self.assertFalse(state["verified"])
            self.assertNotIn("throat", state)
        for params in ({"station_areas_over_throat": [0.9]}, {"station_areas_over_throat": True},
                       {"station_areas_over_throat": [False]}, {"exit_to_throat_area": 1},
                       {"back_to_total_pressure": 1}, {"throat_area_m2": 5}):
            self.assertFalse(solve(**params)["verified"], params)
