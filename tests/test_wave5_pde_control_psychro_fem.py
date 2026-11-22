import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hamon_pde_control_psychro_fem as w5


class TestWave5PDEControlPsychroFEM(unittest.TestCase):
    def test_pde_convective_fin_adiabatic_and_convective_tip(self):
        r_ad = w5.verify_pde_system({
            "mode": "convective_fin_1d",
            "k_w_mk": 200.0,
            "area_m2": 1.5e-4,
            "perimeter_m": 0.05,
            "h_w_m2k": 40.0,
            "length_m": 0.12,
            "t_base_c": 125.0,
            "t_inf_c": 25.0,
            "tip_bc": "adiabatic",
        })
        self.assertTrue(r_ad["passed"])
        m_ex = math.sqrt(40.0 * 0.05 / (200.0 * 1.5e-4))
        q_ex = math.sqrt(40.0 * 0.05 * 200.0 * 1.5e-4) * 100.0 * math.tanh(m_ex * 0.12)
        self.assertAlmostEqual(r_ad["q_fin_w"], q_ex, places=6)
        self.assertLess(r_ad["fvm_rel_error"], 2e-4)

        r_cv = w5.verify_pde_system({
            "mode": "convective_fin_1d",
            "k_w_mk": 180.0,
            "area_m2": 2.0e-4,
            "perimeter_m": 0.06,
            "h_w_m2k": 55.0,
            "h_tip_w_m2k": 55.0,
            "length_m": 0.15,
            "t_base_c": 110.0,
            "t_inf_c": 20.0,
            "tip_bc": "convective",
        })
        self.assertTrue(r_cv["passed"])
        self.assertGreater(r_cv["fin_effectiveness"], 2.0)

    def test_pde_transient_heat_and_2d_laplace(self):
        r_tr = w5.verify_pde_system({
            "mode": "heat_1d_transient",
            "alpha_m2_s": 1.0e-4,
            "length_m": 1.0,
            "t_eval_s": 120.0,
            "x_eval_m": 0.35,
            "modes": [{"n": 1, "amplitude": 80.0}, {"n": 2, "amplitude": -25.0}],
        })
        self.assertTrue(r_tr["passed"])
        self.assertLess(r_tr["linf_error"], 0.15)

        r_lap = w5.verify_pde_system({
            "mode": "laplace_2d_rect",
            "width_m": 1.0,
            "height_m": 1.0,
            "t_top": 100.0,
            "top_profile": "sine",
            "x_eval_m": 0.5,
            "y_eval_m": 0.5,
        })
        self.assertTrue(r_lap["passed"])
        exact_mid = 100.0 * math.sinh(0.5 * math.pi) / math.sinh(math.pi)
        self.assertAlmostEqual(r_lap["t_exact"], exact_mid, places=6)
        self.assertAlmostEqual(r_lap["t_fdm_sor"], exact_mid, delta=0.2)

    def test_rectangular_sine_boundary_layer_across_aspect_and_units(self):
        for height, width, temperature in ((40, 1., 100.), (230, 1., -50.),
                                            (1000, 1e100, 25.), (1000, 1e-100, 25.)):
            with self.subTest(height=height, width=width, temperature=temperature):
                spec = dict(mode="laplace_2d_rect", top_profile="sine", width_m=width,
                            height_m=height*width, t_top=temperature, x_eval_m=0.375*width,
                            y_eval_m=(height-0.05)*width)
                result = w5.verify_pde_system(spec)
                # The reflected lower boundary is below 2e-108 on this panel.
                native_depth = (spec["height_m"] - spec["y_eval_m"]) / width
                expected = temperature * math.sin(0.375*math.pi) * math.exp(-math.pi*native_depth)
                self.assertTrue(result["passed"])
                self.assertAlmostEqual(result["t_exact"], expected, places=12)
                self.assertAlmostEqual(result["t_fdm_sor"], expected, delta=0.0002*abs(temperature))
                self.assertFalse(w5.verify_pde_system({
                    **spec, "claimed_field": "t_exact", "claimed_value": 0.,
                })["passed"])

        for x, y, expected in ((0., 999.9, 0.), (1., 999.9, 0.), (.5, 0., 0.), (.5, 1000., 100.)):
            result = w5.verify_pde_system(dict(mode="laplace_2d_rect", width_m=1., height_m=1000.,
                                              x_eval_m=x, y_eval_m=y, t_top=100.))
            self.assertTrue(result["passed"])
            for field in ("t_exact", "t_fdm_sor"):
                self.assertAlmostEqual(result[field], expected, places=10)

    def test_high_heat_modes_are_verified_between_nodes(self):
        for n, decay, x in ((120, 0., 1/240), (240, 0.3, 1/480), (241, 0., 0.373)):
            with self.subTest(n=n):
                spec = dict(mode="heat_1d_transient", length_m=1., alpha_m2_s=1e-5,
                            t_eval_s=decay / (1e-5 * (n * math.pi) ** 2), x_eval_m=x,
                            modes=[{"n": n, "amplitude": 100.}])
                result = w5.verify_wave5_engineering(spec)
                expected = 100 * math.sin(n * math.pi * x) * math.exp(-decay)
                self.assertTrue(result["passed"])
                self.assertLessEqual(abs(result["u_crank_nicolson"] - expected), result["linf_error"] + 1e-10)
                self.assertLessEqual(result["linf_error"], 0.3)
                self.assertFalse(w5.verify_wave5_engineering({
                    **spec, "claimed_field": "u_crank_nicolson", "claimed_value": 0.,
                })["passed"])

    def test_modal_heat_step_matches_tridiagonal_crank_nicolson(self):
        spec = dict(mode="heat_1d_transient", length_m=1., alpha_m2_s=0.001, t_eval_s=1.,
                    x_eval_m=0.373, modes=[{"n": 2, "amplitude": 80.}, {"n": 7, "amplitude": -25.}])
        result = w5.verify_wave5_engineering(spec)
        cells, steps = result["spatial_cells"], result["time_steps"]
        r = spec["alpha_m2_s"] * spec["t_eval_s"] * cells**2 / (2 * steps)
        temperature = [sum(m["amplitude"] * math.sin(m["n"] * math.pi * i / cells)
                           for m in spec["modes"]) for i in range(cells + 1)]
        temperature[0] = temperature[-1] = 0.
        for _ in range(steps):
            rhs = [r*temperature[i-1] + (1-2*r)*temperature[i] + r*temperature[i+1]
                   for i in range(1, cells)]
            temperature = [0.] + w5._thomas_solve(
                [0.] + [-r]*(cells-2), [1+2*r]*(cells-1), [-r]*(cells-2) + [0.], rhs,
            ) + [0.]
        location = spec["x_eval_m"] * cells
        i = int(location)
        at_query = temperature[i] + (location-i) * (temperature[i+1]-temperature[i])
        self.assertAlmostEqual(result["u_crank_nicolson"], at_query, places=9)
        for x in (0., 1.):
            at_end = w5.verify_wave5_engineering({**spec, "x_eval_m": x})
            self.assertEqual((at_end["u_exact"], at_end["u_crank_nicolson"]), (0., 0.))

    def test_control_routh_hurwitz_and_bode_margins(self):
        r_tf = w5.verify_control_system({
            "mode": "transfer_function",
            "num": [10.0],
            "den": [1.0, 8.0, 17.0, 10.0],
        })
        self.assertTrue(r_tf["passed"])
        self.assertTrue(r_tf["hurwitz_stable"])
        self.assertTrue(r_tf["closed_loop_hurwitz_stable"])
        self.assertAlmostEqual(r_tf["w_pc_rad_s"], math.sqrt(17.0), places=3)
        self.assertAlmostEqual(r_tf["gain_margin_db"], 20.0 * math.log10(12.6), places=2)

        r_unst = w5.verify_control_system({
            "mode": "transfer_function",
            "num": [1.0],
            "den": [1.0, -2.0, 1.0, 2.0],
        })
        self.assertTrue(r_unst["passed"])
        self.assertFalse(r_unst["hurwitz_stable"])
        self.assertEqual(r_unst["rhp_poles_routh"], 2)
        self.assertEqual(r_unst["rhp_poles_eigen"], 2)

    def test_control_state_space_controllability_observability_and_lyapunov(self):
        r_ss = w5.verify_control_system({
            "mode": "state_space",
            "A": [[0.0, 1.0], [-6.0, -5.0]],
            "B": [[0.0], [1.0]],
            "C": [[1.0, 0.0]],
        })
        self.assertTrue(r_ss["passed"])
        self.assertEqual(r_ss["char_poly"], [1.0, 5.0, 6.0])
        self.assertTrue(r_ss["hurwitz_stable"])
        self.assertTrue(r_ss["controllable"])
        self.assertTrue(r_ss["observable"])
        self.assertTrue(r_ss["lyapunov_pd"])
        self.assertLess(r_ss["cayley_hamilton_residual"], 1e-10)

    def test_psychrometric_state_subzero_and_boiling_saturation(self):
        self.assertAlmostEqual(w5._pws_ashrae_pa(20.0), 2338.8, delta=2.0)
        self.assertAlmostEqual(w5._pws_ashrae_pa(100.0), 101325.0, delta=120.0)
        self.assertAlmostEqual(w5._pws_ashrae_pa(-10.0), 259.9, delta=2.0)

        st = w5.verify_psychrometric_process({"mode": "state", "t_db_c": 25.0, "rh": 0.50})
        self.assertTrue(st["passed"])
        st_wb = w5.verify_psychrometric_process({"mode": "state", "t_db_c": 25.0, "t_wb_c": st["t_wb_c"]})
        self.assertTrue(st_wb["passed"])
        self.assertAlmostEqual(st["w_kg_per_kgda"], st_wb["w_kg_per_kgda"], places=5)

    def test_psychrometric_mixing_and_cooling_coil(self):
        mix = w5.verify_psychrometric_process({
            "mode": "adiabatic_mixing",
            "stream1": {"t_db_c": 35.0, "rh": 0.60, "m_da_kg_s": 1.0},
            "stream2": {"t_db_c": 24.0, "rh": 0.50, "m_da_kg_s": 3.0},
        })
        self.assertTrue(mix["passed"])
        self.assertAlmostEqual(mix["m_da_total_kg_s"], 4.0, places=6)

        coil = w5.verify_psychrometric_process({
            "mode": "cooling_coil",
            "m_da_kg_s": 3.0,
            "inlet": {"t_db_c": 32.0, "rh": 0.65},
            "outlet": {"t_db_c": 13.0, "rh": 0.95},
        })
        self.assertTrue(coil["passed"])
        self.assertGreater(coil["condensate_kg_s"], 0.02)
        self.assertGreater(coil["q_total_kw"], coil["q_sensible_kw"])
        self.assertGreater(coil["sensible_heat_ratio"], 0.3)
        self.assertLess(coil["sensible_heat_ratio"], 0.8)

    def test_fem_2d_truss_exact_clapeyron_and_mechanism_rejection(self):
        truss = w5.verify_fem_structure({
            "mode": "truss_2d",
            "nodes": [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]],
            "elements": [
                {"i": 0, "j": 1, "E_pa": 200e9, "A_m2": 1.0e-3},
                {"i": 2, "j": 1, "E_pa": 200e9, "A_m2": 1.0e-3},
            ],
            "supports": [
                {"node": 0, "fix_x": True, "fix_y": True},
                {"node": 2, "fix_x": True, "fix_y": True},
            ],
            "loads": [{"node": 1, "fx_N": 0.0, "fy_N": -30000.0}],
        })
        self.assertTrue(truss["passed"])
        self.assertAlmostEqual(truss["axial_forces_N"][0], -40000.0, places=4)
        self.assertAlmostEqual(truss["axial_forces_N"][1], 50000.0, places=4)
        self.assertAlmostEqual(truss["strain_energy_J"], truss["external_work_J"], places=8)

        mech = w5.verify_fem_structure({
            "mode": "truss_2d",
            "nodes": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]],
            "elements": [
                {"i": 0, "j": 1, "E_pa": 200e9, "A_m2": 1.0e-3},
                {"i": 0, "j": 2, "E_pa": 200e9, "A_m2": 1.0e-3},
            ],
            "supports": [
                {"node": 0, "fix_x": True, "fix_y": True},
                {"node": 2, "fix_x": True, "fix_y": True},
            ],
            "loads": [{"node": 1, "fx_N": 0.0, "fy_N": -10000.0}],
        })
        self.assertFalse(mech["passed"])

    def test_cantilever_stability_is_independent_of_translation_rotation_scale(self):
        # Identical dimensionless deformation on metre and micrometre cantilevers.
        for length, ei in ((1.0, 2e6), (1e-6, 1e-15)):
            force = 0.003 * ei / length**2
            spec = {"mode": "euler_bernoulli_beam_1d", "x_nodes_m": [0.0, length],
                    "EI_Nm2": ei, "supports": [{"node": 0, "fix_w": True, "fix_theta": True}],
                    "nodal_loads": [{"node": 1, "fy_N": force}]}
            result = w5.verify_fem_structure(spec)
            with self.subTest(length=length):
                self.assertTrue(result["passed"])
                self.assertAlmostEqual(result["tip_deflection_m"] / length, 0.001)
                self.assertAlmostEqual(result["rotations_rad"][-1], 0.0015)
                self.assertAlmostEqual(result["reactions_V_M"][0][0] / force, -1)
                self.assertAlmostEqual(result["reactions_V_M"][0][1] / (length * force), -1)
                # The same element pinned at its root still has a rigid rotation.
                spec["supports"][0]["fix_theta"] = False
                self.assertFalse(w5.verify_fem_structure(spec)["passed"])

    def test_independent_axial_stiffness_scales_and_oblique_mechanism(self):
        for k in (1.0, 1e12):
            result = w5.verify_fem_structure({
                "mode": "truss_2d", "nodes": [[0, 0], [1, 0], [0, 1]],
                "elements": [{"i": 0, "j": 1, "E_pa": k, "A_m2": 1},
                             {"i": 0, "j": 2, "E_pa": 1, "A_m2": 1}],
                "supports": [{"node": 0, "fix_x": True, "fix_y": True},
                             {"node": 1, "fix_y": True}, {"node": 2, "fix_x": True}],
                "loads": [{"node": 1, "fx_N": k * 1e-6}, {"node": 2, "fy_N": 1e-6}],
            })
            self.assertTrue(result["passed"])
            self.assertAlmostEqual(result["displacements_xy_m"][1][0], 1e-6, places=17)
            self.assertAlmostEqual(result["displacements_xy_m"][2][1], 1e-6, places=17)
        # Both diagonal stiffnesses are positive; an orthogonal motion still costs zero energy.
        result = w5.verify_fem_structure({
            "mode": "truss_2d", "nodes": [[0, 0], [3, 4]],
            "elements": [{"i": 0, "j": 1, "E_pa": 7e12, "A_m2": 1}],
            "supports": [{"node": 0, "fix_x": True, "fix_y": True}],
            "loads": [{"node": 1, "fx_N": -4, "fy_N": 3}],
        })
        self.assertFalse(result["passed"])

    def test_distributed_load_recovers_clamped_beam_interior_and_energy(self):
        # Exact w(x)=q*x²*(L-x)²/(24EI). Clamps have no nodal motion.
        for nodes in ([0.0, 2.0], [0.0, 0.25, 1.4, 2.0]):
            for load in (-120.0, 120.0):
                with self.subTest(nodes=nodes, load=load):
                    result = w5.verify_fem_structure({
                        "mode": "euler_bernoulli_beam_1d", "x_nodes_m": nodes,
                        "EI_Nm2": 1000.0, "q_uniform_N_m": load,
                        "supports": [{"node": index, "fix_w": True, "fix_theta": True}
                                     for index in (0, len(nodes) - 1)],
                    })
                    self.assertTrue(result["passed"])
                    for mid, a, b in zip(result["element_midspan_deflections_m"], nodes, nodes[1:]):
                        x = (a + b) / 2
                        self.assertAlmostEqual(mid, load * x**2 * (2 - x)**2 / 24000, places=13)
                    self.assertAlmostEqual(result["strain_energy_J"], 0.32, places=12)
                    self.assertAlmostEqual(result["external_work_J"], 0.32, places=12)
                    self.assertEqual(result["support_work_J"], 0.0)

    def test_prescribed_beam_motion_performs_support_work(self):
        # Prescribed displacement with zero end rotations, plus a load acting at
        # that constrained coordinate. The load changes its reaction, not U.
        result = w5.verify_fem_structure({
            "mode": "euler_bernoulli_beam_1d", "x_nodes_m": [0.0, 2.0],
            "EI_Nm2": 1000.0, "nodal_loads": [{"node": 1, "fy_N": 7.0}],
            "supports": [{"node": 0, "fix_w": True, "fix_theta": True},
                         {"node": 1, "fix_w": True, "fix_theta": True, "w_m": 0.01}],
        })
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["strain_energy_J"], 0.075)
        self.assertAlmostEqual(result["element_midspan_deflections_m"][0], 0.005)
        self.assertAlmostEqual(result["external_work_J"], 0.035)
        self.assertAlmostEqual(result["support_work_J"], 0.040)

    def test_fem_euler_bernoulli_beam_propped_cantilever_and_cli(self):
        beam = w5.verify_fem_structure({
            "mode": "euler_bernoulli_beam_1d",
            "x_nodes_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "EI_Nm2": 2.0e7,
            "q_uniform_N_m": -12000.0,
            "supports": [
                {"node": 0, "fix_w": True, "fix_theta": True},
                {"node": 4, "fix_w": True, "fix_theta": False},
            ],
        })
        self.assertTrue(beam["passed"])
        self.assertAlmostEqual(beam["reactions_V_M"][4][0], 18000.0, places=3)
        self.assertAlmostEqual(beam["reactions_V_M"][0][0], 30000.0, places=3)

        proc = subprocess.run(
            [sys.executable, str(ROOT / "hamon_pde_control_psychro_fem.py"), "--json",
             json.dumps({"domain": "control", "mode": "state_space", "A": [[0, 1], [-6, -5]], "B": [[0], [1]], "C": [[1, 0]]})],
            capture_output=True, text=True, check=True,
        )
        out = json.loads(proc.stdout)
        self.assertTrue(out["passed"])
        self.assertTrue(out["hurwitz_stable"])


if __name__ == "__main__":
    unittest.main()
