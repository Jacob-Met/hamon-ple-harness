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
