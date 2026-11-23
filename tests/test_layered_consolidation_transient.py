"""Layered Terzaghi consolidation limits, independently derived modal references and public routes."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scipy.integrate import quad
from hamon_transient_footing import verify_layered_transient
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_pareto import route_from_intake_prompt

ROOT = Path(__file__).resolve().parents[1]
EMPTY = dict(schema="hamon.ple.intake.v1", scanned_files=0, engineering_file_count=0,
             by_extension={}, files=[], warnings=[])


def layer(a, b, mv=1e-7, cv=1e-6):
    return dict(top_depth_m=a, bottom_depth_m=b, mv_per_Pa=mv, cv_m2_s=cv)


def column(*layers, **extra):
    return dict(dict(calculation="layered_consolidation_transient", hydraulic_layers=list(layers),
                     top_boundary="drained", bottom_boundary="impermeable",
                     delta_sigma_v_Pa=100000, time_s=1e6), **extra)


def homogeneous(tv):
    # Average retention after a uniform step; drainage path is H or H/2.
    return math.fsum(8/((2*n+1)*math.pi)**2 * math.exp(-((n+.5)*math.pi)**2 * tv)
                     for n in range(100))


def matched_layers(m1, m2, elapsed):
    """Exact two-layer modes: each layer is one metre thick with Cv=1e-6."""
    angle = math.atan(math.sqrt(m1/m2))
    mass, interface, flux = [], [], []
    for n in range(50):
        for theta in (n*math.pi + angle, (n+1)*math.pi - angle):
            # Top eigenfunction sin(theta*x); bottom tan(theta)*cos(theta*(2-x)).
            slope = math.tan(theta)
            projection = (m1*(1-math.cos(theta)) + m2*slope*math.sin(theta))/theta
            term = math.sin(2*theta)/(4*theta)
            norm = m1*(.5-term) + m2*slope*slope*(.5+term)
            amplitude = 100000*projection/norm*math.exp(-theta*theta*1e-6*elapsed)
            mass.append(projection*amplitude)
            interface.append(amplitude*math.sin(theta))
            flux.append(-1e-6*m1*theta*amplitude*math.cos(theta))
    return 100000*(m1+m2)-math.fsum(mass), math.fsum(interface), math.fsum(flux)


class LayeredConsolidationTests(unittest.TestCase):
    def test_homogeneous_drainage_and_partition(self):
        for setup in ((layer(0,2),), (layer(0,1), layer(1,2))):
            for top,bottom,path in (("drained","impermeable",2), ("impermeable","drained",2),
                                    ("drained","drained",1)):
                spec = column(*setup, top_boundary=top, bottom_boundary=bottom)
                r = verify_layered_transient(spec)
                self.assertTrue(r["conservation_verified"], r)
                self.assertAlmostEqual(r["settlement_at_time_m"], .02*(1-homogeneous(1/path**2)), delta=1e-8)
            closed = verify_layered_transient(column(*setup, top_boundary="impermeable"))
            self.assertTrue(closed["conservation_verified"], closed)
            self.assertEqual(closed["settlement_at_time_m"], 0.)
        instant = verify_layered_transient(column(layer(0,2), time_s=0))
        self.assertEqual(instant["settlement_at_time_m"], 0.)
        self.assertEqual(instant["layers_mean_excess_pressure_Pa"], [100000.])

    def test_layer_order_changes_original_input_pressure_and_flux(self):
        for m1,m2 in ((1e-7,4e-7),(4e-7,1e-7)):
            reference, pressure, flow = matched_layers(m1,m2,1e6)
            out = verify_layered_transient(column(layer(0,1,m1),layer(1,2,m2)))
            self.assertTrue(out["conservation_verified"],out)
            self.assertAlmostEqual(out["settlement_at_time_m"], reference, delta=reference*1e-6)
            face = out["interfaces"][0]
            self.assertAlmostEqual(face["excess_pressure_Pa"], pressure, delta=.02)
            self.assertAlmostEqual(face["downward_darcy_flux_m_s"], flow, delta=2e-14)
            self.assertLess(out["dual_solver_rel_residual"],1e-6)
        out = verify_layered_transient(column(layer(0,1),layer(1,2,4e-7), time_s=1e12))
        self.assertTrue(out["conservation_verified"],out)
        self.assertAlmostEqual(out["settlement_at_time_m"],.05)

    def test_finite_footprint_column_with_spatially_varying_initial_pressure(self):
        s = column(layer(.1,1.1),layer(1.1,2.1), time_s=5e5)
        del s["delta_sigma_v_Pa"]
        s.update(settlement_column={"x_m":0,"y_m":0}, rectangular_loads=[
            dict(center_x_m=0, center_y_m=0,width_m=2,length_m=2,net_pressure_Pa=200000)])
        def stress(x):
            z = x+.1
            radius = math.sqrt(2+z*z)
            return 4*200000/(2*math.pi)*(math.atan(1/(z*radius))+2*z/(radius*(1+z*z)))
        expected = quad(stress,0,2)[0]*1e-7
        for n in range(14):
            k = (n+.5)*math.pi/2
            coeff = quad(lambda x:stress(x)*math.sin(k*x),0,2)[0]
            expected -= 1e-7*coeff/k*math.exp(-1e-6*k*k*s["time_s"])
        r = run_router_conservation_cad_verifier({"mode":"civil_geotech","spec":s})
        self.assertTrue(r["conservation_verified"],r)
        self.assertAlmostEqual(r["settlement_at_time_m"],expected,delta=expected*1e-6)
        closed = verify_layered_transient({**s,"top_boundary":"impermeable","time_s":1e12})
        self.assertEqual(closed["settlement_at_time_m"],0.)
        self.assertAlmostEqual(*closed["layers_mean_excess_pressure_Pa"],delta=.05)

    def test_unqualified_requests_and_bound_provider(self):
        s = column(layer(0,1),layer(1,2,4e-7))
        def routed(spec):
            return route_from_intake_prompt(json.dumps(EMPTY), instruction=json.dumps({"mode":"civil_geotech","spec":spec}))
        good = routed({**s,"proposed_settlement_m":.012478821142420757})
        self.assertEqual(good["thinking_tier"], "low")
        self.assertTrue(json.loads(good["presolve_result_json"])["conservation_verified"])
        for changes in ({"proposed_settlement_m":.040267417471040755},
                        {"hydraulic_layers":[layer(0,1),layer(1.1,2)]},
                        {"hydraulic_layers":[{**layer(0,1),"Cc":.4}]},
                        {"top_boundary":"free"}, {"time_s":-1}, {"load_history":[]}, {"time_s":1e-15}):
            self.assertNotIn("presolve_result_json",routed({**s,**changes}),changes)

    def test_ordinary_dual_verify_cli(self):
        spec = column(layer(0,1),layer(1,2,4e-7))
        with tempfile.TemporaryDirectory() as home:
            path = Path(home)/"settlement.json"
            for proposed, passed in ((.012478821142420757, True),(.040267417471040755,False)):
                path.write_text(json.dumps({"kind":"layered_consolidation_transient",
                                            "spec":{**spec,"proposed_settlement_m":proposed}}))
                proc = subprocess.run([sys.executable,str(ROOT/"engineering_toolkit.py"),"dual-verify",str(path)],
                                      capture_output=True,text=True,check=False)
                out = json.loads(proc.stdout)
                self.assertEqual(out["dual_verified"],passed,out)

    def test_loopback_provider_sends_only_the_verified_response(self):
        import asyncio
        from fin_provider_replay import exchange
        ref = matched_layers(1e-7,4e-7,1e6)[0]
        def inspect(prompt, row, digest):
            self.assertTrue(row["conservation_verified"])
            self.assertLess(row["dual_solver_rel_residual"],1e-6)
            self.assertAlmostEqual(row["settlement_at_time_m"],ref,delta=ref*1e-6)
            return {}
        for claim, valid in ((ref, True), (.040267417471040755,False)):
            s = column(layer(0,1),layer(1,2,4e-7),proposed_settlement_m=claim)
            asyncio.run(exchange(ROOT,json.dumps(dict(mode="civil_geotech",spec=s)),
                                 "independent settlement: %.10g m" % ref,valid,inspect))

    def test_existing_uniform_history_route_agrees_on_shared_domain(self):
        from hamon_layered_consolidation import verify_layered_consolidation
        s = column(layer(0,1),layer(1,2,4e-7))
        history = dict(calculation="layered_consolidation_1d", time_s=s["time_s"], drainage="single",
                       load_history=[dict(time_s=0,surcharge_Pa=s["delta_sigma_v_Pa"])],
                       consolidation_layers=[dict(thickness_m=x["bottom_depth_m"]-x["top_depth_m"],
                                                  mv_per_Pa=x["mv_per_Pa"],cv_m2_s=x["cv_m2_s"])
                                             for x in s["hydraulic_layers"]])
        earlier = verify_layered_consolidation(history)
        new = verify_layered_transient(s)
        self.assertTrue(earlier["conservation_verified"], earlier)
        self.assertTrue(new["conservation_verified"], new)
        self.assertAlmostEqual(new["settlement_at_time_m"],earlier["settlement_at_time_m"],delta=1e-8)
