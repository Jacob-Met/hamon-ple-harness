import math
import unittest
from hamon_dual_verifier import run_dual_verify_spec


def resistor(a,b,r):
    return dict(n1=a,n2=b,R_ohm=r)


def winding(a,b,r=0):
    return dict(n1=a,n2=b,L_H=.01,R_series_ohm=r)


def solve(spec, frequency=None):
    kind = "dc_nodal_kcl" if frequency is None else "ac_phasor_rlc_network"
    if frequency is not None:
        spec = dict(spec, freq_Hz=frequency)
    return run_dual_verify_spec({"mode":"domain","kind":kind,"spec":spec})


def value(entry):
    return complex(*entry) if isinstance(entry,list) else complex(entry)


class DCWindingTests(unittest.TestCase):
    def test_loaded_winding_both_zero_frequency_routes(self):
        for r, voltage, iw in [(0,0,10),(1,10/3,10/3)]:
            for f in (None,0):
                spec = dict(voltage_sources_to_ground={"s":10},
                            resistors=[resistor("s","x",1),resistor("x",0,1)],
                            inductors=[winding("x",0,r)], capacitors=[dict(n1="x",n2=0,C_F=1e-4)])
                o = solve(spec,f)
                self.assertTrue(o["dual_verified"])
                self.assertAlmostEqual(value(o["Vx_V"]),voltage)
                vals = [value(v["current_A"]) for v in o["branch_currents"]]
                for a,b in zip(vals,[10-voltage,voltage,iw,0]):
                    self.assertAlmostEqual(a,b)
                self.assertAlmostEqual(value(o["prescribed_node_supply_A"]["s"]),10-voltage)
                self.assertIsInstance(o["Vx_V"],float if f is None else list)

    def test_tree_currents_and_nonzero_voltage_component(self):
        # Two subtrees branch at x. Reconstruct each oriented current across its cut.
        spec = dict(voltage_sources_to_ground={"s":16},
                    resistors=[resistor("s","x",2),resistor("a",0,3),resistor("b",0,6)],
                    inductors=[winding("a","x"),winding("x","b")],
                    current_injections=[dict(node="a",I_A=3)])
        # (16−V)/2 + 3 = V/3 + V/6 yields V=11.
        o=solve(spec)
        for key in ("Vx_V","Va_V","Vb_V"):
            self.assertEqual(o[key],o["Vx_V"])
            self.assertAlmostEqual(o[key],11)
        winding_rows=[r for r in o["branch_currents"] if r["collection"]=="inductors"]
        for row, expected in zip(winding_rows,[3-11/3,11/6]):
            self.assertAlmostEqual(row["current_A"],expected)
        self.assertAlmostEqual(o["prescribed_node_supply_A"]["s"],2.5)
        self.assertAlmostEqual(16*2.5+11*3,2.5**2*2+11**2/3+11**2/6)

    def test_prescribed_tree_current_injection_and_dc_open(self):
        spec=dict(voltage_sources_to_ground={"bus":12},inductors=[winding("bus","x")],
                  resistors=[resistor("x",0,4)],capacitors=[dict(n1="bus",n2="x",C_F=.2)],
                  current_injections=[dict(node="x",I_A=[.5,.25]),dict(node="bus",I_A=1)])
        o=solve(spec)
        self.assertEqual(value(o["branch_currents"][1]["current_A"]),2.5-.25j)
        self.assertEqual(value(o["prescribed_node_supply_A"]["bus"]),1.5-.25j)
        self.assertEqual(o["branch_currents"][2]["current_A"],0)

    def test_nonunique_windings_and_incompatible_prescriptions(self):
        for winds,fixed in [
            ([winding("s","x"),winding("s","x")],{"s":10}),
            ([winding("s","x"),winding("x","z"),winding("z","s")],{"s":10}),
            ([winding("s","x"),winding("x","z")],{"s":10,"z":10}),
            ([winding("s","x"),winding("x","z")],{"s":10,"z":9}),
            ([winding("s","s")],{"s":10}),
        ]:
            with self.assertRaises(ValueError):
                solve(dict(voltage_sources_to_ground=fixed,inductors=winds))
        with self.assertRaises((ValueError,ArithmeticError)):
            solve(dict(capacitors=[dict(n1="floating",n2=0,C_F=.125)]))

    def test_finite_ac_retains_series_impedance_and_capacitor_admittance(self):
        f=40
        spec=dict(voltage_sources_to_ground={"s":24},
                  inductors=[winding("s","x",.5)],resistors=[resistor("x",0,3)],
                  capacitors=[dict(n1="x",n2=0,C_F=.0002)])
        o=solve(spec,f)
        z=.5+2j*math.pi*f*.01
        y=1/3+2j*math.pi*f*.0002
        vx=24/(1+z*y)
        self.assertAlmostEqual(value(o["Vx_V"]),vx)
        self.assertAlmostEqual(value(o["branch_currents"][1]["current_A"]),y*vx)
        self.assertAlmostEqual(value(o["branch_currents"][2]["current_A"]),vx*2j*math.pi*f*.0002)
