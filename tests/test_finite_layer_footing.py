"""Physical depth-integral, near-surface asymptotics and actual engineering routes."""
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route
from hamon_dual_verifier import solve_domain_problem
from hamon_pareto import _wave8_result


def layer(top, bottom, e0=.8, cc=.32, initial=75000, **extra):
    return dict(top_depth_m=top, bottom_depth_m=bottom, e0=e0, Cc=cc,
                sigma_v0_eff_Pa=initial, **extra)


def footing(*layers, xy=(0, 0), pressure=200000):
    return dict(calculation="layered_footing_settlement", settlement_column=dict(zip(("x_m","y_m"),xy)),
                rectangular_loads=[dict(center_x_m=0, center_y_m=0, width_m=2,
                                        length_m=2, net_pressure_Pa=pressure)], settlement_layers=list(layers))


def check(spec):
    return route({"mode":"civil_geotech", "spec":spec})


def cartesian_reference(top, bottom, e0=.8, cc=.32, initial=75000., pc=None, cr=0.):
    """Direct triple integration in Cartesian coordinates and physical depth."""
    from scipy.integrate import quad
    pc = initial if pc is None else pc

    def strain(z):
        def row(x):
            def kernel(y):
                return 3*z**3 / (2*math.pi*(x*x+y*y+z*z)**2.5)
            return quad(kernel, 0, 1, epsabs=2e-12)[0]
        stress = 800000 * quad(row, 0, 1, epsabs=2e-11)[0]
        below = min(stress, pc-initial)
        loss = cr*math.log10(1+below/initial) + cc*math.log10(1+(stress-below)/(initial+below))
        return loss/(1+e0)
    return quad(strain, top, bottom, epsabs=2e-12, epsrel=2e-10, limit=100)[0]


class FiniteDepthFootingTests(unittest.TestCase):
    def test_midpoint_reverses_actual_limit_and_paired_routes(self):
        # Independent Cartesian integration: first integrate 3*q*z^3/(2*pi*r^5)
        # in x and y over [-1,1]^2, then integrate log strain from .1 to 6.1 m.
        reference = .2406348714868344
        self.assertAlmostEqual(cartesian_reference(.1, 6.1), reference, places=11)
        midpoint = .1725689762142289
        spec = footing(layer(.1, 6.1))
        out = check(spec)
        self.assertTrue(out["conservation_verified"], out)
        self.assertAlmostEqual(out["primary_consolidation_settlement_m"], reference, places=11)
        self.assertAlmostEqual(out["corner_settlement_m"], reference, places=11)
        decision = solve_domain_problem("consolidation_settlement", {**spec, "max_settlement_m":.2})
        self.assertTrue(decision["dual_verified"], decision)
        self.assertFalse(decision["passes_limit"])
        self.assertGreater(reference-midpoint,.068)
        case = {"mode":"civil_geotech", "spec":{**spec, "proposed_settlement_m":midpoint}}
        self.assertIsNone(_wave8_result(case))
        case["spec"]["proposed_settlement_m"] = reference
        self.assertTrue(_wave8_result(case)["conservation_verified"])

    def test_split_and_different_compression_laws(self):
        base = check(footing(layer(.1, 6.1)))
        split = check(footing(layer(.1, 1.1), layer(1.1, 6.1)))
        self.assertTrue(split["conservation_verified"], split)
        self.assertAlmostEqual(base["primary_consolidation_settlement_m"],
                               split["primary_consolidation_settlement_m"], places=10)
        top = layer(.1, 1.1)
        bottom = layer(1.1, 6.1, .65, .22, 80000, Cr=.04, sigma_preconsolidation_Pa=110000)
        mixed = check(footing(top, bottom))
        self.assertTrue(mixed["conservation_verified"], mixed)
        standalone = [check(footing(l))["primary_consolidation_settlement_m"] for l in (top,bottom)]
        self.assertAlmostEqual(sum(standalone), mixed["primary_consolidation_settlement_m"], places=12)
        reference = cartesian_reference(.1, 1.1) + cartesian_reference(1.1, 6.1, .65, .22,
                                                                        80000, 110000, .04)
        self.assertAlmostEqual(mixed["primary_consolidation_settlement_m"], reference, places=10)
        solid = 1/1.8+5/1.65
        initial_void = 6-solid
        self.assertAlmostEqual((initial_void-sum(standalone))/solid, mixed["final_void_ratio"], places=12)

    def test_vanishing_loads_and_surface_limits(self):
        for xy,fraction in (((0,0),1.),((1,0),.5),((1,1),.25)):
            spec = footing(layer(0,1e-7),xy=xy)
            out = check(spec)
            limit = .32*math.log1p(200000*fraction/75000)/(1.8*math.log(10))
            self.assertTrue(out["conservation_verified"], out)
            self.assertAlmostEqual(out["primary_consolidation_settlement_m"]/1e-7,limit,places=10)
        # Off-footprint, stress ~ z^3 and the thin-layer settlement ~ thickness^4.
        exterior = [check(footing(layer(0,h),xy=(2,2))) for h in (1e-8,1e-12)]
        self.assertTrue(all(x["conservation_verified"] for x in exterior), exterior)
        self.assertAlmostEqual(exterior[0]["primary_consolidation_settlement_m"]/
                               exterior[1]["primary_consolidation_settlement_m"]/1e16,1.,places=9)
        vals = [check(footing(layer(0,1),pressure=q)) for q in (0.,1e-9,2e-9)]
        self.assertTrue(all(x["conservation_verified"] for x in vals), vals)
        self.assertEqual(vals[0]["primary_consolidation_settlement_m"],0.)
        self.assertGreater(vals[1]["primary_consolidation_settlement_m"],0.)
        self.assertAlmostEqual(vals[2]["primary_consolidation_settlement_m"]/
                               vals[1]["primary_consolidation_settlement_m"],2.,places=12)

    def test_thin_yielded_zone_at_surface(self):
        # The central surface stress exceeds preconsolidation by 0.01 Pa.
        # Independent bisection puts the yield boundary at depth 0.004054355 m.
        # A midpoint or a quadrature confined to the deep elastic zone gives zero.
        out = check(footing(layer(0, 1, Cr=0., sigma_preconsolidation_Pa=274999.99)))
        self.assertTrue(out["conservation_verified"], out)
        self.assertAlmostEqual(out["primary_consolidation_settlement_m"]/8.53711478665e-12, 1., places=5)

    def test_buried_yield_has_two_boundaries(self):
        # The observation column stands 1 m beyond the nearest footing edge.
        # Direct positive Cartesian pressure attains its interior peak at 2.09075 m.
        from scipy.integrate import quad
        from scipy.optimize import brentq
        threshold = 18966.379138799333

        def pressure(z):
            def x_slice(x):
                return quad(lambda y: 3*z**3/(2*math.pi*(x*x+y*y+z*z)**2.5),0,1)[0]
            return 400000*quad(x_slice,1,3)[0]

        lower = brentq(lambda z: pressure(z)-threshold, 2.08, 2.09075)
        upper = brentq(lambda z: pressure(z)-threshold, 2.09075, 2.10)
        coefficient = .32/(1.8*math.log(10))
        expected = quad(lambda z: coefficient*math.log1p((pressure(z)-threshold)/(75000+threshold)),
                        lower, upper, epsabs=1e-22)[0]
        spec = footing(layer(0,8,Cr=0.,sigma_preconsolidation_Pa=75000+threshold),xy=(2,0))
        out = check(spec)
        self.assertTrue(out["conservation_verified"], out)
        self.assertLess(upper-lower, .00101)
        self.assertAlmostEqual(out["primary_consolidation_settlement_m"]/expected, 1., places=6)
        self.assertFalse(check({**spec, "proposed_settlement_m":0})["conservation_verified"])

    def test_finite_inputs_cannot_be_silently_scalarized(self):
        good = footing(layer(.1,1.1))
        for case in (footing(layer(.1,1,delta_sigma_v_Pa=999)), footing(layer(.1,1,time_s=2)),
                     footing(layer(.1,1,stress_point={"depth_m":.5})),
                     footing(layer(.1,1),layer(1.2,2)), footing(layer(.1,1),layer(.8,2)),
                     {**good,"stress_point":{"x_m":0,"y_m":0,"depth_m":.6}},
                     {**good,"time_s":10}, {**good,"calculation":"consolidation_1d"},
                     {**good,"calculation":"consolidation_history_1d"}):
            self.assertFalse(check(case)["conservation_verified"], case)
        bearing = dict(calculation="bearing_capacity",bearing_method="vesic",footing_B_m=2,
                       depth_Df_m=1,gamma_N_m3=18000,c_prime_Pa=10000,phi_deg=30,FS=3)
        both = check({**good,**bearing})
        self.assertTrue(both["conservation_verified"], both)
        self.assertAlmostEqual(both["primary_consolidation_settlement_m"],.09225437772311096,places=10)
