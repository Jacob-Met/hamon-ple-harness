"""Coupled beam-column displacement and section forces through the structural route."""
import math
import unittest
import numpy as np

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


BOUNDARIES = ("simply_supported_point", "simply_supported_udl", "cantilever_point",
              "cantilever_udl", "fixed_fixed_udl")


def case(bc, ratio=.5, force=850.):
    length, ei = 2., 1.6e6
    factor = 2. if bc.startswith("cantilever") else .5 if bc.startswith("fixed") else 1.
    return dict(boundary=bc, span_m=length, E_Pa=200e9, I_m4=8e-6, area_m2=.05,
                axial_compression_N=ratio*math.pi**2*ei/(factor*length)**2,
                **({"P_N":force} if bc.endswith("point") else {"w_N_m":force}))


def run(p):
    return run_router_conservation_cad_verifier({"mode":"structural", "spec":p})


def finite_elements(p, elements=32):
    # Independent displacement discretization. Integrate EI u''v'' - N u'v'
    # for cubic Hermite shapes; then recover cut forces from global equilibrium.
    length = p["span_m"]
    h = length/elements
    n = 2*(elements+1)
    kg = np.zeros((n,n))
    fg = np.zeros(n)
    force, q = abs(p.get("P_N",0.)), abs(p.get("w_N_m",0.))
    axial = p["axial_compression_N"]
    ei = p["E_Pa"] * p["I_m4"]
    kb = ei/h**3*np.array([[12,6*h,-12,6*h],[6*h,4*h*h,-6*h,2*h*h],
                           [-12,-6*h,12,-6*h],[6*h,2*h*h,-6*h,4*h*h]])
    geometric = axial/(30*h)*np.array([[36,3*h,-36,3*h],[3*h,4*h*h,-3*h,-h*h],
                                      [-36,-3*h,36,-3*h],[3*h,-h*h,-3*h,4*h*h]])
    for i in range(elements):
        indices = slice(2*i,2*i+4)
        kg[indices,indices] += kb-geometric
        fg[indices] += q*np.array([h/2,h*h/12,h/2,-h*h/12])
    bc = p["boundary"]
    cantilever = bc.startswith("cantilever")
    support = [0,1] if cantilever else ([0,1,n-2,n-1] if bc.startswith("fixed") else [0,n-2])
    target = elements if cantilever else elements//2
    fg[2*target] += force
    free = [i for i in range(n) if i not in support]
    u = np.zeros(n)
    u[free] = np.linalg.solve(kg[np.ix_(free,free)],fg[free])
    reaction = kg@u-fg
    s0 = -reaction[0]
    m0 = -reaction[1] if 1 in support else 0.
    sample_m, sample_v = [], []
    for i in range(target):
        v0,t0,v1,t1 = u[2*i:2*i+4]
        for z in np.linspace(0.,1.,9):
            x = h*(i+z)
            y = ((1-3*z*z+2*z**3)*v0+h*(z-2*z*z+z**3)*t0+
                 (3*z*z-2*z**3)*v1+h*(z**3-z*z)*t1)
            rotation = ((6*z*z-6*z)*v0/h+(1-4*z+3*z*z)*t0+
                        (6*z-6*z*z)*v1/h+(3*z*z-2*z)*t1)
            sample_m.append(abs(m0-s0*x+.5*q*x*x-axial*y))
            sample_v.append(abs(s0-q*x+axial*rotation))
    return u[2*target], np.array(sample_m), np.array(sample_v)


class BeamColumnTests(unittest.TestCase):
    def test_serviceability_false_pass_at_half_critical_load(self):
        p=case("simply_supported_point",force=36000.)
        r=run(p)
        theta = math.sqrt(p["axial_compression_N"]/(p["E_Pa"]*p["I_m4"])) * p["span_m"]/2
        delta = p["P_N"]*p["span_m"]/(4*p["axial_compression_N"])*(math.tan(theta)/theta-1)
        self.assertAlmostEqual(delta,.007448579340681537,places=14)
        self.assertAlmostEqual(r["max_deflection_m"],delta,places=14)
        self.assertAlmostEqual(r["max_moment_Nm"],18000.+p["axial_compression_N"]*delta,places=8)
        self.assertFalse(r["passes_deflection"])
        self.assertTrue(r["passes_buckling"] and r["conservation_verified"])

    def test_geometric_stiffness_displacement_and_section_forces_on_all_supports(self):
        for bc in BOUNDARIES:
            for signed in (-850.,850.):
                with self.subTest(boundary=bc,load=signed):
                    p = case(bc,ratio=.375,force=signed)
                    r = run(p)
                    y, m, v = finite_elements(p)
                    self.assertTrue(r["conservation_verified"],r)
                    self.assertTrue(math.isclose(r["max_deflection_m"],y,rel_tol=4e-6))
                    self.assertTrue(math.isclose(r["virtual_work_deflection_m"],y,rel_tol=4e-6))
                    self.assertTrue(math.isclose(r["max_moment_Nm"],max(m),rel_tol=4e-6))
                    self.assertTrue(math.isclose(r["max_shear_N"],max(v),rel_tol=3e-5))
                    self.assertLess(r["section_moment_equilibrium_rel_error"],1e-10)

    def test_positive_stiffness_boundary_and_independent_design_factor(self):
        for bc in BOUNDARIES:
            weak = run(case(bc,ratio=.999999,force=1e-6))
            self.assertTrue(weak["ok"] and weak["conservation_verified"],weak)
            p = case(bc,ratio=1.000001,force=1e-6)
            p["K_buckling"] = .25
            unstable = run(p)
            self.assertFalse(unstable["ok"] or unstable["passes_buckling"])
            self.assertIsNone(unstable["passes_deflection"])
            self.assertNotIn("max_deflection_m",unstable)
            self.assertLess(unstable["in_plane_euler_critical_load_N"],p["axial_compression_N"])

    def test_stability_test_does_not_snap_neighbors_of_the_first_eigenvalue(self):
        p = dict(boundary="simply_supported_point",span_m=1.,E_Pa=1.,I_m4=1.,
                 area_m2=1.,P_N=1e-25,safety_factor=1.)
        # The binary64 value of pi² is below the true eigenvalue; its
        # next larger float is above it. The transverse load is kept small.
        critical_wire = math.pi**2
        below = run(dict(p,axial_compression_N=critical_wire))
        above = run(dict(p,axial_compression_N=math.nextafter(critical_wire,math.inf)))
        self.assertTrue(below["ok"] and below["conservation_verified"],below)
        self.assertFalse(above["ok"])
        self.assertIsNone(above["passes_deflection"])

    def test_tiny_positive_preload_and_zero_preload_share_the_elastic_limit(self):
        for bc in BOUNDARIES:
            p = case(bc)
            plain = run(dict(p,axial_compression_N=0.))
            slight = run(dict(p,axial_compression_N=1e-90))
            self.assertTrue(slight["conservation_verified"],slight)
            for field in ("max_deflection_m","max_moment_Nm","max_shear_N"):
                self.assertTrue(math.isclose(plain[field],slight[field],rel_tol=2e-14))

    def test_combined_tube_stress_uses_collocated_moment_and_local_section_shear(self):
        radius, thickness = .08, .002
        area, inertia = 2*math.pi*radius*thickness, math.pi*radius**3*thickness
        for bc in ("simply_supported_point","cantilever_udl","fixed_fixed_udl"):
            p = case(bc,ratio=.35,force=650.)
            p.update(section={"shape":"thin_walled_circular_tube","mean_radius_m":radius,
                              "wall_thickness_m":thickness},area_m2=area,I_m4=inertia,
                     torque_Nm=-450.,axial_compression_N=p["axial_compression_N"]*inertia/8e-6)
            r = run(p)
            _, m, v = finite_elements(p,48)
            phi = np.linspace(0.,math.pi/2,1001)
            normal = p["axial_compression_N"]/area + m[:,None]*radius/inertia*np.sin(phi)
            shear = abs(p["torque_Nm"])*radius/(2*inertia) + 2*v[:,None]/area*np.cos(phi)
            peak = float(np.max(np.hypot(normal,math.sqrt(3)*shear)))
            bound = r["von_mises_bounds_Pa"]
            self.assertTrue(r["conservation_verified"],r)
            self.assertLessEqual(bound["lower"],peak*(1+5e-6))
            self.assertGreaterEqual(bound["upper"],peak*(1-5e-6))
            self.assertLess(bound["upper"]/peak,1.002)
            for scale, passed in ((.98,False),(1.02,True)):
                limited = run(dict(p,yield_strength_Pa=scale*peak,safety_factor=1.))
                self.assertEqual(limited["passes_stress"],passed)
