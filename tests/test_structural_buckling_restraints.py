"""Euler loads from the admissible bending modes of the declared supports."""
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


BASE = dict(span_m=2., E_Pa=200e9, I_m4=8e-6, area_m2=.05, safety_factor=1.5)


def solve(boundary, **inputs):
    return run_router_conservation_cad_verifier(
        {"mode":"structural", "spec":dict(BASE,boundary=boundary,**inputs)})


def mode_energy(boundary):
    # Pcr = integral(EI*w''² dx)/integral(w'² dx). These modes obey the
    # actual support displacement, rotation, moment and shear conditions.
    length, rigidity = BASE["span_m"], BASE["E_Pa"]*BASE["I_m4"]
    if boundary.startswith("cantilever"):
        wave = math.pi/(2*length)
        slope = lambda x: wave*math.sin(wave*x)
        curvature = lambda x: wave**2*math.cos(wave*x)
    elif boundary.startswith("fixed_fixed"):
        wave = 2*math.pi/length
        slope = lambda x: wave*math.sin(wave*x)
        curvature = lambda x: wave**2*math.cos(wave*x)
    else:
        wave = math.pi/length
        slope = lambda x: wave*math.cos(wave*x)
        curvature = lambda x: -wave**2*math.sin(wave*x)
    points = [(i+.5)*length/64 for i in range(64)]
    return rigidity*math.fsum(curvature(x)**2 for x in points)/math.fsum(slope(x)**2 for x in points)


class MemberBuckling(unittest.TestCase):
    def test_all_declared_restraints_match_their_displacement_energy(self):
        for boundary,k in (("cantilever_point",2.),("cantilever_udl",2.),
                           ("simply_supported_point",1.),("simply_supported_udl",1.),
                           ("fixed_fixed_udl",.5)):
            expected = mode_energy(boundary)
            for fraction in (.95,1.05):
                with self.subTest(boundary=boundary,fraction=fraction):
                    r = solve(boundary, axial_compression_N=fraction*expected/BASE["safety_factor"])
                    self.assertTrue(r["conservation_verified"])
                    self.assertEqual(r["buckling_effective_length_source"],"boundary")
                    self.assertEqual(r["buckling_effective_length_factor"],k)
                    self.assertTrue(math.isclose(r["euler_critical_load_N"],expected,rel_tol=1e-14))
                    self.assertEqual(r["passes_buckling"],fraction<1.)

    def test_allowable_equality_is_distinct_from_the_euler_bifurcation(self):
        for boundary in ("cantilever_point", "simply_supported_point", "fixed_fixed_udl"):
            critical = solve(boundary)["euler_critical_load_N"]
            self.assertTrue(math.isclose(critical,mode_energy(boundary),rel_tol=1e-14))
            allowable = critical / BASE["safety_factor"]
            for load, safety, passes in (
                    (allowable, 1.5, True),
                    (math.nextafter(allowable, math.inf), 1.5, False),
                    (critical, 1.5, False),
                    (critical, 1., True),
                    (math.nextafter(critical, math.inf), 1., False)):
                with self.subTest(boundary=boundary, load=load, safety=safety):
                    r = solve(boundary, axial_compression_N=load, safety_factor=safety)
                    self.assertEqual(r["euler_critical_load_N"],critical)
                    self.assertEqual(r["passes_buckling"],passes)

    def test_unsafe_cantilever_and_safe_fixed_fixed_for_the_same_material(self):
        cantilever = solve("cantilever_point",axial_compression_N=1.2e6,P_N=600.)
        fixed = solve("fixed_fixed_udl",axial_compression_N=4e6,w_N_m=600.)
        self.assertLess(cantilever["euler_critical_load_N"],1.2e6)
        self.assertFalse(cantilever["passes_buckling"])
        self.assertGreater(fixed["euler_critical_load_N"],4e6*BASE["safety_factor"])
        self.assertTrue(fixed["passes_buckling"])

    def test_explicit_end_factor_and_transverse_loads(self):
        for boundary,force in (("cantilever_point",{"P_N":350.}),
                               ("fixed_fixed_udl",{"w_N_m":-250.})):
            r=solve(boundary,**force)
            explicit=solve(boundary,K_buckling=.8,**force)
            self.assertEqual(explicit["buckling_effective_length_source"],"K_buckling")
            self.assertEqual(explicit["buckling_effective_length_factor"],.8)
            self.assertAlmostEqual(explicit["euler_critical_load_N"] /
                (math.pi**2*BASE["E_Pa"]*BASE["I_m4"]/(.8*BASE["span_m"])**2),1.)
            for key in ("max_deflection_m","max_moment_Nm","max_shear_N",
                        "virtual_work_deflection_m","passes_stress","conservation_verified"):
                self.assertEqual(r[key],explicit[key])
            self.assertFalse(solve(boundary,K_buckling=0.)["ok"])
