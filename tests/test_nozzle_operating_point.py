"""Independent isentropic mass-flux, stagnation energy, and two-branch checks."""
import math
import unittest
from unittest.mock import patch

from mpmath import mp

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
import hamon_wave6_physics_verifiers as wave6


def route(**p):
    return run_router_conservation_cad_verifier(
        {"mode": "compressible_acoustics", "spec": {"subkind": "isentropic", **p}})


class NozzleOperatingPoint(unittest.TestCase):
    def check_state(self, result, gamma, target_area):
        # Unit-area mass flux directly from Mach, normalized by its sonic value.
        with mp.workdps(90):
            g, m = mp.mpf(gamma), mp.mpf(result["mach"])
            T, P, V = map(mp.mpf, [result["T_static_K"], result["P_static_Pa"], result["velocity_m_s"]])
            R, T0, P0, throat = map(mp.mpf, [287.05, 300., 101325., .001])
            Tstar = 2*T0/(1+g)
            Pstar = P0*(Tstar/T0)**(g/(g-1))
            mdot = throat*Pstar/(R*Tstar)*mp.sqrt(g*R*Tstar)
            self.assertTrue(result["verified"], result)
            self.assertLess(abs(P/(R*T)*V*throat*target_area/mdot - 1), mp.mpf('3e-13'))
            self.assertLess(abs((T+V*V/(2*g*R/(g-1)))/T0-1), mp.mpf('3e-15'))
            self.assertLess(abs(V/mp.sqrt(g*R*T)/m-1), mp.mpf('3e-15'))
            self.assertLess(abs(mp.mpf(result["mdot_choked_kg_s"])/mdot-1), mp.mpf('3e-15'))

    def test_analytic_two_branch_area_course(self):
        # For gamma=3, A/A*=(M+1/M)/2 exactly; the small root is the reciprocal.
        with mp.workdps(90):
            for a in (1., math.nextafter(1., 2.), 1.+2**-40, 1.25, 64., 2.**40):
                sup = mp.mpf(a)+mp.sqrt(mp.mpf(a)**2-1)
                for branch, root in (("subsonic", 1/sup), ("supersonic", sup)):
                    with self.subTest(area=a, branch=branch):
                        result = route(gamma=3, area_ratio=a, regime=branch)
                        self.assertEqual(result["mach"], float(root))
                        self.check_state(result, 3, a)

    def test_air_near_sonic_pressure_and_ordinary_points(self):
        with mp.workdps(90):
            g, a = mp.mpf(1.4), math.nextafter(1., 2.)
            # Equate stagnation-normalized mass flux on the two sides of its maximum.
            def flux(m):
                return m*(1+(g-1)*m*m/2)**(-(g+1)/(2*(g-1)))
            for sign, branch in ((-1, "subsonic"), (1, "supersonic")):
                estimate = 1+sign*mp.sqrt((g+1)*(mp.mpf(a)-1)/2)
                root = mp.findroot(lambda m: flux(1)-a*flux(m), estimate)
                result = route(area_ratio=a, regime=branch)
                self.assertEqual(result["mach"], float(root))
                self.check_state(result, 1.4, a)
            for pr in (math.nextafter(1., 2.), 2., 10.):
                root = mp.sqrt(2/(g-1)*(mp.mpf(pr)**((g-1)/g)-1))
                result = route(p0_over_p=pr)
                self.assertEqual(result["mach"], float(root))
                self.assertLessEqual(abs(result["P0_over_P"]-pr), 2*math.ulp(pr))
                self.check_state(result, 1.4, float(flux(1)/flux(root)))
            for mach in (.25, 1., 2.):
                result = route(mach=mach)
                self.check_state(result, 1.4, float(flux(1)/flux(mach)))

    def test_invalid_physical_domain_and_erased_branch(self):
        for spec in (dict(area_ratio=math.nextafter(1., 0.)), dict(area_ratio=1., regime='unknown'),
                     dict(area_ratio=1.6875, regime='transonic'), dict(mach=2., A_throat=-.001),
                     dict(mach=2., P0=-101325.), dict(mach=-1.), dict(mach=2., gamma=1.)):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                route(**spec)
        a = math.nextafter(1., 2.)
        for wrong in (1., math.nextafter(1., 0.)):
            with patch.object(wave6, "_nozzle_area_mach", return_value=wrong):
                result = route(area_ratio=a, regime="subsonic")
            self.assertLess(result["residual"], 1e-10)
            self.assertFalse(result["verified"])
