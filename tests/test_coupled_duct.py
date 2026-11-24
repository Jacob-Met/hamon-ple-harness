"""Independent physical-state integration for the coupled heat/friction duct."""
import math
import random
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from hamon_wave6_physics_verifiers import solve_compressible_gas_acoustics as solve


def request(**changes):
    return dict(subkind="fanno_rayleigh", M1=.5, T01=315., P1=101325.,
                L=.5, D_h=.1, f=.01, q_J_kg=40000., **changes)


def case(**changes):
    params = request()
    params.update(changes)
    return params


def physical_oracle(p):
    """Integrate pressure and velocity using the original two balance rows."""
    gamma, gas = p.get('gamma', 1.4), p.get('R', 287.05)
    cp = gamma*gas/(gamma-1)
    tin = p['T01']/(1+(gamma-1)*p['M1']**2/2)
    uin = p['M1']*math.sqrt(gamma*gas*tin)
    mass = p['P1']*uin/(gas*tin)
    f = p['f']*(4 if p.get('friction_factor_convention','fanning') == 'fanning' else 1)
    k = f/(2*p['D_h'])
    heat_rate = p['q_J_kg']/p['L']
    def rates(x, state):
        pressure, velocity = state
        matrix = [[1., mass], [cp*velocity/(mass*gas), cp*pressure/(mass*gas)+velocity]]
        return np.linalg.solve(matrix, [-k*mass*velocity, heat_rate])
    sol = solve_ivp(rates,[0.,p['L']],[p['P1'],uin], method='DOP853',
                    rtol=2e-12,atol=[1e-7,1e-10],max_step=p['L']/32)
    if not sol.success:
        raise AssertionError(sol.message)
    pressure, velocity = sol.y[:,-1]
    temp = pressure*velocity/(mass*gas)
    mach = velocity/math.sqrt(gamma*gas*temp)
    return dict(P2_Pa=pressure,u2_m_s=velocity,T2_K=temp,M2=mach)


class CoupledDuctTests(unittest.TestCase):
    def test_mixed_single_effect_requests_do_not_silently_drop_physics(self):
        for subkind in ('fanno','rayleigh'):
            result = solve(case(subkind=subkind))
            self.assertFalse(result['verified'], result)
            self.assertIn('fanno_rayleigh',result['error'])
        result = solve(case())
        self.assertTrue(result['verified'], result)
        self.assertAlmostEqual(result['M2'], .6069166972327382, places=13)
        self.assertGreater(result['M2'], .5270569917822787)

    def test_independent_pressure_velocity_balances_both_branches_and_heat_signs(self):
        rng = random.Random(6054)
        errors = []
        for i in range(32):
            p = case(M1=rng.uniform(.15,.65) if i%2 == 0 else rng.uniform(1.6,3.8),
                     gamma=rng.uniform(1.2,1.7),R=rng.uniform(180.,430.),
                     T01=rng.uniform(290.,900.),P1=rng.uniform(50000.,400000.),
                     L=rng.uniform(.05,.2),D_h=rng.uniform(.1,.5),
                     f=rng.uniform(.001,.012),q_J_kg=rng.uniform(-12000.,12000.))
            with self.subTest(i=i,p=p):
                result = solve(p)
                self.assertTrue(result['verified'],result)
                reference = physical_oracle(p)
                for key,value in reference.items():
                    error = abs(result[key]/value-1)
                    errors.append(error)
                    self.assertLess(error,2e-10,(key,result,reference))
        print('coupled_duct_independent_32_worst_relative_error',max(errors))

    def test_combined_choking_precedes_both_separate_limits(self):
        p = case(M1=2.)
        result = solve(p)
        self.assertFalse(result['verified'],result)
        self.assertFalse(result['requested_state_feasible'])
        self.assertTrue(result['choked'])
        self.assertNotIn('M2',result)
        self.assertAlmostEqual(result['L_choke_m'],.42522686591353687,places=12)
        self.assertEqual(result['reachable_sonic_state']['M'],1.)
        for isolated in (dict(p,f=0.),dict(p,q_J_kg=0.)):
            self.assertTrue(solve(isolated)['verified'])
        # An ordinary physical integration reaches the state just before the
        # independently reported sonic limit; a longer requested duct is refused.
        short = dict(p,L=.99*result['L_choke_m'],q_J_kg=.99*result['q_choke_at_fixed_heat_rate_J_kg'])
        expected = physical_oracle(short)
        after = solve(short)
        self.assertTrue(after['verified'],after)
        self.assertLess(abs(after['M2']/expected['M2']-1),2e-10)

    def test_uniform_forcing_is_invariant_under_physical_partition(self):
        for mach,heat in ((.5,40000.),(.5,-40000.),(2.,-40000.),(2.,15000.)):
            p = case(M1=mach,q_J_kg=heat)
            whole = solve(p)
            first = solve(dict(p,L=p['L']/2,q_J_kg=heat/2))
            self.assertTrue(whole['verified'],whole)
            self.assertTrue(first['verified'],first)
            second = solve(dict(p,L=p['L']/2,q_J_kg=heat/2,M1=first['M2'],
                                T01=first['T02_K'],P1=first['P2_Pa']))
            self.assertTrue(second['verified'],second)
            for key in ('M2','P2_Pa','T2_K','T02_K','u2_m_s'):
                self.assertLess(abs(second[key]/whole[key]-1),3e-13,(key,whole,second))

    def test_pure_fanno_and_rayleigh_limits_match_existing_analytic_solvers(self):
        for mach in (.2,.5,1.8,3.):
            for kind in ('fanno','rayleigh'):
                p = case(M1=mach,L=.05,q_J_kg=0. if kind=='fanno' else 10000.,
                         f=.01 if kind=='fanno' else 0.)
                actual = solve(p)
                legacy = solve(dict(p,subkind=kind))
                self.assertTrue(actual['verified'],actual)
                self.assertTrue(legacy['verified'],legacy)
                for key in ('M2','P2_over_P1'):
                    self.assertLess(abs(actual[key]/legacy[key]-1),2e-14,(actual,legacy))

    def test_darcy_convention_and_pressure_scale(self):
        a = solve(case())
        b = solve(case(f=.04,friction_factor_convention='darcy',P1=202650.))
        self.assertTrue(b['verified'],b)
        self.assertEqual(a['M2'],b['M2'])
        self.assertEqual(2*a['P2_Pa'],b['P2_Pa'])
        self.assertEqual(2*a['mass_flux_kg_m2_s'],b['mass_flux_kg_m2_s'])

    def test_cooling_separatrix_and_constant_velocity_state(self):
        # gamma=2, R=1, cp=2, M=.5, T01=9/8 => T1=1,
        # u1^2=.5, k=.5 and q=-.5 exactly balance acceleration.
        p = case(gamma=2.,R=1.,cp=2.,M1=.5,T01=1.125,L=1.,D_h=1.,
                 f=1.,friction_factor_convention='darcy',q_J_kg=-.5)
        result = solve(p)
        self.assertTrue(result['verified'],result)
        self.assertEqual(result['T2_K'],.75)
        self.assertAlmostEqual(result['u2_m_s'],math.sqrt(.5))
        for heat in (-.5-1e-8,-.5+1e-8,math.nextafter(-.5,-math.inf),math.nextafter(-.5,0.)):
            varied = dict(p,q_J_kg=heat)
            answer = solve(varied)
            self.assertTrue(answer['verified'],answer)
            ref = physical_oracle(varied)
            self.assertLess(abs(answer['M2']/ref['M2']-1),2e-10)

    def test_invalid_and_unreachable_models_do_not_return_verified_outlets(self):
        changes = [dict(M1=True),dict(T01=float('nan')),dict(f=-.1),dict(D_h=0.),
                   dict(L=0.,q_J_kg=1.),dict(branch='supersonic'),dict(gamma=1.),
                   dict(gamma=1.4,R=287.05,cp=500.),dict(area_ratio=2.),
                   dict(wall_temperature_K=900.),dict(heat_profile=[1,2]),
                   dict(q_J_kg=-1e9),dict(M1=1.)]
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(solve(case(**change))['verified'])

    def test_zero_forcing_returns_the_inlet_and_sonic_identity_is_defined(self):
        for mach in (.1,.5,1.,2.,10.):
            result = solve(case(M1=mach,f=0.,q_J_kg=0.))
            self.assertTrue(result['verified'],result)
            self.assertEqual(result['M2'],mach)
            self.assertEqual(result['P2_Pa'],101325.)


if __name__ == '__main__':
    unittest.main()
