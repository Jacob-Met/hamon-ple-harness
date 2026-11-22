"""Requested duct outlet versus a reachable sonic section in the fixed-inlet model."""
import math
from decimal import Decimal, localcontext

import unittest

from hamon_wave6_physics_verifiers import (
    dispatch_wave6_verifier,
    prune_deterministic_fast_path,
)


def reference(mach, mode, ratio):
    # Use the actual binary inputs at higher precision so a nearly sonic Mach
    # is not replaced by a decimal with a different distance from 1.
    with localcontext() as ctx:
        ctx.prec = 65
        m, gamma = Decimal.from_float(mach), Decimal.from_float(1.4)
        cp, t0 = Decimal('1004.5'), Decimal(300)
        m2 = m*m
        if mode == 'fanno':
            f, diameter = Decimal.from_float(.005), Decimal.from_float(.1)
            x = 2*(1-m2)/((gamma+1)*m2)
            critical = diameter*(gamma+1)*(x-(1+x).ln())/(8*gamma*f)
            change = float(Decimal(str(ratio))*critical)
            spec = {'subkind': mode, 'M1':mach, 'gamma':1.4, 'f':.005, 'D_h':.1, 'L':change}
        else:
            gap = ((1-m2)/(1+gamma*m2))**2
            critical = cp*t0*gap/(1-gap)
            change = float(Decimal(str(ratio))*critical)
            spec = {'subkind':mode, 'M1':mach, 'gamma':1.4, 'T01':300., 'cp':1004.5, 'q_J_kg':change}
    return spec, float(critical)


def check_prescribed_duct(mach, mode):
    def solve(ratio):
        spec, critical = reference(mach, mode, ratio)
        return dispatch_wave6_verifier('compressible_acoustics',spec), critical

    at_inlet, _ = solve(0)
    assert at_inlet['verified'] and math.isclose(at_inlet['M2'], mach, abs_tol=2e-14)
    before, _ = solve(.5)
    assert before['verified'] and before['requested_state_feasible'] and not before['choked']
    assert mach < before['M2'] < 1.
    sonic, _ = solve(1)
    assert sonic['verified'] and sonic['requested_state_feasible'] and sonic['choked']
    assert sonic['M2'] == 1.
    beyond, critical = solve(2)
    assert not beyond['verified'] and not beyond['requested_state_feasible'] and beyond['choked']
    assert beyond['residual'] > 0
    assert not ({'P2_over_P1', 'T2_over_T1', 'M2'} & beyond.keys())
    diagnostic = beyond['reachable_sonic_state']
    assert diagnostic['M'] == 1.
    assert math.isclose(diagnostic['P_over_P1'], sonic['P2_over_P1'], rel_tol=5e-15)
    input_key, limit_key, sonic_key = (
        ('requested_length_m', 'L_choke_m', 'length_m') if mode == 'fanno'
        else ('requested_heat_J_kg', 'q_choke_J_kg', 'heat_J_kg')
    )
    assert math.isclose(beyond[input_key], 2*critical, rel_tol=2e-15)
    assert math.isclose(beyond[limit_key], critical, rel_tol=7e-15)
    assert diagnostic[sonic_key] == beyond[limit_key]
    decision = prune_deterministic_fast_path('medium', 5., presolve_verified=beyond['verified'],
                                             domain='compressible_acoustics', residual=beyond['residual'])
    assert not decision['fast_path_pruned']


class SubsonicDuctEndpointTests(unittest.TestCase):
    def test_limits_and_choking_diagnostic(self):
        for mach in (.5, .9999, .9999999):
            for mode in ('fanno', 'rayleigh'):
                with self.subTest(mach=mach, mode=mode):
                    check_prescribed_duct(mach, mode)

    def test_rayleigh_cooling_keeps_the_subsonic_solution(self):
        spec, _ = reference(.5, 'rayleigh', 0)
        spec['q_J_kg'] = -30000.
        result = dispatch_wave6_verifier('compressible_acoustics', spec)
        assert result['verified'] and not result['choked'] and 0 < result['M2'] < .5
        def t0_ratio(mach):
            return 2*2.4*mach**2*(1+.2*mach**2)/(1+1.4*mach**2)**2
        assert math.isclose(t0_ratio(result['M2'])/t0_ratio(.5), 1-30000/(1004.5*300), rel_tol=2e-14)
