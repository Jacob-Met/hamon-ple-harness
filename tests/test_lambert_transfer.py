"""Independent conic and Cartesian ODE controls for a prescribed-time transfer."""
import json
import math
import random
import unittest

import numpy as np
from mpmath import mp
from scipy.integrate import solve_ivp

from hamon_lambert import solve_lambert_transfer
from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier, solve_lagrangian_orbital


class LambertTransferTests(unittest.TestCase):
    def solve(self, **changes):
        spec = dict(mu=1, r1_m=[1,0,0], r2_m=[0,1,0], tof_s=math.pi/2)
        spec.update(changes)
        result = solve_lambert_transfer(spec)
        json.dumps(result, allow_nan=False)
        return result

    def qualified(self, result):
        self.assertTrue(result['verified'], result)
        self.assertTrue(result['endpoint_verified'], result)
        self.assertLess(result['residual'], 1e-10)
        return result

    def test_circular_arcs_both_directions_and_near_antipodal(self):
        for theta in [.05, math.pi/3, math.pi-1e-7, math.pi+1e-7, 2*math.pi-.01]:
            for sense in [1, -1]:
                with self.subTest(theta=theta, sense=sense):
                    result = self.qualified(self.solve(r2_m=[math.cos(theta),sense*math.sin(theta),0],
                        tof_s=theta, transfer_path='short' if theta < math.pi else 'long'))
                    np.testing.assert_allclose(result['v1_m_s'], [0,sense,0], atol=2e-11, rtol=2e-11)
                    np.testing.assert_allclose(result['v2_m_s'], [-math.sin(theta),sense*math.cos(theta),0], atol=2e-11, rtol=2e-11)
                    self.assertAlmostEqual(result['transfer_angle_rad'], theta, places=12)

    def test_tiny_arc_respects_the_actual_rounded_endpoint(self):
        # Rounded cos(1e-6) is not exactly a circular endpoint. Its 4.45e-17
        # radial displacement changes the required initial radial speed by
        # 4.45e-11. Independently invert the Cartesian Taylor equations through
        # order four; the omitted speed term is below 2e-32 for these inputs.
        ctx = mp.clone()
        ctx.dps = 75
        t = ctx.mpf(1e-6)
        rx = ctx.mpf(math.cos(float(t)))
        expected = (rx-1+t*t/2-t**4/24)/(t+t**3/3)
        for sign in [1,-1]:
            result = self.qualified(self.solve(r2_m=[float(rx),sign*math.sin(float(t)),0],tof_s=float(t)))
            self.assertLess(abs(ctx.mpf(result['v1_m_s'][0])-expected), ctx.mpf('1e-22'))
            self.assertGreater(abs(result['v1_m_s'][0]), 4e-11)
            self.assertEqual(result['v1_m_s'][1], sign)

    def test_analytic_elliptic_arcs(self):
        for eccentricity in [.05, .7, .99]:
            for anomaly in [.1, 1., 2.5, 4.]:
                with self.subTest(e=eccentricity, E=anomaly):
                    e, E = eccentricity, anomaly
                    endpoint = [math.cos(E)-e, math.sqrt(1-e*e)*math.sin(E), 0]
                    result = self.qualified(self.solve(r1_m=[1-e,0,0], r2_m=endpoint,
                        tof_s=E-e*math.sin(E), transfer_path='short' if E < math.pi else 'long'))
                    np.testing.assert_allclose(result['v1_m_s'], [0,math.sqrt((1+e)/(1-e)),0], atol=2e-10, rtol=2e-11)
                    expected = [-math.sin(E)/(1-e*math.cos(E)), math.sqrt(1-e*e)*math.cos(E)/(1-e*math.cos(E)),0]
                    np.testing.assert_allclose(result['v2_m_s'], expected, atol=2e-10, rtol=2e-11)

    def test_analytic_hyperbolic_arcs(self):
        for e in [1.05, 1.4, 2., 5.]:
            for H in [.1, .8, 2.]:
                with self.subTest(e=e, H=H):
                    den = e*math.cosh(H)-1
                    result = self.qualified(self.solve(r1_m=[e-1,0,0],
                        r2_m=[e-math.cosh(H),math.sqrt(e*e-1)*math.sinh(H),0],
                        tof_s=e*math.sinh(H)-H))
                    np.testing.assert_allclose(result['v1_m_s'], [0,math.sqrt((e+1)/(e-1)),0], atol=2e-10, rtol=2e-11)
                    np.testing.assert_allclose(result['v2_m_s'], [-math.sinh(H)/den,math.sqrt(e*e-1)*math.cosh(H)/den,0], atol=2e-10, rtol=2e-11)

    def test_parabolic_barker_equation_and_both_adjacent_float_times(self):
        for D in [.01, .3, 1., 3.]:
            time = math.sqrt(2)*(D+D**3/3)
            for dt in [math.nextafter(time,0), time, math.nextafter(time,math.inf)]:
                with self.subTest(D=D, dt=dt):
                    result = self.qualified(self.solve(r2_m=[1-D*D,2*D,0], tof_s=dt))
                    np.testing.assert_allclose(result['v1_m_s'], [0,math.sqrt(2),0], atol=3e-11, rtol=3e-11)
                    expected = [-math.sqrt(2)*D/(1+D*D), math.sqrt(2)/(1+D*D),0]
                    np.testing.assert_allclose(result['v2_m_s'], expected, atol=3e-11, rtol=3e-11)

    def test_antipodal_requires_a_plane_and_preserves_its_direction(self):
        self.assertEqual(self.solve(r2_m=[-1,0,0], tof_s=math.pi)['error'], 'antipodal_transfer_requires_plane_normal')
        for normal in ([0,0,1], [0,0,-1], [0,2,0]):
            result = self.qualified(self.solve(r2_m=[-1,0,0], tof_s=math.pi, plane_normal=normal))
            self.assertEqual(result['transfer_angle_rad'], math.pi)
            expected = np.cross(np.asarray(normal)/np.linalg.norm(normal), [1,0,0])
            np.testing.assert_allclose(result['v1_m_s'], expected, atol=1e-13)
        self.assertFalse(self.solve(r2_m=[-1,0,0], tof_s=math.pi, plane_normal=[1,0,1])['verified'])

    def test_rotations_and_consistent_unit_rescaling(self):
        rotate = np.array([[.36,.48,-.8],[-.8,.6,0],[.48,.64,.6]])
        r1, r2 = rotate@np.array([1.,0,0]), rotate@np.array([0.,1,0])
        for length, time in [(1.,1.), (2.**100,2.**70), (2.**-100,2.**-70), (7e6,900.)]:
            with self.subTest(length=length, time=time):
                result = self.qualified(self.solve(mu=length**3/time**2,
                    r1_m=(r1*length).tolist(), r2_m=(r2*length).tolist(), tof_s=math.pi/2*time))
                np.testing.assert_allclose(np.asarray(result['v1_m_s'])*time/length, rotate@np.array([0.,1,0]), atol=1e-12)

    def test_independent_cartesian_dop853_replay(self):
        rng = random.Random(6054)
        worst_position = worst_velocity = 0.
        for i in range(24):
            radius = rng.uniform(.4,3.)
            angle = rng.uniform(.5,2.6)
            dt = rng.uniform(1.8,8.)
            target = np.array([radius*math.cos(angle),radius*math.sin(angle),0])
            result = self.qualified(self.solve(r2_m=target.tolist(), tof_s=dt,
                transfer_path='short' if i%2 else 'long'))
            start = [1.,0,0,*result['v1_m_s']]
            def derivative(_t,y):
                return np.r_[y[3:],-y[:3]/np.linalg.norm(y[:3])**3]
            trajectory = solve_ivp(derivative, [0,dt], start, method='DOP853', rtol=2e-12, atol=2e-13)
            self.assertTrue(trajectory.success)
            pos = np.linalg.norm(trajectory.y[:3,-1]-target)/radius
            vel = np.linalg.norm(trajectory.y[3:,-1]-result['v2_m_s'])/max(1,np.linalg.norm(result['v2_m_s']))
            worst_position, worst_velocity = max(pos,worst_position), max(vel,worst_velocity)
            self.assertLess(max(pos,vel), 2e-8, (i,result,trajectory.message,pos,vel))
        print(json.dumps({'lambert_independent_trajectories':24,'worst_position_relative':worst_position,'worst_velocity_relative':worst_velocity}))

    def test_wrong_velocity_claim_cannot_pass_by_energy_conservation(self):
        truth = self.qualified(self.solve())
        accepted = self.solve(proposed_v1_m_s=truth['v1_m_s'], proposed_v2_m_s=truth['v2_m_s'])
        self.assertTrue(accepted['verified'])
        wrong = self.solve(proposed_v1_m_s=[-x for x in truth['v1_m_s']], proposed_v2_m_s=[-x for x in truth['v2_m_s']])
        self.assertFalse(wrong['verified'])
        self.assertTrue(wrong['endpoint_verified'])
        self.assertTrue(all(not x['passed'] for x in wrong['claims'].values()))

    def test_invalid_or_unsupported_models_do_not_succeed(self):
        for changes in [dict(mu=0),dict(tof_s=0),dict(tof_s=float('nan')),dict(mu=True),
            dict(r1_m=[0,0,0]),dict(r1_m=[1,0]),dict(r2_m=[1,0,0]),dict(r2_m=[2,0,0]),
            dict(revolutions=1),dict(revolutions=False),dict(transfer_path='prograde'),
            dict(thrust_N=1),dict(J2=.001),dict(r1=1),dict(plane_normal=[0,0,1]),
            dict(proposed_v1_m_s=[0,1,float('inf')])]:
            with self.subTest(changes=changes):
                self.assertFalse(self.solve(**changes)['verified'])

    def test_registered_python_family_and_hohmann_mixed_schema(self):
        spec = dict(subkind='lambert_transfer',mu=1,r1_m=[1,0,0],r2_m=[0,1,0],tof_s=math.pi/2)
        self.qualified(dispatch_wave7_verifier('lambert',spec))
        self.qualified(solve_lagrangian_orbital(spec))
        legacy = dict(subkind='orbital_transfer',mu=1,r1=1,r2=1)
        self.assertTrue(solve_lagrangian_orbital(legacy)['verified'])
        mixed = solve_lagrangian_orbital({**legacy,**{k:v for k,v in spec.items() if k!='subkind'}})
        self.assertFalse(mixed['verified'])
        self.assertEqual(mixed['error'],'lambert_inputs_require_lambert_transfer')


if __name__ == '__main__':
    unittest.main()
