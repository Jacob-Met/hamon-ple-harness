"""Phase, transverse impulse and work of a symmetric Euler top."""
import math
import unittest

from mpmath import mp
from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


def solve(i, w, torque, t, steps):
    return dispatch_wave7_verifier("lagrangian_orbital", {
        "subkind": "euler_rigid_body", "I": i, "omega0": w, "torque": torque,
        "t_final": t, "n_steps": steps})


class SymmetricTop(unittest.TestCase):
    def test_fast_free_spin_preserves_transverse_phase_at_every_symmetry_axis(self):
        for axis in range(3):
            a,b = (axis+1)%3,(axis+2)%3
            for ia,spin,t,steps in [(2,400,10,2000),(.5,8,1,1)]:
                moments=[1.,1.,1.];moments[axis]=ia
                w=[0.,0.,0.];w[axis]=spin;w[a]=1e-10
                phase=(ia-1)*spin*t
                with self.subTest(axis=axis,phase=phase):
                    r=solve(moments,w,[0,0,0],t,steps)
                    self.assertTrue(r["verified"],r)
                    u=r["omega_final_rad_s"]
                    self.assertEqual(u[axis],spin)
                    self.assertAlmostEqual(u[a]/w[a],math.cos(phase),places=13)
                    self.assertAlmostEqual(u[b]/w[a],math.sin(phase),places=13)
                    self.assertEqual(r["work_done_J"],0.)

    def test_axial_spin_reversal_has_zero_net_phase_and_work(self):
        r=solve([1,1,2],[1.,-.5,3.],[0,0,-12],1.,1)
        self.assertTrue(r["verified"],r)
        self.assertEqual(r["omega_final_rad_s"],[1.,-.5,-3.])
        self.assertEqual(r["work_done_J"],0.)
        self.assertEqual(r["torque_half_H2_change_Nms2"],0.)

    def test_body_torque_original_series_and_independent_work_integral(self):
        # Direct power series for the coupled Euler ODE, including its original
        # torque work integral. This uses neither Fresnel functions nor RK4.
        ctx=mp.clone();ctx.dps=75
        for axis in range(3):
            a,b=(axis+1)%3,(axis+2)%3
            for axial_torque in (-5.,0.,4.):
                it,iz,t=2.,3.,1.25
                I=[it]*3;I[axis]=iz
                w=[0.]*3;w[a],w[b],w[axis]=-.5,1.,.75
                tau=[0.]*3;tau[a],tau[b],tau[axis]=1.5,-.25,axial_torque
                alpha=(iz-it)/it*w[axis]*t
                sweep=ctx.mpf(iz-it)/it*tau[axis]/iz*t*t
                d=ctx.mpc(tau[a],tau[b])*t/it
                terms=[ctx.mpc(w[a],w[b])]
                for n in range(200):
                    terms.append((ctx.j*alpha*terms[n]+(ctx.j*sweep*terms[n-1] if n else d))/(n+1))
                z=sum(terms);mean=sum(c/(n+1) for n,c in enumerate(terms))
                axial=ctx.mpf(w[axis])+ctx.mpf(tau[axis])*t/iz
                wt=t*(tau[a]*mean.real+tau[b]*mean.imag)
                wz=t*tau[axis]*(w[axis]+axial)/2
                with self.subTest(axis=axis,axial_torque=axial_torque):
                    r=solve(I,w,tau,t,1)
                    self.assertTrue(r["verified"],r)
                    actual=r["omega_final_rad_s"]
                    for got,ref in [(actual[a],z.real),(actual[b],z.imag),(actual[axis],axial),
                                    (r["work_done_J"],wt+wz),(r["torque_half_H2_change_Nms2"],it*wt+iz*wz)]:
                        self.assertAlmostEqual(got,float(ref),places=13)

    def test_signed_changes_around_torque_balanced_top(self):
        ctx = mp.clone()
        ctx.dps = 220
        for epsilon in (1e-60, -1e-150):
            for lateral, axial in ((epsilon, 0.), (0., epsilon), (epsilon, epsilon)):
                with self.subTest(lateral=lateral, axial=axial):
                    r = solve([1,1,2], [0,1,1], [1,lateral,axial], 1., 1)
                    self.assertTrue(r["verified"], r)
                    # Direct Euler-series integration about the initial state.
                    # The recurrence integrates angular velocity and torque power.
                    terms = [ctx.j, ctx.j*ctx.mpf(lateral)]
                    chirp = ctx.mpf(axial)/2
                    for n in range(1, 160):
                        terms.append(ctx.j*(terms[n]+chirp*terms[n-1])/(n+1))
                    z = ctx.fsum(terms)
                    mean = ctx.fsum(c/(n+1) for n,c in enumerate(terms))
                    wz = ctx.mpf(axial)*(1+ctx.mpf(axial)/4)
                    wt = mean.real+ctx.mpf(lateral)*mean.imag
                    for got, ref in zip(r["omega_final_rad_s"][:2] +
                                        [r["work_done_J"], r["torque_half_H2_change_Nms2"]],
                                        [z.real, z.imag, wt+wz, wt+2*wz]):
                        self.assertEqual(got, float(ref))

    def test_second_order_coordinate_at_nearly_cancelled_endpoint(self):
        ctx = mp.clone()
        ctx.dps = 220
        angle = 1e-80
        a = ctx.j*ctx.mpf(angle)
        endpoint = -ctx.exp(a)+ctx.expm1(a)/a
        r = solve([1,1,2], [-1,0,angle], [1,0,0], 1, 1)
        self.assertTrue(r["verified"], r)
        self.assertEqual(r["omega_final_rad_s"][:2], [float(endpoint.real), float(endpoint.imag)])

    def test_false_free_triaxial_wobble_cannot_borrow_axial_spin_scale(self):
        r=solve([1,1+2**-24,1.5],[1e-10,0.,8.],[0,0,0],1.,1)
        self.assertFalse(r["verified"],r)
        self.assertGreater(r["solver_residual"],1e-3)
