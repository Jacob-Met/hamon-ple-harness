"""Independent complex KVL in the receiving-bus phase frame."""
import math
import unittest
from mpmath import mp
from hamon_dual_verifier import run_dual_verify_spec


class FeederPortOracle(unittest.TestCase):
    def check(self, volt, amps, pf, resistance, reactance, distance, direction, terminal, branch=None):
        c=mp.clone()
        c.dps=130
        spec=dict(V_ll_V=volt,I_A=amps,pf=pf,length_m=distance,
                  R_ohm_per_km=resistance,X_ohm_per_km=reactance,
                  power_factor_direction=direction,voltage_reference=terminal)
        if branch:
            spec["receiving_voltage_branch"]=branch
        out=run_dual_verify_spec(dict(mode="domain",kind="three_phase_vdrop",spec=spec))
        voltage=c.mpf(volt)
        current=c.mpf(amps)*c.mpc(pf, (1 if direction=="leading" else -1)*c.sqrt(1-c.mpf(pf)**2))
        line=c.mpc(resistance,reactance)*c.mpf(distance)/1000
        # Physical phase voltages and phase current; multiply by sqrt(3)
        # only after solving the complex circuit to compare line magnitudes.
        zi=current*line
        scale=c.sqrt(3)
        if terminal=="receiving":
            vr=voltage/scale
            vs=c.mpc(vr)+zi
            possible=[vr]
        else:
            rad=(voltage/scale)**2-zi.imag**2
            possible=sorted(x for x in ([-zi.real+c.sqrt(rad),-zi.real-c.sqrt(rad)] if rad>=0 else []) if x>0)
            if not possible:
                self.assertFalse(out["feasible"],out)
                self.assertFalse(out["dual_verified"],out)
                return
            vr=min(possible) if branch=="low" else max(possible)
            vs=c.mpc(vr)+zi
        self.assertTrue(out["dual_verified"],out)
        self.assertEqual(len(out["positive_receiving_voltages_ll_V"]),len(possible))
        refs=[(g,p*scale) for g,p in zip(out["positive_receiving_voltages_ll_V"],possible)]
        drop=scale*(abs(vs)-vr)
        refs.extend([(out["V_receiving_ll_V"],scale*vr),(out["V_sending_ll_V"],scale*abs(vs)),
                     (out["V_sending_phasor_ll_V"][0],scale*vs.real),
                     (out["V_sending_phasor_ll_V"][1],scale*vs.imag),
                     (out["vdrop_ll_V"],drop), (out["vdrop_pct"],100*drop/voltage)])
        for got,ref in refs:
            if ref:
                self.assertLess(abs(c.mpf(got)/ref-1),c.mpf("3e-14"))
            else:
                self.assertEqual(got,0.)
        self.assertEqual(out["passes_3pct"],100*drop/voltage<=3)

    def test_signed_current_and_terminal_references(self):
        for terminal in ("receiving","sending"):
            for direction in ("leading","lagging"):
                for values in [(400,100,1,0,1,1000),(96,17,.875,.35,.64,700),
                               (117,8,.625,.7,-.45,850)]:
                    with self.subTest(terminal=terminal,direction=direction,values=values):
                        self.check(*values,direction,terminal)

    def test_both_positive_roots_and_collapse_neighbors(self):
        # With pf=.5, R=0 and X=-8, the real line drop is -12 V and
        # Im(Vs)^2=48 V² exactly. Prescribing |Vs|=13 yields Vr=1 or23 V.
        for branch in ("low","high"):
            self.check(13,1,.5,0,-8,1000,"lagging","sending",branch)
            self.check(1,1,0,.125,2,1000,"leading","sending",branch)
        contact=math.sqrt(3.)
        for volt in (math.nextafter(contact,0.),contact,math.nextafter(contact,math.inf)):
            for branch in ("low","high"):
                with self.subTest(volt=volt,branch=branch):
                    self.check(volt,1,0,1,4,1000,"leading","sending",branch)
        # This exact real double root is negative: real-solution existence
        # alone does not supply an energized positive receiving bus.
        self.check(3,1,.5,2,0,1000,"leading","sending")
