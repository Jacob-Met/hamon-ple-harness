"""Balanced phase-current and complex-power replay for a passive shunt bank."""
import cmath
import math
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def bank(before, after, source_direction, target_direction, connection):
    spec = dict(P_kW=84., pf_initial=before, pf_target=after,
                V_ll_V=690., freq_Hz=60., initial_pf_direction=source_direction,
                target_pf_direction=target_direction, bank_connection=connection)
    return run_dual_verify_spec(dict(mode="domain", kind="pf_correction", spec=spec))


class PassiveBankCurrentOracle(unittest.TestCase):
    def test_signed_node_currents_recover_target_power_for_both_connections(self):
        for before, after, old, new in (
                (.6,.8,"lagging","lagging"), (.8,.6,"lagging","lagging"),
                (.6,.8,"leading","leading"), (.8,.6,"leading","leading"),
                (.8,.8,"lagging","leading"), (.8,.8,"leading","lagging"),
                (1.,.8,"lagging","leading"), (.8,1.,"lagging","lagging"),
                (.8,.8,"leading","leading"), (1.,1.,"lagging","leading")):
            for connection in ("wye", "delta"):
                with self.subTest(before=before, after=after, old=old, new=new,
                                  connection=connection):
                    result = bank(before, after, old, new, connection)
                    self.assertTrue(result["dual_verified"], result)
                    volts = [cmath.rect(690./math.sqrt(3), -i*math.tau/3)
                             for i in range(3)]
                    def demand(pf, direction):
                        angle = math.acos(pf) * (1 if direction=="lagging" else -1)
                        return cmath.rect(84000./pf, angle)
                    load = demand(before, old)
                    goal = demand(after, new)
                    delta_q = goal.imag-load.imag
                    expected = ("capacitor" if delta_q<0 else
                                "reactor" if delta_q>0 else "none")
                    self.assertEqual(result["bank_type"], expected, result)
                    omega = math.tau*60.
                    if expected == "capacitor":
                        c = result["bank_capacitance_per_leg_F"]
                        self.assertGreater(c, 0.)
                        admittance = 1j*omega*c
                    elif expected == "reactor":
                        inductance = result["bank_inductance_per_leg_H"]
                        self.assertGreater(inductance, 0.)
                        admittance = 1/(1j*omega*inductance)
                    else:
                        admittance = 0j
                    if connection == "wye":
                        lines = [admittance*v for v in volts]
                    else:
                        branches = [admittance*(v-volts[(i+1)%3])
                                    for i,v in enumerate(volts)]
                        lines = [x-branches[(i-1)%3] for i,x in enumerate(branches)]
                    bank_power = sum(v*i.conjugate() for v,i in zip(volts,lines))
                    self.assertLess(abs(sum(lines)), 2e-12)
                    self.assertLess(abs(bank_power - 1j*delta_q), 2e-9)
                    self.assertAlmostEqual(result["Q_shunt_kvar"], bank_power.imag/1000, places=9)
                    self.assertAlmostEqual(result["Qc_required_kvar"], -bank_power.imag/1000, places=9)
                    source = [i+(load/(3*v)).conjugate() for v,i in zip(volts,lines)]
                    supply = sum(v*i.conjugate() for v,i in zip(volts,source))
                    self.assertLess(abs(supply-goal), 2e-9)
                    self.assertAlmostEqual(supply.real/abs(supply), after, places=13)
