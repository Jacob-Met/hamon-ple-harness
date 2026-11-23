"""2017 ordinary branch-circuit conductor feasibility at fixed noncontinuous load."""
import math
import unittest
from fractions import Fraction

from hamon_dual_verifier import run_dual_verify_spec


class AmpacityEnvelopeOracle(unittest.TestCase):
    def test_continuous_load_ceiling_for_terminal_and_installation_limits(self):
        for table, terminal, correction, bundling, other, listed in (
                (100,95,1.125,.75,20,False), (125,125,.875,.75,10,False),
                (160,125,1.,.625,0,False), (80,100,1.25,1.,5,False),
                (160,125,1.,.625,0,True), (80,75,1.25,1.,15,True),
                (2**56,2**56,1.,1.,0,True)):
            load_factor = Fraction(1) if listed else Fraction(5,4)
            factor = Fraction(correction)*Fraction(bundling)
            thermal_limit = table*factor-other
            terminal_limit = (min(terminal,table)-other)/load_factor
            boundary = min(thermal_limit,terminal_limit)
            # Some computed ceilings are fractional; these examples are exact in binary.
            at=float(boundary)
            self.assertEqual(Fraction(at), boundary)
            for continuous in (0,math.nextafter(at,-math.inf),at,math.nextafter(at,math.inf)):
                with self.subTest(table=table,terminal=terminal,correction=correction,
                                  bundling=bundling,listed=listed,continuous=continuous):
                    spec=dict(base_ampacity_A=table,termination_ampacity_A=terminal,
                              temp_factor=correction,bundling_factor=bundling,
                              I_noncontinuous_A=other,I_continuous_A=continuous,
                              listed_100_percent_assembly=listed)
                    ans=run_dual_verify_spec(dict(mode="domain",kind="ampacity_derating",spec=spec))
                    served=Fraction(continuous)
                    self.assertEqual(ans["ampacity_ok"],served<=boundary)
                    self.assertEqual(ans["unadjusted_sizing_ok"],served<=terminal_limit)
                    self.assertEqual(ans["adjusted_load_ok"],served<=thermal_limit)
                    adjusted_current=other+load_factor*served
                    actual_current=other+served
                    needed=max(adjusted_current,actual_current/factor)
                    self.assertEqual(ans["I_required_table_A"],float(needed))
                    self.assertEqual(ans["I_installed_allowable_A"],float(min(terminal,table*factor)))

    def test_integer_ampere_neighbors_and_explicit_assembly_listing(self):
        base=2**56
        for current, expected in ((base-1,True),(base,True),(base+1,False)):
            spec=dict(base_ampacity_A=base,termination_ampacity_A=base,
                      temp_factor=1,bundling_factor=1,I_continuous_A=current,
                      listed_100_percent_assembly=True)
            out=run_dual_verify_spec(dict(mode="domain",kind="ampacity_derating",spec=spec))
            self.assertEqual(out["ampacity_ok"],expected)
        for flag in (1,"true","listed"):
            spec["listed_100_percent_assembly"]=flag
            with self.assertRaises(ValueError):
                run_dual_verify_spec(dict(mode="domain",kind="ampacity_derating",spec=spec))
