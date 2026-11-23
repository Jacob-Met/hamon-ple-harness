import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def cycle(kind, **observations):
    spec = dict(subkind=kind, r=4., gamma=1.5, T1=300., cv=750., P1=100000.)
    return run_router_conservation_cad_verifier({"mode":"thermo_cycles",
                                                "spec":dict(spec,**observations)})


class AirStandardHeatAddition(unittest.TestCase):
    def test_otto_cannot_discard_co_supplied_constant_volume_heat(self):
        r = cycle("otto",T3=900.,q_in=90000.)
        self.assertFalse(r["verified"],r)
        self.assertTrue(cycle("otto",T3=900.,q_in=225000.)["verified"])

    def test_diesel_cannot_discard_temperature_or_constant_pressure_heat(self):
        for extra in (dict(T3=750.), dict(q_in=90000.), dict(T3=900.,q_in=-1.)):
            with self.subTest(extra=extra):
                r = cycle("diesel",r_c=1.5,**extra)
                self.assertFalse(r["verified"],r)
        self.assertTrue(cycle("diesel",r_c=1.5,T3=900.,q_in=337500.)["verified"])
