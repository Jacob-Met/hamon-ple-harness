"""Normal-incidence acoustic interface checks against the two boundary equations."""
import math
import unittest
from decimal import Decimal, localcontext

from hamon_wave6_physics_verifiers import dispatch_wave6_verifier


def solve(**params):
    return dispatch_wave6_verifier("compressible_acoustics", {"subkind": "acoustics", **params})


def boundary_oracle(rho1, c1, rho2, c2):
    # Solve x + 1 = y, (1 - x)/Z1 = y/Z2 by eliminating reflected pressure.
    with localcontext() as ctx:
        ctx.prec = 190
        z1 = Decimal.from_float(float(rho1)) * Decimal.from_float(float(c1))
        z2 = Decimal.from_float(float(rho2)) * Decimal.from_float(float(c2))
        conductance1, conductance2 = 1 / z1, 1 / z2
        transmitted = 2 * conductance1 / (conductance1 + conductance2)
        # Recover reflection directly for extreme ratios where y - 1 cancels.
        reflected = (conductance1 - conductance2) / (conductance1 + conductance2)
        power = transmitted**2 * conductance2 / conductance1
        loss = -10 * power.log10()
        return [float(x) for x in (z1, z2, reflected, transmitted, power, loss)]


class PassiveAcousticInterface(unittest.TestCase):
    def test_boundary_and_scale(self):
        for media in ((1, 1, 2, 1), (2, 1, 1, 1), (1.204, 343, 998, 1481),
                      (1, 1, 1, 1), (1, 1, 1 + 2**-30, 1),
                      (1e180, 1, 2e180, 1), (1e-180, 1, 2e-180, 1),
                      (1e-170, 1, 1e150, 1), (1e150, 1, 1e-170, 1),
                      (2.0**40, 2.0**-40, 8, 2.0**-3)):
            with self.subTest(media=media):
                args = dict(zip(("rho1", "c1", "rho2", "c2"), media))
                result = solve(**args)
                self.assertTrue(result["verified"], result)
                keys = ("Z1_rayl", "Z2_rayl", "pressure_reflection_R", "pressure_transmission_T",
                        "intensity_transmission_TI", "transmission_loss_dB")
                for key, expected in zip(keys, boundary_oracle(*media)):
                    actual = result[key]
                    self.assertLessEqual(abs(actual - expected), 5 * math.ulp(expected), (key,actual,expected))

    def test_integer_impedances_preserve_signed_reflection_and_decibel_loss(self):
        for exponent in (53, 100, 536, 537, 538):
            high = 2**exponent + 1
            low = high - 1
            for a, b in ((low, high), (high, low)):
                with self.subTest(exponent=exponent, rising=b > a):
                    result = solve(rho1=a, c1=1, rho2=b, c2=1)
                    self.assertTrue(result["verified"], result)
                    # Compute power directly from the unrounded impedances,
                    # independently of the reflected-power log-series path.
                    with localcontext() as context:
                        context.prec = 410
                        power = Decimal(4*a*b) / Decimal((a+b)**2)
                        loss = float(-10 * power.log10())
                    self.assertEqual(result["pressure_reflection_R"], (b-a)/(a+b))
                    if exponent == 53:
                        oblique = solve(rho1=a, c1=1, rho2=b, c2=1, incidence_sine=0)
                        self.assertTrue(oblique["verified"], oblique)
                        self.assertEqual(result["pressure_reflection_R"], oblique["pressure_reflection_R"])
                    self.assertAlmostEqual(result["transmission_loss_dB"], loss,
                                           delta=math.ulp(loss) if exponent < 500 else 0)

    def test_reciprocal_power_scattering(self):
        for ratio in (0.01, 0.75, 1.0, 1.5, 100.0):
            forward = solve(rho1=1, c1=1, rho2=ratio, c2=1)
            reverse = solve(rho1=ratio, c1=1, rho2=1, c2=1)
            self.assertEqual(forward["pressure_reflection_R"], -reverse["pressure_reflection_R"])
            self.assertEqual(forward["intensity_transmission_TI"], reverse["intensity_transmission_TI"])
            # Pressure waves divided by sqrt(Z) carry squared-magnitude power.
            s21 = forward["pressure_transmission_T"] / math.sqrt(ratio)
            s12 = reverse["pressure_transmission_T"] * math.sqrt(ratio)
            for a, b in ((1, 1), (1j, -0.5), (0.2-0.3j, 0.7+0.8j)):
                left = forward["pressure_reflection_R"]*a + s12*b
                right = s21*a + reverse["pressure_reflection_R"]*b
                self.assertAlmostEqual(abs(left)**2 + abs(right)**2, abs(a)**2 + abs(b)**2, places=13)

    def test_refusals_and_optional_resonator(self):
        fields = {"rho1": 1., "c1": 343., "rho2": 2., "c2": 343.}
        geometry = {"A_neck": .002, "V_cavity": .03, "L_eff": .04}
        self.assertIsNone(solve(**fields)["helmholtz_f0_Hz"])
        self.assertAlmostEqual(solve(**fields, **geometry)["helmholtz_f0_Hz"],
                               343*math.sqrt(.002/(.03*.04))/math.tau)
        invalid = [dict(fields, rho2=-2.),dict(fields, rho1=0.),dict(fields,c1=-1.,rho1=-1.),
                   dict(fields,A_neck=.002),dict(fields,**dict(geometry,V_cavity=-1.,L_eff=-2.)),
                   dict(fields,**dict(geometry,A_neck=0.)),dict(fields,rho1=float("nan")),
                   dict(fields,rho1=1e-300,c1=1e-300),dict(fields,rho2=1e300,c2=1e300),
                   dict(fields,rho1=1e-200,c1=1.,rho2=1e200,c2=1.)]
        for spec in invalid:
            with self.subTest(spec=spec):
                result = solve(**spec)
                self.assertFalse(result["verified"])
                self.assertIn("error", result)
                self.assertNotIn("transmission_loss_dB", result)
