"""Singly reinforced rectangular concrete section under pure bending."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math


def rectangular_rc_section(width, depth, area, concrete, yield_stress, modulus):
    """Use a Whitney block, plane sections, and elastic–perfectly plastic tension steel.

    The nonspiral strength factor uses ACI CODE-318-19(22), Table 21.2.2,
    column "Other". This calculation gives section strengths only.
    """
    values = width, depth, area, concrete, yield_stress, modulus
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Require positive finite section geometry, strengths and steel modulus")
    b, d, steel, fc, fy, es = map(Fraction, values)
    if fc < 17_000_000:
        raise ValueError("The specified concrete strength falls below the 17 MPa Whitney-block table range")
    beta = max(Fraction(65, 100), Fraction(85, 100)
               - max(Fraction(0), fc-28_000_000)/140_000_000)
    concrete_stress = Fraction(85, 100)*fc
    k = concrete_stress*b*beta       # C = k*c
    crushing = Fraction(3, 1000)
    yield_strain = fy/es
    yield_axis = steel*fy/k
    trial_strain = crushing*(d-yield_axis)/yield_axis
    yielded = trial_strain >= yield_strain

    def dec(q):
        return Decimal(q.numerator)/Decimal(q.denominator)

    def wire(x):
        f = float(x)
        if not math.isfinite(f) or (x and not f):
            raise ValueError("Section property is outside the finite binary64 reporting range")
        return f

    with localcontext() as ctx:
        ctx.prec = 90
        if yielded:
            c, strain, stress = dec(yield_axis), dec(trial_strain), dec(fy)
        else:
            # C=T implies k*c² + As*Es*0.003*c - As*Es*0.003*d = 0.
            # The positive, rationalized root remains well conditioned at large As.
            c = 2*dec(d)/(1+(1+dec(4*k*d/(steel*es*crushing))).sqrt())
            stress = dec(k/steel)*c
            strain = stress/dec(es)
        if not yielded or trial_strain <= yield_strain:
            phi, classification = Fraction(65, 100), "compression_controlled"
        elif trial_strain >= yield_strain + crushing:
            phi, classification = Fraction(9, 10), "tension_controlled"
        else:
            phi = Fraction(65, 100) + Fraction(1, 4)*(trial_strain-yield_strain)/crushing
            classification = "transition"
        a = dec(beta)*c
        tension = dec(steel)*stress
        compression = dec(concrete_stress*b)*a
        nominal = tension*(dec(d)-a/2)
        design = dec(phi)*nominal

        result = dict(beta1=wire(dec(beta)), whitney_a_m=wire(a), neutral_axis_c_m=wire(c),
                      tensile_strain=wire(strain), steel_stress_Pa=wire(stress),
                      steel_yield_strain=wire(dec(yield_strain)), steel_yielded=yielded,
                      concrete_compression_N=wire(compression), steel_tension_N=wire(tension),
                      Mn_Nm=wire(nominal), phi=wire(dec(phi)), phi_Mn_Nm=wire(design),
                      strain_classification=classification, analysis_scope="rectangular_section_pure_flexure",
                      strength_factor_basis="ACI CODE-318-19(22), Table 21.2.2, other transverse reinforcement")

    # Replay the published coordinates on the original compatibility, material,
    # and force equations. Taking moments about the concrete face also checks Mn.
    c, a, eps, fs, comp, tens, moment, factor, reduced = map(
        lambda key: Fraction(result[key]), ("neutral_axis_c_m","whitney_a_m","tensile_strain",
        "steel_stress_Pa","concrete_compression_N","steel_tension_N","Mn_Nm","phi","phi_Mn_Nm"))
    checks = [(eps*c, crushing*(d-c)), (fs, min(fy,es*eps)), (a,beta*c),
              (comp,concrete_stress*b*a), (tens,steel*fs), (comp,tens),
              (moment,tens*d-comp*a/2), (reduced,factor*moment)]
    error = max(abs(lhs-rhs)/max(abs(lhs),abs(rhs)) for lhs,rhs in checks)
    result.update(dual_rel_discrepancy=float(error), dual_verified=error < Fraction(1,10**9))
    return result
