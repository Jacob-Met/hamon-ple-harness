"""Qualify joint mass and molar rates of a single neutral formula.

The 2024 CIAAW abridged standard atomic weights and their quoted uncertainties
describe ordinary terrestrial materials. Isotope-specific samples and elements
without a standard atomic weight have no joint mass certificate on this route.
Source: https://ciaaw.org/abridged-atomic-weights.htm
"""

import math
from fractions import Fraction


# Symbol, abridged atomic weight, absolute uncertainty.
_ATOMIC_WEIGHTS = """
H 1.0080 0.0002
He 4.0026 0.0001
Li 6.94 0.06
Be 9.0122 0.0001
B 10.81 0.02
C 12.011 0.002
N 14.007 0.001
O 15.999 0.001
F 18.998 0.001
Ne 20.180 0.001
Na 22.990 0.001
Mg 24.305 0.002
Al 26.982 0.001
Si 28.085 0.001
P 30.974 0.001
S 32.06 0.02
Cl 35.45 0.01
Ar 39.95 0.16
K 39.098 0.001
Ca 40.078 0.004
Sc 44.956 0.001
Ti 47.867 0.001
V 50.942 0.001
Cr 51.996 0.001
Mn 54.938 0.001
Fe 55.845 0.002
Co 58.933 0.001
Ni 58.693 0.001
Cu 63.546 0.003
Zn 65.38 0.02
Ga 69.723 0.001
Ge 72.630 0.008
As 74.922 0.001
Se 78.971 0.008
Br 79.904 0.003
Kr 83.798 0.002
Rb 85.468 0.001
Sr 87.62 0.01
Y 88.906 0.001
Zr 91.222 0.003
Nb 92.906 0.001
Mo 95.95 0.01
Ru 101.07 0.02
Rh 102.91 0.01
Pd 106.42 0.01
Ag 107.87 0.01
Cd 112.41 0.01
In 114.82 0.01
Sn 118.71 0.01
Sb 121.76 0.01
Te 127.60 0.03
I 126.90 0.01
Xe 131.29 0.01
Cs 132.91 0.01
Ba 137.33 0.01
La 138.91 0.01
Ce 140.12 0.01
Pr 140.91 0.01
Nd 144.24 0.01
Sm 150.36 0.02
Eu 151.96 0.01
Gd 157.25 0.01
Tb 158.93 0.01
Dy 162.50 0.01
Ho 164.93 0.01
Er 167.26 0.01
Tm 168.93 0.01
Yb 173.05 0.02
Lu 174.97 0.01
Hf 178.49 0.01
Ta 180.95 0.01
W 183.84 0.01
Re 186.21 0.01
Os 190.23 0.03
Ir 192.22 0.01
Pt 195.08 0.02
Au 196.97 0.01
Hg 200.59 0.01
Tl 204.38 0.01
Pb 207.2 1.1
Bi 208.98 0.01
Th 232.04 0.01
Pa 231.04 0.01
U 238.03 0.01
"""


def _atomic_ranges():
    ranges = {}
    for row in _ATOMIC_WEIGHTS.strip().splitlines():
        symbol, center, uncertainty = row.split()
        center, uncertainty = Fraction(center), Fraction(uncertainty)
        ranges[symbol] = (center - uncertainty, center + uncertainty)
    return ranges


_ATOMIC_RANGES = _atomic_ranges()


def check_stream_mass(elements, mass_rate, molar_rate, rtol):
    """Return a reason if mdot cannot arise from ndot at the declared tolerance."""
    if not all(math.isfinite(x) and x >= 0 for x in (mass_rate, molar_rate)):
        return "mass and molar rates must be finite and nonnegative"
    if not math.isfinite(rtol) or not 0 <= rtol < 1:
        return "relative tolerance must be finite and between zero and one (exclusive of one)"
    if bool(mass_rate) != bool(molar_rate):
        return "zero mass and zero molar flow must coincide"

    lower = upper = Fraction(0)
    for symbol, count in elements.items():
        if not count:
            continue
        if symbol not in _ATOMIC_RANGES:
            return f"no standard atomic weight for {symbol}"
        low, high = _ATOMIC_RANGES[symbol]
        lower += Fraction(count) * low
        upper += Fraction(count) * high

    # Work in g/s. Neither a large formula nor a tiny rate should narrow or
    # overflow the comparison interval before both coordinates are examined.
    observed = Fraction(mass_rate) * 1000
    n = Fraction(molar_rate)
    retained = 1 - Fraction(rtol)
    if observed < retained * n * lower or observed * retained > n * upper:
        return "mass and molar rates disagree with the formula mass"
    return None
