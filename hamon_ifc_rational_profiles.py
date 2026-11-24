"""Source-area qualification for clamped convex rational quadratic IFC profiles."""
import math
from fractions import Fraction


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("Nonfinite rational profile parameter")
    return Fraction(value)


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _curve_fraction(weight):
    """Fraction of the oriented control triangle between a span and its chord.

    Under an affine map to endpoints (0,0), (0,1) and control (1,0),
    Green's formula gives integral[0,1] 2*w*t^2 / D^2 dt, where
    D=1+2*(w-1)*t*(1-t). Integrate the binomial expansion term by term.
    The remainder bound uses |D-1| <= |w-1|/2 <= 1/2 over the entire span.
    """
    if not Fraction(1, 2) <= weight <= 2:
        raise ValueError("Rational middle weight outside the qualified range")
    terms = 64
    term = 2*weight/3
    value = term
    for k in range(terms-1):
        term *= -2*(weight-1)*(k+2)*(k+3) / ((2*k+5)*(2*k+4))
        value += term
    q = abs(weight-1)/2
    error = 2*weight/3 * q**terms * ((terms+1)-terms*q) / (1-q)**2
    return value, error


def rational_profile_area(curve):
    points = [tuple(_number(v) for v in p.Coordinates) for p in curve.ControlPointsList]
    weights = [_number(w) for w in curve.WeightsData]
    knots = [_number(k) for k in curve.Knots]
    if (curve.Degree != 2 or curve.ClosedCurve is not True or curve.SelfIntersect is not False
            or not 7 <= len(points) <= 65 or len(points) % 2 != 1
            or len(weights) != len(points) or any(w <= 0 for w in weights)
            or any(len(p) != 2 for p in points) or points[0] != points[-1]
            or len(knots) != (len(points)+1)//2 or any(a >= b for a,b in zip(knots, knots[1:]))
            or list(curve.KnotMultiplicities) != [3]+[2]*(len(knots)-2)+[3]):
        raise ValueError("Expected a closed clamped piecewise quadratic rational profile")
    boundary = points[:-1]
    if len(set(boundary)) != len(boundary):
        raise ValueError("Repeated rational profile control vertex")
    direction = 0
    # Every directed side of the control polygon must support its convex hull.
    # Ordered distinct hull points preclude crossing spans. A span with positive
    # weights lies in its own control triangle; adjacent triangles meet at most
    # on their shared endpoint for this cyclic convex polygon.
    for a,b in zip(boundary, boundary[1:]+boundary[:1]):
        for c in boundary:
            turn = _cross(a,b,c)
            if turn:
                sign = 1 if turn > 0 else -1
                if direction and direction != sign:
                    raise ValueError("Nonconvex or retraced rational profile control polygon")
                direction = sign
    if not direction:
        raise ValueError("Collinear rational profile")
    twice_area = error = Fraction(0)
    origin = points[0]
    factors = {}
    for i in range(0, len(points)-1, 2):
        a,b,c = points[i:i+3]
        if weights[i] != weights[i+2]:
            raise ValueError("Rational span endpoint weights must agree")
        w = weights[i+1]/weights[i]
        if w not in factors:
            factors[w] = _curve_fraction(w)
        fraction, bound = factors[w]
        triangle = _cross(a,b,c)
        twice_area += _cross(origin,a,c) + triangle*fraction
        error += abs(triangle)*bound
    if twice_area*direction <= 0 or error > abs(twice_area)*Fraction(1,10**14):
        raise ValueError("Rational profile area is unresolved")
    result = float(abs(twice_area)/2)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Rational profile area is outside the finite output range")
    return result
