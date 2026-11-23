"""Schema-correct circular STEP solids with exact section moments.

Only a complete cylinder or sphere is supported here. Boolean combinations of
curved primitives remain outside this bounded analytic kernel.
"""
from fractions import Fraction
import math
from mpmath.ctx_iv import MPIntervalContext
from hamon_step_csg import _number, _ref

CURVED_PRIMITIVES = {"RIGHT_CIRCULAR_CYLINDER", "SPHERE"}
_MATERIAL = CURVED_PRIMITIVES | {
    "BLOCK", "RIGHT_CIRCULAR_CONE", "TORUS", "ELLIPSOID", "ECCENTRIC_CONE",
    "CYCLIDE_SEGMENT_SOLID", "RECTANGULAR_PYRAMID", "RIGHT_ANGULAR_WEDGE",
    "FACETED_PRIMITIVE", "BOUNDED_PRIMITIVE_2D", "BOOLEAN_RESULT",
    "MAPPED_ITEM", "SOLID_REPLICA", "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
    "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION",
}

def select_analytic_primitive(entities, supported=CURVED_PRIMITIVES):
    candidates = [eid for eid, (kind, _) in entities.items()
                  if kind in supported]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("STEP analytic primitive selection is ambiguous")
    selected = candidates[0]
    for eid, (kind, args) in entities.items():
        if eid != selected and (kind in _MATERIAL or kind.endswith("_SOLID")
                                or kind in {"FACETED_BREP", "MANIFOLD_SOLID_BREP",
                                            "BREP_WITH_VOIDS"}):
            raise ValueError("STEP analytic primitive has other material definitions")
        if kind == "SHAPE_REPRESENTATION" or kind.endswith("_SHAPE_REPRESENTATION"):
            if len(args) != 3 or not (args[1].startswith("(") and args[1].endswith(")")):
                raise ValueError("STEP analytic representation is incomplete")
            items = [_ref(v) for v in args[1][1:-1].split(",")]
            if any(i != selected and entities.get(i, (None, []))[0] not in
                   {"AXIS1_PLACEMENT", "AXIS2_PLACEMENT_3D"} for i in items):
                raise ValueError("STEP analytic representation includes other material")
    return selected

def _vector(entities, token, required):
    kind, args = entities.get(_ref(token), (None, []))
    if kind != required or len(args) != 2 or not (
            args[1].startswith("(") and args[1].endswith(")")):
        raise ValueError("STEP primitive requires a correctly typed point or direction")
    values = tuple(_number(v) for v in args[1][1:-1].split(","))
    if len(values) != 3 or (required == "DIRECTION" and not any(values)):
        raise ValueError("STEP primitive requires three coordinates and a nonzero axis")
    return values

def _integral(coefficients, lo, hi, power=0):
    return sum((c * (hi**(i+power+1)-lo**(i+power+1)) / (i+power+1)
                for i, c in enumerate(coefficients)), Fraction(0))

def _scalar(value, name, positive):
    if isinstance(value, bool):
        raise ValueError(name + " must be a finite physical scalar")
    try:
        result = _number(value) if isinstance(value, str) else Fraction(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(name + " must be finite") from exc
    if result < 0 or (positive and not result):
        raise ValueError(name + " must be positive" if positive else name + " must be nonnegative")
    return result

def _rational_sqrt(value):
    numerator, denominator = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if numerator*numerator == value.numerator and denominator*denominator == value.denominator:
        return Fraction(numerator, denominator)
    return None

def _determinant(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))

def _round_public(evaluator, tensor_scales=None):
    def endpoint_float(endpoint):
        # Convert exact dyadic endpoints with Python's ties-to-even rounding.
        sign, mantissa, exponent, bits = endpoint._mpi_[0]
        if bits < 0:
            raise ValueError("STEP analytic result is not finite")
        if not mantissa:
            return 0.0
        if exponent+bits > 1025:
            raise ValueError("STEP analytic result exceeds binary64 range")
        if exponent+bits < -1200:
            return -0.0 if sign else 0.0
        value = (Fraction(mantissa << exponent) if exponent >= 0 else
                 Fraction(mantissa, 1 << -exponent))
        return float(-value if sign else value)

    for digits in (80, 160, 320, 640, 1280, 2560, 5120):
        iv = MPIntervalContext()
        iv.dps = digits
        values = evaluator(iv)
        scales = {} if tensor_scales is None else tensor_scales(iv, values)
        rounded = []
        for index, value in enumerate(values):
            try:
                low, high = endpoint_float(value.a), endpoint_float(value.b)
            except (OverflowError, ValueError) as exc:
                raise ValueError("STEP analytic result exceeds finite binary64 range") from exc
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError("STEP analytic result exceeds finite binary64 range")
            if low != high:
                break
            if low == 0 and index not in scales and (value.a > 0 or value.b < 0):
                raise ValueError("STEP analytic nonzero result is lost in binary64")
            if value != iv.mpf(0) and not (
                    abs(iv.mpf(low)-value) <= scales.get(index, abs(value))*iv.mpf("1e-12")):
                break
            rounded.append(low)
        if len(rounded) == len(values):
            break
    else:
        raise ValueError("STEP analytic output cannot be qualified within binary64 tolerance")
    return rounded

def analytic_mass_properties(entities, selected, unit_scale, density):
    scale = _scalar(unit_scale, "unit scale", True)
    rho = _scalar(density, "density", False)
    kind, args = entities.get(selected, (None, []))
    if kind == "CSG_SOLID":
        if len(args) != 2:
            raise ValueError("STEP CSG primitive root is incomplete")
        selected = _ref(args[1])
        kind, args = entities.get(selected, (None, []))
    if kind == "SPHERE" and len(args) == 3:
        radius = _number(args[1])*scale
        origin = tuple(v*scale for v in _vector(entities, args[2], "CARTESIAN_POINT"))
        axis = (Fraction(0), Fraction(0), Fraction(1))
        height = 2*radius
        q, lo, hi = [radius**2, Fraction(0), Fraction(-1)], -radius, radius
        expected_volume, surface = 4*radius**3/3, 4*radius**2
    elif kind == "RIGHT_CIRCULAR_CYLINDER" and len(args) == 4:
        position_kind, position = entities.get(_ref(args[1]), (None, []))
        if position_kind != "AXIS1_PLACEMENT" or len(position) != 3:
            raise ValueError("STEP cylinder position must be AXIS1_PLACEMENT")
        origin = tuple(v*scale for v in _vector(entities, position[1], "CARTESIAN_POINT"))
        axis = ((Fraction(0), Fraction(0), Fraction(1)) if position[2] == "$"
                else _vector(entities, position[2], "DIRECTION"))
        height, radius = (_number(v)*scale for v in args[2:])
        q, lo, hi = [radius**2], Fraction(0), height
        expected_volume, surface = radius**2*height, 2*radius*(radius+height)
    else:
        raise ValueError("STEP analytic primitive is unsupported or incomplete")
    if radius <= 0 or height <= 0:
        raise ValueError("STEP analytic primitive dimensions must be positive")
    volume = _integral(q, lo, hi)
    first, second = (_integral(q, lo, hi, n) for n in (1, 2))
    squared = [Fraction(0)]*(2*len(q)-1)
    for i, a in enumerate(q):
        for j, b in enumerate(q):
            squared[i+j] += a*b
    radial = _integral(squared, lo, hi)
    center = first/volume
    axial = rho*radial/2
    transverse = rho*(radial/4+second-first**2/volume)
    norm = sum(v*v for v in axis)
    tensor = [[(transverse if i == j else Fraction(0)) +
               (axial-transverse)*axis[i]*axis[j]/norm
               for j in range(3)] for i in range(3)]
    principal = sorted((axial, transverse, transverse))
    trace = sum(tensor[i][i] for i in range(3))
    second_invariant = sum(tensor[i][i]*tensor[j][j]-tensor[i][j]**2
                           for i in range(3) for j in range(i+1, 3))
    if (volume != expected_volume or volume <= 0 or
            trace != sum(principal) or _determinant(tensor) != math.prod(principal) or
            second_invariant != sum(principal[i]*principal[j]
                                    for i in range(3) for j in range(i+1, 3)) or
            (rho > 0 and min(principal) <= 0)):
        raise ValueError("STEP analytic section moments failed their exact invariants")

    rational_norm = _rational_sqrt(norm)
    exact_extents = []
    for coordinate in axis:
        radial_root = _rational_sqrt((norm-coordinate**2)/norm)
        if kind == "SPHERE" or coordinate == 0:
            exact_extents.append(2*radius)
        elif rational_norm is not None and radial_root is not None:
            exact_extents.append(height*abs(coordinate)/rational_norm+2*radius*radial_root)
        else:
            exact_extents.append(None)

    def evaluate(iv):
        def num(value):
            return iv.mpf(value.numerator)/value.denominator
        root = iv.sqrt(num(norm))
        def translated(a, b):
            if rational_norm is not None:
                return num(a+b/rational_norm)
            # Rationalize cancellation in a+b/sqrt(norm).
            if a*b < 0:
                numerator = a*a*norm-b*b
                if not numerator:
                    return iv.mpf(0)
                return num(numerator)/(root*(num(a)*root-num(b)))
            return num(a)+num(b)/root
        centers = [translated(origin[i], center*axis[i]) for i in range(3)]
        extents = [num(exact_extents[i]) if exact_extents[i] is not None else
                   num(height*abs(axis[i]))/root
                   +2*num(radius)*iv.sqrt(num((norm-axis[i]**2)/norm))
                   for i in range(3)]
        return ([num(volume)*iv.pi, num(rho*volume)*iv.pi,
                   num(surface)*iv.pi, num(rho)] + centers + extents
                  + [num(v)*iv.pi for row in tensor for v in row]
                  + [num(v)*iv.pi for v in principal])
    rounded = _round_public(evaluate)
    if min(rounded[0], rounded[2], *rounded[7:10]) <= 0 or (
            rho > 0 and min(rounded[1], rounded[3], *rounded[19:22]) <= 0):
        raise ValueError("STEP analytic positive material property vanished on output")
    properties = {
        "volume_m3": rounded[0], "mass_kg": rounded[1],
        "surface_area_m2": rounded[2], "density_kg_m3": rounded[3],
        "centroid_m": rounded[4:7], "aabb_extents_m": rounded[7:10],
        "inertia_tensor_centroid_kg_m2": [rounded[10+3*i:13+3*i] for i in range(3)],
        "principal_moments_kg_m2": rounded[19:22],
        "watertight": True, "connected_component_count": 1,
        "solid_geometry_kind": "analytic_" + kind.lower(),
        "mass_properties_method": "exact circular section moments",
        "analytic_invariants_verified": True,
        "output_rounding_verified": True,
        "output_relative_error_bound": 1e-12,
    }
    return properties, rounded[0]
