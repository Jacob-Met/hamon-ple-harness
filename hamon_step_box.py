"""Exact material moments of one positioned STEP BLOCK."""
from fractions import Fraction
import math
from hamon_step_csg import _number, _ref, _cross
from hamon_step_analytic import _vector, _scalar, _rational_sqrt, _determinant, _round_public

def _radical_terms(terms):
    """Combine b/sqrt(q) in exact rational square classes, including q=1."""
    groups = {}
    for coefficient, norm in terms:
        if not coefficient:
            continue
        root = _rational_sqrt(norm)
        if root is not None:
            groups[Fraction(1)] = groups.get(Fraction(1), Fraction(0))+coefficient/root
            continue
        for representative in groups:
            ratio = _rational_sqrt(norm/representative)
            if ratio is not None:
                groups[representative] += coefficient/ratio
                break
        else:
            groups[norm] = coefficient
    return [(coefficient, norm) for norm, coefficient in groups.items() if coefficient]

def block_mass_properties(entities, selected, unit_scale, density):
    scale, rho = _scalar(unit_scale, "unit scale", True), _scalar(density, "density", False)
    kind, args = entities.get(selected, (None, []))
    if kind != "BLOCK" or len(args) != 5:
        raise ValueError("STEP block requires exactly three positive dimensions")
    kind, placement = entities.get(_ref(args[1]), (None, []))
    if kind != "AXIS2_PLACEMENT_3D" or len(placement) != 4:
        raise ValueError("STEP block requires a three-dimensional placement")
    origin = tuple(v*scale for v in _vector(entities, placement[1], "CARTESIAN_POINT"))
    z = ((Fraction(0), Fraction(0), Fraction(1)) if placement[2] == "$"
         else _vector(entities, placement[2], "DIRECTION"))
    reference = ((Fraction(1), Fraction(0), Fraction(0)) if placement[3] == "$"
                 else _vector(entities, placement[3], "DIRECTION"))
    y = _cross(z, reference)
    if not any(y):
        if placement[3] != "$":
            raise ValueError("STEP block placement has parallel declared directions")
        y = _cross(z, (Fraction(0), Fraction(1), Fraction(0)))
    axes = (_cross(y, z), y, z)
    norms = [sum(v*v for v in axis) for axis in axes]
    lengths = [_number(v)*scale for v in args[2:]]
    if any(v <= 0 for v in lengths):
        raise ValueError("STEP block dimensions must be positive")
    frame = [[axes[j][i] for j in range(3)] for i in range(3)]
    determinant = _determinant(frame)
    if determinant <= 0 or determinant**2 != math.prod(norms):
        raise ValueError("STEP block frame failed its exact volume identity")
    projectors = [[[axis[i]*axis[j]/norm for j in range(3)] for i in range(3)]
                  for axis, norm in zip(axes, norms)]
    if any(sum(projector[i][j] for projector in projectors) != int(i == j)
           for i in range(3) for j in range(3)):
        raise ValueError("STEP block frame is not orthonormal")
    volume = math.prod(lengths)
    mass = rho*volume
    surface = 2*sum(lengths[i]*lengths[j] for i in range(3) for j in range(i+1, 3))
    moments = [mass*sum(lengths[j]**2 for j in range(3) if j != i)/12 for i in range(3)]
    tensor = [[sum(moments[k]*projectors[k][i][j] for k in range(3))
               for j in range(3)] for i in range(3)]
    principal = sorted(moments)
    if (sum(tensor[i][i] for i in range(3)) != sum(principal) or
            _determinant(tensor) != math.prod(principal) or
            sum(tensor[i][i]*tensor[j][j]-tensor[i][j]**2
                for i in range(3) for j in range(i+1, 3)) !=
            sum(principal[i]*principal[j] for i in range(3) for j in range(i+1, 3)) or
            (rho > 0 and min(principal) <= 0)):
        raise ValueError("STEP block moments failed their exact characteristic identities")
    centers = [_radical_terms([(origin[i], Fraction(1))]+[
        (lengths[j]*axes[j][i]/2, norms[j]) for j in range(3)]) for i in range(3)]
    extents = [_radical_terms([(lengths[j]*abs(axes[j][i]), norms[j])
                              for j in range(3)]) for i in range(3)]

    def evaluate(iv):
        def num(value):
            return iv.mpf(value.numerator)/value.denominator
        def radical_sum(terms):
            return sum((num(b)/iv.sqrt(num(q)) for b, q in terms), iv.mpf(0))
        return ([num(volume), num(mass), num(surface), num(rho)]
                + [radical_sum(terms) for terms in centers]
                + [radical_sum(terms) for terms in extents]
                + [num(value) for row in tensor for value in row]
                + [num(value) for value in principal])
    rounded = _round_public(evaluate)
    if min(rounded[0], rounded[2], *rounded[7:10]) <= 0 or (
            rho > 0 and min(rounded[1], rounded[3], *rounded[19:22]) <= 0):
        raise ValueError("STEP block positive material property vanished on output")
    properties = {
        "volume_m3": rounded[0], "mass_kg": rounded[1],
        "surface_area_m2": rounded[2], "density_kg_m3": rounded[3],
        "centroid_m": rounded[4:7], "aabb_extents_m": rounded[7:10],
        "inertia_tensor_centroid_kg_m2": [rounded[10+3*i:13+3*i] for i in range(3)],
        "principal_moments_kg_m2": rounded[19:22],
        "watertight": True, "connected_component_count": 1,
        "solid_geometry_kind": "analytic_block",
        "mass_properties_method": "exact positioned rectangular material moments",
        "analytic_invariants_verified": True,
        "output_rounding_verified": True,
        "output_relative_error_bound": 1e-12,
    }
    return properties, rounded[0]
