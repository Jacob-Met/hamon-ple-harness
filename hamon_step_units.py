"""STEP length units from representation contexts and typed conversion graphs.

Labels are opaque; conversion factors remain rational until the metre scale.
See ISO 10303-41 measure_schema, global_unit_assigned_context and named_unit.
"""
from fractions import Fraction
import math
import re

_LENGTH = (Fraction(1),) + (Fraction(0),) * 6
_PREFIX = dict(zip(
    ("YOCTO", "ZEPTO", "ATTO", "FEMTO", "PICO", "NANO", "MICRO", "MILLI",
     "CENTI", "DECI", "DECA", "HECTO", "KILO", "MEGA", "GIGA", "TERA",
     "PETA", "EXA", "ZETTA", "YOTTA"),
    (-24, -21, -18, -15, -12, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24)))
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"


def _statements(text):
    """Split outside strings/comments, respecting doubled apostrophes."""
    out, quoted, i = [], False, 0
    while i < len(text):
        ch = text[i]
        if quoted:
            out.append(ch)
            if ch == "'":
                if i + 1 < len(text) and text[i+1] == "'":
                    out.append("'")
                    i += 2
                    continue
                quoted = False
        elif ch == "'":
            quoted = True
            out.append(ch)
        elif text.startswith("/*", i):
            end = text.find("*/", i+2)
            if end < 0:
                raise ValueError("STEP contains an unterminated comment")
            out.append(" ")
            i = end + 2
            continue
        elif ch == ";":
            yield "".join(out).strip()
            out = []
        else:
            out.append(ch)
        i += 1
    if quoted:
        raise ValueError("STEP contains an unterminated string")
    if "".join(out).strip().startswith("#"):
        raise ValueError("STEP entity is missing its terminator")


def _components(expression, split_args):
    complex_record = expression.startswith("(")
    if complex_record:
        if not expression.endswith(")"):
            raise ValueError("STEP complex entity is incomplete")
        expression = expression[1:-1]
    result, pos = {}, 0
    while pos < len(expression):
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*\(", expression[pos:])
        if not match:
            if not expression[pos:].strip():
                break
            raise ValueError("STEP entity component is malformed")
        name, start = match[1].upper(), pos + match.end()
        depth, quoted, end = 1, False, start
        while end < len(expression) and depth:
            ch = expression[end]
            if quoted:
                if ch == "'":
                    if end+1 < len(expression) and expression[end+1] == "'":
                        end += 2
                        continue
                    quoted = False
            elif ch == "'":
                quoted = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            end += 1
        if depth or quoted or name in result:
            raise ValueError("STEP entity has incomplete or repeated components")
        result[name] = split_args(expression[start:end-1])
        pos = end
        if not complex_record and expression[pos:].strip():
            raise ValueError("STEP simple entity has trailing components")
    if not result:
        raise ValueError("STEP entity has no components")
    return result, complex_record


def parse_step_records(text, split_args):
    entities, components = {}, {}
    for statement in _statements(text):
        if not statement.startswith("#"):
            continue
        match = re.fullmatch(r"#(\d+)\s*=\s*(.+)", statement, flags=re.DOTALL)
        if not match or int(match[1]) in components:
            raise ValueError("STEP entity identifier is malformed or repeated")
        eid = int(match[1])
        parts, complex_record = _components(match[2].strip(), split_args)
        components[eid] = parts
        entities[eid] = ("COMPLEX", []) if complex_record else next(iter(parts.items()))
    return entities, components


def _reference(token):
    match = re.fullmatch(r"#\s*(\d+)", token.strip())
    if not match:
        raise ValueError("STEP unit/context requires a complete entity reference")
    return int(match[1])


def _references(token):
    token = token.strip()
    if not (token.startswith("(") and token.endswith(")")):
        raise ValueError("STEP unit/context requires a reference aggregate")
    result = [_reference(item) for item in token[1:-1].split(",")]
    if len(result) != len(set(result)):
        raise ValueError("STEP unit/context has repeated references")
    return result


def _number(token):
    if not re.fullmatch(_NUMBER, token.strip()):
        raise ValueError("STEP unit conversion requires a finite numeric value")
    return Fraction(token.strip())


def resolve_step_length_scale(components, geometry_ids):
    """Resolve only contexts containing selected definitions; refuse ambiguity."""
    def parts(eid):
        if eid not in components:
            raise ValueError("STEP unit/context references a missing entity")
        return components[eid]

    def dimensions(token):
        if token == "*":
            return None
        values = parts(_reference(token)).get("DIMENSIONAL_EXPONENTS")
        if values is None or len(values) != 7:
            raise ValueError("STEP named unit requires seven dimensional exponents")
        return tuple(_number(value) for value in values)

    def dimension_token(value):
        if "NAMED_UNIT" in value:
            args = value["NAMED_UNIT"]
            if len(args) != 1:
                raise ValueError("STEP named unit dimensions are incomplete")
            return args[0]
        for kind, count in (("CONVERSION_BASED_UNIT", 3), ("CONTEXT_DEPENDENT_UNIT", 2),
                            ("LENGTH_UNIT", 1)):
            if kind in value and len(value[kind]) == count:
                return value[kind][0]
        return None

    def is_length(eid):
        value = parts(eid)
        if "LENGTH_UNIT" in value:
            return True
        si = value.get("SI_UNIT", [])
        if si and si[-1].upper() == ".METRE.":
            return True
        token = dimension_token(value)
        return token is not None and dimensions(token) == _LENGTH

    def scale(eid, path=frozenset()):
        if eid in path:
            raise ValueError("STEP length-unit conversion graph is cyclic")
        value = parts(eid)
        if not is_length(eid):
            raise ValueError("STEP length conversion references a different physical dimension")
        allowed = {"LENGTH_UNIT", "NAMED_UNIT", "SI_UNIT", "CONVERSION_BASED_UNIT"}
        if set(value) - allowed or ("SI_UNIT" in value) == ("CONVERSION_BASED_UNIT" in value):
            raise ValueError("STEP length unit is unsupported or has conflicting definitions")
        if value.get("LENGTH_UNIT"):
            raise ValueError("STEP complex length-unit component has unexpected attributes")
        token = dimension_token(value)
        declared = dimensions(token) if token is not None else None
        if declared is not None and declared != _LENGTH:
            raise ValueError("STEP length unit has inconsistent dimensions")
        if "SI_UNIT" in value:
            args = value["SI_UNIT"]
            if "NAMED_UNIT" not in value and len(args) == 3 and args[0] == "*":
                args = args[1:]
            if len(args) != 2 or args[1].upper() != ".METRE.":
                raise ValueError("STEP length SI unit must be metre")
            prefix = args[0].upper()
            if prefix == "$":
                return Fraction(1)
            if not (prefix.startswith(".") and prefix.endswith(".") and prefix[1:-1] in _PREFIX):
                raise ValueError("STEP length SI prefix is unsupported")
            return Fraction(10) ** _PREFIX[prefix[1:-1]]
        args = value["CONVERSION_BASED_UNIT"]
        if "NAMED_UNIT" not in value and len(args) == 3:
            args = args[1:]
        if len(args) != 2 or declared != _LENGTH:
            raise ValueError("STEP converted length unit requires consistent explicit dimensions")
        factor = parts(_reference(args[1]))
        if set(factor) - {"MEASURE_WITH_UNIT", "LENGTH_MEASURE_WITH_UNIT"}:
            raise ValueError("STEP length conversion requires a length measure with unit")
        if "MEASURE_WITH_UNIT" in factor and factor.get("LENGTH_MEASURE_WITH_UNIT"):
            raise ValueError("STEP complex measure has conflicting attributes")
        measure = factor.get("MEASURE_WITH_UNIT") or factor.get("LENGTH_MEASURE_WITH_UNIT")
        if measure is None or len(measure) != 2:
            raise ValueError("STEP length conversion factor is incomplete")
        match = re.fullmatch(r"(?:POSITIVE_)?LENGTH_MEASURE\s*\(\s*(" + _NUMBER + r")\s*\)",
                             measure[0], flags=re.IGNORECASE)
        if not match:
            raise ValueError("STEP length conversion factor has the wrong measure type")
        amount = _number(match[1])
        if amount <= 0:
            raise ValueError("STEP length conversion factor must be positive")
        return amount * scale(_reference(measure[1]), path | {eid})

    geometry_ids = set(geometry_ids)
    contexts, represented, has_representations = set(), set(), False
    for value in components.values():
        kinds = [kind for kind in value if kind == "REPRESENTATION" or kind == "SHAPE_REPRESENTATION"
                 or kind.endswith("_SHAPE_REPRESENTATION")]
        if not kinds:
            continue
        has_representations = True
        candidates = [value[kind] for kind in kinds if value[kind]]
        if len(candidates) != 1 or len(candidates[0]) != 3:
            raise ValueError("STEP representation context is incomplete or unsupported")
        args = candidates[0]
        possible = {int(item) for item in re.findall(r"#\s*(\d+)", args[1])}
        if not geometry_ids.intersection(possible):
            continue
        items = set(_references(args[1]))
        represented.update(geometry_ids.intersection(items))
        contexts.add(_reference(args[2]))

    def unit_definition(value):
        return any(kind.endswith("_UNIT") and not kind.endswith("_WITH_UNIT") for kind in value)

    unit_ids = [eid for eid, value in components.items() if unit_definition(value)]
    if contexts:
        if represented != geometry_ids:
            raise ValueError("STEP selected geometry has incomplete unit-context coverage")
        resolved, length_ids = [], []
        for context in sorted(contexts):
            value = parts(context)
            assigned = value.get("GLOBAL_UNIT_ASSIGNED_CONTEXT")
            if assigned is None:
                if unit_ids or not ({"GEOMETRIC_REPRESENTATION_CONTEXT", "REPRESENTATION_CONTEXT"} & set(value)):
                    raise ValueError("STEP selected representation has no assigned length unit")
                resolved.append(Fraction(1))
                continue
            if len(assigned) == 3:
                assigned = assigned[2:]
            if len(assigned) != 1:
                raise ValueError("STEP global unit assignment is incomplete")
            references = _references(assigned[0])
            for eid in references:
                if not unit_definition(parts(eid)):
                    raise ValueError("STEP context references a non-unit entity")
            length = [eid for eid in references if is_length(eid)]
            if len(length) != 1:
                raise ValueError("STEP representation requires exactly one length unit")
            length_ids.extend(length)
            resolved.append(scale(length[0]))
        if len(set(resolved)) != 1:
            raise ValueError("STEP selected geometry belongs to conflicting length contexts")
        exact = resolved[0]
        source = "representation_context" if length_ids else "implicit_metre"
    else:
        if has_representations or any("GLOBAL_UNIT_ASSIGNED_CONTEXT" in value for value in components.values()):
            if unit_ids:
                raise ValueError("STEP length units are not attached to the selected geometry")
        length_ids = [eid for eid in unit_ids if is_length(eid)]
        if len(length_ids) > 1:
            raise ValueError("STEP standalone length units are ambiguous; provide unit_scale_m")
        exact = scale(length_ids[0]) if length_ids else Fraction(1)
        source = "standalone_length_unit" if length_ids else "implicit_metre"
    try:
        result = float(exact)
    except OverflowError as exc:
        raise ValueError("STEP length scale is outside the representable range") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("STEP length scale must be finite and positive")
    return result, {"unit_scale_source": source,
                    "representation_context_entity_ids": sorted(contexts),
                    "length_unit_entity_ids": sorted(set(length_ids))}
