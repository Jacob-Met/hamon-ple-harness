"""Independent signed-indicator integration for rectilinear STEP CSG."""
from collections import defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
from math import prod
import unittest

import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry


def box(lower, upper):
    return tuple((Fraction(str(a)), Fraction(str(b))) for a, b in zip(lower, upper))


def intersection(a, b):
    limits = tuple((max(al, bl), min(ah, bh)) for (al, ah), (bl, bh) in zip(a, b))
    return limits if all(l < h for l, h in limits) else None


def indicator(expr, boxes):
    if isinstance(expr, str):
        return {boxes[expr]: 1}
    op, left, right = expr
    a, b = indicator(left, boxes), indicator(right, boxes)
    terms = defaultdict(int)
    if op in ("UNION", "DIFFERENCE"):
        terms.update(a)
    if op == "UNION":
        for shape, coefficient in b.items():
            terms[shape] += coefficient
    for x, c in a.items():
        for y, d in b.items():
            overlap = intersection(x, y)
            if overlap is not None:
                terms[overlap] += c*d*(1 if op == "INTERSECTION" else -1)
    return {shape: c for shape, c in terms.items() if c}


def monomial(terms, powers):
    return sum(c * prod((b**(e+1)-a**(e+1))/(e+1)
                        for (a,b), e in zip(shape, powers)) for shape,c in terms.items())


def surface_area(terms):
    # On each face plane the squared jump in the material indicator is one
    # exactly on the exposed boundary. Integrate its rectangular cross terms.
    total = Fraction(0)
    for axis in range(3):
        jumps = defaultdict(list)
        for shape, c in terms.items():
            rect = shape[:axis]+shape[axis+1:]
            lo, hi = shape[axis]
            jumps[lo].append((rect, c))
            jumps[hi].append((rect, -c))
        for rectangles in jumps.values():
            for a,c in rectangles:
                for b,d in rectangles:
                    overlap = intersection(a,b)
                    if overlap is not None:
                        total += c*d*prod(hi-lo for lo,hi in overlap)
    return total


def decimal_text(q):
    with localcontext() as ctx:
        ctx.prec = 160
        return str(Decimal(q.numerator)/Decimal(q.denominator))


def document(boxes, expr, second_root=False):
    records, refs = [], {}

    def emit_record(kind, args):
        ref = f"#{len(records)+1}"
        records.append(f"{ref}={kind}({args});")
        return ref

    for name, shape in boxes.items():
        origin = ",".join(decimal_text(lo) for lo,_ in shape)
        dims = ",".join(decimal_text(hi-lo) for lo,hi in shape)
        point = emit_record("CARTESIAN_POINT", f"'',({origin})")
        frame = emit_record("AXIS2_PLACEMENT_3D", f"'',{point},$,$")
        refs[name] = emit_record("BLOCK", f"'',{frame},{dims}")

    def walk(node):
        if node not in refs:
            op, left, right = node
            refs[node] = emit_record("BOOLEAN_RESULT", f"'',.{op}.,{walk(left)},{walk(right)}")
        return refs[node]

    root = emit_record("CSG_SOLID", f"'',{walk(expr)}")
    if second_root:
        emit_record("CSG_SOLID", f"'',{refs['A']}")
    return "DATA;"+"".join(records)+"ENDSEC;", int(root[1:])


class StepCsgIndicatorOracle(unittest.TestCase):
    def setUp(self):
        self.boxes = {
            "A": box((3,-6,4), (14,7,13)),
            "B": box((5,-3,6), (11,4,11)),
            "C": box((9,-1,5), (15,3,12)),
            "D": box((4,0,0), (8,3,20)),
            "rib": box((7,-7,5), (8,11,12)),
            "island": box((6,-2,7), (8,1,9)),
            "face": box((14,-3,8), (19,3,10)),
            "edge": box((14,7,5), (16,8,10)),
            "out": box((22,22,22), (25,25,25)),
        }

    def compare_properties(self, boxes, expr, scale=Fraction(1), second_root=False):
        text, root = document(boxes, expr, second_root)
        spec = dict(step_text=text, density_kg_m3=7., unit_scale_m=float(scale))
        if second_root:
            spec["solid_entity_id"] = root
        result = verify_step_cad_geometry(spec)
        self.assertTrue(result["conservation_verified"], result)
        terms = indicator(expr, boxes)
        volume = monomial(terms, (0,0,0))
        center = [monomial(terms, [int(i==k) for i in range(3)])/volume for k in range(3)]
        covariance = [[monomial(terms, [int(i==k)+int(j==k) for k in range(3)])
                       - volume*center[i]*center[j] for j in range(3)] for i in range(3)]
        trace = sum(covariance[i][i] for i in range(3))
        inertia = np.array([[float(7*scale**5*((trace if i==j else 0)-covariance[i][j]))
                             for j in range(3)] for i in range(3)])
        for name, expected in (("volume_m3", volume*scale**3), ("mass_kg", 7*volume*scale**3),
                               ("surface_area_m2", surface_area(terms)*scale**2)):
            np.testing.assert_allclose(result[name], float(expected), rtol=4e-15, atol=0.)
        np.testing.assert_allclose(result["centroid_m"], [float(q*scale) for q in center], rtol=4e-15, atol=0.)
        np.testing.assert_allclose(result["inertia_tensor_centroid_kg_m2"], inertia, rtol=4e-14, atol=0.)
        np.testing.assert_allclose(result["principal_moments_kg_m2"], np.linalg.eigvalsh(inertia),
                                   rtol=7e-14, atol=0.)

    def test_cavity_cut_restoration_and_touching_properties(self):
        cases = [
            ("DIFFERENCE", "A", "B"), ("DIFFERENCE", "A", "D"),
            ("DIFFERENCE", "A", ("UNION", "B", "C")),
            ("DIFFERENCE", "A", ("DIFFERENCE", "B", "rib")),
            ("UNION", "A", "face"), ("INTERSECTION", "A", ("UNION", "A", "out")),
            ("INTERSECTION", ("UNION", "A", "C"), ("UNION", "B", "D")),
            ("UNION", ("DIFFERENCE", "A", "B"), ("INTERSECTION", "A", "B")),
        ]
        for expr in cases:
            for scale in (Fraction(1), Fraction(1,1000)):
                with self.subTest(expr=expr, scale=scale):
                    self.compare_properties(self.boxes, expr, scale)

    def test_selected_root_and_sub_ulp_wall(self):
        self.compare_properties(self.boxes, ("DIFFERENCE","A","B"), second_root=True)
        boxes = {"wall": box((0,0,0), ("1.0000000000000000000000000000000000000001",2,3)),
                 "cut": box((0,0,0), (1,2,3))}
        self.compare_properties(boxes, ("DIFFERENCE", "wall", "cut"))

    def test_empty_or_disconnected_material(self):
        for expr in [("UNION","A","out"), ("UNION","A","edge"), ("INTERSECTION","A","face"),
                     ("DIFFERENCE","A","A"), ("DIFFERENCE","A",("DIFFERENCE","B","island"))]:
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                verify_step_cad_geometry(dict(step_text=document(self.boxes,expr)[0], density_kg_m3=7))
