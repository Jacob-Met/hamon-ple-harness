"""Source-bound IFC indexed profile segments and curved extrusion measurements."""
import math


def ordered_segments(points, segments):
    """Resolve an explicitly segmented closed profile. Point-list order has no geometric meaning."""
    edges = []
    first = previous = None
    for kind, raw in segments:
        if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], list):
            raise ValueError("Malformed IFC indexed profile segment")
        indices = raw[0]
        arc = kind == "IFCARCINDEX"
        if (kind not in ("IFCLINEINDEX", "IFCARCINDEX") or len(indices) < 2
                or (arc and len(indices) != 3)
                or any(type(i) is not int or not 1 <= i <= len(points) for i in indices)):
            raise ValueError("Invalid IFC indexed profile segment or point index")
        if previous is not None and previous != indices[0]:
            raise ValueError("Indexed profile segments do not join")
        if first is None:
            first = indices[0]
        previous = indices[-1]
        coords = [tuple(points[i - 1]) for i in indices]
        if any(len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                                  for v in p) for p in coords):
            raise ValueError("Indexed profile needs finite two-dimensional points")
        pieces = [tuple(coords)] if arc else list(zip(coords[:-1], coords[1:]))
        for edge in pieces:
            if len(set(edge)) != len(edge):
                raise ValueError("Indexed profile edge has repeated points")
            edges.append(edge)
    if not edges or (previous != first and tuple(points[previous-1]) != tuple(points[first-1])):
        raise ValueError("Indexed profile is not closed")
    return edges


def convex_arc_area(edges):
    """Signed chord areas plus circular segments, with a single convex winding."""
    import mpmath as mp
    if len(edges) > 512:
        raise ValueError("Indexed arc profile exceeds the certified edge count")
    with mp.workdps(70):
        def cross(a, b):
            return a.real*b.imag - a.imag*b.real

        def dot(a, b):
            return a.real*b.real + a.imag*b.imag

        def turn(a, b):
            return mp.atan2(cross(a, b), dot(a, b))

        area = mp.mpf(0)
        bends = []
        tangents = []
        origin = mp.mpc(*edges[0][0])
        for edge in edges:
            pts = [mp.mpc(*p) - origin for p in edge]
            a, b = pts[0], pts[-1]
            chord = b - a
            area += cross(a, b) / 2
            if len(pts) == 2:
                tangents.append((chord, chord))
                continue
            u, v = pts[1] - a, chord
            det = cross(u, v)
            if det == 0:
                raise ValueError("Collinear IFC circular arc control points")
            # The center is the intersection of the two perpendicular bisectors.
            center = a + 1j * (abs(v)**2 * u - abs(u)**2 * v) / (2*det)
            start, end = a - center, b - center
            sign = mp.sign(det)
            angle = turn(start, end)
            if sign * angle <= 0:
                angle += sign * 2 * mp.pi
            area += abs(start)**2 * (angle - mp.sin(angle)) / 2
            bends.append(angle)
            tangents.append((sign*1j*start, sign*1j*end))
        direction = mp.sign(area)
        for previous, current in zip(tangents[-1:] + tangents[:-1], tangents):
            u, v = previous[1], current[0]
            if dot(u, v) < 0 and abs(cross(u, v)) < mp.mpf('1e-55')*abs(u)*abs(v):
                raise ValueError("Reversing tangent at indexed profile vertex")
            bends.append(turn(u, v))
        # A closed piecewise circular curve with one complete monotone tangent
        # rotation is a convex boundary. This excludes star and repeated windings.
        if (not direction or any(direction*x < -mp.mpf('1e-55') for x in bends)
                or abs(sum(bends) - direction*2*mp.pi) > mp.mpf('1e-50')):
            raise ValueError("Only simple convex curved indexed profiles are qualified")
        return float(abs(area))


def _indexed_area(curve):
    if curve.SelfIntersect is True or not curve.Points.is_a("IfcCartesianPointList2D"):
        raise ValueError("Expected a non-self-intersecting two-dimensional indexed profile")
    points = curve.Points.CoordList
    segments = [(part.is_a().upper(), [list(part.wrappedValue)]) for part in curve.Segments]
    return convex_arc_area(ordered_segments(points, segments))


def has_direct_curved_profile(entities, product_rows):
    """Choose the native route only for a supported curve referenced by an actual Body."""
    def fields(ref, kind):
        if isinstance(ref, tuple) and len(ref) == 2 and ref[0] == "ref":
            actual, args = entities.get(ref[1], (None, []))
            if actual == kind:
                return args
        return []

    for _, _, _, shape in product_rows:
        for ref in shape[2]:
            rep = fields(ref, "IFCSHAPEREPRESENTATION")
            if len(rep) < 4 or str(rep[1]).upper() != "BODY" or not isinstance(rep[3], list):
                continue
            for item in rep[3]:
                sweep = fields(item, "IFCEXTRUDEDAREASOLID")
                profile = fields(sweep[0], "IFCARBITRARYCLOSEDPROFILEDEF") if sweep else []
                if len(profile) > 2 and fields(profile[2], "IFCRATIONALBSPLINECURVEWITHKNOTS"):
                    return True
                curve = fields(profile[2], "IFCINDEXEDPOLYCURVE") if len(profile) > 2 else []
                if len(curve) > 1 and isinstance(curve[1], list) and any(
                        isinstance(s, tuple) and s[0] == "IFCARCINDEX" for s in curve[1]):
                    return True
    return False


def curved_scene(ifc_text, product_rows, required_clearance, cross_only, discipline):
    """Measure direct curved Body occurrences using the source OCCT BRep, without tessellation."""
    import ifcopenshell
    import ifcopenshell.geom as geom
    from ifcopenshell import ifcopenshell_wrapper as wrapper
    from ifcopenshell.util.unit import calculate_unit_scale

    model = ifcopenshell.file.from_string(ifc_text)
    bodies = []
    curved_types = set()
    for step, kind, args, _ in product_rows:
        product = model.by_id(step)
        reps = [rep for rep in product.Representation.Representations
                if str(rep.RepresentationIdentifier).upper() not in ("AXIS", "BOX", "FOOTPRINT")]
        items = [item for rep in reps for item in rep.Items]
        ring = None
        for item in items:
            if item.is_a() == "IfcExtrudedAreaSolid" and item.SweptArea.is_a() == "IfcArbitraryClosedProfileDef":
                candidate = item.SweptArea.OuterCurve
                if candidate.is_a("IfcIndexedPolyCurve") and candidate.Segments and any(
                        part.is_a("IfcArcIndex") for part in candidate.Segments):
                    ring = candidate
                    curved_types.add("arc")
                elif candidate.is_a() == "IfcRationalBSplineCurveWithKnots":
                    ring = candidate
                    curved_types.add("rational")
        bodies.append((product, kind, args, reps, items, ring))
    if not curved_types:
        raise ValueError("Source and native parsers disagree on the curved profile roster")
    if model.by_type("IfcRelVoidsElement"):
        raise ValueError("Indexed-curve BRep does not qualify relational openings")
    scale = calculate_unit_scale(model)
    if "rational" in curved_types:
        from hamon_ifc_rational_profiles import rational_scene
        return rational_scene(model, bodies, scale, required_clearance, cross_only, discipline)
    settings = geom.settings()
    settings.set("iterator-output", wrapper.NATIVE)
    settings.set("use-world-coords", True)
    shapes = []
    volumes = []
    for product, kind, args, reps, items, ring in bodies:
        if (len(reps) != 1 or str(reps[0].RepresentationIdentifier).upper() != "BODY"
                or str(reps[0].RepresentationType).upper() != "SWEPTSOLID"
                or len(items) != 1 or items[0].is_a() != "IfcExtrudedAreaSolid"):
            raise ValueError("Indexed-curve scenes require one direct extruded Body per product")
        item = items[0]
        if ring is not None:
            area = _indexed_area(ring)
        elif item.SweptArea.is_a() == "IfcRectangleProfileDef":
            dims = item.SweptArea.XDim, item.SweptArea.YDim
            if not all(math.isfinite(d) and d > 0 for d in dims):
                raise ValueError("Invalid rectangular peer dimensions")
            area = math.prod(dims)
        elif item.SweptArea.is_a() == "IfcCircleProfileDef":
            if not math.isfinite(item.SweptArea.Radius) or item.SweptArea.Radius <= 0:
                raise ValueError("Invalid circular peer radius")
            area = math.pi * item.SweptArea.Radius**2
        else:
            raise ValueError("Unqualified peer profile in indexed-curve scene")
        ratios = item.ExtrudedDirection.DirectionRatios
        if (len(ratios) != 3 or not all(math.isfinite(v) for v in ratios) or ratios[2] == 0
                or not math.isfinite(item.Depth) or item.Depth <= 0):
            raise ValueError("Invalid indexed-scene extrusion depth or direction")
        expected = area * scale**2 * (item.Depth * scale) * abs(ratios[2]) / math.hypot(*ratios)
        element = geom.create_shape(settings, product, geometry_library="opencascade")
        if element is None:
            raise ValueError("OpenCASCADE could not construct the source Body")
        geometry = element.geometry
        if geometry.size() != 1 or geometry.item_id(0) != item.id():
            raise ValueError("BRep does not identify its single source Body item")
        shape = geometry.item(0)
        volume = shape.volume().to_double()
        if (not shape.is_manifold() or shape.surface_genus() != 0 or not math.isfinite(volume)
                or volume <= 0 or not math.isclose(volume, expected, rel_tol=1e-9, abs_tol=0)):
            raise ValueError("BRep volume disagrees with the source extrusion or lacks a closed boundary")
        volumes.append(volume)
        tree = geom.tree()
        tree.add_element(element)
        shapes.append((product, element, tree, discipline(kind, str(args[2] or ""))))
    rows = []
    for i, (left, element, _, left_discipline) in enumerate(shapes):
        for right, _, tree, right_discipline in shapes[i+1:]:
            if cross_only and left_discipline == right_discipline:
                continue
            # Extrema on two closed boundaries can be positive when one solid is
            # entirely inside the other. Test the common material first.
            common = tree.select(element)
            if common:
                if [p.id() for p in common] != [right.id()]:
                    raise ValueError("Unexpected peer in the BRep material intersection")
                d = 0.0
            else:
                # A missed range query supplies no distance. Enlarge the broad
                # search until the kernel returns this exact peer and one extremum.
                for exponent in range(13):
                    hits = tree.select(element, extend=10.0**exponent)
                    if hits:
                        break
                distances = tree.distances()
                if ([p.id() for p in hits] != [right.id()] or len(distances) != 1
                        or not math.isfinite(distances[0]) or distances[0] < 0):
                    raise ValueError("Source BRep minimum distance could not be established")
                d = distances[0]
            status = ("hard_clash" if d <= 1e-9 else
                      "clearance_violation" if d < required_clearance - 1e-9 else "clear")
            rows.append({"element_a": left.GlobalId, "element_b": right.GlobalId, "status": status,
                         "min_distance_m": d, "overlap_aabb_volume_m3": None, "witness_point_m": None,
                         "closest_points_m": None, "required_clearance_m": required_clearance})
    hard = [r for r in rows if r["status"] == "hard_clash"]
    violations = [r for r in rows if r["status"] == "clearance_violation"]
    return {"solid_count": len(shapes), "pair_count": len(rows), "hard_clash_count": len(hard),
            "clearance_violation_count": len(violations), "min_clearance_m": min(
                (r["min_distance_m"] for r in rows), default=None),
            "total_solid_volume_m3": math.fsum(volumes), "clashes": hard + violations, "all_pairs": rows,
            "geometry_basis": "source-indexed circular arcs and OpenCASCADE solid BReps"}
