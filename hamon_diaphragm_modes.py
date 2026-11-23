"""Separate membrane deformation, connector extension, and strain-free motions.

A common edge welds two membrane panels at zero strain; a shared point
only enforces equal displacement there. Bar constraints then split the
compatible patch motions into those that do and do not extend a connector.
Independent ground reactions must restrain the latter. The material and
spring stiffnesses enter the solve after this kinematic decomposition.
"""
from collections import defaultdict
from fractions import Fraction
from typing import NamedTuple

import numpy as np
from scipy.linalg import svd
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.csgraph import connected_components


class DiaphragmBasis(NamedTuple):
    transform: csc_matrix
    panel_elastic: np.ndarray
    bar_free: np.ndarray
    elastic_only_bars: np.ndarray
    pinned_rows: list
    pinned_motion: list


def _groups(size, edges):
    rows, columns = zip(*edges) if edges else ([], [])
    graph = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(size, size))
    _, labels = connected_components(graph, directed=False)
    return labels


def _row_spaces(rows, width):
    """Separate independent kinematic constraints from zero-extension motions."""
    if not rows:
        return np.empty((width, 0)), np.eye(width)
    matrix = np.asarray(rows)
    scale = np.linalg.norm(matrix, axis=1)
    matrix = matrix[scale > 0] / scale[scale > 0, None]
    if not len(matrix):
        return np.empty((width, 0)), np.eye(width)
    _, singular, right = svd(matrix, full_matrices=len(matrix) < width)
    threshold = np.finfo(float).eps * max(matrix.shape) * singular[0]
    rank = np.count_nonzero(singular > threshold)
    return right[:rank].T, right[rank:].T


def _independent_rows(modes, weights):
    """Select independent measurements, preferring strong supported directions."""
    residual = modes.copy()
    norms = np.linalg.norm(modes, axis=1)
    cutoff = np.finfo(float).eps * max(1, modes.shape[1]) * np.max(norms, initial=0)
    chosen = []
    for _ in range(modes.shape[1]):
        length = np.linalg.norm(residual, axis=1)
        usable = (weights > 0) & (length > cutoff)
        if not np.any(usable):
            raise ValueError("unsupported diaphragm: unrestrained rotation or translation")
        candidates = np.flatnonzero(usable)
        score = np.log(weights[candidates]) + 2*np.log(length[candidates])
        index = candidates[np.argmax(score)]
        direction = residual[index] / length[index]
        residual -= (residual @ direction)[:, None] * direction
        residual[index] = 0
        chosen.append(index)
    return np.asarray(chosen, dtype=int)


def _patch_motions(xy, patches):
    """Independent rigid patch motions, with displacement continuity at common pins."""
    width = sum(2 if len(patch) == 1 else 3 for patch in patches)
    dtype = object if xy.dtype == object else float
    modes = np.zeros((2*len(xy), width), dtype=dtype)
    owners = [set() for _ in xy]
    joints = []
    start = 0
    for body, patch in enumerate(patches):
        slots = 2 if len(patch) == 1 else 3
        base = xy[patch[0]]
        offsets = xy[patch] - base
        span = np.max(abs(offsets))
        if xy.dtype != object and not np.isfinite(span):
            raise ValueError("nonfinite diaphragm patch extent")
        for node, (dx, dy) in zip(patch, offsets):
            velocity = np.zeros((2, width), dtype=dtype)
            velocity[0, start] = velocity[1, start+1] = 1
            if slots == 3:
                velocity[:, start+2] = [-dy/span, dx/span]
            rows = slice(2*node, 2*node+2)
            if owners[node]:
                joints.extend(velocity - modes[rows])
            else:
                modes[rows] = velocity
            owners[node].add(body)
        start += slots
    return modes, owners, joints


def _exact_kernel(rows, width):
    """Nullspace of geometric constraints and independent coordinate columns."""
    matrix = np.array(rows, dtype=object).reshape(len(rows), width)
    pivots = []
    for j in range(width):
        i = len(pivots)
        pivot = next((k for k in range(i, len(matrix)) if matrix[k, j]), None)
        if pivot is None:
            continue
        matrix[[i, pivot]] = matrix[[pivot, i]]
        matrix[i] /= Fraction(matrix[i, j])
        for k in range(len(matrix)):
            if k != i and matrix[k, j]:
                matrix[k] -= matrix[k, j] * matrix[i]
        pivots.append(j)
    remaining = [j for j in range(width) if j not in pivots]
    kernel = np.zeros((width, len(remaining)), dtype=object)
    kernel[remaining] = np.eye(len(remaining), dtype=int)
    kernel[pivots] = -matrix[:len(pivots), remaining]
    return pivots, kernel


def _cardinal_exact(modes, weights):
    """Use independent nodal components as displacement coordinates."""
    basis = modes.copy()
    anchors = []
    for j in range(basis.shape[1]):
        # The strongest remaining weighted lever chooses a well-scaled anchor.
        # Only an exactly zero remaining row is kinematically dependent.
        available = [(Fraction(weight) * max(map(abs, row[j:]))**2, i)
                     for i, (weight, row) in enumerate(zip(weights, basis)) if weight]
        strength, anchor = max(available, default=(0, 0))
        if not strength:
            raise ValueError("unsupported diaphragm: unrestrained rotation or translation")
        pivot = max(range(j, basis.shape[1]), key=lambda col: abs(basis[anchor, col]))
        basis[:, [j, pivot]] = basis[:, [pivot, j]]
        basis[:, j] /= Fraction(basis[anchor, j])
        for k in range(basis.shape[1]):
            if k != j and basis[anchor, k]:
                basis[:, k] -= basis[anchor, k] * basis[:, j]
        anchors.append(anchor)
    return basis, np.asarray(anchors, dtype=int)


def _pinned_component(xy, patches, bars, ground):
    # Rigid patch and bar equations are rational in the supplied binary64
    # coordinates if the unnormalized bar chord is used. Keep them in that
    # form until the connector extension in each cardinal coordinate is known.
    exact_xy = np.array([[Fraction(x), Fraction(y)] for x, y in xy], dtype=object)
    rigid, _, joints = _patch_motions(exact_xy, patches)
    _, compatible = _exact_kernel(joints, rigid.shape[1])
    rigid = rigid @ compatible
    chords = [exact_xy[b] - exact_xy[a] for a, b, _ in bars]
    rows = [edge @ (rigid[2*b:2*b+2] - rigid[2*a:2*a+2])
            for edge, (a, b, _) in zip(chords, bars)]
    pivots, free_modes = _exact_kernel(rows, rigid.shape[1])
    free, support = _cardinal_exact(rigid @ free_modes, ground)
    straining = rigid[:, pivots]
    straining -= free @ straining[support]
    flex, moving = _cardinal_exact(straining, np.ones(len(ground)))

    projected = []
    for edge, (a, b, _) in zip(chords, bars):
        extension = edge @ (flex[2*b:2*b+2] - flex[2*a:2*a+2])
        length = Fraction(float(np.linalg.norm(xy[b] - xy[a])))
        projected.append(np.asarray(extension / length, dtype=float))
    return (free, support, flex, moving,
            [not any(row) for row in rows], projected)


def _component(xy, patches, bars, ground):
    rigid, owners, joints = _patch_motions(xy, patches)
    if joints:
        if any(owners[a].isdisjoint(owners[b]) for a, b, _ in bars):
            return _pinned_component(xy, patches, bars, ground)
        _, compatible = _row_spaces(joints, rigid.shape[1])
        rigid = rigid @ compatible
    bar_rows = []
    elastic_only_bars = []
    for a, b, direction in bars:
        internal = not owners[a].isdisjoint(owners[b])
        elastic_only_bars.append(internal)
        if not internal:
            bar_rows.append(direction @ (rigid[2*b:2*b+2] - rigid[2*a:2*a+2]))
    straining, free = _row_spaces(bar_rows, rigid.shape[1])
    free = rigid @ free
    straining = rigid @ straining

    # Give every fully strain-free motion its own independent grounded
    # displacement. A small positive spring remains usable even when other
    # restraints are much stiffer. An absent independent restraint is singular.
    free_dofs = _independent_rows(free, ground)
    if len(free_dofs):
        free = np.linalg.solve(free[free_dofs].T, free.T).T
        free[free_dofs] = np.eye(len(free_dofs))
        straining -= free @ straining[free_dofs]
        straining[free_dofs] = 0

    flexible_dofs = _independent_rows(straining, np.ones(len(ground)))
    if len(flexible_dofs):
        straining = np.linalg.solve(straining[flexible_dofs].T, straining.T).T
        straining[flexible_dofs] = np.eye(len(flexible_dofs))
    return free, free_dofs, straining, flexible_dofs, elastic_only_bars, None


def diaphragm_coordinates(xy, panels, bars, ground):
    """Return Cartesian map and masks for panel-elastic and bar-free coordinates."""
    panel_connections, structural_edges, edge_owner = [], [], {}
    for i, ns in enumerate(panels):
        structural_edges.extend((ns[0], node) for node in ns[1:])
        for a, b in zip(ns, ns[1:] + ns[:1]):
            edge = tuple(sorted((a, b)))
            if edge in edge_owner:
                panel_connections.append((edge_owner[edge], i))
            else:
                edge_owner[edge] = i
    panel_labels = _groups(len(panels), panel_connections)
    regions = defaultdict(set)
    for label, ns in zip(panel_labels, panels):
        regions[label].update(ns)
    structural_edges.extend((int(dofs[0]//2), int(dofs[2]//2)) for dofs, _, _ in bars)
    labels = _groups(len(xy), structural_edges)
    nodes_by_group = defaultdict(list)
    patches_by_group = defaultdict(list)
    bars_by_group = defaultdict(list)
    for node, label in enumerate(labels):
        nodes_by_group[label].append(node)
    covered = set()
    for ns in regions.values():
        patch = sorted(ns)
        patches_by_group[labels[patch[0]]].append(patch)
        covered.update(ns)
    for node in range(len(xy)):
        if node not in covered:
            patches_by_group[labels[node]].append([node])
    for index, (dofs, axis, _) in enumerate(bars):
        a, b = int(dofs[0]//2), int(dofs[2]//2)
        bars_by_group[labels[a]].append((index, a, b, axis[2:]))

    panel_elastic = np.ones(2*len(xy), dtype=bool)
    bar_free = np.zeros_like(panel_elastic)
    elastic_only_bars = np.zeros(len(bars), dtype=bool)
    rows, cols, data = [], [], []
    pinned_rows = []
    pinned_motion = []
    for label, members in nodes_by_group.items():
        local = {node: i for i, node in enumerate(members)}
        patches = [[local[n] for n in patch] for patch in patches_by_group[label]]
        links = [(local[a], local[b], axis) for _, a, b, axis in bars_by_group[label]]
        nodes = np.asarray(members)
        dofs = np.column_stack((2*nodes, 2*nodes+1)).ravel()
        free, free_dofs, flex, flex_dofs, internal, projected = _component(
            xy[nodes], patches, links, ground[dofs])
        if projected is not None:
            pinned_motion.append((dofs, dofs[np.r_[free_dofs, flex_dofs]],
                                  np.column_stack((free, flex))))
            pinned_rows.extend((bar[0], dofs[flex_dofs], values)
                               for bar, values in zip(bars_by_group[label], projected))
        for (index, _, _, _), same_patch in zip(bars_by_group[label], internal):
            elastic_only_bars[index] = same_patch
        for block, indices in [(free, free_dofs), (flex, flex_dofs)]:
            r, c = np.nonzero(block)
            rows.extend(dofs[r])
            cols.extend(dofs[indices[c]])
            data.extend(np.asarray(block[r, c], dtype=float))
            panel_elastic[dofs[indices]] = False
        bar_free[dofs[free_dofs]] = True
    ordinary = np.flatnonzero(panel_elastic)
    rows.extend(ordinary)
    cols.extend(ordinary)
    data.extend(np.ones(len(ordinary)))
    transform = coo_matrix((data, (rows, cols)), shape=(len(ground), len(ground))).tocsc()
    return DiaphragmBasis(transform, panel_elastic, bar_free, elastic_only_bars,
                          pinned_rows, pinned_motion)
