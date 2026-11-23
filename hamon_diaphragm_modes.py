"""Separate membrane deformation, connector extension, and strain-free motions.

A common edge welds two membrane panels at zero strain; a shared point
only enforces equal displacement there. Bar constraints then split the
compatible patch motions into those that do and do not extend a connector.
Independent ground reactions must restrain the latter. The material and
spring stiffnesses enter the solve after this kinematic decomposition.
"""
from collections import defaultdict

import numpy as np
from scipy.linalg import svd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


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
    modes = np.zeros((2*len(xy), width))
    owners = [set() for _ in xy]
    joints = []
    start = 0
    for body, patch in enumerate(patches):
        slots = 2 if len(patch) == 1 else 3
        base = xy[patch[0]]
        offsets = xy[patch] - base
        span = np.max(abs(offsets))
        if not np.isfinite(span):
            raise ValueError("nonfinite diaphragm patch extent")
        for node, (dx, dy) in zip(patch, offsets):
            velocity = np.zeros((2, width))
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


def _component(xy, patches, bars, ground):
    rigid, owners, joints = _patch_motions(xy, patches)
    compatible = None
    if joints:
        _, compatible = _row_spaces(joints, rigid.shape[1])
    bar_rows = []
    elastic_only_bars = []
    for a, b, direction in bars:
        internal = not owners[a].isdisjoint(owners[b])
        elastic_only_bars.append(internal)
        if not internal:
            row = direction @ (rigid[2*b:2*b+2] - rigid[2*a:2*a+2])
            if compatible is not None:
                projected = row @ compatible
                # A brace on a hinge's rotation line can be exactly redundant.
                # Compare against the unprojected geometry before normalizing.
                tol = np.finfo(float).eps * len(row) * np.linalg.norm(row)
                if np.linalg.norm(projected) <= tol:
                    elastic_only_bars[-1] = True
                    continue
                row = projected
            bar_rows.append(row)
    if compatible is not None:
        rigid = rigid @ compatible
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
    return free, free_dofs, straining, flexible_dofs, elastic_only_bars


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
    for label, members in nodes_by_group.items():
        local = {node: i for i, node in enumerate(members)}
        patches = [[local[n] for n in patch] for patch in patches_by_group[label]]
        links = [(local[a], local[b], axis) for _, a, b, axis in bars_by_group[label]]
        nodes = np.asarray(members)
        dofs = np.column_stack((2*nodes, 2*nodes+1)).ravel()
        free, free_dofs, flex, flex_dofs, internal = _component(xy[nodes], patches, links, ground[dofs])
        for (index, _, _, _), same_patch in zip(bars_by_group[label], internal):
            elastic_only_bars[index] = same_patch
        for block, indices in [(free, free_dofs), (flex, flex_dofs)]:
            r, c = np.nonzero(block)
            rows.extend(dofs[r])
            cols.extend(dofs[indices[c]])
            data.extend(block[r, c])
            panel_elastic[dofs[indices]] = False
        bar_free[dofs[free_dofs]] = True
    ordinary = np.flatnonzero(panel_elastic)
    rows.extend(ordinary)
    cols.extend(ordinary)
    data.extend(np.ones(len(ordinary)))
    transform = coo_matrix((data, (rows, cols)), shape=(len(ground), len(ground))).tocsc()
    return transform, panel_elastic, bar_free, elastic_only_bars
