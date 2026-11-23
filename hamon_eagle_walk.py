"""Find simple, source-addressed copper paths with demonstrated plated landings."""

from collections import deque


def _edge_id(start, end, source):
    return (min(start, end), max(start, end), source)


def _walk_edge(step):
    return _edge_id(tuple(step["start"]), tuple(step["end"]), tuple(step["source"]))


def _landing_conflict(steps, drills):
    """Return one consecutive subpath that cannot demonstrate a plated landing."""
    for i, step in enumerate(steps):
        diameter = drills.get(tuple(step["source"]))
        if diameter is None:
            continue
        for center, direction in ((step["start"], -1), (step["end"], 1)):
            if center[0] != "copper":
                continue
            j = i + direction
            while 0 <= j < len(steps):
                edge = steps[j]
                if edge["source"][0] != "wire":
                    break
                far = edge["start"] if direction < 0 else edge["end"]
                dx, dy = far[2] - center[2], far[3] - center[3]
                if 4 * (dx * dx + dy * dy) > diameter * diameter:
                    break
                j += direction
            else:
                # The path ended before it reached the annular copper.
                j = min(max(j, 0), len(steps) - 1)
                return steps[min(i, j) : max(i, j) + 1]
            if edge["source"][0] != "wire":
                # Include the non-wire boundary. A valid path can retain all of
                # the inner wires if it continues past this boundary instead.
                return steps[min(i, j) : max(i, j) + 1]
    return None


def _path(previous, destination):
    steps = []
    current = destination
    while previous[current] is not None:
        parent, source = previous[current]
        steps.append(dict(source=list(source), start=list(parent), end=list(current)))
        current = parent
    return list(reversed(steps))


def rooted_tree(edges, root):
    previous = {root: None}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor, source in edges[current]:
            if neighbor not in previous:
                previous[neighbor] = current, source
                queue.append(neighbor)
    return previous


def plated_walk(edges, previous, root, destination, drills, remaining):
    """Find a verified simple route while sharing a fallback-work bound across terminals."""
    initial = _path(previous, destination)
    conflict = _landing_conflict(initial, drills)
    if conflict is None:
        return initial, remaining, False

    # Within a simple path, a consecutive conflict's edges must stay together
    # in one orientation or the other. Every valid simple route omits at least
    # one of them. Explore these cuts; a drill-only shortcut cannot mask a
    # longer valid path. The unchanged ordinary tree is the fast path above.
    pending = deque()
    scheduled = {frozenset()}

    def alternatives(banned, bad):
        nonlocal remaining
        for step in bad:
            if remaining < len(banned) + 1:
                return False
            remaining -= len(banned) + 1
            cut = banned | {_walk_edge(step)}
            if cut not in scheduled:
                scheduled.add(cut)
                pending.append(cut)
        return True

    if not alternatives(frozenset(), conflict):
        return None, remaining, True
    while pending:
        banned = pending.popleft()
        parents = {root: None}
        queue = deque([root])
        while queue and destination not in parents:
            current = queue.popleft()
            for neighbor, source in edges[current]:
                if remaining == 0:
                    return None, remaining, True
                remaining -= 1
                if neighbor in parents or _edge_id(current, neighbor, source) in banned:
                    continue
                parents[neighbor] = current, source
                queue.append(neighbor)
        if destination not in parents:
            continue
        steps = _path(parents, destination)
        conflict = _landing_conflict(steps, drills)
        if conflict is None:
            return steps, remaining, False
        if not alternatives(banned, conflict):
            return None, remaining, True
    return None, remaining, False
