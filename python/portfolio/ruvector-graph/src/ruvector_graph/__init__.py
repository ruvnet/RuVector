"""Property graph traversal backed by RuVector Rust bindings."""
from collections import deque
from ruvector import GraphDB

__all__ = ["GraphDB", "reachable"]

def reachable(graph: GraphDB, start: str, *, max_depth: int = 3, max_nodes: int = 1000) -> list[str]:
    """Return existing nodes in breadth first order, including start.

    Caps node visits and depth. Raises on a node budget overflow rather
    than returning an apparently complete but truncated answer.
    """
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    if graph.get_node(start) is None:
        raise KeyError(start)
    seen = {start}
    result = [start]
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth == max_depth:
            continue
        for edge in graph.get_outgoing_edges(node):
            target = edge["to"]
            if target in seen:
                continue
            if len(seen) >= max_nodes:
                raise ValueError("max_nodes budget exceeded")
            seen.add(target)
            result.append(target)
            queue.append((target, depth + 1))
    return result
