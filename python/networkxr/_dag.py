"""Directed acyclic graphs: topological order and longest paths."""

import networkx as _nx
from networkx.utils import pairwise

from ._dispatch import Fallback, accelerate, need_directed, snapshot

__all__ = [
    "topological_generations",
    "topological_sort",
    "is_directed_acyclic_graph",
    "dag_longest_path",
    "dag_longest_path_length",
    "dag_longest_paths",
]

_CYCLE_MESSAGE = "Graph contains a cycle or graph changed during iteration"


def _generations(generations, acyclic, flatten):
    # NetworkX yields what it can and raises at the end if there is a cycle.
    for generation in generations:
        if flatten:
            yield from generation
        else:
            yield generation
    if not acyclic:
        raise _nx.NetworkXUnfeasible(_CYCLE_MESSAGE)


@accelerate(_nx.topological_generations)
def topological_generations(snap, G):
    need_directed(snap)
    generations, acyclic = snap.topological_generations()
    return _generations(generations, acyclic, flatten=False)


@accelerate(_nx.topological_sort)
def topological_sort(snap, G):
    need_directed(snap)
    generations, acyclic = snap.topological_generations()
    return _generations(generations, acyclic, flatten=True)


@accelerate(_nx.is_directed_acyclic_graph)
def is_directed_acyclic_graph(snap, G):
    return snap.directed and snap.is_acyclic()


@accelerate(_nx.dag_longest_path)
def dag_longest_path(snap, G, weight="weight", default_weight=1, topo_order=None):
    need_directed(snap)
    if topo_order is not None:
        raise Fallback
    return snap.dag_longest_path(weight, default_weight)


@accelerate(_nx.dag_longest_path_length)
def dag_longest_path_length(snap, G, weight="weight", default_weight=1):
    need_directed(snap)
    path = snap.dag_longest_path(weight, default_weight)
    # Summed in Python, as NetworkX does, so the result is an int or a float
    # exactly when NetworkX's is.
    path_length = 0
    for u, v in pairwise(path):
        path_length += G[u][v].get(weight, default_weight)
    return path_length


def _dag_longest_paths_python(G, weight, default_weight, max_paths):
    """Pure-Python version of the Rust ``dag_longest_paths``.

    Used for graphs the Rust path does not take, such as graph views or
    weights that are not plain ints or floats.
    """
    order = list(_nx.topological_sort(G))
    pred = G.pred
    # Negate the weights, so the longest path becomes the shortest.
    negated = {}
    for u, v, w in G.edges(data=weight, default=default_weight):
        if w < 0:
            raise ValueError(
                f"dag_longest_paths requires non-negative weights; edge ({u!r}, {v!r}) has weight {w}"
            )
        negated[u, v] = -w
    # Shortest distances, every node starting at 0.
    dist = dict.fromkeys(order, 0)
    for v in order:
        for u in pred[v]:
            through_u = dist[u] + negated[u, v]
            if through_u < dist[v]:
                dist[v] = through_u
    if not dist or max_paths == 0:
        return []
    shortest = min(dist.values())
    # Walk back from every end node along edges that achieve the distance.
    paths = []
    for end in G:
        if dist[end] != shortest:
            continue
        path = [end]
        iterators = [iter(pred[end])]
        if dist[end] == 0:
            paths.append([end])
            if len(paths) == max_paths:
                return paths
        while path:
            v = path[-1]
            for u in iterators[-1]:
                if dist[u] + negated[u, v] == dist[v]:
                    path.append(u)
                    iterators.append(iter(pred[u]))
                    if dist[u] == 0:
                        paths.append(path[::-1])
                        if len(paths) == max_paths:
                            return paths
                    break
            else:
                path.pop()
                iterators.pop()
    return paths


def dag_longest_paths(G, weight="weight", default_weight=1, max_paths=None):
    """Return every longest path in a directed acyclic graph.

    :func:`networkx.dag_longest_path` returns one longest path. This returns
    all of them. It is an addition in networkxr; NetworkX has no equivalent.

    The method is to negate every edge weight, compute shortest distances
    (on a DAG, one pass in topological order), and return every path whose
    length equals the best distance.

    Parameters
    ----------
    G : DiGraph
        A directed acyclic graph.
    weight : str, optional
        Edge attribute holding the weight. Default ``"weight"``.
    default_weight : int or float, optional
        Weight of an edge without that attribute. Default 1.
    max_paths : int, optional
        Stop after finding this many paths. The number of longest paths can
        grow exponentially with the size of the graph.

    Returns
    -------
    list of lists
        Each inner list is a path, as nodes from start to end. All have the
        same total weight, the maximum over all paths in ``G``. For a graph
        with no nodes the result is ``[]``. For a graph with nodes but no
        edges, every node is a longest path (of weight 0) on its own.

    Raises
    ------
    NetworkXNotImplemented
        If ``G`` is undirected or a multigraph.
    NetworkXUnfeasible
        If ``G`` has a cycle.
    ValueError
        If an edge weight is negative.

    Notes
    -----
    A path is included if its total weight equals the maximum. With
    zero-weight edges this includes paths that extend one another: for
    ``a -(0)-> b -(5)-> c`` both ``[b, c]`` and ``[a, b, c]`` are returned.

    Weights are compared exactly. With float weights, two paths whose sums
    differ only by rounding error are not treated as equal.

    Examples
    --------
    >>> import networkxr as nx
    >>> G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    >>> nx.dag_longest_paths(G)
    [[0, 1, 3], [0, 2, 3]]
    """
    if not G.is_directed():
        raise _nx.NetworkXNotImplemented("not implemented for undirected type")
    if G.is_multigraph():
        raise _nx.NetworkXNotImplemented("not implemented for multigraph type")
    if max_paths is not None and (type(max_paths) is not int or max_paths < 0):
        raise ValueError("max_paths must be a non-negative int or None")
    snap = snapshot(G)
    if snap is not None:
        try:
            paths = snap.dag_longest_paths(weight, default_weight, max_paths)
        except Fallback:
            pass
        else:
            if paths is None:
                raise _nx.NetworkXUnfeasible(_CYCLE_MESSAGE)
            return paths
    return _dag_longest_paths_python(G, weight, default_weight, max_paths)
