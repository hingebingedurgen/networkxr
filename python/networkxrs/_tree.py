"""Spanning trees, tree recognition and cycle bases."""

import networkx as _nx

from ._dispatch import LOCAL, Fallback, accelerate, as_generator, need_undirected

__all__ = [
    "minimum_spanning_edges",
    "maximum_spanning_edges",
    "minimum_spanning_tree",
    "maximum_spanning_tree",
    "is_tree",
    "is_forest",
    "cycle_basis",
]


def _kruskal(snap, algorithm, weight, data, minimum):
    need_undirected(snap)
    if algorithm != "kruskal":
        raise Fallback
    # NaN weights (and `ignore_nan`) are handled by NetworkX: reading a NaN
    # weight raises Fallback.
    return snap.kruskal(weight, 1, minimum, bool(data))


@accelerate(_nx.minimum_spanning_edges)
def minimum_spanning_edges(snap, G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False):
    return as_generator(_kruskal(snap, algorithm, weight, data, True))


@accelerate(_nx.maximum_spanning_edges)
def maximum_spanning_edges(snap, G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False):
    return as_generator(_kruskal(snap, algorithm, weight, data, False))


def _tree_from(G, edges):
    T = G.__class__()
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(edges)
    return T


@accelerate(_nx.minimum_spanning_tree)
def minimum_spanning_tree(snap, G, weight="weight", algorithm="kruskal", ignore_nan=False):
    return _tree_from(G, _kruskal(snap, algorithm, weight, True, True))


@accelerate(_nx.maximum_spanning_tree)
def maximum_spanning_tree(snap, G, weight="weight", algorithm="kruskal", ignore_nan=False):
    return _tree_from(G, _kruskal(snap, algorithm, weight, True, False))


# NetworkX first compares the node and edge counts, which settles most
# graphs that are not trees without any search.
@accelerate(_nx.is_tree, tier=LOCAL)
def is_tree(snap, G):
    if snap.number_of_nodes == 0:
        raise Fallback
    return snap.is_tree()


@accelerate(_nx.is_forest)
def is_forest(snap, G):
    if snap.number_of_nodes == 0:
        raise Fallback
    return snap.is_forest()


@accelerate(_nx.cycle_basis)
def cycle_basis(snap, G, root=None):
    need_undirected(snap)
    return snap.cycle_basis(root)
