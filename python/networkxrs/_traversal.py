"""Breadth-first and depth-first search.

``bfs_predecessors`` is left to NetworkX: it is deprecated as of NetworkX 3.7.
"""

import networkx as _nx

from ._dispatch import GLOBAL, LOCAL, Fallback, accelerate, accelerate_stream, depth, from_list, reach

__all__ = [
    "bfs_edges",
    "bfs_tree",
    "bfs_successors",
    "bfs_layers",
    "descendants",
    "ancestors",
    "dfs_edges",
    "dfs_tree",
    "dfs_predecessors",
    "dfs_successors",
    "dfs_preorder_nodes",
    "dfs_postorder_nodes",
]


def _bfs_edges(snap, source, reverse, depth_limit, sort_neighbors):
    if sort_neighbors is not None:
        raise Fallback
    return snap.bfs_edges(source, bool(reverse) and snap.directed, depth(depth_limit))


@accelerate_stream(_nx.bfs_edges)
def bfs_edges(snap, G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    return from_list(_bfs_edges(snap, source, reverse, depth_limit, sort_neighbors))


def _bfs_tree_tier(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    return reach(source, reverse) if depth_limit is None else LOCAL


@accelerate(_nx.bfs_tree, tier=_bfs_tree_tier)
def bfs_tree(snap, G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    edges = _bfs_edges(snap, source, reverse, depth_limit, sort_neighbors)
    T = _nx.DiGraph()
    T.add_node(source)
    T.add_edges_from(edges)
    return T


def _group_successors(source, edges):
    parent = source
    children = []
    for p, c in edges:
        if p == parent:
            children.append(c)
            continue
        yield (parent, children)
        children = [c]
        parent = p
    yield (parent, children)


@accelerate_stream(_nx.bfs_successors)
def bfs_successors(snap, G, source, depth_limit=None, sort_neighbors=None):
    edges = _bfs_edges(snap, source, False, depth_limit, sort_neighbors)
    return from_list(list(_group_successors(source, edges)))


@accelerate_stream(_nx.bfs_layers)
def bfs_layers(snap, G, sources):
    # NetworkX also accepts an iterable of sources; only the single-node form
    # is handled here.
    return from_list(snap.bfs_layers(sources))


@accelerate(_nx.descendants, tier=lambda G, source: reach(source))
def descendants(snap, G, source):
    return snap.reachable(source, False)


@accelerate(_nx.ancestors, tier=lambda G, source: reach(source, True))
def ancestors(snap, G, source):
    return snap.reachable(source, snap.directed)


def _dfs_edges(snap, source, depth_limit, sort_neighbors):
    if sort_neighbors is not None:
        raise Fallback
    return snap.dfs_edges(source, depth(depth_limit))


def _dfs_tier(G, source=None, depth_limit=None, *, sort_neighbors=None):
    if source is None:
        return GLOBAL
    return reach(source) if depth_limit is None else LOCAL


@accelerate_stream(_nx.dfs_edges)
def dfs_edges(snap, G, source=None, depth_limit=None, *, sort_neighbors=None):
    return from_list(_dfs_edges(snap, source, depth_limit, sort_neighbors))


@accelerate(_nx.dfs_tree, tier=_dfs_tier)
def dfs_tree(snap, G, source=None, depth_limit=None, *, sort_neighbors=None):
    edges = _dfs_edges(snap, source, depth_limit, sort_neighbors)
    T = _nx.DiGraph()
    if source is None:
        T.add_nodes_from(G)
    else:
        T.add_node(source)
    T.add_edges_from(edges)
    return T


@accelerate(_nx.dfs_predecessors, tier=_dfs_tier)
def dfs_predecessors(snap, G, source=None, depth_limit=None, *, sort_neighbors=None):
    return {t: s for s, t in _dfs_edges(snap, source, depth_limit, sort_neighbors)}


@accelerate(_nx.dfs_successors, tier=_dfs_tier)
def dfs_successors(snap, G, source=None, depth_limit=None, *, sort_neighbors=None):
    successors = {}
    for s, t in _dfs_edges(snap, source, depth_limit, sort_neighbors):
        successors.setdefault(s, []).append(t)
    return successors


@accelerate_stream(_nx.dfs_preorder_nodes)
def dfs_preorder_nodes(snap, G, source=None, depth_limit=None, *, sort_neighbors=None):
    if sort_neighbors is not None:
        raise Fallback
    return from_list(snap.dfs_nodes(source, depth(depth_limit), False))


@accelerate_stream(_nx.dfs_postorder_nodes)
def dfs_postorder_nodes(snap, G, source=None, depth_limit=None, *, sort_neighbors=None):
    if sort_neighbors is not None:
        raise Fallback
    return from_list(snap.dfs_nodes(source, depth(depth_limit), True))
