"""Connected, weakly connected and strongly connected components."""

import networkx as _nx

from ._dispatch import GLOBAL, Fallback, accelerate, as_generator, need_directed, need_undirected, reach

__all__ = [
    "connected_components",
    "number_connected_components",
    "is_connected",
    "node_connected_component",
    "weakly_connected_components",
    "number_weakly_connected_components",
    "is_weakly_connected",
    "strongly_connected_components",
    "number_strongly_connected_components",
    "is_strongly_connected",
]


@accelerate(_nx.connected_components)
def connected_components(snap, G):
    need_undirected(snap)
    return as_generator(snap.components())


@accelerate(_nx.number_connected_components)
def number_connected_components(snap, G):
    need_undirected(snap)
    return len(snap.components())


def _first_node_tier(G):
    # NetworkX searches from the first node and compares what it finds with
    # the size of the graph. If that node sits in a small component the call
    # is over almost at once.
    for node in G._adj:
        return reach(node)
    return GLOBAL


@accelerate(_nx.is_connected, tier=_first_node_tier)
def is_connected(snap, G):
    need_undirected(snap)
    if snap.number_of_nodes == 0:
        raise Fallback  # NetworkX raises NetworkXPointlessConcept
    return snap.is_connected()


@accelerate(_nx.node_connected_component, tier=lambda G, n: reach(n))
def node_connected_component(snap, G, n):
    need_undirected(snap)
    return snap.node_component(n)


@accelerate(_nx.weakly_connected_components)
def weakly_connected_components(snap, G):
    need_directed(snap)
    return as_generator(snap.components())


@accelerate(_nx.number_weakly_connected_components)
def number_weakly_connected_components(snap, G):
    need_directed(snap)
    return len(snap.components())


@accelerate(_nx.is_weakly_connected)
def is_weakly_connected(snap, G):
    need_directed(snap)
    if snap.number_of_nodes == 0:
        raise Fallback
    return snap.is_connected()


@accelerate(_nx.strongly_connected_components)
def strongly_connected_components(snap, G):
    need_directed(snap)
    return as_generator(snap.strong_components())


@accelerate(_nx.number_strongly_connected_components)
def number_strongly_connected_components(snap, G):
    need_directed(snap)
    return len(snap.strong_components())


@accelerate(_nx.is_strongly_connected)
def is_strongly_connected(snap, G):
    need_directed(snap)
    if snap.number_of_nodes == 0:
        raise Fallback
    return len(snap.strong_components()) == 1
