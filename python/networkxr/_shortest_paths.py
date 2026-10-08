"""Shortest paths: unweighted, Dijkstra and Bellman-Ford."""

import functools
import math

import networkx as _nx

from ._dispatch import ACCELERATED, LOCAL, Fallback, NegativeCycle, accelerate, accelerate_stream, chunked, reach

__all__ = [
    "shortest_path",
    "shortest_path_length",
    "has_path",
    "average_shortest_path_length",
    "single_source_shortest_path",
    "single_source_shortest_path_length",
    "single_target_shortest_path",
    "single_target_shortest_path_length",
    "bidirectional_shortest_path",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    "dijkstra_path",
    "dijkstra_path_length",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "bidirectional_dijkstra",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "bellman_ford_path",
    "bellman_ford_path_length",
    "single_source_bellman_ford",
    "single_source_bellman_ford_path",
    "single_source_bellman_ford_path_length",
]

# What the Rust single-source methods should return.
LENGTHS, PATHS, BOTH = 0, 1, 2


def _hops(cutoff):
    """An unweighted ``cutoff`` as a whole number of edges, or None.

    NetworkX's loop runs while ``cutoff > level``, so a fractional cutoff
    behaves like its ceiling and a negative one like zero.
    """
    if cutoff is None:
        return None
    if type(cutoff) is int:
        return max(cutoff, 0)
    if type(cutoff) is float:
        if cutoff == math.inf:
            return None
        if math.isnan(cutoff) or cutoff <= 0:
            return 0
        return math.ceil(cutoff)
    raise Fallback


def _distance(cutoff):
    """A weighted ``cutoff`` as a float, or None."""
    if cutoff is None:
        return None
    if type(cutoff) is float and not math.isnan(cutoff):
        return cutoff
    if type(cutoff) is int and abs(cutoff) < 2**53:
        return float(cutoff)
    raise Fallback


def _need_node(G, node):
    if node not in G:
        raise Fallback  # NetworkX raises NodeNotFound with its own message


def _bellman_ford(snap, source, weight, reverse, want):
    try:
        return snap.bellman_ford(source, weight, 1, reverse, want)
    except NegativeCycle:
        raise _nx.NetworkXUnbounded("Negative cycle detected.") from None


# --- unweighted --------------------------------------------------------------


def _source_tier(G, source, cutoff=None):
    return reach(source) if cutoff is None else LOCAL


def _target_tier(G, target, cutoff=None):
    return reach(target, True) if cutoff is None else LOCAL


@accelerate(_nx.single_source_shortest_path_length, tier=_source_tier)
def single_source_shortest_path_length(snap, G, source, cutoff=None):
    return snap.bfs_lengths(source, _hops(cutoff), False)


@accelerate(_nx.single_source_shortest_path, tier=_source_tier)
def single_source_shortest_path(snap, G, source, cutoff=None):
    return snap.bfs_paths(source, _hops(cutoff), False)


@accelerate(_nx.single_target_shortest_path_length, tier=_target_tier)
def single_target_shortest_path_length(snap, G, target, cutoff=None):
    return snap.bfs_lengths(target, _hops(cutoff), True)


@accelerate(_nx.single_target_shortest_path, tier=_target_tier)
def single_target_shortest_path(snap, G, target, cutoff=None):
    return snap.bfs_paths(target, _hops(cutoff), True)


def _bidirectional_shortest_path(snap, source, target):
    path = snap.bidirectional_bfs(source, target)
    if path is None:
        raise _nx.NetworkXNoPath(f"No path between {source} and {target}.")
    return path


@accelerate(_nx.bidirectional_shortest_path, tier=LOCAL)
def bidirectional_shortest_path(snap, G, source, target):
    return _bidirectional_shortest_path(snap, source, target)


@accelerate(_nx.has_path, tier=LOCAL)
def has_path(snap, G, source, target):
    return snap.bidirectional_bfs(source, target) is not None


@accelerate_stream(_nx.all_pairs_shortest_path_length)
def all_pairs_shortest_path_length(snap, G, cutoff=None):
    hops = _hops(cutoff)
    return chunked(G, snap, lambda start, stop: snap.all_bfs(start, stop, hops, False))


@accelerate_stream(_nx.all_pairs_shortest_path)
def all_pairs_shortest_path(snap, G, cutoff=None):
    hops = _hops(cutoff)
    return chunked(G, snap, lambda start, stop: snap.all_bfs(start, stop, hops, True))


# --- Dijkstra ----------------------------------------------------------------


def _dijkstra_target(snap, G, source, target, weight, cutoff, no_path):
    result = snap.dijkstra_target(source, target, weight, 1, _distance(cutoff))
    if result is None:
        raise _nx.NetworkXNoPath(no_path)
    return result


def _dijkstra_source_tier(G, source, cutoff=None, weight="weight"):
    return reach(source) if cutoff is None else LOCAL


def _single_source_dijkstra_tier(G, source, target=None, cutoff=None, weight="weight"):
    return reach(source) if target is None and cutoff is None else LOCAL


@accelerate(_nx.single_source_dijkstra, tier=_single_source_dijkstra_tier)
def single_source_dijkstra(snap, G, source, target=None, cutoff=None, weight="weight"):
    if target is None:
        return snap.dijkstra(source, weight, 1, _distance(cutoff), False, BOTH)
    _need_node(G, source)
    if target == source:
        return (0, [target])
    return _dijkstra_target(snap, G, source, target, weight, cutoff, f"No path to {target}.")


@accelerate(_nx.single_source_dijkstra_path, tier=_dijkstra_source_tier)
def single_source_dijkstra_path(snap, G, source, cutoff=None, weight="weight"):
    return snap.dijkstra(source, weight, 1, _distance(cutoff), False, PATHS)


@accelerate(_nx.single_source_dijkstra_path_length, tier=_dijkstra_source_tier)
def single_source_dijkstra_path_length(snap, G, source, cutoff=None, weight="weight"):
    return snap.dijkstra(source, weight, 1, _distance(cutoff), False, LENGTHS)


@accelerate(_nx.dijkstra_path, tier=LOCAL)
def dijkstra_path(snap, G, source, target, weight="weight"):
    _need_node(G, source)
    if target == source:
        return [target]
    return _dijkstra_target(snap, G, source, target, weight, None, f"No path to {target}.")[1]


@accelerate(_nx.dijkstra_path_length, tier=LOCAL)
def dijkstra_path_length(snap, G, source, target, weight="weight"):
    _need_node(G, source)
    if source == target:
        return 0
    message = f"Node {target} not reachable from {source}"
    return _dijkstra_target(snap, G, source, target, weight, None, message)[0]


def _bidirectional_dijkstra(snap, G, source, target, weight):
    _need_node(G, source)
    _need_node(G, target)
    if source == target:
        return (0, [source])
    result = snap.bidirectional_dijkstra(source, target, weight, 1)
    if result is None:
        raise _nx.NetworkXNoPath(f"No path between {source} and {target}.")
    return result


@accelerate(_nx.bidirectional_dijkstra, tier=LOCAL)
def bidirectional_dijkstra(snap, G, source, target, weight="weight"):
    return _bidirectional_dijkstra(snap, G, source, target, weight)


def _all_dijkstra(snap, G, cutoff, weight, want):
    distance = _distance(cutoff)
    return chunked(G, snap, lambda start, stop: snap.all_dijkstra(start, stop, weight, 1, distance, want))


@accelerate_stream(_nx.all_pairs_dijkstra)
def all_pairs_dijkstra(snap, G, cutoff=None, weight="weight"):
    return _all_dijkstra(snap, G, cutoff, weight, BOTH)


@accelerate_stream(_nx.all_pairs_dijkstra_path_length)
def all_pairs_dijkstra_path_length(snap, G, cutoff=None, weight="weight"):
    return _all_dijkstra(snap, G, cutoff, weight, LENGTHS)


@accelerate_stream(_nx.all_pairs_dijkstra_path)
def all_pairs_dijkstra_path(snap, G, cutoff=None, weight="weight"):
    return _all_dijkstra(snap, G, cutoff, weight, PATHS)


# --- Bellman-Ford ------------------------------------------------------------


def _bellman_ford_target(snap, source, target, weight):
    try:
        return snap.bellman_ford_target(source, target, weight, 1)
    except NegativeCycle:
        raise _nx.NetworkXUnbounded("Negative cycle detected.") from None


# Bellman-Ford always relaxes everything its source can reach, even when asked
# about one target.
def _bellman_ford_tier(G, source, *args, **kwargs):
    return reach(source)


@accelerate(_nx.single_source_bellman_ford, tier=_bellman_ford_tier)
def single_source_bellman_ford(snap, G, source, target=None, weight="weight"):
    if source == target:
        _need_node(G, source)
        return (0, [source])
    if target is None:
        return _bellman_ford(snap, source, weight, False, BOTH)
    result = _bellman_ford_target(snap, source, target, weight)
    if result is None:
        raise _nx.NetworkXNoPath(f"Target {target} cannot be reached from given sources")
    return result


@accelerate(_nx.single_source_bellman_ford_path, tier=_bellman_ford_tier)
def single_source_bellman_ford_path(snap, G, source, weight="weight"):
    return _bellman_ford(snap, source, weight, False, PATHS)


@accelerate(_nx.single_source_bellman_ford_path_length, tier=_bellman_ford_tier)
def single_source_bellman_ford_path_length(snap, G, source, weight="weight"):
    return _bellman_ford(snap, source, weight, False, LENGTHS)


@accelerate(_nx.bellman_ford_path, tier=_bellman_ford_tier)
def bellman_ford_path(snap, G, source, target, weight="weight"):
    if source == target:
        _need_node(G, source)
        return [source]
    result = _bellman_ford_target(snap, source, target, weight)
    if result is None:
        raise _nx.NetworkXNoPath(f"Target {target} cannot be reached from given sources")
    return result[1]


@accelerate(_nx.bellman_ford_path_length, tier=_bellman_ford_tier)
def bellman_ford_path_length(snap, G, source, target, weight="weight"):
    if source == target:
        _need_node(G, source)
        return 0
    result = _bellman_ford_target(snap, source, target, weight)
    if result is None:
        raise _nx.NetworkXNoPath(f"node {target} not reachable from {source}")
    return result[0]


# --- the general entry points -------------------------------------------------
#
# These mirror the branching in networkx.shortest_path and
# networkx.shortest_path_length, calling the Rust methods directly.


def _method(weight, method):
    if method not in ("dijkstra", "bellman-ford"):
        raise Fallback  # NetworkX raises ValueError
    return "unweighted" if weight is None else method


def _general_tier(G, source=None, target=None, weight=None, method="dijkstra"):
    if target is None:
        return reach(source)
    if source is None:
        return reach(target, True)
    if weight is not None and method == "bellman-ford":
        return reach(source)
    return LOCAL


@accelerate(_nx.shortest_path, tier=_general_tier)
def _shortest_path(snap, G, source=None, target=None, weight=None, method="dijkstra"):
    method = _method(weight, method)
    if source is None:
        if target is None:
            raise Fallback  # all pairs; see shortest_path below
        # Every path that ends at target: search backwards from it.
        if method == "unweighted":
            return snap.bfs_paths(target, None, True)
        if method == "dijkstra":
            return snap.dijkstra(target, weight, 1, None, True, PATHS)
        return _bellman_ford(snap, target, weight, True, PATHS)
    if target is None:
        if method == "unweighted":
            return snap.bfs_paths(source, None, False)
        if method == "dijkstra":
            return snap.dijkstra(source, weight, 1, None, False, PATHS)
        return _bellman_ford(snap, source, weight, False, PATHS)
    if method == "unweighted":
        return _bidirectional_shortest_path(snap, source, target)
    if method == "dijkstra":
        return _bidirectional_dijkstra(snap, G, source, target, weight)[1]
    return bellman_ford_path.__networkxr_impl__(snap, G, source, target, weight)


@accelerate(_nx.shortest_path_length, tier=_general_tier)
def _shortest_path_length(snap, G, source=None, target=None, weight=None, method="dijkstra"):
    method = _method(weight, method)
    if source is None:
        if target is None:
            raise Fallback  # all pairs; see shortest_path_length below
        if method == "unweighted":
            return snap.bfs_lengths(target, None, True)
        if method == "dijkstra":
            return snap.dijkstra(target, weight, 1, None, True, LENGTHS)
        return _bellman_ford(snap, target, weight, True, LENGTHS)
    if target is None:
        if method == "unweighted":
            return snap.bfs_lengths(source, None, False)
        if method == "dijkstra":
            return snap.dijkstra(source, weight, 1, None, False, LENGTHS)
        return _bellman_ford(snap, source, weight, False, LENGTHS)
    if method == "unweighted":
        return len(_bidirectional_shortest_path(snap, source, target)) - 1
    if method == "dijkstra":
        return dijkstra_path_length.__networkxr_impl__(snap, G, source, target, weight)
    return bellman_ford_path_length.__networkxr_impl__(snap, G, source, target, weight)


# With neither a source nor a target these functions return all pairs, as a
# generator. That form is routed to the all-pairs functions above, which know
# how to serve a generator lazily.


@functools.wraps(_nx.shortest_path)
def shortest_path(G, source=None, target=None, weight=None, method="dijkstra", **kwargs):
    if source is None and target is None and not kwargs:
        if weight is None and method in ("dijkstra", "bellman-ford"):
            return all_pairs_shortest_path(G)
        if method == "dijkstra":
            return all_pairs_dijkstra_path(G, weight=weight)
    return _shortest_path(G, source, target, weight, method, **kwargs)


@functools.wraps(_nx.shortest_path_length)
def shortest_path_length(G, source=None, target=None, weight=None, method="dijkstra", **kwargs):
    if source is None and target is None and not kwargs:
        if weight is None and method in ("dijkstra", "bellman-ford"):
            return all_pairs_shortest_path_length(G)
        if method == "dijkstra":
            return all_pairs_dijkstra_path_length(G, weight=weight)
    return _shortest_path_length(G, source, target, weight, method, **kwargs)


ACCELERATED["shortest_path"] = shortest_path
ACCELERATED["shortest_path_length"] = shortest_path_length


@accelerate(_nx.average_shortest_path_length)
def average_shortest_path_length(snap, G, weight=None, method=None):
    if method is None:
        method = "unweighted" if weight is None else "dijkstra"
    if method not in ("unweighted", "dijkstra"):
        raise Fallback
    n = len(G)
    if n == 0:
        raise Fallback  # NetworkX raises NetworkXPointlessConcept
    if n == 1:
        return 0
    connected = len(snap.strong_components()) == 1 if snap.directed else snap.is_connected()
    if not connected:
        raise Fallback  # NetworkX raises NetworkXError
    if method == "unweighted":
        total = snap.bfs_distance_sum()
    else:
        # The same expression NetworkX uses. Python's `sum` compensates for
        # float rounding, so a hand-written loop would not give the same
        # total.
        every_source = _all_dijkstra(snap, G, None, weight, LENGTHS)(0)
        total = sum(length for _, lengths in every_source for length in lengths.values())
    return total / (n * (n - 1))
