"""networkxrs must return exactly what NetworkX returns.

Every accelerated function is called, with a range of arguments, on every
graph in the zoo (see conftest.py), once through ``networkx`` and once
through ``networkxrs``. The two outcomes must match: same values, same types
(int versus float), same dict and list order, and for bad input the same
exception type and message.
"""

import pytest
from conftest import ZOO, check, probes

import networkxrs

#: function name -> (builder, float tolerance or None for exact)
CASES = {}


def case(name, approx=None):
    """Register a builder: a generator of calls ``(module, G) -> result``."""

    def register(builder):
        CASES[name] = (builder, approx)
        return builder

    return register


def pairs(G):
    """(source, target) pairs, including a node paired with itself and a
    node that is not in the graph."""
    nodes = probes(G)
    return [(s, t) for s in nodes for t in nodes]


WEIGHTS = ["weight", "absent_attribute", None]

# --- traversal -----------------------------------------------------------------


@case("bfs_edges")
def _(G):
    for s in probes(G):
        yield lambda m, G, s=s: m.bfs_edges(G, s)
        yield lambda m, G, s=s: m.bfs_edges(G, s, reverse=True)
        for limit in (0, 1, 2, -1):
            yield lambda m, G, s=s, limit=limit: m.bfs_edges(G, s, depth_limit=limit)
        yield lambda m, G, s=s: m.bfs_edges(G, s, sort_neighbors=lambda ns: sorted(ns, key=repr))


@case("bfs_tree")
def _(G):
    for s in probes(G):
        yield lambda m, G, s=s: m.bfs_tree(G, s)
        yield lambda m, G, s=s: m.bfs_tree(G, s, reverse=True, depth_limit=2)


@case("bfs_successors")
def _(G):
    for s in probes(G):
        yield lambda m, G, s=s: m.bfs_successors(G, s)
        yield lambda m, G, s=s: m.bfs_successors(G, s, depth_limit=1)


@case("bfs_layers")
def _(G):
    for s in probes(G):
        yield lambda m, G, s=s: m.bfs_layers(G, s)
    yield lambda m, G: m.bfs_layers(G, list(G)[:1])


@case("descendants")
def _(G):
    for s in probes(G):
        yield lambda m, G, s=s: m.descendants(G, s)


@case("ancestors")
def _(G):
    for s in probes(G):
        yield lambda m, G, s=s: m.ancestors(G, s)


def _dfs_calls(name):
    def builder(G):
        yield lambda m, G: getattr(m, name)(G)
        for s in probes(G):
            yield lambda m, G, s=s: getattr(m, name)(G, s)
            yield lambda m, G, s=s: getattr(m, name)(G, source=s)
            for limit in (0, 1, 2, 3):
                yield lambda m, G, s=s, limit=limit: getattr(m, name)(G, s, depth_limit=limit)
        yield lambda m, G: getattr(m, name)(G, depth_limit=2)
        yield lambda m, G: getattr(m, name)(G, sort_neighbors=lambda ns: sorted(ns, key=repr))

    return builder


for _name in ("dfs_edges", "dfs_tree", "dfs_predecessors", "dfs_successors", "dfs_preorder_nodes", "dfs_postorder_nodes"):
    case(_name)(_dfs_calls(_name))

# --- components ----------------------------------------------------------------

for _name in (
    "connected_components",
    "number_connected_components",
    "is_connected",
    "weakly_connected_components",
    "number_weakly_connected_components",
    "is_weakly_connected",
    "strongly_connected_components",
    "number_strongly_connected_components",
    "is_strongly_connected",
    "topological_generations",
    "topological_sort",
    "is_directed_acyclic_graph",
    "is_tree",
    "is_forest",
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
):

    @case(_name)
    def _(G, name=_name):
        yield lambda m, G: getattr(m, name)(G)


@case("node_connected_component")
def _(G):
    for s in probes(G):
        yield lambda m, G, s=s: m.node_connected_component(G, s)


# --- DAG -----------------------------------------------------------------------


@case("dag_longest_path")
def _(G):
    for w in WEIGHTS:
        yield lambda m, G, w=w: m.dag_longest_path(G, weight=w)
    yield lambda m, G: m.dag_longest_path(G, default_weight=2.5)
    yield lambda m, G: m.dag_longest_path(G, weight="absent_attribute", default_weight=-1)


@case("dag_longest_path_length")
def _(G):
    for w in WEIGHTS:
        yield lambda m, G, w=w: m.dag_longest_path_length(G, weight=w)
    yield lambda m, G: m.dag_longest_path_length(G, default_weight=2.5)


# --- shortest paths ------------------------------------------------------------

CUTOFFS = [None, 0, 1, 2, 2.5, -1, float("inf")]


def _single_source(name, with_cutoff=True):
    def builder(G):
        for s in probes(G):
            yield lambda m, G, s=s: getattr(m, name)(G, s)
            if with_cutoff:
                for c in CUTOFFS:
                    yield lambda m, G, s=s, c=c: getattr(m, name)(G, s, cutoff=c)

    return builder


for _name in (
    "single_source_shortest_path",
    "single_source_shortest_path_length",
    "single_target_shortest_path",
    "single_target_shortest_path_length",
):
    case(_name)(_single_source(_name))


def _all_pairs(name, weighted):
    def builder(G):
        yield lambda m, G: getattr(m, name)(G)
        yield lambda m, G: getattr(m, name)(G, cutoff=2)
        if weighted:
            for w in WEIGHTS:
                yield lambda m, G, w=w: getattr(m, name)(G, weight=w)
            yield lambda m, G: getattr(m, name)(G, cutoff=3.5, weight="weight")

    return builder


for _name in ("all_pairs_shortest_path", "all_pairs_shortest_path_length"):
    case(_name)(_all_pairs(_name, weighted=False))
for _name in ("all_pairs_dijkstra", "all_pairs_dijkstra_path", "all_pairs_dijkstra_path_length"):
    case(_name)(_all_pairs(_name, weighted=True))


def _source_target(name, weighted=True):
    def builder(G):
        for s, t in pairs(G):
            yield lambda m, G, s=s, t=t: getattr(m, name)(G, s, t)
            if weighted:
                for w in WEIGHTS:
                    yield lambda m, G, s=s, t=t, w=w: getattr(m, name)(G, s, t, weight=w)

    return builder


for _name in ("bidirectional_shortest_path", "has_path"):
    case(_name)(_source_target(_name, weighted=False))
for _name in (
    "dijkstra_path",
    "dijkstra_path_length",
    "bidirectional_dijkstra",
    "bellman_ford_path",
    "bellman_ford_path_length",
):
    case(_name)(_source_target(_name))


def _weighted_single_source(name, cutoff, target):
    def builder(G):
        for s in probes(G):
            for w in WEIGHTS:
                yield lambda m, G, s=s, w=w: getattr(m, name)(G, s, weight=w)
            if cutoff:
                for c in (0, 2, 3.5, 100):
                    yield lambda m, G, s=s, c=c: getattr(m, name)(G, s, cutoff=c)
        if target:
            for s, t in pairs(G):
                yield lambda m, G, s=s, t=t: getattr(m, name)(G, s, target=t)
                if cutoff:
                    yield lambda m, G, s=s, t=t: getattr(m, name)(G, s, target=t, cutoff=3)

    return builder


case("single_source_dijkstra")(_weighted_single_source("single_source_dijkstra", True, True))
case("single_source_dijkstra_path")(_weighted_single_source("single_source_dijkstra_path", True, False))
case("single_source_dijkstra_path_length")(_weighted_single_source("single_source_dijkstra_path_length", True, False))
case("single_source_bellman_ford")(_weighted_single_source("single_source_bellman_ford", False, True))
case("single_source_bellman_ford_path")(_weighted_single_source("single_source_bellman_ford_path", False, False))
case("single_source_bellman_ford_path_length")(
    _weighted_single_source("single_source_bellman_ford_path_length", False, False)
)


def _general(name):
    def builder(G):
        nodes = probes(G)
        for method in ("dijkstra", "bellman-ford", "nonsense"):
            for w in ("weight", None):
                kw = {"weight": w, "method": method}
                yield lambda m, G, kw=kw: getattr(m, name)(G, **kw)
                for s in nodes:
                    yield lambda m, G, s=s, kw=kw: getattr(m, name)(G, source=s, **kw)
                    yield lambda m, G, s=s, kw=kw: getattr(m, name)(G, target=s, **kw)
                for s, t in pairs(G):
                    yield lambda m, G, s=s, t=t, kw=kw: getattr(m, name)(G, s, t, **kw)

    return builder


case("shortest_path")(_general("shortest_path"))
case("shortest_path_length")(_general("shortest_path_length"))


@case("average_shortest_path_length")
def _(G):
    yield lambda m, G: m.average_shortest_path_length(G)
    yield lambda m, G: m.average_shortest_path_length(G, weight="weight")
    yield lambda m, G: m.average_shortest_path_length(G, weight="weight", method="bellman-ford")


# --- centrality ----------------------------------------------------------------


# PageRank agrees to rounding, not bit for bit: SciPy sums in another order.
@case("pagerank", approx=1e-12)
def _(G):
    yield lambda m, G: m.pagerank(G)
    yield lambda m, G: m.pagerank(G, alpha=0.6, weight=None)
    yield lambda m, G: m.pagerank(G, tol=1e-10, max_iter=500)
    yield lambda m, G: m.pagerank(G, max_iter=2)
    nodes = list(G)
    if nodes:
        some = {n: i + 1 for i, n in enumerate(nodes[::3])}
        yield lambda m, G: m.pagerank(G, personalization=some)
        yield lambda m, G: m.pagerank(G, nstart=some, dangling=some)
        yield lambda m, G: m.pagerank(G, personalization=dict.fromkeys(nodes, 0))


@case("betweenness_centrality")
def _(G):
    for normalized in (True, False):
        for endpoints in (True, False):
            for w in (None, "weight"):
                kw = {"normalized": normalized, "endpoints": endpoints, "weight": w}
                yield lambda m, G, kw=kw: m.betweenness_centrality(G, **kw)
    yield lambda m, G: m.betweenness_centrality(G, k=len(G))
    yield lambda m, G: m.betweenness_centrality(G, k=max(1, len(G) // 2), seed=7)


@case("closeness_centrality")
def _(G):
    for wf in (True, False):
        yield lambda m, G, wf=wf: m.closeness_centrality(G, wf_improved=wf)
        yield lambda m, G, wf=wf: m.closeness_centrality(G, distance="weight", wf_improved=wf)
    for u in probes(G):
        yield lambda m, G, u=u: m.closeness_centrality(G, u=u)
        yield lambda m, G, u=u: m.closeness_centrality(G, u, "weight")


# --- trees and cycles ----------------------------------------------------------


def _spanning(name, edges):
    def builder(G):
        yield lambda m, G: getattr(m, name)(G)
        for w in WEIGHTS:
            yield lambda m, G, w=w: getattr(m, name)(G, weight=w)
        yield lambda m, G: getattr(m, name)(G, algorithm="prim")
        yield lambda m, G: getattr(m, name)(G, algorithm="nonsense")
        if edges:
            yield lambda m, G: getattr(m, name)(G, data=False)
            yield lambda m, G: getattr(m, name)(G, keys=False, ignore_nan=True)

    return builder


case("minimum_spanning_edges")(_spanning("minimum_spanning_edges", True))
case("maximum_spanning_edges")(_spanning("maximum_spanning_edges", True))
case("minimum_spanning_tree")(_spanning("minimum_spanning_tree", False))
case("maximum_spanning_tree")(_spanning("maximum_spanning_tree", False))


@case("cycle_basis")
def _(G):
    yield lambda m, G: m.cycle_basis(G)
    for root in probes(G)[:-1]:
        yield lambda m, G, root=root: m.cycle_basis(G, root)


# --- the test ------------------------------------------------------------------


def test_every_accelerated_function_has_cases():
    assert sorted(CASES) == networkxrs.accelerated()


@pytest.fixture(params=["eager", "adaptive"])
def policy(request):
    """Run under both policies: "eager" sends every call it can to Rust,
    "adaptive" mixes NetworkX and Rust the way real use does, including
    generators that start in one and finish in the other."""
    networkxrs.set_policy(request.param)
    yield request.param
    networkxrs.set_policy("adaptive")


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("graph", sorted(ZOO))
def test_same_as_networkx(name, graph, policy):
    builder, approx = CASES[name]
    G = ZOO[graph]
    for i, call in enumerate(builder(G)):
        try:
            check(call, G, approx)
        except AssertionError as failure:
            # Say which of the builder's calls disagreed.
            defaults = call.__defaults__ or ()
            raise AssertionError(f"{name} call #{i} with {defaults!r} on {graph}:\n{failure}") from None
