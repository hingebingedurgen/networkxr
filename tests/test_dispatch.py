"""Checks on the dispatch itself: what ran in Rust and what fell back.

These tests use the "eager" policy, under which a call goes to Rust whenever
it can. The decisions the default policy makes are tested in test_policy.py.
"""

import copy
import pickle

import networkx
import pytest

import networkxrs
from networkxrs._dispatch import snapshot


@pytest.fixture(autouse=True)
def eager():
    networkxrs.set_policy("eager")
    yield
    networkxrs.set_policy("adaptive")


def test_every_function_runs_in_rust_for_some_input():
    """Guards against a function that silently always falls back, which the
    equivalence tests alone would not notice."""
    from conftest import ZOO, outcome
    from test_equivalence import CASES

    networkxrs.dispatch_counts(reset=True)
    for builder, _ in CASES.values():
        for G in ZOO.values():
            for call in builder(G):
                outcome(call, networkxrs, G)
    counts = networkxrs.dispatch_counts(reset=True)
    never = [name for name in networkxrs.accelerated() if counts.get(name, {}).get("rust", 0) == 0]
    assert never == []
    # Under the eager policy most calls on these plain graphs should be Rust.
    rust = sum(c["rust"] for c in counts.values())
    total = rust + sum(c["networkx"] for c in counts.values())
    assert rust > 0.5 * total


def _counts_for(name, call):
    networkxrs.dispatch_counts(reset=True)
    result = call()
    if hasattr(result, "__next__"):
        list(result)
    return networkxrs.dispatch_counts(reset=True).get(name, {"rust": 0, "networkx": 0})


def test_plain_graph_runs_in_rust():
    G = networkx.path_graph(5)
    assert _counts_for("shortest_path", lambda: networkxrs.shortest_path(G, 0, 4)) == {"rust": 1, "networkx": 0}


@pytest.mark.parametrize(
    "make",
    [
        lambda: networkx.MultiGraph([(0, 1), (0, 1), (1, 2)]),
        lambda: networkx.MultiDiGraph([(0, 1), (1, 2)]),
        lambda: networkx.path_graph(5).subgraph([0, 1, 2]),
        lambda: networkx.path_graph(5, create_using=networkx.DiGraph).reverse(copy=False),
        lambda: networkx.path_graph(5).to_undirected(as_view=True),
    ],
    ids=["multigraph", "multidigraph", "subgraph_view", "reverse_view", "same_type_view"],
)
def test_unsupported_graphs_fall_back_and_agree(make):
    G = make()
    counts = _counts_for("shortest_path_length", lambda: networkxrs.shortest_path_length(G, 0))
    assert counts == {"rust": 0, "networkx": 1}
    assert networkxrs.shortest_path_length(G, 0) == networkx.shortest_path_length(G, 0)


@pytest.mark.parametrize(
    "weight",
    [lambda u, v, d: 1, 1.5, True, "nan", "inf", "decimal", "huge", "string"],
    ids=repr,
)
def test_unsupported_weights_fall_back_and_agree(weight):
    import decimal

    G = networkx.path_graph(4)
    values = {
        "nan": float("nan"),
        "inf": float("inf"),
        "decimal": decimal.Decimal(2),
        "huge": 10**30,
        "string": "heavy",
    }
    if weight in values:
        networkx.set_edge_attributes(G, values[weight], weight)
    else:
        networkx.set_edge_attributes(G, weight, "w") if not callable(weight) else None
    key = "w" if weight in (1.5, True) else weight

    def run(module):
        try:
            # repr, because nan != nan would make equal results look unequal
            return repr(module.single_source_dijkstra_path_length(G, 0, weight=key))
        except Exception as exc:  # noqa: BLE001
            return type(exc), str(exc)

    assert run(networkxrs) == run(networkx)
    if weight != 1.5:
        counts = _counts_for("single_source_dijkstra_path_length", lambda: run(networkxrs))
        assert counts == {"rust": 0, "networkx": 1}


def test_snapshot_is_rebuilt_after_every_kind_of_mutation():
    G = networkx.Graph([(0, 1), (1, 2)])
    mutations = [
        lambda: G.add_edge(2, 3),
        lambda: G.add_node(9),
        lambda: G.add_edges_from([(3, 4), (9, 0)]),
        lambda: G.add_weighted_edges_from([(4, 5, 2.0)]),
        lambda: G.remove_edge(0, 1),
        lambda: G.remove_node(9),
        lambda: G.add_nodes_from([20, 21]),
        lambda: G.remove_nodes_from([20]),
        lambda: G.remove_edges_from([(3, 4)]),
        lambda: G.update(edges=[(0, 1), (3, 4)]),
        lambda: G.clear_edges(),
        lambda: G.add_path(range(6)) if hasattr(G, "add_path") else networkx.add_path(G, range(6)),
        lambda: G.clear(),
    ]
    for mutate in mutations:
        before = snapshot(G)
        mutate()
        assert snapshot(G) is not before
        assert dict(networkxrs.all_pairs_shortest_path_length(G)) == dict(networkx.all_pairs_shortest_path_length(G))
        assert list(networkxrs.connected_components(G)) == list(networkx.connected_components(G))


def test_view_sees_changes_to_the_graph_it_shows():
    G = networkx.path_graph(4)
    view = G.to_undirected(as_view=True)  # shares G's adjacency dicts
    assert networkxrs.shortest_path(view, 0, 3) == [0, 1, 2, 3]
    G.add_edge(0, 3)
    assert networkxrs.shortest_path(view, 0, 3) == [0, 3]


def test_subclasses_are_accelerated_only_if_they_behave_like_graph():
    class Tagged(networkx.Graph):  # adds something, overrides nothing
        def tag(self):
            return "tagged"

    class Quiet(networkx.Graph):  # overrides a writer
        def add_edge(self, u, v, **attr):
            self._adj.setdefault(u, {})[v] = self._adj.setdefault(v, {})[u] = attr
            self._node.setdefault(u, {})
            self._node.setdefault(v, {})

    class Complement(networkx.Graph):  # overrides a reader
        def neighbors(self, n):
            return iter(set(self._adj) - set(self._adj[n]) - {n})

    for cls, expected in ((Tagged, True), (Quiet, False), (Complement, False)):
        G = cls([(0, 1), (1, 2)])
        assert (snapshot(G) is not None) is expected
        assert networkxrs.shortest_path(G, 0, 2) == networkx.shortest_path(G, 0, 2)


def test_graph_with_caching_disabled_falls_back():
    G = networkx.path_graph(4)
    G.__networkx_cache__ = None  # NetworkX does this to graphs it edits directly
    assert snapshot(G) is None
    assert networkxrs.shortest_path(G, 0, 3) == [0, 1, 2, 3]


def test_snapshot_is_reused_when_nothing_changes():
    G = networkx.path_graph(4)
    networkxrs.shortest_path(G, 0, 3)
    first = snapshot(G)
    networkxrs.pagerank(G)
    assert snapshot(G) is first


def test_weights_changed_in_place_are_seen_without_a_rebuild():
    G = networkx.Graph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 2, 5)])
    assert networkxrs.dijkstra_path(G, 0, 2) == [0, 1, 2]
    before = snapshot(G)
    G[0][2]["weight"] = 1  # NetworkX does not clear its cache for this
    assert snapshot(G) is before
    assert networkxrs.dijkstra_path(G, 0, 2) == [0, 2]
    G.edges[0, 2]["weight"] = 10
    assert networkxrs.dijkstra_path(G, 0, 2) == [0, 1, 2]
    del G[0][2]["weight"]  # now defaults to 1
    assert networkxrs.dijkstra_path(G, 0, 2) == [0, 2]


def test_graph_with_snapshot_can_be_copied_and_pickled():
    G = networkx.path_graph(4)
    networkxrs.shortest_path(G, 0, 3)
    assert snapshot(G) is not None
    for clone in (copy.deepcopy(G), pickle.loads(pickle.dumps(G)), G.copy()):
        clone.add_edge(0, 3)
        assert networkxrs.shortest_path(clone, 0, 3) == [0, 3]
    assert networkxrs.shortest_path(G, 0, 3) == [0, 1, 2, 3]
    # A shallow copy shares its adjacency with the original, in NetworkX too.
    shallow = copy.copy(G)
    shallow.add_edge(0, 3)
    assert networkxrs.shortest_path(G, 0, 3) == networkx.shortest_path(G, 0, 3) == [0, 3]


def test_unknown_keyword_goes_to_networkx_and_fails_there():
    """A parameter added by a newer NetworkX must reach NetworkX. With the
    installed NetworkX an invented one is an error, and it must be
    NetworkX's error."""
    G = networkx.path_graph(4)
    networkxrs.dispatch_counts(reset=True)
    for call in (networkxrs.pagerank, networkxrs.bfs_edges):
        with pytest.raises(TypeError) as ours:
            list(call(G, invented_parameter=1) or ())
        with pytest.raises(TypeError) as theirs:
            list(getattr(networkx, call.__name__)(G, invented_parameter=1) or ())
        assert str(ours.value) == str(theirs.value)
    counts = networkxrs.dispatch_counts(reset=True)
    assert counts["pagerank"]["rust"] == 0


def test_backend_keyword_is_passed_to_networkx():
    G = networkx.path_graph(4)
    counts = _counts_for("shortest_path", lambda: networkxrs.shortest_path(G, 0, 3, backend=None))
    assert counts == {"rust": 0, "networkx": 1}


def test_wrapper_looks_like_the_networkx_function():
    assert networkxrs.shortest_path.__name__ == "shortest_path"
    assert networkxrs.shortest_path.__doc__ == networkx.shortest_path.__doc__
    assert networkxrs.shortest_path.__wrapped__ is networkx.shortest_path
