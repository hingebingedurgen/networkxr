"""The default ("adaptive") policy: when a snapshot is built and when not.

The aim of the policy is that networkxr is never much slower than NetworkX.
A snapshot costs time proportional to the size of the graph, so it must not
be built for a call NetworkX would answer by looking at a handful of nodes.
"""

import networkx
import pytest

import networkxr
from networkxr import _dispatch


def built(G):
    """True if G currently has a snapshot."""
    return bool(G.__networkx_cache__.get(_dispatch._SNAPSHOT))


def debt(G):
    return G.__networkx_cache__.get(_dispatch._DEBT, 0.0)


@pytest.fixture
def snapshots_expensive(monkeypatch):
    """Make a snapshot cost far more than any test spends in NetworkX, so
    that "does not build" tests check the rules and not the speed of the
    machine they happen to run on."""
    monkeypatch.setattr(_dispatch, "BUILD_SECONDS_PER_UNIT", 1.0)


@pytest.fixture
def snapshots_free(monkeypatch):
    """The opposite: any time at all spent in NetworkX justifies a snapshot."""
    monkeypatch.setattr(_dispatch, "BUILD_SECONDS_PER_UNIT", 1e-12)
    monkeypatch.setattr(_dispatch, "RUST_SECONDS_PER_UNIT", 1e-12)


@pytest.fixture
def big():
    """A connected graph large enough that the cost model's thresholds are
    well above the cost of one small query."""
    return networkx.connected_watts_strogatz_graph(20_000, 6, 0.1, seed=1)


@pytest.fixture
def islands():
    """Many small components: any one search reaches very little."""
    G = networkx.Graph()
    for i in range(0, 20_000, 4):
        networkx.add_path(G, range(i, i + 4))
    return G


def test_global_function_builds_at_once(big):
    networkxr.number_connected_components(big)
    assert built(big)


def test_point_to_point_query_does_not_build(big, snapshots_expensive):
    assert networkxr.has_path(big, 0, 1)
    assert networkxr.shortest_path(big, 0, 1) == [0, 1]
    assert networkxr.dijkstra_path_length(big, 0, 1) == 1
    assert not built(big)
    assert debt(big) > 0


def test_enough_point_to_point_queries_build(big, snapshots_free):
    # With snapshots nearly free, the first query's cost pays for one.
    networkxr.has_path(big, 0, 10_000)
    assert not built(big)  # the decision is made before a call, not after
    networkxr.has_path(big, 0, 10_000)
    assert built(big)
    counts_before = networkxr.dispatch_counts(reset=True)
    assert networkxr.has_path(big, 0, 10_000)
    assert networkxr.dispatch_counts()["has_path"] == {"rust": 1, "networkx": 0}
    del counts_before


def test_changing_the_graph_resets_the_count(big, snapshots_expensive):
    networkxr.has_path(big, 0, 1)
    assert debt(big) > 0
    big.add_edge(0, 5000)
    assert debt(big) == 0.0
    assert not built(big)


def test_a_loop_that_edits_then_queries_never_builds(big, snapshots_expensive):
    for i in range(50):
        big.add_edge(i, i + 7000)
        assert networkxr.has_path(big, i, i + 7000)
        assert not built(big)


def test_single_source_search_builds_when_it_reaches_far(big):
    networkxr.single_source_shortest_path_length(big, 0)
    assert built(big)


def test_single_source_search_does_not_build_when_it_reaches_little(islands, snapshots_expensive):
    assert networkxr.single_source_shortest_path_length(islands, 0) == {0: 0, 1: 1, 2: 2, 3: 3}
    assert networkxr.descendants(islands, 5) == {4, 6, 7}
    assert networkxr.node_connected_component(islands, 9) == {8, 9, 10, 11}
    assert networkxr.single_source_dijkstra_path_length(islands, 0) == {0: 0, 1: 1, 2: 2, 3: 3}
    assert not networkxr.is_connected(islands)
    assert not built(islands)


def test_search_with_a_cutoff_does_not_build(big, snapshots_expensive):
    networkxr.single_source_shortest_path_length(big, 0, cutoff=1)
    networkxr.single_source_dijkstra_path_length(big, 0, cutoff=1)
    assert not built(big)


def test_reverse_search_probes_predecessors(snapshots_expensive):
    # 0 -> 1 -> ... -> 9999: everything reaches the last node, nothing is
    # reachable from it.
    G = networkx.path_graph(10_000, create_using=networkx.DiGraph)
    networkxr.descendants(G, 9_999)
    assert not built(G)
    networkxr.ancestors(G, 9_999)
    assert built(G)


def test_partly_read_generator_does_not_build(big, snapshots_expensive):
    edges = networkxr.bfs_edges(big, 0)
    assert next(edges) == next(networkx.bfs_edges(big, 0))
    edges.close()
    assert not built(big)
    for _, lengths in networkxr.all_pairs_shortest_path_length(big, cutoff=1):
        break
    assert not built(big)


def test_fully_read_generator_switches_to_rust_midway(big, snapshots_free):
    networkxr.dispatch_counts(reset=True)
    assert list(networkxr.bfs_edges(big, 0)) == list(networkx.bfs_edges(big, 0))
    assert list(networkxr.dfs_preorder_nodes(big, 0)) == list(networkx.dfs_preorder_nodes(big, 0))
    assert built(big)
    counts = networkxr.dispatch_counts()
    assert counts["bfs_edges"]["rust"] == 1
    assert counts["dfs_preorder_nodes"]["rust"] == 1


def test_generator_stays_with_networkx_if_the_graph_changes_midway(big, snapshots_free):
    expected = networkx.bfs_edges(big, 0)
    actual = networkxr.bfs_edges(big, 0)
    assert next(actual) == next(expected)
    big.add_node("new")  # does not affect a search already under way from 0
    assert list(actual) == list(expected)


def test_generator_reads_the_graph_when_first_asked_not_when_created():
    G = networkx.path_graph(3)
    networkxr.set_policy("eager")
    try:
        networkxr.number_connected_components(G)  # G now has a snapshot
        edges = networkxr.bfs_edges(G, 0)
        G.add_edge(2, 3)
        assert list(edges) == [(0, 1), (1, 2), (2, 3)]
    finally:
        networkxr.set_policy("adaptive")


def test_off_policy_never_uses_rust(big):
    networkxr.set_policy("off")
    try:
        networkxr.dispatch_counts(reset=True)
        networkxr.number_connected_components(big)
        list(networkxr.bfs_edges(big, 0))
        assert not built(big)
        counts = networkxr.dispatch_counts()
        assert counts["number_connected_components"] == {"rust": 0, "networkx": 1}
        assert counts["bfs_edges"] == {"rust": 0, "networkx": 1}
    finally:
        networkxr.set_policy("adaptive")


def test_set_policy_rejects_unknown_names():
    with pytest.raises(ValueError):
        networkxr.set_policy("fast")
    assert networkxr.get_policy() == "adaptive"


def test_small_searches_on_a_built_graph_are_cheap(islands):
    """With a snapshot in place, a search that reaches four nodes must not
    cost time proportional to the 20,000-node graph."""
    import time

    networkxr.set_policy("eager")
    try:
        networkxr.number_connected_components(islands)

        def per_call(function, repeat=2000):
            start = time.perf_counter()
            for i in range(repeat):
                function(i)
            return (time.perf_counter() - start) / repeat

        ours = per_call(lambda i: networkxr.single_source_dijkstra_path_length(islands, i))
        theirs = per_call(lambda i: networkx.single_source_dijkstra_path_length(islands, i))
        # Very generous, because shared CI machines time things erratically.
        # A per-call cost that grew with the graph would be hundreds of
        # times NetworkX's here, so this still catches it.
        assert ours < 20 * theirs
        ours = per_call(lambda i: networkxr.has_path(islands, i, i + 1))
        theirs = per_call(lambda i: networkx.has_path(islands, i, i + 1))
        assert ours < 20 * theirs
    finally:
        networkxr.set_policy("adaptive")
