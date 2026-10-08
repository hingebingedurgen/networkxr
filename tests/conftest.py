"""Shared helpers: a zoo of graphs and a strict comparison."""

import math
import random
import types

import networkx
import pytest

import networkxr


def strict_equal(a, b, approx=None, path="result"):
    """Assert a == b, and also that types and dict order match.

    ``1 == 1.0`` and ``{1: 2, 3: 4} == {3: 4, 1: 2}`` in Python, but a
    drop-in replacement should return the same types in the same order.
    """
    assert type(a) is type(b), f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{path}: dict keys or their order differ"
        for key in a:
            strict_equal(a[key], b[key], approx, f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            strict_equal(x, y, approx, f"{path}[{i}]")
    elif isinstance(a, float):
        if approx is None:
            assert a == b or (math.isnan(a) and math.isnan(b)), f"{path}: {a!r} != {b!r}"
        else:
            assert a == pytest.approx(b, rel=approx, abs=approx), f"{path}: {a!r} != {b!r}"
    elif isinstance(a, networkx.Graph):
        strict_equal(dict(a.nodes(data=True)), dict(b.nodes(data=True)), approx, path + ".nodes")
        strict_equal(list(a.edges(data=True)), list(b.edges(data=True)), approx, path + ".edges")
        strict_equal(a.graph, b.graph, approx, path + ".graph")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def materialise(value):
    """Turn generators into lists, recursively through pairs."""
    if isinstance(value, (types.GeneratorType, map, zip)):
        return [materialise(v) for v in value]
    if isinstance(value, tuple):
        return tuple(materialise(v) for v in value)
    return value


def outcome(call, module, G):
    """Run ``call(module, G)`` and describe what happened.

    Returns ``("ok", value)`` or ``("raise", when, type, message)``, where
    ``when`` says whether the exception came from the call itself or only
    once the returned generator was read. NetworkX is particular about that,
    and code that wraps a call in try/except depends on it.
    """
    try:
        result = call(module, G)
    except Exception as exc:  # noqa: BLE001 - comparing exceptions is the point
        return ("raise", "at call", type(exc), str(exc))
    try:
        return ("ok", materialise(result))
    except Exception as exc:  # noqa: BLE001
        return ("raise", "when read", type(exc), str(exc))


def check(call, G, approx=None):
    """Assert networkxr and networkx agree on ``call``, results or errors."""
    expected = outcome(call, networkx, G)
    actual = outcome(call, networkxr, G)
    assert actual[0] == expected[0], f"networkx: {expected!r}\nnetworkxr: {actual!r}"
    if expected[0] == "raise":
        assert actual[1:] == expected[1:]
    else:
        strict_equal(actual[1], expected[1], approx)


# --- the graph zoo -----------------------------------------------------------


def _shuffled(G, seed):
    """A copy of G with nodes and edges inserted in a random order, so that
    insertion order differs from sorted order."""
    rng = random.Random(seed)
    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))
    rng.shuffle(nodes)
    rng.shuffle(edges)
    H = G.__class__()
    H.add_nodes_from(nodes)
    for u, v, data in edges:
        if not G.is_directed() and rng.random() < 0.5:
            u, v = v, u
        H.add_edge(u, v, **data)
    return H


def _weighted(G, seed, kind):
    """Add a 'weight' attribute to the edges of G."""
    rng = random.Random(seed)
    for u, v, data in G.edges(data=True):
        if kind == "int":  # few distinct values, so many ties
            data["weight"] = rng.randint(1, 4)
        elif kind == "float":
            data["weight"] = rng.random() * 10
        elif kind == "halves":  # floats with many ties
            data["weight"] = rng.randint(1, 6) / 2
        elif kind == "zeros":  # zero-weight edges
            data["weight"] = rng.choice([0, 0, 1, 2])
        elif kind == "sparse":  # some edges lack the attribute
            if rng.random() < 0.6:
                data["weight"] = rng.randint(1, 5)
        elif kind == "mixed":  # ints and floats together
            data["weight"] = rng.choice([1, 2, 1.5, 3.0])
        elif kind == "negative":
            data["weight"] = rng.randint(-2, 6)
    return G


def _random_dag(n, p, seed):
    rng = random.Random(seed)
    G = networkx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p)
    return G


def _with_self_loops(G):
    nodes = list(G)
    G.add_edges_from((n, n) for n in nodes[::7])
    return G


def _build_zoo():
    nx = networkx
    zoo = {}
    for cls in (nx.Graph, nx.DiGraph):
        d = "di" if cls is nx.DiGraph else ""
        zoo[d + "empty"] = cls()
        one = cls()
        one.add_node("only")
        zoo[d + "single"] = one
        two = cls()
        two.add_nodes_from([0, 1])
        zoo[d + "two_isolated"] = two
        zoo[d + "path"] = nx.path_graph(7, create_using=cls)
        zoo[d + "cycle"] = nx.cycle_graph(6, create_using=cls)
        zoo[d + "complete"] = nx.complete_graph(6, create_using=cls)
        # Built by hand: older NetworkX releases refuse a directed star_graph.
        star = cls()
        star.add_nodes_from(range(6))
        star.add_edges_from((0, leaf) for leaf in range(1, 6))
        zoo[d + "star"] = star
        base = nx.gnp_random_graph(45, 0.07, seed=3, directed=cls is nx.DiGraph)
        zoo[d + "sparse_random"] = _shuffled(base, 1)
        dense = nx.gnp_random_graph(30, 0.25, seed=4, directed=cls is nx.DiGraph)
        zoo[d + "dense_random"] = _shuffled(dense, 2)
        zoo[d + "self_loops"] = _with_self_loops(_shuffled(base.copy(), 5))
        for kind in ("int", "float", "halves", "zeros", "sparse", "mixed", "negative"):
            zoo[f"{d}weighted_{kind}"] = _weighted(_shuffled(dense.copy(), 6), 7, kind)
        zoo[d + "weighted_sparse_random"] = _weighted(_shuffled(base.copy(), 8), 9, "int")
    zoo["grid"] = nx.grid_2d_graph(4, 5)  # tuple nodes
    zoo["karate"] = nx.karate_club_graph()
    zoo["strings"] = nx.relabel_nodes(nx.gnp_random_graph(25, 0.15, seed=11), lambda n: f"n{n}")
    zoo["mixed_node_types"] = nx.relabel_nodes(
        nx.gnp_random_graph(12, 0.3, seed=12), {0: "a", 1: (1, 2), 2: 2.5, 3: None.__class__, 4: frozenset({1})}
    )
    zoo["tree"] = _shuffled(nx.random_labeled_tree(25, seed=13), 14)
    zoo["forest"] = nx.union(nx.path_graph(4), nx.star_graph(3), rename=("a", "b"))
    dag = _random_dag(30, 0.15, 15)
    zoo["dag"] = _shuffled(dag, 16)
    zoo["dag_weighted_int"] = _weighted(_shuffled(dag.copy(), 17), 18, "int")
    zoo["dag_weighted_float"] = _weighted(_shuffled(dag.copy(), 19), 20, "float")
    zoo["dag_weighted_zeros"] = _weighted(_shuffled(dag.copy(), 21), 22, "zeros")
    zoo["dag_weighted_negative"] = _weighted(_shuffled(dag.copy(), 23), 24, "negative")
    zoo["dag_weighted_mixed"] = _weighted(_shuffled(dag.copy(), 25), 26, "mixed")
    negative_cycle = nx.DiGraph()
    negative_cycle.add_weighted_edges_from([(0, 1, 1), (1, 2, -3), (2, 0, 1), (2, 3, 4)])
    zoo["negative_cycle"] = negative_cycle
    return zoo


ZOO = _build_zoo()


@pytest.fixture(params=sorted(ZOO))
def G(request):
    return ZOO[request.param]


def probes(G, count=3):
    """A few nodes of G to use as sources and targets, plus one that is not
    in G."""
    nodes = list(G)
    if not nodes:
        return ["missing"]
    step = max(1, len(nodes) // count)
    return nodes[::step][:count] + ["missing"]
