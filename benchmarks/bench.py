"""Time networkxrs against NetworkX (and rustworkx and rustnx, if installed).

    python benchmarks/bench.py            # full run, a few minutes
    python benchmarks/bench.py --quick    # smaller graphs, under a minute
    python benchmarks/bench.py --markdown # print Markdown tables
    python benchmarks/bench.py --scaling  # per-node algorithms at growing sizes

The first table times single calls. Three timings are reported for networkxrs:

- *cold*: the first call on a graph, which includes building the Rust
  snapshot of it (if the policy decides to build one);
- *warm*: a later call on the same, unchanged graph, which reuses the
  snapshot;
- *speed-up*: NetworkX time divided by the cold time, the pessimistic one.

The second table times whole usage patterns, including ones chosen to be
awkward for a snapshot-based design: many tiny queries, and queries
interleaved with edits to the graph.

Every networkxrs result is checked against NetworkX's before its time is
reported. rustworkx and rustnx results are not checked: neither promises
results identical to NetworkX's.

rustnx is another Rust engine for NetworkX, used as a NetworkX backend
(``networkx.pagerank(G, backend="rustnx")``). Like networkxrs it converts the
graph on the first call and caches the conversion, so it gets the same cold
and warm columns. An empty cell means rustnx does not implement the call.

``--scaling`` runs a different experiment: the algorithms that do one search
per node (betweenness and closeness), on graphs from 2,000 to 50,000 nodes.
NetworkX needs minutes to hours for the larger ones, so past a time budget
(``--budget``, 120 seconds per call by default) its time is not measured but
estimated: the search is run from 50 of the nodes and the time is multiplied
up. Estimates are marked with ``~``, and a row with an estimate is not
checked against NetworkX. Pass a large ``--budget`` to measure everything.
"""

import argparse
import functools
import gc
import math
import os
import platform
import random
import time

import networkx

import networkxrs

try:
    import rustworkx
except ImportError:
    rustworkx = None

try:
    import rustnx
except ImportError:
    rustnx = None
else:
    # NetworkX warns every time a backend reuses a cached conversion.
    networkx.config.warnings_to_ignore.add("cache")


class Backend:
    """Looks like the networkx module, but sends every call to one backend:
    ``Backend("rustnx").pagerank(G)`` is ``networkx.pagerank(G, backend="rustnx")``."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, function):
        return functools.partial(getattr(networkx, function), backend=self.name)


def timed_or_none(function):
    """`timed`, or None if the backend does not implement the call."""
    try:
        return timed(function)[0]
    except NotImplementedError:
        return None


def make_graph(n, degree, directed, seed, dag=False):
    """A random graph with n nodes, about ``degree`` edges per node, and
    integer edge weights."""
    rng = random.Random(seed)
    if dag:
        G = networkx.DiGraph()
        G.add_nodes_from(range(n))
        for _ in range(n * degree):
            u, v = rng.randrange(n), rng.randrange(n)
            if u != v:
                G.add_edge(min(u, v), max(u, v))
    else:
        G = networkx.fast_gnp_random_graph(n, degree / n, seed=seed, directed=directed)
    for _, _, data in G.edges(data=True):
        data["weight"] = rng.randint(1, 20)
    return G


def consume(value):
    """Exhaust generators so lazy results are fully computed."""
    if hasattr(value, "__next__"):
        return list(value)
    return value


def timed(function):
    start = time.perf_counter()
    result = consume(function())
    return time.perf_counter() - start, result


def same(a, b):
    if isinstance(a, dict) and a and isinstance(next(iter(a.values())), float):
        return a.keys() == b.keys() and all(math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-12) for k in a)
    if isinstance(a, networkx.Graph):
        return list(a.edges(data=True)) == list(b.edges(data=True))
    return a == b


# (label, graph kind, call). `m` is the module: networkx or networkxrs.
BENCHMARKS = [
    ("connected_components", "big", lambda m, G: m.connected_components(G)),
    ("bfs_edges", "big", lambda m, G: m.bfs_edges(G, 0)),
    ("dfs_preorder_nodes", "big", lambda m, G: m.dfs_preorder_nodes(G, 0)),
    ("single_source_shortest_path_length", "big", lambda m, G: m.single_source_shortest_path_length(G, 0)),
    ("single_source_shortest_path", "big", lambda m, G: m.single_source_shortest_path(G, 0)),
    ("shortest_path (one pair)", "big", lambda m, G: m.shortest_path(G, 0, len(G) - 1)),
    ("single_source_dijkstra_path_length", "big", lambda m, G: m.single_source_dijkstra_path_length(G, 0)),
    ("single_source_dijkstra", "big", lambda m, G: m.single_source_dijkstra(G, 0)),
    ("dijkstra_path (one pair)", "big", lambda m, G: m.dijkstra_path(G, 0, len(G) - 1)),
    ("single_source_bellman_ford_path_length", "big", lambda m, G: m.single_source_bellman_ford_path_length(G, 0)),
    ("minimum_spanning_tree", "big", lambda m, G: m.minimum_spanning_tree(G)),
    ("pagerank", "big", lambda m, G: m.pagerank(G)),
    ("degree_centrality", "big", lambda m, G: m.degree_centrality(G)),
    ("strongly_connected_components", "big_directed", lambda m, G: m.strongly_connected_components(G)),
    ("pagerank (directed)", "big_directed", lambda m, G: m.pagerank(G)),
    ("topological_sort", "big_dag", lambda m, G: m.topological_sort(G)),
    ("dag_longest_path", "big_dag", lambda m, G: m.dag_longest_path(G)),
    ("cycle_basis", "medium", lambda m, G: m.cycle_basis(G)),
    ("betweenness_centrality", "medium", lambda m, G: m.betweenness_centrality(G)),
    ("betweenness_centrality (weighted)", "medium", lambda m, G: m.betweenness_centrality(G, weight="weight")),
    ("closeness_centrality", "medium", lambda m, G: m.closeness_centrality(G)),
    ("all_pairs_shortest_path_length", "medium", lambda m, G: dict(m.all_pairs_shortest_path_length(G))),
    ("all_pairs_dijkstra_path_length", "medium", lambda m, G: dict(m.all_pairs_dijkstra_path_length(G))),
    ("average_shortest_path_length", "medium_connected", lambda m, G: m.average_shortest_path_length(G)),
]

# The same job in rustworkx, where it has one. `R` is the converted graph.
RUSTWORKX = {
    "connected_components": lambda R: rustworkx.connected_components(R),
    "single_source_dijkstra_path_length": lambda R: rustworkx.dijkstra_shortest_path_lengths(R, 0, float),
    
    "betweenness_centrality": lambda R: rustworkx.betweenness_centrality(R),
    "closeness_centrality": lambda R: rustworkx.closeness_centrality(R),
    "minimum_spanning_tree": lambda R: rustworkx.minimum_spanning_tree(R, weight_fn=float),
}


def pattern_edit_then_query(m, G, rounds):
    """Add an edge, ask a point-to-point question, repeat. Changes G."""
    n = len(G)
    rng = random.Random(5)
    answers = []
    for _ in range(rounds):
        u, v = rng.randrange(n), rng.randrange(n)
        G.add_edge(u, v, weight=1)
        answers.append(m.has_path(G, u, rng.randrange(n)))
    return answers


def pattern_pairs(m, G, rounds):
    """Shortest paths between random pairs on a graph that does not change."""
    n = len(G)
    rng = random.Random(6)
    return [m.shortest_path_length(G, rng.randrange(n), rng.randrange(n)) for _ in range(rounds)]


def pattern_weighted_pairs(m, G, rounds):
    n = len(G)
    rng = random.Random(7)
    return [m.dijkstra_path_length(G, rng.randrange(n), rng.randrange(n)) for _ in range(rounds)]


def pattern_neighbourhoods(m, G, rounds):
    """Everything within two steps of each of many nodes."""
    return [len(m.single_source_shortest_path_length(G, u, cutoff=2)) for u in range(rounds)]


def pattern_first_edge(m, G, rounds):
    """Ask for a generator and read one item from it."""
    return [next(m.bfs_edges(G, u), None) for u in range(rounds)]


def pattern_sources(m, G, rounds):
    """Full single-source Dijkstra from several sources."""
    return [m.single_source_dijkstra_path_length(G, u) for u in range(rounds)]


# (label, graph kind, function, rounds full, rounds quick)
PATTERNS = [
    ("add an edge, then has_path", "big", pattern_edit_then_query, 300, 100),
    ("shortest_path_length, random pairs", "big", pattern_pairs, 3000, 500),
    ("dijkstra_path_length, random pairs", "big", pattern_weighted_pairs, 30, 10),
    ("2-step neighbourhood of each node", "big", pattern_neighbourhoods, 20000, 2000),
    ("first item of bfs_edges", "big", pattern_first_edge, 20000, 2000),
    ("Dijkstra from each of several sources", "big", pattern_sources, 10, 5),
]


def human(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 120:
        return f"{seconds:.2f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


SCALING_SIZES = [2_000, 5_000, 10_000, 20_000, 50_000]
SCALING_SAMPLE = 50

# (label, the full call, the same work from only the sources in `nodes`).
# The third entry is only ever run on NetworkX, to estimate its full time.
SCALING = [
    (
        "betweenness_centrality",
        lambda m, G: m.betweenness_centrality(G),
        lambda G, nodes: networkx.betweenness_centrality(G, k=len(nodes), seed=1),
    ),
    (
        "betweenness_centrality (weighted)",
        lambda m, G: m.betweenness_centrality(G, weight="weight"),
        lambda G, nodes: networkx.betweenness_centrality(G, k=len(nodes), weight="weight", seed=1),
    ),
    (
        "closeness_centrality",
        lambda m, G: m.closeness_centrality(G),
        lambda G, nodes: [networkx.closeness_centrality(G, u=u) for u in nodes],
    ),
]


def scaling(args):
    """The per-node algorithms on graphs of growing size."""
    sizes = [n for n in SCALING_SIZES if not args.quick or n <= 5_000]
    rows = []
    for n in sizes:
        G = make_graph(n, 10, False, 4)
        gc.collect()
        gc.freeze()  # see the comment in main()
        rng = random.Random(8)
        sample = rng.sample(range(n), SCALING_SAMPLE)
        for label, call, sampled in SCALING:
            if args.only and args.only not in label:
                continue
            G.__networkx_cache__.clear()
            t_sample, _ = timed(lambda: sampled(G, sample))
            estimate = t_sample * n / SCALING_SAMPLE
            measured = estimate <= args.budget
            if measured:
                t_nx, expected = timed(lambda: call(networkx, G))
            else:
                t_nx = estimate
            t_ours, actual = timed(lambda: call(networkxrs, G))
            if measured and not same(actual, expected):
                raise SystemExit(f"{label}: networkxrs and NetworkX disagree")
            row = [label, f"{n:,}", f"{G.number_of_edges():,}",
                   ("" if measured else "~") + human(t_nx), human(t_ours),
                   ("" if measured else "~") + f"{t_nx / t_ours:.0f}x"]
            if rustnx is not None:
                G.__networkx_cache__.clear()
                t_rustnx = timed_or_none(lambda: call(Backend("rustnx"), G))
                row.append("" if t_rustnx is None else human(t_rustnx))
            rows.append(row)
            print("  " + "  ".join(row), flush=True)
    header = ["function", "nodes", "edges", "NetworkX", "networkxrs", "speed-up"]
    if rustnx is not None:
        header.append("rustnx")
    return header, rows


def print_table(table_header, table_rows, markdown):
    print()
    if markdown:
        print("| " + " | ".join(table_header) + " |")
        print("|" + "|".join(["---"] + ["---:"] * (len(table_header) - 1)) + "|")
        for row in table_rows:
            print("| " + " | ".join(row) + " |")
    else:
        table = [table_header, *table_rows]
        widths = [max(len(r[i]) for r in table) for i in range(len(table_header))]
        for r in table:
            print("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))))


def environment():
    return (f"networkx {networkx.__version__}, networkxrs {networkxrs.__networkxrs_version__}, "
            f"Python {platform.python_version()}, {os.cpu_count()} CPU cores, {platform.machine()}, "
            f"policy {networkxrs.get_policy()!r}"
            + (f", rustnx {rustnx.__version__}" if rustnx is not None else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smaller graphs")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("--only", help="run benchmarks whose label contains this text")
    parser.add_argument("--scaling", action="store_true", help="per-node algorithms at growing sizes")
    parser.add_argument("--budget", type=float, default=120.0,
                        help="with --scaling: estimate a NetworkX call instead of running it "
                             "when it would take longer than this many seconds")
    args = parser.parse_args()

    if args.scaling:
        table_header, table_rows = scaling(args)
        print()
        print(environment())
        print(f"~ marks a NetworkX time estimated from {SCALING_SAMPLE} of the nodes")
        print_table(table_header, table_rows, args.markdown)
        return

    big, medium = (20_000, 400) if args.quick else (200_000, 2_000)
    print(f"building graphs ({big:,} and {medium:,} nodes, about 5 edges per node) ...", flush=True)
    graphs = {
        "big": make_graph(big, 10, False, 1),
        "big_directed": make_graph(big, 5, True, 2),
        "big_dag": make_graph(big, 5, True, 3, dag=True),
        "medium": make_graph(medium, 10, False, 4),
    }
    largest = max(networkx.connected_components(graphs["medium"]), key=len)
    graphs["medium_connected"] = graphs["medium"].subgraph(largest).copy()

    # The graphs are millions of Python objects. Python's garbage collector
    # periodically re-examines every object it tracks, and with a heap this
    # size one such pass takes longer than most of the calls being timed. It
    # lands on whichever call happens to be running: on Python 3.14, the
    # second of two identical NetworkX calls took three times as long as the
    # first for this reason alone. `freeze` tells the collector to leave
    # everything that exists now out of its passes, so the timings below
    # measure the calls and not the collector.
    gc.collect()
    gc.freeze()

    rows = []
    for label, kind, call in BENCHMARKS:
        if args.only and args.only not in label:
            continue
        G = graphs[kind]
        G.__networkx_cache__.clear()  # forget any snapshot from a previous row
        t_nx, expected = timed(lambda: call(networkx, G))
        t_cold, actual = timed(lambda: call(networkxrs, G))
        t_warm = min(timed(lambda: call(networkxrs, G))[0] for _ in range(3))
        if not same(actual, expected):
            raise SystemExit(f"{label}: networkxrs and NetworkX disagree")
        row = [label, f"{len(G):,}", human(t_nx), human(t_cold), human(t_warm), f"{t_nx / t_cold:.0f}x"]
        if rustworkx is not None:
            rx_call = RUSTWORKX.get(label)
            if rx_call is None:
                row += ["", ""]
            else:
                t_convert, R = timed(lambda: rustworkx.networkx_converter(G, keep_attributes=False))
                # rustworkx stores one payload per edge; give it the weight.
                for (u, v), index in zip(G.edges, R.edge_indices()):
                    R.update_edge_by_index(index, G[u][v]["weight"])
                t_rx = min(timed(lambda: rx_call(R))[0] for _ in range(3))
                row += [human(t_convert + t_rx), human(t_rx)]
        if rustnx is not None:
            G.__networkx_cache__.clear()
            backend = Backend("rustnx")
            t_cold = timed_or_none(lambda: call(backend, G))
            if t_cold is None:
                row += ["", ""]
            else:
                row += [human(t_cold), human(min(timed(lambda: call(backend, G))[0] for _ in range(3)))]
        rows.append(row)
        print("  " + "  ".join(row), flush=True)

    header = ["function", "nodes", "NetworkX", "networkxrs cold", "networkxrs warm", "speed-up (cold)"]
    if rustworkx is not None:
        header += ["rustworkx incl. conversion", "rustworkx"]
    if rustnx is not None:
        header += ["rustnx cold", "rustnx warm"]

    pattern_rows = []
    for label, kind, function, full, quick in PATTERNS:
        if args.only and args.only not in label:
            continue
        rounds = quick if args.quick else full
        # Each side gets its own copy, made outside the timed region: one
        # pattern edits the graph, and copying a large graph takes seconds.
        first, second = graphs[kind].copy(), graphs[kind].copy()
        t_nx, expected = timed(lambda: function(networkx, first, rounds))
        t_ours, actual = timed(lambda: function(networkxrs, second, rounds))
        if actual != expected:
            raise SystemExit(f"{label}: networkxrs and NetworkX disagree")
        row = [label, f"{rounds:,}", human(t_nx), human(t_ours), f"{t_nx / t_ours:.1f}x"]
        if rustnx is not None:
            third = graphs[kind].copy()
            t_rustnx = timed_or_none(lambda: function(Backend("rustnx"), third, rounds))
            row.append("" if t_rustnx is None else human(t_rustnx))
        pattern_rows.append(row)
        print("  " + "  ".join(row), flush=True)
    pattern_header = ["pattern", "calls", "NetworkX", "networkxrs", "speed-up"]
    if rustnx is not None:
        pattern_header.append("rustnx")

    print()
    print(environment())
    print_table(header, rows, args.markdown)
    print_table(pattern_header, pattern_rows, args.markdown)

if __name__ == "__main__":
    main()
