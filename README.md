# networkxrs

NetworkX, with the hot algorithms running in Rust. Change one line:

```python
import networkxrs as nx
```

Everything NetworkX provides is there under the same names. 65 of its
functions run in Rust and return what NetworkX returns. The rest *are*
NetworkX.

> **Status: first milestone (0.1).** Not yet published. See
> [Before publishing](#before-publishing).

## Using it

```python
import networkxrs as nx

G = nx.gnp_random_graph(100_000, 1e-4, seed=1)   # an ordinary networkx.Graph
ranks = nx.pagerank(G)                            # Rust
dist = nx.single_source_dijkstra_path_length(G, 0)  # Rust
tri = nx.triangles(G)                             # NetworkX, as before
```

Graphs are plain NetworkX graphs: `networkxrs.Graph is networkx.Graph`. You
can hand them to any other library, mix both imports in one program, and
pickle them as usual.

Three functions exist only here:

| | |
|---|---|
| `nx.accelerated()` | names of the functions that run in Rust |
| `nx.dispatch_counts()` | how many calls each of them answered in Rust and how many it passed to NetworkX |
| `nx.set_policy("adaptive" \| "eager" \| "off")` | when to use Rust; see [below](#when-it-uses-rust) |

And one algorithm: `nx.dag_longest_paths(G)` returns *every* longest path of
a DAG with non-negative weights, where `nx.dag_longest_path` returns one.

## What runs in Rust

| Family | Functions |
|---|---|
| Traversal | `bfs_edges` `bfs_tree` `bfs_successors` `bfs_layers` `descendants` `ancestors` `dfs_edges` `dfs_tree` `dfs_predecessors` `dfs_successors` `dfs_preorder_nodes` `dfs_postorder_nodes` |
| Components | `connected_components` `number_connected_components` `is_connected` `node_connected_component`, and the `weakly_` and `strongly_` versions |
| DAGs | `topological_sort` `topological_generations` `is_directed_acyclic_graph` `dag_longest_path` `dag_longest_path_length` `dag_longest_paths` |
| Shortest paths | `shortest_path` `shortest_path_length` `has_path` `average_shortest_path_length`; `single_source_`, `single_target_`, `bidirectional_` and `all_pairs_shortest_path` and `_length`; the `dijkstra` and `bellman_ford` functions in their `_path`, `_path_length`, `single_source_`, `bidirectional_` and `all_pairs_` forms |
| Centrality | `pagerank` `betweenness_centrality` `closeness_centrality` `degree_centrality` `in_degree_centrality` `out_degree_centrality` |
| Trees and cycles | `minimum_spanning_tree` `minimum_spanning_edges` and the `maximum_` versions (Kruskal), `is_tree` `is_forest` `cycle_basis` |

A call is handed to NetworkX, silently, when the Rust code would not
reproduce NetworkX exactly:

- multigraphs, graph views (`G.subgraph(...)`, `G.reverse(copy=False)`), and
  graph subclasses that override NetworkX's own methods;
- weights given as a function, and weight values that are not plain Python
  `int` or `float` (NumPy scalars, `Decimal`, `bool`, NaN, infinity);
- a mix of `int` and `float` weights, for functions that return distances;
- negative weights with Dijkstra;
- less common options: `sort_neighbors`, sampling in betweenness (`k`),
  Prim's and Borůvka's spanning-tree algorithms, a caller-supplied
  `topo_order`;
- any argument networkxrs does not know, such as one added by a NetworkX
  release newer than it.

`nx.dispatch_counts()` shows which way calls went.

## The same as NetworkX, exactly

For every accelerated function the goal is the identical result: same
values, same types (`int` where NetworkX returns `int`), same order of dict
keys and list items, the same path when several are equally short, and for
bad input the same exception with the same message raised at the same
moment.

How that is checked:

- **`tests/`** calls every accelerated function with a range of arguments on
  about 50 graphs (directed and not, weighted in seven ways, disconnected,
  with self-loops, with odd node types) through both libraries and compares
  the outcomes strictly. It does this under both the `adaptive` and `eager`
  policies: 6,412 tests.
- **NetworkX's own test-suite**, run with networkxrs's functions patched into
  the `networkx` namespace, so that NetworkX's unmodified tests (and
  NetworkX's other algorithms, which call these functions internally)
  exercise the Rust code. All of it passes under both policies with one
  exception, listed below: 7,047 of 7,048 tests on NetworkX 3.6.1 and 9,360
  of 9,361 on NetworkX 3.7. `python -m networkxrs.conformance` runs the
  modules that cover the accelerated functions;
  `pytest -p networkxrs.conformance --pyargs networkx` runs everything.

Results are matched to NetworkX **3.6 and 3.7**. Older releases behave
slightly differently from these in places (dict order, tie-breaking), so they
are not supported.

Known differences:

1. **Changing a graph while reading a generator from it.** NetworkX's
   generators compute as they go. Here, `connected_components`, the
   `weakly_`/`strongly_` versions, `topological_sort`,
   `topological_generations` and the spanning-edge functions compute their
   whole result when called, and the other generators compute the rest of
   theirs at once after handing out the first items. Editing the graph
   part-way through reading is therefore not noticed. One NetworkX test,
   which checks that `topological_sort` raises in that situation, is the
   exception mentioned above.
2. **PageRank agrees to rounding, not bit for bit** (within 1e-15 relative
   on the test graphs).
   NetworkX computes it with SciPy, which adds some terms in a different
   order. Everything else is bit-identical, including float results.
3. **Equal but distinct node objects.** If a graph holds node `1` and a
   function is called with `1.0`, NetworkX's result can contain the `1.0`
   that was passed; here it contains the graph's own `1`. They compare
   equal.
4. **A few microseconds of overhead per call**, which shows on calls that do
   almost no work (see the last benchmark table).

## How fast

Measured on a 16-core Apple Silicon MacBook Pro, NetworkX 3.7, Python 3.14.
Reproduce with `python benchmarks/bench.py`; timings vary from run to run.

*Cold* is the first call on a graph, which includes building the Rust
snapshot of it. *Warm* is a later call on the same unchanged graph. The
speed-up column uses the cold time. Graphs are random with about five edges
per node.

| function | nodes | NetworkX | cold | warm | speed-up (cold) |
|---|---:|---:|---:|---:|---:|
| connected_components | 200,000 | 153 ms | 49 ms | 7.3 ms | 3x |
| bfs_edges | 200,000 | 235 ms | 233 ms | 41 ms | 1x |
| dfs_preorder_nodes | 200,000 | 349 ms | 173 ms | 28 ms | 2x |
| single_source_shortest_path_length | 200,000 | 171 ms | 58 ms | 9.4 ms | 3x |
| single_source_shortest_path | 200,000 | 199 ms | 78 ms | 21 ms | 3x |
| shortest_path (one pair) | 200,000 | 521 µs | 202 µs | 115 µs | 3x |
| single_source_dijkstra_path_length | 200,000 | 768 ms | 179 ms | 104 ms | 4x |
| single_source_dijkstra | 200,000 | 967 ms | 210 ms | 129 ms | 5x |
| dijkstra_path (one pair) | 200,000 | 188 ms | 139 ms | 42 ms | 1x |
| single_source_bellman_ford_path_length | 200,000 | 1.93 s | 190 ms | 118 ms | 10x |
| minimum_spanning_tree | 200,000 | 1.51 s | 454 ms | 345 ms | 3x |
| pagerank | 200,000 | 742 ms | 97 ms | 32 ms | 8x |
| degree_centrality | 200,000 | 34 ms | 32 ms | 4.5 ms | 1x |
| strongly_connected_components | 200,000 | 271 ms | 56 ms | 16 ms | 5x |
| pagerank (directed) | 200,000 | 346 ms | 71 ms | 23 ms | 5x |
| topological_sort | 200,000 | 225 ms | 46 ms | 7.1 ms | 5x |
| dag_longest_path | 200,000 | 675 ms | 88 ms | 32 ms | 8x |
| cycle_basis | 2,000 | 23 ms | 8.8 ms | 7.8 ms | 3x |
| betweenness_centrality | 2,000 | 4.78 s | 20 ms | 18 ms | 240x |
| betweenness_centrality (weighted) | 2,000 | 11.36 s | 64 ms | 61 ms | 178x |
| closeness_centrality | 2,000 | 1.07 s | 1.5 ms | 0.76 ms | 721x |
| all_pairs_shortest_path_length | 2,000 | 1.09 s | 74 ms | 65 ms | 15x |
| all_pairs_dijkstra_path_length | 2,000 | 6.02 s | 123 ms | 116 ms | 49x |
| average_shortest_path_length | 2,000 | 1.12 s | 1.7 ms | 0.71 ms | 648x |

Reading the table:

- The biggest wins are the algorithms that run one search per node
  (betweenness, closeness, all pairs). They are pure Rust loops and use every
  CPU core. Unweighted closeness and average shortest path length also run
  64 searches at a time, one per bit of a machine word.
- Next come single passes where NetworkX does a lot of work per edge in
  Python: weighted searches, PageRank, longest paths.
- A single cheap pass such as BFS gains less on a cold call, because reading
  the graph out of NetworkX's dicts (about 40 ms for the 200,000-node graph)
  is a sizeable fraction of what NetworkX's own pass costs. The warm column
  shows the algorithm without that cost.
- `bfs_edges` cold gains nothing. It is a generator, so it starts in
  NetworkX (see below), and NetworkX's BFS produces nearly all of its items
  early and then spends most of its time finishing without producing any,
  which leaves no point at which to switch.
- One-pair `dijkstra_path` and `degree_centrality` are left to NetworkX on a
  first call by design: NetworkX may answer them faster than a snapshot can
  be built.
- `minimum_spanning_tree` spends most of its remaining time building the
  result graph in Python.
- Weighted searches are slower than unweighted ones by more than the
  algorithm explains (104 ms against 9 ms warm) because every call reads the
  weights afresh from the graph's attribute dicts.

### Larger graphs

The per-node algorithms are where NetworkX stops being usable well before a
graph is large, because their cost grows with nodes times edges.
`python benchmarks/bench.py --scaling --budget 100000` runs them at growing
sizes:

| function | nodes | edges | NetworkX | networkxrs | speed-up |
|---|---:|---:|---:|---:|---:|
| betweenness_centrality | 2,000 | 9,991 | 4.46 s | 21.0 ms | 212x |
| betweenness_centrality | 5,000 | 24,834 | 30.97 s | 127.6 ms | 243x |
| betweenness_centrality | 10,000 | 49,853 | 2.3 min | 542.2 ms | 250x |
| betweenness_centrality | 20,000 | 99,903 | 9.6 min | 2.21 s | 260x |
| betweenness_centrality | 50,000 | 249,748 | 77.8 min | 15.07 s | 309x |
| betweenness_centrality (weighted) | 2,000 | 9,991 | 10.94 s | 71.4 ms | 153x |
| betweenness_centrality (weighted) | 5,000 | 24,834 | 74.03 s | 422.9 ms | 175x |
| betweenness_centrality (weighted) | 10,000 | 49,853 | 5.4 min | 1.72 s | 190x |
| betweenness_centrality (weighted) | 20,000 | 99,903 | 23.6 min | 7.83 s | 181x |
| betweenness_centrality (weighted) | 50,000 | 249,748 | 3.2 h | 62.49 s | 185x |
| closeness_centrality | 2,000 | 9,991 | 1.03 s | 1.4 ms | 714x |
| closeness_centrality | 5,000 | 24,834 | 8.23 s | 4.6 ms | 1796x |
| closeness_centrality | 10,000 | 49,853 | 31.69 s | 14.6 ms | 2168x |
| closeness_centrality | 20,000 | 99,903 | 2.4 min | 57.6 ms | 2468x |
| closeness_centrality | 50,000 | 249,748 | 17.0 min | 358.0 ms | 2851x |

Every time in this table is measured, none estimated, and every result was
checked against NetworkX's and matches exactly. The whole run takes about
six hours, nearly all of it NetworkX. Without `--budget 100000` the script
finishes in about ten minutes by estimating the slow NetworkX calls: it runs
the search from 50 of the nodes, multiplies up, and marks the figure with a
`~`. Those estimates came out 2 to 13% above the measured times here.

### Usage patterns

Whole usage patterns, on the 200,000-node graph, including ones that are
awkward for this design:

| pattern | calls | NetworkX | networkxrs | speed-up |
|---|---:|---:|---:|---:|
| add an edge, then `has_path` | 300 | 43 ms | 38 ms | 1.1x |
| `shortest_path_length`, random pairs | 3,000 | 383 ms | 123 ms | 3.1x |
| `dijkstra_path_length`, random pairs | 30 | 10.5 s | 1.74 s | 6.0x |
| 2-step neighbourhood of each node | 20,000 | 328 ms | 179 ms | 1.8x |
| first item of `bfs_edges` | 20,000 | 66 ms | 85 ms | 0.8x |
| Dijkstra from each of several sources | 10 | 7.46 s | 940 ms | 7.9x |

The one row below 1x is the per-call overhead: 20,000 calls that each do
a few microseconds of work in NetworkX cost about one microsecond more here.

### Other Rust engines

The benchmark script prints columns for these if they are installed. Their
results are not checked against NetworkX, since neither promises identical
output.

[rustnx](https://github.com/fabuseless/rustnx) is the closest project: a
Rust engine used as a NetworkX backend, with far more functions than this
one covers. Version 0.1.0a3, same machine, same graphs:

| | networkxrs | rustnx |
|---|---:|---:|
| betweenness_centrality, 50,000 nodes | 15.3 s | 14.4 s |
| betweenness_centrality (weighted), 50,000 nodes | 62.8 s | 60.1 s |
| closeness_centrality, 50,000 nodes | 362 ms | 323 ms |
| all_pairs_dijkstra_path_length, 2,000 nodes | 123 ms | 302 ms |
| pagerank, 200,000 nodes, cold / warm | 97 / 32 ms | 160 / 15 ms |
| single_source_dijkstra_path_length, cold / warm | 179 / 104 ms | 220 / 72 ms |
| dijkstra_path (one pair), cold / warm | 139 / 42 ms | 157 / 7.4 ms |
| bfs_edges, cold / warm | 233 / 41 ms | 117 / 32 ms |
| add an edge, then `has_path`, 300 times | 38 ms | 23.1 s |
| first item of `bfs_edges`, 20,000 times | 85 ms | 2.6 min |
| 2-step neighbourhood of each node, 20,000 times | 179 ms | 442 ms |
| `dijkstra_path_length`, 30 random pairs | 1.74 s | 910 ms |

The heavy algorithms are within about 10% of each other. The differences
come from two design choices. rustnx keeps a copy of the edge weights, which
makes repeated weighted queries faster; networkxrs reads them from the graph
on every call, so a weight changed in place is never out of date. rustnx
converts the graph whenever it is asked to run, and computes a generator's
whole result before yielding; networkxrs first decides whether Rust will pay
off (see [When it uses Rust](#when-it-uses-rust)), which is what the edit
and first-item rows show. rustnx's betweenness in that release differs from
NetworkX's in the last bits; here it is identical.

Bit-parallel BFS for closeness, and reading betweenness predecessors off the
BFS levels, were adopted here after reading rustnx's source.

[rustworkx](https://www.rustworkx.org/) has its own graph type and API. On
the same graphs, once they are in its format: single-source Dijkstra took
85 ms against networkxrs's 104 ms warm, connected components 40 ms against
7.3 ms, betweenness 46 ms against 18 ms, closeness 37 ms against 0.76 ms,
and the spanning tree 33 ms against 345 ms (it returns the tree in its own
format, where networkxrs builds a NetworkX graph). Converting the
200,000-node NetworkX graph with rustworkx's own converter took 420 to
660 ms, against about 40 ms for a snapshot here.

### A note on measuring

The benchmark holds millions of Python objects, and on a heap that size one
pass of Python's garbage collector takes longer than most of the calls being
timed and lands on whichever call is running. The script freezes the heap
after building its graphs (`gc.freeze()`) so that both libraries are timed
without it. A program holding large NetworkX graphs pays that cost whichever
library it calls.

## How it works

**A snapshot.** The first accelerated call on a graph copies its structure
into a compact Rust form: nodes become the integers `0..n` in NetworkX's
iteration order, and each node's neighbours go into one flat array. The
algorithm runs on that and the integer results are mapped back to your node
objects. The Rust code never hashes or compares a Python object.

**Kept up to date by NetworkX.** The snapshot is stored in
`G.__networkx_cache__`, the dict NetworkX gives every graph for backends to
cache converted copies in. NetworkX empties it whenever the graph is changed
through its methods, so the snapshot is dropped at the right moments and
rebuilt when next needed.

**Weights are never cached.** NetworkX does not empty that cache for
`G[u][v]["weight"] = 3`. So the snapshot holds references to the edge
attribute dicts and reads weights from them on every call.

**Same answers by construction.** Each Rust function is a port of
NetworkX's, down to the details that decide ties: neighbour order, the
order entries leave the priority queue, which of two equal maxima `max`
keeps. `docs/DESIGN.md` lists them.

### When it uses Rust

Building a snapshot takes time proportional to the size of the graph. That
is well spent on PageRank and wasted on a `has_path` between two
neighbours, especially in a loop that edits the graph between calls and so
invalidates the snapshot each time. The default policy, `adaptive`, decides
per call:

- Functions that read the whole graph build a snapshot at once.
- Searches from one node first run a quick probe on the NetworkX graph. If
  the node reaches a large part of the graph, build.
- Point-to-point queries and searches with a cutoff are answered by
  NetworkX, and the time it takes is added up per graph. When the total
  reaches what a snapshot would cost, one is built. Changing the graph
  resets the total.
- Generators you might stop reading early (`bfs_edges`, the `all_pairs_`
  functions) hand out NetworkX's items first and switch to the Rust result
  once you have read enough to show it is worth computing.

Once a snapshot exists, every search costs time proportional to what it
touches, not to the size of the graph.

`nx.set_policy("eager")` builds a snapshot on the first accelerated call
regardless: right when graphs are large, rarely change and are queried many
times. `nx.set_policy("off")` sends everything to NetworkX. The environment
variable `NETWORKXRS_POLICY` sets the starting policy.

## Building from source

Needs Rust (1.82 or newer), Python 3.12 or newer and
[maturin](https://www.maturin.rs/).

```sh
pip install maturin pytest numpy scipy "networkx>=3.6"
./build.sh                     # builds an optimised wheel and installs it
cargo test -p networkxrs-core   # the Rust unit tests
pytest tests                   # every accelerated function against NetworkX
python -m networkxrs.conformance            # NetworkX's own tests, on networkxrs
python benchmarks/bench.py --quick
```

## Layout

```
crates/networkxrs-core/   the algorithms, in pure Rust; knows nothing of Python
crates/networkxrs-py/     the PyO3 bindings: builds snapshots, converts results
python/networkxrs/        the Python package: the import swap and the dispatch
tests/                   equivalence tests against NetworkX
benchmarks/bench.py      the tables above
docs/READING_GUIDE.md    where to start reading the Rust, and where each
                         language feature is explained
docs/DESIGN.md           why it is built this way
tools/rename.py          renames the project in one step
.github/workflows/       CI (tests on every push) and release (wheels to PyPI)
```

The Rust is written to be read by someone learning the language: each
feature is explained in a comment where it first appears, in the order given
in `docs/READING_GUIDE.md`.

## Before publishing

- **The name.** `networkxrs` was free on PyPI and crates.io when this was
  written. It is one letter away from `networkxr`, an unrelated PyPI project
  with the same aim, which this project was first named after by accident.
  To change the name again, run `python tools/rename.py <name>`; it changes
  the package, the import name, the crates and the docs.
- **Repository URL.** None is set in `Cargo.toml` or `pyproject.toml`.
- **CI.** `.github/workflows/` holds two GitHub Actions workflows. `ci.yml`
  runs the Rust and Python
  tests and NetworkX's suite on Linux, macOS and Windows. `release.yml`
  builds wheels on a version tag and publishes them to PyPI; it needs the
  repository registered as a trusted publisher on PyPI first. `ci.yml` passes
  on all three systems with Python 3.12 to 3.14; `release.yml` has not been
  run yet.
- **Benchmarks.** The timings above are from one Mac. Nothing has been
  measured on Windows, and only an earlier version on Linux.

## What is not done

- Multigraphs.
- Everything outside the six families above: flow, matching, communities,
  isomorphism, link prediction and the rest all still run in NetworkX.
- Registering as a NetworkX backend, so that plain `import networkx` could
  use the Rust code through NetworkX's own dispatch.
- Caching weights between calls. It would roughly halve the time of a warm
  weighted search, and needs a way to notice in-place edits to edge
  attributes (CPython's dict watchers could provide one).

## Licence

BSD 3-Clause, the same as NetworkX. The algorithms are ports of NetworkX's
and the wrappers reuse its signatures and docstrings; NetworkX's copyright
notice is in `LICENSES/NetworkX.txt`.
