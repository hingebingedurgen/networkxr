# networkxr

NetworkX, with the hot algorithms running in Rust. Change one line:

```python
import networkxr as nx
```

Everything NetworkX provides is there under the same names. 65 of its
functions run in Rust and return what NetworkX returns. The rest *are*
NetworkX.

> **Status: first milestone (0.1).** The name `networkxr` is a working name
> and is already taken on PyPI by another project, so this cannot be
> published under it. See [Before publishing](#before-publishing).

## Using it

```python
import networkxr as nx

G = nx.gnp_random_graph(100_000, 1e-4, seed=1)   # an ordinary networkx.Graph
ranks = nx.pagerank(G)                            # Rust
dist = nx.single_source_dijkstra_path_length(G, 0)  # Rust
tri = nx.triangles(G)                             # NetworkX, as before
```

Graphs are plain NetworkX graphs: `networkxr.Graph is networkx.Graph`. You
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
- any argument networkxr does not know, such as one added by a NetworkX
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
- **NetworkX's own test-suite**, run with networkxr's functions patched into
  the `networkx` namespace, so that NetworkX's unmodified tests (and
  NetworkX's other algorithms, which call these functions internally)
  exercise the Rust code. All of it passes under both policies with one
  exception, listed below: 7,047 of 7,048 tests on NetworkX 3.6.1 and 9,360
  of 9,361 on NetworkX 3.7. `python -m networkxr.conformance` runs the
  modules that cover the accelerated functions;
  `pytest -p networkxr.conformance --pyargs networkx` runs everything.

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

Measured on a 2-core cloud VM (x86-64), NetworkX 3.6.1, Python 3.13.
Reproduce with `python benchmarks/bench.py`; timings vary from run to run.

*Cold* is the first call on a graph, which includes building the Rust
snapshot of it. *Warm* is a later call on the same unchanged graph. The
speed-up column uses the cold time. Graphs are random with about five edges
per node.

| function | nodes | NetworkX | cold | warm | speed-up (cold) |
|---|---:|---:|---:|---:|---:|
| connected_components | 200,000 | 787 ms | 311 ms | 37 ms | 3x |
| bfs_edges | 200,000 | 2.37 s | 2.08 s | 161 ms | 1x |
| dfs_preorder_nodes | 200,000 | 2.54 s | 1.55 s | 131 ms | 2x |
| single_source_shortest_path_length | 200,000 | 825 ms | 279 ms | 56 ms | 3x |
| single_source_shortest_path | 200,000 | 2.06 s | 383 ms | 149 ms | 5x |
| shortest_path (one pair) | 200,000 | 1.2 ms | 730 µs | 218 µs | 2x |
| single_source_dijkstra_path_length | 200,000 | 4.67 s | 798 ms | 356 ms | 6x |
| single_source_dijkstra | 200,000 | 7.30 s | 973 ms | 543 ms | 7x |
| dijkstra_path (one pair) | 200,000 | 2.10 s | 760 ms | 123 ms | 3x |
| single_source_bellman_ford_path_length | 200,000 | 11.72 s | 808 ms | 454 ms | 15x |
| minimum_spanning_tree | 200,000 | 8.77 s | 1.34 s | 2.71 s | 7x |
| pagerank | 200,000 | 5.02 s | 535 ms | 98 ms | 9x |
| degree_centrality | 200,000 | 96 ms | 88 ms | 85 ms | 1x |
| strongly_connected_components | 200,000 | 2.05 s | 255 ms | 74 ms | 8x |
| pagerank (directed) | 200,000 | 1.92 s | 344 ms | 73 ms | 6x |
| topological_sort | 200,000 | 1.29 s | 305 ms | 40 ms | 4x |
| dag_longest_path | 200,000 | 3.98 s | 601 ms | 158 ms | 7x |
| cycle_basis | 2,000 | 73 ms | 29 ms | 27 ms | 3x |
| betweenness_centrality | 2,000 | 12.29 s | 166 ms | 163 ms | 74x |
| betweenness_centrality (weighted) | 2,000 | 39.23 s | 621 ms | 672 ms | 63x |
| closeness_centrality | 2,000 | 3.21 s | 80 ms | 77 ms | 40x |
| all_pairs_shortest_path_length | 2,000 | 2.62 s | 337 ms | 320 ms | 8x |
| all_pairs_dijkstra_path_length | 2,000 | 20.25 s | 905 ms | 907 ms | 22x |
| average_shortest_path_length | 2,000 | 3.16 s | 79 ms | 80 ms | 40x |

Reading the table:

- The big wins are where NetworkX does a lot of work per edge in Python:
  weighted searches, PageRank, and anything that runs one search per node
  (those also use every CPU core).
- A single linear pass such as BFS gains less on a cold call, because
  reading the graph out of NetworkX's dicts is itself a sizeable fraction of
  what NetworkX's own pass costs. `bfs_edges` cold is the extreme: it is a
  generator, so it starts in NetworkX (see below) and gains almost nothing
  until the second call.
- `minimum_spanning_tree` spends most of its remaining time building the
  result graph in Python; its warm figure above is slower than its cold one
  only through noise from that.
- `degree_centrality` is already one quick pass in NetworkX, so it is left
  there unless a snapshot happens to exist.

Whole usage patterns, on the 200,000-node graph, including ones that are
awkward for this design:

| pattern | calls | NetworkX | networkxr | speed-up |
|---|---:|---:|---:|---:|
| add an edge, then `has_path` | 300 | 212 ms | 166 ms | 1.3x |
| `shortest_path_length`, random pairs | 3,000 | 1.57 s | 754 ms | 2.1x |
| `dijkstra_path_length`, random pairs | 30 | 56.3 s | 8.2 s | 6.8x |
| 2-step neighbourhood of each node | 20,000 | 1.24 s | 925 ms | 1.3x |
| first item of `bfs_edges` | 20,000 | 65 ms | 117 ms | 0.6x |
| Dijkstra from each of several sources | 10 | 39.4 s | 4.2 s | 9.5x |

The one row below 1x is the per-call overhead: 20,000 calls that each do
three microseconds of work in NetworkX cost about six here.

For comparison, [rustworkx](https://www.rustworkx.org/) 0.18 on the same
graphs, once they are in its own format: single-source Dijkstra took about
the same time as networkxr's warm figure, betweenness and closeness two to
four times longer on this 2-core machine, and the spanning tree far less
(80 ms; it returns the tree in its own format, where networkxr builds a
NetworkX graph). Converting a NetworkX graph with rustworkx's own converter
took 5 to 6 seconds for the 200,000-node graph, against about 0.2 seconds
for a snapshot here. The benchmark script prints these if rustworkx is
installed.

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
variable `NETWORKXR_POLICY` sets the starting policy.

## Building from source

Needs Rust (1.82 or newer), Python 3.12 or newer and
[maturin](https://www.maturin.rs/).

```sh
pip install maturin pytest numpy scipy "networkx>=3.6"
./build.sh                     # builds an optimised wheel and installs it
cargo test -p networkxr-core   # the Rust unit tests
pytest tests                   # every accelerated function against NetworkX
python -m networkxr.conformance            # NetworkX's own tests, on networkxr
python benchmarks/bench.py --quick
```

## Layout

```
crates/networkxr-core/   the algorithms, in pure Rust; knows nothing of Python
crates/networkxr-py/     the PyO3 bindings: builds snapshots, converts results
python/networkxr/        the Python package: the import swap and the dispatch
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

- **The name.** `networkxr` is taken on PyPI (a project with the same aim, at
  version 0.1.6 when this was written). Pick a free name and run
  `python tools/rename.py <name>`; it changes the package, the import name,
  the crates and the docs, and the tests pass afterwards.
- **Repository URL.** None is set in `Cargo.toml` or `pyproject.toml`.
- **CI.** `.github/workflows/` holds two GitHub Actions workflows. `ci.yml`
  runs the Rust and Python
  tests and NetworkX's suite on Linux, macOS and Windows. `release.yml`
  builds wheels on a version tag and publishes them to PyPI; it needs the
  repository registered as a trusted publisher on PyPI first. Neither has
  been run yet.
- **macOS and Windows.** Everything above was built and tested on Linux only.

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
