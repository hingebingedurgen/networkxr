# Reading guide

The Rust in this project is commented for someone who knows Python and is
learning Rust. Each language feature is explained where it first appears,
and the files are meant to be read in the order below. Later files assume
the earlier ones.

`cargo doc --open` renders the `///` and `//!` comments as a website, which
is a pleasant way to get the overview before reading the code.

## 1. The algorithms: `crates/networkxrs-core/src/`

Pure Rust. Nothing here knows about Python. Every file ends with its tests,
which double as usage examples; run them with `cargo test -p networkxrs-core`.

| Order | File | What it does | Rust it introduces |
|---|---|---|---|
| 1 | `lib.rs` | Lists the modules | modules, `pub`, doc comments |
| 2 | `graph.rs` | The graph: three flat arrays (CSR) | `use`, type aliases, enums, `#[derive]`, structs, `Vec`, `impl`, `&self` and borrowing, slices, integer casts, iterators (`iter`, `copied`, `zip`), traits (`Display`, `Error`), `match`, `Option`, moving values, `Result` and `?`, `let`/`mut`, closures, generic functions, tests |
| 3 | `workspace.rs` | Scratch memory that searches share, and why | `pub(crate)`, `#[derive(Default)]`, `drain` |
| 4 | `traversal.rs` | Breadth-first and depth-first search | `&mut` borrows, loop labels, `std::mem::take`, `FnMut` closures passed as arguments, `while let`, `if let`, an explicit stack in place of recursion |
| 5 | `components.rs` | Connected and strongly connected components | mutable slices, lending a buffer to a function, a non-recursive Tarjan |
| 6 | `dag.rs` | Topological sort, longest paths | an error type that carries data, `map`/`filter`/`collect`, `flatten`, match guards, `map_err`, `fold` |
| 7 | `shortest_paths.rs` | BFS, Dijkstra, Bellman-Ford, bidirectional searches | a generic struct (`Tree<D>`), `BinaryHeap`, implementing `Ord` by hand, functions generic over a closure, `par_iter` and `map_init` (parallelism with rayon), `Sync`, fixed-size arrays, `let ... else`, `VecDeque` |
| 8 | `tree.rs` | Union-find and Kruskal's algorithm | constructors (`new`, `Self`), `&mut self`, `mem::replace`, `mem::swap`, stable `sort_by` |
| 9 | `cycles.rs` | Cycle basis | `HashSet`, `Option::take` |
| 10 | `centrality.rs` | PageRank, betweenness, closeness | lifetime parameters on a struct, implementing `Default`, struct update syntax, per-thread scratch state, borrowing two fields at once |

If you only read three, read `graph.rs`, `traversal.rs` and
`shortest_paths.rs`.

Something worth noticing as you go: almost every function is a line-for-line
port of a NetworkX function, and the comments say which. Having the Python
original open beside the Rust (`python -c "import networkx, inspect;
print(inspect.getsource(networkx.bfs_edges))"`) makes both easier to read.

## 2. The bindings: `crates/networkxrs-py/src/`

The layer between Python and the algorithms, written with
[PyO3](https://pyo3.rs).

| Order | File | What it does | Rust it introduces |
|---|---|---|---|
| 1 | `lib.rs` | The `Snapshot` class: builds the Rust graph from a NetworkX graph, converts results back | PyO3's macros and types (`#[pyclass]`, `#[pymethods]`, `Python<'py>`, `Bound`, `Py`), `PyOnceLock` (lazy initialisation), `Mutex` and `try_lock`, `FnOnce` |
| 2 | `adjacency.rs` | Reads NetworkX's dicts through CPython's C API | `unsafe`, raw pointers, `SAFETY` comments, `unsafe fn`. The only unsafe code in the project. |
| 3 | `weights.rs` | Reads edge weights, all at once or on demand | `Cell` and `RefCell` (interior mutability), two lifetimes on one struct |
| 4 | `traversal.rs` | Python methods for BFS and DFS | releasing the GIL with `py.detach` |
| 5 | `shortest_paths.rs` | Python methods for shortest paths | building Python dicts and lists from Rust, closures returning `Result` |
| 6 | `structure.rs`, `centrality.rs` | The remaining Python methods | `format!`, `as_deref` |

## 3. The Python package: `python/networkxrs/`

| File | What it does |
|---|---|
| `__init__.py` | Re-exports NetworkX, then replaces the accelerated functions |
| `_dispatch.py` | Decides, per call, between Rust and NetworkX. Its docstring explains the policy. Read this one. |
| `_alias.py` | Makes `import networkxrs.algorithms.bipartite` work |
| `_traversal.py`, `_components.py`, `_dag.py`, `_shortest_paths.py`, `_centrality.py`, `_tree.py` | One thin wrapper per accelerated function |
| `conformance.py` | Runs NetworkX's own tests against networkxrs |

## Things to try

Changing code and watching what breaks teaches more than reading.

- In `shortest_paths.rs`, change Dijkstra's `through_v < ws.dist[...]` to
  `<=`. `cargo test` still passes; `pytest tests` does not, because a
  different one of several equally short paths is now returned.
- In `traversal.rs`, delete the `ws.release()` at the end of `bfs_edges`.
  The tests that assert `ws.is_clean()` fail.
- In `shortest_paths.rs`, try to push to a shared `Vec` from inside the
  closure given to `par_iter().map_init(...)` in `bfs_many`. It will not
  compile, and the error message explains why that would be a data race.
- Add an algorithm. `networkx.descendants_at_distance` is a good first one:
  it is a few lines on top of `bfs_layers`. You need a function in
  `traversal.rs`, a method in `crates/networkxrs-py/src/traversal.rs`, a
  wrapper in `python/networkxrs/_traversal.py` and a case in
  `tests/test_equivalence.py`.
