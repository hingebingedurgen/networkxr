# Design notes

Why networkxrs is built the way it is. The README says what it does; this
says why, and records the alternatives that were turned down.

## The goal and the constraint

The goal is a speed-up that needs one changed line: `import networkxrs as
nx`. That fixes the constraint: for the same input, return what NetworkX
returns. Not an equivalent answer; the same one. Real code depends, often
without its author knowing, on which of two equally short paths comes back,
on the order of a dict's keys, on a distance being `3` and not `3.0`.

## Where the graph lives

**Chosen: the graph stays a NetworkX graph; Rust works on a snapshot.**
`networkxrs.Graph` *is* `networkx.Graph`. The first accelerated call copies
the structure into a compact Rust form, cached on the graph.

The alternative was a Rust-owned graph with Python classes imitating
NetworkX's dict-of-dicts interface. It would avoid the copy, but every
NetworkX function that was *not* ported would have to work through the
imitation, and NetworkX's interface is large and subtle (live views, mutable
attribute dicts, arbitrary hashable nodes). With the snapshot design those
functions are simply NetworkX, and correctness of the unported 95% is not
our problem.

The cost is the copy. Much of the design below exists to make it cheap and
to avoid making it when it would not pay.

## The snapshot

Nodes are numbered `0..n` in NetworkX's iteration order, and each node's
neighbours are stored in NetworkX's order, in compressed sparse row form
(`graph.rs`). Order is kept because results depend on it.

**Knowing when it is stale.** NetworkX gives every graph a dict,
`__networkx_cache__`, for backends to cache converted graphs in, and clears
it in every method that changes the graph. The snapshot lives there, so
NetworkX invalidates it for us. Three cases need care:

- NetworkX sets that attribute to `None` on graphs it is about to edit
  directly (flow residual networks). Such graphs are never snapshotted.
- Graph views have a cache of their own that is *not* cleared when the
  underlying graph changes. Views are never snapshotted.
- A subclass could override `add_edge` and forget the cache, or override
  `neighbors` and change what the adjacency dict means (NetworkX has a
  private class that does exactly that). Subclasses are snapshotted only if
  they inherit all of `Graph`'s readers and writers unchanged.

**Weights are not in the snapshot.** NetworkX does not clear its cache for
`G[u][v]["weight"] = 3`, so cached weights could silently go stale. The
snapshot keeps references to the edge attribute dicts and reads from them
on every call.

**Building it fast.** The obvious way to build the neighbour arrays is to
look up each neighbour's index in a `{node: index}` dict. On a large graph
each lookup is several cache misses; built that way, a snapshot of a
200,000-node, million-edge graph took about 730 ms, against about 200 ms now. The trick used instead: in NetworkX, `adj[u][v]` and `adj[v][u]`
are the *same dict object*. So its address identifies the edge, and pairing
up equal addresses tells us each entry's neighbour without touching a node
object at all. Addresses are read with CPython's `PyDict_Next`, which hands
out borrowed pointers (`adjacency.rs`, the project's only unsafe code), then
sorted and paired (`Graph::from_edge_tokens`). If the pairing does not work
out, which NetworkX's own methods never cause, the slow way is used.

## The fallback rule

Any input the Rust path does not reproduce exactly raises `Fallback` inside
the bindings, and the wrapper calls NetworkX. This one rule covers
unsupported graph types, unsupported options, bad input (so error messages
are NetworkX's own, raised at the moment NetworkX raises them) and
arguments added by future NetworkX releases.

## When to build a snapshot

A snapshot costs time proportional to the graph. NetworkX answers some
queries by touching a handful of nodes. A design that built a snapshot on
every first call would be thousands of times slower than NetworkX on, say,
a loop that adds an edge and then calls `has_path`.

The rule adopted is the classic rent-or-buy one: keep "renting" (calling
NetworkX) until the rent paid equals the purchase price (a snapshot), then
buy. However the calls are arranged, the total then stays within a small
constant factor of what NetworkX alone would have cost, and nothing is lost
when there is real work to do. In detail (`_dispatch.py`):

- Whole-graph functions buy at once. NetworkX would spend more than the
  price anyway.
- Single-source searches probe first: a bounded search on the NetworkX
  dicts, in Rust, that stops as soon as it has reached 30% of the nodes.
  Large reach: buy. Small: rent.
- Point-to-point and cutoff searches rent, with the time NetworkX takes
  added to a per-graph total kept in the same cache dict (so editing the
  graph resets it).
- Generators start by yielding NetworkX's items, in blocks that double in
  size, timing them. When the time reaches the cost of computing the whole
  result in Rust, that is done and the rest is yielded from it, skipping
  what was already handed out. This works only because the Rust sequence is
  item-for-item the one NetworkX produces.

Two things had to be true for the Rust path itself to be safe to take:

- **A search must cost what it touches.** A `vec![false; n]` at the top of
  a search costs time proportional to the graph. So searches borrow
  preallocated arrays and reset only the entries they wrote
  (`workspace.rs`), and return compact results.
- **So must reading weights.** Single-source searches read each weight when
  they first reach the edge. After reading an eighth of them that way, the
  rest are read in one sequential pass, which is faster per weight
  (`weights.rs`).

## Details that had to be copied

Each of these changes the result if it is done any other way.

| NetworkX behaviour | Where it is reproduced |
|---|---|
| Nodes and neighbours iterate in insertion order | the snapshot keeps both orders, including the separate order of `G.pred` |
| An undirected edge is reported from its first endpoint in node order | edge numbering in `graph.rs` |
| Dijkstra pushes `(distance, counter, node)`, so equal distances pop in push order | `MinQueue` in `shortest_paths.rs` |
| A tie keeps the first path found (`<`, not `<=`) | every relaxation |
| `shortest_path(G, s, t)` uses a *bidirectional* search, which can return a different shortest path from the single-source one | `bidirectional_bfs`, `bidirectional_dijkstra` |
| Distances are `int` if the weights are, and the source's distance is the int `0` even among floats | `Numbers` in `weights.rs`; a mix of int and float weights falls back |
| `sum()` over floats is compensated (Neumaier) since Python 3.12 | `python_sum` in `centrality.rs`; closeness would otherwise differ in the last bit |
| Bellman-Ford is the queue-based variant with a specific negative-cycle heuristic, and builds paths by a depth-first walk of predecessor lists | `bellman_ford`, `path_to_position` |
| Betweenness adds each source's contribution in node order | sources run in parallel batches, then contributions are added in order, so the float result is identical |
| Weighted betweenness counts the source's own queue entry, doubling its path count | kept as is; see `count_paths_weighted` |
| `strongly_connected_components` yields in the order of a particular non-recursive Tarjan | ported line for line |
| Kruskal sorts stably, and with `reverse=True` for a maximum tree | `sort_by` with the comparison reversed, not a reversed list |
| `max()` returns the first of equal maxima | `dag_longest_path` |
| Errors from generator functions are raised when the generator is first read, except argument checks made by decorators, which are raised at the call | wrappers fall back in a way that preserves both; the tests compare when an error is raised, not only which |
| Topological sort yields what it can before raising on a cycle | `CycleError` carries the partial result |

PageRank is the exception: NetworkX computes it with SciPy, whose summation
order is not reproduced, so results agree to rounding only.

## Testing strategy

Three layers, because they catch different things:

1. `cargo test`: each algorithm on tiny graphs, with the expected answers
   written out. Catches plain bugs, and documents the functions.
2. `tests/test_equivalence.py`: every accelerated function, many arguments,
   about fifty graphs, strict comparison with NetworkX (types, order, error
   messages and their timing), under both policies. Catches differences.
3. NetworkX's own test-suite with networkxrs patched in
   (`networkxrs.conformance`). Catches what we did not think to test, and
   shows what a new NetworkX release changed.

Layer 3 is what found the graphs whose `__networkx_cache__` is `None`, and
both of the changes NetworkX 3.7 made to these functions (a new parameter
and a deprecation).

## Version policy

Results are matched to specific NetworkX releases, currently 3.6 and 3.7.
NetworkX changes tie-breaking and dict order between releases now and then,
so "matches NetworkX" only means something per release. The CI matrix lists
the supported ones; supporting a new release means adding it there and
fixing whatever the conformance run reports.
