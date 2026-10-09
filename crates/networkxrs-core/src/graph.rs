//! The graph data structure every algorithm in this crate runs on.
//!
//! # The idea
//!
//! NetworkX stores a graph as a dict of dicts: `adj[u][v] = {attributes}`.
//! That is flexible and slow, because following an edge means hashing a
//! Python object. Here a graph is three flat arrays, a layout known as
//! *compressed sparse row* (CSR):
//!
//! ```text
//!   node:      0        1     2
//!   offsets: [ 0,       3,    4,    6 ]
//!   targets: [ 1, 2, 3, 0,    0, 3      ]
//!              └ node 0 ┘ └1┘ └ node 2 ┘
//! ```
//!
//! The neighbours of node `u` are `targets[offsets[u]..offsets[u + 1]]`.
//! Following an edge is one array read, and a node's neighbours sit next to
//! each other in memory, which is what CPU caches like.
//!
//! The price is that the structure cannot be edited. That is fine for our
//! purpose: the Python side keeps the editable NetworkX graph and rebuilds
//! this read-only snapshot when the graph changes.
//!
//! # Order matters
//!
//! NetworkX iterates nodes and neighbours in insertion order, and its results
//! depend on that order (which shortest path wins a tie, the order of a
//! topological sort, and so on). So node `i` here is the `i`-th node NetworkX
//! would iterate, and each neighbour list is in NetworkX's order. Nothing in
//! this file sorts anything the caller can observe.

// `use` brings names into scope, like Python's `from x import y`.
use std::fmt;

use rayon::prelude::*;

/// A node is identified by its position in NetworkX's node order.
///
/// `type` creates an alias, not a new type: `NodeId` and `u32` are
/// interchangeable. `u32` is an unsigned 32-bit integer. It halves the memory
/// of the big `targets` array compared to a 64-bit index and still allows
/// about four billion nodes.
pub type NodeId = u32;

/// An edge is identified by its position in NetworkX's edge order
/// (`G.edges`). Edge weights are passed to algorithms as a slice indexed by
/// `EdgeId`. In an undirected graph both directions of an edge share one id.
pub type EdgeId = u32;

/// A marker meaning "no node", used where Python would use `None` and an
/// `Option<NodeId>` per element would double the size of a large array.
pub const NO_NODE: NodeId = NodeId::MAX;

/// Which way to follow edges.
///
/// An `enum` is a type whose value is exactly one of the listed variants.
///
/// `#[derive(...)]` asks the compiler to write trait implementations for us:
/// - `Debug`: printable with `{:?}`
/// - `Clone`, `Copy`: can be duplicated; `Copy` means implicitly, like an integer
/// - `PartialEq`, `Eq`: comparable with `==`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Follow edges from source to target (NetworkX's `G.successors`).
    Forward,
    /// Follow edges from target to source (NetworkX's `G.predecessors`).
    Reverse,
}

/// One adjacency structure in compressed sparse row form.
///
/// A `struct` is a record with named fields. Fields are private by default,
/// which lets us guarantee the invariants below: nobody outside this module
/// can build a `Csr` whose arrays disagree with each other.
///
/// Invariants:
/// - `offsets.len() == node_count + 1`, `offsets[0] == 0`, non-decreasing
/// - `targets.len() == edge_ids.len() == offsets[node_count]`
#[derive(Debug, Clone)]
pub struct Csr {
    /// `Vec<T>` is a growable array that owns its elements, like a Python
    /// list where every element has the same type.
    offsets: Vec<usize>,
    targets: Vec<NodeId>,
    edge_ids: Vec<EdgeId>,
}

// An `impl` block attaches functions to a type. Functions taking `&self` are
// methods, called as `csr.neighbors(u)`.
impl Csr {
    /// The neighbours of `u`, in NetworkX order.
    ///
    /// `&self` borrows the `Csr` without taking ownership: the caller keeps
    /// it and may lend it out again. The return type `&[NodeId]` is a
    /// *slice*: a pointer and a length describing part of `self.targets`. No
    /// data is copied. The compiler checks that the slice is not used after
    /// the `Csr` it points into is gone; that check is the "borrow checker".
    ///
    /// `#[inline]` suggests copying this small function's body into its
    /// callers instead of emitting a call. It is called in every inner loop.
    #[inline]
    pub fn neighbors(&self, u: NodeId) -> &[NodeId] {
        // `u as usize` converts between integer types. Indexing needs `usize`,
        // the pointer-sized unsigned integer. `a..b` is a half-open range.
        // The last expression of a function, with no semicolon, is its
        // return value.
        &self.targets[self.offsets[u as usize]..self.offsets[u as usize + 1]]
    }

    /// The edge ids parallel to [`neighbors`](Self::neighbors): element `i`
    /// is the id of the edge to neighbour `i`.
    #[inline]
    pub fn edge_ids(&self, u: NodeId) -> &[EdgeId] {
        &self.edge_ids[self.offsets[u as usize]..self.offsets[u as usize + 1]]
    }

    /// Pairs of `(neighbour, edge id)` for `u`.
    ///
    /// `impl Iterator<Item = ...>` means "some type that implements
    /// `Iterator`; the caller need not know which". Iterators are lazy:
    /// nothing happens until something loops over the result.
    ///
    /// - `.iter()` yields references (`&NodeId`)
    /// - `.copied()` turns those into plain values
    /// - `.zip(other)` pairs two iterators element by element
    ///
    /// The `+ '_` says the returned iterator borrows from `self`. `'_` is a
    /// *lifetime* the compiler fills in.
    #[inline]
    pub fn arcs(&self, u: NodeId) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        self.neighbors(u)
            .iter()
            .copied()
            .zip(self.edge_ids(u).iter().copied())
    }

    /// The number of neighbours of `u`.
    #[inline]
    pub fn degree(&self, u: NodeId) -> usize {
        self.offsets[u as usize + 1] - self.offsets[u as usize]
    }

    /// The raw offsets array: `offsets()[u]..offsets()[u + 1]` is where
    /// node `u`'s entries are.
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }
}

/// Everything that can be wrong with the input to [`Graph::from_csr`].
///
/// Enum variants can carry data. Each variant here says what went wrong and
/// carries what the message needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The offsets array is not a valid CSR offsets array.
    BadOffsets,
    /// A neighbour index is `>= node_count`.
    TargetOutOfRange { node: NodeId, target: NodeId },
    /// An undirected graph lists `v` under `u` but not `u` under `v`, or a
    /// predecessor list disagrees with the successor lists.
    Inconsistent { from: NodeId, to: NodeId },
    /// An edge token does not pair up as [`Graph::from_edge_tokens`] requires.
    UnpairedToken,
    /// More nodes or edges than a `u32` can index.
    TooLarge,
}

// A *trait* is an interface. `fmt::Display` is the trait behind `{}` in
// format strings and `.to_string()`. Implementing it for our error type
// gives it a human-readable message.
impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `match` is a switch that must cover every variant; adding a variant
        // later makes this a compile error until it is handled here.
        match self {
            GraphError::BadOffsets => write!(f, "offsets do not describe a valid adjacency"),
            GraphError::TargetOutOfRange { node, target } => {
                write!(
                    f,
                    "node {node} lists neighbour {target}, which does not exist"
                )
            }
            GraphError::Inconsistent { from, to } => {
                write!(
                    f,
                    "edge {from} -> {to} has no matching entry in the other direction"
                )
            }
            GraphError::UnpairedToken => write!(f, "an edge token does not appear exactly twice"),
            GraphError::TooLarge => write!(f, "graph is too large for 32-bit indices"),
        }
    }
}

// `std::error::Error` marks a type as an error other code can handle
// generically. It needs no methods beyond `Debug` and `Display`.
impl std::error::Error for GraphError {}

/// A read-only simple graph (no parallel edges) with nodes `0..n`.
#[derive(Debug, Clone)]
pub struct Graph {
    directed: bool,
    /// For each node, its successors. For an undirected graph, its neighbours.
    succ: Csr,
    /// For each node, its predecessors. `Option<T>` is either `Some(value)`
    /// or `None`; Rust has no null, so "might be absent" is always spelled
    /// out in the type. Undirected graphs store `None` here and reuse `succ`.
    pred: Option<Csr>,
    /// `endpoints[e]` is edge `e` as `(u, v)`, in NetworkX's `G.edges` order.
    endpoints: Vec<(NodeId, NodeId)>,
}

impl Graph {
    /// Builds a graph from adjacency already in CSR layout.
    ///
    /// `succ_offsets`/`succ_targets` describe each node's successors
    /// (neighbours, if undirected), in NetworkX order.
    ///
    /// For a directed graph, `pred` may give each node's predecessors in
    /// NetworkX's `G.pred` order. If it is `None`, predecessors are listed in
    /// increasing node order.
    ///
    /// The arguments are taken *by value* (no `&`): the caller hands over
    /// ownership of the vectors and this function stores them without
    /// copying. After the call the caller can no longer use them. That is a
    /// "move".
    ///
    /// `Result<Graph, GraphError>` is either `Ok(graph)` or `Err(error)`.
    /// Rust has no exceptions; a function that can fail says so in its
    /// return type and the caller must deal with both cases.
    pub fn from_csr(
        directed: bool,
        succ_offsets: Vec<usize>,
        succ_targets: Vec<NodeId>,
        pred: Option<(Vec<usize>, Vec<NodeId>)>,
    ) -> Result<Graph, GraphError> {
        // The `?` operator: if the call returned `Err(e)`, return `Err(e)`
        // from this function right away; otherwise unwrap the `Ok` value.
        let n = validate(&succ_offsets, &succ_targets)?;

        // `let` declares a variable. Variables are immutable unless declared
        // `let mut`.
        let mut edge_ids: Vec<EdgeId> = Vec::with_capacity(succ_targets.len());
        let mut endpoints: Vec<(NodeId, NodeId)> = Vec::new();

        if directed {
            // Every arc is its own edge, numbered in iteration order.
            for u in 0..n {
                for &v in &succ_targets[succ_offsets[u]..succ_offsets[u + 1]] {
                    edge_ids.push(endpoints.len() as EdgeId);
                    endpoints.push((u as NodeId, v));
                }
            }
            let succ = Csr {
                offsets: succ_offsets,
                targets: succ_targets,
                edge_ids,
            };

            // The transpose lists, for each node, the arcs coming *into* it,
            // sorted by source node.
            let sorted_pred = transpose(n, &succ, |_u, _v| true);

            // `match` on an Option handles both cases.
            let pred = match pred {
                None => sorted_pred,
                Some((offsets, sources)) => {
                    if validate(&offsets, &sources)? != n || sources.len() != succ.targets.len() {
                        return Err(GraphError::BadOffsets);
                    }
                    // The caller's order is the one we keep. Each incoming
                    // arc still needs its edge id, which we look up in the
                    // sorted transpose.
                    let edge_ids =
                        lookup_edge_ids(n, &offsets, &sources, &sorted_pred, |_v, _u| true)?;
                    Csr {
                        offsets,
                        targets: sources,
                        edge_ids,
                    }
                }
            };
            Ok(Graph {
                directed,
                succ,
                pred: Some(pred),
                endpoints,
            })
        } else {
            // Undirected: NetworkX lists edge {u, v} under both u and v, and
            // reports it once, from whichever endpoint comes first in node
            // order. So an arc u -> v with v >= u is the "first sighting" and
            // gets a fresh id, and the arc v -> u reuses it.
            let mut first_sightings = Csr {
                offsets: succ_offsets,
                targets: succ_targets,
                edge_ids: Vec::new(),
            };
            for u in 0..n as NodeId {
                for &v in first_sightings.neighbors(u) {
                    if v >= u {
                        edge_ids.push(endpoints.len() as EdgeId);
                        endpoints.push((u, v));
                    } else {
                        edge_ids.push(0); // placeholder, filled in below
                    }
                }
            }
            first_sightings.edge_ids = edge_ids;

            // For each node v, the first sightings u -> v with u < v, sorted
            // by u. Self-loops (u == v) are left out: they appear only once.
            let back = transpose(n, &first_sightings, |u, v| v > u);
            let offsets = &first_sightings.offsets;
            let targets = &first_sightings.targets;
            // The arcs that still need an id are exactly those with v < u.
            let looked_up = lookup_edge_ids(n, offsets, targets, &back, |u, v| v < u)?;

            // Every edge must have been matched exactly once.
            let backward = targets.len() - endpoints.len();
            if backward != back.targets.len() {
                // Some edge is listed under its first endpoint only. Find it
                // for the error message. `find` returns the first item for
                // which the closure is true. `|&&(u, v)| ...` is a closure
                // (an anonymous function); its argument is a reference to a
                // reference to a pair, which the pattern takes apart.
                let one_sided = endpoints
                    .iter()
                    .find(|&&(u, v)| u != v && !first_sightings.neighbors(v).contains(&u));
                let &(from, to) = one_sided.unwrap_or(&(0, 0));
                return Err(GraphError::Inconsistent { from, to });
            }
            let mut succ = first_sightings;
            for u in 0..n as NodeId {
                let range = succ.offsets[u as usize]..succ.offsets[u as usize + 1];
                for i in range {
                    if succ.targets[i] < u {
                        succ.edge_ids[i] = looked_up[i];
                    }
                }
            }
            Ok(Graph {
                directed,
                succ,
                pred: None,
                endpoints,
            })
        }
    }

    /// Builds a graph from adjacency lists in which the neighbours are not
    /// known, but every entry carries a *token* identifying its edge.
    ///
    /// This is the fast way to read a NetworkX graph. In NetworkX,
    /// `G.adj[u][v]` and `G.adj[v][u]` are the same attribute dict object, so
    /// that object's address is a token shared by exactly the two entries of
    /// one edge. Matching up equal tokens tells us each entry's neighbour
    /// without hashing a single Python node. (The slower alternative,
    /// [`from_csr`](Self::from_csr), needs every neighbour looked up in a
    /// node-to-index dict first.)
    ///
    /// `succ_offsets` delimits each node's entries as in CSR, and
    /// `succ_tokens` holds one token per entry.
    ///
    /// - Undirected: a token must appear twice, under two different nodes.
    ///   A token that appears once is taken to be a self-loop.
    /// - Directed: `pred` gives the predecessor entries the same way, and a
    ///   token must appear once among the successor entries and once among
    ///   the predecessor entries.
    ///
    /// Returns the graph and, for an undirected graph, the positions in
    /// `succ_tokens` of the entries assumed to be self-loops, so the caller
    /// can check that assumption.
    pub fn from_edge_tokens(
        directed: bool,
        succ_offsets: Vec<usize>,
        succ_tokens: &[u64],
        pred: Option<(Vec<usize>, &[u64])>,
    ) -> Result<(Graph, Vec<usize>), GraphError> {
        // A dummy targets array of the right length lets us reuse `validate`
        // for the offsets checks.
        let n = validate(&succ_offsets, &vec![0; succ_tokens.len()])?;
        if n == 0 && !succ_tokens.is_empty() {
            // `validate` could not catch this with dummy targets.
            return Err(GraphError::BadOffsets);
        }
        let arcs = succ_tokens.len();
        let succ_owner = owners(&succ_offsets);

        // `if let` with a tuple pattern: taken when the graph is directed and
        // `pred` was given, and it binds the two parts of `pred`.
        if let (true, Some((pred_offsets, pred_tokens))) = (directed, pred) {
            if validate(&pred_offsets, &vec![0; pred_tokens.len()])? != n
                || pred_tokens.len() != arcs
            {
                return Err(GraphError::BadOffsets);
            }
            let pred_owner = owners(&pred_offsets);

            // Sort all entries by token. Successor entries get slot numbers
            // 0..arcs and predecessor entries arcs..2*arcs, so after sorting
            // the two entries of an edge sit side by side, successor first.
            let mut entries: Vec<(u64, u32)> = Vec::with_capacity(2 * arcs);
            entries.extend(
                succ_tokens
                    .iter()
                    .enumerate()
                    .map(|(slot, &t)| (t, slot as u32)),
            );
            entries.extend(
                pred_tokens
                    .iter()
                    .enumerate()
                    .map(|(slot, &t)| (t, (arcs + slot) as u32)),
            );
            // rayon's parallel sort. "unstable" means equal elements may be
            // reordered, which is faster and harmless here.
            entries.par_sort_unstable();

            let mut succ_targets = vec![0 as NodeId; arcs];
            let mut pred_sources = vec![0 as NodeId; arcs];
            let mut pred_edge_ids = vec![0 as EdgeId; arcs];
            // `chunks_exact(2)` yields consecutive pairs.
            for pair in entries.chunks_exact(2) {
                let ((token_a, a), (token_b, b)) = (pair[0], pair[1]);
                let (a, b) = (a as usize, b as usize);
                if token_a != token_b || a >= arcs || b < arcs {
                    return Err(GraphError::UnpairedToken);
                }
                let b = b - arcs;
                succ_targets[a] = pred_owner[b];
                pred_sources[b] = succ_owner[a];
                // A directed edge's id is its position among the successor
                // entries, which is NetworkX's `G.edges` order.
                pred_edge_ids[b] = a as EdgeId;
            }
            let endpoints = succ_owner
                .iter()
                .copied()
                .zip(succ_targets.iter().copied())
                .collect();
            let succ = Csr {
                offsets: succ_offsets,
                targets: succ_targets,
                edge_ids: (0..arcs as EdgeId).collect(),
            };
            let pred = Csr {
                offsets: pred_offsets,
                targets: pred_sources,
                edge_ids: pred_edge_ids,
            };
            return Ok((
                Graph {
                    directed: true,
                    succ,
                    pred: Some(pred),
                    endpoints,
                },
                Vec::new(),
            ));
        }
        if directed {
            return Err(GraphError::BadOffsets);
        }

        let mut entries: Vec<(u64, u32)> = succ_tokens
            .iter()
            .enumerate()
            .map(|(slot, &t)| (t, slot as u32))
            .collect();
        entries.par_sort_unstable();

        let mut targets = vec![0 as NodeId; arcs];
        // partner[slot] is the other entry of the same edge.
        let mut partner = vec![0u32; arcs];
        let mut self_loops = Vec::new();
        let mut i = 0;
        while i < entries.len() {
            let (token, a) = entries[i];
            let a = a as usize;
            // `get` returns None past the end instead of panicking.
            let next = entries.get(i + 1).filter(|&&(t, _)| t == token);
            match next {
                None => {
                    targets[a] = succ_owner[a];
                    partner[a] = a as u32;
                    self_loops.push(a);
                    i += 1;
                }
                Some(&(_, b)) => {
                    let b = b as usize;
                    let third_matches = entries.get(i + 2).is_some_and(|&(t, _)| t == token);
                    if succ_owner[a] == succ_owner[b] || third_matches {
                        return Err(GraphError::UnpairedToken);
                    }
                    targets[a] = succ_owner[b];
                    targets[b] = succ_owner[a];
                    partner[a] = b as u32;
                    partner[b] = a as u32;
                    i += 2;
                }
            }
        }

        // Number the edges in NetworkX order: an edge is reported from
        // whichever endpoint comes first in node order.
        let mut edge_ids = vec![0 as EdgeId; arcs];
        let mut endpoints = Vec::with_capacity((arcs + self_loops.len()) / 2);
        for slot in 0..arcs {
            let (u, v) = (succ_owner[slot], targets[slot]);
            if v >= u {
                let id = endpoints.len() as EdgeId;
                edge_ids[slot] = id;
                edge_ids[partner[slot] as usize] = id;
                endpoints.push((u, v));
            }
        }
        let succ = Csr {
            offsets: succ_offsets,
            targets,
            edge_ids,
        };
        Ok((
            Graph {
                directed: false,
                succ,
                pred: None,
                endpoints,
            },
            self_loops,
        ))
    }

    /// Builds a graph from an edge list. Convenient for tests and for Rust
    /// callers; the Python bindings use [`from_csr`](Self::from_csr).
    ///
    /// Neighbour order follows the order of `edges`, as it would if you
    /// called `G.add_edge` in a loop on an empty NetworkX graph that already
    /// had nodes `0..n` added in order. `edges` must not repeat an edge.
    ///
    /// `&[(NodeId, NodeId)]` borrows a slice of pairs: this function reads
    /// the caller's data and gives it back untouched.
    pub fn from_edges(
        n: usize,
        edges: &[(NodeId, NodeId)],
        directed: bool,
    ) -> Result<Graph, GraphError> {
        // `vec![value; n]` makes a Vec of n copies of value. Here that is n
        // empty lists.
        let mut succ: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        let mut pred: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        // `for &(u, v) in edges` destructures each pair while iterating.
        for &(u, v) in edges {
            if u as usize >= n || v as usize >= n {
                return Err(GraphError::TargetOutOfRange { node: u, target: v });
            }
            succ[u as usize].push(v);
            if directed {
                pred[v as usize].push(u);
            } else if u != v {
                succ[v as usize].push(u);
            }
        }
        let (succ_offsets, succ_targets) = flatten(&succ);
        // `then` turns a bool into an Option: `Some(closure result)` if true.
        let pred = directed.then(|| flatten(&pred));
        Graph::from_csr(directed, succ_offsets, succ_targets, pred)
    }

    /// The number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.succ.offsets.len() - 1
    }

    /// The number of edges. An undirected edge counts once.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.endpoints.len()
    }

    #[inline]
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// An iterator over all node ids, `0..n`.
    #[inline]
    pub fn nodes(&self) -> std::ops::Range<NodeId> {
        0..self.node_count() as NodeId
    }

    /// Successor lists (neighbour lists, if undirected).
    #[inline]
    pub fn succ(&self) -> &Csr {
        &self.succ
    }

    /// Predecessor lists (neighbour lists, if undirected).
    #[inline]
    pub fn pred(&self) -> &Csr {
        // `as_ref()` turns `&Option<Csr>` into `Option<&Csr>`, so we can
        // look inside without moving the Csr out of `self`. `unwrap_or`
        // supplies the fallback for `None`.
        self.pred.as_ref().unwrap_or(&self.succ)
    }

    /// The adjacency to use when following edges in direction `dir`.
    #[inline]
    pub fn adjacency(&self, dir: Direction) -> &Csr {
        match dir {
            Direction::Forward => self.succ(),
            Direction::Reverse => self.pred(),
        }
    }

    /// Edge `e` as `(u, v)`, oriented as NetworkX's `G.edges` reports it.
    #[inline]
    pub fn endpoints(&self, e: EdgeId) -> (NodeId, NodeId) {
        self.endpoints[e as usize]
    }

    /// All edges in NetworkX's `G.edges` order.
    #[inline]
    pub fn edges(&self) -> &[(NodeId, NodeId)] {
        &self.endpoints
    }

    /// NetworkX's `G.degree(u)`: in-degree plus out-degree for a directed
    /// graph; for an undirected graph a self-loop counts twice.
    pub fn degree(&self, u: NodeId) -> usize {
        if self.directed {
            self.succ.degree(u) + self.pred().degree(u)
        } else {
            // `usize::from(bool)` is 1 for true and 0 for false.
            self.succ.degree(u) + usize::from(self.succ.neighbors(u).contains(&u))
        }
    }
}

/// For each entry of a CSR structure, the node whose list it is in.
fn owners(offsets: &[usize]) -> Vec<NodeId> {
    let mut owner = Vec::with_capacity(*offsets.last().unwrap_or(&0));
    for u in 0..offsets.len().saturating_sub(1) {
        // `extend` appends everything an iterator yields; `repeat_n` yields
        // the same value a given number of times.
        owner.extend(std::iter::repeat_n(
            u as NodeId,
            offsets[u + 1] - offsets[u],
        ));
    }
    owner
}

/// Checks a CSR offsets/targets pair and returns the node count.
///
/// A function without `pub` is private to this module.
fn validate(offsets: &[usize], targets: &[NodeId]) -> Result<usize, GraphError> {
    // `first()` and `last()` return an Option because the slice may be empty.
    if offsets.first() != Some(&0) || offsets.last() != Some(&targets.len()) {
        return Err(GraphError::BadOffsets);
    }
    // `windows(2)` yields every adjacent pair as a slice of length 2.
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(GraphError::BadOffsets);
    }
    let n = offsets.len() - 1;
    if n >= NO_NODE as usize || targets.len() >= EdgeId::MAX as usize {
        return Err(GraphError::TooLarge);
    }
    for u in 0..n {
        for &v in &targets[offsets[u]..offsets[u + 1]] {
            if v as usize >= n {
                return Err(GraphError::TargetOutOfRange {
                    node: u as NodeId,
                    target: v,
                });
            }
        }
    }
    Ok(n)
}

/// Turns a list of lists into CSR offsets and targets.
fn flatten(lists: &[Vec<NodeId>]) -> (Vec<usize>, Vec<NodeId>) {
    let mut offsets = Vec::with_capacity(lists.len() + 1);
    let mut targets = Vec::new();
    offsets.push(0);
    for list in lists {
        // `extend_from_slice` appends a copy of every element.
        targets.extend_from_slice(list);
        offsets.push(targets.len());
    }
    (offsets, targets)
}

/// Reverses the arcs of `csr` for which `keep(u, v)` is true.
///
/// The result lists, for each node `v`, every kept arc `u -> v` as source
/// `u` with its edge id, sorted by `u`. This is a counting sort: count how
/// many arcs land on each node, turn the counts into offsets, then drop each
/// arc into its slot.
///
/// `F: Fn(NodeId, NodeId) -> bool` makes the function *generic* over the
/// type of the closure passed in. The compiler generates a specialised copy
/// for each closure, so the call is as fast as writing the test inline.
fn transpose<F>(n: usize, csr: &Csr, keep: F) -> Csr
where
    F: Fn(NodeId, NodeId) -> bool,
{
    let mut offsets = vec![0usize; n + 1];
    for u in 0..n as NodeId {
        for &v in csr.neighbors(u) {
            if keep(u, v) {
                offsets[v as usize + 1] += 1;
            }
        }
    }
    for v in 0..n {
        offsets[v + 1] += offsets[v];
    }
    let total = offsets[n];
    let mut sources = vec![0 as NodeId; total];
    let mut edge_ids = vec![0 as EdgeId; total];
    // `cursor[v]` is the next free slot for node v. `clone()` makes an
    // explicit deep copy; Rust never copies a Vec implicitly.
    let mut cursor = offsets.clone();
    for u in 0..n as NodeId {
        for (v, e) in csr.arcs(u) {
            if keep(u, v) {
                let slot = cursor[v as usize];
                sources[slot] = u;
                edge_ids[slot] = e;
                cursor[v as usize] += 1;
            }
        }
    }
    Csr {
        offsets,
        targets: sources,
        edge_ids,
    }
}

/// For every arc `a -> b` in `offsets`/`targets` for which `wanted(a, b)` is
/// true, finds the id of the edge between them by binary search in
/// `sorted`, whose neighbour lists are sorted.
///
/// Returns one id per arc; arcs that were not wanted get 0.
fn lookup_edge_ids<F>(
    n: usize,
    offsets: &[usize],
    targets: &[NodeId],
    sorted: &Csr,
    wanted: F,
) -> Result<Vec<EdgeId>, GraphError>
where
    F: Fn(NodeId, NodeId) -> bool,
{
    let mut ids = vec![0 as EdgeId; targets.len()];
    for a in 0..n as NodeId {
        let candidates = sorted.neighbors(a);
        let candidate_ids = sorted.edge_ids(a);
        for i in offsets[a as usize]..offsets[a as usize + 1] {
            let b = targets[i];
            if !wanted(a, b) {
                continue;
            }
            // `binary_search` returns `Ok(position)` if found and
            // `Err(insertion point)` if not.
            match candidates.binary_search(&b) {
                Ok(position) => ids[i] = candidate_ids[position],
                Err(_) => return Err(GraphError::Inconsistent { from: a, to: b }),
            }
        }
    }
    Ok(ids)
}

// Tests live next to the code. `#[cfg(test)]` compiles this module only for
// `cargo test`.
#[cfg(test)]
mod tests {
    // `super::*` imports everything from the enclosing module.
    use super::*;

    #[test]
    fn undirected_edges_share_one_id_in_both_directions() {
        // Path 0 - 1 - 2 plus a self-loop on 1.
        let g = Graph::from_edges(3, &[(0, 1), (1, 1), (1, 2)], false).unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
        assert_eq!(g.succ().neighbors(1), &[0, 1, 2]);
        assert_eq!(g.edges(), &[(0, 1), (1, 1), (1, 2)]);
        // 0 -> 1 and 1 -> 0 are the same edge.
        assert_eq!(g.succ().edge_ids(0), &[0]);
        assert_eq!(g.succ().edge_ids(1), &[0, 1, 2]);
        assert_eq!(g.succ().edge_ids(2), &[2]);
        // A self-loop adds two to the degree, as in NetworkX.
        assert_eq!(g.degree(1), 4);
    }

    #[test]
    fn directed_predecessors_keep_caller_order() {
        // 2 -> 0 is added before 1 -> 0, so pred(0) is [2, 1], not sorted.
        let g = Graph::from_edges(3, &[(2, 0), (1, 0), (0, 1)], true).unwrap();
        assert_eq!(g.pred().neighbors(0), &[2, 1]);
        assert_eq!(g.pred().edge_ids(0), &[2, 1]);
        assert_eq!(g.edges(), &[(0, 1), (1, 0), (2, 0)]);
        assert_eq!(g.degree(0), 3);
    }

    #[test]
    fn edge_tokens_give_the_same_graph_as_explicit_neighbours() {
        // Path 0 - 1 - 2 with a self-loop on 1. Tokens: 70 for {0,1}, 50 for
        // the loop, 60 for {1,2}. Their numeric order is deliberately not
        // the edge order.
        let (g, loops) =
            Graph::from_edge_tokens(false, vec![0, 1, 4, 5], &[70, 70, 50, 60, 60], None).unwrap();
        let expected = Graph::from_edges(3, &[(0, 1), (1, 1), (1, 2)], false).unwrap();
        assert_eq!(g.succ().neighbors(1), expected.succ().neighbors(1));
        assert_eq!(g.succ().edge_ids(1), expected.succ().edge_ids(1));
        assert_eq!(g.edges(), expected.edges());
        assert_eq!(loops, vec![2]);

        // Directed: 2 -> 0 (token 9), 1 -> 0 (token 8), 0 -> 1 (token 7).
        // succ: 0:[1] 1:[0] 2:[0]; pred: 0:[2, 1] 1:[0] 2:[].
        let (g, _) = Graph::from_edge_tokens(
            true,
            vec![0, 1, 2, 3],
            &[7, 8, 9],
            Some((vec![0, 2, 3, 3], &[9, 8, 7])),
        )
        .unwrap();
        let expected = Graph::from_edges(3, &[(2, 0), (1, 0), (0, 1)], true).unwrap();
        assert_eq!(g.edges(), expected.edges());
        assert_eq!(g.pred().neighbors(0), expected.pred().neighbors(0));
        assert_eq!(g.pred().edge_ids(0), expected.pred().edge_ids(0));
    }

    #[test]
    fn edge_tokens_that_do_not_pair_up_are_rejected() {
        // The same token three times.
        assert!(Graph::from_edge_tokens(false, vec![0, 1, 2, 3], &[5, 5, 5], None).is_err());
        // The same token twice under one node.
        assert!(Graph::from_edge_tokens(false, vec![0, 2], &[5, 5], None).is_err());
        // Directed, with a token missing from the predecessor side.
        assert!(
            Graph::from_edge_tokens(true, vec![0, 1, 1], &[5], Some((vec![0, 0, 1], &[6])))
                .is_err()
        );
    }

    #[test]
    fn one_sided_undirected_adjacency_is_rejected() {
        // Node 0 lists 1, but node 1 lists nothing.
        let result = Graph::from_csr(false, vec![0, 1, 1], vec![1], None);
        assert!(result.is_err());
    }

    #[test]
    fn out_of_range_neighbour_is_rejected() {
        let result = Graph::from_csr(true, vec![0, 1], vec![5], None);
        assert_eq!(
            result.unwrap_err(),
            GraphError::TargetOutOfRange { node: 0, target: 5 }
        );
    }
}
