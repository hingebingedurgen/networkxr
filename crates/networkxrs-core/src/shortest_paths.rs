//! Shortest paths: breadth-first search, Dijkstra and Bellman-Ford.
//!
//! Ports of `networkx.algorithms.shortest_paths`. When several shortest
//! paths exist, each function returns the one NetworkX returns. That depends
//! on details such as the order nodes leave the priority queue, so those
//! details are ported too, and the comments point them out.
//!
//! # Weights
//!
//! Weighted functions take the weights as a function from edge id to `f64`,
//! written `W: Fn(EdgeId) -> f64`. With the weights in a slice you pass a
//! closure:
//!
//! ```text
//! dijkstra(&g, |e| weights[e as usize], ...)
//! ```
//!
//! The compiler generates a copy of `dijkstra` specialised to that closure
//! and inlines it, so this costs the same as indexing the slice directly.
//! The Python bindings pass a different closure, one that reads the weight
//! out of the NetworkX graph only when the search reaches the edge.
//!
//! # Cost
//!
//! Every search borrows a [`Workspace`] and returns only the nodes it
//! reached, so its cost depends on how far it went, not on the size of the
//! graph.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::fmt;

// `prelude::*` imports the traits that add `.par_iter()` and friends.
use rayon::prelude::*;

use crate::graph::{Csr, Direction, EdgeId, Graph, NodeId, NO_NODE};
use crate::workspace::{Workspace, UNSEEN};

/// In [`Tree::parent`], marks the root, which has no parent.
pub const NO_PARENT: u32 = u32::MAX;

/// A shortest-path tree: the nodes a search reached, how far each is from
/// the source, and how it was reached.
///
/// `Tree<D>` is *generic* over the distance type `D`: the same struct serves
/// breadth-first search with `D = u32` (a number of edges) and Dijkstra with
/// `D = f64`. The three Vecs are parallel: entry `i` of each describes the
/// same node.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree<D> {
    /// The reached nodes, in the order NetworkX lists them in its dicts:
    /// discovery order for breadth-first search, order of increasing
    /// distance for Dijkstra. The source is first.
    pub order: Vec<NodeId>,
    /// `dist[i]` is the distance from the source to `order[i]`.
    pub dist: Vec<D>,
    /// `parent[i]` is the position in `order` of the node before `order[i]`
    /// on its shortest path, or [`NO_PARENT`] for the source. A parent
    /// always comes earlier in `order` than its child.
    pub parent: Vec<u32>,
}

/// The tree of a breadth-first search.
pub type BfsTree = Tree<u32>;
/// The tree of a weighted search.
pub type ShortestPaths = Tree<f64>;

// `impl<D>` implements these methods for every choice of `D`.
impl<D> Tree<D> {
    /// The position of `node` in [`order`](Self::order), or `None` if the
    /// search did not reach it.
    pub fn position(&self, node: NodeId) -> Option<usize> {
        self.order.iter().position(|&v| v == node)
    }

    /// The shortest path from the source to the node at position `i`.
    pub fn path_to_position(&self, i: usize) -> Vec<NodeId> {
        let mut path = vec![self.order[i]];
        let mut at = self.parent[i];
        while at != NO_PARENT {
            path.push(self.order[at as usize]);
            at = self.parent[at as usize];
        }
        path.reverse();
        path
    }

    /// The shortest path from the source to `target`, or `None` if the
    /// search did not reach it.
    pub fn path_to(&self, target: NodeId) -> Option<Vec<NodeId>> {
        // `Option::map` applies the closure to the value inside `Some`.
        self.position(target).map(|i| self.path_to_position(i))
    }
}

// ---------------------------------------------------------------------------
// Unweighted
// ---------------------------------------------------------------------------

/// Breadth-first search from `source`, visiting nodes at most `cutoff` edges
/// away (`None` for no limit).
///
/// Ports the core of `networkx.single_source_shortest_path_length` and
/// `networkx.single_source_shortest_path`.
pub fn bfs(
    g: &Graph,
    source: NodeId,
    dir: Direction,
    cutoff: Option<usize>,
    ws: &mut Workspace,
) -> BfsTree {
    let adj = g.adjacency(dir);
    ws.prepare(g.node_count());

    let mut order = vec![source];
    let mut dist = vec![0u32];
    let mut parent = vec![NO_PARENT];
    // A node's slot holds its position in `order`, which doubles as the
    // visited mark.
    ws.slot[source as usize] = 0;

    // `order` is also the queue: `head` walks through it while newly found
    // nodes are appended. Nodes are appended in non-decreasing distance, so
    // once the node at `head` is at the cutoff, so is everything after it.
    let mut head = 0;
    while head < order.len() {
        let v = order[head];
        let d = dist[head];
        // `is_some_and` is true if the Option is Some and the closure holds.
        if cutoff.is_some_and(|c| d as usize >= c) {
            break;
        }
        for &w in adj.neighbors(v) {
            if ws.slot[w as usize] == UNSEEN {
                ws.slot[w as usize] = order.len() as u32;
                order.push(w);
                dist.push(d + 1);
                parent.push(head as u32);
            }
        }
        head += 1;
    }

    // Leave the workspace clean. The nodes written are exactly `order`.
    for &v in &order {
        ws.slot[v as usize] = UNSEEN;
    }
    Tree {
        order,
        dist,
        parent,
    }
}

/// [`bfs`] from each of `sources`, run in parallel.
///
/// `par_iter()` is rayon's parallel version of `iter()`. It splits the work
/// across a pool of threads, one per CPU core, and `collect` puts the
/// results back in the original order. `map_init` is `map` with per-thread
/// state: the first closure makes one [`Workspace`] per worker thread, and
/// the second receives it for every item that thread handles.
///
/// This compiles only because it is safe: the closure reads `g` through a
/// shared reference and each thread writes only its own workspace. If it
/// tried to modify a shared variable, the compiler would reject it. That
/// guarantee is the main reason to write this in Rust.
pub fn bfs_many(
    g: &Graph,
    sources: &[NodeId],
    dir: Direction,
    cutoff: Option<usize>,
) -> Vec<BfsTree> {
    sources
        .par_iter()
        .map_init(Workspace::new, |ws, &s| bfs(g, s, dir, cutoff, ws))
        .collect()
}

/// What one breadth-first search found, when only totals are wanted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BfsTotal {
    /// How many nodes the search reached, the source included.
    pub reached: usize,
    /// The sum of the distances to them.
    pub distance_sum: u64,
}

/// How many searches [`MultiBfs`] runs at once: the number of bits in a
/// `u64`.
const LANES: usize = 64;

/// Working memory for [`MultiBfs::run`]. Each entry of each array belongs to
/// one node and is a `u64` used as 64 separate yes/no flags, one per search.
/// Bit `i` of a node's entry is that node's flag for search number `i`.
struct MultiBfs {
    /// Flag `i` set: search `i` has reached this node.
    seen: Vec<u64>,
    /// Flag `i` set: search `i` reached this node in the level just finished.
    frontier: Vec<u64>,
    /// Flag `i` set: search `i` reaches this node in the level being built.
    next: Vec<u64>,
    /// The nodes whose `frontier` entry is not zero.
    frontier_nodes: Vec<NodeId>,
    /// The nodes whose `next` entry is not zero.
    next_nodes: Vec<NodeId>,
}

impl MultiBfs {
    fn new(n: usize) -> Self {
        Self {
            seen: vec![0; n],
            frontier: vec![0; n],
            next: vec![0; n],
            frontier_nodes: Vec::new(),
            next_nodes: Vec::new(),
        }
    }

    /// Breadth-first search from up to 64 sources at once.
    ///
    /// An ordinary search asks, for every edge `v -> w` out of the current
    /// level, "has the search seen `w`?". Running 64 searches means asking
    /// that 64 times per edge. Here the 64 answers sit side by side in one
    /// integer, so one `|` (bitwise or) and one `&` (bitwise and) answer the
    /// question for all 64 searches together. The CPU does either in a
    /// single instruction.
    ///
    /// The searches do not interfere: bit `i` only ever meets bit `i`. Each
    /// one visits exactly the nodes, at exactly the distances, that it would
    /// on its own, so the totals are the same as from 64 separate searches.
    /// The saving is largest when the searches overlap, which they do on any
    /// graph where most nodes are a few steps from most others.
    ///
    /// The technique is known as multi-source or bit-parallel BFS. rustnx,
    /// another Rust engine for NetworkX, uses it for closeness, which is
    /// where the idea to use it here came from.
    fn run(&mut self, adj: &Csr, sources: &[NodeId]) -> Vec<BfsTotal> {
        debug_assert!(sources.len() <= LANES);
        // One total per search. A source reaches itself at distance 0.
        let mut totals = vec![
            BfsTotal {
                reached: 1,
                distance_sum: 0
            };
            sources.len()
        ];

        // `enumerate` pairs each item with its position: search `lane`
        // starts at node `s`. `1 << lane` is the integer with only bit
        // `lane` set.
        for (lane, &s) in sources.iter().enumerate() {
            let si = s as usize;
            // The same node may be listed twice, so check before pushing.
            if self.frontier[si] == 0 {
                self.frontier_nodes.push(s);
            }
            self.frontier[si] |= 1 << lane;
            self.seen[si] |= 1 << lane;
        }

        let n = self.seen.len();
        let mut level = 0u64;
        while !self.frontier_nodes.is_empty() {
            level += 1;

            // There are two ways to find the nodes reached in this level.
            // *Listing* notes each one in `next_nodes` as it is reached,
            // which costs nothing for the nodes not reached. *Sweeping*
            // looks at every node afterwards, which is cheaper per node
            // reached and leaves them in index order, the order that is
            // kindest to the CPU's memory cache. Sweep when the level is
            // likely to reach a good share of the graph.
            let sweep = self.frontier_nodes.len() >= n / 16;

            // Push every flag in the frontier along every edge.
            for &v in &self.frontier_nodes {
                let arriving = self.frontier[v as usize];
                for &w in adj.neighbors(v) {
                    let wi = w as usize;
                    // `!x` flips every bit. `a & !b` keeps the flags set in
                    // `a` and not in `b`: the searches arriving at `w` that
                    // have not been here before.
                    let new = arriving & !self.seen[wi];
                    if sweep {
                        self.next[wi] |= new;
                    } else if new != 0 {
                        if self.next[wi] == 0 {
                            self.next_nodes.push(w);
                        }
                        self.next[wi] |= new;
                    }
                }
            }

            // The frontier has been used; clear its flags.
            for &v in &self.frontier_nodes {
                self.frontier[v as usize] = 0;
            }
            self.frontier_nodes.clear();

            if sweep {
                // `filter` keeps the items for which the closure is true;
                // `extend` appends everything an iterator yields.
                let next = &self.next;
                self.next_nodes
                    .extend((0..n as NodeId).filter(|&w| next[w as usize] != 0));
            }

            // The nodes just reached become the frontier of the next level.
            for &w in &self.next_nodes {
                let wi = w as usize;
                let new = self.next[wi];
                self.next[wi] = 0;
                self.seen[wi] |= new;
                self.frontier[wi] = new;

                // Credit each search that arrived. `trailing_zeros` is the
                // position of the lowest set bit, and `bits & (bits - 1)`
                // clears that bit, so the loop runs once per set bit.
                let mut bits = new;
                while bits != 0 {
                    let total = &mut totals[bits.trailing_zeros() as usize];
                    total.reached += 1;
                    total.distance_sum += level;
                    bits &= bits - 1;
                }
            }
            // `swap` exchanges the two Vecs without copying their contents.
            // `next_nodes` receives the cleared `frontier_nodes`.
            std::mem::swap(&mut self.frontier_nodes, &mut self.next_nodes);
        }

        // Leave the memory clean for the next batch. `fill` is a plain
        // memory write, far cheaper than the search was.
        self.seen.fill(0);
        totals
    }
}

/// For each of `sources`: how many nodes a breadth-first search from it
/// reaches, and the sum of the distances to them. Results are in the order
/// of `sources`.
///
/// The searches run 64 at a time (see [`MultiBfs::run`]), and the batches of
/// 64 run in parallel. `par_chunks(64)` is rayon's parallel version of
/// `chunks(64)`, which yields consecutive sub-slices of at most 64 items.
/// `flatten` then turns the list of per-batch lists into one list.
pub fn bfs_totals(g: &Graph, sources: &[NodeId], dir: Direction) -> Vec<BfsTotal> {
    let adj = g.adjacency(dir);
    let n = g.node_count();
    let batches: Vec<Vec<BfsTotal>> = sources
        .par_chunks(LANES)
        .map_init(|| MultiBfs::new(n), |state, batch| state.run(adj, batch))
        .collect();
    batches.into_iter().flatten().collect()
}

/// The sum of shortest-path lengths over all ordered pairs of nodes,
/// counting only pairs where a path exists.
pub fn bfs_distance_sum(g: &Graph) -> u64 {
    let sources: Vec<NodeId> = g.nodes().collect();
    bfs_totals(g, &sources, Direction::Forward)
        .iter()
        .map(|total| total.distance_sum)
        .sum()
}

/// Follows `ws.parent` links from `from` until a node with no parent.
fn follow_parents(ws: &Workspace, from: NodeId) -> Vec<NodeId> {
    let mut nodes = vec![from];
    let mut v = from;
    while ws.parent[v as usize] != NO_NODE {
        v = ws.parent[v as usize];
        nodes.push(v);
    }
    nodes
}

/// Joins the two halves of a bidirectional search at `meet`: the forward
/// search's parent links lead back to the source, the backward search's
/// lead on to the target.
fn join_at(forward: &Workspace, backward: &Workspace, meet: NodeId) -> Vec<NodeId> {
    let mut path = follow_parents(forward, meet);
    path.reverse();
    // `[1..]` skips `meet` itself, which is already in the path.
    path.extend_from_slice(&follow_parents(backward, meet)[1..]);
    path
}

/// One shortest path from `source` to `target` by number of edges, or `None`
/// if there is none. Searches from both ends and stops where they meet.
///
/// Port of `networkx.bidirectional_shortest_path`. `networkx.shortest_path`
/// uses it when given a source, a target and no weight.
pub fn bidirectional_bfs(
    g: &Graph,
    source: NodeId,
    target: NodeId,
    forward: &mut Workspace,
    backward: &mut Workspace,
) -> Option<Vec<NodeId>> {
    if source == target {
        return Some(vec![source]);
    }
    let n = g.node_count();
    forward.prepare(n);
    backward.prepare(n);
    // NetworkX keeps dicts `pred` and `succ` and tests `w not in pred`.
    // Here a node's slot says whether it is in the dict, and `parent` holds
    // the value.
    const IN_DICT: u32 = 0;
    forward.slot[source as usize] = IN_DICT;
    forward.touch(source);
    backward.slot[target as usize] = IN_DICT;
    backward.touch(target);

    let mut forward_fringe = vec![source];
    let mut reverse_fringe = vec![target];
    let mut meet = None;

    // `'search:` labels the outer loop so the inner loops can break out of it.
    'search: while !forward_fringe.is_empty() && !reverse_fringe.is_empty() {
        if forward_fringe.len() <= reverse_fringe.len() {
            let this_level = std::mem::take(&mut forward_fringe);
            for &v in &this_level {
                for &w in g.succ().neighbors(v) {
                    if forward.slot[w as usize] == UNSEEN {
                        forward_fringe.push(w);
                        forward.slot[w as usize] = IN_DICT;
                        forward.parent[w as usize] = v;
                        forward.touch(w);
                    }
                    if backward.slot[w as usize] != UNSEEN {
                        meet = Some(w);
                        break 'search;
                    }
                }
            }
        } else {
            let this_level = std::mem::take(&mut reverse_fringe);
            for &v in &this_level {
                for &w in g.pred().neighbors(v) {
                    if backward.slot[w as usize] == UNSEEN {
                        backward.slot[w as usize] = IN_DICT;
                        backward.parent[w as usize] = v;
                        backward.touch(w);
                        reverse_fringe.push(w);
                    }
                    if forward.slot[w as usize] != UNSEEN {
                        meet = Some(w);
                        break 'search;
                    }
                }
            }
        }
    }

    let path = meet.map(|w| join_at(forward, backward, w));
    forward.release();
    backward.release();
    path
}

// ---------------------------------------------------------------------------
// Dijkstra
// ---------------------------------------------------------------------------

/// An entry in Dijkstra's priority queue.
///
/// NetworkX pushes tuples `(distance, counter, node)` onto a `heapq`. The
/// counter increases with every push, so among entries with equal distance
/// the one pushed first is popped first. Which shortest path is found
/// depends on that, so we keep the counter.
#[derive(Debug, Clone, Copy)]
struct QueueEntry {
    dist: f64,
    seq: u64,
    node: NodeId,
    /// The node this entry was reached from. Only betweenness uses it.
    from: NodeId,
}

// `BinaryHeap` needs to compare entries, which means implementing the `Ord`
// trait. `Ord` requires `PartialOrd`, `Eq` and `PartialEq` as well; they all
// defer to `cmp` below.
//
// `BinaryHeap` is a *max*-heap: `pop` returns the greatest entry. We want
// the smallest distance first, so `cmp` is written backwards (`other`
// compared to `self`).
impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // f64 only implements `PartialOrd`, because NaN compares as neither
        // less, equal nor greater. `partial_cmp` returns an Option; our
        // distances are never NaN, and we treat the impossible case as equal.
        // `.then_with` breaks ties using the next key.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for QueueEntry {}

/// A priority queue that pops the smallest distance first and, among equal
/// distances, the entry pushed first. Wraps `BinaryHeap` and the counter.
///
/// `pub(crate)` makes it visible inside this crate only. Betweenness
/// centrality reuses it.
#[derive(Debug, Default)]
pub(crate) struct MinQueue {
    heap: BinaryHeap<QueueEntry>,
    seq: u64,
}

impl MinQueue {
    pub(crate) fn push(&mut self, dist: f64, node: NodeId, from: NodeId) {
        self.heap.push(QueueEntry {
            dist,
            seq: self.seq,
            node,
            from,
        });
        self.seq += 1;
    }

    /// Returns `(distance, node, from)`.
    pub(crate) fn pop(&mut self) -> Option<(f64, NodeId, NodeId)> {
        self.heap.pop().map(|e| (e.dist, e.node, e.from))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.heap.clear();
        self.seq = 0;
    }
}

/// Dijkstra's algorithm from `source`.
///
/// - `weight(e)` is the weight of edge `e`.
/// - `target`: stop as soon as this node's distance is final.
/// - `cutoff`: ignore paths longer than this.
///
/// All weights must be non-negative and not NaN. The Python wrapper checks
/// this and hands graphs with negative weights to NetworkX.
///
/// Port of `networkx`'s `_dijkstra_multisource` for a single source.
pub fn dijkstra<W>(
    g: &Graph,
    weight: W,
    source: NodeId,
    dir: Direction,
    target: Option<NodeId>,
    cutoff: Option<f64>,
    ws: &mut Workspace,
) -> ShortestPaths
where
    W: Fn(EdgeId) -> f64,
{
    let adj = g.adjacency(dir);
    ws.prepare(g.node_count());

    // While the search runs:
    // - `ws.dist[v]` is the best distance seen so far for v, infinity
    //   standing for "not seen" where NetworkX tests `u not in seen`;
    // - `ws.parent[v]` is the node that gave v that distance;
    // - `ws.slot[v]` is v's position in `order` once its distance is final.
    let mut order = Vec::new();
    let mut dist = Vec::new();
    let mut parent = Vec::new();

    // `MinQueue::default()` calls the `Default` implementation that
    // `#[derive(Default)]` generated: an empty heap and a zero counter.
    let mut queue = MinQueue::default();
    ws.dist[source as usize] = 0.0;
    ws.touch(source);
    queue.push(0.0, source, NO_NODE);

    while let Some((d, v, _)) = queue.pop() {
        if ws.slot[v as usize] != UNSEEN {
            // A stale entry: v was pushed again with a smaller distance and
            // that entry was popped first.
            continue;
        }
        ws.slot[v as usize] = order.len() as u32;
        order.push(v);
        dist.push(d);
        parent.push(match ws.parent[v as usize] {
            NO_NODE => NO_PARENT,
            // The parent's distance became final before v's, so its slot
            // already holds its position.
            p => ws.slot[p as usize],
        });
        if target == Some(v) {
            break;
        }
        for (u, e) in adj.arcs(v) {
            let through_v = d + weight(e);
            if cutoff.is_some_and(|c| through_v > c) {
                continue;
            }
            // Strictly less: on a tie the first path found is kept.
            if ws.slot[u as usize] == UNSEEN && through_v < ws.dist[u as usize] {
                if ws.dist[u as usize] == f64::INFINITY {
                    ws.touch(u);
                }
                ws.dist[u as usize] = through_v;
                ws.parent[u as usize] = v;
                queue.push(through_v, u, v);
            }
        }
    }
    ws.release();
    Tree {
        order,
        dist,
        parent,
    }
}

/// [`dijkstra`] from each of `sources`, run in parallel.
///
/// `W: ... + Sync` adds a requirement: the closure must be safe to call
/// from several threads at once. One that only reads a slice is.
pub fn dijkstra_many<W>(
    g: &Graph,
    weight: W,
    sources: &[NodeId],
    dir: Direction,
    cutoff: Option<f64>,
) -> Vec<ShortestPaths>
where
    W: Fn(EdgeId) -> f64 + Sync,
{
    sources
        .par_iter()
        // `&weight` lends the closure to each call; a reference to a
        // closure is itself callable.
        .map_init(Workspace::new, |ws, &s| {
            dijkstra(g, &weight, s, dir, None, cutoff, ws)
        })
        .collect()
}

/// One shortest weighted path from `source` to `target` and its length, or
/// `None` if there is no path. Runs Dijkstra from both ends, alternating.
///
/// Port of `networkx.bidirectional_dijkstra`. `networkx.shortest_path` uses
/// it when given a source, a target and a weight.
pub fn bidirectional_dijkstra<W>(
    g: &Graph,
    weight: W,
    source: NodeId,
    target: NodeId,
    forward: &mut Workspace,
    backward: &mut Workspace,
) -> Option<(f64, Vec<NodeId>)>
where
    W: Fn(EdgeId) -> f64,
{
    if source == target {
        return Some((0.0, vec![source]));
    }
    let n = g.node_count();
    forward.prepare(n);
    backward.prepare(n);
    const FINAL: u32 = 0;

    // Index 0 is the search forwards from the source, index 1 the search
    // backwards from the target. `[a, b]` is a fixed-size array. Holding
    // the two workspaces in one lets the loop below pick "this side" and
    // "the other side" by index.
    let adjacency = [g.succ(), g.pred()];
    let side = [forward, backward];
    // NetworkX shares one counter between both heaps. Each MinQueue here has
    // its own, which gives the same order: entries are only ever compared
    // with others in the same heap.
    let mut fringe = [MinQueue::default(), MinQueue::default()];

    for (i, start) in [source, target].into_iter().enumerate() {
        side[i].dist[start as usize] = 0.0;
        side[i].touch(start);
        fringe[i].push(0.0, start, NO_NODE);
    }

    // The best complete path found so far: its length and the node where
    // the two halves join.
    let mut best: Option<(f64, NodeId)> = None;
    let mut found = None;
    let mut dir = 1;

    while !fringe[0].is_empty() && !fringe[1].is_empty() {
        dir = 1 - dir;
        let other = 1 - dir;
        // `let ... else` binds the pattern or runs the else block, which
        // must leave (here with `continue`).
        let Some((d, v, _)) = fringe[dir].pop() else {
            continue;
        };
        if side[dir].slot[v as usize] == FINAL {
            continue;
        }
        side[dir].slot[v as usize] = FINAL;
        if side[other].slot[v as usize] == FINAL {
            // Both searches have settled v: the best path found is optimal.
            found = best;
            break;
        }
        for (w, e) in adjacency[dir].arcs(v) {
            let wi = w as usize;
            let through_v = d + weight(e);
            if side[dir].slot[wi] != FINAL && through_v < side[dir].dist[wi] {
                if side[dir].dist[wi] == f64::INFINITY {
                    side[dir].touch(w);
                }
                side[dir].dist[wi] = through_v;
                side[dir].parent[wi] = v;
                fringe[dir].push(through_v, w, v);
                let from_other_side = side[other].dist[wi];
                if from_other_side < f64::INFINITY {
                    let total = through_v + from_other_side;
                    // `is_none_or(f)`: true for None, else `f(value)`.
                    if best.is_none_or(|(length, _)| length > total) {
                        best = Some((total, w));
                    }
                }
            }
        }
    }

    // Destructure the array to get the two `&mut Workspace`s back out.
    let [forward, backward] = side;
    let result = found.map(|(length, meet)| (length, join_at(forward, backward, meet)));
    forward.release();
    backward.release();
    result
}

// ---------------------------------------------------------------------------
// Bellman-Ford
// ---------------------------------------------------------------------------

/// A cycle of negative total weight is reachable from the source, so
/// shortest paths are not defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegativeCycle;

impl fmt::Display for NegativeCycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "negative cycle detected")
    }
}

impl std::error::Error for NegativeCycle {}

/// Working memory for [`bellman_ford`], reusable like a [`Workspace`].
/// Bellman-Ford keeps more per-node state than the other searches, so it
/// has a workspace of its own.
#[derive(Debug, Default, Clone)]
pub struct BellmanFordWorkspace {
    /// Position in the result's `order`, or [`UNSEEN`] if not reached.
    slot: Vec<u32>,
    dist: Vec<f64>,
    /// Every node that comes immediately before this one on some shortest
    /// path.
    preds: Vec<Vec<NodeId>>,
    // State for NetworkX's negative-cycle heuristic.
    pred_edge: Vec<NodeId>,
    recent_update: Vec<(NodeId, NodeId)>,
    /// How many times each node has entered the queue.
    count: Vec<u32>,
    in_queue: Vec<bool>,
}

impl BellmanFordWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self, n: usize) {
        if self.slot.len() != n {
            // `*self = ...` replaces the whole struct through the reference.
            *self = Self {
                slot: vec![UNSEEN; n],
                dist: vec![f64::INFINITY; n],
                preds: vec![Vec::new(); n],
                pred_edge: vec![NO_NODE; n],
                recent_update: vec![(NO_NODE, NO_NODE); n],
                count: vec![0; n],
                in_queue: vec![false; n],
            };
        }
    }
}

/// The result of the Bellman-Ford algorithm from one node.
#[derive(Debug, Clone, PartialEq)]
pub struct BellmanFord {
    /// The reached nodes, in the order they were first reached. The source
    /// is first.
    pub order: Vec<NodeId>,
    /// `dist[i]` is the distance to `order[i]`.
    pub dist: Vec<f64>,
    /// `preds[i]` lists, as positions in `order`, every node that comes
    /// immediately before `order[i]` on some shortest path.
    pub preds: Vec<Vec<u32>>,
}

impl BellmanFord {
    /// The position of `node` in [`order`](Self::order), or `None` if it
    /// was not reached.
    pub fn position(&self, node: NodeId) -> Option<usize> {
        self.order.iter().position(|&v| v == node)
    }

    /// The shortest path from the source to the node at position `i`.
    ///
    /// `on_stack` is working memory, all `false`, one entry per reached
    /// node. It is all `false` again on return. Passing it in lets
    /// [`all_paths`](Self::all_paths) reuse one array for every target.
    ///
    /// Port of NetworkX's `_build_paths_from_predecessors`, taking the first
    /// path it yields.
    pub fn path_to_position(&self, i: usize, on_stack: &mut [bool]) -> Vec<NodeId> {
        // Depth-first search back along predecessor lists. `on_stack`
        // prevents walking round a zero-weight cycle forever. Each stack
        // entry is a position and how many of its predecessors were tried.
        let mut stack: Vec<(u32, usize)> = vec![(i as u32, 0)];
        on_stack[i] = true;
        let mut path = Vec::new();
        while let Some(top) = stack.last_mut() {
            let at = top.0 as usize;
            if at == 0 {
                // Position 0 is the source: the stack, read from the bottom
                // up, is the path backwards.
                path = stack
                    .iter()
                    .rev()
                    .map(|&(p, _)| self.order[p as usize])
                    .collect();
                break;
            }
            let preds = &self.preds[at];
            if top.1 < preds.len() {
                let next = preds[top.1];
                top.1 += 1;
                if !on_stack[next as usize] {
                    on_stack[next as usize] = true;
                    stack.push((next, 0));
                }
            } else {
                on_stack[at] = false;
                stack.pop();
            }
        }
        for &(p, _) in &stack {
            on_stack[p as usize] = false;
        }
        path
    }

    /// The shortest path from the source to `target`, or `None` if it was
    /// not reached.
    pub fn path_to(&self, target: NodeId) -> Option<Vec<NodeId>> {
        let mut on_stack = vec![false; self.order.len()];
        self.position(target)
            .map(|i| self.path_to_position(i, &mut on_stack))
    }

    /// The shortest path to every reached node, in `order`.
    pub fn all_paths(&self) -> Vec<Vec<NodeId>> {
        let mut on_stack = vec![false; self.order.len()];
        (0..self.order.len())
            .map(|i| self.path_to_position(i, &mut on_stack))
            .collect()
    }
}

/// The Bellman-Ford algorithm from `source`. Weights may be negative.
///
/// Port of `networkx`'s `_inner_bellman_ford`, which is the queue-based
/// variant (also called SPFA) with a heuristic that spots most negative
/// cycles early.
pub fn bellman_ford<W>(
    g: &Graph,
    weight: W,
    source: NodeId,
    dir: Direction,
    ws: &mut BellmanFordWorkspace,
) -> Result<BellmanFord, NegativeCycle>
where
    W: Fn(EdgeId) -> f64,
{
    let n = g.node_count();
    ws.prepare(n);
    let mut order = vec![source];
    ws.slot[source as usize] = 0;
    ws.dist[source as usize] = 0.0;

    // The search proper is a separate function so that, whichever way it
    // ends, the clean-up below runs.
    let outcome = relax_until_stable(g, weight, source, dir, ws, &mut order);

    // Move the results out of the workspace and blank it in the same pass.
    let mut dist = Vec::with_capacity(order.len());
    let mut preds = Vec::with_capacity(order.len());
    for &v in &order {
        let vi = v as usize;
        dist.push(ws.dist[vi]);
        // `take` moves the list out and leaves an empty one in its place.
        // Each predecessor is translated to its position while the slots
        // still hold them.
        let list = std::mem::take(&mut ws.preds[vi]);
        preds.push(list.into_iter().map(|p| ws.slot[p as usize]).collect());
    }
    for &v in &order {
        let vi = v as usize;
        ws.slot[vi] = UNSEEN;
        ws.dist[vi] = f64::INFINITY;
        ws.pred_edge[vi] = NO_NODE;
        ws.recent_update[vi] = (NO_NODE, NO_NODE);
        ws.count[vi] = 0;
        ws.in_queue[vi] = false;
    }

    // `outcome?` returns the error, if there was one, now that the
    // workspace is clean.
    outcome?;
    Ok(BellmanFord { order, dist, preds })
}

/// The main loop of [`bellman_ford`]. Appends newly reached nodes to
/// `order`.
fn relax_until_stable<W>(
    g: &Graph,
    weight: W,
    source: NodeId,
    dir: Direction,
    ws: &mut BellmanFordWorkspace,
    order: &mut Vec<NodeId>,
) -> Result<(), NegativeCycle>
where
    W: Fn(EdgeId) -> f64,
{
    let adj = g.adjacency(dir);
    let n = g.node_count() as u32;

    // `VecDeque` is a double-ended queue, like Python's `collections.deque`.
    let mut queue = VecDeque::from([source]);
    ws.in_queue[source as usize] = true;

    while let Some(u) = queue.pop_front() {
        let ui = u as usize;
        ws.in_queue[ui] = false;
        // Skip u if one of its predecessors is waiting in the queue: u would
        // only be relaxed again afterwards. `any` is true if the closure
        // holds for at least one item.
        if ws.preds[ui].iter().any(|&p| ws.in_queue[p as usize]) {
            continue;
        }
        let dist_u = ws.dist[ui];
        for (v, e) in adj.arcs(u) {
            let vi = v as usize;
            let dist_v = dist_u + weight(e);
            if dist_v < ws.dist[vi] {
                let (a, b) = ws.recent_update[ui];
                if v == a || v == b {
                    return Err(NegativeCycle);
                }
                if ws.pred_edge[vi] == u {
                    ws.recent_update[vi] = ws.recent_update[ui];
                } else {
                    ws.recent_update[vi] = (u, v);
                }
                if ws.slot[vi] == UNSEEN {
                    // First time v is reached. Recorded before any early
                    // return so that the clean-up knows about it.
                    ws.slot[vi] = order.len() as u32;
                    order.push(v);
                }
                if !ws.in_queue[vi] {
                    queue.push_back(v);
                    ws.in_queue[vi] = true;
                    ws.count[vi] += 1;
                    if ws.count[vi] == n {
                        return Err(NegativeCycle);
                    }
                }
                ws.dist[vi] = dist_v;
                // Replace the list with one holding just u, reusing its
                // memory.
                ws.preds[vi].clear();
                ws.preds[vi].push(u);
                ws.pred_edge[vi] = u;
            } else if ws.slot[vi] != UNSEEN && dist_v == ws.dist[vi] {
                // Another equally short way to reach v.
                ws.preds[vi].push(u);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square 0-1-3 / 0-2-3 with a pendant node 4 on 3.
    fn square() -> Graph {
        Graph::from_edges(5, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], false).unwrap()
    }

    /// Searches run 64 at a time must find what one search at a time finds.
    #[test]
    fn bfs_totals_match_separate_searches() {
        // 200 nodes so that there are several batches of 64, with a ring,
        // some chords, a second component (150..160) and isolated nodes.
        let mut edges: Vec<(NodeId, NodeId)> = (0..150).map(|u| (u, (u + 1) % 150)).collect();
        edges.extend((0..150).step_by(7).map(|u| (u, (u * 3 + 11) % 150)));
        edges.extend((150..159).map(|u| (u, u + 1)));
        for directed in [false, true] {
            let g = Graph::from_edges(200, &edges, directed).unwrap();
            // Every node, then node 5 again: a repeated source is allowed.
            let mut sources: Vec<NodeId> = g.nodes().collect();
            sources.push(5);
            for dir in [Direction::Forward, Direction::Reverse] {
                let mut ws = Workspace::new();
                let expected: Vec<BfsTotal> = sources
                    .iter()
                    .map(|&s| {
                        let tree = bfs(&g, s, dir, None, &mut ws);
                        BfsTotal {
                            reached: tree.order.len(),
                            distance_sum: tree.dist.iter().map(|&d| u64::from(d)).sum(),
                        }
                    })
                    .collect();
                assert_eq!(bfs_totals(&g, &sources, dir), expected);
            }
        }
    }

    #[test]
    fn bfs_distances_and_paths() {
        let g = square();
        let mut ws = Workspace::new();
        let tree = bfs(&g, 0, Direction::Forward, None, &mut ws);
        assert_eq!(tree.order, vec![0, 1, 2, 3, 4]);
        assert_eq!(tree.dist, vec![0, 1, 1, 2, 3]);
        // Node 3 is found via 1, the first of its parents to be expanded.
        assert_eq!(tree.path_to(4), Some(vec![0, 1, 3, 4]));
        let limited = bfs(&g, 0, Direction::Forward, Some(1), &mut ws);
        assert_eq!(limited.order, vec![0, 1, 2]);
        assert_eq!(limited.path_to(3), None);
        assert!(ws.is_clean());
    }

    #[test]
    fn bfs_sum_and_parallel_agree_with_serial() {
        let g = square();
        let many = bfs_many(&g, &[0, 4], Direction::Forward, None);
        assert_eq!(
            many[1],
            bfs(&g, 4, Direction::Forward, None, &mut Workspace::new())
        );
        // Sum of all pairwise distances, each pair counted in both directions.
        assert_eq!(
            bfs_distance_sum(&g),
            2 * (1 + 1 + 2 + 3 + 2 + 1 + 2 + 1 + 2 + 1)
        );
    }

    #[test]
    fn bidirectional_bfs_finds_a_shortest_path() {
        let g = square();
        let (mut a, mut b) = (Workspace::new(), Workspace::new());
        assert_eq!(
            bidirectional_bfs(&g, 0, 4, &mut a, &mut b),
            Some(vec![0, 1, 3, 4])
        );
        assert_eq!(bidirectional_bfs(&g, 2, 2, &mut a, &mut b), Some(vec![2]));
        let split = Graph::from_edges(2, &[], false).unwrap();
        assert_eq!(bidirectional_bfs(&split, 0, 1, &mut a, &mut b), None);
        assert!(a.is_clean() && b.is_clean());
    }

    #[test]
    fn queue_pops_smallest_then_oldest() {
        let mut q = MinQueue::default();
        q.push(2.0, 10, NO_NODE);
        q.push(1.0, 11, NO_NODE);
        q.push(1.0, 12, NO_NODE);
        assert_eq!(q.pop().unwrap().1, 11);
        assert_eq!(q.pop().unwrap().1, 12);
        assert_eq!(q.pop().unwrap().1, 10);
        assert!(q.pop().is_none());
    }

    #[test]
    fn dijkstra_prefers_lighter_route() {
        let g = square();
        // Edge ids: (0,1)=0 (0,2)=1 (1,3)=2 (2,3)=3 (3,4)=4.
        let weights = [5.0, 1.0, 1.0, 1.0, 1.0];
        let weight = |e: EdgeId| weights[e as usize];
        let (mut a, mut b) = (Workspace::new(), Workspace::new());
        let sp = dijkstra(&g, weight, 0, Direction::Forward, None, None, &mut a);
        assert_eq!(sp.order, vec![0, 2, 3, 1, 4]);
        assert_eq!(sp.dist, vec![0.0, 1.0, 2.0, 3.0, 3.0]);
        assert_eq!(sp.path_to(4), Some(vec![0, 2, 3, 4]));
        let cut = dijkstra(&g, weight, 0, Direction::Forward, None, Some(1.5), &mut a);
        assert_eq!(cut.order, vec![0, 2]);
        assert_eq!(cut.path_to(4), None);
        let early = dijkstra(&g, weight, 0, Direction::Forward, Some(3), None, &mut a);
        assert_eq!(early.order, vec![0, 2, 3]);
        assert_eq!(
            bidirectional_dijkstra(&g, weight, 0, 4, &mut a, &mut b),
            Some((3.0, vec![0, 2, 3, 4]))
        );
        assert!(a.is_clean() && b.is_clean());
        assert_eq!(
            dijkstra_many(&g, weight, &[0], Direction::Forward, None)[0],
            sp
        );
    }

    #[test]
    fn bellman_ford_handles_negative_edges() {
        // 0 -> 1 (4), 0 -> 2 (1), 2 -> 1 (-2)
        let g = Graph::from_edges(3, &[(0, 1), (0, 2), (2, 1)], true).unwrap();
        let weights = [4.0, 1.0, -2.0];
        let mut ws = BellmanFordWorkspace::new();
        let bf = bellman_ford(&g, |e| weights[e as usize], 0, Direction::Forward, &mut ws).unwrap();
        assert_eq!(bf.order, vec![0, 1, 2]);
        assert_eq!(bf.dist, vec![0.0, -1.0, 1.0]);
        assert_eq!(bf.all_paths(), vec![vec![0], vec![0, 2, 1], vec![0, 2]]);
        assert_eq!(bf.path_to(1), Some(vec![0, 2, 1]));
        // The workspace is reusable: a second run gives the same answer.
        let again =
            bellman_ford(&g, |e| weights[e as usize], 0, Direction::Forward, &mut ws).unwrap();
        assert_eq!(again, bf);
    }

    #[test]
    fn bellman_ford_reports_negative_cycle_and_recovers() {
        let g = Graph::from_edges(2, &[(0, 1), (1, 0)], true).unwrap();
        let mut ws = BellmanFordWorkspace::new();
        let bad = [1.0, -2.0];
        assert_eq!(
            bellman_ford(&g, |e| bad[e as usize], 0, Direction::Forward, &mut ws),
            Err(NegativeCycle)
        );
        // The failed run must not leave anything behind in the workspace.
        let good = [1.0, 2.0];
        let bf = bellman_ford(&g, |e| good[e as usize], 0, Direction::Forward, &mut ws).unwrap();
        assert_eq!(bf.dist, vec![0.0, 1.0]);
    }
}
