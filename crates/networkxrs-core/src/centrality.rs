//! Centrality measures: PageRank, degree, betweenness and closeness.
//!
//! Ports of `networkx.pagerank` and functions in
//! `networkx.algorithms.centrality`.

use std::fmt;

use rayon::prelude::*;

use crate::graph::{Direction, EdgeId, Graph, NodeId};
use crate::shortest_paths::{bfs, bfs_totals, dijkstra, MinQueue};
use crate::workspace::Workspace;

// ---------------------------------------------------------------------------
// PageRank
// ---------------------------------------------------------------------------

/// Settings for [`pagerank`].
///
/// The `'a` is a *lifetime parameter*. The struct holds borrowed slices, and
/// `'a` names how long those borrows last, so the compiler can check that a
/// `PageRankOptions` is not kept around longer than the data it points to.
/// You rarely write lifetimes in function signatures because the compiler
/// infers them, but a struct that stores a reference must name one.
#[derive(Debug, Clone)]
pub struct PageRankOptions<'a> {
    /// Damping factor.
    pub alpha: f64,
    /// Give up after this many iterations.
    pub max_iter: usize,
    /// Stop when the total change in one iteration is below `n * tol`.
    pub tol: f64,
    /// Where a random jump lands, one probability per node, summing to 1.
    /// `None` means uniform.
    pub personalization: Option<&'a [f64]>,
    /// The starting vector, summing to 1. `None` means uniform.
    pub nstart: Option<&'a [f64]>,
    /// Where the rank of a node without out-edges goes, summing to 1.
    /// `None` means the same as `personalization`.
    pub dangling: Option<&'a [f64]>,
}

// `Default` is the trait behind `PageRankOptions::default()`. Implementing
// it lets callers set only the fields they care about:
//
//     PageRankOptions { alpha: 0.9, ..Default::default() }
//
// The lifetime is written `'_` here because no field of the default value
// borrows anything.
impl Default for PageRankOptions<'_> {
    fn default() -> Self {
        Self {
            alpha: 0.85,
            max_iter: 100,
            tol: 1.0e-6,
            personalization: None,
            nstart: None,
            dangling: None,
        }
    }
}

/// [`pagerank`] did not converge within `max_iter` iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotConverged {
    pub iterations: usize,
}

impl fmt::Display for NotConverged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "power iteration failed to converge within {} iterations",
            self.iterations
        )
    }
}

impl std::error::Error for NotConverged {}

/// PageRank by power iteration.
///
/// `weights` is indexed by edge id; `None` gives every edge weight 1. An
/// undirected graph is treated as directed with an edge each way.
///
/// Port of `networkx.pagerank`, which does the same arithmetic with a SciPy
/// sparse matrix. Results agree with NetworkX to floating-point rounding
/// rather than bit for bit, because SciPy adds some terms in a different
/// order.
pub fn pagerank(
    g: &Graph,
    weights: Option<&[f64]>,
    options: &PageRankOptions,
) -> Result<Vec<f64>, NotConverged> {
    let n = g.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }
    let adj = g.succ();
    // A closure that hides whether the graph is weighted.
    let weight = |e: EdgeId| weights.map_or(1.0, |w| w[e as usize]);

    // 1 / (total weight leaving each node), or 0 for a node with nothing
    // leaving it. Such "dangling" nodes are handled separately below.
    let inverse_out: Vec<f64> = g
        .nodes()
        .map(|u| {
            let total: f64 = adj.edge_ids(u).iter().map(|&e| weight(e)).sum();
            if total != 0.0 {
                1.0 / total
            } else {
                0.0
            }
        })
        .collect();
    let dangling_nodes: Vec<usize> = (0..n).filter(|&u| inverse_out[u] == 0.0).collect();

    let uniform = vec![1.0 / n as f64; n];
    // `Option<&[f64]>` is `Copy`, so reading it from behind `&options` is
    // fine. `unwrap_or` falls back to the uniform vector.
    let p = options.personalization.unwrap_or(&uniform);
    let dangling_weights = options.dangling.unwrap_or(p);
    // `to_vec` copies a slice into a new Vec we own and can modify.
    let mut x = options.nstart.unwrap_or(&uniform).to_vec();
    let mut next = vec![0.0; n];
    let alpha = options.alpha;

    for _ in 0..options.max_iter {
        // next = x * A, where A[u][v] = weight(u, v) / out_weight(u):
        // each node shares its rank among its successors.
        next.fill(0.0);
        for u in g.nodes() {
            let share = x[u as usize];
            let scale = inverse_out[u as usize];
            for (v, e) in adj.arcs(u) {
                next[v as usize] += (scale * weight(e)) * share;
            }
        }
        let dangling_sum: f64 = dangling_nodes.iter().map(|&u| x[u]).sum();

        let mut err = 0.0;
        for v in 0..n {
            let value =
                alpha * (next[v] + dangling_sum * dangling_weights[v]) + (1.0 - alpha) * p[v];
            err += (value - x[v]).abs();
            next[v] = value;
        }
        // `swap` exchanges the two Vecs' contents by swapping their
        // pointers; no element is copied.
        std::mem::swap(&mut x, &mut next);
        if err < n as f64 * options.tol {
            return Ok(x);
        }
    }
    Err(NotConverged {
        iterations: options.max_iter,
    })
}

// ---------------------------------------------------------------------------
// Degree
// ---------------------------------------------------------------------------

/// Which degree [`degree_centrality`] counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegreeKind {
    /// All edges at the node (`networkx.degree_centrality`).
    Total,
    /// Incoming edges (`networkx.in_degree_centrality`).
    In,
    /// Outgoing edges (`networkx.out_degree_centrality`).
    Out,
}

/// Each node's degree divided by `n - 1`. Needs at least two nodes.
pub fn degree_centrality(g: &Graph, kind: DegreeKind) -> Vec<f64> {
    let scale = 1.0 / (g.node_count() as f64 - 1.0);
    g.nodes()
        .map(|u| {
            let degree = match kind {
                DegreeKind::Total => g.degree(u),
                DegreeKind::In => g.pred().degree(u),
                DegreeKind::Out => g.succ().degree(u),
            };
            degree as f64 * scale
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Betweenness
// ---------------------------------------------------------------------------

/// Marks a node the unweighted search has not reached.
const NO_LEVEL: u32 = u32::MAX;

/// Working memory for one single-source pass of Brandes' algorithm.
///
/// Betweenness runs one shortest-path search per node. Allocating these
/// arrays afresh for each search would dominate the running time on sparse
/// graphs, so each worker thread makes one `Scratch` and reuses it.
///
/// Between passes every array is back in its starting state. A pass cleans
/// up only the entries it wrote, which are those of the nodes in `order`, so
/// a search that reaches few nodes costs little however large the graph is.
struct Scratch {
    /// Number of shortest paths from the source to each node.
    sigma: Vec<f64>,
    /// Nodes in the order their distance became final. The unweighted
    /// search also uses it as its queue.
    order: Vec<NodeId>,
    /// Dependency of the source on each node (Brandes' delta).
    delta: Vec<f64>,
    // Unweighted search.
    /// Distance from the source, or `NO_LEVEL`.
    level: Vec<u32>,
    // Weighted search.
    /// Predecessors of each node on shortest paths from the source.
    preds: Vec<Vec<NodeId>>,
    seen: Vec<f64>,
    done: Vec<bool>,
    queue: MinQueue,
}

impl Scratch {
    fn new(n: usize, weighted: bool) -> Self {
        // Only one of the two searches is used, so give the other's arrays
        // no memory. An `if` is an expression: it produces a value.
        let (unweighted_n, weighted_n) = if weighted { (0, n) } else { (n, 0) };
        Self {
            sigma: vec![0.0; n],
            order: Vec::with_capacity(n),
            delta: vec![0.0; n],
            level: vec![NO_LEVEL; unweighted_n],
            preds: vec![Vec::new(); weighted_n],
            seen: vec![f64::INFINITY; weighted_n],
            done: vec![false; weighted_n],
            queue: MinQueue::default(),
        }
    }

    /// Counts shortest paths from `s` by breadth-first search.
    /// Port of `_single_source_shortest_path_basic`.
    ///
    /// NetworkX also records each node's predecessors here. This version
    /// does not: after a breadth-first search they can be read off the
    /// levels (see [`Scratch::accumulate`]), which saves building a list per
    /// node.
    fn count_paths_unweighted(&mut self, g: &Graph, s: NodeId) {
        let adj = g.succ();
        self.sigma[s as usize] = 1.0;
        self.level[s as usize] = 0;
        self.order.push(s);
        // `order` is the queue: `head` walks through it while newly found
        // nodes are appended at the end.
        let mut head = 0;
        while head < self.order.len() {
            let v = self.order[head];
            head += 1;
            let level_v = self.level[v as usize];
            let sigma_v = self.sigma[v as usize];
            for &w in adj.neighbors(v) {
                let wi = w as usize;
                if self.level[wi] == NO_LEVEL {
                    self.order.push(w);
                    self.level[wi] = level_v + 1;
                }
                if self.level[wi] == level_v + 1 {
                    self.sigma[wi] += sigma_v;
                }
            }
        }
    }

    /// Counts shortest paths from `s` with Dijkstra's algorithm.
    /// Port of `_single_source_dijkstra_path_basic`, including the detail
    /// that the source ends up with sigma 2: its queue entry names itself as
    /// its own predecessor. The accumulation step divides that back out.
    fn count_paths_weighted(&mut self, g: &Graph, weights: &[f64], s: NodeId) {
        let adj = g.succ();
        self.sigma[s as usize] = 1.0;
        self.seen[s as usize] = 0.0;
        self.queue.push(0.0, s, s);
        while let Some((dist, v, pred)) = self.queue.pop() {
            let vi = v as usize;
            if self.done[vi] {
                continue;
            }
            self.sigma[vi] += self.sigma[pred as usize];
            self.order.push(v);
            self.done[vi] = true;
            for (w, e) in adj.arcs(v) {
                let wi = w as usize;
                let through_v = dist + weights[e as usize];
                if !self.done[wi] && through_v < self.seen[wi] {
                    self.seen[wi] = through_v;
                    self.queue.push(through_v, w, v);
                    self.sigma[wi] = 0.0;
                    self.preds[wi].clear();
                    self.preds[wi].push(v);
                } else if through_v == self.seen[wi] {
                    // Another shortest path to w.
                    self.sigma[wi] += self.sigma[vi];
                    self.preds[wi].push(v);
                }
            }
        }
    }

    /// Turns the path counts into this source's contribution to every
    /// node's betweenness, written into `contribution`, then cleans up.
    /// Ports `_accumulate_basic` and `_accumulate_endpoints`.
    ///
    /// `contribution` must be all zeros on entry. Only the entries of nodes
    /// the search reached are written.
    fn accumulate(
        &mut self,
        g: &Graph,
        weighted: bool,
        s: NodeId,
        endpoints: bool,
        contribution: &mut [f64],
    ) {
        if endpoints {
            contribution[s as usize] = (self.order.len() - 1) as f64;
        }
        // Visit nodes farthest first. `.rev()` walks an iterator backwards.
        for &w in self.order.iter().rev() {
            let wi = w as usize;
            let coeff = (1.0 + self.delta[wi]) / self.sigma[wi];
            if weighted {
                // `self.preds[wi]` and `self.delta` are different fields, so
                // the compiler allows reading one while writing the other.
                for &v in &self.preds[wi] {
                    self.delta[v as usize] += self.sigma[v as usize] * coeff;
                }
            } else {
                // The predecessors of w are its in-neighbours one level
                // closer to the source. NetworkX may list them in another
                // order, but each one's delta receives a single addition
                // here, so the order cannot change any result.
                let level_w = self.level[wi];
                // The source, at level 0, has no predecessors.
                if level_w > 0 {
                    for &v in g.pred().neighbors(w) {
                        let vi = v as usize;
                        // An unreached node has level NO_LEVEL, which is
                        // larger than any real level, so it never matches.
                        if self.level[vi] == level_w - 1 {
                            self.delta[vi] += self.sigma[vi] * coeff;
                        }
                    }
                }
            }
            if w != s {
                contribution[wi] = if endpoints {
                    self.delta[wi] + 1.0
                } else {
                    self.delta[wi]
                };
            }
        }

        // Put back every entry this pass wrote. Both searches write only to
        // nodes that end up in `order`.
        for &v in &self.order {
            let vi = v as usize;
            self.sigma[vi] = 0.0;
            self.delta[vi] = 0.0;
            if weighted {
                self.preds[vi].clear();
                self.seen[vi] = f64::INFINITY;
                self.done[vi] = false;
            } else {
                self.level[vi] = NO_LEVEL;
            }
        }
        self.order.clear();
        // The queue is empty by now; this also restarts its tie-break counter.
        self.queue.clear();
    }
}

/// How many nodes one task handles when contributions are added up.
const ADD_BLOCK: usize = 4096;

/// Betweenness centrality of every node: the share of shortest paths that
/// pass through it.
///
/// - `weights`: `None` counts edges; otherwise paths are weighted.
/// - `normalized`: scale by the number of node pairs.
/// - `endpoints`: count a path's endpoints as lying on it.
///
/// Port of `networkx.betweenness_centrality` without sampling (`k=None`).
///
/// # Why the result is identical to NetworkX's
///
/// Floating-point addition gives slightly different answers depending on
/// the order of the additions. NetworkX handles the sources one after
/// another and adds each one's contribution to a running total per node. To
/// match it to the last bit, each node's total here receives the same
/// numbers in the same order.
///
/// The work is still parallel, in two ways. The searches of a batch of
/// sources run in parallel, each writing its contribution into its own
/// buffer. Then the buffers are added to the totals, and that is parallel
/// across *nodes*: each task owns a block of nodes and, for those nodes,
/// adds the buffers in source order. No two tasks touch the same number, and
/// every number sees its additions in NetworkX's order.
pub fn betweenness_centrality(
    g: &Graph,
    weights: Option<&[f64]>,
    normalized: bool,
    endpoints: bool,
) -> Vec<f64> {
    let n = g.node_count();
    let weighted = weights.is_some();
    let mut betweenness = vec![0.0; n];
    let sources: Vec<NodeId> = g.nodes().collect();

    // Each buffer is n floats, so cap how many exist at about 32 MB.
    let batch = (4_000_000 / n.max(1)).clamp(1, 1024).min(n.max(1));
    // The buffers are made once and reused by every batch.
    let mut buffers: Vec<Vec<f64>> = vec![vec![0.0; n]; batch];

    // `chunks` yields consecutive sub-slices of at most `batch` elements.
    for chunk in sources.chunks(batch) {
        // `[..chunk.len()]` because the last batch may be shorter.
        let buffers = &mut buffers[..chunk.len()];

        // Step 1: one search per source, in parallel. `zip` pairs each
        // buffer with its source. `for_each_init` is `for_each` with
        // per-thread state: the first closure builds one Scratch per worker
        // thread, and the second receives it mutably for every item that
        // thread handles.
        buffers.par_iter_mut().zip(chunk).for_each_init(
            || Scratch::new(n, weighted),
            |scratch, (buffer, &s)| {
                match weights {
                    None => scratch.count_paths_unweighted(g, s),
                    Some(w) => scratch.count_paths_weighted(g, w, s),
                }
                scratch.accumulate(g, weighted, s, endpoints, buffer);
            },
        );

        // Step 2: add the buffers to the totals, a block of nodes per task.
        //
        // The compiler must be shown that the tasks write to different
        // memory. `chunks_mut` does that: it splits one slice into
        // non-overlapping mutable pieces. Cut every buffer and the totals
        // at the same places, then hand each task the matching piece of
        // each.
        //
        // `pieces[j]` is an iterator over buffer j's pieces.
        let mut pieces: Vec<_> = buffers
            .iter_mut()
            .map(|buffer| buffer.chunks_mut(ADD_BLOCK))
            .collect();
        // One entry per block of nodes: that block of the totals, and the
        // same block of every buffer, in source order. Calling `next()` on
        // each iterator in turn takes the next block from each buffer.
        let blocks: Vec<(&mut [f64], Vec<&mut [f64]>)> = betweenness
            .chunks_mut(ADD_BLOCK)
            .map(|totals| {
                // `filter_map` keeps the `Some` values; every buffer has as
                // many blocks as the totals, so nothing is dropped.
                let parts = pieces.iter_mut().filter_map(|piece| piece.next()).collect();
                (totals, parts)
            })
            .collect();
        // `into_par_iter` consumes the Vec and gives each entry to a task.
        blocks.into_par_iter().for_each(|(totals, parts)| {
            for part in parts {
                for (total, value) in totals.iter_mut().zip(part.iter_mut()) {
                    *total += *value;
                    // Zero the buffer for the next batch while it is in the
                    // CPU cache anyway.
                    *value = 0.0;
                }
            }
        });
    }

    // Port of NetworkX's `_rescale` for the unsampled case.
    let big_n = if endpoints { n } else { n.saturating_sub(1) };
    if big_n >= 2 {
        let scale = if normalized {
            1.0 / (big_n * (big_n - 1)) as f64
        } else {
            let correction = if g.is_directed() { 1 } else { 2 };
            big_n as f64 / (big_n * correction) as f64
        };
        if scale != 1.0 {
            for value in betweenness.iter_mut() {
                *value *= scale;
            }
        }
    }
    betweenness
}

// ---------------------------------------------------------------------------
// Closeness
// ---------------------------------------------------------------------------

/// Adds floats the way Python's built-in `sum` does.
///
/// Since Python 3.12, `sum` over floats uses Neumaier's compensated
/// summation: alongside the running total it tracks the rounding error of
/// each addition and adds the accumulated error back at the end. NetworkX
/// computes closeness with `sum(distances)`, so a plain loop here would
/// differ from it in the last digit.
///
/// `impl Iterator<Item = f64>` in argument position accepts any iterator of
/// f64, so the caller does not have to build a Vec first.
fn python_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0f64;
    let mut compensation = 0.0f64;
    for x in values {
        let t = total + x;
        // The rounding error of `total + x` can be recovered exactly, by
        // subtracting in the right order: larger magnitude first.
        if total.abs() >= x.abs() {
            compensation += (total - t) + x;
        } else {
            compensation += (x - t) + total;
        }
        total = t;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// Closeness centrality of one node: the reciprocal of its average distance
/// *from* the other nodes that can reach it.
///
/// - `weight`: `None` counts edges; otherwise `weight(e)` is the weight of
///   edge `e`, which must be non-negative.
/// - `wf_improved`: scale by the fraction of nodes that can reach the node
///   (Wasserman and Faust), which makes values comparable across components.
///
/// Port of the loop body of `networkx.closeness_centrality`.
///
/// `Option<W>` with `W` a closure type: pass `None::<fn(EdgeId) -> f64>` for
/// unweighted, where the `::<...>` tells the compiler what type the absent
/// closure would have had.
pub fn closeness_of<W>(
    g: &Graph,
    weight: Option<W>,
    u: NodeId,
    wf_improved: bool,
    ws: &mut Workspace,
) -> f64
where
    W: Fn(EdgeId) -> f64,
{
    let n = g.node_count();
    // NetworkX reverses a directed graph first: closeness uses distances
    // *to* u. Following edges backwards from u is the same.
    let (reached, total) = match weight {
        None => {
            let tree = bfs(g, u, Direction::Reverse, None, ws);
            let total: f64 = tree.dist.iter().map(|&d| f64::from(d)).sum();
            (tree.order.len(), total)
        }
        Some(weight) => {
            let sp = dijkstra(g, weight, u, Direction::Reverse, None, None, ws);
            (sp.order.len(), python_sum(sp.dist.iter().copied()))
        }
    };
    closeness_score(reached, total, n, wf_improved)
}

/// The closeness of a node that `reached` nodes can reach (itself included)
/// over distances adding up to `total`, in a graph of `n` nodes.
fn closeness_score(reached: usize, total: f64, n: usize, wf_improved: bool) -> f64 {
    let mut closeness = 0.0;
    if total > 0.0 && n > 1 {
        closeness = (reached as f64 - 1.0) / total;
        if wf_improved {
            closeness *= (reached as f64 - 1.0) / (n - 1) as f64;
        }
    }
    closeness
}

/// [`closeness_of`] for each node in `nodes`, in parallel. `weights` is
/// indexed by edge id.
///
/// Unweighted, the searches run 64 at a time (see
/// [`bfs_totals`](crate::shortest_paths::bfs_totals)). Closeness needs only
/// how many nodes each search reached and the sum of the distances, and both
/// are whole numbers, so the result is exactly what one search per node
/// gives. Weighted, each worker thread runs one Dijkstra search at a time
/// with its own [`Workspace`].
pub fn closeness_centrality(
    g: &Graph,
    weights: Option<&[f64]>,
    nodes: &[NodeId],
    wf_improved: bool,
) -> Vec<f64> {
    let n = g.node_count();
    match weights {
        // Closeness uses distances *to* each node, hence `Reverse`.
        None => bfs_totals(g, nodes, Direction::Reverse)
            .iter()
            // The sum is a whole number well below 2^53, the point up to
            // which an f64 holds every whole number exactly.
            .map(|t| closeness_score(t.reached, t.distance_sum as f64, n, wf_improved))
            .collect(),
        Some(w) => nodes
            .par_iter()
            .map_init(Workspace::new, |ws, &u| {
                // A closure over the slice, as `closeness_of` expects.
                let weight = Some(move |e: EdgeId| w[e as usize]);
                closeness_of(g, weight, u, wf_improved, ws)
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts two float slices are equal to within 1e-9.
    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-9, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn pagerank_of_a_cycle_is_uniform() {
        let g = Graph::from_edges(3, &[(0, 1), (1, 2), (2, 0)], true).unwrap();
        let ranks = pagerank(&g, None, &PageRankOptions::default()).unwrap();
        assert_close(&ranks, &[1.0 / 3.0; 3]);
    }

    #[test]
    fn pagerank_reports_non_convergence() {
        let g = Graph::from_edges(3, &[(0, 1), (1, 2)], true).unwrap();
        // Struct update syntax: set one field, default the rest.
        let options = PageRankOptions {
            max_iter: 0,
            ..Default::default()
        };
        assert_eq!(
            pagerank(&g, None, &options),
            Err(NotConverged { iterations: 0 })
        );
    }

    #[test]
    fn degree_of_a_star() {
        let g = Graph::from_edges(4, &[(0, 1), (0, 2), (0, 3)], false).unwrap();
        assert_close(
            &degree_centrality(&g, DegreeKind::Total),
            &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        );
    }

    #[test]
    fn betweenness_of_a_path() {
        // 0 - 1 - 2: only the middle node lies between others.
        let g = Graph::from_edges(3, &[(0, 1), (1, 2)], false).unwrap();
        assert_close(
            &betweenness_centrality(&g, None, true, false),
            &[0.0, 1.0, 0.0],
        );
        assert_close(
            &betweenness_centrality(&g, None, false, false),
            &[0.0, 1.0, 0.0],
        );
        // Unit weights must give the same answer as no weights.
        assert_close(
            &betweenness_centrality(&g, Some(&[1.0, 1.0]), true, false),
            &[0.0, 1.0, 0.0],
        );
    }

    #[test]
    fn python_sum_recovers_lost_digits() {
        // A plain left-to-right sum loses the 1.0 entirely.
        let values = [1e100, 1.0, -1e100];
        assert_eq!(values.iter().sum::<f64>(), 0.0);
        assert_eq!(python_sum(values.iter().copied()), 1.0);
    }

    #[test]
    fn closeness_of_a_path() {
        let g = Graph::from_edges(3, &[(0, 1), (1, 2)], false).unwrap();
        let c = closeness_centrality(&g, None, &[0, 1, 2], true);
        assert_close(&c, &[2.0 / 3.0, 1.0, 2.0 / 3.0]);
        let weighted = closeness_centrality(&g, Some(&[2.0, 2.0]), &[1], true);
        assert_close(&weighted, &[0.5]);
    }
}
