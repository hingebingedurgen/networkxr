//! Centrality measures: PageRank, degree, betweenness and closeness.
//!
//! Ports of `networkx.pagerank` and functions in
//! `networkx.algorithms.centrality`.

use std::collections::VecDeque;
use std::fmt;

use rayon::prelude::*;

use crate::graph::{Direction, EdgeId, Graph, NodeId};
use crate::shortest_paths::{bfs, dijkstra, MinQueue};
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

/// Working memory for one single-source pass of Brandes' algorithm.
///
/// Betweenness runs one shortest-path search per node. Allocating these
/// arrays afresh for each search would dominate the running time on sparse
/// graphs, so each worker thread makes one `Scratch` and reuses it.
struct Scratch {
    /// Number of shortest paths from the source to each node.
    sigma: Vec<f64>,
    /// Predecessors of each node on shortest paths from the source.
    preds: Vec<Vec<NodeId>>,
    /// Nodes in the order their distance became final.
    order: Vec<NodeId>,
    /// Dependency of the source on each node (Brandes' delta).
    delta: Vec<f64>,
    // Unweighted search.
    level: Vec<u32>,
    fifo: VecDeque<NodeId>,
    // Weighted search.
    seen: Vec<f64>,
    done: Vec<bool>,
    queue: MinQueue,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            sigma: vec![0.0; n],
            preds: vec![Vec::new(); n],
            order: Vec::with_capacity(n),
            delta: vec![0.0; n],
            level: vec![u32::MAX; n],
            fifo: VecDeque::new(),
            seen: vec![f64::INFINITY; n],
            done: vec![false; n],
            queue: MinQueue::default(),
        }
    }

    /// Clears everything the previous pass wrote.
    fn reset(&mut self) {
        self.sigma.fill(0.0);
        // `iter_mut()` yields mutable references, so each inner Vec can be
        // cleared in place. `clear` keeps the allocation for reuse.
        for list in self.preds.iter_mut() {
            list.clear();
        }
        self.order.clear();
        self.delta.fill(0.0);
        self.level.fill(u32::MAX);
        self.seen.fill(f64::INFINITY);
        self.done.fill(false);
        self.queue.clear();
    }

    /// Counts shortest paths from `s` by breadth-first search.
    /// Port of `_single_source_shortest_path_basic`.
    fn count_paths_unweighted(&mut self, g: &Graph, s: NodeId) {
        let adj = g.succ();
        self.sigma[s as usize] = 1.0;
        self.level[s as usize] = 0;
        self.fifo.push_back(s);
        while let Some(v) = self.fifo.pop_front() {
            self.order.push(v);
            let level_v = self.level[v as usize];
            let sigma_v = self.sigma[v as usize];
            for &w in adj.neighbors(v) {
                let wi = w as usize;
                if self.level[wi] == u32::MAX {
                    self.fifo.push_back(w);
                    self.level[wi] = level_v + 1;
                }
                if self.level[wi] == level_v + 1 {
                    self.sigma[wi] += sigma_v;
                    self.preds[wi].push(v);
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
    /// node's betweenness. Ports `_accumulate_basic` and
    /// `_accumulate_endpoints`.
    fn accumulate(&mut self, s: NodeId, endpoints: bool, n: usize) -> Vec<f64> {
        let mut contribution = vec![0.0; n];
        if endpoints {
            contribution[s as usize] = (self.order.len() - 1) as f64;
        }
        // Visit nodes farthest first.
        while let Some(w) = self.order.pop() {
            let wi = w as usize;
            let coeff = (1.0 + self.delta[wi]) / self.sigma[wi];
            // `self.preds[wi]` and `self.delta` are different fields, so the
            // compiler allows reading one while writing the other.
            for &v in &self.preds[wi] {
                self.delta[v as usize] += self.sigma[v as usize] * coeff;
            }
            if w != s {
                contribution[wi] = if endpoints {
                    self.delta[wi] + 1.0
                } else {
                    self.delta[wi]
                };
            }
        }
        contribution
    }
}

/// Betweenness centrality of every node: the share of shortest paths that
/// pass through it.
///
/// - `weights`: `None` counts edges; otherwise paths are weighted.
/// - `normalized`: scale by the number of node pairs.
/// - `endpoints`: count a path's endpoints as lying on it.
///
/// Port of `networkx.betweenness_centrality` without sampling (`k=None`).
///
/// The searches run in parallel, a batch of sources at a time. Each source's
/// contribution is then added in source order, the same order NetworkX adds
/// them, so the floating-point result is identical to NetworkX's.
pub fn betweenness_centrality(
    g: &Graph,
    weights: Option<&[f64]>,
    normalized: bool,
    endpoints: bool,
) -> Vec<f64> {
    let n = g.node_count();
    let mut betweenness = vec![0.0; n];
    let sources: Vec<NodeId> = g.nodes().collect();

    // Each contribution is a Vec of n floats, so cap how many are alive at
    // once at about 32 MB.
    let batch = (4_000_000 / n.max(1)).clamp(1, 1024);

    // `chunks` yields consecutive sub-slices of at most `batch` elements.
    for chunk in sources.chunks(batch) {
        let contributions: Vec<Vec<f64>> = chunk
            .par_iter()
            // `map_init` is `map` with per-thread state: the first closure
            // builds one Scratch per worker thread, and the second receives
            // it mutably for every item that thread handles.
            .map_init(
                || Scratch::new(n),
                |scratch, &s| {
                    scratch.reset();
                    match weights {
                        None => scratch.count_paths_unweighted(g, s),
                        Some(w) => scratch.count_paths_weighted(g, w, s),
                    }
                    scratch.accumulate(s, endpoints, n)
                },
            )
            .collect();
        for contribution in &contributions {
            // `zip` walks two iterators together; `iter_mut` on the left
            // gives mutable references so we can add in place.
            for (total, c) in betweenness.iter_mut().zip(contribution) {
                *total += c;
            }
        }
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
    let mut closeness = 0.0;
    if total > 0.0 && n > 1 {
        closeness = (reached as f64 - 1.0) / total;
        if wf_improved {
            closeness *= (reached as f64 - 1.0) / (n - 1) as f64;
        }
    }
    closeness
}

/// [`closeness_of`] for each node in `nodes`, in parallel, each worker
/// thread with its own [`Workspace`]. `weights` is indexed by edge id.
pub fn closeness_centrality(
    g: &Graph,
    weights: Option<&[f64]>,
    nodes: &[NodeId],
    wf_improved: bool,
) -> Vec<f64> {
    nodes
        .par_iter()
        .map_init(Workspace::new, |ws, &u| {
            // Turn the optional slice into an optional closure over it.
            let weight = weights.map(|w| move |e: EdgeId| w[e as usize]);
            closeness_of(g, weight, u, wf_improved, ws)
        })
        .collect()
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
