//! Directed acyclic graphs: topological order and longest paths.
//!
//! Ports of functions in `networkx.algorithms.dag`, plus
//! [`dag_longest_paths`], which NetworkX does not have.

use std::fmt;

use crate::graph::{EdgeId, Graph, NodeId, NO_NODE};

/// The graph has a cycle, so no topological order exists.
///
/// An error type can carry data. NetworkX yields the generations it can
/// compute and only then raises; `partial` holds those generations so the
/// Python wrapper can do the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleError {
    pub partial: Vec<Vec<NodeId>>,
}

impl fmt::Display for CycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "graph contains a cycle")
    }
}

impl std::error::Error for CycleError {}

/// Groups the nodes of a directed acyclic graph into generations: a node is
/// in generation `k` if its longest chain of ancestors has `k` nodes.
///
/// Port of `networkx.topological_generations` (Kahn's algorithm).
pub fn topological_generations(g: &Graph) -> Result<Vec<Vec<NodeId>>, CycleError> {
    let n = g.node_count();

    // `.map(...)` transforms each item of an iterator, `.collect()` gathers
    // the results into a container. The annotation on the variable tells
    // `collect` which container to build.
    let mut indegree: Vec<usize> = g.nodes().map(|v| g.pred().degree(v)).collect();
    // `.filter(...)` keeps the items for which the closure returns true.
    let mut zero_indegree: Vec<NodeId> = g.nodes().filter(|&v| indegree[v as usize] == 0).collect();

    let mut generations = Vec::new();
    let mut placed = 0;
    while !zero_indegree.is_empty() {
        let this_generation = std::mem::take(&mut zero_indegree);
        for &node in &this_generation {
            for &child in g.succ().neighbors(node) {
                indegree[child as usize] -= 1;
                if indegree[child as usize] == 0 {
                    zero_indegree.push(child);
                }
            }
        }
        placed += this_generation.len();
        generations.push(this_generation);
    }

    // A node on a cycle never reaches in-degree zero, so it is never placed.
    if placed == n {
        Ok(generations)
    } else {
        Err(CycleError {
            partial: generations,
        })
    }
}

/// The nodes of a directed acyclic graph, each before all its successors.
///
/// Port of `networkx.topological_sort`.
pub fn topological_sort(g: &Graph) -> Result<Vec<NodeId>, CycleError> {
    // `?` passes the error on. `into_iter()` consumes the Vec and yields
    // its elements by value; `flatten()` concatenates the inner Vecs.
    Ok(topological_generations(g)?.into_iter().flatten().collect())
}

/// True if a directed graph has no cycle.
pub fn is_acyclic(g: &Graph) -> bool {
    topological_generations(g).is_ok()
}

/// One longest path in a directed acyclic graph, by total edge weight.
///
/// Port of `networkx.dag_longest_path`. `weights[e]` is the weight of edge
/// `e`. Ties are broken as NetworkX breaks them, so this returns the same
/// path. For all the longest paths, see [`dag_longest_paths`].
pub fn dag_longest_path(g: &Graph, weights: &[f64]) -> Result<Vec<NodeId>, CycleError> {
    let n = g.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }
    let order = topological_sort(g)?;

    // best[v] is the weight of the heaviest path ending at v, and from[v] is
    // the node before v on it (v itself if the path starts at v).
    let mut best = vec![0.0f64; n];
    let mut from = vec![NO_NODE; n];
    for &v in &order {
        // The heaviest way to arrive at v. Python's `max` keeps the first of
        // several equal maxima, so a later candidate must be strictly
        // greater to replace the current one.
        let mut arrive: Option<(f64, NodeId)> = None;
        for (u, e) in g.pred().arcs(v) {
            let candidate = best[u as usize] + weights[e as usize];
            // `is_none_or(f)`: true for None, `f(value)` for Some.
            if arrive.is_none_or(|(w, _)| candidate > w) {
                arrive = Some((candidate, u));
            }
        }
        // A `match` arm can have a guard (`if ...`). `_` matches the rest:
        // no predecessors, or only paths of negative weight, in which case
        // starting fresh at v is better.
        match arrive {
            Some((w, u)) if w >= 0.0 => {
                best[v as usize] = w;
                from[v as usize] = u;
            }
            _ => {
                best[v as usize] = 0.0;
                from[v as usize] = v;
            }
        }
    }

    // The end of the longest path: first maximum in topological order.
    let mut end = order[0];
    for &v in &order {
        if best[v as usize] > best[end as usize] {
            end = v;
        }
    }

    // Walk back to the start, then reverse.
    let mut path = Vec::new();
    let mut v = end;
    loop {
        path.push(v);
        let u = from[v as usize];
        if u == v {
            break;
        }
        v = u;
    }
    path.reverse();
    Ok(path)
}

/// Why [`dag_longest_paths`] could not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LongestPathsError {
    /// The graph has a cycle.
    Cycle,
    /// This edge has a negative weight.
    NegativeWeight(EdgeId),
}

impl fmt::Display for LongestPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LongestPathsError::Cycle => write!(f, "graph contains a cycle"),
            LongestPathsError::NegativeWeight(e) => write!(f, "edge {e} has a negative weight"),
        }
    }
}

impl std::error::Error for LongestPathsError {}

/// *All* longest paths in a directed acyclic graph with non-negative weights.
///
/// The method: negate every weight, so that the longest path becomes the
/// shortest one; compute shortest distances; then return every path whose
/// length equals the best distance found.
///
/// With negated weights the graph has negative edges, which rules out
/// Dijkstra. On a DAG that does not matter: relaxing the edges once, in
/// topological order, gives exact shortest distances.
///
/// A path may start at any node, so each node starts at distance 0, as if a
/// hidden source were joined to every node by a zero-weight edge.
///
/// A path is returned if its total weight equals the maximum. With
/// zero-weight edges this includes paths that are extensions of one another,
/// since all of them have the maximum weight. Weights are compared exactly;
/// with floating-point weights, two paths whose sums differ only by rounding
/// are not considered equal.
///
/// The number of longest paths can grow exponentially with the size of the
/// graph. `max_paths` stops the search after that many paths.
///
/// Paths are ordered by end node, then by the predecessor order of each
/// node from the end of the path backwards.
pub fn dag_longest_paths(
    g: &Graph,
    weights: &[f64],
    max_paths: Option<usize>,
) -> Result<Vec<Vec<NodeId>>, LongestPathsError> {
    let n = g.node_count();
    // `position` returns the index of the first item matching the closure.
    if let Some(e) = weights.iter().position(|&w| w < 0.0) {
        return Err(LongestPathsError::NegativeWeight(e as EdgeId));
    }
    // `map_err` converts one error type into another before `?` returns it.
    let order = topological_sort(g).map_err(|_| LongestPathsError::Cycle)?;
    let limit = max_paths.unwrap_or(usize::MAX);
    if n == 0 || limit == 0 {
        return Ok(Vec::new());
    }

    // Step 1: negate the weights.
    let negated: Vec<f64> = weights.iter().map(|&w| -w).collect();

    // Step 2: shortest distances, every node starting at 0.
    let mut dist = vec![0.0f64; n];
    for &v in &order {
        for (u, e) in g.pred().arcs(v) {
            let through_u = dist[u as usize] + negated[e as usize];
            if through_u < dist[v as usize] {
                dist[v as usize] = through_u;
            }
        }
    }

    // The most negative distance is minus the longest path weight.
    // `fold` reduces an iterator to one value, starting from the first
    // argument. (f64 has no `.min()` over iterators because NaN has no
    // order, so we fold with the two-argument `f64::min`.)
    let shortest = dist.iter().copied().fold(f64::INFINITY, f64::min);

    // Step 3: collect every path of that length by walking backwards from
    // each node at the best distance, following only "tight" edges, the
    // ones that achieve the distance of the node they lead to.
    let is_tight = |u: NodeId, e: EdgeId, v: NodeId| {
        dist[u as usize] + negated[e as usize] == dist[v as usize]
    };

    let mut paths = Vec::new();
    // The path being built, end node first, and for each node on it how
    // many of its incoming edges have been tried. Explicit stacks again,
    // because a path can be as long as the graph.
    let mut path: Vec<NodeId> = Vec::new();
    let mut tried: Vec<usize> = Vec::new();

    for end in g.nodes().filter(|&v| dist[v as usize] == shortest) {
        path.push(end);
        tried.push(0);
        // A node at distance 0 is where a path can start, so the current
        // stack is a complete path. `iter().rev()` walks it start to end.
        if dist[end as usize] == 0.0 {
            paths.push(path.iter().rev().copied().collect());
            if paths.len() == limit {
                return Ok(paths);
            }
        }
        while let Some(&v) = path.last() {
            let incoming = g.pred().neighbors(v);
            let incoming_edges = g.pred().edge_ids(v);
            // `tried.last_mut().unwrap()`: `unwrap` takes the value out of
            // an Option and panics if it is None. `tried` always has the
            // same length as `path`, which we just saw is non-empty.
            let next = tried.last_mut().unwrap();
            let mut step = None;
            while *next < incoming.len() {
                let (u, e) = (incoming[*next], incoming_edges[*next]);
                *next += 1;
                if is_tight(u, e, v) {
                    step = Some(u);
                    break;
                }
            }
            match step {
                Some(u) => {
                    path.push(u);
                    tried.push(0);
                    if dist[u as usize] == 0.0 {
                        paths.push(path.iter().rev().copied().collect());
                        if paths.len() == limit {
                            return Ok(paths);
                        }
                    }
                }
                None => {
                    path.pop();
                    tried.pop();
                }
            }
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generations_and_sort() {
        // 0 -> 1 -> 3, 0 -> 2 -> 3
        let g = Graph::from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], true).unwrap();
        assert_eq!(
            topological_generations(&g).unwrap(),
            vec![vec![0], vec![1, 2], vec![3]]
        );
        assert_eq!(topological_sort(&g).unwrap(), vec![0, 1, 2, 3]);
        assert!(is_acyclic(&g));
    }

    #[test]
    fn cycle_is_an_error_carrying_what_was_sorted() {
        // 0 -> 1, then 1 <-> 2
        let g = Graph::from_edges(3, &[(0, 1), (1, 2), (2, 1)], true).unwrap();
        let err = topological_generations(&g).unwrap_err();
        assert_eq!(err.partial, vec![vec![0]]);
        assert!(!is_acyclic(&g));
    }

    #[test]
    fn longest_path_single() {
        let g = Graph::from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], true).unwrap();
        // Edge ids follow G.edges order: (0,1) (0,2) (1,3) (2,3).
        assert_eq!(
            dag_longest_path(&g, &[1.0, 5.0, 1.0, 1.0]).unwrap(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn longest_paths_returns_every_tie() {
        // Two routes of weight 2 from 0 to 3.
        let g = Graph::from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], true).unwrap();
        let paths = dag_longest_paths(&g, &[1.0; 4], None).unwrap();
        assert_eq!(paths, vec![vec![0, 1, 3], vec![0, 2, 3]]);
        // The limit stops early.
        assert_eq!(dag_longest_paths(&g, &[1.0; 4], Some(1)).unwrap().len(), 1);
    }

    #[test]
    fn longest_paths_with_zero_weight_edge_includes_both_extensions() {
        // 0 -(0)-> 1 -(5)-> 2. Both [1, 2] and [0, 1, 2] weigh 5.
        let g = Graph::from_edges(3, &[(0, 1), (1, 2)], true).unwrap();
        let paths = dag_longest_paths(&g, &[0.0, 5.0], None).unwrap();
        assert_eq!(paths, vec![vec![1, 2], vec![0, 1, 2]]);
    }

    #[test]
    fn longest_paths_without_edges_is_every_single_node() {
        let g = Graph::from_edges(2, &[], true).unwrap();
        assert_eq!(
            dag_longest_paths(&g, &[], None).unwrap(),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn longest_paths_rejects_bad_input() {
        let g = Graph::from_edges(2, &[(0, 1)], true).unwrap();
        assert_eq!(
            dag_longest_paths(&g, &[-1.0], None),
            Err(LongestPathsError::NegativeWeight(0))
        );
        let cyclic = Graph::from_edges(2, &[(0, 1), (1, 0)], true).unwrap();
        assert_eq!(
            dag_longest_paths(&cyclic, &[1.0, 1.0], None),
            Err(LongestPathsError::Cycle)
        );
    }
}
