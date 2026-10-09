//! Breadth-first and depth-first search.
//!
//! Ports of `networkx.algorithms.traversal`. NetworkX writes these as
//! generators that yield one edge at a time. Here they either return a `Vec`
//! of everything, or call a function you pass in for each event.
//!
//! Each function borrows a [`Workspace`] for its visited marks, so a search
//! that reaches few nodes is cheap however large the graph is. See the
//! [`workspace`](crate::workspace) module for why.

use crate::graph::{Direction, Graph, NodeId};
use crate::workspace::{Workspace, UNSEEN};

/// The value written to a node's slot to mark it visited.
const SEEN: u32 = 0;

/// The edges of a breadth-first search from `source`, as `(parent, child)`.
///
/// Port of `networkx.generic_bfs_edges`. `depth_limit` of `None` means no
/// limit.
///
/// `ws: &mut Workspace` is a mutable borrow: this function may modify the
/// workspace, and while it runs nobody else can use it.
pub fn bfs_edges(
    g: &Graph,
    source: NodeId,
    dir: Direction,
    depth_limit: Option<usize>,
    ws: &mut Workspace,
) -> Vec<(NodeId, NodeId)> {
    let adj = g.adjacency(dir);
    let n = g.node_count();
    // `unwrap_or` gives the value inside `Some`, or the fallback for `None`.
    let depth_limit = depth_limit.unwrap_or(n);

    // NetworkX tracks visited nodes in a Python set. Because our nodes are
    // the integers 0..n, an array indexed by node does the same job with no
    // hashing.
    ws.prepare(n);
    ws.slot[source as usize] = SEEN;
    ws.touch(source);
    let mut seen_count = 1;

    let mut edges = Vec::new();
    let mut next_level = vec![source];
    let mut depth = 0;

    // `'search:` labels the loop so that the inner loop can leave it.
    'search: while !next_level.is_empty() && depth < depth_limit {
        // `std::mem::take` moves the Vec out of `next_level` and leaves an
        // empty Vec behind. It is how you say "this level = next level;
        // next level = []" without copying.
        let this_level = std::mem::take(&mut next_level);
        // `&this_level` iterates by reference; `&parent` in the pattern
        // copies each element out of its reference.
        for &parent in &this_level {
            for &child in adj.neighbors(parent) {
                if ws.slot[child as usize] == UNSEEN {
                    ws.slot[child as usize] = SEEN;
                    ws.touch(child);
                    seen_count += 1;
                    next_level.push(child);
                    edges.push((parent, child));
                }
            }
            if seen_count == n {
                break 'search;
            }
        }
        depth += 1;
    }
    ws.release();
    edges
}

/// The nodes reachable from `source`, grouped by distance. Layer 0 is
/// `[source]`.
///
/// Port of `networkx.bfs_layers` for a single source.
pub fn bfs_layers(g: &Graph, source: NodeId, ws: &mut Workspace) -> Vec<Vec<NodeId>> {
    let adj = g.succ();
    ws.prepare(g.node_count());
    ws.slot[source as usize] = SEEN;
    ws.touch(source);
    let mut layers = Vec::new();
    let mut current = vec![source];
    while !current.is_empty() {
        let mut next = Vec::new();
        for &node in &current {
            for &child in adj.neighbors(node) {
                if ws.slot[child as usize] == UNSEEN {
                    ws.slot[child as usize] = SEEN;
                    ws.touch(child);
                    next.push(child);
                }
            }
        }
        // `current` is moved into `layers` here; `next` takes its place.
        layers.push(current);
        current = next;
    }
    ws.release();
    layers
}

/// What a depth-first search is doing with an edge.
///
/// These mirror the string labels of `networkx.dfs_labeled_edges`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfsLabel {
    /// The search moves to an unvisited node. Also reported once as
    /// `(root, root)` when a new search tree starts.
    Forward,
    /// The edge leads to a node that was already visited.
    Nontree,
    /// The search has finished a node and steps back to its parent. Also
    /// reported once as `(root, root)` when a search tree is finished.
    Reverse,
    /// The search reached an unvisited node but will not explore it because
    /// of the depth limit.
    ReverseDepthLimit,
}

/// Runs a depth-first search and calls `visit(u, v, label)` for every event.
///
/// Port of `networkx.dfs_labeled_edges`. If `source` is `None`, the search
/// restarts from each unvisited node in node order until every node has
/// been visited.
///
/// `F: FnMut(...)` accepts any closure with this signature. `FnMut` (rather
/// than `Fn`) allows the closure to modify variables it captured, such as
/// pushing onto a Vec owned by the caller. See [`dfs_edges`] for an example.
pub fn dfs_labeled_edges<F>(
    g: &Graph,
    source: Option<NodeId>,
    depth_limit: Option<usize>,
    ws: &mut Workspace,
    mut visit: F,
) where
    F: FnMut(NodeId, NodeId, DfsLabel),
{
    let adj = g.succ();
    let n = g.node_count();
    let depth_limit = depth_limit.unwrap_or(n);
    ws.prepare(n);

    // Either the single source, or every node. Both arms of the `match` must
    // have the same type, so both are ranges.
    let starts = match source {
        Some(s) => s..s + 1,
        None => g.nodes(),
    };

    // A recursive DFS would overflow the call stack on a long path graph, so
    // the stack is explicit. Each entry is a node and how many of its
    // neighbours have been looked at so far. NetworkX keeps a Python
    // iterator in that second slot; a position into the neighbour slice is
    // the same thing.
    let mut stack: Vec<(NodeId, usize)> = Vec::new();

    for start in starts {
        if ws.slot[start as usize] != UNSEEN {
            continue;
        }
        visit(start, start, DfsLabel::Forward);
        ws.slot[start as usize] = SEEN;
        ws.touch(start);
        stack.push((start, 0));
        let mut depth_now = 1;

        // `while let` loops for as long as the pattern matches.
        // `last_mut()` gives `Some(mutable reference to the top entry)` or
        // `None` when the stack is empty.
        while let Some(top) = stack.last_mut() {
            let parent = top.0;
            let neighbors = adj.neighbors(parent);
            // Set when we find an unvisited child to descend into.
            let mut descend_into = None;

            while top.1 < neighbors.len() {
                let child = neighbors[top.1];
                top.1 += 1;
                if ws.slot[child as usize] != UNSEEN {
                    visit(parent, child, DfsLabel::Nontree);
                } else {
                    visit(parent, child, DfsLabel::Forward);
                    ws.slot[child as usize] = SEEN;
                    ws.touch(child);
                    if depth_now < depth_limit {
                        descend_into = Some(child);
                        break;
                    }
                    visit(parent, child, DfsLabel::ReverseDepthLimit);
                }
            }

            // The mutable borrow `top` ends above, which is what allows us
            // to push to and pop from `stack` here. Two live mutable borrows
            // of the same Vec would not compile.
            match descend_into {
                Some(child) => {
                    stack.push((child, 0));
                    depth_now += 1;
                }
                None => {
                    stack.pop();
                    depth_now -= 1;
                    // `if let` runs the block only if the pattern matches.
                    if let Some(&(grandparent, _)) = stack.last() {
                        visit(grandparent, parent, DfsLabel::Reverse);
                    }
                }
            }
        }
        visit(start, start, DfsLabel::Reverse);
    }
    ws.release();
}

/// The tree edges of a depth-first search, as `(parent, child)`.
///
/// Port of `networkx.dfs_edges`.
pub fn dfs_edges(
    g: &Graph,
    source: Option<NodeId>,
    depth_limit: Option<usize>,
    ws: &mut Workspace,
) -> Vec<(NodeId, NodeId)> {
    let mut edges = Vec::new();
    // The closure `|u, v, label| { ... }` captures `edges` by mutable
    // reference and pushes to it each time the search calls it.
    dfs_labeled_edges(g, source, depth_limit, ws, |u, v, label| {
        // The `(root, root)` event has u == v. A self-loop never produces a
        // Forward event, because its target is already visited.
        if label == DfsLabel::Forward && u != v {
            edges.push((u, v));
        }
    });
    edges
}

/// Nodes in the order a depth-first search first reaches them.
///
/// Port of `networkx.dfs_preorder_nodes`.
pub fn dfs_preorder(
    g: &Graph,
    source: Option<NodeId>,
    depth_limit: Option<usize>,
    ws: &mut Workspace,
) -> Vec<NodeId> {
    let mut nodes = Vec::new();
    dfs_labeled_edges(g, source, depth_limit, ws, |_, v, label| {
        if label == DfsLabel::Forward {
            nodes.push(v);
        }
    });
    nodes
}

/// Nodes in the order a depth-first search finishes them.
///
/// Port of `networkx.dfs_postorder_nodes`. As in NetworkX, a node cut off by
/// the depth limit is not reported.
pub fn dfs_postorder(
    g: &Graph,
    source: Option<NodeId>,
    depth_limit: Option<usize>,
    ws: &mut Workspace,
) -> Vec<NodeId> {
    let mut nodes = Vec::new();
    dfs_labeled_edges(g, source, depth_limit, ws, |_, v, label| {
        if label == DfsLabel::Reverse {
            nodes.push(v);
        }
    });
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0 - 1 - 3
    /// |   |
    /// 2 - 4
    fn small() -> Graph {
        Graph::from_edges(5, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 4)], false).unwrap()
    }

    #[test]
    fn bfs_visits_level_by_level() {
        let g = small();
        let mut ws = Workspace::new();
        assert_eq!(
            bfs_edges(&g, 0, Direction::Forward, None, &mut ws),
            vec![(0, 1), (0, 2), (1, 3), (1, 4)]
        );
        assert_eq!(
            bfs_edges(&g, 0, Direction::Forward, Some(1), &mut ws),
            vec![(0, 1), (0, 2)]
        );
        assert_eq!(
            bfs_layers(&g, 0, &mut ws),
            vec![vec![0], vec![1, 2], vec![3, 4]]
        );
        assert!(ws.is_clean());
    }

    #[test]
    fn bfs_reverse_follows_predecessors() {
        let g = Graph::from_edges(3, &[(0, 1), (1, 2)], true).unwrap();
        let mut ws = Workspace::new();
        assert_eq!(bfs_edges(&g, 2, Direction::Forward, None, &mut ws), vec![]);
        assert_eq!(
            bfs_edges(&g, 2, Direction::Reverse, None, &mut ws),
            vec![(2, 1), (1, 0)]
        );
        assert!(ws.is_clean());
    }

    #[test]
    fn dfs_orders() {
        let g = small();
        let mut ws = Workspace::new();
        assert_eq!(
            dfs_edges(&g, Some(0), None, &mut ws),
            vec![(0, 1), (1, 3), (1, 4), (4, 2)]
        );
        assert_eq!(
            dfs_preorder(&g, Some(0), None, &mut ws),
            vec![0, 1, 3, 4, 2]
        );
        assert_eq!(
            dfs_postorder(&g, Some(0), None, &mut ws),
            vec![3, 2, 4, 1, 0]
        );
        assert!(ws.is_clean());
    }

    #[test]
    fn dfs_without_source_covers_every_component() {
        let g = Graph::from_edges(4, &[(0, 1), (2, 3)], false).unwrap();
        let mut ws = Workspace::new();
        assert_eq!(dfs_edges(&g, None, None, &mut ws), vec![(0, 1), (2, 3)]);
        assert_eq!(dfs_preorder(&g, None, None, &mut ws), vec![0, 1, 2, 3]);
        assert!(ws.is_clean());
    }

    #[test]
    fn dfs_depth_limit_cuts_off_and_skips_postorder() {
        let g = Graph::from_edges(3, &[(0, 1), (1, 2)], false).unwrap();
        let mut ws = Workspace::new();
        assert_eq!(dfs_edges(&g, Some(0), Some(1), &mut ws), vec![(0, 1)]);
        // Node 1 was reached at the limit, so it is never "finished".
        assert_eq!(dfs_postorder(&g, Some(0), Some(1), &mut ws), vec![0]);
        assert!(ws.is_clean());
    }
}
