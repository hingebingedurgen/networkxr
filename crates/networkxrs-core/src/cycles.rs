//! Cycles in undirected graphs.
//!
//! Port of `networkx.cycle_basis`.

use std::collections::HashSet;

use crate::graph::{Graph, NodeId, NO_NODE};

/// A list of cycles that forms a basis for all cycles of an undirected
/// graph: every cycle is a combination of these. Each cycle is a list of
/// nodes.
///
/// `root` picks the node the search starts from. With `None`, NetworkX
/// starts from the last node in node order, and so does this.
///
/// Port of `networkx.cycle_basis`, kept line for line so that the cycles,
/// their order and their starting points match.
pub fn cycle_basis(g: &Graph, root: Option<NodeId>) -> Vec<Vec<NodeId>> {
    let adj = g.succ();
    let n = g.node_count();
    let mut cycles = Vec::new();

    // NetworkX keeps the unvisited nodes in a dict and calls `popitem()`,
    // which removes the most recently inserted key: the last node. Nodes are
    // only ever removed, so "the last remaining node" can be found by
    // scanning downwards from where the previous scan stopped.
    let mut remaining = vec![true; n];
    let mut scan_from = n;
    let mut next_root = root;

    // pred[v] is v's parent in the search tree. in_tree[v] says whether v
    // has been reached, where NetworkX tests `nbr not in used`.
    let mut pred = vec![NO_NODE; n];
    let mut in_tree = vec![false; n];
    // used[v] is the set of neighbours of v whose edge to v is accounted for.
    // `HashSet<T>` is a hash set, like Python's `set`. An empty one costs no
    // allocation, so a Vec of n of them is cheap.
    let mut used: Vec<HashSet<NodeId>> = vec![HashSet::new(); n];

    loop {
        // `Option::take` returns the value and leaves `None` in its place,
        // so the caller's root is used once.
        let root = match next_root.take() {
            Some(r) => r,
            None => {
                while scan_from > 0 && !remaining[scan_from - 1] {
                    scan_from -= 1;
                }
                if scan_from == 0 {
                    break;
                }
                (scan_from - 1) as NodeId
            }
        };

        let mut stack = vec![root];
        let mut tree_nodes = vec![root];
        pred[root as usize] = root;
        in_tree[root as usize] = true;

        while let Some(z) = stack.pop() {
            for &nbr in adj.neighbors(z) {
                if !in_tree[nbr as usize] {
                    // A tree edge: first time we reach nbr.
                    pred[nbr as usize] = z;
                    in_tree[nbr as usize] = true;
                    tree_nodes.push(nbr);
                    stack.push(nbr);
                    used[nbr as usize].insert(z);
                } else if nbr == z {
                    // A self-loop is a cycle on its own.
                    cycles.push(vec![z]);
                } else if !used[z as usize].contains(&nbr) {
                    // A non-tree edge closes a cycle: walk up from z until
                    // we meet a node already linked to nbr.
                    let mut cycle = vec![nbr, z];
                    let mut p = pred[z as usize];
                    while !used[nbr as usize].contains(&p) {
                        cycle.push(p);
                        p = pred[p as usize];
                    }
                    cycle.push(p);
                    cycles.push(cycle);
                    used[nbr as usize].insert(z);
                }
            }
        }

        for &v in &tree_nodes {
            remaining[v as usize] = false;
        }
    }
    cycles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_with_tail_has_one_cycle() {
        let g = Graph::from_edges(4, &[(0, 1), (1, 2), (2, 0), (2, 3)], false).unwrap();
        let cycles = cycle_basis(&g, Some(0));
        assert_eq!(cycles.len(), 1);
        let mut nodes = cycles[0].clone();
        nodes.sort_unstable();
        assert_eq!(nodes, vec![0, 1, 2]);
    }

    #[test]
    fn tree_has_no_cycles_and_self_loop_is_one() {
        let tree = Graph::from_edges(3, &[(0, 1), (1, 2)], false).unwrap();
        assert!(cycle_basis(&tree, None).is_empty());
        let looped = Graph::from_edges(2, &[(0, 1), (1, 1)], false).unwrap();
        assert_eq!(cycle_basis(&looped, None), vec![vec![1]]);
    }
}
