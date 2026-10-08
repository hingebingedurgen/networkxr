//! Spanning trees and tree recognition.
//!
//! Ports of `networkx.minimum_spanning_edges` (Kruskal's algorithm),
//! `networkx.is_tree` and `networkx.is_forest`.

use crate::components::{connected_components, is_connected};
use crate::graph::{EdgeId, Graph, NodeId};

/// A disjoint-set (union-find) structure over the integers `0..n`.
///
/// It tracks a partition of the nodes into groups and supports two fast
/// operations: find which group a node is in, and merge two groups. Kruskal's
/// algorithm uses it to ask "would this edge join two different trees?".
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// `parent[x]` is another member of x's group, closer to the group's
    /// representative. The representative is its own parent.
    parent: Vec<NodeId>,
    /// For a representative, the number of nodes in its group.
    size: Vec<u32>,
}

impl UnionFind {
    /// `n` groups of one node each.
    ///
    /// By convention the constructor is an associated function called `new`.
    /// It has no `self` parameter and is called as `UnionFind::new(n)`.
    /// `Self` is shorthand for the type being implemented.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as NodeId).collect(),
            size: vec![1; n],
        }
    }

    /// The representative of the group containing `x`.
    ///
    /// `&mut self` because looking something up also flattens the structure
    /// ("path compression"): every node visited is re-pointed straight at
    /// the representative, so the next lookup is one step.
    pub fn find(&mut self, x: NodeId) -> NodeId {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // Second pass: point everything on the path at the root.
        let mut node = x;
        while self.parent[node as usize] != root {
            // `std::mem::replace` stores a new value and returns the old one.
            node = std::mem::replace(&mut self.parent[node as usize], root);
        }
        root
    }

    /// Merges the groups of `a` and `b`. Returns `false` if they were
    /// already in the same group.
    pub fn union(&mut self, a: NodeId, b: NodeId) -> bool {
        let (mut big, mut small) = (self.find(a), self.find(b));
        if big == small {
            return false;
        }
        // Hang the smaller group under the larger one. This keeps the
        // parent chains short.
        if self.size[big as usize] < self.size[small as usize] {
            // `swap` exchanges two values in place.
            std::mem::swap(&mut big, &mut small);
        }
        self.parent[small as usize] = big;
        self.size[big as usize] += self.size[small as usize];
        true
    }
}

/// The edges of a minimum (or, with `minimum == false`, maximum) spanning
/// forest, as edge ids in the order Kruskal's algorithm accepts them.
///
/// Port of `networkx`'s `kruskal_mst_edges`. Weights must not be NaN.
pub fn kruskal(g: &Graph, weights: &[f64], minimum: bool) -> Vec<EdgeId> {
    let mut edges: Vec<EdgeId> = (0..g.edge_count() as EdgeId).collect();
    // `sort_by` takes a closure that compares two elements. It is a *stable*
    // sort: equal elements keep their original order, as with Python's
    // `sorted`. NetworkX relies on that to break ties by edge order, and for
    // a maximum tree it sorts with `reverse=True`, which also keeps equal
    // elements in their original order. Comparing b to a does the same.
    //
    // `expect` is `unwrap` with a message for the panic.
    let weight = |e: &EdgeId| weights[*e as usize];
    if minimum {
        edges.sort_by(|a, b| weight(a).partial_cmp(&weight(b)).expect("NaN edge weight"));
    } else {
        edges.sort_by(|a, b| weight(b).partial_cmp(&weight(a)).expect("NaN edge weight"));
    }

    let mut groups = UnionFind::new(g.node_count());
    let mut chosen = Vec::new();
    for e in edges {
        let (u, v) = g.endpoints(e);
        if groups.union(u, v) {
            chosen.push(e);
        }
    }
    chosen
}

/// True if every component is a tree. Direction is ignored when finding
/// components. The graph must have at least one node.
///
/// Port of `networkx.is_forest`. A component with `k` nodes has at least
/// `k - 1` edges, and exactly that many if it is a tree, so the graph is a
/// forest exactly when edges = nodes - components.
pub fn is_forest(g: &Graph) -> bool {
    g.edge_count() + connected_components(g).len() == g.node_count()
}

/// True if the graph is connected (ignoring direction) and has one edge
/// fewer than it has nodes. The graph must have at least one node.
///
/// Port of `networkx.is_tree`.
pub fn is_tree(g: &Graph) -> bool {
    // `&&` short-circuits: the search only runs if the count matches.
    g.edge_count() + 1 == g.node_count() && is_connected(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find_merges_groups() {
        let mut uf = UnionFind::new(4);
        assert!(uf.union(0, 1));
        assert!(uf.union(2, 3));
        assert_ne!(uf.find(0), uf.find(2));
        assert!(uf.union(1, 3));
        assert_eq!(uf.find(0), uf.find(2));
        assert!(!uf.union(0, 3));
    }

    #[test]
    fn kruskal_minimum_and_maximum() {
        // Triangle 0-1-2 with weights 1, 2, 3 and a pendant edge 2-3.
        let g = Graph::from_edges(4, &[(0, 1), (1, 2), (0, 2), (2, 3)], false).unwrap();
        // G.edges order is (0,1) (0,2) (1,2) (2,3).
        let weights = [1.0, 3.0, 2.0, 1.0];
        assert_eq!(kruskal(&g, &weights, true), vec![0, 3, 2]);
        assert_eq!(kruskal(&g, &weights, false), vec![1, 2, 3]);
    }

    #[test]
    fn tree_and_forest() {
        let path = Graph::from_edges(3, &[(0, 1), (1, 2)], false).unwrap();
        assert!(is_tree(&path) && is_forest(&path));
        let two_trees = Graph::from_edges(4, &[(0, 1), (2, 3)], false).unwrap();
        assert!(!is_tree(&two_trees) && is_forest(&two_trees));
        let triangle = Graph::from_edges(3, &[(0, 1), (1, 2), (0, 2)], false).unwrap();
        assert!(!is_tree(&triangle) && !is_forest(&triangle));
    }
}
