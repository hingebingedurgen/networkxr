//! Connected, weakly connected and strongly connected components.
//!
//! Ports of `networkx.algorithms.components`.

use crate::graph::{Graph, NodeId};
use crate::workspace::{Workspace, UNSEEN};

/// The value written to a node's slot to mark it visited.
const SEEN: u32 = 0;

/// Every node that can reach or be reached from `source` when edge direction
/// is ignored. Marks each one in `marks` and skips nodes already marked.
///
/// Port of NetworkX's `_plain_bfs` helpers.
///
/// `&mut [u32]` is a mutable borrow of a slice: this function may change
/// the elements, and while it runs nobody else can read or write them.
fn component_from(g: &Graph, source: NodeId, marks: &mut [u32]) -> Vec<NodeId> {
    let mut component = vec![source];
    marks[source as usize] = SEEN;
    // `component` doubles as the BFS queue: `head` is the next node to
    // expand, and newly found nodes are appended at the end.
    let mut head = 0;
    while head < component.len() {
        let v = component[head];
        head += 1;
        for &w in g.succ().neighbors(v) {
            if marks[w as usize] == UNSEEN {
                marks[w as usize] = SEEN;
                component.push(w);
            }
        }
        // For an undirected graph the predecessor lists *are* the successor
        // lists, so there is nothing more to look at.
        if g.is_directed() {
            for &w in g.pred().neighbors(v) {
                if marks[w as usize] == UNSEEN {
                    marks[w as usize] = SEEN;
                    component.push(w);
                }
            }
        }
    }
    component
}

/// The connected components of an undirected graph, or the weakly connected
/// components of a directed one.
///
/// Components come out in the order NetworkX yields them: by their first
/// node in node order.
///
/// Ports `networkx.connected_components` and
/// `networkx.weakly_connected_components`.
pub fn connected_components(g: &Graph) -> Vec<Vec<NodeId>> {
    // This visits every node anyway, so a fresh array is as good as a
    // borrowed workspace.
    let mut marks = vec![UNSEEN; g.node_count()];
    let mut components = Vec::new();
    for v in g.nodes() {
        if marks[v as usize] == UNSEEN {
            // `&mut marks` lends the array to the callee for the duration
            // of the call, then we have it back.
            components.push(component_from(g, v, &mut marks));
        }
    }
    components
}

/// The component containing `node`, ignoring edge direction.
///
/// Port of `networkx.node_connected_component`.
pub fn node_component(g: &Graph, node: NodeId, ws: &mut Workspace) -> Vec<NodeId> {
    ws.prepare(g.node_count());
    let component = component_from(g, node, &mut ws.slot);
    for &v in &component {
        ws.slot[v as usize] = UNSEEN;
    }
    component
}

/// True if the graph is one component when edge direction is ignored.
/// A graph with no nodes is reported as not connected; NetworkX raises an
/// error for it and the Python wrapper handles that case.
pub fn is_connected(g: &Graph) -> bool {
    let n = g.node_count();
    n > 0 && node_component(g, 0, &mut Workspace::new()).len() == n
}

/// The strongly connected components of a directed graph.
///
/// Port of `networkx.strongly_connected_components`, which is a
/// non-recursive version of Tarjan's algorithm with Nuutila's modifications.
/// The port is line for line so that components come out in the same order.
pub fn strongly_connected_components(g: &Graph) -> Vec<Vec<NodeId>> {
    let adj = g.succ();
    let n = g.node_count();

    // NetworkX uses dicts and tests `v not in preorder`. Preorder numbers
    // start at 1, so 0 can mean "not numbered yet".
    let mut preorder = vec![0u32; n];
    let mut lowlink = vec![0u32; n];
    let mut scc_found = vec![false; n];
    let mut scc_queue: Vec<NodeId> = Vec::new();
    let mut counter = 0u32;
    // How far we have iterated through each node's neighbours. NetworkX
    // keeps one Python iterator per node for this.
    let mut position = vec![0usize; n];

    let mut components = Vec::new();
    let mut queue: Vec<NodeId> = Vec::new();

    for source in g.nodes() {
        if scc_found[source as usize] {
            continue;
        }
        queue.push(source);
        // `.last()` returns `Option<&NodeId>`; the `&v` pattern copies the
        // node id out so that `queue` is not borrowed inside the loop body.
        while let Some(&v) = queue.last() {
            let vi = v as usize;
            if preorder[vi] == 0 {
                counter += 1;
                preorder[vi] = counter;
            }
            let neighbors = adj.neighbors(v);
            let mut done = true;
            while position[vi] < neighbors.len() {
                let w = neighbors[position[vi]];
                position[vi] += 1;
                if preorder[w as usize] == 0 {
                    queue.push(w);
                    done = false;
                    break;
                }
            }
            if !done {
                continue;
            }

            lowlink[vi] = preorder[vi];
            for &w in neighbors {
                let wi = w as usize;
                if !scc_found[wi] {
                    if preorder[wi] > preorder[vi] {
                        lowlink[vi] = lowlink[vi].min(lowlink[wi]);
                    } else {
                        lowlink[vi] = lowlink[vi].min(preorder[wi]);
                    }
                }
            }
            queue.pop();
            if lowlink[vi] == preorder[vi] {
                // v is the root of a component: it and everything above it
                // on scc_queue with a larger preorder number.
                let mut scc = vec![v];
                while let Some(&k) = scc_queue.last() {
                    if preorder[k as usize] <= preorder[vi] {
                        break;
                    }
                    scc_queue.pop();
                    scc.push(k);
                }
                for &k in &scc {
                    scc_found[k as usize] = true;
                }
                components.push(scc);
            } else {
                scc_queue.push(v);
            }
        }
    }
    components
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sorts each component so tests do not depend on order within one.
    /// `mut components` takes ownership of the argument and lets us modify it.
    fn sorted(mut components: Vec<Vec<NodeId>>) -> Vec<Vec<NodeId>> {
        for c in &mut components {
            c.sort_unstable();
        }
        components
    }

    #[test]
    fn undirected_components() {
        let g = Graph::from_edges(5, &[(0, 1), (3, 4)], false).unwrap();
        assert_eq!(
            sorted(connected_components(&g)),
            vec![vec![0, 1], vec![2], vec![3, 4]]
        );
        assert!(!is_connected(&g));
        assert_eq!(node_component(&g, 4, &mut Workspace::new()), vec![4, 3]);
    }

    #[test]
    fn weak_components_ignore_direction() {
        let g = Graph::from_edges(3, &[(1, 0), (1, 2)], true).unwrap();
        assert_eq!(sorted(connected_components(&g)), vec![vec![0, 1, 2]]);
        assert!(is_connected(&g));
    }

    #[test]
    fn strong_components_in_networkx_order() {
        // A cycle 0 -> 1 -> 2 -> 0 with a tail 2 -> 3.
        let g = Graph::from_edges(4, &[(0, 1), (1, 2), (2, 0), (2, 3)], true).unwrap();
        // NetworkX yields the sink component first.
        assert_eq!(
            sorted(strongly_connected_components(&g)),
            vec![vec![3], vec![0, 1, 2]]
        );
    }
}
