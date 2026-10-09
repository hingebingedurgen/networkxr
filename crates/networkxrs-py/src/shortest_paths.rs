//! Python methods for shortest paths.
//!
//! The dicts built here list nodes in the order NetworkX lists them: the
//! order the search reached (BFS, Bellman-Ford) or finalised (Dijkstra)
//! them.
//!
//! Single-source and point-to-point searches read weights lazily and keep
//! the GIL, since reading a weight touches a Python dict. The all-pairs
//! methods read every weight first, then release the GIL and search in
//! parallel.

use networkxrs_core::shortest_paths::{self as sp, Tree, NO_PARENT};
use networkxrs_core::NodeId;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::traversal::direction;
use crate::weights::Numbers;
use crate::{fallback, NegativeCycle, Snapshot};

/// What a single-source method should return.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    Lengths,
    Paths,
    /// A `(lengths, paths)` tuple.
    Both,
}

impl Want {
    /// Python passes 0, 1 or 2.
    fn from_code(code: u8) -> PyResult<Want> {
        match code {
            0 => Ok(Want::Lengths),
            1 => Ok(Want::Paths),
            2 => Ok(Want::Both),
            _ => Err(fallback()),
        }
    }
}

impl Snapshot {
    /// `{node: number of edges}` for a breadth-first search result.
    fn bfs_length_dict<'py>(
        &self,
        py: Python<'py>,
        tree: &sp::BfsTree,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        // `zip` walks the two parallel Vecs together.
        for (&v, &d) in tree.order.iter().zip(&tree.dist) {
            out.set_item(self.node(py, v), d)?;
        }
        Ok(out)
    }

    /// `{node: distance}` in the number type NetworkX would use. The first
    /// node in `order` is the source.
    fn length_dict<'py>(
        &self,
        py: Python<'py>,
        order: &[NodeId],
        dist: &[f64],
        numbers: Numbers,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (i, (&v, &d)) in order.iter().zip(dist).enumerate() {
            out.set_item(self.node(py, v), numbers.number(py, d, i == 0)?)?;
        }
        Ok(out)
    }

    /// `{node: path}` from a shortest-path tree.
    ///
    /// NetworkX builds each path as `paths[parent] + [node]`, and so does
    /// this: a node's parent always comes before it in the tree, so the
    /// parent's list already exists. With `reversed`, each path runs from
    /// the node to the root instead, which is what a search on the reversed
    /// graph needs.
    ///
    /// The function is generic over the tree's distance type `D`, which it
    /// never looks at.
    fn path_dict<'py, D>(
        &self,
        py: Python<'py>,
        tree: &Tree<D>,
        reversed: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        // lists[i] is the path of the node at position i.
        let mut lists: Vec<Bound<'py, PyList>> = Vec::with_capacity(tree.order.len());
        for (&v, &parent) in tree.order.iter().zip(&tree.parent) {
            let single = PyList::new(py, [self.node(py, v)])?;
            let path = if parent == NO_PARENT {
                single
            } else {
                let parent_path = &lists[parent as usize];
                // `concat` is Python's `a + b` for sequences.
                let joined = if reversed {
                    single.as_sequence().concat(parent_path.as_sequence())?
                } else {
                    parent_path.as_sequence().concat(single.as_sequence())?
                };
                joined.cast_into::<PyList>()?
            };
            out.set_item(self.node(py, v), &path)?;
            lists.push(path);
        }
        Ok(out)
    }

    /// Converts one Dijkstra result into what the caller asked for.
    fn dijkstra_output<'py>(
        &self,
        py: Python<'py>,
        tree: &sp::ShortestPaths,
        numbers: Numbers,
        want: Want,
        reverse_paths: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // A closure for each half, so only the requested half is built.
        let lengths = || self.length_dict(py, &tree.order, &tree.dist, numbers);
        let paths = || self.path_dict(py, tree, reverse_paths);
        Ok(match want {
            Want::Lengths => lengths()?.into_any(),
            Want::Paths => paths()?.into_any(),
            Want::Both => (lengths()?, paths()?).into_pyobject(py)?.into_any(),
        })
    }
}

#[pymethods]
impl Snapshot {
    // ----- unweighted ------------------------------------------------------

    fn bfs_lengths<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        cutoff: Option<usize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let source = self.idx(py, source)?;
        let tree = self.with_scratch(|scratch| {
            py.detach(|| {
                sp::bfs(
                    &self.graph,
                    source,
                    direction(reverse),
                    cutoff,
                    &mut scratch.a,
                )
            })
        });
        self.bfs_length_dict(py, &tree)
    }

    fn bfs_paths<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        cutoff: Option<usize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let source = self.idx(py, source)?;
        let tree = self.with_scratch(|scratch| {
            py.detach(|| {
                sp::bfs(
                    &self.graph,
                    source,
                    direction(reverse),
                    cutoff,
                    &mut scratch.a,
                )
            })
        });
        // A reversed search wants its paths written target-last.
        self.path_dict(py, &tree, reverse)
    }

    /// The set of nodes reachable from `source`, without `source` itself.
    fn reachable<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        reverse: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        let source = self.idx(py, source)?;
        let tree = self.with_scratch(|scratch| {
            py.detach(|| {
                sp::bfs(
                    &self.graph,
                    source,
                    direction(reverse),
                    None,
                    &mut scratch.a,
                )
            })
        });
        // `[1..]` skips the source, which is always first.
        pyo3::types::PySet::new(py, tree.order[1..].iter().map(|&v| self.node(py, v)))
    }

    /// Returns `None` if there is no path.
    fn bidirectional_bfs<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        target: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let (source, target) = (self.idx(py, source)?, self.idx(py, target)?);
        let path = self.with_scratch(|scratch| {
            py.detach(|| {
                sp::bidirectional_bfs(&self.graph, source, target, &mut scratch.a, &mut scratch.b)
            })
        });
        match path {
            Some(path) => Ok(Some(self.node_list(py, &path)?)),
            None => Ok(None),
        }
    }

    /// One dict per source node in `start..stop`, searched in parallel.
    /// Dicts map to path lists if `paths` is true, else to lengths.
    fn all_bfs<'py>(
        &self,
        py: Python<'py>,
        start: NodeId,
        stop: NodeId,
        cutoff: Option<usize>,
        paths: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let sources: Vec<NodeId> = (start..stop).collect();
        let trees = py.detach(|| sp::bfs_many(&self.graph, &sources, direction(false), cutoff));
        let out = PyList::empty(py);
        for tree in &trees {
            if paths {
                out.append(self.path_dict(py, tree, false)?)?;
            } else {
                out.append(self.bfs_length_dict(py, tree)?)?;
            }
        }
        Ok(out)
    }

    fn bfs_distance_sum(&self, py: Python<'_>) -> u64 {
        py.detach(|| sp::bfs_distance_sum(&self.graph))
    }

    // ----- Dijkstra --------------------------------------------------------

    /// Single-source Dijkstra. `want` is 0 for lengths, 1 for paths, 2 for
    /// `(lengths, paths)`.
    // `#[allow(...)]` silences one lint for the item that follows. Clippy
    // dislikes long argument lists; this one mirrors the Python call.
    #[allow(clippy::too_many_arguments)]
    fn dijkstra<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
        cutoff: Option<f64>,
        reverse: bool,
        want: u8,
    ) -> PyResult<Bound<'py, PyAny>> {
        let want = Want::from_code(want)?;
        let source = self.idx(py, source)?;
        // `with_scratch` returns whatever the closure returns, here a
        // PyResult, and the trailing `?` unwraps it.
        let (tree, numbers) = self.with_scratch(|scratch| {
            // The search needs one part of the scratch and the weights
            // another. Borrowing the fields separately lets both be mutable
            // at once; borrowing `scratch` as a whole twice would not compile.
            let (workspace, memo) = (&mut scratch.a, &mut scratch.weights);
            let weights = self.lazy_weights(py, weight, default_weight, memo)?;
            // The closure `|e| weights.get(e)` is the weight function: it
            // reads an edge's weight from the NetworkX graph when the search
            // asks for it.
            let tree = sp::dijkstra(
                &self.graph,
                |e| weights.get(e),
                source,
                direction(reverse),
                None,
                cutoff,
                workspace,
            );
            // Only now do we know whether every weight read was acceptable.
            // `Ok::<_, PyErr>` names the closure's error type, which the
            // compiler cannot infer here.
            Ok::<_, PyErr>((tree, weights.seen().for_dijkstra()?))
        })?;
        self.dijkstra_output(py, &tree, numbers, want, reverse)
    }

    /// Returns `(length, path)`, or `None` if `target` is not reachable.
    fn dijkstra_target<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        target: &Bound<'py, PyAny>,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
        cutoff: Option<f64>,
    ) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyList>)>> {
        let (source, target) = (self.idx(py, source)?, self.idx(py, target)?);
        let (tree, numbers) = self.with_scratch(|scratch| {
            let (workspace, memo) = (&mut scratch.a, &mut scratch.weights);
            let weights = self.lazy_weights(py, weight, default_weight, memo)?;
            let forward = direction(false);
            let tree = sp::dijkstra(
                &self.graph,
                |e| weights.get(e),
                source,
                forward,
                Some(target),
                cutoff,
                workspace,
            );
            Ok::<_, PyErr>((tree, weights.seen().for_dijkstra()?))
        })?;
        // The search stops when it finalises the target, so if the target
        // was reached it is the last node in the tree.
        let last = tree.order.len() - 1;
        if tree.order[last] != target {
            return Ok(None);
        }
        let length = numbers.number(py, tree.dist[last], last == 0)?;
        Ok(Some((
            length,
            self.node_list(py, &tree.path_to_position(last))?,
        )))
    }

    /// Returns `(length, path)`, or `None` if there is no path.
    fn bidirectional_dijkstra<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        target: &Bound<'py, PyAny>,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
    ) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyList>)>> {
        let (source, target) = (self.idx(py, source)?, self.idx(py, target)?);
        let (found, numbers) = self.with_scratch(|scratch| {
            let (forward, backward, memo) = (&mut scratch.a, &mut scratch.b, &mut scratch.weights);
            let weights = self.lazy_weights(py, weight, default_weight, memo)?;
            let found = sp::bidirectional_dijkstra(
                &self.graph,
                |e| weights.get(e),
                source,
                target,
                forward,
                backward,
            );
            Ok::<_, PyErr>((found, weights.seen().for_dijkstra()?))
        })?;
        match found {
            Some((length, path)) => {
                let length = numbers.number(py, length, source == target)?;
                Ok(Some((length, self.node_list(py, &path)?)))
            }
            None => Ok(None),
        }
    }

    /// One result per source node in `start..stop`, searched in parallel.
    #[allow(clippy::too_many_arguments)]
    fn all_dijkstra<'py>(
        &self,
        py: Python<'py>,
        start: NodeId,
        stop: NodeId,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
        cutoff: Option<f64>,
        want: u8,
    ) -> PyResult<Bound<'py, PyList>> {
        let want = Want::from_code(want)?;
        let weights = self.weights(py, weight, default_weight)?;
        let numbers = weights.seen.for_dijkstra()?;
        let sources: Vec<NodeId> = (start..stop).collect();
        let values = &weights.values;
        let trees = py.detach(|| {
            sp::dijkstra_many(
                &self.graph,
                |e| values[e as usize],
                &sources,
                direction(false),
                cutoff,
            )
        });
        let out = PyList::empty(py);
        for tree in &trees {
            out.append(self.dijkstra_output(py, tree, numbers, want, false)?)?;
        }
        Ok(out)
    }

    // ----- Bellman-Ford ----------------------------------------------------

    /// Single-source Bellman-Ford. `want` as for `dijkstra`. Raises
    /// `NegativeCycle` if one is reachable.
    fn bellman_ford<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
        reverse: bool,
        want: u8,
    ) -> PyResult<Bound<'py, PyAny>> {
        let want = Want::from_code(want)?;
        let source = self.idx(py, source)?;
        let (outcome, numbers) = self.with_scratch(|scratch| {
            let (workspace, memo) = (&mut scratch.bellman_ford, &mut scratch.weights);
            let weights = self.lazy_weights(py, weight, default_weight, memo)?;
            let outcome = sp::bellman_ford(
                &self.graph,
                |e| weights.get(e),
                source,
                direction(reverse),
                workspace,
            );
            // Check the weights before the outcome: a "negative cycle" found
            // while reading unusable weights means nothing.
            Ok::<_, PyErr>((outcome, weights.seen().for_distances()?))
        })?;
        let result = outcome.map_err(|_| NegativeCycle::new_err(()))?;

        let lengths = || self.length_dict(py, &result.order, &result.dist, numbers);
        let paths = || -> PyResult<Bound<'py, PyDict>> {
            let out = PyDict::new(py);
            // `all_paths` returns the paths in `order`, so zip lines them up.
            for (&v, mut path) in result.order.iter().zip(result.all_paths()) {
                if reverse {
                    path.reverse();
                }
                out.set_item(self.node(py, v), self.node_list(py, &path)?)?;
            }
            Ok(out)
        };
        Ok(match want {
            Want::Lengths => lengths()?.into_any(),
            Want::Paths => paths()?.into_any(),
            Want::Both => (lengths()?, paths()?).into_pyobject(py)?.into_any(),
        })
    }

    /// Returns `(length, path)`, or `None` if `target` is not reachable.
    fn bellman_ford_target<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        target: &Bound<'py, PyAny>,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
    ) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyList>)>> {
        let (source, target) = (self.idx(py, source)?, self.idx(py, target)?);
        let (outcome, numbers) = self.with_scratch(|scratch| {
            let (workspace, memo) = (&mut scratch.bellman_ford, &mut scratch.weights);
            let weights = self.lazy_weights(py, weight, default_weight, memo)?;
            let outcome = sp::bellman_ford(
                &self.graph,
                |e| weights.get(e),
                source,
                direction(false),
                workspace,
            );
            Ok::<_, PyErr>((outcome, weights.seen().for_distances()?))
        })?;
        let result = outcome.map_err(|_| NegativeCycle::new_err(()))?;
        // `let ... else` again: bind the position or return early.
        let Some(i) = result.position(target) else {
            return Ok(None);
        };
        let mut on_stack = vec![false; result.order.len()];
        let path = result.path_to_position(i, &mut on_stack);
        let length = numbers.number(py, result.dist[i], i == 0)?;
        Ok(Some((length, self.node_list(py, &path)?)))
    }
}
