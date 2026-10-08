//! Python methods for breadth-first and depth-first search.

use networkxr_core::{traversal, Direction};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::Snapshot;

/// `Direction::Reverse` if `reverse` is true.
pub(crate) fn direction(reverse: bool) -> Direction {
    if reverse {
        Direction::Reverse
    } else {
        Direction::Forward
    }
}

#[pymethods]
impl Snapshot {
    fn bfs_edges<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        reverse: bool,
        depth_limit: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let source = self.idx(py, source)?;
        // `with_scratch` lends the snapshot's working memory to the closure.
        // Inside, `py.detach` releases the GIL while its closure runs, so
        // other Python threads can work in the meantime. That closure may
        // not touch Python objects, and the compiler enforces it: `Bound`
        // values cannot be captured by it.
        let edges = self.with_scratch(|scratch| {
            py.detach(|| {
                traversal::bfs_edges(
                    &self.graph,
                    source,
                    direction(reverse),
                    depth_limit,
                    &mut scratch.a,
                )
            })
        });
        self.pair_list(py, &edges)
    }

    fn bfs_layers<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let source = self.idx(py, source)?;
        let layers = self.with_scratch(|scratch| {
            py.detach(|| traversal::bfs_layers(&self.graph, source, &mut scratch.a))
        });
        self.node_lists(py, &layers)
    }

    fn dfs_edges<'py>(
        &self,
        py: Python<'py>,
        source: Option<&Bound<'py, PyAny>>,
        depth_limit: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let source = self.opt_idx(py, source)?;
        let edges = self.with_scratch(|scratch| {
            py.detach(|| traversal::dfs_edges(&self.graph, source, depth_limit, &mut scratch.a))
        });
        self.pair_list(py, &edges)
    }

    /// Preorder, or postorder if `postorder` is true.
    fn dfs_nodes<'py>(
        &self,
        py: Python<'py>,
        source: Option<&Bound<'py, PyAny>>,
        depth_limit: Option<usize>,
        postorder: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let source = self.opt_idx(py, source)?;
        let nodes = self.with_scratch(|scratch| {
            py.detach(|| {
                if postorder {
                    traversal::dfs_postorder(&self.graph, source, depth_limit, &mut scratch.a)
                } else {
                    traversal::dfs_preorder(&self.graph, source, depth_limit, &mut scratch.a)
                }
            })
        });
        self.node_list(py, &nodes)
    }
}
