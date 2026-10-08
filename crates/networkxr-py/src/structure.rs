//! Python methods for components, DAGs, spanning trees and cycles.

use networkxr_core::dag::LongestPathsError;
use networkxr_core::{components, cycles, dag, tree, NodeId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};

use crate::{fallback, Snapshot};

impl Snapshot {
    /// A Python list of sets of node objects.
    fn node_sets<'py>(
        &self,
        py: Python<'py>,
        groups: &[Vec<NodeId>],
    ) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for group in groups {
            out.append(PySet::new(py, group.iter().map(|&i| self.node(py, i)))?)?;
        }
        Ok(out)
    }
}

#[pymethods]
impl Snapshot {
    /// Connected components, or weakly connected ones if directed.
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let groups = py.detach(|| components::connected_components(&self.graph));
        self.node_sets(py, &groups)
    }

    fn node_component<'py>(
        &self,
        py: Python<'py>,
        node: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PySet>> {
        let node = self.idx(py, node)?;
        let group = self.with_scratch(|scratch| {
            py.detach(|| components::node_component(&self.graph, node, &mut scratch.a))
        });
        PySet::new(py, group.iter().map(|&i| self.node(py, i)))
    }

    fn is_connected(&self, py: Python<'_>) -> bool {
        py.detach(|| components::is_connected(&self.graph))
    }

    fn strong_components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let groups = py.detach(|| components::strongly_connected_components(&self.graph));
        self.node_sets(py, &groups)
    }

    /// Returns `(generations, acyclic)`. If the graph has a cycle,
    /// `generations` holds what could be sorted before getting stuck.
    fn topological_generations<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyList>, bool)> {
        match py.detach(|| dag::topological_generations(&self.graph)) {
            Ok(generations) => Ok((self.node_lists(py, &generations)?, true)),
            Err(cycle) => Ok((self.node_lists(py, &cycle.partial)?, false)),
        }
    }

    fn is_acyclic(&self, py: Python<'_>) -> bool {
        py.detach(|| dag::is_acyclic(&self.graph))
    }

    fn dag_longest_path<'py>(
        &self,
        py: Python<'py>,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let weights = self.weights(py, weight, default_weight)?;
        // On a cycle NetworkX raises; fall back so that it does.
        let path = py
            .detach(|| dag::dag_longest_path(&self.graph, &weights.values))
            .map_err(|_| fallback())?;
        self.node_list(py, &path)
    }

    /// Returns `None` if the graph has a cycle.
    fn dag_longest_paths<'py>(
        &self,
        py: Python<'py>,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
        max_paths: Option<usize>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let weights = self.weights(py, weight, default_weight)?;
        match py.detach(|| dag::dag_longest_paths(&self.graph, &weights.values, max_paths)) {
            Ok(paths) => Ok(Some(self.node_lists(py, &paths)?)),
            Err(LongestPathsError::Cycle) => Ok(None),
            Err(LongestPathsError::NegativeWeight(e)) => {
                let (u, v) = self.graph.endpoints(e);
                // `format!` builds a String, like an f-string. `repr()`
                // gives the Python repr of the node.
                Err(PyValueError::new_err(format!(
                    "dag_longest_paths requires non-negative weights; edge ({}, {}) has weight {}",
                    self.node(py, u).repr()?,
                    self.node(py, v).repr()?,
                    weights.values[e as usize]
                )))
            }
        }
    }

    /// Spanning forest edges as `(u, v, data)` tuples, or `(u, v)` if `data`
    /// is false.
    fn kruskal<'py>(
        &self,
        py: Python<'py>,
        weight: &Bound<'py, PyAny>,
        default_weight: &Bound<'py, PyAny>,
        minimum: bool,
        data: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let weights = self.weights(py, weight, default_weight)?;
        let chosen = py.detach(|| tree::kruskal(&self.graph, &weights.values, minimum));
        let out = PyList::empty(py);
        for e in chosen {
            let (u, v) = self.graph.endpoints(e);
            let (u, v) = (self.node(py, u), self.node(py, v));
            if data {
                out.append((u, v, self.data_of(py, e)?))?;
            } else {
                out.append((u, v))?;
            }
        }
        Ok(out)
    }

    fn is_forest(&self, py: Python<'_>) -> bool {
        py.detach(|| tree::is_forest(&self.graph))
    }

    fn is_tree(&self, py: Python<'_>) -> bool {
        py.detach(|| tree::is_tree(&self.graph))
    }

    fn cycle_basis<'py>(
        &self,
        py: Python<'py>,
        root: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let root = self.opt_idx(py, root)?;
        let basis = py.detach(|| cycles::cycle_basis(&self.graph, root));
        self.node_lists(py, &basis)
    }
}
