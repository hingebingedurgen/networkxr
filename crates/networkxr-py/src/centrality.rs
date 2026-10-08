//! Python methods for centrality measures.

use networkxr_core::centrality::{self, DegreeKind, PageRankOptions};
use networkxr_core::{EdgeId, NodeId};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{fallback, Snapshot};

#[pymethods]
impl Snapshot {
    /// Returns `{node: rank}`, or `None` if the iteration did not converge.
    ///
    /// `personalization`, `nstart` and `dangling` are one float per node,
    /// already normalised to sum to 1, or `None`.
    ///
    /// PyO3 converts a Python list of floats to `Vec<f64>` for us.
    #[allow(clippy::too_many_arguments)]
    fn pagerank<'py>(
        &self,
        py: Python<'py>,
        alpha: f64,
        max_iter: usize,
        tol: f64,
        weight: &Bound<'py, PyAny>,
        personalization: Option<Vec<f64>>,
        nstart: Option<Vec<f64>>,
        dangling: Option<Vec<f64>>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let n = self.nodes.len();
        // `iter()` on an Option yields its value if there is one, so this
        // checks the length of whichever vectors were given.
        for vector in [&personalization, &nstart, &dangling] {
            if vector.iter().any(|v| v.len() != n) {
                return Err(fallback());
            }
        }
        // `1.into_pyobject(py)` makes the Python int 1, the default weight.
        let one = 1i64.into_pyobject(py)?;
        let weights = self.weights(py, weight, one.as_any())?;
        let options = PageRankOptions {
            alpha,
            max_iter,
            tol,
            // `as_deref` turns `&Option<Vec<f64>>` into `Option<&[f64]>`.
            personalization: personalization.as_deref(),
            nstart: nstart.as_deref(),
            dangling: dangling.as_deref(),
        };
        match py.detach(|| centrality::pagerank(&self.graph, Some(&weights.values), &options)) {
            Ok(ranks) => Ok(Some(self.float_dict(py, &ranks)?)),
            Err(_) => Ok(None),
        }
    }

    /// `kind` is 0 for degree, 1 for in-degree, 2 for out-degree.
    fn degree_centrality<'py>(&self, py: Python<'py>, kind: u8) -> PyResult<Bound<'py, PyDict>> {
        let kind = match kind {
            0 => DegreeKind::Total,
            1 => DegreeKind::In,
            2 => DegreeKind::Out,
            _ => return Err(fallback()),
        };
        if self.nodes.len() < 2 {
            return Err(fallback());
        }
        self.float_dict(py, &centrality::degree_centrality(&self.graph, kind))
    }

    /// `weight` of `None` means unweighted.
    fn betweenness<'py>(
        &self,
        py: Python<'py>,
        weight: Option<&Bound<'py, PyAny>>,
        normalized: bool,
        endpoints: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let weights = match weight {
            Some(key) => {
                let one = 1i64.into_pyobject(py)?;
                let weights = self.weights(py, key, one.as_any())?;
                if weights.seen.negative {
                    return Err(fallback());
                }
                Some(weights)
            }
            None => None,
        };
        // `as_ref()` borrows the Weights inside the Option; `map` then
        // borrows its values as a slice.
        let values = weights.as_ref().map(|w| w.values.as_slice());
        let scores = py.detach(|| {
            centrality::betweenness_centrality(&self.graph, values, normalized, endpoints)
        });
        self.float_dict(py, &scores)
    }

    /// Closeness of every node, as a dict. `distance` of `None` means
    /// unweighted.
    fn closeness_all<'py>(
        &self,
        py: Python<'py>,
        distance: Option<&Bound<'py, PyAny>>,
        wf_improved: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let weights = match distance {
            Some(key) => {
                let one = 1i64.into_pyobject(py)?;
                let weights = self.weights(py, key, one.as_any())?;
                // NetworkX sums the distances with Python's `sum`, which
                // treats ints and floats differently, so a mix is left to
                // NetworkX. `for_dijkstra` checks that and the signs.
                weights.seen.for_dijkstra()?;
                Some(weights)
            }
            None => None,
        };
        let values = weights.as_ref().map(|w| w.values.as_slice());
        let all: Vec<NodeId> = self.graph.nodes().collect();
        let scores =
            py.detach(|| centrality::closeness_centrality(&self.graph, values, &all, wf_improved));
        self.float_dict(py, &scores)
    }

    /// Closeness of one node. Reads only the weights the search reaches.
    fn closeness_of<'py>(
        &self,
        py: Python<'py>,
        node: &Bound<'py, PyAny>,
        distance: Option<&Bound<'py, PyAny>>,
        wf_improved: bool,
    ) -> PyResult<f64> {
        let u = self.idx(py, node)?;
        match distance {
            None => Ok(self.with_scratch(|scratch| {
                // `None::<fn(EdgeId) -> f64>`: an absent closure still needs
                // a type, and a plain function pointer type will do.
                py.detach(|| {
                    centrality::closeness_of(
                        &self.graph,
                        None::<fn(EdgeId) -> f64>,
                        u,
                        wf_improved,
                        &mut scratch.a,
                    )
                })
            })),
            Some(key) => {
                let one = 1i64.into_pyobject(py)?;
                self.with_scratch(|scratch| {
                    let (workspace, memo) = (&mut scratch.a, &mut scratch.weights);
                    let weights = self.lazy_weights(py, key, one.as_any(), memo)?;
                    let weight = Some(|e| weights.get(e));
                    let score =
                        centrality::closeness_of(&self.graph, weight, u, wf_improved, workspace);
                    weights.seen().for_dijkstra()?;
                    Ok(score)
                })
            }
        }
    }
}
