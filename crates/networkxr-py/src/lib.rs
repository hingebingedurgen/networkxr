//! Python bindings for `networkxr-core`, built with PyO3.
//!
//! This crate compiles to the Python extension module `networkxr._core`. It
//! is the boundary layer: it turns a NetworkX graph into a
//! [`networkxr_core::Graph`], calls an algorithm, and turns the integer
//! results back into Python objects.
//!
//! # How PyO3 looks
//!
//! - `#[pyclass]` on a struct makes it a Python class.
//! - `#[pymethods]` on an `impl` block makes its functions Python methods.
//!   PyO3 generates the argument parsing and converts return values.
//! - `Python<'py>` (always called `py` here) is a token proving the current
//!   thread holds the Global Interpreter Lock. Functions that touch Python
//!   objects require it, so touching Python without the GIL does not compile.
//! - `Bound<'py, T>` is a reference to a Python object of type `T` that is
//!   tied to holding the GIL. `Py<T>` is the same reference without that
//!   tie; it is what you store in a struct. `.bind(py)` and `.unbind()`
//!   convert between them.
//! - `PyResult<T>` is `Result<T, PyErr>`. Returning `Err` raises the
//!   exception in Python.
//!
//! # The fallback rule
//!
//! Any input this layer does not handle exactly as NetworkX would (a missing
//! node, a weight that is not a plain number, a multigraph) raises
//! [`Fallback`]. The Python wrapper catches it and calls NetworkX instead.
//! So "unsupported" is never an error the user sees, and error messages for
//! bad input come from NetworkX itself.

use std::sync::Mutex;

use networkxr_core::shortest_paths::BellmanFordWorkspace;
use networkxr_core::{EdgeId, Graph, NodeId, Workspace};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use adjacency::{object_at, scan};

mod adjacency;
// Each of these files adds a `#[pymethods] impl Snapshot` block.
mod centrality;
mod shortest_paths;
mod structure;
mod traversal;
mod weights;

// A macro (note the `!`) that defines a new Python exception class.
create_exception!(
    _core,
    Fallback,
    PyException,
    "Raised when the Rust path does not apply. The caller should use NetworkX."
);
create_exception!(
    _core,
    NegativeCycle,
    PyException,
    "A negative cycle was detected."
);

/// The error to return for anything unsupported.
pub(crate) fn fallback() -> PyErr {
    Fallback::new_err(())
}

/// A read-only snapshot of a NetworkX graph.
///
/// `frozen` tells PyO3 that methods only ever take `&self`, a shared
/// reference, so it need not guard the object against simultaneous mutable
/// access. The few fields that do change after construction say so in their
/// types (`PyOnceLock`, `Mutex`).
#[pyclass(frozen, module = "networkxr._core")]
pub struct Snapshot {
    pub(crate) graph: Graph,
    /// `nodes[i]` is the Python object for node `i`.
    pub(crate) nodes: Vec<Py<PyAny>>,
    /// The graph's own adjacency dict (`G._adj`), kept so that edge
    /// attributes can be read later.
    adjacency: Py<PyDict>,
    /// A checksum of the adjacency as it was when the snapshot was built.
    /// See `Scan::fingerprint` in `adjacency.rs`.
    fingerprint: u64,
    /// The reverse of `nodes`: a Python dict `{node: i}`. A Python dict
    /// rather than a Rust HashMap because only Python knows how to hash and
    /// compare arbitrary Python objects.
    ///
    /// `PyOnceLock<T>` is a cell that starts empty and can be filled exactly
    /// once, even through a shared `&self`. It gives us lazy
    /// initialisation: an algorithm that never looks up a node by its
    /// Python object (connected components, say) never pays for the dict.
    index: PyOnceLock<Py<PyDict>>,
    /// `edge_data[e]` is the attribute dict of edge `e`. These are the live
    /// dicts inside the NetworkX graph, not copies, so weights are always
    /// read fresh. Filled on the first weighted call.
    edge_data: PyOnceLock<Vec<Py<PyAny>>>,
    /// Working memory lent to searches. See [`Snapshot::with_scratch`].
    scratch: Mutex<Scratch>,
}

/// The reusable working memory a snapshot keeps: two workspaces, because a
/// bidirectional search needs one for each end, one for Bellman-Ford, and
/// a memo of the weights the current search has read. They start empty and
/// size themselves on first use.
#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) a: Workspace,
    pub(crate) b: Workspace,
    pub(crate) bellman_ford: BellmanFordWorkspace,
    pub(crate) weights: weights::WeightMemo,
}

/// Builds the node-to-index dict.
fn index_of(py: Python<'_>, nodes: &[Py<PyAny>]) -> PyResult<Py<PyDict>> {
    let index = PyDict::new(py);
    // `enumerate` pairs each item with its position.
    for (i, node) in nodes.iter().enumerate() {
        index.set_item(node.bind(py), i)?;
    }
    Ok(index.unbind())
}

/// Reads one adjacency dict into CSR offsets and targets by looking every
/// neighbour up in `index`. This is the slow way; see `Snapshot::new`.
fn read_by_key<'py>(
    index: &Bound<'py, PyDict>,
    adjacency: &Bound<'py, PyDict>,
) -> PyResult<(Vec<usize>, Vec<NodeId>)> {
    let mut offsets = Vec::with_capacity(adjacency.len() + 1);
    let mut targets = Vec::new();
    offsets.push(0);
    for (_node, neighbors) in adjacency.iter() {
        // `cast_exact` checks the Python type at runtime.
        let neighbors = neighbors.cast_exact::<PyDict>().map_err(|_| fallback())?;
        for (neighbor, _data) in neighbors.iter() {
            let v: NodeId = match index.get_item(&neighbor)? {
                Some(i) => i.extract()?,
                None => return Err(fallback()),
            };
            targets.push(v);
        }
        offsets.push(targets.len());
    }
    Ok((offsets, targets))
}

#[pymethods]
impl Snapshot {
    /// `Snapshot(G._adj, None)` for an undirected graph, or
    /// `Snapshot(G._succ, G._pred)` for a directed one.
    ///
    /// `#[new]` marks the function Python calls for `Snapshot(...)`.
    #[new]
    fn new(
        py: Python<'_>,
        adjacency: &Bound<'_, PyDict>,
        pred: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let directed = pred.is_some();
        let succ_scan = scan(adjacency, true)?;
        let fingerprint = succ_scan.fingerprint();
        // `map` + `transpose` runs the fallible scan on the dict inside the
        // Option and gives back PyResult<Option<Scan>>.
        let pred_scan = pred.map(|p| scan(p, false)).transpose()?;
        let nodes = succ_scan.nodes;
        let index = PyOnceLock::new();

        // The fast way: pair up entries by the address of their data dict.
        // `as_ref()` lets us look inside the Option without consuming it.
        let pred_parts = pred_scan
            .as_ref()
            .map(|p| (p.offsets.clone(), p.tokens.as_slice()));
        let by_token = Graph::from_edge_tokens(
            directed,
            succ_scan.offsets.clone(),
            &succ_scan.tokens,
            pred_parts,
        );
        let graph = match by_token {
            // A match arm with a guard: taken only if the `if` also holds.
            Ok((graph, loops))
                if self_loops_are_real(
                    py,
                    adjacency,
                    &nodes,
                    &succ_scan.offsets,
                    &succ_scan.tokens,
                    &loops,
                )? =>
            {
                graph
            }
            // The slow way, for a graph whose two directions do not share
            // their data dicts. NetworkX's own methods never produce one.
            _ => {
                let node_index = index_of(py, &nodes)?;
                let (succ_offsets, succ_targets) = read_by_key(node_index.bind(py), adjacency)?;
                let pred = match pred {
                    Some(pred) => Some(read_by_key(node_index.bind(py), pred)?),
                    None => None,
                };
                // Store the dict we just built. `set` fails only if the cell
                // is already full, which it is not.
                let _ = index.set(py, node_index);
                Graph::from_csr(directed, succ_offsets, succ_targets, pred)
                    .map_err(|_| fallback())?
            }
        };
        Ok(Snapshot {
            graph,
            nodes,
            adjacency: adjacency.clone().unbind(),
            fingerprint,
            index,
            edge_data: PyOnceLock::new(),
            scratch: Mutex::new(Scratch::default()),
        })
    }

    /// `#[getter]` exposes a method as a read-only Python attribute.
    #[getter]
    fn directed(&self) -> bool {
        self.graph.is_directed()
    }

    #[getter]
    fn number_of_nodes(&self) -> usize {
        self.graph.node_count()
    }

    #[getter]
    fn number_of_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Makes pickling or deep-copying a graph that carries a cached snapshot
    /// work: the snapshot is replaced by `None` and rebuilt on demand.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        // `type(None)` called with no arguments returns `None`.
        let none_type = py.None().into_bound(py).get_type().into_any();
        Ok((none_type, PyTuple::empty(py)))
    }
}

/// Checks the entries `from_edge_tokens` assumed to be self-loops: each
/// must be `adjacency[node][node]`.
fn self_loops_are_real(
    py: Python<'_>,
    adjacency: &Bound<'_, PyDict>,
    nodes: &[Py<PyAny>],
    offsets: &[usize],
    tokens: &[u64],
    loops: &[usize],
) -> PyResult<bool> {
    for &slot in loops {
        // `partition_point` is a binary search for the first offset beyond
        // `slot`; the node owning the slot is the one before it.
        let owner = offsets.partition_point(|&o| o <= slot) - 1;
        let node = nodes[owner].bind(py);
        let Some(neighbors) = adjacency.get_item(node)? else {
            return Ok(false);
        };
        let Ok(neighbors) = neighbors.cast_exact::<PyDict>() else {
            return Ok(false);
        };
        match neighbors.get_item(node)? {
            // `as_ptr` gives the object's address without touching it.
            Some(data) if data.as_ptr() as u64 == tokens[slot] => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

// Helper methods that are not exposed to Python: a plain `impl` block
// without `#[pymethods]`.
impl Snapshot {
    /// The `{node: index}` dict, built on first use.
    fn index<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyDict>> {
        // Runs the closure only if the cell is still empty.
        let index = self
            .index
            .get_or_try_init(py, || index_of(py, &self.nodes))?;
        Ok(index.bind(py))
    }

    /// Runs `f` with the snapshot's working memory.
    ///
    /// A `Mutex<T>` guards a value so that only one thread at a time can
    /// use it. `try_lock` takes the lock if it is free and fails at once if
    /// it is not. It never waits, and that matters: a thread that waited
    /// here while holding the GIL could deadlock with the thread holding
    /// the lock, which may itself be waiting for the GIL. If the scratch is
    /// busy, the search gets fresh, temporary memory instead.
    ///
    /// `impl FnOnce(&mut Scratch) -> R` is a closure that will be called
    /// once; `R` is whatever it returns.
    pub(crate) fn with_scratch<R>(&self, f: impl FnOnce(&mut Scratch) -> R) -> R {
        match self.scratch.try_lock() {
            // `guard` unlocks the mutex when it goes out of scope.
            // `&mut guard` reaches through it to the Scratch.
            Ok(mut guard) => f(&mut guard),
            Err(_) => f(&mut Scratch::default()),
        }
    }

    /// The attribute dict of every edge, by edge id, collected on first use.
    pub(crate) fn edge_data(&self, py: Python<'_>) -> PyResult<&Vec<Py<PyAny>>> {
        self.edge_data.get_or_try_init(py, || {
            let scanned = scan(self.adjacency.bind(py), false)?;
            // The dispatch layer only uses a snapshot while NetworkX's cache
            // says the graph is unchanged. This is a second line of defence.
            if scanned.offsets != self.graph.succ().offsets()
                || scanned.fingerprint() != self.fingerprint
            {
                return Err(fallback());
            }
            let mut data = Vec::with_capacity(self.graph.edge_count());
            let directed = self.graph.is_directed();
            for u in self.graph.nodes() {
                let first = scanned.offsets[u as usize];
                for (i, &v) in self.graph.succ().neighbors(u).iter().enumerate() {
                    // The same rule that numbers the edges: an undirected
                    // edge belongs to its first endpoint in node order.
                    if directed || v >= u {
                        // SAFETY: the tokens come from the scan a few lines
                        // up, and no Python code has run since.
                        data.push(unsafe { object_at(py, scanned.tokens[first + i]) });
                    }
                }
            }
            Ok(data)
        })
    }

    /// The index of a Python node object, or `Fallback` if it is not in the
    /// graph (or cannot be hashed).
    pub(crate) fn idx(&self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<NodeId> {
        match self.index(py)?.get_item(node) {
            Ok(Some(i)) => i.extract(),
            _ => Err(fallback()),
        }
    }

    /// The attribute dict of edge `e`.
    pub(crate) fn data_of<'py>(&self, py: Python<'py>, e: EdgeId) -> PyResult<&Bound<'py, PyAny>> {
        Ok(self.edge_data(py)?[e as usize].bind(py))
    }

    /// Like [`idx`](Self::idx) for an optional node.
    pub(crate) fn opt_idx(
        &self,
        py: Python<'_>,
        node: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<NodeId>> {
        // `transpose` turns Option<Result<T>> into Result<Option<T>>.
        node.map(|n| self.idx(py, n)).transpose()
    }

    /// The Python object for node `i`.
    pub(crate) fn node<'py>(&self, py: Python<'py>, i: NodeId) -> &Bound<'py, PyAny> {
        self.nodes[i as usize].bind(py)
    }

    /// A Python list of node objects.
    pub(crate) fn node_list<'py>(
        &self,
        py: Python<'py>,
        ids: &[NodeId],
    ) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, ids.iter().map(|&i| self.node(py, i)))
    }

    /// A Python list of lists of node objects.
    pub(crate) fn node_lists<'py>(
        &self,
        py: Python<'py>,
        lists: &[Vec<NodeId>],
    ) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for ids in lists {
            out.append(self.node_list(py, ids)?)?;
        }
        Ok(out)
    }

    /// A Python list of `(u, v)` tuples.
    pub(crate) fn pair_list<'py>(
        &self,
        py: Python<'py>,
        pairs: &[(NodeId, NodeId)],
    ) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for &(u, v) in pairs {
            out.append((self.node(py, u), self.node(py, v)))?;
        }
        Ok(out)
    }

    /// A Python dict `{node: value}` over all nodes, from one float per node.
    pub(crate) fn float_dict<'py>(
        &self,
        py: Python<'py>,
        values: &[f64],
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (node, &value) in self.nodes.iter().zip(values) {
            out.set_item(node.bind(py), value)?;
        }
        Ok(out)
    }
}

/// Reports whether at least `limit` nodes can be reached from `source` by
/// following `adjacency`, a NetworkX `{node: {neighbour: data}}` dict.
///
/// This runs directly on the NetworkX dicts, before any snapshot exists. The
/// dispatch layer uses it to decide whether a search from `source` will be
/// big enough to justify building one. It stops as soon as the answer is
/// known, so it never visits more of the graph than NetworkX's own search
/// from `source` would.
///
/// `#[pyfunction]` exposes a plain function to Python.
#[pyfunction]
fn reaches(
    adjacency: &Bound<'_, PyDict>,
    source: &Bound<'_, PyAny>,
    limit: usize,
) -> PyResult<bool> {
    let seen = PySet::empty(adjacency.py())?;
    seen.add(source)?;
    let mut stack = vec![source.clone()];
    while let Some(node) = stack.pop() {
        if seen.len() >= limit {
            return Ok(true);
        }
        let Some(neighbors) = adjacency.get_item(&node)? else {
            return Ok(false);
        };
        let Ok(neighbors) = neighbors.cast_exact::<PyDict>() else {
            return Ok(false);
        };
        for (neighbor, _data) in neighbors.iter() {
            // Adding to a set is a no-op for a member, so the set's length
            // tells us whether the neighbour was new.
            let before = seen.len();
            seen.add(&neighbor)?;
            if seen.len() > before {
                stack.push(neighbor);
            }
        }
    }
    Ok(seen.len() >= limit)
}

/// The module initialiser. Python calls it on `import networkxr._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Snapshot>()?;
    // `wrap_pyfunction!` builds the Python function object.
    m.add_function(wrap_pyfunction!(reaches, m)?)?;
    m.add("Fallback", m.py().get_type::<Fallback>())?;
    m.add("NegativeCycle", m.py().get_type::<NegativeCycle>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
