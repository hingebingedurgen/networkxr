//! Reading edge weights out of the NetworkX graph.
//!
//! Weights are never cached. Every call reads them from the graph's own
//! attribute dicts, because NetworkX gives no signal when a caller writes
//! `G[u][v]["weight"] = 3`.
//!
//! There are two ways to read them:
//!
//! - [`Weights`]: all at once into a `Vec<f64>`. Algorithms that look at
//!   every edge anyway (PageRank, betweenness, spanning trees) use this,
//!   then release the GIL and run in parallel.
//! - [`LazyWeights`]: one at a time, when a search first reaches an edge.
//!   A search that stays near its source then reads only the weights near
//!   its source. Searches from a single node use this.
//!
//! Only plain Python `int` and `float` weights are handled. Anything else
//! (bool, NumPy scalar, Decimal, NaN, infinity, an int too large for an f64)
//! raises `Fallback`, and NetworkX deals with it in its own way.

use std::cell::{Cell, RefCell};

use networkxr_core::EdgeId;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt};

use crate::{fallback, Snapshot};

/// Integers up to 2^53 are exactly representable as f64. Larger ones are
/// handed back to NetworkX, which computes with Python's unbounded ints.
const MAX_EXACT_INT: f64 = 9_007_199_254_740_992.0;

/// What kinds of weight have been read so far.
///
/// `Copy` because it is four bools: cheap to pass around by value.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Seen {
    int: bool,
    float: bool,
    pub(crate) negative: bool,
    /// A weight was not a usable number.
    invalid: bool,
}

impl Seen {
    /// Converts a Python weight to f64 and records what kind it was.
    /// Returns `None` for anything that is not a usable number.
    fn read(&mut self, value: &Bound<'_, PyAny>) -> Option<f64> {
        // `is_exact_instance_of` is `type(value) is float`: subclasses such
        // as numpy.float64 do not count.
        let number = if value.is_exact_instance_of::<PyFloat>() {
            self.float = true;
            // `.ok()` turns Result into Option; `?` on an Option returns
            // None from this function if it is None.
            value.extract::<f64>().ok()?
        } else if value.is_exact_instance_of::<PyInt>() {
            self.int = true;
            // Fails for ints beyond 64 bits.
            value.extract::<i64>().ok()? as f64
        } else {
            return None;
        };
        if !number.is_finite() || number.abs() >= MAX_EXACT_INT {
            return None;
        }
        self.negative |= number < 0.0;
        Some(number)
    }

    /// Adds what `other` has seen to this record.
    fn merge(&mut self, other: Seen) {
        self.int |= other.int;
        self.float |= other.float;
        self.negative |= other.negative;
        self.invalid |= other.invalid;
    }

    /// For algorithms that return distances and assume no negative weights.
    pub(crate) fn for_dijkstra(self) -> PyResult<Numbers> {
        if self.negative {
            return Err(fallback());
        }
        self.for_distances()
    }

    /// For algorithms that return distances.
    ///
    /// NetworkX adds weights with Python arithmetic, so a distance is an
    /// int if every weight on the path is an int. We compute in f64 and
    /// convert at the end, which is only possible if the weights seen were
    /// all ints or all floats.
    pub(crate) fn for_distances(self) -> PyResult<Numbers> {
        if self.invalid || (self.int && self.float) {
            return Err(fallback());
        }
        Ok(Numbers {
            all_int: !self.float,
        })
    }
}

/// How to turn computed distances back into Python numbers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Numbers {
    all_int: bool,
}

impl Numbers {
    /// Converts a distance to the Python number NetworkX would return.
    /// The source's distance is always the int 0, even with float weights.
    pub(crate) fn number<'py>(
        &self,
        py: Python<'py>,
        value: f64,
        is_source: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if is_source {
            return Ok(0i64.into_pyobject(py)?.into_any());
        }
        if self.all_int {
            if value.abs() >= MAX_EXACT_INT {
                return Err(fallback());
            }
            Ok((value as i64).into_pyobject(py)?.into_any())
        } else {
            Ok(PyFloat::new(py, value).into_any())
        }
    }
}

/// Every edge's weight, read up front.
pub(crate) struct Weights {
    /// One weight per edge id.
    pub(crate) values: Vec<f64>,
    pub(crate) seen: Seen,
}

/// Remembers the weights one search has read, so each is read once.
///
/// Lives in the snapshot's [`Scratch`](crate::Scratch) and is reused by
/// every search. It is never carried over from one search to the next:
/// `begin` starts a new *epoch*, and an entry counts only if it was stamped
/// with the current epoch. That makes forgetting everything a single
/// increment instead of a pass over the whole array.
#[derive(Default)]
pub(crate) struct WeightMemo {
    value: Vec<f64>,
    stamp: Vec<u32>,
    epoch: u32,
}

impl WeightMemo {
    /// Forgets everything and makes room for `edges` weights.
    fn begin(&mut self, edges: usize) {
        if self.stamp.len() != edges || self.epoch == u32::MAX {
            self.value = vec![0.0; edges];
            self.stamp = vec![0; edges];
            self.epoch = 0;
        }
        self.epoch += 1;
    }
}

/// Edge weights read on demand.
///
/// A search that stays near its source reads only the weights near its
/// source. A search that turns out to cover much of the graph would read
/// most weights this way, one at a time and in a scattered order, which is
/// slower than reading them all in sequence. So after it has read an eighth
/// of them, the rest are read in one go.
///
/// The two lifetimes: `'py` is how long the GIL is held, `'a` how long the
/// borrowed snapshot data, key and memo live.
pub(crate) struct LazyWeights<'a, 'py> {
    py: Python<'py>,
    edge_data: &'a [Py<PyAny>],
    key: &'a Bound<'py, PyAny>,
    default: f64,
    default_kind: Seen,
    /// The search algorithms take the weight function as `Fn`, which can
    /// only capture by shared reference, yet reading a weight has to update
    /// the memo and the record of what was seen. The types below allow
    /// changing a value through a shared reference ("interior mutability"):
    ///
    /// - `Cell<T>` for small `Copy` values: `get()` copies the value out,
    ///   `set()` copies a new one in.
    /// - `RefCell<T>` for anything else: `borrow_mut()` hands out a mutable
    ///   borrow and checks *at run time* that there is no other borrow,
    ///   panicking if there is. The compile-time rule is the same; only the
    ///   moment of checking moves.
    seen: Cell<Seen>,
    memo: RefCell<&'a mut WeightMemo>,
    /// How many more weights to read one at a time before reading the rest.
    patience: Cell<usize>,
}

impl LazyWeights<'_, '_> {
    /// The weight of edge `e`.
    ///
    /// On an unusable weight this records the fact and returns 1.0, so the
    /// search can finish harmlessly; the caller then finds out from
    /// [`seen`](Self::seen) and discards the result.
    pub(crate) fn get(&self, e: EdgeId) -> f64 {
        let mut memo = self.memo.borrow_mut();
        let i = e as usize;
        if memo.stamp[i] == memo.epoch {
            return memo.value[i];
        }
        let weight = self.read(i);
        memo.value[i] = weight;
        memo.stamp[i] = memo.epoch;

        let patience = self.patience.get();
        if patience > 1 {
            self.patience.set(patience - 1);
        } else if patience == 1 {
            self.patience.set(0);
            for j in 0..memo.stamp.len() {
                if memo.stamp[j] != memo.epoch {
                    memo.value[j] = self.read(j);
                    memo.stamp[j] = memo.epoch;
                }
            }
        }
        weight
    }

    /// Reads the weight of the edge with index `i` from the graph.
    fn read(&self, i: usize) -> f64 {
        let mut seen = self.seen.get();
        let weight = self.read_checked(i, &mut seen);
        if weight.is_none() {
            seen.invalid = true;
        }
        self.seen.set(seen);
        weight.unwrap_or(1.0)
    }

    fn read_checked(&self, i: usize, seen: &mut Seen) -> Option<f64> {
        let data = self.edge_data[i]
            .bind(self.py)
            .cast_exact::<PyDict>()
            .ok()?;
        match data.get_item(self.key).ok()? {
            Some(value) => seen.read(&value),
            None => {
                seen.merge(self.default_kind);
                Some(self.default)
            }
        }
    }

    /// What kinds of weight the search read.
    pub(crate) fn seen(&self) -> Seen {
        self.seen.get()
    }
}

impl Snapshot {
    /// Reads `data.get(key, default)` for every edge.
    pub(crate) fn weights(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: &Bound<'_, PyAny>,
    ) -> PyResult<Weights> {
        // A callable weight is a function NetworkX calls per edge.
        if key.is_callable() {
            return Err(fallback());
        }
        let edge_data = self.edge_data(py)?;
        let mut seen = Seen::default();
        let mut values = Vec::with_capacity(edge_data.len());
        for data in edge_data {
            let data = data
                .bind(py)
                .cast_exact::<PyDict>()
                .map_err(|_| fallback())?;
            // An unhashable key raises TypeError here; let NetworkX report it.
            let found = data.get_item(key).map_err(|_| fallback())?;
            // `as_ref()` borrows the value inside the Option.
            let value = found.as_ref().unwrap_or(default);
            // `ok_or_else` turns None into an error.
            values.push(seen.read(value).ok_or_else(fallback)?);
        }
        Ok(Weights { values, seen })
    }

    /// Prepares to read `data.get(key, default)` edge by edge, remembering
    /// what is read in `memo`.
    pub(crate) fn lazy_weights<'a, 'py>(
        &'a self,
        py: Python<'py>,
        key: &'a Bound<'py, PyAny>,
        default: &Bound<'py, PyAny>,
        memo: &'a mut WeightMemo,
    ) -> PyResult<LazyWeights<'a, 'py>> {
        if key.is_callable() {
            return Err(fallback());
        }
        // The default's kind (int or float) only counts if the default ends
        // up being used, so it is recorded separately.
        let mut kind = Seen::default();
        let default_value = kind.read(default).ok_or_else(fallback)?;
        let edge_data = self.edge_data(py)?;
        memo.begin(edge_data.len());
        Ok(LazyWeights {
            py,
            edge_data,
            key,
            default: default_value,
            default_kind: kind,
            seen: Cell::new(Seen::default()),
            memo: RefCell::new(memo),
            patience: Cell::new((edge_data.len() / 8).max(64)),
        })
    }
}
