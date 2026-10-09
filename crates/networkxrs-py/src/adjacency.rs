//! Reading NetworkX's adjacency dicts as fast as CPython allows.
//!
//! This is the only file in the project with `unsafe` code.
//!
//! # Why unsafe
//!
//! Building a snapshot means visiting every entry of
//! `{node: {neighbour: data}}`. PyO3's safe dict iterator hands out an owned
//! reference for every key and value, which means incrementing and later
//! decrementing a reference count inside each object. Those objects are
//! scattered across memory, so on a large graph each touch is a cache miss,
//! and the misses dominated the time to build a snapshot.
//!
//! CPython's own `PyDict_Next` yields *borrowed* pointers: plain addresses,
//! with no reference count touched. We only want the addresses (see
//! [`networkxrs_core::Graph::from_edge_tokens`] for why), so the objects
//! themselves are never read.
//!
//! # What `unsafe` means
//!
//! Rust normally proves that every pointer is valid when used. Calling a C
//! function is outside what it can check, so those calls must be wrapped in
//! `unsafe { }`, which means "I have checked this myself". Each block below
//! has a `SAFETY:` comment saying why it is sound. Everything outside the
//! blocks is still checked by the compiler.

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fallback;

/// The shape of an adjacency dict and the address of every data dict in it.
pub(crate) struct Scan {
    /// The outer dict's keys, in order. Empty unless requested.
    pub(crate) nodes: Vec<Py<PyAny>>,
    /// CSR offsets: node `i`'s entries are `tokens[offsets[i]..offsets[i + 1]]`.
    pub(crate) offsets: Vec<usize>,
    /// For every entry, the address of its data dict, as a plain integer.
    /// Never dereferenced unless turned back into a reference by
    /// [`object_at`] while the dict is known to be unchanged.
    pub(crate) tokens: Vec<u64>,
}

impl Scan {
    /// A number that changes if any data dict is replaced or the entries
    /// are reordered. Used to notice a graph that changed behind our back.
    pub(crate) fn fingerprint(&self) -> u64 {
        self.tokens
            .iter()
            .fold(self.offsets.len() as u64, |h, &t| h.rotate_left(5) ^ t)
    }
}

/// Walks `{node: {neighbour: data}}` once.
///
/// Raises `Fallback` if an inner value is not exactly a dict.
pub(crate) fn scan(adjacency: &Bound<'_, PyDict>, want_nodes: bool) -> PyResult<Scan> {
    let py = adjacency.py();
    let mut nodes = Vec::new();
    let mut offsets = Vec::with_capacity(adjacency.len() + 1);
    let mut tokens = Vec::new();
    offsets.push(0);

    // `PyDict_Next` writes its progress and results through these.
    // `*mut T` is a raw pointer: an address with no validity guarantees.
    let mut position: ffi::Py_ssize_t = 0;
    let mut node: *mut ffi::PyObject = std::ptr::null_mut();
    let mut neighbors: *mut ffi::PyObject = std::ptr::null_mut();

    // SAFETY:
    // - `adjacency` is a `Bound`, which can only exist while this thread
    //   holds the GIL, so no other thread can modify these dicts.
    // - `PyDict_Next` only reads. The pointers it yields are valid for as
    //   long as the dict is not modified, and nothing in this loop can
    //   modify it: no Python code runs here (taking a reference with
    //   `from_borrowed_ptr` only increments a counter).
    // - `neighbors` is checked to be a dict before it is iterated.
    unsafe {
        while ffi::PyDict_Next(adjacency.as_ptr(), &mut position, &mut node, &mut neighbors) != 0 {
            if ffi::PyDict_CheckExact(neighbors) == 0 {
                return Err(fallback());
            }
            if want_nodes {
                nodes.push(Bound::from_borrowed_ptr(py, node).unbind());
            }
            let mut inner: ffi::Py_ssize_t = 0;
            let mut neighbor: *mut ffi::PyObject = std::ptr::null_mut();
            let mut data: *mut ffi::PyObject = std::ptr::null_mut();
            while ffi::PyDict_Next(neighbors, &mut inner, &mut neighbor, &mut data) != 0 {
                tokens.push(data as u64);
            }
            offsets.push(tokens.len());
        }
    }
    Ok(Scan {
        nodes,
        offsets,
        tokens,
    })
}

/// Turns an address recorded by [`scan`] back into a reference.
///
/// # Safety
///
/// `token` must come from a [`scan`] of a dict that has not been modified
/// since, with no Python code having run in between. Then the object is
/// still alive, because the dict still holds it.
///
/// An `unsafe fn` passes the obligation to its caller: calling it requires
/// an `unsafe` block.
pub(crate) unsafe fn object_at(py: Python<'_>, token: u64) -> Py<PyAny> {
    // SAFETY: guaranteed by the caller, as described above.
    unsafe { Bound::from_borrowed_ptr(py, token as *mut ffi::PyObject).unbind() }
}
