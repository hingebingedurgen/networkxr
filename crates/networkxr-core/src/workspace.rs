//! Reusable working memory for searches.
//!
//! # The problem this solves
//!
//! A search needs per-node state: has this node been visited, what is its
//! best distance so far. The obvious way to hold it is an array with one
//! slot per node, created at the start of the search:
//!
//! ```text
//! let mut visited = vec![false; n];
//! ```
//!
//! That line costs time proportional to `n` even if the search then visits
//! three nodes. NetworkX uses Python sets and dicts, which start empty, so a
//! small search on a huge graph costs it almost nothing. Called in a loop
//! (the two-hop neighbourhood of every node, say) the array version would be
//! slower than NetworkX, in the worst case by a factor of `n`.
//!
//! # The fix
//!
//! Allocate the arrays once, in a [`Workspace`], and lend it to every
//! search. A search writes only the slots of nodes it touches, remembers
//! which those were, and puts them back to their blank values before
//! returning. Its cost is then proportional to what it touched.
//!
//! Every search function that takes a `&mut Workspace` leaves it clean.

use crate::graph::{NodeId, NO_NODE};

/// The value of [`Workspace::slot`] for a node the current search has not
/// recorded.
pub const UNSEEN: u32 = u32::MAX;

/// Per-node scratch arrays, all at their blank values between searches.
///
/// The fields are `pub(crate)`: visible to the algorithms in this crate,
/// hidden from its users, who only create a workspace and pass it in.
#[derive(Debug, Default, Clone)]
pub struct Workspace {
    /// A small integer per node, [`UNSEEN`] when blank. Searches use it as a
    /// visited mark or to remember a node's position in their output.
    pub(crate) slot: Vec<u32>,
    /// A tentative distance per node, infinity when blank.
    pub(crate) dist: Vec<f64>,
    /// A tentative parent per node, [`NO_NODE`] when blank.
    pub(crate) parent: Vec<NodeId>,
    /// The nodes whose entries a search has written, for [`release`].
    ///
    /// [`release`]: Self::release
    pub(crate) touched: Vec<NodeId>,
}

impl Workspace {
    /// An empty workspace. It sizes itself to the graph on first use, so
    /// creating one is free.
    ///
    /// `Self::default()` comes from `#[derive(Default)]`: every field gets
    /// its type's default, which for a `Vec` is empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// Makes the arrays fit a graph of `n` nodes. Does nothing if they
    /// already do, which is the usual case.
    pub(crate) fn prepare(&mut self, n: usize) {
        if self.slot.len() != n {
            // `clear` then `resize` refills the whole array with the blank
            // value, whatever it held before.
            self.slot.clear();
            self.slot.resize(n, UNSEEN);
            self.dist.clear();
            self.dist.resize(n, f64::INFINITY);
            self.parent.clear();
            self.parent.resize(n, NO_NODE);
            self.touched.clear();
        }
    }

    /// Records that `v`'s entries are about to be written.
    #[inline]
    pub(crate) fn touch(&mut self, v: NodeId) {
        self.touched.push(v);
    }

    /// Blanks every entry recorded with [`touch`](Self::touch).
    pub(crate) fn release(&mut self) {
        // `drain(..)` removes and yields every element, leaving the Vec
        // empty but keeping its allocation for the next search. The loop
        // borrows `self.touched` mutably while writing three other fields;
        // the compiler allows that because the fields do not overlap.
        for v in self.touched.drain(..) {
            self.slot[v as usize] = UNSEEN;
            self.dist[v as usize] = f64::INFINITY;
            self.parent[v as usize] = NO_NODE;
        }
    }

    /// True if every entry is blank. Used by tests.
    #[cfg(test)]
    pub(crate) fn is_clean(&self) -> bool {
        self.touched.is_empty()
            && self.slot.iter().all(|&s| s == UNSEEN)
            && self.dist.iter().all(|&d| d == f64::INFINITY)
            && self.parent.iter().all(|&p| p == NO_NODE)
    }
}
