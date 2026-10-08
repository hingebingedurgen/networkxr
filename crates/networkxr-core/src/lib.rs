//! # networkxr-core
//!
//! Graph algorithms over a compact, read-only graph whose nodes are the
//! integers `0..n`. This crate knows nothing about Python. The Python package
//! `networkxr` converts a NetworkX graph into a [`Graph`] (a "snapshot"), runs
//! an algorithm from this crate, and translates the integer results back.
//!
//! Every algorithm here is a port of the corresponding NetworkX function and
//! is written to return *exactly* what NetworkX returns for the same input,
//! including the order of results and which path wins a tie. Where that
//! forces an unusual choice, a comment says so.
//!
//! ## Reading order
//!
//! If you are new to Rust, read the modules in this order. Each one
//! introduces a few language features and explains them where they first
//! appear.
//!
//! 1. [`graph`]: structs, `impl` blocks, `Vec`, slices, borrowing, `Option`,
//!    `Result`, enums, traits
//! 2. [`workspace`]: mutable borrows, and why searches share scratch memory
//! 3. [`traversal`]: loops, closures, generic functions, `std::mem::take`
//! 4. [`components`]: more of the same, with a non-recursive Tarjan
//! 5. [`dag`]: errors that carry data, iterator adaptors
//! 6. [`shortest_paths`]: generic structs, `BinaryHeap`, implementing `Ord`,
//!    parallelism
//! 7. [`tree`] and [`cycles`]: a small data structure of our own, `HashSet`
//! 8. [`centrality`]: structs with defaults, lifetimes, per-thread state
//!
//! Lines starting with `//!` (like these) document the enclosing item, here
//! the whole crate. Lines starting with `///` document the item that follows.
//! `cargo doc --open` renders both as a website.

// `pub mod x;` says: there is a module `x` in the file `x.rs`, and it is
// visible to users of this crate.
pub mod centrality;
pub mod components;
pub mod cycles;
pub mod dag;
pub mod graph;
pub mod shortest_paths;
pub mod traversal;
pub mod tree;
pub mod workspace;

// Re-export the most used names so callers can write `networkxr_core::Graph`
// instead of `networkxr_core::graph::Graph`.
pub use graph::{Csr, Direction, EdgeId, Graph, GraphError, NodeId, NO_NODE};
pub use workspace::Workspace;
