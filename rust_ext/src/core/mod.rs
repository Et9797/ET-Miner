//! Core algorithms for ET-Miner - framework-agnostic Rust implementations.
//!
//! This module contains pure Rust algorithms without any Python bindings.
//! All functions operate on raw slices and return owned Vecs.
//!
//! # Module Organization
//! - `matrix`: CSR/CSC matrix operations and construction
//! - `cooccurrence`: k=2 co-occurrence matrix computation
//! - `candidates`: Candidate generation for Apriori
//! - `apriori`: Complete Apriori algorithm
//! - `groups`: K>=3 prefix-group build and the free-set prune

pub mod matrix;
pub mod cooccurrence;
pub mod candidates;
pub mod apriori;
pub mod groups;

// Re-export commonly used types
pub use matrix::CscMatrix;
pub use apriori::AprioriResult;
