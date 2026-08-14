//! Core algorithms for ET-Miner - framework-agnostic Rust implementations.
//!
//! This module contains pure Rust algorithms without any Python bindings.
//! All functions operate on raw slices and return owned Vecs.
//!
//! # Module Organization
//! - `matrix`: CSR/CSC matrix operations and construction
//! - `counting`: Itemset support counting algorithms
//! - `cooccurrence`: k=2 co-occurrence matrix computation
//! - `bitvec`: Bitvector operations for GPU acceleration
//! - `candidates`: Candidate generation for Apriori
//! - `apriori`: Complete Apriori algorithm
//! - `utils`: Thread control and utility functions

pub mod matrix;
pub mod counting;
pub mod cooccurrence;
pub mod bitvec;
pub mod candidates;
pub mod apriori;
pub mod groups;
pub mod utils;

// Re-export commonly used types
pub use matrix::CscMatrix;
pub use apriori::AprioriResult;
