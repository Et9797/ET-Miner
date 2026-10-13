//! Utility functions for thread control and configuration.

/// Get the number of threads Rayon will use for parallel operations.
pub fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Set the number of threads Rayon will use for parallel operations.
///
/// Note: This can only be called once per process. Subsequent calls are ignored.
/// Returns true if the thread pool was successfully configured, false otherwise.
pub fn set_num_threads(n_threads: usize) -> bool {
    if n_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .is_ok()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_num_threads() {
        // Should return at least 1 thread
        assert!(get_num_threads() >= 1);
    }
}
