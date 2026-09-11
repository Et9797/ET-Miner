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

/// Run `f` on a rayon pool of exactly `n_threads` threads.
///
/// `set_num_threads` builds the GLOBAL pool and can only succeed once per
/// process, so it cannot express a per-call budget. A caller who asks for
/// `n_jobs=1` -- documented as "sequential execution" -- must actually get one
/// thread, not every core, or a job told to constrain itself quietly takes the
/// whole machine. `n_threads == 0` means "no budget": run on the global pool.
pub fn with_thread_budget<T, F>(n_threads: usize, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    if n_threads == 0 {
        return f();
    }
    // Pools are cached per thread count. Building one costs ~0.7 ms (measured),
    // which is nothing against a campaign level but real against the many short
    // calls a small dataset makes -- and the counting entry points are called
    // once per K level, so a fresh pool per call is pure overhead for a budget
    // that does not change within a run.
    match pool_for(n_threads) {
        Some(pool) => pool.install(f),
        // A pool we cannot build is not a reason to fail the count; the global
        // pool is a correct, merely unbudgeted, fallback.
        None => f(),
    }
}

fn pool_for(n_threads: usize) -> Option<std::sync::Arc<rayon::ThreadPool>> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = pools.lock().ok()?;
    if let Some(pool) = guard.get(&n_threads) {
        return Some(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .ok()?,
    );
    guard.insert(n_threads, Arc::clone(&pool));
    Some(pool)
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
