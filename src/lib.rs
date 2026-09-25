pub mod builder_macro;
pub mod clustering;
pub mod feature_extraction;
pub mod io;
pub mod plotting;
pub mod sirius;
pub mod spectrum;
pub mod workflow;
pub use spectrum::*;

/// Sets the maximum number of threads used by all parallel operations in this library.
///
/// This configures Rayon's global thread pool, which is shared by every `par_iter`,
/// `par_sort`, and matrix operation (via `faer`). Must be called before any parallel
/// work starts. Subsequent calls are no-ops (Rayon only allows one global initialization).
///
/// Passing `0` lets Rayon choose automatically (one thread per logical CPU).
pub fn set_global_thread_limit(n: usize) -> Result<(), rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
}
