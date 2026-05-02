use indicatif::ProgressBar;
use rayon::prelude::*;
use crate::core::Distance;

/// Compute the k nearest neighbors for every query against every reference, O(N×M).
///
/// Returns a `Vec` of length `queries.len()`. Each element is a `Vec<(ref_idx, distance)>`
/// sorted nearest-first and truncated to at most `k` entries.
///
/// Parallelized with rayon: each query row is processed independently.
///
/// `threads` controls parallelism: `0` uses all available CPUs, otherwise the given count.
/// 
/// `pb` is optional — when `Some`, it is incremented once per completed query.
pub fn exact_nearest_neighbors<T, D>(
    queries: &[T],
    references: &[T],
    distance: &D,
    k: usize,
    threads: usize,
    pb: Option<&ProgressBar>,
) -> Vec<Vec<(usize, f32)>>
where
    T: Sync,
    D: Distance<T>,
{
    let num_threads = if threads == 0 { rayon::current_num_threads() } else { threads };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build rayon thread pool");
    pool.install(|| {
        queries
            .par_iter()
            .map(|q| {
                let mut dists: Vec<(usize, f32)> = references
                    .iter()
                    .enumerate()
                    .map(|(j, r)| (j, distance.call(q, r)))
                    .collect();

                dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                dists.truncate(k);
                if let Some(pb) = pb { pb.inc(references.len() as u64); }
                dists
            })
            .collect()
    })
}
/// Total number of (query, reference) pairs for an N×M computation.
/// Use this to initialize the progress bar length before calling [`exact_nearest_neighbors`].
pub fn exact_nearest_neighbors_total(n_queries: usize, n_refs: usize) -> u64 {
    n_queries as u64 * n_refs as u64
}
