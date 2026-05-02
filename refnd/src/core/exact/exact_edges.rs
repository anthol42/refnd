use indicatif::ProgressBar;
use rayon::prelude::*;
use crate::core::Distance;

/// Compute the exact edge list by evaluating all pairwise distances (upper triangle).
///
/// Only pairs `(i, j)` with `i < j` whose distance is strictly below `threshold` are included.
/// Parallelized with rayon: each row `i` is processed by an independent thread.
///
/// `threads` controls parallelism: `0` uses all available CPUs, otherwise the given count.
///
/// `pb` is optional — when `Some`, its total should be set to `n*(n-1)/2` (see
/// [`exact_edges_total`]) and it is incremented by the number of pairs in each completed row.
pub fn exact_edges<T, D>(
    data: &[T],
    distance: &D,
    threshold: f32,
    threads: usize,
    pb: Option<&ProgressBar>,
) -> Vec<(usize, usize, f32)>
where
    T: Sync,
    D: Distance<T>,
{
    let n = data.len();
    let num_threads = if threads == 0 { rayon::current_num_threads() } else { threads };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build rayon thread pool");
    pool.install(|| {
        (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let mut row = Vec::new();
                for j in (i + 1)..n {
                    let d = distance.call(&data[i], &data[j]);
                    if d < threshold {
                        row.push((i, j, d));
                    }
                }
                if let Some(pb) = pb {
                    // Row i compares against (n - i - 1) partners
                    pb.inc((n - i - 1) as u64);
                }
                row
            })
            .collect()
    })
}

/// Total number of pairs in the upper triangle for a dataset of size `n`.
/// Use this to initialize the progress bar length before calling [`exact_edges`].
pub fn exact_edges_total(n: usize) -> u64 {
    (n as u64 * n.saturating_sub(1) as u64) / 2
}
