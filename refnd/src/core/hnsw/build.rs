use rayon::prelude::*;
use indicatif::ProgressBar;
use rand::seq::SliceRandom;
use super::{HNSWState, ScratchBuffers, RNG, SCRATCH};
#[cfg(feature = "monitor")]
use super::{STAT_SNAPSHOT, STAT_ADD_EDGE_LO, STAT_ADD_EDGE_HI, STAT_SET_NEIGHBOURHOOD, STAT_DASHMAP, STAT_CACHE_HIT, STAT_CACHE_MISS, STAT_ALIGNMENT, STAT_CACHE_GET, STAT_CACHE_INSERT};
use crate::core::Distance;

impl<T: Sync, D: Distance<T>> HNSWState<T, D> {
    pub fn build(&self, pb: Option<&ProgressBar>) {
        let n = self.data.len();

        let mut order: Vec<usize> = (0..n).collect();
        if self.config.shuffle {
            RNG.with(|r| order.shuffle(&mut *r.borrow_mut()));
        }

        let run = || {
            // Bootstrap: insert order[0] single-threaded to establish the entry point
            RNG.with(|r| {
                SCRATCH.with(|s| {
                    let mut s = s.borrow_mut();
                    if s.is_none() {
                        *s = Some(ScratchBuffers::with_capacity(
                            n,
                            self.config.ef_construction,
                            self.config.m_max0,
                        ));
                    }
                    self.insert_parallel(order[0], s.as_mut().unwrap(), &mut *r.borrow_mut());
                })
            });
            if let Some(pb) = pb { pb.inc(1); }

            // Parallel insertion: each thread uses its own thread-local RNG and scratch buffers
            order[1..].into_par_iter().for_each(|&idx| {
                RNG.with(|r| {
                    SCRATCH.with(|s| {
                        let mut s = s.borrow_mut();
                        if s.is_none() {
                            *s = Some(ScratchBuffers::with_capacity(
                                n,
                                self.config.ef_construction,
                                self.config.m_max0,
                            ));
                        }
                        self.insert_parallel(idx, s.as_mut().unwrap(), &mut *r.borrow_mut());
                    })
                });
                if let Some(pb) = pb { pb.inc(1); }
            });
        };

        if self.config.n_threads == 0 {
            run();
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.config.n_threads)
                .build()
                .expect("failed to build thread pool")
                .install(run);
        }

        #[cfg(feature = "monitor")]
        {
            let hits  = STAT_CACHE_HIT.calls.load(std::sync::atomic::Ordering::Relaxed);
            let misses = STAT_CACHE_MISS.calls.load(std::sync::atomic::Ordering::Relaxed);
            let total = hits + misses;
            let hit_rate = if total > 0 { 100.0 * hits as f64 / total as f64 } else { 0.0 };
            eprintln!("cache: {hits} hits / {total} calls ({hit_rate:.1}% hit rate)");
            STAT_ALIGNMENT.report("alignment          (FFI)   ");
            STAT_CACHE_GET.report("cache get          (lookup)");
            STAT_CACHE_INSERT.report("cache insert                ");
            STAT_SNAPSHOT.report("neighbors_snapshot (read)  ");
            STAT_ADD_EDGE_LO.report("add_edge lo        (write) ");
            STAT_ADD_EDGE_HI.report("add_edge hi        (write) ");
            STAT_SET_NEIGHBOURHOOD.report("set_neighbourhood  (write) ");
            STAT_DASHMAP.report("proximity_edges    (insert)");
        }
    }
}
