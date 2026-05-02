use std::{cmp::Reverse,cell::RefCell};
use rayon::prelude::*;
use fixedbitset::FixedBitSet;
use indicatif::ProgressBar;
use crate::core::Distance;
use super::{HNSWState, Candidate, ScratchBuffers, MinHeap, MaxHeap};

thread_local! {
    static SCRATCH: RefCell<Option<ScratchBuffers>> = const { RefCell::new(None) };
}

impl<T: Sync, D: Distance<T>> HNSWState<T, D> {

    pub fn parallel_search(&self, queries: &[T], k: usize, ef: usize, threads: usize, pb: Option<&ProgressBar>) -> Vec<Vec<(usize, f32)>> {
        let num_threads = if threads == 0 { rayon::current_num_threads() } else { threads };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build rayon thread pool");
        pool.install(|| {
            queries
                .par_iter()
                .map(|seq| {
                    SCRATCH.with(|s| {
                        let mut s = s.borrow_mut();
                        if s.is_none() {
                            *s = Some(ScratchBuffers::with_capacity(self.data.len(), ef, self.config.m_max));
                        }
                        let neighbors = self.search(seq, k, ef, s.as_mut().unwrap());
                        if let Some(p) = pb {
                            p.inc(1);
                        }
                        neighbors
                    })
                })
                .collect()
        })
    }

    /// Algorithm 5: K-NN-SEARCH
    /// Returns the `k` nearest neighbors to `query` from the index.
    pub fn search(&self, query: &T, k: usize, ef: usize, scratch: &mut ScratchBuffers) -> Vec<(usize, f32)> {
        let Some((ep_node, ep_layer)) = self.entry_point.get() else {
            return Vec::new();
        };

        scratch.nearest_neighbors.push(Candidate {
            idx: ep_node,
            distance: self.query_distance(query, ep_node),
        });

        // Coarse search: descend from entry layer down to layer 1
        for lc in (1..=ep_layer).rev() {
            self.search_layer_query(
                query, lc, ef, self.config.proximity_threshold, false,
                &mut scratch.visited,
                &mut scratch.candidates,
                &mut scratch.nearest_neighbors,
                &mut scratch.snapshot,
            );
        }

        // Fine search: layer 0 with full ef
        self.search_layer_query(
            query, 0, ef, self.config.proximity_threshold, self.config.strict_ef,
            &mut scratch.visited,
            &mut scratch.candidates,
            &mut scratch.nearest_neighbors,
            &mut scratch.snapshot,
        );

        // Pop from the MaxHeap (largest-first) into a vec, then reverse to get nearest-first
        let mut results: Vec<Candidate> =
            std::iter::from_fn(|| scratch.nearest_neighbors.pop()).collect();
        scratch.clear();

        results.reverse(); // pop() gives largest-first; reverse gives nearest-first
        results.truncate(k);
        results.into_iter().map(|c| (c.idx, c.distance)).collect()
    }

    /// Scratch buffers should be empty when passed to this function.
    /// They are cleared before exiting, except for `nearest_neighbors` which is used as
    /// both input (entry points) and output (result set).
    pub(super) fn search_layer_query(
        &self,
        query: &T,
        layer: usize,
        ef: usize,
        threshold: f32,
        strict_ef: bool,
        // Scratch buffers
        visited: &mut FixedBitSet,
        candidates: &mut MinHeap<Candidate>,
        // Input/Output
        nearest_neighbors: &mut MaxHeap<Candidate>,
        // Snapshot buffer: clones a node's neighbor list under a brief read lock
        snapshot: &mut Vec<usize>,
    ) {
        debug_assert!(!nearest_neighbors.is_empty(), "search_layer requires at least one entry point");
        debug_assert!(candidates.is_empty(), "candidates requires to be empty");
        debug_assert!(visited.is_clear(), "visited requires to be empty");

        for &v in nearest_neighbors.iter() {
            candidates.push(Reverse(v));
            visited.set(v.idx, true);
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let f = nearest_neighbors.peek().expect("nearest_neighbors should always be non-empty");
            if c.distance > f.distance {
                break; // Best candidate is worse than our worst nn — done
            }

            // Snapshot the neighbor list under a brief read lock, then release before
            // calling distance() (which may invoke the parasail FFI).
            self.hgraph.neighbors_snapshot(layer, c.idx, snapshot);

            for &neighbor in snapshot.iter() {
                if !visited.put(neighbor) {
                    let f = nearest_neighbors.peek().expect("nearest_neighbors should be non-empty").clone();
                    let dist_neighbor2query = self.query_distance(query, neighbor);
                    if dist_neighbor2query < f.distance || nearest_neighbors.len() < ef {
                        let new_candidate = Candidate { idx: neighbor, distance: dist_neighbor2query };
                        candidates.push(Reverse(new_candidate));
                        nearest_neighbors.push(new_candidate);
                        if nearest_neighbors.len() > ef {
                            // If `strict_ef` is disabled, the nearest_neighbors are allowed
                            // to grow larger than ef, as long as all are valid
                            // neighbors (distance < threshold)
                            if strict_ef || f.distance >= threshold {
                                nearest_neighbors.pop();
                            }
                        }
                    }
                }
            }
        }

        // Housekeeping: visited and candidates are not output, reset their state
        candidates.clear();
        visited.clear();
        snapshot.clear();
    }
}
