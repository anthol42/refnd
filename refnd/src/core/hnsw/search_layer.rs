use std::cmp::Reverse;
use fixedbitset::FixedBitSet;
use crate::core::Distance;
use super::{HNSWState, Candidate, MinHeap, MaxHeap};

impl<T: Sync, D: Distance<T>> HNSWState<T, D> {
    /// Scratch buffers should be empty when passed to this function.
    /// They are cleared before exiting, except for `nearest_neighbors` which is used as
    /// both input (entry points) and output (result set).
    pub(super) fn search_layer(
        &self,
        query: usize,
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
                    let dist_neighbor2query = self.distance(neighbor, query);
                    if dist_neighbor2query < f.distance || nearest_neighbors.len() < ef {
                        let new_candidate = Candidate { idx: neighbor, distance: dist_neighbor2query };
                        candidates.push(Reverse(new_candidate));
                        nearest_neighbors.push(new_candidate);
                        if nearest_neighbors.len() > ef {
                            // If `strict_ef` is disabled, the nearest_neighbors are allowed
                            // to grow larger than ef, as long as all are valid
                            // neighbors (distance < threshold)
                            if strict_ef || f.distance >= threshold{
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
